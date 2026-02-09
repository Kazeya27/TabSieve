import os
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# ---------- 配置与全局状态 ----------
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", str(RANK)))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}") if torch.cuda.is_available() else torch.device("cpu")
print(f"RANK:{RANK}")
print(f"WORLD_SIZE:{WORLD_SIZE}")
print(f"LOCAL_RANK:{LOCAL_RANK}")
print(f"DEVICE:{DEVICE}")

# RPC
import torch.distributed.rpc as rpc
RPC_NAME = f"worker{RANK}"

# 这些全局变量在 init_shards() 后可用
TEXT_SHARD = None          # 放 GPU：仅文本检索用，矩阵大，需并行
IMAGE_SHARD_CPU = None     # 放 CPU：图像只拉少量行，没必要占 GPU 显存
SHARD_START = 0
SHARD_END = 0
EMB_DIM = None

# ---------- 数据模型 ----------
class InputData(BaseModel):
    response: str
    index: int
    k: int

# ---------- 实用函数 ----------
def compute_shard_ranges(total: int, world_size: int) -> Tuple[List[int], List[int]]:
    """
    返回 offsets, sizes：把 total 行平均切到 world_size 份，前 remainder 份多 1 行。
    """
    base = total // world_size
    rem = total % world_size
    sizes = [base + (1 if i < rem else 0) for i in range(world_size)]
    offsets = [0]
    for i in range(world_size - 1):
        offsets.append(offsets[-1] + sizes[i])
    return offsets, sizes

def owner_rank_for_index(idx: int, offsets: List[int], sizes: List[int]) -> int:
    for r, (st, sz) in enumerate(zip(offsets, sizes)):
        if st <= idx < st + sz:
            return r
    raise IndexError(f"Global index {idx} out of range")

# ---------- RPC 远程函数（在各 rank 上执行） ----------
@torch.no_grad()
def rpc_local_topk(q_cpu: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    每个 worker 在本地 TEXT_SHARD（GPU）上做局部 Top-K。
    返回 (vals_cpu[k], global_inds_cpu[k])
    """
    assert q_cpu.dim() == 1 or (q_cpu.dim() == 2 and q_cpu.shape[0] == 1)
    q = q_cpu.reshape(-1).to(DEVICE)
    # 确保 q 单位范数（安全起见）
    q = F.normalize(q, p=2, dim=0)

    global SHARD_START, SHARD_END, TEXT_SHARD
    if TEXT_SHARD is None or TEXT_SHARD.numel() == 0:
        # 该分片为空（可能 world_size > 样本数）
        empty_vals = torch.empty(0, dtype=torch.float32)
        empty_inds = torch.empty(0, dtype=torch.long)
        return empty_vals, empty_inds

    sims = TEXT_SHARD @ q  # [n_shard]
    k_local = min(k, sims.shape[0])
    vals, local_inds = torch.topk(sims, k_local, largest=True, sorted=True)
    global_inds = local_inds + SHARD_START
    return vals.detach().cpu(), global_inds.detach().cpu()

@torch.no_grad()
def rpc_get_image_embed(global_idx: int) -> torch.Tensor:
    """
    返回该 rank 所持有的图像向量（CPU 张量，已归一化）。
    仅在“ owner rank ”上被调用。
    """
    global SHARD_START, IMAGE_SHARD_CPU
    local_idx = global_idx - SHARD_START
    vec = IMAGE_SHARD_CPU[local_idx:local_idx+1]  # [1, d]
    return vec.squeeze(0).contiguous()  # [d] CPU float32

@torch.no_grad()
def rpc_text_sim_at_index(q_cpu: torch.Tensor, global_idx: int) -> float:
    """
    返回 q 与指定 global_idx 对应文本向量的相似度（余弦）。
    仅在该 index 所属的 owner rank 上调用。
    """
    assert q_cpu.dim() == 1 or (q_cpu.dim() == 2 and q_cpu.shape[0] == 1)
    q = q_cpu.reshape(-1).to(DEVICE)
    q = F.normalize(q, p=2, dim=0)

    global SHARD_START, TEXT_SHARD
    local_idx = global_idx - SHARD_START
    # TEXT_SHARD 已按行归一化
    vec = TEXT_SHARD[local_idx]  # [d] on DEVICE
    sim = torch.sum(vec * q)
    return float(sim.detach().cpu().item())

@torch.no_grad()
def rpc_count_greater(q_cpu: torch.Tensor, pivot_sim: float) -> int:
    """
    统计本地分片中与 q 的相似度严格大于 pivot_sim 的样本数量。
    用于计算 dense rank：k = 1 + sum(count(sim > pivot)).
    """
    assert q_cpu.dim() == 1 or (q_cpu.dim() == 2 and q_cpu.shape[0] == 1)
    q = q_cpu.reshape(-1).to(DEVICE)
    q = F.normalize(q, p=2, dim=0)

    global TEXT_SHARD
    if TEXT_SHARD is None or TEXT_SHARD.numel() == 0:
        return 0

    sims = TEXT_SHARD @ q  # [n_shard]
    pivot = torch.tensor(pivot_sim, dtype=sims.dtype, device=sims.device)
    cnt = torch.count_nonzero(sims > pivot).item()
    return int(cnt)

# ---------- 分片初始化 ----------
def init_shards(text_paths: str, image_paths: str):
    global TEXT_SHARD, IMAGE_SHARD_CPU, SHARD_START, SHARD_END, EMB_DIM

    # 1) CPU 读入（各进程各读一遍，数据量不大时最简单可靠）
    text_all = []
    image_all = []
    for text_path in text_paths:
        text_all.append(torch.load(text_path, map_location="cpu").float())
    for image_path in image_paths:
        image_all.append(torch.load(image_path, map_location="cpu").float())
    text_all = torch.concat(text_all, dim=0)
    image_all = torch.concat(image_all, dim=0)
    assert text_all.shape[0] == image_all.shape[0]

    # 2) 归一化（和你的 encode(normalize_embeddings=True) 对齐）
    text_all = F.normalize(text_all, p=2, dim=1)
    image_all = F.normalize(image_all, p=2, dim=1)

    n, d = text_all.shape
    print(f"text_all.shape[0]: {n}")
    EMB_DIM = d
    offsets, sizes = compute_shard_ranges(n, WORLD_SIZE)
    SHARD_START = offsets[RANK]
    SHARD_END = SHARD_START + sizes[RANK]

    # 3) 切片：文本分片上 GPU，图像分片留 CPU
    if sizes[RANK] > 0:
        TEXT_SHARD = text_all[SHARD_START:SHARD_END].to(DEVICE, non_blocking=True)
        IMAGE_SHARD_CPU = image_all[SHARD_START:SHARD_END].contiguous()  # CPU
    else:
        TEXT_SHARD = torch.empty((0, d), device=DEVICE, dtype=torch.float32)
        IMAGE_SHARD_CPU = torch.empty((0, d), device="cpu", dtype=torch.float32)

    # 释放大 tensor 引用
    del text_all, image_all

    # 记录 offsets/sizes 到全局（rank=0 用来判 owner）
    init_shards.offsets = offsets
    init_shards.sizes = sizes

# 给函数挂属性，方便其它地方访问
init_shards.offsets = None
init_shards.sizes = None

# ---------- FastAPI（仅 rank=0 启） ----------
def build_app(model_path: str) -> FastAPI:
    app = FastAPI()

    # 仅 rank=0 加载模型到本地 GPU
    model_device = DEVICE
    model = SentenceTransformer(model_path, device=str(model_device))

    @app.post("/topk")
    @torch.no_grad()
    def topk(data: InputData):
        """
        1) rank=0 编码查询文本 -> q
        2) RPC 到各 rank 做局部 Top-K（文本库）
        3) 汇总出全局 Top-k 的 global indices
        4) RPC 拉取 target 图像向量 & k 个候选图像向量（CPU），在 rank=0 点积 + 加权求和
        """
        k = data.k

        # 1) 文本编码（已归一化）
        q = model.encode([data.response], convert_to_tensor=True, normalize_embeddings=True).float()
        q_cpu = q.squeeze(0).detach().cpu()  # [d]

        # 2) 并发 RPC：各 rank 局部 Top-K
        futs = []
        for r in range(WORLD_SIZE):
            futs.append(
                rpc.rpc_async(
                    to=f"worker{r}",
                    func=rpc_local_topk,
                    args=(q_cpu, k),
                )
            )
        results = [f.wait() for f in futs]
        # results: List[(vals[k_r], inds[k_r])]
        all_vals = torch.cat([v for (v, _) in results if v.numel() > 0], dim=0) if results else torch.empty(0)
        all_inds = torch.cat([i for (_, i) in results if i.numel() > 0], dim=0) if results else torch.empty(0, dtype=torch.long)

        if all_inds.numel() == 0:
            return float(0.0)

        k_global = min(k, all_inds.numel())
        top_vals, top_sel = torch.topk(all_vals, k=k_global, largest=True, sorted=True)
        top_global_inds = all_inds[top_sel].tolist()  # 全局 top-k 的全局索引（按文本相似度降序）

        # 3) 拉取 target 图像向量（CPU）
        offsets, sizes = init_shards.offsets, init_shards.sizes
        owner_t = owner_rank_for_index(data.index, offsets, sizes)
        target_vec = rpc.rpc_sync(to=f"worker{owner_t}", func=rpc_get_image_embed, args=(data.index,))  # [d] CPU

        # 4) 拉取候选图像向量（CPU，数量很小）
        cand_futs = []
        for gi in top_global_inds:
            owner = owner_rank_for_index(gi, offsets, sizes)
            cand_futs.append(rpc.rpc_async(to=f"worker{owner}", func=rpc_get_image_embed, args=(gi,)))
        cand_vecs = [f.wait() for f in cand_futs]  # List[Tensor[d] CPU]

        # 5) 在 rank=0 计算相似度与加权和（CPU，量很小）
        target_vec = target_vec.to(dtype=torch.float32)
        sims = []
        vec_list = []
        for v in cand_vecs:
            v = v.to(dtype=torch.float32)
            sims.append(float(torch.dot(v, target_vec).item()))  # 归一化后内积=余弦
            vec_list.append(v)
        sims_t = torch.tensor(sims, dtype=torch.float32)  # [k]
        weights = 0.5 ** torch.arange(len(sims_t), dtype=torch.float32)
        sim_score = float(torch.sum(weights * sims_t).item())
        
        mrl_score = float(torch.linalg.norm(torch.stack(vec_list, dim=0).mean(dim=0), ord=2).item())
        return sim_score + mrl_score
    
    @app.post("/sim")
    @torch.no_grad()
    def sim(data: InputData):
        """
        1) rank=0 编码查询文本 -> q
        2) RPC 到各 rank 做局部 Top-K（文本库）
        3) 汇总出全局 Top-k 的 global indices
        4) RPC 拉取 target 图像向量 & k 个候选图像向量（CPU），在 rank=0 点积 + 加权求和
        """
        k = data.k

        # 1) 文本编码（已归一化）
        q = model.encode([data.response], convert_to_tensor=True, normalize_embeddings=True).float()
        q_cpu = q.squeeze(0).detach().cpu()  # [d]

        # 2) 并发 RPC：各 rank 局部 Top-K
        futs = []
        for r in range(WORLD_SIZE):
            futs.append(
                rpc.rpc_async(
                    to=f"worker{r}",
                    func=rpc_local_topk,
                    args=(q_cpu, k),
                )
            )
        results = [f.wait() for f in futs]
        # results: List[(vals[k_r], inds[k_r])]
        all_vals = torch.cat([v for (v, _) in results if v.numel() > 0], dim=0) if results else torch.empty(0)
        all_inds = torch.cat([i for (_, i) in results if i.numel() > 0], dim=0) if results else torch.empty(0, dtype=torch.long)

        if all_inds.numel() == 0:
            return float(0.0)

        k_global = min(k, all_inds.numel())
        top_vals, top_sel = torch.topk(all_vals, k=k_global, largest=True, sorted=True)
        top_global_inds = all_inds[top_sel].tolist()  # 全局 top-k 的全局索引（按文本相似度降序）

        # 3) 拉取 target 图像向量（CPU）
        offsets, sizes = init_shards.offsets, init_shards.sizes
        owner_t = owner_rank_for_index(data.index, offsets, sizes)
        target_vec = rpc.rpc_sync(to=f"worker{owner_t}", func=rpc_get_image_embed, args=(data.index,))  # [d] CPU

        # 4) 拉取候选图像向量（CPU，数量很小）
        cand_futs = []
        for gi in top_global_inds:
            owner = owner_rank_for_index(gi, offsets, sizes)
            cand_futs.append(rpc.rpc_async(to=f"worker{owner}", func=rpc_get_image_embed, args=(gi,)))
        cand_vecs = [f.wait() for f in cand_futs]  # List[Tensor[d] CPU]

        # 5) 在 rank=0 计算相似度与加权和（CPU，量很小）
        target_vec = target_vec.to(dtype=torch.float32)
        sims = []
        vec_list = []
        for v in cand_vecs:
            v = v.to(dtype=torch.float32)
            sims.append(float(torch.dot(v, target_vec).item()))  # 归一化后内积=余弦
            vec_list.append(v)
        sims_t = torch.tensor(sims, dtype=torch.float32)  # [k]
        weights = 0.5 ** torch.arange(len(sims_t), dtype=torch.float32)
        sim_score = float(torch.sum(weights * sims_t).item())
        return sim_score
    
    @app.post("/mrl")
    @torch.no_grad()
    def mrl(data: InputData):
        """
        1) rank=0 编码查询文本 -> q
        2) RPC 到各 rank 做局部 Top-K（文本库）
        3) 汇总出全局 Top-k 的 global indices
        4) RPC 拉取 target 图像向量 & k 个候选图像向量（CPU），在 rank=0 点积 + 加权求和
        """
        k = data.k

        # 1) 文本编码（已归一化）
        q = model.encode([data.response], convert_to_tensor=True, normalize_embeddings=True).float()
        q_cpu = q.squeeze(0).detach().cpu()  # [d]

        # 2) 并发 RPC：各 rank 局部 Top-K
        futs = []
        for r in range(WORLD_SIZE):
            futs.append(
                rpc.rpc_async(
                    to=f"worker{r}",
                    func=rpc_local_topk,
                    args=(q_cpu, k),
                )
            )
        results = [f.wait() for f in futs]
        # results: List[(vals[k_r], inds[k_r])]
        all_vals = torch.cat([v for (v, _) in results if v.numel() > 0], dim=0) if results else torch.empty(0)
        all_inds = torch.cat([i for (_, i) in results if i.numel() > 0], dim=0) if results else torch.empty(0, dtype=torch.long)

        if all_inds.numel() == 0:
            return float(0.0)

        k_global = min(k, all_inds.numel())
        top_vals, top_sel = torch.topk(all_vals, k=k_global, largest=True, sorted=True)
        top_global_inds = all_inds[top_sel].tolist()  # 全局 top-k 的全局索引（按文本相似度降序）

        # 3) 拉取 target 图像向量（CPU）
        offsets, sizes = init_shards.offsets, init_shards.sizes
        owner_t = owner_rank_for_index(data.index, offsets, sizes)
        target_vec = rpc.rpc_sync(to=f"worker{owner_t}", func=rpc_get_image_embed, args=(data.index,))  # [d] CPU

        # 4) 拉取候选图像向量（CPU，数量很小）
        cand_futs = []
        for gi in top_global_inds:
            owner = owner_rank_for_index(gi, offsets, sizes)
            cand_futs.append(rpc.rpc_async(to=f"worker{owner}", func=rpc_get_image_embed, args=(gi,)))
        cand_vecs = [f.wait() for f in cand_futs]  # List[Tensor[d] CPU]

        # 5) 在 rank=0 计算相似度与加权和（CPU，量很小）
        vec_list = []
        for v in cand_vecs:
            v = v.to(dtype=torch.float32)
            vec_list.append(v)

        mrl_score = float(torch.linalg.norm(torch.stack(vec_list, dim=0).mean(dim=0), ord=2).item())
        return mrl_score

    @app.post("/rr")
    @torch.no_grad()
    def reciprocal_rank(data: InputData):
        """
        计算 data.index 在“文本与查询 q 的余弦相似度”全局排序中的名次 k（dense rank，按 sim 严格大于计数），返回 1/k。
        """
        # 1) 编码与归一化查询
        q = model.encode([data.response], convert_to_tensor=True, normalize_embeddings=True).float()
        q_cpu = q.squeeze(0).detach().cpu()  # [d]

        # 2) 先得到目标 index 的相似度（在 owner rank 上点积）
        offsets, sizes = init_shards.offsets, init_shards.sizes
        owner = owner_rank_for_index(data.index, offsets, sizes)
        pivot_sim = rpc.rpc_sync(
            to=f"worker{owner}",
            func=rpc_text_sim_at_index,
            args=(q_cpu, data.index),
        )

        # 3) 并发统计所有分片中 sim > pivot_sim 的数量
        futs = []
        for r in range(WORLD_SIZE):
            futs.append(
                rpc.rpc_async(
                    to=f"worker{r}",
                    func=rpc_count_greater,
                    args=(q_cpu, float(pivot_sim)),
                )
            )
        counts = [f.wait() for f in futs]
        total_greater = int(sum(counts))

        # 4) dense rank 与倒数
        k = total_greater + 1
        return float(1.0 / k)
    
    return app

# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/oss_bucket_0/jiahn/models/all-MiniLM-L6-v2")
    parser.add_argument("--text_paths", type=str, nargs='+', default=["/data/oss_bucket_0/jiahn/datasets/sc-captioner/train_coco6k_caption_all-MiniLM-L6-v2.pt"])
    parser.add_argument("--image_paths", type=str, nargs='+', default=["/data/oss_bucket_0/jiahn/datasets/sc-captioner/train_coco6k_image_ViT-H-14.pt"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3545)
    parser.add_argument("--rpc_threads", type=int, default=16)
    args = parser.parse_args()

    # 每个进程固定到对应 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    # 初始化分片（加载 & 切片 & 归一化）
    init_shards(args.text_paths, args.image_paths)

    # 初始化 RPC（TensorPipe，名字 worker{rank}）
    # 注：用 torchrun 启动时，MASTER_ADDR/MASTER_PORT 已自动设置
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads)
    rpc.init_rpc(name=RPC_NAME, rank=RANK, world_size=WORLD_SIZE, rpc_backend_options=rpc_backend_options)

    if RANK == 0:
        # 只在 rank=0 启动 HTTP 服务
        app = build_app(args.model_path)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        # 退出时优雅关闭 RPC
        rpc.shutdown()
    else:
        # 其它 rank 常驻等待 RPC 调用
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            rpc.shutdown()

if __name__ == "__main__":
    main()