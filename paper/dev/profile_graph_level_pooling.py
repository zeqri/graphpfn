"""GPU-side profiling for the graph-level pooling pipeline (needs a real GPU
-- CPU-only profiling could not settle whether the sparse dgl.ops in
GeometricAttentionStack/SparseAttentionPooler accelerate well on GPU, or
quantify the actual benefit of GraphLevelPriorSampler's background
prefetching under real GPU compute speeds).

Usage (from paper/, on a GPU node/allocation):
    python dev/profile_graph_level_pooling.py

Prints:
  1. A torch.profiler table of the top ops by CUDA time over a few real
     training steps at production scale -- directly shows whether dense ops
     (encoder_x, transformer_encoder) or the dgl sparse spmm/gspmm ops
     (geometric_attention, pooler) actually dominate GPU time.
  2. A synchronous-vs-prefetching wall-clock comparison (n_workers=0 vs the
     configured n_workers) over several steps, on the SAME GPU, to quantify
     background prefetching's real benefit (not just the CPU-only estimate
     from the earlier profiling).

Also writes a chrome trace to dev/profile_trace.json (open in
chrome://tracing or https://ui.perfetto.dev for a visual timeline) if you
want more detail than the printed table.
"""

import sys
import time
from pathlib import Path

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

import numpy as np
import tomllib
import torch
from torch.profiler import ProfilerActivity, profile

import lib.tfm.limix as limix_mod

_LOCAL_CKPT = str(REPO / "checkpoints/LimiX-16M.ckpt")
limix_mod._download_limix_checkpoint = lambda: _LOCAL_CKPT

from lib.graphpfn.model import GraphPFN
from lib.graphpfn.pooling import GraphLevelGraphPFN
from lib.graphpfn.prior.graph_level import GraphLevelPriorSampler, raw_sample_to_graph


def context_query_split(n_graphs, train_ratio, rng):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all, train_idx):
    mean = y_all.mean()
    std = y_all.std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std


def next_valid_split(sampler, rng, max_attempts=50):
    for _ in range(max_attempts):
        atom_features, graph, y_per_molecule, train_ratio = raw_sample_to_graph(next(sampler))
        n_graphs = int(graph.batch_num_nodes().shape[0])
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        y_np = y_per_molecule.numpy()
        if len(test_idx) > 0 and len(train_idx) >= 2 and np.std(y_np[test_idx]) != 0:
            return atom_features, graph, y_np, train_idx, test_idx, n_graphs
    raise RuntimeError("could not get a valid split sample")


def main():
    assert torch.cuda.is_available(), "This script needs a real GPU allocation."
    device = torch.device("cuda")

    with open(
        REPO / "exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/pretrain.toml", "rb"
    ) as f:
        toml_config = tomllib.load(f)
    config = toml_config["base_config"]
    prior_config = config["prior"]
    sampler_config = config.get("sampler", {})

    print("Building model on GPU...")
    graphpfn = GraphPFN(
        edge_head=None, feat_head=False, layer_ids=list(range(12)),
        freeze_tfm=True, random_init_tfm=False,
    ).to(device)
    model = GraphLevelGraphPFN(
        graphpfn, embed_dim=graphpfn.tfm.module.embed_dim, **config.get("pooling", {})
    ).to(device)
    model.train()
    print("Model built.\n")

    rng = np.random.default_rng(0)

    def run_one_step(sampler):
        atom_features, graph, y_np, train_idx, test_idx, n_graphs = next_valid_split(sampler, rng)
        y_std = standardize_by_context(y_np, train_idx)
        is_context = torch.zeros(n_graphs, dtype=torch.bool, device=device)
        is_context[torch.from_numpy(train_idx)] = True
        y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
        atom_features = atom_features.to(device)
        graph = graph.to(device)
        with torch.autocast(device.type, enabled=True, dtype=torch.bfloat16):
            pred = model(graph, atom_features, y_std_t, is_context)
        query_idx = torch.from_numpy(test_idx).to(device)
        loss = torch.nn.functional.mse_loss(pred[query_idx], y_std_t[query_idx])
        loss.backward()
        model.zero_grad()
        return n_graphs, graph.num_nodes()

    # === Part 1: torch.profiler over a few real steps (with prefetching) ===
    print("=== Part 1: op-level GPU profiling (torch.profiler) ===")
    sampler = GraphLevelPriorSampler(base_prior_config=prior_config, seed=1, **sampler_config)
    print("Warming up (2 steps, untimed -- CUDA context/kernel init)...")
    for _ in range(2):
        run_one_step(sampler)
    torch.cuda.synchronize()

    N_PROFILE_STEPS = 3
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(N_PROFILE_STEPS):
            run_one_step(sampler)
            torch.cuda.synchronize()

    print(f"\nTop ops by CUDA time, over {N_PROFILE_STEPS} steps:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    trace_path = REPO / "dev/profile_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace written to {trace_path} (open in chrome://tracing or ui.perfetto.dev)")

    # === Part 2: synchronous vs. prefetching wall-clock, same GPU ===
    print("\n=== Part 2: synchronous vs. prefetching wall-clock (same GPU) ===")
    N_TIMED_STEPS = 15

    print(f"Running {N_TIMED_STEPS} steps with n_workers=0 (synchronous sampling)...")
    sync_sampler = GraphLevelPriorSampler(base_prior_config=prior_config, seed=2, n_workers=0)
    for _ in range(2):
        run_one_step(sync_sampler)  # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TIMED_STEPS):
        run_one_step(sync_sampler)
    torch.cuda.synchronize()
    sync_time = time.perf_counter() - t0

    print(f"Running {N_TIMED_STEPS} steps with configured prefetching ({sampler_config})...")
    prefetch_sampler = GraphLevelPriorSampler(base_prior_config=prior_config, seed=3, **sampler_config)
    for _ in range(2):
        run_one_step(prefetch_sampler)  # warm up (lets prefetch buffer fill too)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TIMED_STEPS):
        run_one_step(prefetch_sampler)
    torch.cuda.synchronize()
    prefetch_time = time.perf_counter() - t0

    print(f"\nsynchronous:  {sync_time:.2f}s total, {sync_time/N_TIMED_STEPS:.3f}s/step")
    print(f"prefetching:  {prefetch_time:.2f}s total, {prefetch_time/N_TIMED_STEPS:.3f}s/step")
    speedup = sync_time / prefetch_time if prefetch_time > 0 else float("nan")
    print(f"speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
