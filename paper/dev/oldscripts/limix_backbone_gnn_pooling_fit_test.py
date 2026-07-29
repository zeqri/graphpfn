"""Extends limix_encoder_gnn_fit_test.py to use the REAL pretrained
LimiX-16M backbone end-to-end, not just encoder_x in isolation with a
from-scratch supervised head. Runs on multiple GPUs via DDP.

encoder_x is loaded from the real checkpoint (paper/checkpoints/LimiX-16M.ckpt)
and kept FROZEN. Atom embeddings from encoder_x are pooled into per-molecule
tokens by a NEW, trainable GNN (real bonds, shared weights across the
feature-group dimension -- see PoolingGNN), preserving the group dimension
the backbone's transformer_encoder actually attends over (per
FeaturesTransformer.forward's exact add_embeddings -> mixed_y_embedding ->
transformer_encoder -> y_decoder chain, replicated manually here since the
"sequence" the transformer attends over needs to be MOLECULES, not the raw
node-level sequence FeaturesTransformer.forward assumes).

Only NEW/trainable parameters: PoolingGNN (message passing + atom->molecule
pooling). encoder_x, transformer_encoder, y_decoder, add_embeddings all stay
frozen -- real pretrained weights, doing real in-context learning: context
molecules' labels are revealed, query molecules' labels are predicted by the
frozen backbone's own attention over pooled molecule tokens.

DDP design: the one sampled dataset (fixed seed) and its context-first
molecule reorder are identical on every rank (no broadcast needed -- same
seed, same deterministic RNG call sequence). The expensive part
(transformer_encoder, ~44% of forward time per earlier profiling) is what
gets sharded: each rank's forward pass attends over the SAME shared context
molecules plus only ITS OWN disjoint slice of query molecules, so the
"sequence" length -- and the O(seq^2)-ish attention cost -- shrinks
per-rank instead of every rank redundantly processing all query molecules.
PoolingGNN (the only trainable module) is wrapped in DistributedDataParallel
so its gradients get averaged across ranks after each backward call; the
frozen backbone needs no DDP wrapping (no trainable params to sync). Every
LOG_EVERY epochs, predictions are all-gathered across ranks to report one
true test R2 over every query molecule, not just rank 0's shard.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_fit_test.py
(Falls back to a single-process, single-GPU/CPU run if launched with plain
`python`, no torchrun -- lib.is_ddp() is False whenever RANK isn't set.)
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import dgl  # noqa: E402
import dgl.nn.pytorch as dglnn  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lib  # noqa: E402
import lib.deep  # noqa: E402
from lib.graphpfn.prior.graph_level import sample_graph_level_dataset  # noqa: E402
from vendor.limix.utils.loading import load_model  # noqa: E402
from dev.limix_encoder_pooling_probe import BASE_PRIOR_CONFIG  # noqa: E402

SEED = 0
CHECKPOINT_PATH = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
TRAIN_FRACTION = 0.8
N_EPOCHS = 1000
N_GNN_LAYERS = 3
DROPOUT = 0.0  # matches the real graph-adapter modules' own default (layers.py:
# GraphPFNGraphAttentionModule/GraphPFNMLPModule both default dropout=0.0)
LOG_EVERY = 20

# Optimizer/scheduler/clipping/EMA settings taken directly from the real
# pretraining config (exp/graphpfn/pretrain/main/pretrain.toml), which
# already tunes exactly this scenario: new adapter parameters trained behind
# a frozen backbone. Only the warmup fraction is preserved as a RATIO rather
# than copied as an absolute step count -- that config's n_warmup_steps=1000
# is against n_steps=10000 (10% warmup); our N_EPOCHS=300 total steps is a
# different scale, so warmup is scaled proportionally to keep the same
# shape of LR schedule instead of never finishing warmup.
OPTIMIZER_TYPE = "AdamW"
LR = 0.001
WEIGHT_DECAY = 0.1
GRADIENT_CLIPPING_NORM = 1.0
LR_SCHEDULER = "cosine"
N_WARMUP_STEPS = max(1, round(N_EPOCHS * 1000 / 10000))
EMA_DECAY = 0.98


class PoolingGNN(nn.Module):
    """Message passing (real bonds) + atom->molecule pooling, applied
    independently per feature-group with SHARED weights across groups (the
    same SAGEConv/LayerNorm instances are called once per group, looping
    over the group dim) -- matches how the real graph adapter
    (GraphPFNGraphAttentionModule) natively operates on the
    (n_nodes, n_groups, embed_dim) shape, instead of collapsing groups to a
    single vector before message passing (an earlier version of this
    exploration did that and it discarded real signal -- see GroupFusion's
    docstring in limix_encoder_pooling_probe.py for the same lesson applied
    to a different collapse point).
    """

    def __init__(self, embed_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(
            dglnn.SAGEConv(embed_dim, embed_dim, aggregator_type="mean") for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_embeddings_grouped: torch.Tensor,
        molecule_id: torch.Tensor,
        n_molecules: int,
    ) -> torch.Tensor:
        n_atoms, n_groups, embed_dim = atom_embeddings_grouped.shape
        dtype = atom_embeddings_grouped.dtype

        group_outputs = []
        for g in range(n_groups):
            h = atom_embeddings_grouped[:, g, :].float()  # SAGEConv wants float32
            for conv, norm in zip(self.convs, self.norms):
                h = self.dropout(F.relu(norm(conv(graph, h))))
            group_outputs.append(h)
        atom_out = torch.stack(group_outputs, dim=1)  # (n_atoms, n_groups, embed_dim)

        pooled = torch.zeros(n_molecules, n_groups, embed_dim, device=atom_out.device)
        counts = torch.zeros(n_molecules, 1, 1, device=atom_out.device)
        pooled = pooled.index_add(0, molecule_id, atom_out)
        counts = counts.index_add(0, molecule_id, torch.ones(n_atoms, 1, 1, device=atom_out.device))
        pooled = pooled / counts.clamp(min=1)  # (n_molecules, n_groups, embed_dim)
        return pooled.to(dtype)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    if lib.is_ddp():
        lib.configure_ddp()
    rank = lib.get_rank()
    world_size = lib.get_world_size()
    device = lib.get_device()
    is_main = lib.is_master_process()

    # Same seed on every rank -> identical dataset/reorder without needing a
    # broadcast (see module docstring).
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if is_main:
        print(f"DDP: world_size={world_size}" if lib.is_ddp() else "Single-process run (no torchrun)")
        print(f"Loading real LimiX-16M checkpoint from {CHECKPOINT_PATH}...")
    model = load_model(str(CHECKPOINT_PATH), mask_prediction=False)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    features_per_group = model.features_per_group
    embed_dim = model.embed_dim
    if is_main:
        print(f"features_per_group={features_per_group}, embed_dim={embed_dim}, nlayers={model.nlayers}")
        print("Sampling molecules from the synthetic prior...")
    atom_features, graph, y_per_molecule, _ = sample_graph_level_dataset(BASE_PRIOR_CONFIG)
    n_features = atom_features.shape[-1]
    if is_main:
        print(f"n_atoms={graph.num_nodes()}, n_molecules={graph.batch_num_nodes().shape[0]}, n_features={n_features}")

    # Context-first molecule reorder (unbatch/rebatch keeps each molecule's
    # own atoms + real bonds + features together, correctly, without manual
    # edge-index math).
    graph.ndata["feat"] = atom_features
    mol_graphs = dgl.unbatch(graph)
    n_molecules = len(mol_graphs)
    perm = np.random.permutation(n_molecules)
    n_train = int(n_molecules * TRAIN_FRACTION)
    train_mol_idx, test_mol_idx = perm[:n_train], perm[n_train:]
    mol_order = np.concatenate([train_mol_idx, test_mol_idx])

    reordered_graph = dgl.batch([mol_graphs[i] for i in mol_order]).to(device)
    atom_features_reordered = reordered_graph.ndata["feat"]
    counts_reordered = reordered_graph.batch_num_nodes()
    molecule_id = torch.repeat_interleave(torch.arange(n_molecules, device=device), counts_reordered)
    eval_pos_atoms = int(counts_reordered[:n_train].sum().item())
    eval_pos_molecules = n_train

    y_reordered = y_per_molecule[torch.from_numpy(mol_order)].to(device)
    y_mean = y_reordered[:n_train].mean()
    y_std = y_reordered[:n_train].std().clamp(min=1e-6)
    y_norm = (y_reordered - y_mean) / y_std

    n_atoms = atom_features_reordered.shape[0]
    feature_to_add = n_features % features_per_group
    if feature_to_add > 0:
        pad = torch.zeros(
            n_atoms, features_per_group - feature_to_add, dtype=atom_features_reordered.dtype, device=device
        )
        atom_features_reordered = torch.cat([atom_features_reordered, pad], dim=-1)
    n_groups = atom_features_reordered.shape[-1] // features_per_group

    def encode_atoms_frozen() -> torch.Tensor:
        """x_preprocess -> process_4_x -> encoder_x, all frozen, no_grad
        (matches FeaturesTransformer.forward's exact order, called manually
        since our "sequence" for this stage is ATOMS, not molecules).
        """
        x = atom_features_reordered.unsqueeze(0)  # (1, n_atoms, n_features_padded)
        x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
        x_dict = {k: v.reshape(1, n_atoms, n_groups, features_per_group) for k, v in x_dict.items()}
        x_dict["eval_pos"] = eval_pos_atoms
        with torch.no_grad():
            preprocessed = model.x_preprocess(x_dict)
            preprocessed = model.process_4_x(preprocessed)
            x_encoder_result = model.encoder_x(preprocessed)
        return x_encoder_result["data"].squeeze(0)  # (n_atoms, n_groups, embed_dim)

    if is_main:
        print("Encoding atoms with frozen encoder_x (one-time, features don't change across epochs)...")
    atom_embeddings_grouped = encode_atoms_frozen()

    # Shard QUERY molecules round-robin across ranks. Context (the first
    # eval_pos_molecules indices) is shared/duplicated on every rank -- every
    # query prediction needs to attend over the full context.
    query_indices_global = np.arange(eval_pos_molecules, n_molecules)
    local_query_indices = torch.from_numpy(query_indices_global[rank::world_size]).to(device)
    context_indices = torch.arange(eval_pos_molecules, device=device)
    local_indices = torch.cat([context_indices, local_query_indices])
    n_local_query = local_query_indices.shape[0]

    pooler_without_ddp = PoolingGNN(embed_dim=embed_dim, n_layers=N_GNN_LAYERS, dropout=DROPOUT).to(device)
    pooler = pooler_without_ddp
    if lib.is_ddp():
        # device_ids is only valid for GPU modules; CPU (or CPU-fallback,
        # e.g. no CUDA available) DDP must omit it entirely.
        ddp_device_ids = [lib.get_local_rank()] if device.type == "cuda" else None
        pooler = DistributedDataParallel(pooler_without_ddp, device_ids=ddp_device_ids)

    params = lib.deep.make_parameter_groups(pooler_without_ddp)
    optimizer = lib.deep.make_optimizer(type=OPTIMIZER_TYPE, lr=LR, weight_decay=WEIGHT_DECAY, params=params)
    lr_scheduler = lib.deep.get_lr_scheduler(
        optimizer, n_warmup_steps=N_WARMUP_STEPS, n_steps=N_EPOCHS, scheduler=LR_SCHEDULER
    )
    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler_without_ddp, device, multi_avg_fn=ema_multi_avg_fn)

    def forward_pass(
        indices: torch.Tensor, n_local_eval_pos: int, pooler_module: nn.Module
    ) -> torch.Tensor:
        """indices: molecule indices to include in this forward pass's
        sequence (context + some slice of query); n_local_eval_pos: how many
        of the leading `indices` are context (always eval_pos_molecules).
        Returns predictions for the query tail of `indices`.
        """
        pooled_grouped_full = pooler_module(reordered_graph, atom_embeddings_grouped, molecule_id, n_molecules)
        pooled_grouped = pooled_grouped_full[indices]
        pooled_x = pooled_grouped.unsqueeze(0).to(next(model.encoder_x.parameters()).dtype)

        embedded_x = model.add_embeddings(pooled_x)  # (1, len(indices), n_groups, embed_dim)

        y_local = y_norm[indices].unsqueeze(0).unsqueeze(-1).clone()  # (1, len(indices), 1)
        y_dict = {"data": y_local}
        y_dict["data"][:, n_local_eval_pos:] = torch.nan
        y_type = torch.ones_like(y_dict["data"])  # all regression
        embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=n_local_eval_pos)

        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2).to(embedded_x.dtype)), dim=2)
        encoder_out = model.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=n_local_eval_pos)[0]
        encoder_out = model.encoder_out_norm(encoder_out)

        test_encoder_out = encoder_out[:, n_local_eval_pos:, -1]
        test_y_type = y_type[:, n_local_eval_pos:]
        _, reg_output = model.y_decoder(test_encoder_out, test_y_type)
        return reg_output.float().squeeze(0).squeeze(-1)  # (n_local_query,)

    y_test_target_local = y_norm[local_query_indices]
    y_test_target_full = y_norm[eval_pos_molecules:]

    def gather_predictions(local_pred: torch.Tensor) -> torch.Tensor | None:
        """Reconstructs one global prediction vector (query order) from every
        rank's local shard, on rank 0 only. Returns None on other ranks.
        """
        if not lib.is_ddp():
            return local_pred
        gathered: list[tuple[np.ndarray, np.ndarray]] = [None] * world_size  # type: ignore
        torch.distributed.all_gather_object(
            gathered, (query_indices_global[rank::world_size], local_pred.detach().cpu().numpy())
        )
        if not is_main:
            return None
        full = np.empty(n_molecules - eval_pos_molecules, dtype=np.float32)
        for idx_shard, pred_shard in gathered:
            full[idx_shard - eval_pos_molecules] = pred_shard
        return torch.from_numpy(full).to(device)

    best_test_r2 = -float("inf")
    best_epoch = -1

    iterator = range(N_EPOCHS)
    if is_main:
        iterator = tqdm(iterator, desc="training")

    for epoch in iterator:
        pooler.train()
        optimizer.zero_grad()
        pred = forward_pass(local_indices, eval_pos_molecules, pooler)
        loss = F.mse_loss(pred, y_test_target_local)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pooler.parameters(), GRADIENT_CLIPPING_NORM)
        optimizer.step()
        pooler_ema.update_parameters(pooler_without_ddp)
        lr_scheduler.step()

        if is_main and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=loss.item(), grad_norm=grad_norm.item(), lr=lib.deep.get_lr(optimizer))

        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            pooler_ema.eval()
            with torch.no_grad():
                pred = forward_pass(local_indices, eval_pos_molecules, pooler_ema)
                full_pred = gather_predictions(pred)
                if is_main:
                    test_r2 = r2_score(full_pred, y_test_target_full)
                    if test_r2 > best_test_r2:
                        best_test_r2 = test_r2
                        best_epoch = epoch
                    tqdm.write(
                        f"epoch {epoch:4d} | loss {loss.item():.4f} | grad_norm {grad_norm.item():.4f} | "
                        f"lr {lib.deep.get_lr(optimizer):.6f} | test R2 (EMA) {test_r2:.4f}"
                    )
            lib.barrier()

    if is_main:
        print(f"\nbest test R2 = {best_test_r2:.4f} at epoch {best_epoch} (early-stopping proxy)")


if __name__ == "__main__":
    main()
