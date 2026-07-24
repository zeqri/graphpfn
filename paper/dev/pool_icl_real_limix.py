"""Sim-to-real transfer check: train the graph-adapter purely on the
synthetic prior, but evaluate on real QM9 molecules instead of more
synthetic prior samples (or vice versa, via --eval_on). This tests whether
whatever the adapter learns from synthetic molecule-skeleton data actually
generalizes to real chemistry, rather than just to more of the same
generative process.

Architecture:
  - each molecule's atoms -> encoder_x (feature_dim -> embed_dim), per atom
  - GeometricAtomAttention: one round of distance-biased self-attention
    among a molecule's own atoms, using real bond distances (the prior's own
    edge_distance for synthetic data, real 3D-coordinate-derived distances
    for QM9) and reusing the exact EdgeDistanceEncoder (RBF expansion +
    linear bias) from lib.graphpfn.model -- the same mechanism the real
    graph adapter uses, just applied to atoms before pooling instead of to
    pooled molecule tokens after.
  - AttentionPooler: learnable Set-Transformer-style pooling (a learnable
    query cross-attends over the geometrically-enhanced atom tokens),
    replacing a plain mean -- pooling weights are learned, not uniform.
  - the graph-adapter's message-passing slot (inside LimiX's 12 wrapped
    layers) is repurposed to full (fully connected) attention among the
    pooled per-molecule tokens, repeating at every layer.
  - manually replicates FeaturesTransformer.forward (bypassing it as a black
    box, since it hard-codes the label tensor's sequence length to match the
    un-pooled sample count) to insert pooling right after encoder_x.

No feature-dimension matching is needed between training (synthetic, fixed
at N_FEATURES_FIXED=16) and eval (real QM9, 11-dim data.x): LimiX's own
per-feature-group tokenization handles variable feature counts natively,
and the graph-adapter/pooler/geometric-attention all operate purely in
embed_dim space, so a model whose adapter was only ever trained on 16-dim
synthetic features can still be run forward on 11-dim real ones with no
code changes.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import dgl

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

# >>> Point the checkpoint loader at the local copy instead of HF Hub (no
# network access from compute nodes).
import lib.tfm.limix as limix_mod  # noqa: E402
_LOCAL_CKPT = str(REPO / "checkpoints/LimiX-16M.ckpt")
limix_mod._download_limix_checkpoint = lambda: _LOCAL_CKPT

from lib.graphpfn.model import GraphPFN, SDPAInput, EdgeDistanceEncoder  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.attributes import sample_attributes_gnn  # noqa: E402
from lib.graphpfn.prior.postprocessing import process_features  # noqa: E402
from lib.util import TaskType  # noqa: E402

import tomllib  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from torch_geometric.datasets import QM9  # noqa: E402

TOML_PATH = REPO / "exp/graphpfn/pretrain/multigraph_molecule_geometric_rbf_only_warmstart/pretrain.toml"
SENTINEL_DISTANCE = 0.0
N_FEATURES_FIXED = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QM9_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9"
QM9_TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298",
    "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom", "A", "B", "C",
]


def load_base_prior_config() -> dict:
    with open(TOML_PATH, "rb") as f:
        toml = tomllib.load(f)
    return toml["base_config"]["prior"]


def add_virtual_node_and_get_labels(graph: dgl.DGLGraph, scm_config: dict):
    """Star-connect one virtual node per sub-graph, run the GNN-SCM to get a
    graph-level label per molecule, and ALSO return each molecule's own real
    atom-atom edges + bond distances (from BEFORE virtual-node augmentation,
    i.e. the actual molecule-skeleton bonds) so the caller can run geometric
    attention over real chemistry, not the virtual star edges.
    """
    counts = graph.batch_num_nodes().tolist()
    n_atoms_total = graph.num_nodes()
    n_graphs = len(counts)

    orig_src, orig_dst = graph.edges()
    orig_src, orig_dst = orig_src.numpy(), orig_dst.numpy()
    orig_distance = graph.edata.get("distance")
    orig_distance = orig_distance.numpy() if orig_distance is not None else np.full(len(orig_src), SENTINEL_DISTANCE, dtype=np.float32)

    atom_idx_per_molecule: list[np.ndarray] = []
    edges_per_molecule: list[tuple[np.ndarray, np.ndarray]] = []
    distances_per_molecule: list[np.ndarray] = []
    offset = 0
    for n in counts:
        lo, hi = offset, offset + n
        atom_idx_per_molecule.append(np.arange(lo, hi))
        in_range = (orig_src >= lo) & (orig_src < hi) & (orig_dst >= lo) & (orig_dst < hi)
        edges_per_molecule.append((orig_src[in_range] - lo, orig_dst[in_range] - lo))
        distances_per_molecule.append(orig_distance[in_range])
        offset += n

    # Now build the virtual-node-augmented graph (for label generation only).
    src, dst, distance = orig_src.tolist(), orig_dst.tolist(), orig_distance.tolist()
    new_src, new_dst, new_distance = list(src), list(dst), list(distance)
    offset = 0
    for g_i, n in enumerate(counts):
        virtual_idx = n_atoms_total + g_i
        for a in range(offset, offset + n):
            new_src.append(a); new_dst.append(virtual_idx); new_distance.append(SENTINEL_DISTANCE)
            new_src.append(virtual_idx); new_dst.append(a); new_distance.append(SENTINEL_DISTANCE)
        offset += n

    total_nodes = n_atoms_total + n_graphs
    graph_aug = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)), num_nodes=total_nodes)
    graph_aug.edata["distance"] = torch.tensor(new_distance, dtype=torch.float32)

    X, y = sample_attributes_gnn(graph_aug, scm_config)
    X = process_features(X, p_cat=0.0, max_categories=256, do_permute_features=False)
    X, y = X.numpy(), y.numpy()

    atom_features_per_molecule = [X[atoms] for atoms in atom_idx_per_molecule]
    y_per_molecule = y[n_atoms_total:]
    return atom_features_per_molecule, edges_per_molecule, distances_per_molecule, y_per_molecule


def sample_one_dataset(base_prior_config: dict):
    configs = sample_configs(base_prior_config, batch_size=1)
    config = configs[0]["prior"]
    config["scm"]["base"]["n_features"] = N_FEATURES_FIXED

    sampler_cfg = config["graph"]["sampler"]
    graph = sample_multi_graph(
        n_graphs=sampler_cfg["n_graphs"],
        sub_graph=sampler_cfg["sub_graph"],
        base_n_nodes=sampler_cfg.get("base_n_nodes"),
        size_jitter=sampler_cfg.get("size_jitter", 0.0),
    )
    atom_features_per_molecule, edges_per_molecule, distances_per_molecule, y_per_molecule = \
        add_virtual_node_and_get_labels(graph, config["scm"])
    train_ratio = config["train_ratio"]
    return atom_features_per_molecule, edges_per_molecule, distances_per_molecule, y_per_molecule, train_ratio


def load_qm9():
    return QM9(root=QM9_ROOT)


def sample_one_qm9_dataset(dataset, n_graphs_range: tuple[int, int], target_idx: int, rng: np.random.Generator):
    """Draw a random subset of real QM9 molecules as one eval 'dataset' --
    same output shape as sample_one_dataset. Real per-atom features (PyG's
    own 11-dim data.x), real bonds (data.edge_index), and real bond
    distances computed from data.pos (actual 3D conformer coordinates, not
    a fitted prior) -- one real graph-level label per molecule (data.y); no
    virtual node / SCM needed since QM9 already provides the label.
    """
    n_graphs = int(rng.integers(n_graphs_range[0], n_graphs_range[1] + 1))
    mol_indices = rng.choice(len(dataset), size=n_graphs, replace=False)
    atom_features_per_molecule = []
    edges_per_molecule = []
    distances_per_molecule = []
    labels = []
    for idx in mol_indices:
        d = dataset[int(idx)]
        atom_features_per_molecule.append(d.x.numpy())
        pos = d.pos.numpy()
        src, dst = d.edge_index.numpy()
        dist = np.linalg.norm(pos[src] - pos[dst], axis=-1)
        edges_per_molecule.append((src, dst))
        distances_per_molecule.append(dist)
        labels.append(float(d.y[0, target_idx]))
    return atom_features_per_molecule, edges_per_molecule, distances_per_molecule, np.array(labels)


def context_query_split(n_graphs: int, train_ratio: float, rng: np.random.Generator):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Standardize using ALL nodes (context + query), matching the real
    pipeline's own standard_scale_labels convention (applied at data-
    generation time, before any train/test split exists) -- NOT context-only
    stats. This isn't label leakage: the model still only ever receives
    y_context (context labels) as input regardless of what population the
    scale was fixed from. The earlier context-only version was a deliberate
    but costly choice: with context sizes as small as 2-14 molecules, its
    std estimate could be near-zero, blowing up the standardized target for
    that step -- this is what caused every extreme loss spike seen across
    every ablation in this series, and with the backbone unfrozen it was
    severe enough to corrupt the backbone into NaN permanently (no frozen
    anchor left to recover from a bad step). Standardizing over all 10-30
    molecules in the dataset instead of as few as 2 makes that near-zero-std
    degenerate case far rarer.
    """
    mean = y_all.mean()
    std = y_all.std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std


class GeometricAtomAttention(torch.nn.Module):
    """Distance-biased self-attention among one molecule's own atoms, reusing
    the exact EdgeDistanceEncoder (RBF expansion + linear bias) from
    lib.graphpfn.model -- the same mechanism GraphPFNGraphAttentionModule
    uses for its graph adapter, just applied here to real atoms (using real
    bonds/distances) before pooling, instead of to already-pooled molecule
    tokens after. Restricted to real bonds via an additive -inf bias on
    non-edges (self-loops always allowed, same convention as SDPAInput's
    zero_degree_mask handling).
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.out = torch.nn.Linear(embed_dim, embed_dim)
        self.distance_encoder = EdgeDistanceEncoder(n_heads=n_heads)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, atom_embeddings: torch.Tensor, edges: tuple[np.ndarray, np.ndarray], distances: np.ndarray) -> torch.Tensor:
        # atom_embeddings: (1, n_atoms, n_groups, embed_dim)
        _, n_atoms, n_groups, embed_dim = atom_embeddings.shape
        x = atom_embeddings.squeeze(0).permute(1, 0, 2)  # (n_groups, n_atoms, embed_dim)
        device = x.device

        bias_mask = torch.full((n_atoms, n_atoms), float("-inf"), device=device)
        bias_mask.fill_diagonal_(0.0)
        dist_dense = torch.zeros(n_atoms, n_atoms, device=device)
        src, dst = edges
        if len(src) > 0:
            src_t = torch.as_tensor(src, device=device, dtype=torch.long)
            dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
            dist_t = torch.as_tensor(distances, device=device, dtype=torch.float32)
            bias_mask[dst_t, src_t] = 0.0
            dist_dense[dst_t, src_t] = dist_t

        edge_bias = self.distance_encoder(dist_dense)  # (n_atoms, n_atoms, n_heads)
        edge_bias = edge_bias.permute(2, 0, 1)  # (n_heads, n_atoms, n_atoms)
        full_bias = (edge_bias + bias_mask.unsqueeze(0)).unsqueeze(0).expand(n_groups, -1, -1, -1)

        qkv = self.qkv(x).reshape(n_groups, n_atoms, self.n_heads, 3 * self.d_head)
        q, k, v = qkv.split(self.d_head, dim=-1)
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))  # (n_groups, n_heads, n_atoms, d_head)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q.to(full_bias.dtype), k.to(full_bias.dtype), v.to(full_bias.dtype), attn_mask=full_bias
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(n_groups, n_atoms, embed_dim)
        attn_out = self.out(attn_out.to(x.dtype))

        x = self.norm(x + attn_out)
        return x.permute(1, 0, 2).unsqueeze(0)  # (1, n_atoms, n_groups, embed_dim)


class AttentionPooler(torch.nn.Module):
    """Learnable, Set-Transformer-style pooling (Lee et al. 2019, "Pooling by
    Multihead Attention"): a single learnable query cross-attends over a
    molecule's (geometrically-enhanced) atom tokens, producing a *weighted*
    combination -- weights come from learned attention scores, not a fixed
    uniform 1/n_atoms average.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(1, embed_dim))
        torch.nn.init.xavier_uniform_(self.query)
        self.mha = torch.nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        # atom_embeddings: (1, n_atoms, n_groups, embed_dim)
        _, n_atoms, n_groups, embed_dim = atom_embeddings.shape
        x = atom_embeddings.squeeze(0).permute(1, 0, 2)  # (n_groups, n_atoms, embed_dim)
        query = self.query.unsqueeze(0).expand(n_groups, 1, embed_dim).to(x.dtype)
        pooled, _ = self.mha(query, x, x)  # (n_groups, 1, embed_dim)
        pooled = self.norm(pooled.squeeze(1))  # (n_groups, embed_dim)
        return pooled.unsqueeze(0)  # (1, n_groups, embed_dim)


def build_model(random_adapter_init: bool = True, freeze_backbone: bool = True) -> GraphPFN:
    """Real pretrained LimiX backbone + graph adapter. NOTE: the adapter's own
    weights have no pretrained checkpoint here (only the LimiX-16M.ckpt base
    backbone was supplied) -- GraphPFNGraphAttentionModule zero-inits its
    output projection by default, which would make the adapter a pure no-op
    (identity) at init. We instead construct it with zero_init=False so the
    (untrained, random) adapter actually participates in this one-shot
    forward pass -- otherwise "repurposing message-passing as full
    attention" would have literally zero effect to observe. The
    geometric-attention and pooler modules are NOT zero-init (they have no
    such no-op requirement) and are attached as real submodules so their
    params show up in model.parameters() automatically.

    freeze_backbone=False switches from the ICL/pretraining regime (only the
    adapter + geometric attention + pooler train, LimiX itself stays frozen,
    as GraphPFN's own real pretraining does) to the paper's finetuning
    regime (Section 5.1: "we finetune the entire model") -- GraphPFN.__init__
    already implements this distinction via its own freeze_tfm argument; we
    just expose it here.
    """
    model = GraphPFN(
        edge_head=None,
        feat_head=False,
        layer_ids=list(range(12)),
        freeze_tfm=freeze_backbone,
        random_init_tfm=False,
    )
    if random_adapter_init:
        for idx in range(12):
            layer = model.tfm.module.transformer_encoder.layers[idx]
            for module in [layer.conv.base, layer.mlp.base]:
                for p in module.parameters():
                    if p.dim() >= 2:
                        torch.nn.init.xavier_uniform_(p, gain=0.1)
                    else:
                        torch.nn.init.zeros_(p)
    embed_dim = model.tfm.module.embed_dim
    model.geometric_attention = GeometricAtomAttention(embed_dim=embed_dim)
    model.attention_pooler = AttentionPooler(embed_dim=embed_dim)
    model = model.to(DEVICE)
    model.eval()
    return model


def pooled_forward(model: GraphPFN, atom_features_per_molecule, edges_per_molecule, distances_per_molecule,
                    y_context, train_idx, test_idx, task_type=TaskType.REGRESSION):
    """Manually replicate FeaturesTransformer.forward, inserting geometric
    atom attention + learnable pooling right after encoder_x, and
    repurposing the graph-adapter's message-passing slot as full (fully
    connected) attention among the pooled molecule tokens.
    """
    tfm = model.tfm.module  # the real FeaturesTransformer
    n_graphs = len(atom_features_per_molecule)
    is_context = torch.zeros(n_graphs, dtype=torch.bool, device=DEVICE)
    is_context[train_idx] = True

    # --- reorder so context molecules come first (LimiX's own convention) ---
    perm = torch.argsort((~is_context).float(), stable=True)
    ordered_idx = perm.tolist()
    n_context = int(is_context.sum().item())

    pooled_tokens = []
    for i in ordered_idx:
        atoms = atom_features_per_molecule[i]
        x = torch.tensor(atoms, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, n_atoms, n_features)
        batch_size, seq_len, num_feature = x.shape
        feature_to_add = num_feature % tfm.features_per_group
        if feature_to_add > 0:
            x = torch.cat([x, torch.zeros(batch_size, seq_len, feature_to_add, device=x.device, dtype=x.dtype)], dim=-1)
        x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
        x_dict = {k: v.reshape(batch_size, seq_len, v.shape[2] // tfm.features_per_group, tfm.features_per_group) for k, v in x_dict.items()}
        x_dict["eval_pos"] = seq_len  # irrelevant here, x_preprocess just needs *a* value
        preprocessed = tfm.x_preprocess(x_dict)
        preprocessed = tfm.process_4_x(preprocessed)
        x_encoder_result = tfm.encoder_x(preprocessed)
        x_emb = x_encoder_result["data"]  # (1, n_atoms, n_feature_groups, embed_dim)

        x_emb = model.geometric_attention(x_emb, edges_per_molecule[i], distances_per_molecule[i])
        pooled_tokens.append(model.attention_pooler(x_emb))  # (1, n_feature_groups, embed_dim)

    embedded_x = torch.cat(pooled_tokens, dim=0).unsqueeze(0)  # (1, n_graphs, n_feature_groups, embed_dim)
    embedded_x = tfm.add_embeddings(embedded_x)

    # --- build the y tensor directly at molecule granularity (bypassing
    # FeaturesTransformer.forward's own atom-sized y construction) ---
    y_full = torch.full((1, n_graphs), float("nan"), device=DEVICE, dtype=torch.float32)
    y_context_ordered = torch.tensor(y_context, dtype=torch.float32, device=DEVICE)
    y_full[0, :n_context] = y_context_ordered
    y_dict = {"data": y_full.unsqueeze(-1)}
    y_type = torch.ones_like(y_dict["data"])  # regression
    embedded_y = tfm.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=n_context)

    embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)  # (1, n_graphs, n_feature_groups+1, embed_dim)

    # --- fully connected "graph adapter" mask over the n_graphs pooled tokens ---
    attn_mask = torch.ones(n_graphs, n_graphs, dtype=torch.bool, device=DEVICE)
    zero_degree_mask = torch.zeros(n_graphs, dtype=torch.bool, device=DEVICE)
    model._graph_holder.graph = SDPAInput(attn_mask=attn_mask, zero_degree_mask=zero_degree_mask, edge_distance=None)

    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        encoder_out = tfm.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=n_context)[0]
    model._graph_holder.graph = None

    encoder_out = tfm.encoder_out_norm(encoder_out)
    test_encoder_out = encoder_out[:, n_context:, -1]  # query molecules' y-token
    test_y_type = y_type[:, n_context:, 0]
    _, reg_pred = tfm.y_decoder(test_encoder_out, test_y_type)
    pred_ordered = reg_pred.float().squeeze(0).squeeze(-1)  # (n_query,)

    # map predictions back from "context-first" order to original molecule order
    query_order_orig_idx = perm[n_context:]
    pred_by_orig_idx = torch.zeros(n_graphs, device=DEVICE)
    pred_by_orig_idx[query_order_orig_idx] = pred_ordered
    return pred_by_orig_idx[test_idx]


def train_adapters(model: GraphPFN, base_prior_config: dict, n_steps: int, rng: np.random.Generator,
                    lr: float = 1e-3, max_norm: float = 1.0):
    """Train the graph-adapter params, the geometric-attention module, and
    the learnable pooler jointly (everything with requires_grad=True --
    the pretrained LimiX backbone itself stays frozen). Mirrors the real
    pretraining loop's structure (sample dataset -> forward with context
    revealed -> MSE loss on query -> backward -> clip -> step), just at a
    much smaller scale (single dataset per step, no grad accumulation).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    backbone_state = "frozen" if n_trainable < n_total * 0.5 else "UNFROZEN (finetuning entire model)"
    print(f"training {n_trainable}/{n_total} params (adapters + geometric attention + pooler; backbone {backbone_state})")
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.train()

    losses = []
    step = 0
    attempts = 0
    while step < n_steps and attempts < n_steps * 3:
        attempts += 1
        try:
            atom_feats, edges, distances, y_all, train_ratio = sample_one_dataset(base_prior_config)
        except Exception:
            continue
        n_graphs = len(atom_feats)
        if n_graphs < 4:
            continue
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        if len(test_idx) == 0 or len(train_idx) < 2:
            continue
        y_std = standardize_by_context(y_all, train_idx)

        optimizer.zero_grad()
        pred = pooled_forward(model, atom_feats, edges, distances, y_std[train_idx], train_idx, test_idx)
        loss = torch.nn.functional.mse_loss(
            pred, torch.tensor(y_std[test_idx], dtype=torch.float32, device=DEVICE)
        )
        if not torch.isfinite(loss):
            # Never let a non-finite loss reach backward()/step() -- with the
            # backbone frozen a bad step could only push the small adapter
            # around and recover next step, but with it unfrozen a single
            # non-finite gradient can permanently corrupt the pretrained
            # weights into NaN (as happened at step 300 of the unfrozen run).
            # Skipping here makes that impossible regardless of what's frozen.
            print(f"[train step {step}] non-finite loss ({loss.item()}), skipping step")
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_norm)
        if not torch.isfinite(grad_norm):
            print(f"[train step {step}] non-finite grad norm, skipping step")
            optimizer.zero_grad()
            continue
        optimizer.step()
        losses.append(loss.item())
        step += 1
        if step % 10 == 0:
            print(f"[train step {step}/{n_steps}] recent loss = {np.mean(losses[-10:]):.4f}")

    model.eval()
    return losses


def main(
    n_eval_datasets: int = 60,
    seed: int = 0,
    random_adapter_init: bool = True,
    n_train_steps: int = 0,
    eval_on: str = "qm9",
    qm9_target: str = "cv",
    qm9_n_graphs_min: int = 10,
    qm9_n_graphs_max: int = 30,
    qm9_train_ratio: float = 0.3,
    freeze_backbone: bool = True,
):
    print(f"device = {DEVICE}")
    base_prior_config = load_base_prior_config()
    # When actually training, start from the default zero-init adapter (a
    # true no-op at step 0, same convention the real pretraining pipeline
    # uses) rather than the random-init hack used for the untrained one-shot
    # check -- zero-init + residual wrapping is what makes training stable.
    model = build_model(random_adapter_init=random_adapter_init and n_train_steps == 0, freeze_backbone=freeze_backbone)
    rng = np.random.default_rng(seed)

    if n_train_steps > 0:
        # Training always uses the synthetic prior -- this script is
        # specifically the "train on prior, evaluate on <eval_on>" check.
        train_adapters(model, base_prior_config, n_train_steps, rng)

    qm9_dataset = None
    target_idx = None
    if eval_on == "qm9":
        target_idx = QM9_TARGETS.index(qm9_target)
        print(f"evaluating on real QM9 molecules, target '{qm9_target}' (index {target_idx})")
        qm9_dataset = load_qm9()
        print(f"loaded QM9: {len(qm9_dataset)} molecules (no train/eval pool split needed -- "
              f"training never touches QM9 in this script, so there's no overlap risk)")
    else:
        print("evaluating on freshly sampled synthetic prior datasets (same distribution as training)")

    scores = []
    n_skipped = 0
    i = 0
    attempts = 0
    with torch.no_grad():
        while i < n_eval_datasets and attempts < n_eval_datasets * 3:
            attempts += 1
            try:
                if eval_on == "qm9":
                    atom_feats, edges, distances, y_all = sample_one_qm9_dataset(
                        qm9_dataset, (qm9_n_graphs_min, qm9_n_graphs_max), target_idx, rng
                    )
                    train_ratio = qm9_train_ratio
                else:
                    atom_feats, edges, distances, y_all, train_ratio = sample_one_dataset(base_prior_config)
            except Exception as e:
                n_skipped += 1
                continue
            n_graphs = len(atom_feats)
            if n_graphs < 4:
                n_skipped += 1
                continue
            train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
            if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_all[test_idx]) == 0:
                n_skipped += 1
                continue
            y_std = standardize_by_context(y_all, train_idx)
            try:
                pred = pooled_forward(model, atom_feats, edges, distances, y_std[train_idx], train_idx, test_idx)
            except Exception as e:
                print(f"forward failed: {type(e).__name__}: {e}")
                n_skipped += 1
                continue
            scores.append(r2_score(y_std[test_idx], pred.cpu().numpy()))
            i += 1
            if i % 10 == 0:
                print(f"[{i}/{n_eval_datasets}] evaluated, skipped={n_skipped}")

    arr = np.array(scores)
    print(f"\n=== Train-on-prior, eval-on-{eval_on} Pool+ICL R^2 (n={len(arr)}, skipped={n_skipped}) ===")
    if len(arr) == 0:
        print("no successful evaluations")
        return
    print(f"mean={arr.mean():.3f} median={np.median(arr):.3f}")
    print(f"quantiles [10,25,50,75,90]%: {np.round(np.quantile(arr, [0.1,0.25,0.5,0.75,0.9]), 3)}")
    print(f"frac R2>0.5: {(arr > 0.5).mean():.3f}  frac R2>0.8: {(arr > 0.8).mean():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval_datasets", type=int, default=60)
    parser.add_argument("--n_train_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero_init_adapter", action="store_true")
    parser.add_argument("--eval_on", type=str, default="qm9", choices=["qm9", "prior"])
    parser.add_argument("--qm9_target", type=str, default="cv", choices=QM9_TARGETS)
    parser.add_argument("--qm9_n_graphs_min", type=int, default=10)
    parser.add_argument("--qm9_n_graphs_max", type=int, default=30)
    parser.add_argument("--qm9_train_ratio", type=float, default=0.3)
    parser.add_argument("--unfreeze_backbone", action="store_true",
                         help="Finetune the entire pretrained LimiX backbone too, not just the "
                              "adapter/geometric-attention/pooler (paper's FT regime vs ICL/pretraining regime).")
    args = parser.parse_args()
    main(
        n_eval_datasets=args.n_eval_datasets,
        seed=args.seed,
        random_adapter_init=not args.zero_init_adapter,
        n_train_steps=args.n_train_steps,
        eval_on=args.eval_on,
        qm9_target=args.qm9_target,
        qm9_n_graphs_min=args.qm9_n_graphs_min,
        qm9_n_graphs_max=args.qm9_n_graphs_max,
        qm9_train_ratio=args.qm9_train_ratio,
        freeze_backbone=not args.unfreeze_backbone,
    )
