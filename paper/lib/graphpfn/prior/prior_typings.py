from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

import torch

from lib.util import TaskType


def unpack(config: Mapping[str, Any]) -> dict[str, Any]:
    """Strip _-prefixed keys for unpacking config into function args."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


# >>> Distribution Config Types


class UniformDistribution(TypedDict):
    _distribution_: Literal["uniform"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: float
    max: float


class LogUniformDistribution(TypedDict):
    _distribution_: Literal["log_uniform"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: float
    max: float


class LogUniformIntDistribution(TypedDict):
    _distribution_: Literal["log_uniform_int"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: int
    max: int


class UniformIntDistribution(TypedDict):
    _distribution_: Literal["uniform_int"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: int
    max: int


class ChoiceDistribution(TypedDict):
    """Uniform random selection from values, with optional weights.

    The optional `weights` dict specifies non-default weights for choices.
    Keys are string representations of choice values, values are weight multipliers.
    Default weight for unlisted choices is 1.0.

    Example:
        {"_distribution_": "choice",
         "values": ["relu", "tanh", "random_fourier"],
         "weights": {"random_fourier": 10}}  # random_fourier is 10x more likely
    """

    _distribution_: Literal["choice"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    values: list[ConfigValue]
    weights: NotRequired[dict[str, float]]


class BetaDistribution(TypedDict):
    _distribution_: Literal["beta"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    alpha: float
    beta: float


class BernoulliDistribution(TypedDict):
    """Sample True with probability p, False with probability 1-p."""

    _distribution_: Literal["bernoulli"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    p: float


class MixedLogUniformDistribution(TypedDict):
    """Bimodal log-uniform: samples from first range with prob p_first, else second."""

    _distribution_: Literal["mixed_log_uniform"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min_first: float
    max_first: float
    min_second: float
    max_second: float
    p_first: float


class UniformIntWithDefaultDistribution(TypedDict):
    """Uniform int in [min, max] with prob (1-p_default), else default."""

    _distribution_: Literal["uniform_int_with_default"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: int
    max: int
    default: int
    p_default: float


class MetaBetaDistribution(TypedDict):
    """Meta-beta: samples alpha/beta from uniform [min, max], then beta * scale."""

    _distribution_: Literal["meta_beta"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    scale: float
    min: float
    max: float


class MetaTruncNormLogScaledDistribution(TypedDict):
    """Log-scaled truncated normal: samples mean/std on log scale.

    Samples:
        log_mean ~ uniform(log(min_mean), log(max_mean))
        log_std ~ uniform(log(min_std), log(max_std))
        mu = exp(log_mean)
        sigma = mu * exp(log_std)
        sample ~ TruncNorm(mu, sigma) + lower_bound
    """

    _distribution_: Literal["meta_trunc_norm_log_scaled"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min_mean: float
    max_mean: float
    min_std: float
    max_std: float
    lower_bound: float
    round: bool


type DistributionSpec = (
    UniformDistribution
    | LogUniformDistribution
    | ChoiceDistribution
    | BetaDistribution
    | BernoulliDistribution
    | MixedLogUniformDistribution
    | MetaBetaDistribution
    | MetaTruncNormLogScaledDistribution
    | LogUniformIntDistribution
    | UniformIntDistribution
    | UniformIntWithDefaultDistribution
)


type ConfigValue = (
    None | bool | int | float | str | DistributionSpec | dict[str, "ConfigValue"]
)

type DistributionConfig = dict[str, ConfigValue]


# >>> Sampled Config Types


# Graph sampler configs
class SBMSamplerConfig(TypedDict):
    _type_: Literal["sbm"]
    n_groups: int
    offdiagonal_coef: float


class GeometricSamplerConfig(TypedDict):
    _type_: Literal["geometric"]
    n_latent_features: int


class PASamplerConfig(TypedDict):
    _type_: Literal["preferential-attachment"]


class ERSamplerConfig(TypedDict):
    _type_: Literal["erdos-renyi"]


class MultiLevelSBMWithPASamplerConfig(TypedDict):
    _type_: Literal["multi-level-sbm-with-pa"]
    pa_nodes_ratio: float
    pa_max_degree: int
    n_first_level_subgraphs: int
    first_level_degree_ratio: float
    min_first_level_n_nodes: int
    n_groups: int
    offdiagonal_coef: float


GraphSamplerConfig = (
    SBMSamplerConfig
    | GeometricSamplerConfig
    | PASamplerConfig
    | ERSamplerConfig
    | MultiLevelSBMWithPASamplerConfig
)


class GraphConfig(TypedDict):
    n_nodes: int
    avg_degree: float
    sampler: GraphSamplerConfig


# SCM sub-configs (shared between GNN and MLP SCMs)


class CausesConfig(TypedDict):
    strategy: Literal["normal", "uniform", "mixed"]
    pre_sample_stats: bool


class NoiseConfig(TypedDict):
    std: float
    pre_sample_std: bool


class InitConfig(TypedDict):
    std: float
    block_wise_dropout: bool
    p_dropout: float
    scale_std_by_dropout: bool


class CausalConfig(TypedDict):
    enabled: bool
    y_is_effect: bool
    in_clique: bool
    sort_features: bool


class StructuralConfig(TypedDict):
    use_degree: bool
    use_pagerank: bool
    lappe_k: int


# SCM configs (discriminated by _type_)


class MLPSCMConfig(TypedDict):
    _type_: Literal["mlp"]
    n_features: int
    n_causes: int
    n_layers: int
    hidden_dim: int
    activation_type: str | DistributionSpec
    causes: CausesConfig
    noise: NoiseConfig
    init: InitConfig
    causal: CausalConfig


class GNNSCMConfig(TypedDict):
    """GNN SCM config nests MLPSCMConfig as `base`, adding graph-specific params."""

    _type_: Literal["gnn"]
    base: MLPSCMConfig
    conv_type: Literal["gcn", "sage-mean", "sage-min", "sage-max", "gt"]
    graph_conv_ratio: float
    structural: StructuralConfig


class RegressionTaskConfig(TypedDict):
    _type_: Literal["regression"]


class BinclassTaskConfig(TypedDict):
    _type_: Literal["binclass"]
    quantile: float
    p_reverse: float


class MulticlassTaskConfig(TypedDict):
    _type_: Literal["multiclass"]
    n_classes: int
    multiclass_type: Literal["rank", "value"]
    p_ordered: float
    p_reverse: float


TaskConfig = RegressionTaskConfig | BinclassTaskConfig | MulticlassTaskConfig


class PostprocessingConfig(TypedDict):
    p_cat: float
    max_categories: int
    permute_features: bool
    permute_labels: bool


class GraphThenAttributesPriorConfig(TypedDict):
    _type_: Literal["graph_then_attributes"]
    graph: GraphConfig
    scm: GNNSCMConfig
    task: TaskConfig
    postprocessing: PostprocessingConfig
    train_ratio: float


class AttributesThenGraphPriorConfig(TypedDict):
    _type_: Literal["attributes_then_graph"]
    scm: MLPSCMConfig
    graph: GraphConfig
    task: TaskConfig
    postprocessing: PostprocessingConfig
    train_ratio: float


class SmallGraphStructureConfig(TypedDict):
    """Bounds for one small graph's structure, resampled independently by
    `graph_level.py` for each of the `n_graphs` graphs in one dataset (unlike
    `GraphConfig` above, whose fields are resolved once per whole dataset).
    """

    min_nodes: int
    max_nodes: int
    avg_degree_min: float
    avg_degree_max: float
    n_groups_min: int
    n_groups_max: int
    offdiagonal_coef: float


class GraphLevelPriorConfig(TypedDict):
    """Many small graphs, one virtual (readout) node each, disjoint-unioned.

    Only virtual nodes ever carry a label; real nodes are structural-only.
    Each of the `n_graphs` small graphs gets its own independently sampled
    size/degree (from `graph`'s bounds), subject to `total_n_nodes_budget`
    (see graph_level.py), mirroring how `graph.n_nodes` bounds the single
    graph in the node-level priors above. The SCM itself is instantiated
    once for the whole dataset (shared "task" across all graphs), matching
    how one graph's many nodes share one SCM in the node-level priors.
    `train_ratio` is applied to the graph population (not the node
    population): it picks what fraction of virtual nodes are context.
    """

    _type_: Literal["graph_level"]
    total_n_nodes_budget: int
    n_graphs: int
    graph: SmallGraphStructureConfig
    scm: GNNSCMConfig
    postprocessing: PostprocessingConfig
    train_ratio: float
    label_aggregation: Literal["mean", "sum", "max", "min"]


PriorConfig = (
    GraphThenAttributesPriorConfig | AttributesThenGraphPriorConfig | GraphLevelPriorConfig
)


class SanityCheckConfig(TypedDict):
    min_features: int


class SampledConfig(TypedDict):
    sanity_check: SanityCheckConfig
    prior: PriorConfig


# >>> Prior Data Types
# NOTE: Split is contiguous: train=[0, n_train_nodes), test=[n_train_nodes, n_nodes)


class PriorDataset(TypedDict):
    features: torch.Tensor
    labels: torch.Tensor
    edges: torch.Tensor
    n_train_nodes: int
    task_type: TaskType
    # True for nodes eligible to ever be context/query (train or test); False
    # for structural-only nodes (e.g. the real/atom nodes in the graph_level
    # prior) that get a forward pass but never contribute a label or a loss
    # term. Node-level priors mark every node True (unchanged behavior).
    labeled_mask: torch.Tensor
    # Which nodes' features are representative enough to fit normalization
    # stats on. Node-level priors reuse train=[0, n_train_nodes) (unchanged
    # behavior). The graph_level prior instead marks real/atom nodes, since
    # its train (context virtual-node) rows are all-zero placeholders and
    # would otherwise make every feature column look constant/zero-variance.
    feature_fit_mask: torch.Tensor


class PriorDatasetBatch(TypedDict):
    features: torch.Tensor
    labels: torch.Tensor
    edges: torch.Tensor
    n_nodes: torch.Tensor
    n_features: torch.Tensor
    n_edges: torch.Tensor
    n_train_nodes: int
    task_type: TaskType
    labeled_mask: torch.Tensor
    feature_fit_mask: torch.Tensor
