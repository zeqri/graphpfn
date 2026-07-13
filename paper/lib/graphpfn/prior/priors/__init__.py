"""Prior pipelines that combine building blocks to generate datasets."""

from .attributes_then_graph import (
    sample_dataset as sample_dataset_attributes_then_graph,
)
from .graph_level import sample_dataset as sample_dataset_graph_level
from .graph_then_attributes import (
    sample_dataset as sample_dataset_graph_then_attributes,
)

from ..prior_typings import PriorConfig, PriorDataset


def sample_dataset(config: PriorConfig) -> PriorDataset:
    match config["_type_"]:
        case "graph_then_attributes":
            return sample_dataset_graph_then_attributes(config)  # type: ignore
        case "attributes_then_graph":
            return sample_dataset_attributes_then_graph(config)  # type: ignore
        case "graph_level":
            return sample_dataset_graph_level(config)  # type: ignore
        case _:
            raise ValueError(f"Unknown prior type: {config['_type_']}")


__all__ = [
    "sample_dataset",
    "sample_dataset_attributes_then_graph",
    "sample_dataset_graph_level",
    "sample_dataset_graph_then_attributes",
]
