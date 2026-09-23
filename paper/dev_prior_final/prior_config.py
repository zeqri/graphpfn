"""Base GraphPFN prior config for multi-molecule (graph-level) datasets: molecule-skeleton graphs
with a GNN MLP-SCM producing atom features and one label per molecule. train_pooler.py widens it
(molecule counts, randomized conv_type / structural features, calibrated topology, causal SCM)."""

BASE_PRIOR_CONFIG = {
    "sanity_check": {"min_features": 4, "min_train_ratio": 0.05, "max_train_ratio": 0.5},
    "prior": {
        "_distribution_": "choice",
        "_shared_": True,
        "values": [
            {
                "_type_": "graph_then_attributes",
                "graph": {
                    "n_nodes": {"_distribution_": "choice", "values": [3000], "_shared_": True},
                    "avg_degree": {"_distribution_": "choice", "values": [2.2], "_shared_": True},
                    "sampler": {
                        "_distribution_": "choice",
                        "values": [
                            {
                                "_type_": "multi-graph",
                                "size_jitter": 0.15,
                                "n_graphs": {"_distribution_": "choice", "values": [3000]},
                                "base_n_nodes": {"_distribution_": "choice", "values": [25]},
                                "sub_graph": {
                                    "avg_degree": {"_distribution_": "choice", "values": [2.2]},
                                    "sampler": {
                                        "_distribution_": "choice",
                                        "values": [
                                            {
                                                "_type_": "molecule-skeleton",
                                                "heavy_atom_fraction": 0.42,
                                                "min_ring_size": 3,
                                                "max_ring_size": 7,
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    },
                },
                "scm": {
                    "_type_": "gnn",
                    "base": {
                        "_type_": "mlp",
                        "n_features": {"_distribution_": "choice", "values": [16], "_shared_": True},
                        "n_layers": {"_distribution_": "choice", "values": [3]},
                        "hidden_dim": {"_distribution_": "choice", "values": [32]},
                        "activation_type": {"_distribution_": "choice", "values": ["tanh"]},
                        "causes": {
                            "strategy": {"_distribution_": "choice", "values": ["normal"]},
                            "pre_sample_stats": {"_distribution_": "choice", "values": [False]},
                        },
                        "noise": {
                            "std": {"_distribution_": "choice", "values": [0.02]},
                            "pre_sample_std": {"_distribution_": "choice", "values": [False]},
                        },
                        "init": {
                            "std": {"_distribution_": "choice", "values": [1.0]},
                            "block_wise_dropout": {"_distribution_": "choice", "values": [False]},
                            "p_dropout": {"_distribution_": "choice", "values": [0.0]},
                            "scale_std_by_dropout": {"_distribution_": "choice", "values": [True]},
                        },
                        "causal": {
                            "enabled": {"_distribution_": "choice", "values": [False]},
                            "y_is_effect": {"_distribution_": "choice", "values": [True]},
                            "in_clique": {"_distribution_": "choice", "values": [True]},
                            "sort_features": {"_distribution_": "choice", "values": [True]},
                        },
                    },
                    "conv_type": {"_distribution_": "choice", "values": ["sage-mean"]},
                    "graph_conv_ratio": {"_distribution_": "choice", "values": [1.0]},
                    "structural": {
                        "use_degree": {"_distribution_": "choice", "values": [True]},
                        "use_pagerank": {"_distribution_": "choice", "values": [False]},
                        "lappe_k": 0,
                    },
                },
                "postprocessing": {"p_cat": 0.0, "max_categories": 256, "permute_features": False},
                "train_ratio": {"_distribution_": "choice", "values": [0.8], "_shared_": True},
            }
        ],
    },
}
