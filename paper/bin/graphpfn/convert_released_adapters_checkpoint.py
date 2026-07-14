"""One-time conversion: released GraphPFN-1.3 adapter weights -> the internal
`base_checkpoint` format `main()` (bin/graphpfn/pretrain.py) already knows
how to load.

The released checkpoint (`hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt`,
downloaded via `graphpfn.model.graphpfn.resolve_checkpoint`) is a plain
`{"state_dict": {...}}` with 144 entries -- exactly the trainable graph
adapter parameters (`tfm.module.transformer_encoder.layers.{i}.{conv,mlp}...`
for each of the 12 layers), keyed identically to
`lib.graphpfn.model.GraphPFN.state_dict()` (verified: 0 shape mismatches, 0
unexpected keys when loaded with strict=False). Everything else (the frozen
LimiX backbone) is loaded separately from LimiX-16M.ckpt at model
construction time regardless, so this checkpoint intentionally contains
nothing else.

`main()`'s existing `base_checkpoint` loading code expects OUR OWN internal
training-run format instead: `checkpoint["model_ema"]` with every key
prefixed by "module." (an artifact of `torch.optim.swa_utils.AveragedModel`
wrapping). This script bridges the two formats once, offline, so
`base_checkpoint` in a toml can point at a local file and `main()` needs
zero code changes.

Usage (run with cwd=paper/, needs the env active):
    python -m bin.graphpfn.convert_released_adapters_checkpoint
"""

from pathlib import Path

import torch

CHECKPOINT_DIR = Path("checkpoints")
RELEASE_CHECKPOINT = CHECKPOINT_DIR / "graphpfn-adapters-1_3.pt"
OUTPUT_PATH = CHECKPOINT_DIR / "graphpfn-adapters-1_3_internal_format.pt"


def main() -> None:
    if not RELEASE_CHECKPOINT.exists():
        import os
        import sys

        sys.path.insert(0, str(Path("..") / "src"))
        os.environ.setdefault(
            "HF_HOME", str(CHECKPOINT_DIR / ".cache" / "huggingface")
        )
        from graphpfn.model.graphpfn import resolve_checkpoint

        resolve_checkpoint(
            "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt",
            local_dir=str(CHECKPOINT_DIR),
        )
        assert RELEASE_CHECKPOINT.exists()

    release = torch.load(RELEASE_CHECKPOINT, map_location="cpu", weights_only=True)
    state_dict = release["state_dict"]
    print(f"Loaded release checkpoint: {len(state_dict)} tensors from {RELEASE_CHECKPOINT}")

    # Verify against a freshly-constructed internal model before trusting
    # this conversion -- shape mismatches or unexpected keys here would
    # silently produce a garbage warm-start otherwise.
    import tomllib

    from lib.graphpfn.model import GraphPFN

    with open("exp/graphpfn/pretrain/graph_level/pretrain.toml", "rb") as f:
        toml_config = tomllib.load(f)
    model = GraphPFN(**toml_config["base_config"].get("model", {}))
    internal_keys = set(model.state_dict().keys())

    unexpected = set(state_dict.keys()) - internal_keys
    assert not unexpected, f"release checkpoint has keys not in internal model: {unexpected}"
    for k, v in state_dict.items():
        expected_shape = tuple(model.state_dict()[k].shape)
        assert tuple(v.shape) == expected_shape, (
            f"shape mismatch for {k}: release={tuple(v.shape)} internal={expected_shape}"
        )
    print("Verified: all release keys exist in the internal model with matching shapes.")

    wrapped = {"module." + k: v for k, v in state_dict.items()}
    torch.save({"model_ema": wrapped}, OUTPUT_PATH)
    print(f"Wrote internal-format checkpoint ({len(wrapped)} tensors) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
