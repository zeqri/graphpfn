# MolPFN research code

This directory contains the MolPFN training and evaluation pipeline and code adapted from [GraphPFN](https://github.com/yandex-research/graphpfn), the implementation accompanying "GraphPFN: A Prior-Data Fitted Graph Foundation Model" ([paper](https://arxiv.org/abs/2509.21489)). This README has been modified for the MolPFN release.

For MolPFN setup and usage, see the root README:

- [Setup](../README.md#setup)
- [Train the pooler](../README.md#1-train-the-pooler)
- [Evaluate](../README.md#2-evaluate)

The original GraphPFN experiment configurations in `exp/` have been removed from this release. To reproduce those experiments, use the [upstream GraphPFN repository](https://github.com/yandex-research/graphpfn). MolPFN builds its synthetic prior from [`dev_prior_final/prior_config.py`](dev_prior_final/prior_config.py); its training and evaluation commands are documented in the root README.

## Project Structure

- `dev_prior_final/` - MolPFN training, synthetic prior configuration, and evaluation scripts
- `bin/` - Retained GraphPFN training and evaluation scripts; their original experiment configurations are omitted
- `lib/` - GraphPFN prior, model code, and utilities, including MolPFN additions and modifications
- `vendor/` - Vendored third-party code

## Licenses

Original GraphPFN license and attribution notices are retained in [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). See also the [root license documentation](../README.md#licenses).

- This project uses third-party components [LimiX](https://github.com/limix-ldm/LimiX), [TabICL](https://github.com/soda-inria/tabicl) and [TabPFN](https://github.com/PriorLabs/TabPFN). See the `NOTICE` file and `LICENSES/` directory for details.
- GraphPFN prior in `lib/graphpfn/prior` is largely based on the [TabICLv1](https://github.com/soda-inria/tabicl) prior.
- LimiX serves as the backbone for GraphPFN, and its weights have a separate license – see the LimiX [repository](https://github.com/limix-ldm/LimiX).
