# BayesLID-Estimators

This repository implements a Bayesian framework for robust Local Intrinsic Dimensionality (LID) estimation, as described in the paper "A Bayesian Framework for Robust Local Intrinsic Dimensionality Estimation". It includes:

- LID estimators (MLE/Quadratic, Stein, and Brown)
- Pooling methods (arithmetic, harmonic, geometric)
- Sequential update methods (One-Step and Accumulative updates)

The code uses CIFAR-10 and a pretrained ResNet18 (with the final layer removed) as a feature extractor.

## Usage

1. Install the dependencies from `requirements.txt`.
2. Run the main script: `python main.py`.