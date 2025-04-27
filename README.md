# DAGMM for Anomaly Detection

This repository implements DAGMM (Deep Autoencoding Gaussian Mixture Model) for unsupervised anomaly detection on multiple datasets including KDDCup99, KDDCUP-Rev, Arrhythmia, and Thyroid.

## Structure
- `model.py` — DAGMM model definition.
- `solver.py` — Training and testing logic.
- `data_loader.py` — Data loading and preprocessing.
- `main.py` — Entrypoint script.
- `DAGMM_KDDCup99.ipynb`, `DAGMM_KDDCUP-Rev.ipynb`, etc. — Jupyter notebooks for experiments.

## Datasets
Small datasets like `kddcup.data_10_percent` and `arrhythmia.data` are included.

## Supported Distributions
The code supports three types of distributions for anomaly detection:
- **Gaussian**
- **Laplace**
- **Student's t-distribution**

## How to Run
1. Install required libraries:
    ```bash
    pip install torch scikit-learn pandas numpy
    ```

2. Open any of the Jupyter notebooks and configure settings inside the notebook if needed (like distribution type(`dist_type`), student_nu, etc.).

3. Run all cells to train and test the model.
