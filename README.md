# Program Clone Detection (Spectral + ML)

This repository contains the code and experimental artifacts for a bachelor thesis on binary clone detection using
spectral features (eigenvalues, edge counts) and Siamese neural networks. The work extends Program Spectral Similarity (PSS)
by learning embeddings from graph-derived features to improve clone search robustness under compiler/architecture variation.

**Thesis**: "Incorporating Machine Learning into Spectral Analysis for Program Clone Detection" (LMU)

## Dataset
The experiments use a BinKit-derived binary corpus. BinKit is a public benchmark for binary code similarity analysis.
In the thesis, we focus on a 67,680‑binary subset dominated by Coreutils builds, compiled across optimization levels,
compilers, and architectures.

## Repository Structure
- `src/` — Models and utilities (dataset, normalization, split handling).
- `scripts/` — Training, clone search, and metadata generation scripts.
- `data/` — Local dataset layout (raw, features, splits, metadata).
- `experiments/` — Trained weights and training logs (metrics + summary).

## Quickstart
```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (example)
python scripts/training_scripts/train_deepset_full.py

# Run clone search (example)
python scripts/clone_search/clone_search_deepset_full.py
```

## Environment Variables (Optional)
You can override dataset/experiment locations using env vars:
- `PCD_DATA_DIR`, `PCD_SAMPLES_FILE`, `PCD_SPLITS_DIR`
- `PCD_TRAIN_SPLIT_DIR`, `PCD_VAL_SPLIT_DIR`, `PCD_TEST_SPLIT_DIR`
- `PCD_NORMALIZED_TEST_SPLIT_DIR`, `PCD_CLONE_SEARCH_CONFIG`
- `PCD_EXPERIMENTS_DIR`, `PCD_RESULTS_DIR`, `PCD_WEIGHTS_PATH`

## License
This repo is provided under the MIT license (see `LICENSE`).
