# Experiments

Each experiment folder contains **exactly three files**:
- `metrics.csv` — Full training/validation loss logs
- `summary.csv` — Best epoch + hyperparameter summary
- `weights.pth` — Best model checkpoint

## Subfolders
- `mlp_activation_sweep/` — 9‑layer MLP activation/regularization variants
- `mlp_depth_sweep/` — Depth sweep for base MLP
- `deepset_full/` — DeepSets with eigenvalues + edge counts
- `deepset_eigen_only/` — DeepSets with eigenvalues only
- `self_attention_full/` — Self‑attention on eigenvalues + edges
- `self_attention_eigen_only/` — Self‑attention on eigenvalues only
- `pss_baselines/` — PSS baselines (eigen‑only / eigen+edges)
