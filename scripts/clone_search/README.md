# Clone Search Scripts

Clone search scripts load trained weights and evaluate clone retrieval using the
config file in `data/meta/clone_search_config.json`.

## Scripts
- `clone_search_deepset_full.py` — DeepSets (eigen + edges)
- `clone_search_deepset_eigen_only.py` — DeepSets (eigen only)
- `clone_search_self_attention_eigen_only.py` — Self‑attention (eigen only)
- `clone_search_mlp_full.py` — MLP (eigen + edges)
- `clone_search_mlp_eigen_only.py` — MLP (eigen only)
- `clone_search_pss_full.py` — PSS baseline (eigen + edges)
- `clone_search_pss_eigen_only.py` — PSS baseline (eigen only)

## Outputs
Results are written to `results/clone_search/*.csv` by default.
