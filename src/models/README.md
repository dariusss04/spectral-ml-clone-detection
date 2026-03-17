# Models

Model families:
- **MLP baselines**: `siamese_mlp_relu_base.py` and 9/12‑layer variants with different activations.
- **DeepSets**: `deepset_siamese.py`, `deepset_siamese_eigenonly.py`.
- **Self‑Attention**: `deepset_self_attention.py`, `deepset_self_attention_eigenonly.py`.

Input conventions:
- MLPs use concatenated feature vectors (eigenvalues + edge counts → 200 dims).
- DeepSets/Self‑Attention use set‑structured features (B, S, 1).
