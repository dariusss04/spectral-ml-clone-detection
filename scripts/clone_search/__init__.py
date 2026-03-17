"""Clone search entrypoints."""

from .clone_search_deepset_full import main as run_deepset_full
from .clone_search_deepset_eigen_only import main as run_deepset_eigen_only
from .clone_search_self_attention_eigen_only import main as run_self_attention_eigen_only
from .clone_search_mlp_full import main as run_mlp_full
from .clone_search_mlp_eigen_only import main as run_mlp_eigen_only
from .clone_search_pss_full import main as run_pss_full
from .clone_search_pss_eigen_only import main as run_pss_eigen_only

__all__ = [
    "run_deepset_full",
    "run_deepset_eigen_only",
    "run_self_attention_eigen_only",
    "run_mlp_full",
    "run_mlp_eigen_only",
    "run_pss_full",
    "run_pss_eigen_only",
]
