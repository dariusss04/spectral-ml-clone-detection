"""Metadata generation utilities."""

from .summarize_projects import main as summarize_projects
from .build_clone_search_config import main as build_clone_search_config

__all__ = [
    "summarize_projects",
    "build_clone_search_config",
]
