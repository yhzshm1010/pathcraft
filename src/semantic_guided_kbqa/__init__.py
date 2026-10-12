"""Semantic-guided Freebase KBQA pipeline."""

__version__ = "0.6.2"

from .config import AppConfig
from .pipeline import SemanticGuidedPipeline

__all__ = ["AppConfig", "SemanticGuidedPipeline"]
