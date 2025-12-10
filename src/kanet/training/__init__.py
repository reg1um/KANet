"""Training utilities for KANet."""

from .pruning import (
    SparsityRegularizer,
    KANPruner,
    KANTrainerWithPruning,
)

__all__ = [
    "SparsityRegularizer",
    "KANPruner", 
    "KANTrainerWithPruning",
]
