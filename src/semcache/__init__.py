"""A semantic cache for language-model calls."""

from .cache import Embedder, Hit, SemanticCache, Stats
from .store import MemoryStore, Store

__version__ = "0.1.0"

__all__ = ["Embedder", "Hit", "MemoryStore", "SemanticCache", "Stats", "Store", "__version__"]
