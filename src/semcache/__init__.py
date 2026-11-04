"""A semantic cache for language-model calls."""

from .cache import Embedder, Hit, SemanticCache, Stats
from .decorator import semcache
from .embedding import LocalEmbedder
from .store import MemoryStore, Store

__version__ = "0.1.0"

__all__ = [
    "Embedder",
    "Hit",
    "LocalEmbedder",
    "MemoryStore",
    "SemanticCache",
    "Stats",
    "Store",
    "__version__",
    "semcache",
]
