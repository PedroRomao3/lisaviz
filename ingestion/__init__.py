"""Data Ingestion layer (depends on Domain; hides HDF5)."""

from .hdf5_adapter import HDF5RepositoryAdapter
from .mock import InMemoryMockRepository
from .pytables import ChunkLayoutError

__all__ = ["HDF5RepositoryAdapter", "InMemoryMockRepository", "ChunkLayoutError"]
