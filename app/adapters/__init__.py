"""__init__.py para o módulo de adaptadores."""

from app.adapters.chromadb_adapter import ChromaDBAdapter
from app.adapters.pinecone_adapter import PineconeAdapter
from app.adapters.mongodb_adapter import MongoDBAdapter

__all__ = [
    "ChromaDBAdapter",
    "PineconeAdapter",
    "MongoDBAdapter",
]
