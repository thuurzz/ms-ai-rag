"""__init__.py para o módulo core."""

from app.core.config import settings, Settings
from app.core.vector_store import VectorStoreAdapter, Document, SearchResult
from app.core.pdf_processor import PDFProcessor
from app.core.vector_store_factory import VectorStoreFactory

__all__ = [
    "settings",
    "Settings",
    "VectorStoreAdapter",
    "Document",
    "SearchResult",
    "PDFProcessor",
    "VectorStoreFactory",
]
