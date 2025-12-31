from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Configurações da aplicação."""

    # Servidor
    API_TITLE: str = "MS AI RAG"
    API_DESCRIPTION: str = "Microsserviço para processamento de PDFs e RAG com IA"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Vector Store
    VECTOR_STORE_TYPE: Literal["chromadb", "pinecone", "mongodb"] = "chromadb"

    # ChromaDB
    CHROMADB_PERSIST_DIRECTORY: str = "./chroma_data"

    # Pinecone (v3.0+)
    PINECONE_API_KEY: str = ""
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # MongoDB
    MONGODB_CONNECTION_STRING: str = "mongodb://localhost:27017"
    MONGODB_DATABASE_NAME: str = "rag_system"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # PDF Processing
    PDF_CHUNK_SIZE: int = 500
    PDF_CHUNK_OVERLAP: int = 50
    MAX_PDF_SIZE_MB: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instância global das configurações
settings = Settings()
