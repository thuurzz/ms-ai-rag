from app.core.config import settings
from app.core.vector_store import VectorStoreAdapter
from app.adapters import ChromaDBAdapter, PineconeAdapter, MongoDBAdapter


class VectorStoreFactory:
    """Factory para criar a instância apropriada de VectorStoreAdapter."""

    @staticmethod
    def create_vector_store() -> VectorStoreAdapter:
        """
        Cria uma instância do adaptador de banco vetorial baseado na configuração.

        Returns:
            Instância de VectorStoreAdapter

        Raises:
            ValueError: Se o tipo de vector store configurado não é suportado
        """
        store_type = settings.VECTOR_STORE_TYPE.lower()

        if store_type == "chromadb":
            return ChromaDBAdapter(model_name=settings.EMBEDDING_MODEL)

        elif store_type == "pinecone":
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY não configurada")
            return PineconeAdapter(
                api_key=settings.PINECONE_API_KEY,
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
                model_name=settings.EMBEDDING_MODEL
            )

        elif store_type == "mongodb":
            return MongoDBAdapter(
                connection_string=settings.MONGODB_CONNECTION_STRING,
                database_name=settings.MONGODB_DATABASE_NAME,
                model_name=settings.EMBEDDING_MODEL
            )

        else:
            raise ValueError(
                f"Vector store '{store_type}' não suportado. "
                f"Use: chromadb, pinecone ou mongodb"
            )
