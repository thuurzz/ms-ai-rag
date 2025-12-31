from typing import List, Optional
from app.core.vector_store import VectorStoreAdapter, Document, SearchResult
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class MongoDBAdapter(VectorStoreAdapter):
    """Adaptador para MongoDB como banco vetorial (com Atlas Vector Search)."""

    def __init__(
        self,
        connection_string: str,
        database_name: str = "rag_system",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Inicializa o adaptador MongoDB.

        Args:
            connection_string: String de conexão MongoDB
            database_name: Nome do banco de dados
            model_name: Modelo de embeddings do Sentence Transformers
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.embedding_model = SentenceTransformer(model_name)

    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """Adiciona documentos ao MongoDB."""
        try:
            collection = self.db[collection_name]

            # Gerar embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode(
                texts, convert_to_tensor=False).tolist()

            # Preparar documentos
            documents_to_insert = []
            ids = []

            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.doc_id or f"doc_{len(documents_to_insert)}"
                ids.append(doc_id)

                documents_to_insert.append({
                    "_id": doc_id,
                    "content": doc.content,
                    "embedding": embedding,
                    "metadata": doc.metadata
                })

            # Inserir no MongoDB
            if documents_to_insert:
                collection.insert_many(documents_to_insert, ordered=False)

            return ids
        except Exception as e:
            print(f"Erro ao adicionar documentos no MongoDB: {str(e)}")
            raise

    async def search(self, query: str, collection_name: str, top_k: int = 5) -> List[SearchResult]:
        """Busca documentos similares no MongoDB usando vector search."""
        try:
            collection = self.db[collection_name]

            # Gerar embedding da query
            query_embedding = self.embedding_model.encode(
                [query], convert_to_tensor=False).tolist()[0]

            # Buscar usando aggregation pipeline com $search (requer Atlas Vector Search)
            # Se não tiver Atlas Vector Search, usa busca aproximada com score calculado
            pipeline = [
                {
                    "$addFields": {
                        "similarity": {
                            "$divide": [
                                {"$reduce": {
                                    "input": {"$zip": {
                                        "inputs": ["$embedding", query_embedding]
                                    }},
                                    "initialValue": 0,
                                    "in": {"$add": [
                                        "$$value",
                                        {"$multiply": [
                                            {"$arrayElemAt": ["$$this", 0]},
                                            {"$arrayElemAt": ["$$this", 1]}
                                        ]}
                                    ]}
                                }},
                                {
                                    "$sqrt": {
                                        "$reduce": {
                                            "input": "$embedding",
                                            "initialValue": 0,
                                            "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": top_k}
            ]

            results = list(collection.aggregate(pipeline))

            # Converter para SearchResult
            search_results = []
            for result in results:
                search_results.append(
                    SearchResult(
                        content=result.get("content", ""),
                        score=result.get("similarity", 0),
                        metadata=result.get("metadata", {})
                    )
                )

            return search_results
        except Exception as e:
            print(f"Erro ao buscar no MongoDB: {str(e)}")
            # Fallback: retornar busca simples se vector search não funcionar
            return []

    async def delete_documents(self, doc_ids: List[str], collection_name: str) -> bool:
        """Remove documentos do MongoDB."""
        try:
            collection = self.db[collection_name]
            result = collection.delete_many({"_id": {"$in": doc_ids}})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Erro ao deletar documentos no MongoDB: {str(e)}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Remove uma coleção do MongoDB."""
        try:
            self.db[collection_name].drop()
            return True
        except Exception as e:
            print(f"Erro ao deletar coleção no MongoDB: {str(e)}")
            return False

    async def health_check(self) -> bool:
        """Verifica a saúde da conexão com MongoDB."""
        try:
            self.client.admin.command("ping")
            return True
        except ConnectionFailure:
            return False
