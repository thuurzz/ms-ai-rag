import chromadb
from typing import List
from app.core.vector_store import VectorStoreAdapter, Document, SearchResult
from sentence_transformers import SentenceTransformer


class ChromaDBAdapter(VectorStoreAdapter):
    """Adaptador para ChromaDB como banco vetorial."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o adaptador ChromaDB.

        Args:
            model_name: Modelo de embeddings do Sentence Transformers
        """
        self.client = chromadb.Client()
        self.embedding_model = SentenceTransformer(model_name)
        self.collections = {}

    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """Adiciona documentos ao ChromaDB."""
        try:
            # Obter ou criar coleção
            if collection_name not in self.collections:
                self.collections[collection_name] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

            collection = self.collections[collection_name]

            # Gerar embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode(
                texts, convert_to_tensor=False).tolist()

            # Preparar dados
            ids = [doc.doc_id or f"doc_{i}" for i, doc in enumerate(documents)]
            metadatas = [doc.metadata for doc in documents]

            # Adicionar ao ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )

            return ids
        except Exception as e:
            print(f"Erro ao adicionar documentos no ChromaDB: {str(e)}")
            raise

    async def search(self, query: str, collection_name: str, top_k: int = 5) -> List[SearchResult]:
        """Busca documentos similares no ChromaDB."""
        try:
            if collection_name not in self.collections:
                return []

            collection = self.collections[collection_name]

            # Gerar embedding da query
            query_embedding = self.embedding_model.encode(
                [query], convert_to_tensor=False).tolist()[0]

            # Buscar
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Converter para SearchResult
            search_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # ChromaDB retorna distância, convertemos para score de similaridade
                    score = 1 - distance
                    search_results.append(
                        SearchResult(content=doc, score=score,
                                     metadata=metadata)
                    )

            return search_results
        except Exception as e:
            print(f"Erro ao buscar no ChromaDB: {str(e)}")
            raise

    async def delete_documents(self, doc_ids: List[str], collection_name: str) -> bool:
        """Remove documentos do ChromaDB."""
        try:
            if collection_name not in self.collections:
                return False

            collection = self.collections[collection_name]
            collection.delete(ids=doc_ids)
            return True
        except Exception as e:
            print(f"Erro ao deletar documentos no ChromaDB: {str(e)}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Remove uma coleção do ChromaDB."""
        try:
            if collection_name in self.collections:
                self.client.delete_collection(name=collection_name)
                del self.collections[collection_name]
            return True
        except Exception as e:
            print(f"Erro ao deletar coleção no ChromaDB: {str(e)}")
            return False

    async def health_check(self) -> bool:
        """Verifica a saúde do ChromaDB."""
        try:
            # ChromaDB em memória sempre está saudável se inicializado
            return self.client is not None
        except Exception:
            return False
