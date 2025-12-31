from typing import List
from app.core.vector_store import VectorStoreAdapter, Document, SearchResult
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time


class PineconeAdapter(VectorStoreAdapter):
    """Adaptador para Pinecone v3.0+ como banco vetorial."""

    def __init__(self, api_key: str, cloud: str = "aws", region: str = "us-east-1", model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o adaptador Pinecone v3.0+.

        Args:
            api_key: Chave de API do Pinecone
            cloud: Provedor cloud (aws, gcp, azure)
            region: Região (ex: us-east-1)
            model_name: Modelo de embeddings do Sentence Transformers
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.pc = Pinecone(api_key=api_key)
        self.cloud = cloud
        self.region = region
        self.namespace = "documents"  # Namespace padrão para documentos

    def _validate_index_name(self, index_name: str) -> str:
        """
        Valida o nome do índice para ser compatível com Pinecone.

        Pinecone v3.0+ aceita apenas:
        - lowercase alphanumeric
        - hífens (-)

        Não aceita:
        - Underscores (_)
        - Espaços
        - Caracteres especiais
        - MAIÚSCULAS

        Args:
            index_name: Nome do índice a validar

        Raises:
            ValueError: Se o nome for inválido

        Returns:
            Nome validado (em lowercase)
        """
        if " " in index_name:
            raise ValueError(
                f"❌ Nome de coleção inválido: '{index_name}'\n"
                f"Pinecone não aceita espaços.\n"
                f"Sugestão: Use hífens (-) no lugar de espaços.\n"
                f"Exemplo: '{index_name.replace(' ', '-').lower()}'"
            )

        if "_" in index_name:
            raise ValueError(
                f"❌ Nome de coleção inválido: '{index_name}'\n"
                f"Pinecone não aceita underscores (_).\n"
                f"Sugestão: Use hífens (-) no lugar de underscores.\n"
                f"Exemplo: '{index_name.replace('_', '-').lower()}'"
            )

        # Converter para lowercase (sempre fazer isso)
        validated = index_name.lower()

        # Verificar se tem caracteres especiais (exceto hífens)
        if not all(c.isalnum() or c == "-" for c in validated):
            special_chars = "".join(
                set(c for c in validated if not (c.isalnum() or c == "-")))
            raise ValueError(
                f"❌ Nome de coleção inválido: '{index_name}'\n"
                f"Pinecone não aceita caracteres especiais: {special_chars}\n"
                f"Use apenas: letras minúsculas, números e hífens (-)."
            )

        return validated

    def _ensure_index_exists(self, index_name: str):
        """
        Verifica e cria o índice se necessário.

        Args:
            index_name: Nome do índice (já validado)
        """
        if not self.pc.has_index(index_name):
            # Criar índice serverless com dimensão apropriada
            self.pc.create_index(
                name=index_name,
                dimension=384,  # Dimensão do modelo all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            # Aguardar índice ficar pronto
            time.sleep(2)

    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """Adiciona documentos ao Pinecone."""
        try:
            # Validar nome da coleção
            collection_name = self._validate_index_name(collection_name)

            # Garantir que o índice existe
            self._ensure_index_exists(collection_name)

            # Obter índice
            index = self.pc.Index(collection_name)

            # Gerar embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode(
                texts, convert_to_tensor=False
            )

            # Preparar dados para upsert
            records_to_upsert = []
            ids = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = doc.doc_id or f"doc_{i}"
                ids.append(doc_id)

                record = {
                    "id": doc_id,
                    "values": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    "metadata": {
                        "content": doc.content,
                        **doc.metadata
                    }
                }
                records_to_upsert.append(record)

            # Upsert em batches de 100 registros
            batch_size = 100
            for i in range(0, len(records_to_upsert), batch_size):
                batch = records_to_upsert[i:i + batch_size]
                index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )

            return ids

        except Exception as e:
            print(f"Erro ao adicionar documentos no Pinecone: {str(e)}")
            raise

    async def search(self, query: str, collection_name: str, top_k: int = 5) -> List[SearchResult]:
        """Busca documentos similares no Pinecone."""
        try:
            # Validar nome da coleção
            collection_name = self._validate_index_name(collection_name)

            # Verificar se índice existe
            if not self.pc.has_index(collection_name):
                return []

            index = self.pc.Index(collection_name)

            # Gerar embedding da query
            query_embedding = self.embedding_model.encode(
                [query], convert_to_tensor=False)[0]

            # Buscar
            results = index.query(
                namespace=self.namespace,
                vector=query_embedding.tolist() if hasattr(
                    query_embedding, 'tolist') else query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Converter para SearchResult
            search_results = []
            for match in results.matches:
                if match.metadata:
                    content = match.metadata.get("content", "")
                    metadata = {k: v for k,
                                v in match.metadata.items() if k != "content"}
                else:
                    content = ""
                    metadata = {}

                search_results.append(
                    SearchResult(
                        content=content,
                        score=match.score,
                        metadata=metadata
                    )
                )

            return search_results

        except Exception as e:
            print(f"Erro ao buscar no Pinecone: {str(e)}")
            raise

    async def delete_documents(self, doc_ids: List[str], collection_name: str) -> bool:
        """Remove documentos do Pinecone."""
        try:
            # Validar nome da coleção
            collection_name = self._validate_index_name(collection_name)

            if not self.pc.has_index(collection_name):
                return False

            index = self.pc.Index(collection_name)
            index.delete(ids=doc_ids, namespace=self.namespace)
            return True

        except Exception as e:
            print(f"Erro ao deletar documentos no Pinecone: {str(e)}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Remove um índice do Pinecone."""
        try:
            # Validar nome da coleção
            collection_name = self._validate_index_name(collection_name)

            if self.pc.has_index(collection_name):
                self.pc.delete_index(collection_name)
            return True

        except Exception as e:
            print(f"Erro ao deletar índice no Pinecone: {str(e)}")
            return False

    async def health_check(self) -> bool:
        """Verifica a saúde da conexão com Pinecone."""
        try:
            # Tentar listar índices como teste de conexão
            self.pc.list_indexes()
            return True
        except Exception:
            return False
