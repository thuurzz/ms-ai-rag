from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Document:
    """Representação de um documento no sistema."""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id


class SearchResult:
    """Resultado de busca com conteúdo e pontuação de relevância."""

    def __init__(self, content: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.score = score
        self.metadata = metadata or {}


class VectorStoreAdapter(ABC):
    """
    Interface abstrata para adaptadores de bancos de dados vetoriais.
    Permite implementações modulares (ChromaDB, Pinecone, MongoDB, etc.)
    """

    @abstractmethod
    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """
        Adiciona documentos ao banco vetorial.

        Args:
            documents: Lista de documentos a adicionar
            collection_name: Nome da coleção/índice

        Returns:
            Lista de IDs dos documentos adicionados
        """
        pass

    @abstractmethod
    async def search(self, query: str, collection_name: str, top_k: int = 5) -> List[SearchResult]:
        """
        Busca documentos similares no banco vetorial.

        Args:
            query: Texto a buscar
            collection_name: Nome da coleção/índice
            top_k: Número de resultados a retornar

        Returns:
            Lista de resultados de busca ordenados por relevância
        """
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str], collection_name: str) -> bool:
        """
        Remove documentos do banco vetorial.

        Args:
            doc_ids: IDs dos documentos a remover
            collection_name: Nome da coleção/índice

        Returns:
            True se bem-sucedido, False caso contrário
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Remove uma coleção inteira.

        Args:
            collection_name: Nome da coleção a remover

        Returns:
            True se bem-sucedido, False caso contrário
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica a saúde da conexão com o banco vetorial.

        Returns:
            True se operacional, False caso contrário
        """
        pass
