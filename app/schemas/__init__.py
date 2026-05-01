from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DocumentUploadRequest(BaseModel):
    """Schema para requisição de upload de documento."""

    collection_name: str = Field(
        ...,
        description="Nome da coleção onde o documento será armazenado"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadados adicionais para o documento"
    )


class DocumentUploadResponse(BaseModel):
    """Schema para resposta de upload de documento."""

    document_id: str = Field(description="ID único do documento")
    filename: str = Field(description="Nome do arquivo")
    collection_name: str = Field(description="Coleção onde foi armazenado")
    chunks_created: int = Field(description="Número de chunks criados")
    chunks_ids: List[str] = Field(description="IDs dos chunks criados")
    status: str = Field(default="success", description="Status do upload")


class SearchQuery(BaseModel):
    """Schema para requisição de busca."""

    query: str = Field(
        ...,
        min_length=1,
        description="Texto a ser buscado"
    )
    collection_name: str = Field(
        ...,
        description="Coleção onde fazer a busca"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Número máximo de resultados"
    )


class SearchResultItem(BaseModel):
    """Um item de resultado de busca."""

    content: str = Field(description="Conteúdo do documento")
    score: float = Field(description="Score de relevância (0-1)")
    metadata: Dict[str, Any] = Field(description="Metadados do documento")


class SearchResponse(BaseModel):
    """Schema para resposta de busca."""

    query: str = Field(description="Query executada")
    collection_name: str = Field(description="Coleção pesquisada")
    results: List[SearchResultItem] = Field(
        description="Resultados encontrados")
    total_results: int = Field(description="Total de resultados")


class ChunkItem(BaseModel):
    """Representa um chunk armazenado."""

    chunk_id: str = Field(description="ID do chunk")
    chunk_index: int = Field(description="Índice do chunk no documento")
    content: str = Field(description="Conteúdo textual do chunk")
    metadata: Dict[str, Any] = Field(description="Metadados do chunk")


class DocumentWithChunksItem(BaseModel):
    """Representa um documento e seus chunks."""

    document_id: str = Field(description="ID lógico do documento")
    source_file: Optional[str] = Field(
        default=None,
        description="Arquivo de origem",
    )
    chunk_total: int = Field(description="Quantidade total de chunks do documento")
    metadata: Dict[str, Any] = Field(description="Metadados do documento")
    chunks: List[ChunkItem] = Field(description="Lista de chunks do documento")


class CollectionDocumentsResponse(BaseModel):
    """Resposta de listagem de documentos/chunks por coleção."""

    collection_name: str = Field(description="Nome da coleção")
    total_documents: int = Field(description="Total de documentos")
    total_chunks: int = Field(description="Total de chunks")
    page: int = Field(description="Página atual")
    page_size: int = Field(description="Tamanho da página")
    total_pages: int = Field(description="Total de páginas")
    document_id_filter: Optional[str] = Field(
        default=None,
        description="Filtro aplicado por document_id",
    )
    documents: List[DocumentWithChunksItem] = Field(
        description="Documentos com seus respectivos chunks"
    )


class CollectionListResponse(BaseModel):
    """Resposta de listagem de coleções."""

    total_collections: int = Field(description="Total de coleções")
    collections: List[str] = Field(description="Nomes das coleções")


class HealthCheckResponse(BaseModel):
    """Schema para resposta de health check."""

    status: str = Field(description="Status da aplicação")
    vector_store: str = Field(description="Tipo de vector store")
    vector_store_healthy: bool = Field(
        description="Vector store está operacional")


class ErrorResponse(BaseModel):
    """Schema para respostas de erro."""

    detail: str = Field(description="Descrição do erro")
    error_code: str = Field(description="Código do erro")
