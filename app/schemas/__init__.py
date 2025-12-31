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
