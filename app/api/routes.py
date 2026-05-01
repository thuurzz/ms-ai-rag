from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from typing import Any, Dict, List, Literal, Optional
import json
import uuid
from datetime import date
from app.core import PDFProcessor, VectorStoreFactory, settings
from app.schemas import (
    DocumentUploadResponse,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    CollectionListResponse,
    CollectionDocumentsResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/api/v1", tags=["RAG Operations"])


ConfidentialityType = Literal["publico_interno", "restrito", "confidencial"]

# Instâncias globais
pdf_processor = PDFProcessor(
    chunk_size=settings.PDF_CHUNK_SIZE,
    chunk_overlap=settings.PDF_CHUNK_OVERLAP,
)
vector_store = VectorStoreFactory.create_vector_store()


def _parse_json_metadata(metadata: Optional[str]) -> Optional[Dict[str, Any]]:
    if metadata is None:
        return None

    raw = metadata.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Metadata inválido: {str(e)}")

    if not isinstance(parsed, dict):
        raise ValueError("Metadata deve ser um objeto JSON")

    return parsed


def _parse_tags(tags: Optional[str]) -> Optional[List[str]]:
    if tags is None:
        return None

    raw = tags.strip()
    if not raw:
        return None

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tags inválidas: {str(e)}")

        if not isinstance(parsed, list):
            raise ValueError("Tags devem ser uma lista JSON ou string separada por vírgulas")

        normalized = [str(item).strip() for item in parsed if str(item).strip()]
        return normalized or None

    normalized = [item.strip() for item in raw.split(",") if item.strip()]
    return normalized or None


def _serialize_metadata_values(metadata: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, date):
            serialized[key] = value.isoformat()
        elif isinstance(value, dict):
            serialized[key] = _serialize_metadata_values(value)
        elif isinstance(value, list):
            serialized[key] = [
                item.isoformat() if isinstance(item, date) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def _build_standard_metadata(
    *,
    tenant_id: Optional[str],
    domain: Optional[str],
    doc_type: Optional[str],
    language: Optional[str],
    country: Optional[str],
    source_system: Optional[str],
    effective_date: Optional[str],
    confidentiality: Optional[ConfidentialityType],
    version: Optional[str],
    tags: Optional[str],
    custom_metadata: Optional[str],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    if tenant_id and tenant_id.strip():
        metadata["tenant_id"] = tenant_id.strip()
    if domain and domain.strip():
        metadata["domain"] = domain.strip()
    if doc_type and doc_type.strip():
        metadata["doc_type"] = doc_type.strip()
    if language and language.strip():
        metadata["language"] = language.strip()
    if country and country.strip():
        metadata["country"] = country.strip()
    if source_system and source_system.strip():
        metadata["source_system"] = source_system.strip()
    if effective_date and effective_date.strip():
        try:
            metadata["effective_date"] = date.fromisoformat(effective_date.strip()).isoformat()
        except ValueError:
            raise ValueError("effective_date deve estar no formato YYYY-MM-DD")
    if confidentiality:
        metadata["confidentiality"] = confidentiality
    if version and version.strip():
        metadata["version"] = version.strip()

    parsed_tags = _parse_tags(tags)
    if parsed_tags is not None:
        metadata["tags"] = parsed_tags

    parsed_custom = _parse_json_metadata(custom_metadata)
    if parsed_custom is not None:
        metadata["custom_metadata"] = parsed_custom

    return metadata


def _build_documents_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    documents_map: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        chunk_id = str(chunk.get("chunk_id", ""))
        content = chunk.get("content", "")

        document_id = metadata.get("document_id")
        if not document_id and "_chunk_" in chunk_id:
            document_id = chunk_id.rsplit("_chunk_", 1)[0]
        if not document_id:
            document_id = chunk_id or "unknown_document"

        chunk_index = metadata.get("chunk_index", 0)
        try:
            chunk_index = int(chunk_index)
        except (ValueError, TypeError):
            chunk_index = 0

        if document_id not in documents_map:
            document_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {"chunk_index", "chunk_total"}
            }
            documents_map[document_id] = {
                "document_id": str(document_id),
                "source_file": metadata.get("source_file"),
                "chunk_total": int(metadata.get("chunk_total", 0) or 0),
                "metadata": document_metadata,
                "chunks": [],
            }

        documents_map[document_id]["chunks"].append(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": metadata,
            }
        )

    documents = list(documents_map.values())
    for document in documents:
        document["chunks"].sort(key=lambda item: item.get("chunk_index", 0))
        inferred_total = len(document["chunks"])
        if document["chunk_total"] <= 0:
            document["chunk_total"] = inferred_total

    documents.sort(key=lambda item: item["document_id"])
    return documents


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload de Documento PDF",
    description="Faz upload de um documento PDF, quebra em chunks e gera embeddings"
)
async def upload_document(
    file: UploadFile = File(..., description="Arquivo PDF para processar"),
    collection_name: str = Form(...,
                                description="Coleção para armazenar o documento"),
    metadata: Optional[str] = Form(
        None,
        description="JSON string legado para metadados livres (opcional)"
    ),
    tenant_id: Optional[str] = Form(None, description="ID do tenant/cliente (ex.: empresa_alpha)"),
    domain: Optional[str] = Form(
        None,
        description="Domínio do conteúdo (texto livre, ex.: rh, juridico, financeiro)",
    ),
    doc_type: Optional[str] = Form(
        None,
        description="Tipo do documento (texto livre, ex.: politica, manual, contrato)",
    ),
    language: Optional[str] = Form(None, description="Idioma (ex.: pt-BR, en-US)"),
    country: Optional[str] = Form(None, description="País (ex.: BR, US)"),
    source_system: Optional[str] = Form(None, description="Sistema de origem (ex.: sharepoint)"),
    effective_date: Optional[str] = Form(None, description="Data efetiva (YYYY-MM-DD)"),
    confidentiality: Optional[ConfidentialityType] = Form(None, description="Nível de confidencialidade"),
    version: Optional[str] = Form(None, description="Versão do documento (ex.: v1, 2026.04)"),
    tags: Optional[str] = Form(None, description="Lista de tags em JSON ou string separada por vírgulas"),
    custom_metadata: Optional[str] = Form(None, description="JSON com metadados customizados"),
):
    """
    Endpoint para upload de PDFs.

    - **file**: Arquivo PDF para processar
    - **collection_name**: Nome da coleção para armazenar o documento
    - **metadata**: Metadados adicionais em formato JSON (opcional)

    Retorna informações sobre o upload e os chunks criados.
    """
    try:
        legacy_metadata = _parse_json_metadata(metadata) or {}
        standard_metadata = _build_standard_metadata(
            tenant_id=tenant_id,
            domain=domain,
            doc_type=doc_type,
            language=language,
            country=country,
            source_system=source_system,
            effective_date=effective_date,
            confidentiality=confidentiality,
            version=version,
            tags=tags,
            custom_metadata=custom_metadata,
        )
        upload_metadata = {**legacy_metadata, **standard_metadata}
        if not upload_metadata:
            upload_metadata = None

        # Validar tipo de arquivo
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Apenas arquivos PDF são aceitos"
            )

        # Ler arquivo
        file_content = await file.read()

        # Validar tamanho
        max_size = settings.MAX_PDF_SIZE_MB * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
                detail=f"Arquivo muito grande. Máximo: {settings.MAX_PDF_SIZE_MB}MB"
            )

        # Gerar ID único para o documento
        document_id = str(uuid.uuid4())

        # Processar PDF
        documents = await pdf_processor.process_pdf(
            file_content=file_content,
            filename=file.filename,
            document_id=document_id,
            metadata=upload_metadata,
        )

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nenhum conteúdo extraível encontrado no PDF"
            )

        # Adicionar ao vector store
        chunk_ids = await vector_store.add_documents(documents, collection_name)

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            collection_name=collection_name,
            chunks_created=len(chunk_ids),
            chunks_ids=chunk_ids,
            status="success"
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar documento: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Buscar Documentos",
    description="Busca documentos similares usando embeddings"
)
async def search_documents(query_request: SearchQuery):
    """
    Endpoint para buscar documentos similares.

    - **query**: Texto para buscar
    - **collection_name**: Coleção onde buscar
    - **top_k**: Número máximo de resultados (padrão: 5)

    Retorna os documentos mais similares ordenados por relevância.
    """
    try:
        # Validar coleção
        if not query_request.collection_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nome da coleção é obrigatório"
            )

        # Buscar documentos
        results = await vector_store.search(
            query=query_request.query,
            collection_name=query_request.collection_name,
            top_k=query_request.top_k,
            metadata_filters=(
                _serialize_metadata_values(query_request.metadata_filters.model_dump(exclude_none=True))
                if query_request.metadata_filters else None
            ),
        )

        # Converter para response format
        search_results = [
            SearchResultItem(
                content=result.content,
                score=result.score,
                metadata=result.metadata
            )
            for result in results
        ]

        return SearchResponse(
            query=query_request.query,
            collection_name=query_request.collection_name,
            results=search_results,
            total_results=len(search_results)
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao buscar documentos: {str(e)}"
        )


@router.delete(
    "/collections/{collection_name}",
    summary="Deletar Coleção",
    description="Remove uma coleção inteira e todos seus documentos"
)
async def delete_collection(collection_name: str):
    """
    Endpoint para deletar uma coleção.

    - **collection_name**: Nome da coleção a deletar

    Retorna sucesso ou erro da operação.
    """
    try:
        success = await vector_store.delete_collection(collection_name)

        if success:
            return {
                "status": "success",
                "message": f"Coleção '{collection_name}' deletada com sucesso"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Coleção '{collection_name}' não encontrada"
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao deletar coleção: {str(e)}"
        )


@router.get(
    "/collections",
    response_model=CollectionListResponse,
    summary="Listar Coleções",
    description="Lista coleções disponíveis no backend vetorial"
)
async def list_collections():
    """Endpoint para listar coleções disponíveis."""
    try:
        try:
            collections = await vector_store.list_collections()
        except NotImplementedError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(e)
            )

        return CollectionListResponse(
            total_collections=len(collections),
            collections=collections,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar coleções: {str(e)}"
        )


@router.get(
    "/collections/{collection_name}/documents",
    response_model=CollectionDocumentsResponse,
    summary="Listar Documentos e Chunks",
    description="Lista todos os documentos de uma coleção com seus chunks e metadados"
)
async def list_collection_documents(
    collection_name: str,
    document_id: Optional[str] = Query(
        default=None,
        description="Filtra por document_id específico"
    ),
    page: int = Query(default=1, ge=1, description="Página da listagem"),
    page_size: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Quantidade de documentos por página"
    ),
):
    """
    Endpoint para listar documentos e chunks de uma coleção.

    - **collection_name**: Nome da coleção
    - **document_id**: Filtro opcional por document_id
    - **page**: Página da listagem (inicia em 1)
    - **page_size**: Quantidade de documentos por página
    """
    try:
        if not collection_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nome da coleção é obrigatório"
            )

        try:
            chunks = await vector_store.list_chunks(collection_name.strip())
        except NotImplementedError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(e)
            )

        documents = _build_documents_from_chunks(chunks)

        if document_id and document_id.strip():
            target_document_id = document_id.strip()
            documents = [
                item for item in documents
                if item["document_id"] == target_document_id
            ]

        total_documents = len(documents)
        total_chunks = sum(len(item.get("chunks", [])) for item in documents)
        total_pages = max(1, (total_documents + page_size - 1) // page_size)

        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_documents = documents[start_index:end_index]

        return CollectionDocumentsResponse(
            collection_name=collection_name.strip(),
            total_documents=total_documents,
            total_chunks=total_chunks,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            document_id_filter=document_id.strip() if document_id else None,
            documents=paginated_documents,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar documentos/chunks: {str(e)}"
        )
