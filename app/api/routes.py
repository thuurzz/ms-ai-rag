from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from typing import Any, Dict, List, Optional
import uuid
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

# Instâncias globais
pdf_processor = PDFProcessor(
    chunk_size=settings.PDF_CHUNK_SIZE,
    chunk_overlap=settings.PDF_CHUNK_OVERLAP,
)
vector_store = VectorStoreFactory.create_vector_store()


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
        None, description="JSON string com metadados adicionais"),
):
    """
    Endpoint para upload de PDFs.

    - **file**: Arquivo PDF para processar
    - **collection_name**: Nome da coleção para armazenar o documento
    - **metadata**: Metadados adicionais em formato JSON (opcional)

    Retorna informações sobre o upload e os chunks criados.
    """
    try:
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
            top_k=query_request.top_k
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
