from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional
import uuid
from app.core import PDFProcessor, VectorStoreFactory, settings
from app.schemas import (
    DocumentUploadResponse,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    ErrorResponse,
)

router = APIRouter(prefix="/api/v1", tags=["RAG Operations"])

# Instâncias globais
pdf_processor = PDFProcessor(
    chunk_size=settings.PDF_CHUNK_SIZE,
    chunk_overlap=settings.PDF_CHUNK_OVERLAP,
)
vector_store = VectorStoreFactory.create_vector_store()


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
