from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core import settings
from app.api import router

# Criar aplicação FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, configurar domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Incluir rotas da API
app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check da aplicação.

    Retorna status da API, configurações e saúde do vector store.
    """
    from app.core.vector_store_factory import VectorStoreFactory

    try:
        # Verificar vector store
        vector_store = VectorStoreFactory.create_vector_store()
        vector_store_healthy = await vector_store.health_check()

        return {
            "status": "healthy" if vector_store_healthy else "degraded",
            "api": {
                "title": settings.API_TITLE,
                "version": settings.API_VERSION,
                "description": settings.API_DESCRIPTION,
            },
            "vector_store": {
                "type": settings.VECTOR_STORE_TYPE,
                "healthy": vector_store_healthy,
            },
            "configuration": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "pdf_chunk_size": settings.PDF_CHUNK_SIZE,
                "pdf_chunk_overlap": settings.PDF_CHUNK_OVERLAP,
                "max_pdf_size_mb": settings.MAX_PDF_SIZE_MB,
                "debug_mode": settings.DEBUG,
            }
        }
    except Exception as e:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check falhou: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
