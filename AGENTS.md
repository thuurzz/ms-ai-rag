# AGENTS

## Fast Start (verified)
- Use `python3` in this shell (`python` is not available unless a venv provides it).
- Install deps with `pip install -r requirements.txt`.
- Run API with `python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` (or `python3 app/main.py`).
- Docker path is `docker compose up --build`; it starts only `rag-service` by default.
- For optional DB backends in compose: `docker compose --profile with-mongodb up --build` or `docker compose --profile with-postgres up --build`.

## Verification Reality
- No CI workflows, lint config, or typecheck config are present; rely on focused runtime checks.
- `tests/` exists but is empty.
- `pytest` is not in `requirements.txt`; `python3 -m pytest tests -q` fails unless you install pytest manually.
- Practical smoke check is API-level: call `GET /health`, then exercise `/api/v1/upload` and `/api/v1/search`.

## Real Entry Points
- `app/main.py`: FastAPI app setup, CORS, `/health`, and router registration.
- `app/api/routes.py`: primary API behavior (`/api/v1/upload`, `/api/v1/search`, `/api/v1/collections/{collection_name}`).
- `app/core/config.py`: env loading from `.env` with `case_sensitive=True`.
- `app/core/vector_store_factory.py`: backend selection (`chromadb|pinecone|mongodb|postgres`).

## Gotchas That Affect Changes
- `app/api/routes.py` initializes `pdf_processor` and `vector_store` as module globals at import; changing env/backend requires process restart.
- `/health` instantiates a fresh adapter via `VectorStoreFactory.create_vector_store()` each call, so health status can diverge from the global adapter used by route handlers.
- `ChromaDBAdapter` uses `chromadb.Client()` in-memory and ignores `CHROMADB_PERSIST_DIRECTORY`; data is non-persistent across restarts despite compose volume/env wiring.
- Upload `metadata` is accepted as form text but never parsed or passed into `PDFProcessor.process_pdf`; it is currently ignored.
- Pinecone index names are validated in `PineconeAdapter._validate_index_name`: lowercase letters/numbers/hyphens only (no spaces/underscores/special chars).
- Postgres adapter stores vectors with dimension 384 (`all-MiniLM-L6-v2`); changing embedding model requires matching vector dimension.
