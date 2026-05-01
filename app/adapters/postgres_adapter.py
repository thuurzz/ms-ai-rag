import json
import re
from typing import Any, Dict, List

import psycopg
from psycopg import sql
from sentence_transformers import SentenceTransformer

from app.core.vector_store import Document, SearchResult, VectorStoreAdapter


class PostgresAdapter(VectorStoreAdapter):
    """Adaptador para PostgreSQL com extensão pgvector."""

    def __init__(self, connection_string: str, table_prefix: str = "rag_", model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.connection = psycopg.connect(connection_string)
        self.table_prefix = self._normalize_prefix(table_prefix)
        self.vector_dimension = 384
        self._ensure_vector_extension()

    def _normalize_prefix(self, prefix: str) -> str:
        if not prefix:
            return ""
        normalized = re.sub(r"[^a-zA-Z0-9_]", "_", prefix).lower()
        return normalized

    def _normalize_table_name(self, collection_name: str) -> str:
        if not collection_name or not collection_name.strip():
            raise ValueError("Nome da coleção é obrigatório")

        normalized = re.sub(r"[^a-zA-Z0-9_]", "_", collection_name.strip()).lower()
        normalized = normalized.strip("_")

        if not normalized:
            raise ValueError("Nome da coleção inválido para PostgreSQL")

        table_name = f"{self.table_prefix}{normalized}"

        if len(table_name) > 63:
            table_name = table_name[:63]

        if table_name[0].isdigit():
            table_name = f"c_{table_name}"

        return table_name

    def _vector_literal(self, embedding: List[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in embedding) + "]"

    def _table_name_to_collection_name(self, table_name: str) -> str:
        if self.table_prefix and table_name.startswith(self.table_prefix):
            collection_name = table_name[len(self.table_prefix):]
            return collection_name or table_name
        return table_name

    def _ensure_vector_extension(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Erro ao habilitar extensão vector no PostgreSQL: {str(e)}")
            raise

    def _ensure_collection_table(self, table_name: str):
        create_table_query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector(384) NOT NULL
            )
            """
        ).format(
            table_name=sql.Identifier(table_name),
        )

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_query)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Erro ao criar tabela no PostgreSQL: {str(e)}")
            raise

    def _table_exists(self, table_name: str) -> bool:
        exists_query = "SELECT to_regclass(%s)"
        with self.connection.cursor() as cursor:
            cursor.execute(exists_query, (table_name,))
            row = cursor.fetchone()
        return bool(row and row[0])

    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """Adiciona documentos ao PostgreSQL/pgvector."""
        try:
            table_name = self._normalize_table_name(collection_name)
            self._ensure_collection_table(table_name)

            texts = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)

            ids = []
            with self.connection.cursor() as cursor:
                insert_query = sql.SQL(
                    """
                    INSERT INTO {table_name} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """
                ).format(table_name=sql.Identifier(table_name))

                for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    doc_id = doc.doc_id or f"doc_{i}"
                    ids.append(doc_id)
                    metadata_json = json.dumps(doc.metadata or {})
                    embedding_values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                    cursor.execute(
                        insert_query,
                        (doc_id, doc.content, metadata_json, self._vector_literal(embedding_values)),
                    )

            self.connection.commit()
            return ids

        except Exception as e:
            self.connection.rollback()
            print(f"Erro ao adicionar documentos no PostgreSQL: {str(e)}")
            raise

    async def search(self, query: str, collection_name: str, top_k: int = 5) -> List[SearchResult]:
        """Busca documentos similares no PostgreSQL/pgvector."""
        try:
            table_name = self._normalize_table_name(collection_name)
            if not self._table_exists(table_name):
                return []

            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
            query_embedding_values = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

            search_query = sql.SQL(
                """
                SELECT
                    content,
                    metadata,
                    (embedding <=> %s::vector) AS distance
                FROM {table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
            ).format(table_name=sql.Identifier(table_name))

            with self.connection.cursor() as cursor:
                vector_literal = self._vector_literal(query_embedding_values)
                cursor.execute(search_query, (vector_literal, vector_literal, top_k))
                rows = cursor.fetchall()

            results = []
            for content, metadata, distance in rows:
                score = max(0.0, 1.0 - float(distance))
                results.append(
                    SearchResult(
                        content=content,
                        score=score,
                        metadata=metadata or {},
                    )
                )

            return results

        except Exception as e:
            print(f"Erro ao buscar no PostgreSQL: {str(e)}")
            raise

    async def delete_documents(self, doc_ids: List[str], collection_name: str) -> bool:
        """Remove documentos do PostgreSQL."""
        try:
            table_name = self._normalize_table_name(collection_name)
            if not self._table_exists(table_name):
                return False

            delete_query = sql.SQL(
                "DELETE FROM {table_name} WHERE id = ANY(%s)"
            ).format(table_name=sql.Identifier(table_name))

            with self.connection.cursor() as cursor:
                cursor.execute(delete_query, (doc_ids,))
                deleted_count = cursor.rowcount

            self.connection.commit()
            return deleted_count > 0

        except Exception as e:
            self.connection.rollback()
            print(f"Erro ao deletar documentos no PostgreSQL: {str(e)}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Remove uma coleção (tabela) do PostgreSQL."""
        try:
            table_name = self._normalize_table_name(collection_name)
            drop_query = sql.SQL("DROP TABLE IF EXISTS {table_name}").format(
                table_name=sql.Identifier(table_name)
            )

            with self.connection.cursor() as cursor:
                cursor.execute(drop_query)

            self.connection.commit()
            return True

        except Exception as e:
            self.connection.rollback()
            print(f"Erro ao deletar coleção no PostgreSQL: {str(e)}")
            return False

    async def health_check(self) -> bool:
        """Verifica a saúde da conexão com PostgreSQL e extensão vector."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                row = cursor.fetchone()

            return bool(row and row[0])
        except Exception:
            return False

    async def list_chunks(self, collection_name: str) -> List[Dict[str, Any]]:
        """Lista todos os chunks de uma coleção no PostgreSQL."""
        try:
            table_name = self._normalize_table_name(collection_name)
            if not self._table_exists(table_name):
                return []

            query = sql.SQL(
                """
                SELECT id, content, metadata
                FROM {table_name}
                ORDER BY
                    COALESCE(metadata->>'document_id', ''),
                    COALESCE((metadata->>'chunk_index')::int, 0)
                """
            ).format(table_name=sql.Identifier(table_name))

            with self.connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

            chunks = []
            for chunk_id, content, metadata in rows:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": content,
                        "metadata": metadata or {},
                    }
                )

            return chunks
        except Exception as e:
            print(f"Erro ao listar chunks no PostgreSQL: {str(e)}")
            raise

    async def list_collections(self) -> List[str]:
        """Lista coleções disponíveis no PostgreSQL."""
        try:
            if self.table_prefix:
                query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name LIKE %s
                ORDER BY table_name
                """
                params = (f"{self.table_prefix}%",)
            else:
                query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
                params = ()

            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

            return [self._table_name_to_collection_name(row[0]) for row in rows]
        except Exception as e:
            print(f"Erro ao listar coleções no PostgreSQL: {str(e)}")
            raise
