import json
from typing import Any, Dict, Optional

import requests
import streamlit as st


DEFAULT_API_URL = "http://localhost:8000"


def api_request(
    method: str,
    base_url: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    url = f"{base_url.rstrip('/')}{path}"
    return requests.request(
        method=method,
        url=url,
        params=params,
        data=data,
        files=files,
        json=json_payload,
        timeout=60,
    )


def ensure_collections_state() -> None:
    if "available_collections" not in st.session_state:
        st.session_state["available_collections"] = []
    if "collections_loaded_for_url" not in st.session_state:
        st.session_state["collections_loaded_for_url"] = None


def refresh_collections(base_url: str) -> None:
    response = api_request("GET", base_url, "/api/v1/collections")
    response.raise_for_status()
    data = response.json()
    st.session_state["available_collections"] = data.get("collections", [])
    st.session_state["collections_loaded_for_url"] = base_url.rstrip("/")


def render_collection_selector(base_url: str, key_prefix: str) -> str:
    ensure_collections_state()
    normalized_url = base_url.rstrip("/")

    should_auto_refresh = (
        st.session_state.get("collections_loaded_for_url") != normalized_url
        or not st.session_state.get("available_collections")
    )

    if should_auto_refresh:
        try:
            refresh_collections(base_url)
        except requests.RequestException as exc:
            st.warning(f"Não foi possível carregar coleções automaticamente: {exc}")

    refresh_click = st.button(
        "Atualizar coleções",
        use_container_width=True,
        key=f"{key_prefix}_refresh_collections",
    )

    if refresh_click:
        try:
            refresh_collections(base_url)
            st.success(
                f"{len(st.session_state.get('available_collections', []))} coleção(ões) carregada(s)."
            )
        except requests.RequestException as exc:
            st.error(f"Erro de conexão ao listar coleções: {exc}")

    available_collections = st.session_state.get("available_collections", [])
    selected_collection = ""
    if available_collections:
        selected_collection = st.selectbox(
            "Coleções disponíveis",
            options=available_collections,
            key=f"{key_prefix}_collection_select",
        )
    else:
        st.info("Nenhuma coleção carregada ainda. Clique em 'Atualizar coleções'.")

    return selected_collection


def render_health(base_url: str) -> None:
    if st.button("Health check", use_container_width=True):
        with st.spinner("Consultando serviço..."):
            try:
                response = api_request("GET", base_url, "/health")
                response.raise_for_status()
                st.success("Serviço respondeu com sucesso.")
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"Erro ao chamar /health: {exc}")


def parse_metadata(metadata_text: str) -> Optional[Dict[str, Any]]:
    cleaned = metadata_text.strip()
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        st.error(f"Metadados inválidos (JSON): {exc}")
        return None

    if not isinstance(parsed, dict):
        st.error("Metadados devem ser um objeto JSON.")
        return None

    return parsed


def render_upload(base_url: str) -> None:
    st.subheader("Upload de PDF")
    collection_name = st.text_input("Collection name", key="upload_collection")
    uploaded_pdf = st.file_uploader(
        "Arquivo PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="upload_pdf",
    )
    metadata_text = st.text_area(
        "Metadata (JSON opcional)",
        placeholder='{"cliente": "empresa_xyz", "tipo": "documento"}',
        key="upload_metadata",
    )

    if st.button("Enviar PDF", type="primary", use_container_width=True):
        if not collection_name.strip():
            st.warning("Informe o nome da coleção.")
            return

        if uploaded_pdf is None:
            st.warning("Selecione um arquivo PDF.")
            return

        metadata = parse_metadata(metadata_text)
        if metadata_text.strip() and metadata is None:
            return

        payload = {"collection_name": collection_name.strip()}
        if metadata is not None:
            payload["metadata"] = json.dumps(metadata)

        files = {
            "file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf"),
        }

        with st.spinner("Processando upload..."):
            try:
                response = api_request(
                    "POST",
                    base_url,
                    "/api/v1/upload",
                    data=payload,
                    files=files,
                )
                response.raise_for_status()
                st.success("Upload concluído.")
                st.json(response.json())
            except requests.HTTPError:
                detail = response.text if response is not None else ""
                st.error(f"Falha no upload: {detail}")
            except requests.RequestException as exc:
                st.error(f"Erro de conexão: {exc}")


def render_search(base_url: str) -> None:
    st.subheader("Busca semântica")
    query = st.text_area("Pergunta / Query", key="search_query")
    collection_name = render_collection_selector(base_url, key_prefix="search")
    top_k = st.slider("Top K", min_value=1, max_value=100, value=5)

    if st.button("Buscar", use_container_width=True):
        if not query.strip():
            st.warning("Informe uma query para busca.")
            return
        if not collection_name.strip():
            st.warning("Informe o nome da coleção.")
            return

        payload = {
            "query": query.strip(),
            "collection_name": collection_name.strip(),
            "top_k": top_k,
        }

        with st.spinner("Buscando documentos..."):
            try:
                response = api_request(
                    "POST",
                    base_url,
                    "/api/v1/search",
                    json_payload=payload,
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                st.success(f"Busca concluída. {len(results)} resultado(s).")
                st.json(data)

                for idx, item in enumerate(results, start=1):
                    with st.expander(f"Resultado {idx} | score={item.get('score', 0):.4f}"):
                        st.write(item.get("content", ""))
                        st.caption("Metadata")
                        st.json(item.get("metadata", {}))
            except requests.HTTPError:
                detail = response.text if response is not None else ""
                st.error(f"Falha na busca: {detail}")
            except requests.RequestException as exc:
                st.error(f"Erro de conexão: {exc}")


def render_delete_collection(base_url: str) -> None:
    st.subheader("Deletar coleção")
    collection_name = render_collection_selector(base_url, key_prefix="delete")
    confirm = st.checkbox("Confirmo que desejo apagar a coleção")

    if st.button("Deletar coleção", use_container_width=True):
        if not collection_name.strip():
            st.warning("Informe o nome da coleção.")
            return
        if not confirm:
            st.warning("Marque a confirmação para continuar.")
            return

        path = f"/api/v1/collections/{collection_name.strip()}"
        with st.spinner("Deletando coleção..."):
            try:
                response = api_request("DELETE", base_url, path)
                response.raise_for_status()
                st.success("Coleção removida.")
                st.json(response.json())
            except requests.HTTPError:
                detail = response.text if response is not None else ""
                st.error(f"Falha ao deletar coleção: {detail}")
            except requests.RequestException as exc:
                st.error(f"Erro de conexão: {exc}")


def render_documents(base_url: str) -> None:
    st.subheader("Documentos e chunks")
    collection_name = render_collection_selector(base_url, key_prefix="documents")

    document_id_filter = st.text_input(
        "Filtrar por document_id (opcional)",
        key="documents_document_id",
    )
    col_page, col_page_size = st.columns(2)
    with col_page:
        page = st.number_input("Página", min_value=1, value=1, step=1)
    with col_page_size:
        page_size = st.selectbox("Itens por página", options=[10, 20, 50, 100, 200], index=1)

    if st.button("Listar documentos", use_container_width=True):
        if not collection_name.strip():
            st.warning("Informe o nome da coleção.")
            return

        path = f"/api/v1/collections/{collection_name.strip()}/documents"
        params = {
            "page": int(page),
            "page_size": int(page_size),
        }
        if document_id_filter.strip():
            params["document_id"] = document_id_filter.strip()

        with st.spinner("Buscando documentos e chunks..."):
            try:
                response = api_request("GET", base_url, path, params=params)
                response.raise_for_status()
                data = response.json()

                st.success(
                    f"{data.get('total_documents', 0)} documento(s) e {data.get('total_chunks', 0)} chunk(s)."
                )
                st.caption(
                    f"Página {data.get('page', 1)} de {data.get('total_pages', 1)} | page_size={data.get('page_size', page_size)}"
                )
                st.json(data)

                for document in data.get("documents", []):
                    title = document.get("document_id", "documento")
                    chunk_total = document.get("chunk_total", 0)
                    source_file = document.get("source_file", "-")
                    with st.expander(f"{title} | chunks={chunk_total} | arquivo={source_file}"):
                        st.caption("Metadados do documento")
                        st.json(document.get("metadata", {}))

                        for chunk in document.get("chunks", []):
                            chunk_id = chunk.get("chunk_id", "")
                            chunk_index = chunk.get("chunk_index", 0)
                            st.markdown(f"**Chunk {chunk_index}** · `{chunk_id}`")
                            st.write(chunk.get("content", ""))
                            st.caption("Metadados do chunk")
                            st.json(chunk.get("metadata", {}))
                            st.divider()
            except requests.HTTPError:
                detail = response.text if response is not None else ""
                st.error(f"Falha ao listar documentos: {detail}")
            except requests.RequestException as exc:
                st.error(f"Erro de conexão: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="MS AI RAG UI",
        page_icon="📄",
        layout="wide",
    )

    st.title("MS AI RAG - Streamlit UI")
    st.caption("Interface desacoplada do microsserviço (consumo via HTTP API).")

    with st.sidebar:
        st.header("Configuração")
        base_url = st.text_input("URL da API", value=DEFAULT_API_URL)
        st.markdown("A UI não importa código do backend; apenas chama os endpoints HTTP.")
        st.divider()
        render_health(base_url)

    tab_upload, tab_search, tab_documents, tab_delete = st.tabs(
        ["Upload", "Search", "Documents", "Delete Collection"]
    )

    with tab_upload:
        render_upload(base_url)
    with tab_search:
        render_search(base_url)
    with tab_documents:
        render_documents(base_url)
    with tab_delete:
        render_delete_collection(base_url)


if __name__ == "__main__":
    main()
