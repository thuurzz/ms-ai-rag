# MS AI RAG

Microsserviço FastAPI para processamento de documentos PDF, geração de embeddings e consultas RAG (Retrieval-Augmented Generation) para agentes de IA.

## 🚀 Características

- ✅ **API FastAPI moderna** com documentação automática (Swagger/ReDoc)
- 📄 **Processamento de PDF** com extração de texto e chunking inteligente
- 🧠 **Geração de embeddings** usando Sentence Transformers
- 🔄 **Adaptadores modulares** para bancos vetoriais (ChromaDB, Pinecone, MongoDB)
- 🔍 **Busca semântica** com relevância por cosine similarity
- ⚙️ **Fácil configuração** via variáveis de ambiente
- 🏥 **Health check** para monitoramento

## 📋 Pré-requisitos

- Python 3.9+
- pip ou conda

## 🔧 Instalação

### 1. Clone o repositório

```bash
cd /home/steel-bk2/Development/ms-ai-rag
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o arquivo `.env`

Copie `.env.example` para `.env`:

```bash
cp .env.example .env
```

Edite `.env` com suas configurações:

```env
# Vector Store: chromadb, pinecone ou mongodb
VECTOR_STORE_TYPE=chromadb

# Para Pinecone
PINECONE_API_KEY=sua-chave-aqui
PINECONE_ENVIRONMENT=us-west1-gcp

# Para MongoDB
MONGODB_CONNECTION_STRING=mongodb://localhost:27017
```

## 🚀 Executar a Aplicação

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ou simplesmente:

```bash
python app/main.py
```

A API estará disponível em: **http://localhost:8000**

Acesse a documentação em:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 Endpoints da API

### 1. Health Check

```http
GET /health
```

Verifica a saúde da aplicação, configurações e status do vector store.

**Resposta:**

```json
{
  "status": "healthy",
  "api": {
    "title": "MS AI RAG",
    "version": "1.0.0",
    "description": "Microsserviço para processamento de PDFs e RAG com IA"
  },
  "vector_store": {
    "type": "chromadb",
    "healthy": true
  },
  "configuration": {
    "embedding_model": "all-MiniLM-L6-v2",
    "pdf_chunk_size": 500,
    "pdf_chunk_overlap": 50,
    "max_pdf_size_mb": 50,
    "debug_mode": false
  }
}
```

### 2. Upload de Documento

```http
POST /api/v1/upload
Content-Type: multipart/form-data
```

Faz upload de um PDF, processa e gera embeddings.

**Parâmetros:**

- `file` (obrigatório): Arquivo PDF
- `collection_name` (obrigatório): Nome da coleção
- `metadata` (opcional): JSON string com metadados adicionais

**Resposta:**

```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "documento.pdf",
  "collection_name": "meus_documentos",
  "chunks_created": 12,
  "chunks_ids": ["123e4567-e89b-12d3-a456-426614174000_chunk_0", ...],
  "status": "success"
}
```

### 3. Buscar Documentos

```http
POST /api/v1/search
Content-Type: application/json
```

Busca documentos similares usando embeddings.

**Corpo da requisição:**

```json
{
  "query": "Qual é o assunto principal do documento?",
  "collection_name": "meus_documentos",
  "top_k": 5
}
```

**Resposta:**

```json
{
  "query": "Qual é o assunto principal do documento?",
  "collection_name": "meus_documentos",
  "results": [
    {
      "content": "Este documento aborda os seguintes pontos...",
      "score": 0.92,
      "metadata": {
        "source_file": "documento.pdf",
        "chunk_index": 2,
        "document_id": "123e4567..."
      }
    }
  ],
  "total_results": 1
}
```

### 4. Deletar Coleção

```http
DELETE /api/v1/collections/{collection_name}
```

Remove uma coleção e todos seus documentos.

## 🏗️ Arquitetura

```
app/
├── main.py                 # Aplicação FastAPI principal
├── api/
│   └── routes.py          # Endpoints da API
├── core/
│   ├── config.py          # Configurações da aplicação
│   ├── vector_store.py    # Interface abstrata (VectorStoreAdapter)
│   ├── vector_store_factory.py  # Factory para criar adapters
│   └── pdf_processor.py   # Processador de PDFs
├── adapters/
│   ├── chromadb_adapter.py    # Adaptador ChromaDB
│   ├── pinecone_adapter.py    # Adaptador Pinecone
│   └── mongodb_adapter.py     # Adaptador MongoDB
└── schemas/
    └── __init__.py        # Schemas Pydantic para validação
```

## 🔄 Selecionando um Vector Store

A aplicação suporta múltiplos backends de armazenamento vetorial. Mude apenas a variável de ambiente `VECTOR_STORE_TYPE`:

### ChromaDB (Padrão - Em Memória)

```env
VECTOR_STORE_TYPE=chromadb
CHROMADB_PERSIST_DIRECTORY=./chroma_data
```

### Pinecone (Cloud)

```env
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=pk-xxx...
PINECONE_ENVIRONMENT=us-west1-gcp
```

### MongoDB (Com Vector Search)

```env
VECTOR_STORE_TYPE=mongodb
MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DATABASE_NAME=rag_system
```

## 📝 Exemplo de Uso

### cURL

```bash
# 1. Upload de um PDF
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@documento.pdf" \
  -F "collection_name=meus_docs" \
  -F 'metadata={"cliente": "empresa_xyz", "tipo": "documento"}'

# 2. Buscar documentos
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "qual é o assunto?",
    "collection_name": "meus_docs",
    "top_k": 5
  }'

# 3. Health check
curl "http://localhost:8000/health"

# 4. Deletar coleção
curl -X DELETE "http://localhost:8000/api/v1/collections/meus_docs"
```

### Python

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Upload
with open("documento.pdf", "rb") as f:
    files = {"file": f}
    data = {
        "collection_name": "meus_docs",
        "metadata": json.dumps({"type": "document"})
    }
    response = requests.post(f"{BASE_URL}/api/v1/upload", files=files, data=data)
    print(response.json())

# Search
search_query = {
    "query": "assunto principal",
    "collection_name": "meus_docs",
    "top_k": 5
}
response = requests.post(f"{BASE_URL}/api/v1/search", json=search_query)
print(response.json())
```

## ⚙️ Configuração Avançada

### Ajustar Tamanho de Chunks

```env
PDF_CHUNK_SIZE=1000        # Aumentar tamanho dos chunks
PDF_CHUNK_OVERLAP=100      # Aumentar sobreposição
```

### Usar Modelo de Embeddings Diferente

```env
EMBEDDING_MODEL=all-mpnet-base-v2  # Modelo mais potente
# Opções: all-MiniLM-L6-v2 (padrão), all-mpnet-base-v2, multilingual-e5-large
```

### Modo Debug

```env
DEBUG=True
```

## 🧪 Testes

```bash
# Executar com pytest (quando implementado)
pytest tests/
```

## 📦 Dependências Principais

- **FastAPI** - Framework web moderno
- **Uvicorn** - Servidor ASGI
- **Pydantic** - Validação de dados
- **LangChain** - Text splitting e utilities
- **Sentence Transformers** - Geração de embeddings
- **ChromaDB** - Banco vetorial em memória
- **Pinecone** - Banco vetorial cloud
- **MongoDB** - Banco de dados com vector search
- **PyPDF2/pdfplumber** - Processamento de PDFs

## 🐳 Docker (Opcional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build e execute:

```bash
docker build -t ms-ai-rag .
docker run -p 8000:8000 ms-ai-rag
```

## 🤝 Contribuindo

Sinta-se livre para abrir issues e pull requests!

## 📄 Licença

MIT License

## 📧 Suporte

Para dúvidas ou problemas, abra uma issue no repositório.
