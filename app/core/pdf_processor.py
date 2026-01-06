import io
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from app.core.vector_store import Document


class PDFProcessor:
    """Responsável por processar PDFs e quebrar em chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Inicializa o processador de PDF.

        Args:
            chunk_size: Tamanho de cada chunk em caracteres
            chunk_overlap: Sobreposição entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        document_id: str,
        metadata: dict = None,
    ) -> List[Document]:
        """
        Processa um arquivo PDF e retorna documentos em chunks.

        Args:
            file_content: Conteúdo binário do arquivo PDF
            filename: Nome do arquivo
            document_id: ID único do documento
            metadata: Metadados adicionais para os chunks

        Returns:
            Lista de documentos em chunks
        """
        try:
            # Extrair texto do PDF
            text = self._extract_text_from_pdf(file_content)

            if not text.strip():
                raise ValueError("PDF não contém texto extraível")

            # Dividir em chunks
            chunks = self.text_splitter.split_text(text)

            # Criar documentos com metadados
            documents = []
            default_metadata = {
                "source_file": filename,
                "document_id": document_id,
                **(metadata or {})
            }

            for idx, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    metadata={
                        **default_metadata,
                        "chunk_index": idx,
                        "chunk_total": len(chunks),
                    },
                    doc_id=f"{document_id}_chunk_{idx}"
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Erro ao processar PDF {filename}: {str(e)}")
            raise

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extrai texto do PDF usando pdfplumber.

        Args:
            file_content: Conteúdo binário do PDF

        Returns:
            Texto extraído do PDF
        """
        try:
            pdf_file = io.BytesIO(file_content)
            text = ""

            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Página {page_num + 1} ---\n"
                        text += page_text

            return text

        except Exception as e:
            print(f"Erro ao extrair texto do PDF: {str(e)}")
            raise

    def set_chunk_parameters(self, chunk_size: int, chunk_overlap: int):
        """
        Atualiza os parâmetros de chunking.

        Args:
            chunk_size: Novo tamanho de chunk
            chunk_overlap: Nova sobreposição
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
