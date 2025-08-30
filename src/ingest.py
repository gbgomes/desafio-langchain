"""
Script para ingestão de documentos PDF em um banco de dados vetorial PGVector.

Este script realiza o seguinte processo de pipeline para RAG (Retrieval-Augmented Generation):
1. Carrega as variáveis de ambiente necessárias (chaves de API, URLs de banco de dados).
2. Localiza e carrega um arquivo PDF especificado na raiz do projeto.
3. Divide o documento em trechos (chunks) de texto menores.
4. Limpa e enriquece os metadados de cada trecho.
5. Gera embeddings para cada trecho usando a API da OpenAI.
6. Insere os trechos e seus embeddings correspondentes em uma coleção no PGVector,
   apagando a coleção anterior para garantir dados atualizados.

Pré-requisitos:
- Um arquivo .env na raiz do projeto com as variáveis de ambiente definidas.
- O arquivo PDF a ser processado deve estar na raiz do projeto.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
# Valida se as variáveis de ambiente essenciais estão configuradas
for k in ("OPENAI_API_KEY", "PGVECTOR_URL","PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

# Define o caminho para o arquivo PDF, assumindo que ele está na raiz do projeto
PDF_DIR = Path(__file__).parent.parent
PDF_PATH = PDF_DIR / "LAMINA_28428129000114.pdf"

def ingest_pdf():
    """
    Executa o processo de ingestão do PDF.

    Carrega o documento, o divide em trechos, gera embeddings e os armazena
    no banco de dados PGVector.
    """
    print("Realizando a leitura e split do arquivo.")
    docs = PyPDFLoader(str(PDF_PATH)).load()

    # Divide os documentos em trechos menores para facilitar a busca de similaridade
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, add_start_index=False).split_documents(docs)
    if not splits:
        print("Nenhum documento para processar. Encerrando.")
        raise SystemExit(0)

    # Enriquece os documentos, removendo metadados vazios ou nulos que podem
    # interferir no processo de embedding ou armazenamento.
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    # Gera IDs únicos para cada documento para garantir a idempotência na inserção
    ids = [f"doc-{i}" for i in range(len(enriched))]

    print("Inserindo os dados no banco de dados.")
    # Inicializa o modelo de embeddings da OpenAI
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small"))

    # Conecta ao PGVector e insere os documentos e seus embeddings
    PGVector.from_documents(
        documents=enriched,
        embedding=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        pre_delete_collection=True, # Garante que a coleção seja limpa antes da nova inserção
        ids=ids,
    )
    print("Dados inseridos no banco de dados.")

# Ponto de entrada do script: executa a função de ingestão quando o arquivo é chamado diretamente
if __name__ == "__main__":
    ingest_pdf()