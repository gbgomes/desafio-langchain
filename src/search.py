"""
Módulo de busca e geração de respostas para o sistema RAG.

Este módulo é o coração do chatbot, responsável por orquestrar o processo de
Retrieval-Augmented Generation (RAG). Ele inicializa e configura todos os
componentes necessários da biblioteca LangChain, como o modelo de embeddings,
a conexão com o banco de dados vetorial (PGVector) e a pipeline de LLM.

A função principal, `search_prompt`, recebe uma pergunta do usuário, busca
documentos relevantes no banco de dados e gera uma resposta contextualizada.
"""
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

# Template do prompt, projetado para ser robusto e garantir que o LLM
# responda estritamente com base no contexto fornecido. Inclui regras claras
# e exemplos para guiar o modelo, minimizando alucinações (respostas inventadas).
PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

# Carrega as variáveis de ambiente do arquivo .env.
load_dotenv()
# Valida se todas as variáveis de ambiente essenciais estão configuradas.
# Isso garante que o script falhe rapidamente se a configuração estiver incompleta.
for k in ("OPENAI_API_KEY", "PGVECTOR_URL","PGVECTOR_COLLECTION", "OPENAI_MODEL", "EMBEDDING_MODEL"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

# --- Inicialização dos Componentes LangChain ---
# Estes objetos são criados uma única vez quando o módulo é importado,
# o que otimiza o desempenho, evitando recriações a cada chamada de função.

# Inicializa o modelo de embeddings da OpenAI, que será usado para vetorizar
# as perguntas do usuário para a busca de similaridade.
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small"))

# Configura a conexão com o banco de dados vetorial PGVector.
# Este objeto `store` será usado para realizar as buscas de similaridade.
store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

# Cria um objeto PromptTemplate a partir da string de template definida anteriormente.
template = PromptTemplate.from_template(PROMPT_TEMPLATE)

# Inicializa o modelo de linguagem (LLM) da OpenAI com temperatura 0 para
# obter respostas mais determinísticas e factuais.
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)

# Monta a pipeline de execução usando LangChain Expression Language (LCEL).
# A pipeline define o fluxo: (template -> llm -> parser).
# 1. O `template` recebe o contexto e a pergunta.
# 2. O resultado é enviado para o `llm`.
# 3. A resposta do `llm` é processada pelo `StrOutputParser` para retornar uma string.
pipeline = template | llm | StrOutputParser()


def search_prompt(question=None) :
    """
    Executa a busca por similaridade e a geração da resposta via LLM.

    Esta função recebe uma pergunta, busca os 10 trechos de texto mais
    relevantes no banco de dados vetorial e os utiliza como contexto para
    gerar uma resposta com o modelo de linguagem.

    Args:
        question (str, optional): A pergunta feita pelo usuário. Defaults to None.

    Returns:
        str: A resposta gerada pelo modelo de linguagem, baseada no contexto
             encontrado.
    """
    # Realiza a busca de similaridade no PGVector para encontrar os 10
    # documentos mais relevantes para a pergunta do usuário.
    results = store.similarity_search_with_score(question, k=10)

    # Concatena o conteúdo dos documentos encontrados para formar um único
    # bloco de texto de contexto, separado por quebras de linha duplas.
    context = "\n\n".join([doc.page_content for doc, score in results])

    # Invoca a pipeline de RAG (template | llm | parser) com o contexto
    # e a pergunta para obter a resposta final.
    result = pipeline.invoke({"context": context, "question": question})
    return result