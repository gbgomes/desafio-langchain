# Chatbot com RAG sobre Documento PDF

Este projeto implementa um chatbot que utiliza a técnica de Retrieval-Augmented Generation (RAG) para responder perguntas com base no conteúdo de um documento PDF.

O sistema utiliza a biblioteca LangChain para orquestrar o fluxo, OpenAI para a geração de embeddings e respostas, e um banco de dados PostgreSQL com a extensão PGVector para armazenar e consultar os vetores de texto de forma eficiente.

## Arquitetura

1.  **Ingestão de Dados**: Um script Python (`src/ingest.py`) lê o documento PDF, divide-o em trechos (chunks), gera embeddings para cada trecho usando a API da OpenAI e os armazena no banco de dados PGVector.
2.  **Interface de Chat**: Um segundo script (`src/chat.py`) fornece uma interface de linha de comando onde o usuário pode fazer perguntas.
3.  **Lógica de RAG**: Para cada pergunta do usuário:
   *   O sistema gera um embedding da pergunta.
   *   Realiza uma busca de similaridade no PGVector para encontrar os trechos mais relevantes do PDF.
   *   Monta um prompt contendo o contexto (trechos relevantes) e a pergunta original.
   *   Envia o prompt para um modelo de linguagem da OpenAI (LLM) para gerar uma resposta fundamentada no contexto.

## Pré-requisitos

- Python 3.9+
- Docker e Docker Compose
- Uma chave de API da OpenAI

## Configuração do Ambiente

Siga os passos abaixo para configurar e executar o projeto.

### 1. Clone o Repositório

```bash
git clone <url-do-seu-repositorio>
cd desafio-langchain
```

### 2. Configure as Variáveis de Ambiente

Copie o arquivo de exemplo `.env.example` para um novo arquivo chamado `.env`:

```bash
cp .env.example .env
```

Agora, edite o arquivo `.env` e adicione sua chave da API da OpenAI:

```ini
# .env
OPENAI_API_KEY="sk-sua-chave-aqui"
PGVECTOR_URL="postgresql+psycopg2://user:password@localhost:5432/vectordb"
PGVECTOR_COLLECTION="desafio_fiap"
```

### 3. Inicie o Banco de Dados com Docker

Com o Docker em execução, inicie o container do PostgreSQL com a extensão PGVector usando o Docker Compose:

```bash
docker-compose up -d
```

Isso irá criar e iniciar um banco de dados na porta `5432`, pronto para ser usado pela aplicação.

### 4. Crie e Ative o Ambiente Virtual

É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

```bash
# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente (Linux/macOS)
source venv/bin/activate

# Ative o ambiente (Windows)
.\venv\Scripts\activate
```

### 5. Instale as Dependências

Com o ambiente virtual ativado, instale as bibliotecas Python necessárias:

```bash
pip install -r requirements.txt
```

## Execução

### Passo 1: Ingestão do PDF

Primeiro, execute o script de ingestão para processar o arquivo `document.pdf` (que deve estar na raiz do projeto) e popular o banco de dados.

```bash
python src/ingest.py
```

Você verá uma mensagem de confirmação quando o processo for concluído.

### Passo 2: Inicie o Chat

Agora você pode iniciar o chatbot interativo.

```bash
python src/chat.py
```

O terminal ficará aguardando sua pergunta.

#### Exemplo de Uso

```
Faça sua pergunta (ou 'sair' para encerrar): Qual o faturamento da Empresa SuperTechIABrazil?

O faturamento da SuperTechIABrazil é de R$ 10.000.000,00.

Faça sua pergunta (ou 'sair' para encerrar): sair
```

## Reinicializando o Banco de Dados
Caso você precise limpar completamente o banco de dados e começar do zero (por exemplo, para re-ingerir um novo PDF), você pode remover o volume de dados persistentes do Docker.

Atenção: Este comando apagará todos os dados armazenados no banco de dados.

Pare e remova os containers e o volume de dados:

```bash
docker-compose down -v
Inicie o banco de dados novamente:
```

Reinicie o banco de dados:

```bash
docker-compose up -d
```

Após esses passos, o banco de dados estará vazio. Você precisará executar o script de ingestão novamente para populá-lo (python src/ingest.py).

