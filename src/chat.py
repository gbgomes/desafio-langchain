"""
Ponto de entrada para a interface de chat interativo com o usuário.

Este script inicia um loop infinito que:
1. Solicita uma pergunta ao usuário via linha de comando.
2. Verifica se o usuário deseja sair da aplicação.
3. Chama a função `search_prompt` do módulo `search` para obter uma resposta.
4. Imprime a resposta gerada no console.

Para executar, use: `python src/chat.py`
"""
from search import search_prompt

# Inicia o loop principal da aplicação de chat.
while True:
    # Solicita a entrada do usuário e armazena na variável 'query'.
    query = input("\nFaça sua pergunta (ou 'sair' para encerrar): ")

    # Verifica se a entrada do usuário é um comando para sair do loop.
    if query.lower() in ("sair", "exit", "quit"):
        print("Encerrando o chat. Até logo!")
        break

    # Chama a função de busca e geração de resposta com a pergunta do usuário.
    result = search_prompt(query)

    # Imprime o resultado retornado pela função.
    print(result)