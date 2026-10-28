"""Juiz da avaliação: um LLM compara a resposta de cada configuração com a resposta esperada.

Comparar texto livre com `==` não funciona ("Seu cachorro é o Paçoca" e "Paçoca" dizem a mesma
coisa), então quem decide se a resposta está certa é o próprio modelo, como no LongMemEval. O
juiz devolve duas coisas: `correto` (sim ou não) e uma `justificativa` de uma linha, para o
professor poder conferir o veredito. A saída segue um esquema JSON (saída estruturada), então
o código lê o veredito com `json.loads` em vez de procurar a palavra "sim" no texto.

O prompt é o mesmo para as quatro configurações e para todas as categorias de pergunta. A
categoria vai na mensagem de cada julgamento, não no prompt, para que nenhuma configuração seja
julgada por uma régua diferente das outras.
"""

import json

from memoria.llm import LLM, texto

# Esquema do veredito. `required` e `additionalProperties: False` fazem a API devolver sempre
# um objeto com exatamente estas duas chaves.
FORMATO = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "correto": {"type": "boolean"},
            "justificativa": {"type": "string"},
        },
        "required": ["correto", "justificativa"],
        "additionalProperties": False,
    },
}

# Instruções do juiz. Quase todas as regras existem por um caso real do dataset: a resposta
# esperada traz contexto entre parênteses ("antes: ..."), as perguntas de tempo envolvem contas
# de dias, as de abstenção têm como resposta certa "não sei" e as de várias sessões pedem listas.
# A regra do marcador `[erro: ...]` vem do runner, que passa ao juiz a resposta de um turno que
# falhou.
PROMPT_JUIZ = """\
Você é o juiz de uma avaliação de memória de um assistente. Você recebe a categoria da pergunta, a pergunta, a resposta esperada e a resposta que o assistente deu. Decida se a resposta do assistente está correta.

Regras:
- `correto` é verdadeiro se a resposta do assistente contém a informação esperada, mesmo com palavras diferentes ou com detalhes a mais. É falso se ela contradiz a resposta esperada ou se o assistente disse que não sabe quando havia a informação.
- Na resposta esperada, o texto entre parênteses ou depois de dois-pontos é contexto para você, não conteúdo que o assistente precise repetir. Por exemplo, para "Agrotech Planalto (antes: Cerrado Logística)", uma resposta que cita só Agrotech Planalto está correta. Em perguntas sobre datas e prazos, o que conta é o número; a conta que o assistente mostra não importa.
- Em contagens de dias, aceite diferença de 1 dia (contar ou não o dia inicial).
- Quando a resposta esperada lista vários itens, todos precisam aparecer na resposta do assistente, e itens inventados a mais tornam a resposta incorreta.
- Categoria `abstencao`: a informação nunca foi dita ao assistente. A resposta é correta se ele disse que não sabe ou que isso não foi mencionado, e incorreta se inventou uma resposta.
- Uma resposta que é um marcador de erro, como `[erro: api]`, é incorreta.

Escreva a `justificativa` em uma única linha."""


def julgar(llm: LLM, pergunta: str, esperado: str, resposta: str, categoria: str) -> dict:
    """Julga uma resposta e devolve `{"correto": bool, "justificativa": str}`.

    Se a chamada ao juiz falhar (erro de API, recusa) ou o JSON vier inválido, o resultado é
    `correto=False` e a justificativa é o motivo entre colchetes (`[erro: api]`, `[erro: json
    inválido]`...). Assim uma falha do juiz vira uma linha visível na tabela, não uma exceção
    que derruba a avaliação inteira, e nunca conta ponto a favor de uma configuração.
    """
    pedido = (
        f"Categoria: {categoria}\n"
        f"Pergunta: {pergunta}\n"
        f"Resposta esperada: {esperado}\n"
        f"Resposta do assistente: {resposta}"
    )
    saida = texto(llm.responder(PROMPT_JUIZ, [{"role": "user", "content": pedido}], formato=FORMATO))
    if saida.startswith("[erro:"):
        return {"correto": False, "justificativa": saida}
    try:
        veredito = json.loads(saida)
        return {"correto": veredito["correto"], "justificativa": veredito["justificativa"]}
    except (json.JSONDecodeError, TypeError, KeyError):
        return {"correto": False, "justificativa": "[erro: json inválido]"}
