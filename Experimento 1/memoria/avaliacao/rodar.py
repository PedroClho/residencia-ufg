"""Runner da verificação: roda as quatro configurações no mesmo dataset e monta a tabela de resultados.

Para cada configuração, o runner repete a mesma história:

    dataset.json
       |
       +--> 1. para cada sessão: nova_sessao(data) e uma chamada a responder() por fala do usuário
       |       (as respostas do assistente são geradas ao vivo, para a memória ser alimentada
       |       pelo pipeline de verdade)
       |
       +--> 2. sessão final: nova_sessao(data das perguntas) e responder() para cada pergunta, em
       |       sequência, como turnos normais
       |
       +--> 3. o juiz (juiz.py) compara cada resposta com a esperada: correto ou incorreto
       |
       +--> 4. soma acertos por categoria, erros, chamadas, tokens e custo

No fim grava `resultados/<AAAAMMDD-HHMMSS>/`: um JSON por configuração (todos os turnos, respostas
e vereditos) e a `tabela.md` com a comparação.

Para rodar (usa a API de verdade e custa dinheiro):

    uv run python -m memoria.avaliacao.rodar               # as quatro configurações
    uv run python -m memoria.avaliacao.rodar passiva       # só uma (sem_memoria, contexto_completo,
                                                           #          passiva ou ativa)

Decisões de desenho:

- Cada configuração começa do zero: um banco novo (em memória) e o contador de tokens zerado. Assim
  uma configuração nunca herda memória nem custo da anterior.
- O juiz usa o próprio `LLM`, com contador separado. Tokens, chamadas e custo de cada configuração
  não incluem o juiz, que é um custo da avaliação e não da estratégia de memória. O custo do juiz
  só aparece na linha "Custo total estimado" (ver `main`).
- Uma resposta `[erro: ...]` a uma pergunta continua sendo julgada (o juiz a recebe e a marca como
  incorreta) e continua contando no total de perguntas: a falha pesa contra a configuração.
- Os números vêm de uma única rodada, e saída de LLM não é determinística. A tabela mostra
  `acertos/N`, e não só porcentagem, para o tamanho da amostra ficar à vista.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from memoria.ativa import Ativa
from memoria.avaliacao.baselines import ContextoCompleto, SemMemoria
from memoria.avaliacao.juiz import julgar
from memoria.llm import LLM
from memoria.passiva import Passiva
from memoria.store import Store

# As quatro configurações da comparação, na ordem em que aparecem na tabela. Todas têm a mesma
# interface: Classe(llm, store), nova_sessao(data) e responder(mensagem).
CONFIGURACOES = {
    "sem_memoria": SemMemoria,
    "contexto_completo": ContextoCompleto,
    "passiva": Passiva,
    "ativa": Ativa,
}

# As categorias de pergunta, na ordem das colunas da tabela.
CATEGORIAS = ["sessao_unica", "multi_sessao", "temporal", "atualizacao", "abstencao"]

# Onde cada rodada grava seus arquivos (uma subpasta com data e hora). É constante de módulo para
# o teste poder trocá-la por uma pasta temporária.
PASTA_RESULTADOS = Path("resultados")


def rodar_configuracao(nome: str, dataset: dict, llm: LLM, llm_juiz: LLM) -> dict:
    """Roda uma configuração do começo ao fim no dataset e devolve o resultado dela.

    `llm` conversa (e, na passiva, resume); `llm_juiz` só julga. Mantê-los separados deixa o
    contador de `llm` medir só o custo da configuração. O resultado tem todos os turnos da
    conversa, as respostas às perguntas com o veredito de cada uma e os totais. Para a passiva,
    inclui o resumo S ao fim de cada sessão (para ver o tamanho dele crescer ou ser comprimido);
    para a ativa, quantas vezes o modelo usou cada ferramenta.
    """
    llm.contador.zerar()
    store = Store()
    configuracao = CONFIGURACOES[nome](llm, store)

    turnos = []
    resumos_por_sessao = []
    for sessao in dataset["sessoes"]:
        configuracao.nova_sessao(sessao["data"])
        for fala in sessao["falas"]:
            turnos.append({"sessao": sessao["data"], "usuario": fala, "assistente": configuracao.responder(fala)})
        if isinstance(configuracao, Passiva):
            resumo = store.ler_resumo()
            resumos_por_sessao.append({"sessao": sessao["data"], "palavras": len(resumo.split()), "resumo": resumo})

    perguntas = dataset["perguntas"]
    configuracao.nova_sessao(perguntas["data"])
    respostas = []
    for item in perguntas["itens"]:
        resposta = configuracao.responder(item["pergunta"])
        veredito = julgar(llm_juiz, item["pergunta"], item["esperado"], resposta, item["categoria"])
        respostas.append(
            {
                "id": item["id"],
                "categoria": item["categoria"],
                "pergunta": item["pergunta"],
                "esperado": item["esperado"],
                "resposta": resposta,
                "correto": veredito["correto"],
                "justificativa": veredito["justificativa"],
            }
        )

    # Só entram as categorias que o dataset tem, na ordem de CATEGORIAS, como [acertos, total].
    acertos_por_categoria = {}
    for categoria in CATEGORIAS:
        da_categoria = [r for r in respostas if r["categoria"] == categoria]
        if da_categoria:
            acertos_por_categoria[categoria] = [sum(r["correto"] for r in da_categoria), len(da_categoria)]

    # Um erro é qualquer resposta que começa com "[erro:", seja de uma fala da conversa ou de uma pergunta.
    textos = [t["assistente"] for t in turnos] + [r["resposta"] for r in respostas]
    erros = sum(texto.startswith("[erro:") for texto in textos)

    store.fechar()
    return {
        "configuracao": nome,
        "turnos": turnos,
        "resumos_por_sessao": resumos_por_sessao,
        "ferramentas": dict(configuracao.chamadas_ferramenta) if isinstance(configuracao, Ativa) else {},
        "respostas": respostas,
        "acertos_por_categoria": acertos_por_categoria,
        "acertos_total": [sum(r["correto"] for r in respostas), len(respostas)],
        "erros": erros,
        "chamadas": llm.contador.chamadas,
        "tokens_entrada": llm.contador.tokens_entrada,
        "tokens_saida": llm.contador.tokens_saida,
        "custo_usd": llm.contador.custo_usd,
    }


def tabela(resultados: list[dict]) -> str:
    """Monta a tabela Markdown da comparação: uma linha por configuração.

    As colunas são fixas: configuração, uma por categoria (na ordem de CATEGORIAS), total, erros,
    chamadas, tokens de entrada e de saída, custo e ferramentas. Acertos aparecem como
    `acertos/N`. Categoria que o resultado não tem aparece como `-`, e ferramentas só existem na
    ativa (`buscar/gravar`). O custo é só o da configuração, sem o juiz.
    """
    cabecalho = [
        "configuração", *CATEGORIAS, "total", "erros", "chamadas",
        "tokens entrada", "tokens saída", "custo (USD)", "ferramentas (buscar/gravar)",
    ]  # fmt: skip
    # O separador é "|---|---|" (sem espaço depois da barra) para só as linhas de dados começarem com "| ".
    linhas = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    for r in resultados:
        por_categoria = r["acertos_por_categoria"]
        ferramentas = r["ferramentas"]
        celulas = [
            r["configuracao"],
            *[f"{por_categoria[c][0]}/{por_categoria[c][1]}" if c in por_categoria else "-" for c in CATEGORIAS],
            f"{r['acertos_total'][0]}/{r['acertos_total'][1]}",
            str(r["erros"]),
            str(r["chamadas"]),
            str(r["tokens_entrada"]),
            str(r["tokens_saida"]),
            f"{r['custo_usd']:.4f}",
            f"{ferramentas['buscar_memoria']}/{ferramentas['gravar_memoria']}" if ferramentas else "-",
        ]
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas)


def main(argv: list[str]) -> None:
    """Roda a verificação e grava os arquivos em `resultados/<AAAAMMDD-HHMMSS>/`.

    Sem argumento roda as quatro configurações; com um argumento (o nome de uma delas) roda só
    essa. Não há outras opções. Um nome que não é de configuração encerra com uma mensagem antes de
    criar a pasta, para um erro de digitação não deixar uma pasta vazia em `resultados/`. Cada JSON
    é gravado assim que a configuração termina, então uma falha numa configuração mais adiante não
    perde as anteriores, que já custaram dinheiro.

    No fim imprime a tabela e o "Custo total estimado": a soma do custo das configurações com o
    custo do juiz, ou seja, o que a chave da API realmente gastou. A coluna de custo da tabela
    fica sem o juiz, para comparar só as estratégias de memória.
    """
    dataset = json.loads(Path(__file__).with_name("dataset.json").read_text(encoding="utf-8"))
    nomes = argv[:1] or list(CONFIGURACOES)
    if nomes[0] not in CONFIGURACOES:
        sys.exit(f"configuração desconhecida: {nomes[0]} (use {', '.join(CONFIGURACOES)})")
    llm = LLM()
    llm_juiz = LLM()

    pasta = PASTA_RESULTADOS / datetime.now().strftime("%Y%m%d-%H%M%S")
    pasta.mkdir(parents=True)

    resultados = []
    for nome in nomes:
        print(f"Rodando {nome}...", flush=True)  # a rodada paga é longa: uma linha por configuração mostra que anda
        resultado = rodar_configuracao(nome, dataset, llm, llm_juiz)
        (pasta / f"{nome}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        resultados.append(resultado)

    markdown = tabela(resultados)
    (pasta / "tabela.md").write_text(markdown + "\n", encoding="utf-8")

    custo_total = sum(r["custo_usd"] for r in resultados) + llm_juiz.contador.custo_usd
    print(f"\n{markdown}\n\nCusto total estimado: US$ {custo_total:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
