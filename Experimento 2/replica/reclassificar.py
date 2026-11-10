"""Reclassifica a matéria das questões de 2018 a 2023 com o GPT-4o-mini.

Os rótulos do banco depois de 2018 são pouco confiáveis, e a matéria decide qual lei o agente do H1 recebe.
"""
import random
import re
import sys

from replica import oab
from replica.arquivos import gravar_json, ler_json
from replica.clientes import ClienteOpenAI
from replica.config import DADOS, LIVRO_CAIXA, MODELO_OPENAI, RAIZ, SEMENTE
from replica.livro_caixa import LivroCaixa
from replica.oab import MATERIA_PARA_LEIS, MATERIAS_FORA_H1, formatar_questao

MATERIAS: tuple[str, ...] = (*MATERIA_PARA_LEIS, *MATERIAS_FORA_H1)

SYSTEM = "Você classifica questões da prova da OAB pela matéria."


def prompt_classificacao(q: dict) -> tuple[str, str]:
    """`(system, mensagem)` para classificar a questão em uma das `MATERIAS`."""
    mensagem = (
        "Classifique a questão abaixo em exatamente uma destas matérias:\n"
        + "\n".join(f"- {m}" for m in MATERIAS)
        + "\n\nResponda só com o nome da matéria, exatamente como está na lista.\n\nQuestão:\n"
        + formatar_questao(q)
    )
    return SYSTEM, mensagem


def interpretar(saida: str) -> str | None:
    """Casa com a lista ignorando caixa, espaços e um ponto final; fora da lista vira `None`."""
    limpo = (saida or "").strip()
    if limpo.endswith("."):
        limpo = limpo[:-1]
    limpo = limpo.strip().casefold()
    for m in MATERIAS:
        if m.casefold() == limpo:
            return m
    return None


def reclassificar(qs: list[dict], cliente) -> tuple[dict[str, str], list[str]]:
    """Mapa id → matéria e ids sem resposta válida (que mantêm o rótulo do banco)."""
    mapa: dict[str, str] = {}
    sem_resposta: list[str] = []
    for i, q in enumerate(qs, 1):
        system, mensagem = prompt_classificacao(q)
        r = cliente.conversar(system, mensagem, max_tokens=20)
        materia = None if r.erro else interpretar(r.texto)
        qid = str(q["id"])
        if materia is None:
            sem_resposta.append(qid)
            materia = q["materia"]
        mapa[qid] = materia
        if i % 100 == 0:
            print(f"reclassificadas {i}/{len(qs)}", file=sys.stderr)
    return mapa, sem_resposta


def _trecho(enunciado: str, n: int = 150) -> str:
    return re.sub(r"[\r\n|]", " ", enunciado[:n])


def amostra_conferencia(qs: list[dict], mapa: dict[str, str], n: int = 30,
                        semente: int = SEMENTE) -> str:
    """Tabela Markdown de `n` questões sorteadas para o Pedro conferir a mão."""
    por_id = {str(q["id"]): q for q in qs}
    ids = random.Random(semente).sample(sorted(por_id), n)
    linhas = [
        "Confira cada linha: o rótulo novo é a matéria certa para a questão? Escreva `sim` ou `não` "
        "na coluna \"ok?\". Até 3 erros em 30 aprova a reclassificação.",
        "",
        "| id | trecho (150 caracteres) | rótulo do banco | rótulo novo | ok? |",
        "|---|---|---|---|---|",
    ]
    for qid in ids:
        q = por_id[qid]
        linhas.append(f"| {qid} | {_trecho(q['enunciado'])} | {q['materia']} | {mapa[qid]} |  |")
    return "\n".join(linhas) + "\n"


def main() -> int:
    todas = ler_json(DADOS / "atlas" / "questoes_oab.json")
    qs = oab.deduplicar([q for q in todas if oab.questao_valida(q) and 2018 <= int(q["ano"]) <= 2023])
    livro = LivroCaixa(LIVRO_CAIXA)
    cliente = ClienteOpenAI(MODELO_OPENAI, livro, temperatura=0.0)
    mapa, sem_resposta = reclassificar(qs, cliente)
    gravar_json(DADOS / "materias_2018_2023.json", mapa)
    docs = RAIZ / "docs" / "conferencia-materias.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("# Conferência da reclassificação de matéria\n\n" + amostra_conferencia(qs, mapa),
                    encoding="utf-8")
    mudaram = sum(1 for q in qs if mapa[str(q["id"])] != q["materia"])
    print(f"classificadas: {len(mapa)}")
    print(f"mantiveram o rótulo do banco (sem resposta válida): {len(sem_resposta)}")
    print(f"mudaram de rótulo: {mudaram}")
    print(f"gasto OpenAI no livro-caixa: US$ {livro.gasto('openai'):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
