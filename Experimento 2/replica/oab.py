"""Questões e leis da OAB convertidas para linhas no formato de dados do autor."""
import difflib
import random
import re
import sys
import time
from pathlib import Path

from replica.arquivos import gravar_json, ler_json
from replica.config import DADOS, SEMENTE

LETRAS = ("A", "B", "C", "D")

MATERIA_PARA_LEIS: dict[str, tuple[str, ...]] = {
    "Ética Profissional": ("EAOAB", "RG-EAOAB", "CED"),
    "Direito Constitucional": ("CF",),
    "Direitos Humanos": ("CADH", "DUDH", "Lei 9.474"),
    "Direito Tributário": ("CTN",),
    "Direito Administrativo": ("Lei 8.112", "Lei 14.133", "Lei 8.987", "Lei 8.429", "Lei 9.784",
                               "Lei 12.016", "Lei 4.717"),
    "Direito Ambiental": ("Lei 12.651", "Lei 6.938", "Lei 9.605"),
    "Direito Civil": ("CC",),
    "Direito da Criança e do Adolescente": ("ECA",),
    "Direito do Consumidor": ("CDC",),
    "Direito Empresarial": ("Lei 6.404", "Lei 11.101"),
    "Direito Processual Civil": ("CPC",),
    "Direito Penal": ("CP",),
    "Direito Processual Penal": ("CPP",),
    "Direito do Trabalho": ("CLT",),
    "Direito Processual do Trabalho": ("CLT",),
    "Direito Eleitoral": ("Lei 4.737", "Lei 9.504", "Lei 9.096", "LC 64"),
    "Direito Previdenciário": ("Lei 8.212", "Lei 8.213", "LOAS"),
    "Direito Financeiro": ("LRF", "Lei 4.320"),
}
MATERIAS_FORA_H1 = ("Filosofia do Direito", "Direito Internacional")

_P_GABARITO = re.compile(r"(?i)gabarito\s*:?\s*\(?\s*([abcd])\b")
_P_ALTERNATIVA = re.compile(r"(?i)\b(?:alternativa|letra|op[cç][aã]o)\s*\(?\s*([abcd])\b")
_P_INICIO = re.compile(r"^\W*([ABCD])\s*(?:[).:\-]|\*\*|$)")
_P_ARTIGO = re.compile(r"^\s*Art\.?\s*(\d+(?:\.\d{3})*)\s*(?:º|°|o)?\s*(?:-\s*([A-Z]))?")


def formatar_questao(q: dict) -> str:
    """Enunciado, linha em branco e as alternativas `A) texto`."""
    return q["enunciado"] + "\n\n" + "\n".join(f"{a['letra']}) {a['texto']}" for a in q["alternativas"])


def extrair_letra(texto: str | None) -> str | None:
    """Lê a letra A a D na resposta do modelo; `None` se não achar."""
    if not texto:
        return None
    for padrao in (_P_GABARITO, _P_ALTERNATIVA):
        m = padrao.search(texto)
        if m:
            return m.group(1).upper()
    m = _P_INICIO.match(texto)
    return m.group(1) if m else None


def questao_valida(q: dict) -> bool:
    """Exatamente as alternativas A a D, gabarito em A a D e enunciado não vazio."""
    letras = tuple(a.get("letra") for a in q.get("alternativas") or [])
    return (letras == LETRAS and q.get("gabarito") in LETRAS
            and bool((q.get("enunciado") or "").strip()))


def _parecidas(sm: difflib.SequenceMatcher, a: str, limiar: float) -> bool:
    """`sm` já tem seq2 fixa; compara `a` (seq1). Limites superiores baratos antes do ratio()."""
    sm.set_seq1(a)
    if sm.real_quick_ratio() <= limiar or sm.quick_ratio() <= limiar:
        return False
    return sm.ratio() > limiar


def _similar_a_algum(enunciado: str, outros: list[str], limiar: float) -> bool:
    sm = difflib.SequenceMatcher(None)
    sm.set_seq2(enunciado)
    return any(_parecidas(sm, o, limiar) for o in outros)


def deduplicar(qs: list[dict], limiar: float = 0.8) -> list[dict]:
    """Tira a prova 2018-01 (quase igual à 2018-25) e as quase-duplicatas, ficando a de menor id."""
    mantidas: list[dict] = []
    enunciados: list[str] = []
    for q in sorted((q for q in qs if q["exam_id"] != "2018-01"), key=lambda q: str(q["id"])):
        if not _similar_a_algum(q["enunciado"], enunciados, limiar):
            mantidas.append(q)
            enunciados.append(q["enunciado"])
    return mantidas


def ordenar_artigos(artigos: list[dict], siglas: tuple[str, ...]) -> list[dict]:
    """Ordem de leitura: sigla na ordem dada, número do artigo, sufixo; sem número por último, por id."""
    def chave(a: dict):
        idx = siglas.index(a["law_sigla"]) if a["law_sigla"] in siglas else len(siglas)
        m = _P_ARTIGO.match(a["content"] or "")
        if not m:
            return (idx, 1, 0, "", str(a["id"]))
        return (idx, 0, int(m.group(1).replace(".", "")), m.group(2) or "", str(a["id"]))
    return sorted(artigos, key=chave)


def _base_h1(qs: list[dict]) -> list[dict]:
    return deduplicar([q for q in qs if questao_valida(q) and 2018 <= q["ano"] <= 2023])


def amostrar_h1(qs: list[dict], materias: dict[str, str], n: int = 300,
                semente: int = SEMENTE) -> list[dict]:
    """Amostra estratificada por matéria (maior resto), sorteada com `semente` sobre ids ordenados."""
    por_materia: dict[str, list[dict]] = {}
    for q in _base_h1(qs):
        m = materias.get(str(q["id"]))
        if m in MATERIA_PARA_LEIS and m not in MATERIAS_FORA_H1:
            por_materia.setdefault(m, []).append(q)
    total = sum(len(v) for v in por_materia.values())
    n = min(n, total)
    cotas = {m: n * len(v) / total for m, v in por_materia.items()}
    alocado = {m: int(c) for m, c in cotas.items()}
    restantes = n - sum(alocado.values())
    for m in sorted(cotas, key=lambda m: (-(cotas[m] - alocado[m]), m))[:restantes]:
        alocado[m] += 1
    rng = random.Random(semente)
    escolhidas = []
    for m in sorted(por_materia):
        ordenadas = sorted(por_materia[m], key=lambda q: str(q["id"]))
        escolhidas.extend(rng.sample(ordenadas, alocado[m]))
    return sorted(escolhidas, key=lambda q: str(q["id"]))


def _linha(source: str, context: str, qs: list[dict], leis: tuple[str, ...] | None = None) -> dict:
    meta = {"source": source, "qa_pair_ids": [str(q["id"]) for q in qs],
            "anos": [q["ano"] for q in qs]}
    if leis is not None:
        meta["leis"] = list(leis)
    return {"context": context, "questions": [formatar_questao(q) for q in qs],
            "answers": [[q["gabarito"]] for q in qs], "metadata": meta}


def _contexto_leis(artigos: list[dict], siglas: tuple[str, ...]) -> str:
    doc = [a for a in artigos if a["law_sigla"] in siglas]
    return "\n\n".join(a["content"] for a in ordenar_artigos(doc, siglas))


def montar_h1(amostra: list[dict], materias: dict[str, str], artigos: list[dict]) -> list[dict]:
    """Uma linha por tupla de leis (na ordem de MATERIA_PARA_LEIS) com as questões dessa tupla."""
    linhas = []
    vistas: list[tuple[str, ...]] = []
    for siglas in MATERIA_PARA_LEIS.values():
        if siglas in vistas:
            continue
        vistas.append(siglas)
        nomes = {m for m, s in MATERIA_PARA_LEIS.items() if s == siglas}
        qs = [q for q in amostra if materias[str(q["id"])] in nomes]
        if not qs:
            continue
        linhas.append(_linha(f"oab_h1_{len(linhas):02d}", _contexto_leis(artigos, siglas), qs, siglas))
    return linhas


def montar_h1_corpus(amostra: list[dict], artigos: list[dict]) -> list[dict]:
    """Uma linha só: todas as leis de MATERIA_PARA_LEIS num único contexto."""
    siglas: list[str] = []
    for s in MATERIA_PARA_LEIS.values():
        siglas.extend(x for x in s if x not in siglas)
    siglas_t = tuple(siglas)
    return [_linha("oab_h1_corpus", _contexto_leis(artigos, siglas_t), amostra, siglas_t)]


def separar_h2(qs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(memória, teste): teste são as válidas de 2024 e 2025; memória, as até 2023 sem quase-duplicata de teste."""
    validas = [q for q in qs if questao_valida(q)]
    teste = [q for q in validas if q["ano"] in (2024, 2025)]
    enun_teste = [q["enunciado"] for q in teste]
    memoria = [q for q in deduplicar([q for q in validas if q["ano"] <= 2023])
               if not _similar_a_algum(q["enunciado"], enun_teste, 0.8)]
    memoria.sort(key=lambda q: (q["exam_id"], q["numero_questao"]))
    return memoria, teste


def montar_h2(memoria: list[dict], teste: list[dict]) -> list[dict]:
    """Uma linha: contexto com as questões de memória e o gabarito, no formato `frase\\nlabel: x`."""
    contexto = "".join(f"{formatar_questao(q)}\nGabarito: {q['gabarito']}\n\n" for q in memoria)
    return [_linha("oab_h2", contexto, teste)]


def montar_sem_memoria(qs: list[dict]) -> list[dict]:
    """Uma linha com contexto vazio e todas as questões válidas."""
    return [_linha("oab_sem_memoria", "", [q for q in qs if questao_valida(q)])]


def _resumo(nome: str, linhas: list[dict], enc) -> None:
    print(f"{nome}: {len(linhas)} contexto(s), {sum(len(l['questions']) for l in linhas)} questões")
    for l in linhas:
        print(f"  {l['metadata']['source']}: {len(enc.encode(l['context']))} tokens o200k_base, "
              f"{len(l['questions'])} questões")


def main(dados: Path = DADOS) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    qs = ler_json(dados / "atlas" / "questoes_oab.json")
    artigos = ler_json(dados / "atlas" / "law_articles.json")
    presentes = {a["law_sigla"] for a in artigos}
    for m, siglas in MATERIA_PARA_LEIS.items():
        for s in siglas:
            if s not in presentes:
                print(f"AVISO: sigla {s} ({m}) sem nenhum artigo no snapshot", file=sys.stderr)
    saida = dados / "oab"
    conjuntos = {}

    t0 = time.time()
    memoria, teste = separar_h2(qs)
    print(f"separar_h2 (deduplicação incluída): {time.time() - t0:.1f}s; "
          f"memória {len(memoria)} questões, teste {len(teste)}")
    conjuntos["h2"] = montar_h2(memoria, teste)
    conjuntos["sem_memoria"] = montar_sem_memoria(qs)

    arq_materias = dados / "materias_2018_2023.json"
    if arq_materias.exists():
        materias = {str(k): v for k, v in ler_json(arq_materias).items()}
        amostra = amostrar_h1(qs, materias)
        conjuntos["h1"] = montar_h1(amostra, materias, artigos)
        conjuntos["h1_corpus"] = montar_h1_corpus(amostra, artigos)
    else:
        print(f"{arq_materias} não existe (a reclassificação ainda não rodou): "
              "pulando h1 e h1_corpus.")

    for nome, linhas in conjuntos.items():
        gravar_json(saida / f"{nome}.json", linhas)
        _resumo(nome, linhas, enc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
