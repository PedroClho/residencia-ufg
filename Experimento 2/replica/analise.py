"""Análise: bootstrap pareado, vereditos, teste de reprodução, tabela e gráfico qualidade × tokens.

A regra de decisão foi registrada no desenho antes de rodar; este módulo só a aplica. Linhas com `erro`
contam como resposta errada e são contadas à parte na tabela de custos.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from replica.arquivos import ler_jsonl
from replica.config import FONTE_MHQA, MCC_FONTES, NUMEROS_ARTIGO, SEMENTE
from replica.rodar import CONFIGURACOES, ERROS_FINAIS

PRIMARIAS = [
    ("H1 direta", "direta-mhqa-lc-4omini", "direta-mhqa-bm25-4omini", +1),
    ("H2 direta", "direta-mcc-lc-4omini", "direta-mcc-bm25-4omini", -1),
    ("H1 OAB", "oab-h1-lc-4omini", "oab-h1-bm25-4omini", +1),
    ("H2 OAB", "oab-h2-lc-4omini", "oab-h2-bm25-4omini", -1),
]
# Cada primária é (título, contexto longo, RAG, sentido): sentido +1 quer dizer "o RAG vence" (H1) e -1
# "o contexto longo vence" (H2); a diferença é sempre média(RAG) − média(contexto longo).
_LOTE = 500  # reamostragens por lote, para não alocar n × N inteiros de uma vez
_POR_NOME = {c.nome: c for c in CONFIGURACOES}
NAO_RODADA = "não rodada"


# ---------------------------------------------------------------- estatística

def _medias_bootstrap(valores: np.ndarray, estratos, n: int, semente: int) -> np.ndarray:
    """`n` médias de reamostras (com reposição) de `valores`, dentro de cada estrato se houver."""
    rng = np.random.default_rng(semente)
    if estratos is None:
        grupos = [valores]
    else:
        est = np.asarray(estratos)
        grupos = [valores[est == e] for e in sorted(set(est.tolist()))]
    total = sum(len(g) for g in grupos)
    saida = np.empty(n)
    for ini in range(0, n, _LOTE):
        b = min(_LOTE, n - ini)
        soma = np.zeros(b)
        for g in grupos:
            soma += g[rng.integers(0, len(g), size=(b, len(g)))].sum(axis=1)
        saida[ini:ini + b] = soma / total
    return saida


def _intervalo(medias: np.ndarray) -> tuple[float, float]:
    li, ls = np.percentile(medias, [2.5, 97.5])
    return float(li), float(ls)


def intervalo_acuracia(acertos: list[bool], n: int = 10_000, semente: int = SEMENTE,
                       estratos: list[str] | None = None) -> tuple[float, float, float]:
    """`(acurácia, li, ls)`: bootstrap de percentis 2,5 e 97,5 (estratificado se `estratos`)."""
    v = np.asarray(acertos, dtype=float)
    li, ls = _intervalo(_medias_bootstrap(v, estratos, n, semente))
    return float(v.mean()), li, ls


def bootstrap_pareado(a: list[bool], b: list[bool], estratos: list[str] | None = None, n: int = 10_000,
                      semente: int = SEMENTE) -> tuple[float, float, float]:
    """`(dif = média(b) − média(a), li, ls)`; reamostra os pares (dentro de cada estrato, se houver)."""
    if len(a) != len(b):
        raise ValueError("as duas listas precisam ter o mesmo tamanho (pareadas)")
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    li, ls = _intervalo(_medias_bootstrap(d, estratos, n, semente))
    return float(d.mean()), li, ls


def veredito(li: float, ls: float, sentido: int) -> str:
    """Confirmada se o intervalo exclui zero na direção do artigo, contrariada na oposta, senão inconclusiva."""
    if (sentido > 0 and li > 0) or (sentido < 0 and ls < 0):
        return "confirmada"
    if (sentido > 0 and ls < 0) or (sentido < 0 and li > 0):
        return "contrariada"
    return "inconclusiva"


def reproduz(numero_artigo_pct: float, li: float, ls: float) -> bool:
    """O número do artigo (em %) cai dentro do nosso intervalo (proporções)?"""
    return li * 100 <= numero_artigo_pct <= ls * 100


# ---------------------------------------------------------------- dados

def carregar_rodada(pasta: Path) -> dict[str, list[dict]]:
    """`{nome da configuração: linhas}` para cada `*.jsonl` da pasta."""
    return {p.stem: ler_jsonl(p) for p in sorted(Path(pasta).glob("*.jsonl"))}


def _acerto(linha: dict) -> bool:
    return bool(linha.get("acerto")) and not linha.get("erro")


def _acerto_flexivel(linha: dict) -> bool:
    return bool((linha.get("metricas") or {}).get("exact_match_flexivel")) and not linha.get("erro")


def _estratos_mcc(linhas: list[dict]) -> list[str] | None:
    """Estratos (a `fonte`) se as linhas são de uma configuração MCC; `None` nas demais."""
    if any(l["fonte"] in MCC_FONTES for l in linhas):
        return [l["fonte"] for l in linhas]
    return None


def _acuracia_ic(linhas: list[dict], flexivel: bool = False) -> tuple[float, float, float]:
    """Acurácia com IC; nas MCC o bootstrap é sempre estratificado por `fonte`, em todas as seções."""
    f = _acerto_flexivel if flexivel else _acerto
    return intervalo_acuracia([f(l) for l in linhas], estratos=_estratos_mcc(linhas))


def _por_pergunta(linhas: list[dict]) -> dict[str, dict]:
    return {str(l["pergunta_id"]): l for l in linhas}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ic(acc: float, li: float, ls: float) -> str:
    return f"{_pct(acc)} [{_pct(li)}; {_pct(ls)}]"


def _ic_dif(d: float, li: float, ls: float) -> str:
    return f"{d * 100:+.1f} pp [{li * 100:+.1f}; {ls * 100:+.1f}]"


def _tabela(cabecalho: list[str], linhas: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join("---" for _ in cabecalho) + "|"]
    out += ["| " + " | ".join(l) + " |" for l in linhas]
    return out


# ---------------------------------------------------------------- seções

def _secao_primarias(dados: dict[str, list[dict]]) -> list[str]:
    linhas, notas = [], []
    for titulo, nome_lc, nome_rag, sentido in PRIMARIAS:
        if nome_lc not in dados or nome_rag not in dados:
            ausentes = [n for n in (nome_lc, nome_rag) if n not in dados]
            linhas.append([titulo, NAO_RODADA, "", "", "", "", f"sem: {', '.join(ausentes)}"])
            continue
        lc, rag = _por_pergunta(dados[nome_lc]), _por_pergunta(dados[nome_rag])
        ids = sorted(set(lc) & set(rag))
        if not ids:
            linhas.append([titulo, "0", "", "", "", "", "sem perguntas em comum"])
            continue
        a = [_acerto(lc[i]) for i in ids]  # contexto longo
        b = [_acerto(rag[i]) for i in ids]  # RAG
        estratos = [lc[i]["fonte"] for i in ids] if titulo == "H2 direta" else None
        # Diferença média(RAG) − média(LC). Na H2 o artigo diz LC > RAG, ou seja, diferença negativa.
        d, li, ls = bootstrap_pareado(a, b, estratos)
        _, la, sa = intervalo_acuracia(a, estratos=estratos)
        _, lb, sb = intervalo_acuracia(b, estratos=estratos)
        total = len(set(lc) | set(rag))
        api = sum(1 for l in (*dados[nome_lc], *dados[nome_rag])
                  if l.get("erro") and l.get("erro") not in ERROS_FINAIS)
        nota = f"(parcial: {len(ids)} de {total})" if len(ids) < total or api else ""
        if api and len(ids) == total:
            nota = f"(parcial: {len(ids)} de {total}, {api} erro(s) de API)"
        linhas.append([titulo, str(len(ids)), _ic(np.mean(a), la, sa), _ic(np.mean(b), lb, sb),
                       _ic_dif(d, li, ls), veredito(li, ls, sentido), nota])
        fora_lc, fora_rag = len(lc) - len(ids), len(rag) - len(ids)
        if fora_lc or fora_rag:
            notas.append(f"- {titulo}: {fora_lc} pergunta(s) de `{nome_lc}` e {fora_rag} de `{nome_rag}` "
                         f"descartadas por não estarem nas duas configurações.")
    out = ["## Primárias", "",
           "Contexto longo × RAG BM25, GPT-4o-mini, perguntas pareadas por `pergunta_id`. Diferença = "
           "RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (IC de 95%); linhas com erro "
           "contam como erradas. H1 espera diferença positiva (RAG vence), H2 negativa (contexto longo vence).",
           ""]
    out += _tabela(["Comparação", "n", "Contexto longo", "RAG BM25", "Diferença", "Veredito", "Nota"], linhas)
    if notas:
        out += [""] + notas
    return out + [""]


def _secao_reproducao(dados: dict[str, list[dict]]) -> list[str]:
    out = ["## Reprodução (etapa direta)", "",
           "Número do artigo (em %) contra a nossa acurácia com IC de 95% (bootstrap). "
           "Reproduz se o número do artigo cai dentro do intervalo. Média MCC: bootstrap estratificado por "
           "dataset.", ""]
    linhas, flex = [], []
    for cfg in (c for c in CONFIGURACOES if c.etapa == "direta"):
        if cfg.nome not in dados:
            linhas.append([cfg.nome, NAO_RODADA, "", "", "", ""])
            continue
        todas = dados[cfg.nome]
        fontes = [FONTE_MHQA] if cfg.conjunto == FONTE_MHQA else list(MCC_FONTES)
        itens = []
        for f in fontes:
            ls_f = [l for l in todas if l["fonte"] == f]
            itens.append((f, ls_f, None, NUMEROS_ARTIGO.get((f, cfg.agente))))
        if cfg.conjunto == "mcc":
            itens.append(("média MCC", [l for l in todas if l["fonte"] in MCC_FONTES],
                          "estratificar", NUMEROS_ARTIGO.get(("mcc", cfg.agente))))
        for rotulo, ls_f, modo, numero in itens:
            if not ls_f:
                linhas.append([cfg.nome, rotulo, NAO_RODADA, "", "", ""])
                continue
            est = [l["fonte"] for l in ls_f] if modo else None
            acc, li, ls = intervalo_acuracia([_acerto(l) for l in ls_f], estratos=est)
            if numero is None:
                linhas.append([cfg.nome, rotulo, str(len(ls_f)), _ic(acc, li, ls), "", ""])
            else:
                linhas.append([cfg.nome, rotulo, str(len(ls_f)), _ic(acc, li, ls), f"{numero:.1f}%",
                               "sim" if reproduz(numero, li, ls) else "não"])
            if cfg.conjunto == "mcc":
                accf, lif, lsf = intervalo_acuracia([_acerto_flexivel(l) for l in ls_f], estratos=est)
                flex.append([cfg.nome, rotulo, _ic(accf, lif, lsf)])
    out += _tabela(["Configuração", "Fonte", "n", "Nossa acurácia", "Artigo", "Reproduz"], linhas)
    if flex:
        out += ["", "Secundário, sem veredito: acurácia com leitura flexível de `label:` nas MCC "
                "(`exact_match_flexivel`; a primária usa o `exact_match` estrito do autor).", ""]
        out += _tabela(["Configuração", "Fonte", "Acurácia flexível"], flex)
    return out + [""]


def _nomes_presentes(dados: dict[str, list[dict]]) -> list[str]:
    ordem = [c.nome for c in CONFIGURACOES if c.nome in dados]
    return ordem + [n for n in dados if n not in _POR_NOME]


def _secao_secundarias(dados: dict[str, list[dict]]) -> list[str]:
    primarias = {n for _, a, b, _ in PRIMARIAS for n in (a, b)}
    nomes = [n for n in _nomes_presentes(dados) if n not in primarias]
    out = ["## Secundárias", "", "Acurácia com IC de 95%, sem veredito. As configurações diretas têm a "
           "comparação com o artigo na seção de reprodução.", ""]
    if not nomes:
        return out + ["Nenhuma configuração secundária na pasta.", ""]
    linhas = []
    for n in nomes:
        ls_n = dados[n]
        if not ls_n:
            linhas.append([n, "0", "sem linhas"])
            continue
        acc, li, ls = _acuracia_ic(ls_n)
        linhas.append([n, str(len(ls_n)), _ic(acc, li, ls)])
    return out + _tabela(["Configuração", "n", "Acurácia"], linhas) + [""]


def _secao_custos(dados: dict[str, list[dict]]) -> list[str]:
    out = ["## Custos", "",
           "Por configuração. Erros de API e recusas/respostas vazias são contados à parte e valem como erro.",
           ""]
    linhas = []
    for n in _nomes_presentes(dados):
        ls_n = dados[n]
        entrada = sum(l.get("tokens_entrada", 0) for l in ls_n)
        cache = sum(l.get("tokens_cacheados", 0) for l in ls_n)
        recusas = sum(1 for l in ls_n if l.get("erro") in ERROS_FINAIS)
        api = sum(1 for l in ls_n if l.get("erro") and l.get("erro") not in ERROS_FINAIS)
        linhas.append([n, str(len(ls_n)), f"{entrada:,}", _pct(cache / entrada) if entrada else "-",
                       f"{sum(l.get('tokens_saida', 0) for l in ls_n):,}",
                       f"{sum(l.get('custo_usd', 0.0) for l in ls_n):.4f}", str(api), str(recusas)])
    if not linhas:
        return out + ["Nenhuma configuração na pasta.", ""]
    out += _tabela(["Configuração", "n", "Tokens de entrada", "% cacheado", "Tokens de saída", "US$",
                    "Erros de API", "Recusas/vazias"], linhas)
    return out + [""]


def gerar_relatorio(pasta: Path) -> str:
    """Grava `pasta/tabela.md` (primárias, reprodução, secundárias e custos) e devolve o texto."""
    pasta = Path(pasta)
    dados = carregar_rodada(pasta)
    linhas = [f"# Resultados: {pasta.name}", ""]
    for secao in (_secao_primarias, _secao_reproducao, _secao_secundarias, _secao_custos):
        linhas += secao(dados)
    md = "\n".join(linhas)
    (pasta / "tabela.md").write_text(md, encoding="utf-8")
    return md


# ---------------------------------------------------------------- gráfico

_CURVA_K = [("oab-h1-bm25-4omini-k2", 2), ("oab-h1-bm25-4omini-k5", 5), ("oab-h1-bm25-4omini", 10),
            ("oab-h1-bm25-4omini-k20", 20)]
_CONTEXTOS_LONGOS = [("oab-h1-lc-4omini", "Contexto longo, GPT-4o-mini", "s"),
                     ("oab-h1-lc-haiku", "Contexto longo, Haiku 5.5", "D")]


def _ponto(linhas: list[dict]) -> tuple[float, float, float, float]:
    acc, li, ls = _acuracia_ic(linhas)
    x = float(np.mean([l.get("tokens_entrada", 0) for l in linhas]))
    return x, acc, acc - li, ls - acc


def grafico_qualidade_tokens(pasta: Path) -> Path | None:
    """Grava `pasta/qualidade_tokens.png` (OAB H1: acurácia × tokens de entrada por pergunta); `None` se
    nenhuma configuração da OAB H1 foi rodada."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pasta = Path(pasta)
    dados = carregar_rodada(pasta)
    curva = [(nome, k) for nome, k in _CURVA_K if dados.get(nome)]
    lcs = [(n, rot, m) for n, rot, m in _CONTEXTOS_LONGOS if dados.get(n)]
    if not curva and not lcs:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    if curva:
        pts = [_ponto(dados[n]) for n, _ in curva]
        ax.errorbar([p[0] for p in pts], [p[1] * 100 for p in pts],
                    yerr=[[p[2] * 100 for p in pts], [p[3] * 100 for p in pts]], marker="o", capsize=3,
                    label="RAG BM25, GPT-4o-mini (k = 2, 5, 10, 20)")
        for (_, k), p in zip(curva, pts):
            ax.annotate(f"k={k}", (p[0], p[1] * 100), textcoords="offset points", xytext=(6, -12))
    for n, rot, marcador in lcs:
        x, acc, e1, e2 = _ponto(dados[n])
        ax.errorbar([x], [acc * 100], yerr=[[e1 * 100], [e2 * 100]], marker=marcador, capsize=3, linestyle="",
                    label=rot)
    ax.set_xscale("log")
    ax.set_xlabel("Tokens de entrada médios por pergunta (escala logarítmica)")
    ax.set_ylabel("Acurácia (%) com IC de 95%")
    ax.set_title("OAB H1: qualidade × tokens de entrada")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    caminho = pasta / "qualidade_tokens.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m replica.analise",
                                 description="Tabela de vereditos e gráfico de uma pasta de rodada.")
    ap.add_argument("pasta", type=Path)
    args = ap.parse_args(argv)
    if not args.pasta.is_dir():
        print(f"pasta não existe: {args.pasta}", file=sys.stderr)
        return 2
    gerar_relatorio(args.pasta)
    print(args.pasta / "tabela.md")
    png = grafico_qualidade_tokens(args.pasta)
    if png:
        print(png)
    else:
        print("sem configuração da OAB H1 na pasta: gráfico não gerado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
