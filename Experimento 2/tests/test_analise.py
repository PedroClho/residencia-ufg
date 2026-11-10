"""Análise: bootstrap pareado, vereditos, reprodução, tabela e gráfico. Nenhum teste chama API."""
from pytest import approx

from replica.analise import (
    NAO_RODADA,
    PRIMARIAS,
    bootstrap_pareado,
    carregar_rodada,
    gerar_relatorio,
    grafico_qualidade_tokens,
    intervalo_acuracia,
    main,
    reproduz,
    veredito,
)
from replica.arquivos import anexar_jsonl
from replica.config import FONTE_MHQA, MCC_FONTES


def _linha(nome, fonte, i, acerto, erro=None, entrada=1000, flex=None):
    # nas MCC, `acerto` é o exact_match estrito do autor; `flex` é a leitura flexível de "label: N"
    metricas = {"exact_match": acerto}
    if flex is not None:
        metricas["exact_match_flexivel"] = flex
    return {
        "configuracao": nome, "fonte": fonte, "contexto_id": 0, "pergunta_id": f"{fonte}:{i}", "ano": None,
        "consulta": "Pergunta com acentuação: ção É", "gabarito": ["x"], "saida": "x", "acerto": acerto,
        "metricas": metricas, "tokens_entrada": entrada, "tokens_cacheados": entrada // 2,
        "tokens_escritos_cache": 0, "tokens_saida": 5, "custo_usd": 0.001, "erro": erro,
    }


def _linhas_da_secao(md, titulo):
    """Células das linhas de dados da primeira tabela da seção que começa com `## <titulo>`."""
    bloco = md.split(f"## {titulo}")[1].split("\n## ")[0]
    linhas, em_tabela = [], False
    for l in bloco.splitlines():
        if l.startswith("|"):
            em_tabela = True
            linhas.append(l)
        elif em_tabela:
            break
    linhas = linhas[2:]
    return [[c.strip() for c in l.strip()[1:-1].split("|")] for l in linhas]


def _gravar(pasta, nome, fonte, acertos, flexiveis=None, **kw):
    for i, a in enumerate(acertos):
        flex = None if flexiveis is None else flexiveis[i]
        anexar_jsonl(pasta / f"{nome}.jsonl", _linha(nome, fonte, i, a, flex=flex, **kw))


def test_bootstrap_pareado_contem_a_diferenca():
    a = [True] * 60 + [False] * 40
    b = [True] * 75 + [False] * 25
    dif, li, ls = bootstrap_pareado(a, b)
    assert dif == approx(0.15) and li < 0.15 < ls
    assert bootstrap_pareado(a, b) == (dif, li, ls)  # determinístico


def test_intervalo_acuracia():
    acc, li, ls = intervalo_acuracia([True] * 60 + [False] * 40)
    assert acc == approx(0.6) and li < 0.6 < ls
    assert 0.45 < li and ls < 0.75
    assert intervalo_acuracia([True] * 60 + [False] * 40) == (acc, li, ls)


def test_veredito():
    assert veredito(0.02, 0.18, +1) == "confirmada"
    assert veredito(-0.18, -0.02, +1) == "contrariada"
    assert veredito(-0.01, 0.11, +1) == "inconclusiva"
    assert veredito(-0.12, -0.03, -1) == "confirmada"
    assert veredito(0.03, 0.12, -1) == "contrariada"
    assert veredito(-0.05, 0.05, -1) == "inconclusiva"


def test_bootstrap_estratificado_preserva_estratos():
    # Estrato "x": b ganha de a em todos (dif 1). Estrato "y": empate (dif 0). Dentro de cada estrato a
    # diferença é constante: reamostrando por estrato, a média da diferença é sempre 0,5.
    a = [False] * 50 + [True] * 50
    b = [True] * 50 + [True] * 50
    est = ["x"] * 50 + ["y"] * 50
    dif, li, ls = bootstrap_pareado(a, b, estratos=est)
    assert dif == approx(0.5) and ls - li < 1e-9
    # sem estratos, o intervalo tem largura de verdade
    _, li2, ls2 = bootstrap_pareado(a, b)
    assert ls2 - li2 > 0.1


def test_reproduz():
    assert reproduz(56.0, 0.50, 0.62) and not reproduz(43.0, 0.50, 0.62)


def _rodada_completa(pasta):
    # H1 direta: LC 40%, BM25 60% (100 perguntas)
    _gravar(pasta, "direta-mhqa-lc-4omini", FONTE_MHQA, [True] * 40 + [False] * 60)
    _gravar(pasta, "direta-mhqa-bm25-4omini", FONTE_MHQA, [True] * 60 + [False] * 40)
    # H2 direta: 5 fontes com 100 cada. Leitura flexível: LC 80%, BM25 70%. Estrito ("label: N" vira erro):
    # LC 0%, BM25 0%, como no piloto; com o estrito não haveria veredito.
    for f in MCC_FONTES:
        _gravar(pasta, "direta-mcc-lc-4omini", f, [False] * 100, flexiveis=[True] * 80 + [False] * 20)
        _gravar(pasta, "direta-mcc-bm25-4omini", f, [False] * 100, flexiveis=[True] * 70 + [False] * 30)
    # OAB
    # H1 OAB: o RAG perde de 10 pontos (contrariada); H2 OAB: diferença de 2 pontos (inconclusiva)
    _gravar(pasta, "oab-h1-lc-4omini", "oab_h1", [True] * 60 + [False] * 40, entrada=100_000)
    _gravar(pasta, "oab-h1-bm25-4omini", "oab_h1", [True] * 50 + [False] * 50, entrada=3_000)
    _gravar(pasta, "oab-h2-lc-4omini", "oab_h2", [True] * 52 + [False] * 48, entrada=100_000)
    _gravar(pasta, "oab-h2-bm25-4omini", "oab_h2", [True] * 50 + [False] * 50, entrada=40_000)


def test_relatorio_tem_as_quatro_primarias(tmp_path):
    _rodada_completa(tmp_path)
    md = gerar_relatorio(tmp_path)
    assert (tmp_path / "tabela.md").read_text(encoding="utf-8") == md
    for titulo, *_ in PRIMARIAS:
        assert titulo in md
    for secao in ("Primárias", "Reprodução", "Secundárias", "Custos"):
        assert secao in md
    assert "não rodada" in md  # as demais direta e secundárias não existem na pasta
    # veredito por linha: H1 direta (RAG +20) confirmada; H2 direta (LC +10) confirmada; H1 OAB (RAG -10)
    # contrariada; H2 OAB (LC +2, ruído) inconclusiva
    prim = {c[0]: c for c in _linhas_da_secao(md, "Primárias")}
    assert [prim[t][5] for t in ("H1 direta", "H2 direta", "H1 OAB", "H2 OAB")] == [
        "confirmada", "confirmada", "contrariada", "inconclusiva"]
    assert prim["H1 direta"][1] == "100" and prim["H2 direta"][1] == "500"
    assert prim["H1 direta"][2].startswith("40.0%") and prim["H1 direta"][3].startswith("60.0%")
    assert prim["H1 direta"][4].startswith("+20.0 pp") and prim["H2 direta"][4].startswith("-10.0 pp")
    assert all(prim[t][6] == "" for t in prim)  # rodada completa: sem marca de parcial
    # reprodução: média MCC e linhas por fonte
    rep = _linhas_da_secao(md, "Reprodução")
    por = {(c[0], c[1]): c for c in rep}
    assert por[("direta-mcc-lc-4omini", "média MCC")][2] == "500"
    assert por[("direta-mcc-lc-4omini", "média MCC")][4] == "82.0%"
    assert por[("direta-mcc-lc-4omini", "média MCC")][5] == "sim"  # 80% [76,84] contém 82
    assert por[("direta-mcc-bm25-4omini", "média MCC")][5] == "não"  # 70% [66,74] não contém 75,4
    assert por[("direta-mcc-lc-4omini", MCC_FONTES[0])][5] == "não"  # 80% contra 93
    assert por[("direta-mhqa-lc-4omini", FONTE_MHQA)][5] == "sim"  # 40% [31,50] contém 43
    assert por[("direta-mhqa-bm25-4omini", FONTE_MHQA)][5] == "sim"  # 60% [50,70] contém 56
    # coluna do estrito (literal do README): 0% nas MCC, vazia nas demais
    assert por[("direta-mcc-lc-4omini", "média MCC")][6] == "0.0% [0.0%; 0.0%]"
    assert por[("direta-mcc-bm25-4omini", MCC_FONTES[0])][6] == "0.0% [0.0%; 0.0%]"
    assert por[("direta-mhqa-lc-4omini", FONTE_MHQA)][6] == ""
    assert "Estrito (literal do README)" in md
    assert "leitura flexível de `label:`" in md  # nota sob as primárias
    assert "Acurácia flexível" not in md  # a tabela secundária de flexíveis foi removida
    assert por[("direta-mhqa-emb-4omini", NAO_RODADA)][0] == "direta-mhqa-emb-4omini"


def test_relatorio_pasta_incompleta(tmp_path):
    _gravar(tmp_path, "direta-mhqa-lc-4omini", FONTE_MHQA, [True] * 40 + [False] * 60)
    _gravar(tmp_path, "direta-mhqa-bm25-4omini", FONTE_MHQA, [True] * 60 + [False] * 40)
    md = gerar_relatorio(tmp_path)
    assert "H1 direta" in md and "H2 OAB" in md
    assert md.count("não rodada") >= 3
    # sem nenhuma configuração H1 da OAB, não há gráfico
    assert grafico_qualidade_tokens(tmp_path) is None
    assert not (tmp_path / "qualidade_tokens.png").exists()


def test_pareamento_pela_intersecao_e_erros_contam_como_erro(tmp_path):
    lc, rag = "oab-h1-lc-4omini", "oab-h1-bm25-4omini"
    _gravar(tmp_path, lc, "oab_h1", [True] * 10)
    # no contexto longo, as perguntas 8 e 9 viram um erro de API e uma recusa (valem como erradas)
    ls = carregar_rodada(tmp_path)[lc]
    (tmp_path / f"{lc}.jsonl").unlink()
    for l in ls[:8]:
        anexar_jsonl(tmp_path / f"{lc}.jsonl", l)
    anexar_jsonl(tmp_path / f"{lc}.jsonl", _linha(lc, "oab_h1", 8, True, erro="[erro: api]"))
    anexar_jsonl(tmp_path / f"{lc}.jsonl", _linha(lc, "oab_h1", 9, True, erro="[erro: recusa]"))
    # o RAG acerta as 10 e tem mais uma pergunta (99) que o contexto longo não tem
    _gravar(tmp_path, rag, "oab_h1", [True] * 10)
    anexar_jsonl(tmp_path / f"{rag}.jsonl", _linha(rag, "oab_h1", 99, True))
    md = gerar_relatorio(tmp_path)
    prim = {c[0]: c for c in _linhas_da_secao(md, "Primárias")}
    h1 = prim["H1 OAB"]
    assert h1[1] == "10"  # interseção de 10 perguntas
    assert h1[2].startswith("80.0%")  # os dois erros baixam a acurácia mesmo com acerto=True gravado
    assert h1[3].startswith("100.0%")
    assert "(parcial: 10 de 11" in h1[6]  # união de 11 ids, e há erro de API
    assert "descartad" in md and "1 de `oab-h1-bm25-4omini`" in md
    custos = {c[0]: c for c in _linhas_da_secao(md, "Custos")}
    assert custos[lc][1] == "10" and custos[lc][6] == "1" and custos[lc][7] == "1"  # API, recusas/vazias
    assert custos[rag][6] == "0" and custos[rag][7] == "0" and custos[rag][1] == "11"


def test_resposta_vazia_conta_como_recusa_e_nao_como_erro_de_api(tmp_path):
    _gravar(tmp_path, "oab-h1-lc-4omini", "oab_h1", [True] * 5)
    anexar_jsonl(tmp_path / "oab-h1-lc-4omini.jsonl",
                 _linha("oab-h1-lc-4omini", "oab_h1", 5, False, erro="[erro: resposta vazia]"))
    anexar_jsonl(tmp_path / "oab-h1-lc-4omini.jsonl",
                 _linha("oab-h1-lc-4omini", "oab_h1", 6, False, erro="Connection error"))
    md = gerar_relatorio(tmp_path)
    c = {x[0]: x for x in _linhas_da_secao(md, "Custos")}["oab-h1-lc-4omini"]
    assert c[6] == "1" and c[7] == "1"


def test_ic_das_mcc_e_estratificado_em_todas_as_secoes(tmp_path):
    # três fontes com 100% e duas com 0%: estratificado, a acurácia é 60% e o intervalo não tem largura
    for k, f in enumerate(MCC_FONTES):
        _gravar(tmp_path, "direta-mcc-emb-4omini", f, [False] * 20, flexiveis=[k < 3] * 20)
    md = gerar_relatorio(tmp_path)
    sec = {c[0]: c for c in _linhas_da_secao(md, "Secundárias")}
    assert sec["direta-mcc-emb-4omini"][2] == "60.0% [60.0%; 60.0%]"
    rep = {(c[0], c[1]): c for c in _linhas_da_secao(md, "Reprodução")}
    assert rep[("direta-mcc-emb-4omini", "média MCC")][3] == "60.0% [60.0%; 60.0%]"


def test_grafico_e_main(tmp_path, capsys):
    _rodada_completa(tmp_path)
    for k in (2, 5, 20):
        _gravar(tmp_path, f"oab-h1-bm25-4omini-k{k}", "oab_h1", [True] * (40 + k) + [False] * (60 - k),
                entrada=300 * k)
    png = grafico_qualidade_tokens(tmp_path)
    assert png == tmp_path / "qualidade_tokens.png" and png.stat().st_size > 1000
    png.unlink()
    assert main([str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "tabela.md" in saida and "qualidade_tokens.png" in saida
    assert (tmp_path / "tabela.md").exists() and png.exists()


def test_mcc_secundaria_usa_flexivel_e_erro_continua_errado(tmp_path):
    _gravar(tmp_path, "direta-mcc-emb-4omini", MCC_FONTES[0], [False] * 4, flexiveis=[True] * 4)
    anexar_jsonl(tmp_path / "direta-mcc-emb-4omini.jsonl",
                 _linha("direta-mcc-emb-4omini", MCC_FONTES[0], 4, False, erro="[erro: api]", flex=True))
    md = gerar_relatorio(tmp_path)
    sec = {c[0]: c for c in _linhas_da_secao(md, "Secundárias")}
    assert sec["direta-mcc-emb-4omini"][2].startswith("80.0%")  # 4 de 5; o erro vale como errado
