"""Análise: bootstrap pareado, vereditos, reprodução, tabela e gráfico. Nenhum teste chama API."""
from pytest import approx

from replica.analise import (
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
    metricas = {} if flex is None else {"exact_match_flexivel": flex}
    return {
        "configuracao": nome, "fonte": fonte, "contexto_id": 0, "pergunta_id": f"{fonte}:{i}", "ano": None,
        "consulta": "Pergunta com acentuação: ção É", "gabarito": ["x"], "saida": "x", "acerto": acerto,
        "metricas": metricas, "tokens_entrada": entrada, "tokens_cacheados": entrada // 2,
        "tokens_escritos_cache": 0, "tokens_saida": 5, "custo_usd": 0.001, "erro": erro,
    }


def _gravar(pasta, nome, fonte, acertos, **kw):
    for i, a in enumerate(acertos):
        anexar_jsonl(pasta / f"{nome}.jsonl", _linha(nome, fonte, i, a, **kw))


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
    # H2 direta: 5 fontes com 100 cada; LC 80%, BM25 70%
    for f in MCC_FONTES:
        _gravar(pasta, "direta-mcc-lc-4omini", f, [True] * 80 + [False] * 20, flex=True)
        _gravar(pasta, "direta-mcc-bm25-4omini", f, [True] * 70 + [False] * 30, flex=True)
    # OAB
    _gravar(pasta, "oab-h1-lc-4omini", "oab_h1", [True] * 50 + [False] * 50, entrada=100_000)
    _gravar(pasta, "oab-h1-bm25-4omini", "oab_h1", [True] * 60 + [False] * 40, entrada=3_000)
    _gravar(pasta, "oab-h2-lc-4omini", "oab_h2", [True] * 55 + [False] * 45, entrada=100_000)
    _gravar(pasta, "oab-h2-bm25-4omini", "oab_h2", [True] * 50 + [False] * 50, entrada=40_000)


def test_relatorio_tem_as_quatro_primarias(tmp_path):
    _rodada_completa(tmp_path)
    md = gerar_relatorio(tmp_path)
    assert (tmp_path / "tabela.md").read_text(encoding="utf-8") == md
    for titulo, *_ in PRIMARIAS:
        assert titulo in md
    for secao in ("Primárias", "Reprodução", "Secundárias", "Custos"):
        assert secao in md
    assert "confirmada" in md  # H1 direta: BM25 vence por 20 pontos
    assert "não rodada" in md  # as demais direta e secundárias não existem na pasta
    assert "média MCC" in md


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
    _gravar(tmp_path, "oab-h1-lc-4omini", "oab_h1", [True] * 10)
    # o BM25 tem só 8 perguntas em comum, 1 delas com erro de API, 1 com recusa
    _gravar(tmp_path, "oab-h1-bm25-4omini", "oab_h1", [True] * 8)
    anexar_jsonl(tmp_path / "oab-h1-bm25-4omini.jsonl", _linha("oab-h1-bm25-4omini", "oab_h1", 99, True))
    linhas = carregar_rodada(tmp_path)
    assert set(linhas) == {"oab-h1-lc-4omini", "oab-h1-bm25-4omini"}
    anexar_jsonl(tmp_path / "oab-h1-lc-4omini.jsonl",
                 _linha("oab-h1-lc-4omini", "oab_h1", 50, False, erro="[erro: api]"))
    anexar_jsonl(tmp_path / "oab-h1-lc-4omini.jsonl",
                 _linha("oab-h1-lc-4omini", "oab_h1", 51, False, erro="[erro: recusa]"))
    md = gerar_relatorio(tmp_path)
    assert "descartad" in md  # interseção diferente de um dos lados
    custos = md.split("Custos")[1]
    assert "oab-h1-lc-4omini" in custos


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
