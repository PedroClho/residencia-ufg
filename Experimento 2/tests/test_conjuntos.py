"""Testes de arquivos UTF-8 e dos conjuntos de dados. Tudo offline."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from replica.arquivos import anexar_jsonl, gravar_json, ler_json, ler_jsonl
from replica.conjuntos import Contexto, Pergunta, carregar_autor, carregar_oab


def test_json_utf8_ida_e_volta(tmp_path):
    gravar_json(tmp_path / "a.json", {"matéria": "Ética Profissional", "x": "ção"})
    assert ler_json(tmp_path / "a.json") == {"matéria": "Ética Profissional", "x": "ção"}
    assert "Ética" in (tmp_path / "a.json").read_text(encoding="utf-8")


def test_jsonl_anexa_e_ignora_linhas_em_branco(tmp_path):
    caminho = tmp_path / "sub" / "r.jsonl"
    assert ler_jsonl(caminho) == []
    anexar_jsonl(caminho, {"a": "É"})
    with open(caminho, "a", encoding="utf-8") as f:
        f.write("\n\n")
    anexar_jsonl(caminho, {"a": "ç"})
    assert ler_jsonl(caminho) == [{"a": "É"}, {"a": "ç"}]
    assert "É" in caminho.read_text(encoding="utf-8")


def _parquet(tmp_path, linhas):
    caminho = tmp_path / "x.parquet"
    tabela = pa.table({
        "context": [l[1] for l in linhas],
        "questions": [l[2] for l in linhas],
        "answers": [l[3] for l in linhas],
        "metadata": [{"source": l[0]} for l in linhas],
    })
    pq.write_table(tabela, caminho)
    return caminho


def _falso(caminho, chamadas=None):
    def baixar(**kw):
        if chamadas is not None:
            chamadas.append(kw)
        return str(caminho)
    return baixar


def test_carregar_autor_filtra_a_fonte_e_numera_perguntas(tmp_path):
    caminho = _parquet(tmp_path, [
        ("ruler_qa2_421K", "c" * 3000, ["q0", "q1"], [["yes"], ["no"]]),
        ("outra", "d" * 3000, ["z"], [["w"]]),
    ])
    chamadas = []
    ctx = carregar_autor(["ruler_qa2_421K"], pasta=tmp_path, baixar=_falso(caminho, chamadas))
    assert len(ctx) == 1 and ctx[0].perguntas[1].id == "ruler_qa2_421K:1"
    assert ctx[0].perguntas[0].gabarito == ["yes"]
    assert ctx[0].perguntas[0].texto == "q0"
    assert chamadas[0]["filename"] == "data/Accurate_Retrieval-00000-of-00001.parquet"
    assert chamadas[0]["revision"] == "7ea066982b140a19337e17e60d45d4076e042faf"


def test_carregar_varias_fontes_na_ordem(tmp_path):
    caminho = _parquet(tmp_path, [
        ("icl_a", "a" * 3000, ["q"], [["1"]]),
        ("icl_b", "b" * 3000, ["q"], [["2"]]),
    ])
    chamadas = []
    ctx = carregar_autor(["icl_b", "icl_a"], pasta=tmp_path, baixar=_falso(caminho, chamadas))
    assert [(c.id, c.fonte) for c in ctx] == [(0, "icl_b"), (1, "icl_a")]
    assert chamadas[0]["filename"].startswith("data/Test_Time_Learning-")


def test_fonte_inexistente_levanta(tmp_path):
    caminho = _parquet(tmp_path, [("ruler_x", "a" * 3000, ["q"], [["1"]])])
    with pytest.raises(ValueError, match="ruler_y"):
        carregar_autor(["ruler_y"], pasta=tmp_path, baixar=_falso(caminho))


def test_contexto_curto_falha(tmp_path):
    caminho = _parquet(tmp_path, [("ruler_x", "curto", ["q"], [["1"]])])
    with pytest.raises(AssertionError):
        carregar_autor(["ruler_x"], pasta=tmp_path, baixar=_falso(caminho))


def test_carregar_oab_le_ids_e_anos(tmp_path):
    linhas = [{
        "context": "Questão: ...",
        "questions": ["Qual?", "Quem?"],
        "answers": [["A"], ["C"]],
        "metadata": {"source": "oab_x", "qa_pair_ids": [101, "2018-25-7"], "anos": [2012, 2018]},
    }]
    gravar_json(tmp_path / "oab_x.json", linhas)
    ctx = carregar_oab("oab_x", pasta=tmp_path)
    assert isinstance(ctx[0], Contexto) and ctx[0].fonte == "oab_x" and ctx[0].id == 0
    p = ctx[0].perguntas
    assert p[0] == Pergunta(id="101", texto="Qual?", gabarito=["A"], ano=2012)
    assert p[1].id == "2018-25-7" and p[1].ano == 2018 and p[1].gabarito == ["C"]
