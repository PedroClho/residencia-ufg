"""Testes do snapshot do Atlas: tudo offline, com get_json falso."""
from replica import arquivos
from replica.snapshot import buscar_paginado, salvar_snapshot, TABELAS


def test_paginacao_junta_todas_as_paginas():
    linhas = [{"id": i} for i in range(2345)]
    chamadas = []

    def falso(tabela, params, intervalo):
        chamadas.append(intervalo)
        ini, fim = (int(x) for x in intervalo.split("-"))
        return linhas[ini : fim + 1]

    r = buscar_paginado(falso, "t", "id")
    assert len(chamadas) == 3
    assert r == linhas


def _falso_que_registra(pedidas):
    def falso(tabela, params, intervalo):
        pedidas.append(tabela)
        if tabela == "questoes_oab":
            return [{"id": i, "enunciado": "Ética"} for i in range(3)]
        return [{"id": 1, "content": "art"}]

    return falso


def test_snapshot_grava_manifesto_e_nunca_pede_embeddings(tmp_path):
    pedidas = []
    m = salvar_snapshot(tmp_path, get_json=_falso_que_registra(pedidas))
    assert set(pedidas) == {"questoes_oab", "law_articles"} and "embeddings" not in pedidas
    assert m["tabelas"]["questoes_oab"]["linhas"] == 3
    assert len(m["tabelas"]["law_articles"]["sha256"]) == 64
    assert arquivos.ler_json(tmp_path / "manifesto.json") == m
    assert set(TABELAS) == {"questoes_oab", "law_articles"}


def test_snapshot_preserva_acentos(tmp_path):
    salvar_snapshot(tmp_path, get_json=_falso_que_registra([]))
    dados = arquivos.ler_json(tmp_path / "atlas" / "questoes_oab.json")
    assert dados[0]["enunciado"] == "Ética"
