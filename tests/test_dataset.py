"""Testes do dataset (dataset.json): validação de esquema do gabarito da verificação. Só lê o arquivo, sem API."""

import json
from datetime import date
from pathlib import Path

import pytest

import memoria.avaliacao

CATEGORIAS = {"sessao_unica", "multi_sessao", "temporal", "atualizacao", "abstencao"}


@pytest.fixture
def dataset():
    """O dataset.json versionado ao lado do pacote de avaliação."""
    caminho = Path(memoria.avaliacao.__file__).parent / "dataset.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


def test_estrutura_basica(dataset):
    assert isinstance(dataset["usuario"], str) and dataset["usuario"].strip()
    sessoes = dataset["sessoes"]
    assert 4 <= len(sessoes) <= 6
    for s in sessoes:
        assert 8 <= len(s["falas"]) <= 12
        assert all(isinstance(f, str) and f.strip() for f in s["falas"])
    # date.fromisoformat aceita "2026-03-14" e recusa o que não é data ISO.
    datas = [date.fromisoformat(s["data"]) for s in sessoes]
    assert all(a < b for a, b in zip(datas, datas[1:]))
    assert date.fromisoformat(dataset["perguntas"]["data"]) > datas[-1]


def test_perguntas(dataset):
    itens = dataset["perguntas"]["itens"]
    assert 15 <= len(itens) <= 20
    ids = [i["id"] for i in itens]
    assert len(ids) == len(set(ids))
    for i in itens:
        assert i["categoria"] in CATEGORIAS
        assert i["pergunta"].strip()
        assert i["esperado"].strip()
    for c in CATEGORIAS:
        assert 3 <= sum(i["categoria"] == c for i in itens) <= 4, c


def test_abstencao_espera_nao_sei(dataset):
    abstencoes = [i for i in dataset["perguntas"]["itens"] if i["categoria"] == "abstencao"]
    assert abstencoes
    assert all(i["esperado"].startswith("Não foi dito") for i in abstencoes)
