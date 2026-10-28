"""Testes do banco (store.py): resumo único, notas e busca de texto completo (FTS5). SQLite puro, sem API."""

import pytest

from memoria.store import Store


@pytest.fixture
def s():
    """Um banco novo em memória para cada teste."""
    store = Store()
    yield store
    store.fechar()


def test_resumo_comeca_vazio(s):
    assert s.ler_resumo() == ""


def test_gravar_resumo_substitui(s):
    s.gravar_resumo("a")
    s.gravar_resumo("b")
    assert s.ler_resumo() == "b"


def test_gravar_nota_devolve_ids_crescentes(s):
    assert s.gravar_nota("2026-01-01", "x") < s.gravar_nota("2026-01-02", "y")


def test_buscar_nota_por_palavra(s):
    s.gravar_nota("2026-01-01", "O cachorro da usuária se chama Thor")
    s.gravar_nota("2026-01-02", "Ela trabalha na empresa Alfa")
    [n] = s.buscar_notas("cachorro")
    assert n["conteudo"].startswith("O cachorro")
    assert n["data"] == "2026-01-01"
    assert isinstance(n["id"], int)


def test_buscar_sem_resultado_devolve_lista_vazia(s):
    assert s.buscar_notas("nada") == []


def test_buscar_respeita_n(s):
    for i in range(7):
        s.gravar_nota(f"2026-01-{i + 1:02d}", f"detalhe {i} do projeto")
    assert len(s.buscar_notas("projeto", n=3)) == 3


def test_buscar_ordena_por_relevancia(s):
    # A nota mais relevante é gravada por último, para a ordem não ser só a de inserção.
    s.gravar_nota("2026-01-01", "Ontem fui ao mercado comprar pão, leite, ovos e um brinquedo para o gato")
    s.gravar_nota("2026-01-02", "gato gato")
    achadas = s.buscar_notas("gato")
    assert [n["data"] for n in achadas] == ["2026-01-02", "2026-01-01"]


def test_buscar_com_pontuacao_e_operadores_nao_quebra(s):
    s.gravar_nota("2026-01-01", "mora em Goiânia")
    # Não lança erro de sintaxe do FTS5 e ainda acha a nota pela palavra "Goiânia".
    assert s.buscar_notas('onde eu moro? "AND" (Goiânia) OR NOT *')


def test_buscar_consulta_so_com_pontuacao_devolve_vazio(s):
    s.gravar_nota("2026-01-01", "mora em Goiânia")
    assert s.buscar_notas("???") == []


def test_dados_persistem_em_arquivo(tmp_path):
    caminho = str(tmp_path / "memoria.db")
    primeiro = Store(caminho)
    primeiro.gravar_resumo("o resumo")
    primeiro.gravar_nota("2026-01-01", "gosta de café")
    primeiro.fechar()

    segundo = Store(caminho)
    assert segundo.ler_resumo() == "o resumo"
    assert [n["conteudo"] for n in segundo.buscar_notas("café")] == ["gosta de café"]
    segundo.fechar()
