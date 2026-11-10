"""Testes da conversão da OAB para o formato do autor. Tudo offline, com questões sintéticas."""
import hashlib
from collections import Counter

import pytest

from replica.oab import (LETRAS, MATERIA_PARA_LEIS, MATERIAS_FORA_H1, amostrar_h1, deduplicar,
                         extrair_letra, formatar_questao, montar_h1, montar_h1_corpus,
                         montar_h2, montar_sem_memoria, ordenar_artigos, questao_valida,
                         separar_h2)
from replica.templates_oab import obter_template


def _q(id_, ano=2020, exam=None, num=1, materia="Direito Civil", enun=None, gab="A", n_alt=4):
    return {
        "id": id_, "exam_id": exam or f"{ano}-01", "ano": ano, "numero_questao": num,
        "materia": materia, "gabarito": gab, "anulada": False,
        "enunciado": enun if enun is not None else hashlib.sha256(id_.encode()).hexdigest() * 2,
        "alternativas": [{"letra": L, "texto": f"texto {L} {id_}"} for L in "ABCD"[:n_alt]],
    }


def test_formatar_questao():
    q = {"enunciado": "Qual?", "alternativas": [{"letra": L, "texto": f"t{L}"} for L in "ABCD"]}
    assert formatar_questao(q) == "Qual?\n\nA) tA\nB) tB\nC) tC\nD) tD"


@pytest.mark.parametrize("saida,letra", [("B", "B"), ("Gabarito: c", "C"), ("A resposta correta é a alternativa D.", "D"),
    ("**A)** porque", "A"), ("Letra B", "B"), ("C.", "C"), ("", None), ("Não sei", None), ("E", None),
    ("A resposta correta é D", None), (None, None)])
def test_extrair_letra(saida, letra):
    assert extrair_letra(saida) == letra


def test_obter_template_despacha_oab_e_autor():
    assert obter_template("oab_h1_03", "query", "Simple_rag_bm25").endswith("Agora responda à questão: {question}")
    assert obter_template("ruler_qa2_421K", "system", "Simple_rag_bm25").startswith("You are a helpful assistant")
    assert obter_template("oab_h2", "query", "Long_context_agent").endswith("Gabarito:")
    assert "{{letra}}" in obter_template("oab_h2", "query", "Long_context_agent")
    assert obter_template("oab_h1", "system", "Long_context_agent").startswith("Você é")


def test_questao_valida_rejeita_incompletas():
    assert questao_valida(_q("a"))
    assert not questao_valida(_q("a", n_alt=3))
    assert not questao_valida(_q("a", gab="E"))
    assert not questao_valida(_q("a", enun="  "))
    q = _q("a")
    q["alternativas"][3]["letra"] = "E"
    assert not questao_valida(q)


def test_deduplicar_tira_2018_01_e_quase_iguais():
    base = "A empresa Alfa celebrou contrato de locação com o senhor José, que deixou de pagar três aluguéis"
    qs = [
        _q("2018-01_1", ano=2018, exam="2018-01", enun="Tema totalmente diferente sobre penal"),
        _q("2019-01_1", ano=2019, enun=base),
        _q("2019-01_2", ano=2019, enun=base + "."),
        _q("2019-01_3", ano=2019, enun="Pergunta sobre direito eleitoral e propaganda partidária em rádio"),
    ]
    ids = [q["id"] for q in deduplicar(qs)]
    assert ids == ["2019-01_1", "2019-01_3"]


def test_ordenar_artigos():
    arts = [{"id": str(i), "law_sigla": "CC", "content": c} for i, c in
            enumerate(["Art. 10. x", "Art. 2º x", "Art. 1.000. x", "Art. 2º-A x", "Parágrafo solto"])]
    out = [a["content"] for a in ordenar_artigos(arts, ("CC",))]
    assert out == ["Art. 2º x", "Art. 2º-A x", "Art. 10. x", "Art. 1.000. x", "Parágrafo solto"]


def test_ordenar_artigos_respeita_ordem_das_siglas():
    arts = [{"id": "1", "law_sigla": "CPC", "content": "Art. 1º a"},
            {"id": "2", "law_sigla": "CC", "content": "Art. 9º b"}]
    assert [a["law_sigla"] for a in ordenar_artigos(arts, ("CC", "CPC"))] == ["CC", "CPC"]


def _base_h1():
    todas = list(MATERIA_PARA_LEIS) + list(MATERIAS_FORA_H1)
    qs, mapa = [], {}
    for i in range(1000):
        ano = 2018 + i % 6
        enun = hashlib.sha256(f"s{i}".encode()).hexdigest() * 2
        q = _q(f"{ano}-{i:04d}_{i}", ano=ano, exam=f"{ano}-02", num=i, enun=enun)
        qs.append(q)
        mapa[str(q["id"])] = todas[i % len(todas)]
    return qs, mapa


def test_amostra_h1_estratificada_e_deterministica():
    qs, mapa = _base_h1()
    a1 = amostrar_h1(qs, mapa, n=300)
    a2 = amostrar_h1(qs, mapa, n=300)
    assert len(a1) == 300
    assert [q["id"] for q in a1] == [q["id"] for q in a2]
    assert all(mapa[str(q["id"])] not in MATERIAS_FORA_H1 for q in a1)
    elegiveis = [q for q in qs if mapa[str(q["id"])] in MATERIA_PARA_LEIS]
    pop = Counter(mapa[str(q["id"])] for q in elegiveis)
    got = Counter(mapa[str(q["id"])] for q in a1)
    for m, c in pop.items():
        cota = 300 * c / len(elegiveis)
        assert int(cota) <= got[m] <= int(cota) + 1
    assert [q["id"] for q in amostrar_h1(qs, mapa, n=300, semente=1)] != [q["id"] for q in a1]


def test_h1_agrupa_por_lei_e_nao_vaza():
    qs = [_q(f"2020-01_{i}", num=i, materia=m) for i, m in enumerate(
        ["Direito do Trabalho", "Direito Processual do Trabalho", "Direito Civil"])]
    mapa = {str(q["id"]): q["materia"] for q in qs}
    artigos = [{"id": "1", "law_sigla": "CLT", "content": "Art. 2º CLT texto"},
               {"id": "2", "law_sigla": "CLT", "content": "Art. 1º CLT texto"},
               {"id": "3", "law_sigla": "CC", "content": "Art. 1º CC texto"}]
    linhas = montar_h1(qs, mapa, artigos)
    assert len(linhas) == 2
    clt = next(l for l in linhas if l["metadata"]["leis"] == ["CLT"])
    assert len(clt["questions"]) == 2
    assert clt["context"] == "Art. 1º CLT texto\n\nArt. 2º CLT texto"
    for l in linhas:
        for i, qid in enumerate(l["metadata"]["qa_pair_ids"]):
            q = next(x for x in qs if x["id"] == qid)
            assert l["questions"][i] == formatar_questao(q)
            assert l["answers"][i] == [q["gabarito"]]
            assert q["enunciado"][:80] not in l["context"]
    assert [l["metadata"]["source"] for l in linhas] == ["oab_h1_00", "oab_h1_01"]
    assert all(isinstance(a, int) for l in linhas for a in l["metadata"]["anos"])


def test_h1_corpus_tem_todas_as_leis():
    qs = [_q("2020-01_1", materia="Direito Civil")]
    artigos = [{"id": "1", "law_sigla": "CC", "content": "Art. 1º a"},
               {"id": "2", "law_sigla": "CPC", "content": "Art. 1º b"},
               {"id": "3", "law_sigla": "ZZZ", "content": "Art. 1º fora"}]
    (linha,) = montar_h1_corpus(qs, artigos)
    assert linha["metadata"]["source"] == "oab_h1_corpus"
    assert linha["context"] == "Art. 1º a\n\nArt. 1º b"
    assert len(linha["questions"]) == 1


def test_h2_sem_vazamento_e_em_ordem():
    teste_enun = "Sobre a responsabilidade civil do Estado por omissão e o dever de indenizar o particular lesado"
    qs = [
        _q("2024-01_1", ano=2024, num=1, enun=teste_enun),
        _q("2025-01_1", ano=2025, num=1),
        _q("2023-01_5", ano=2023, exam="2023-01", num=5),
        _q("2019-01_2", ano=2019, exam="2019-01", num=2),
        _q("2019-01_1", ano=2019, exam="2019-01", num=1),
        _q("2021-01_9", ano=2021, exam="2021-01", num=9, enun=teste_enun + "!"),
        _q("2022-01_1", ano=2022, exam="2022-01", num=1, n_alt=3),
    ]
    memoria, teste = separar_h2(qs)
    assert {q["ano"] for q in teste} == {2024, 2025}
    assert [q["id"] for q in memoria] == ["2019-01_1", "2019-01_2", "2023-01_5"]
    (linha,) = montar_h2(memoria, teste)
    assert linha["metadata"]["source"] == "oab_h2"
    assert linha["context"].count("Gabarito: A") == 3
    assert linha["context"].startswith(formatar_questao(memoria[0]) + "\nGabarito: A\n\n")
    assert len(linha["questions"]) == 2
    for q in teste:
        assert q["enunciado"][:80] not in linha["context"]


def test_sem_memoria():
    qs = [_q("2024-01_1", ano=2024), _q("2025-01_1", ano=2025, n_alt=3)]
    (linha,) = montar_sem_memoria(qs)
    assert linha["context"] == "" and len(linha["questions"]) == 1
    assert linha["metadata"]["source"] == "oab_sem_memoria"
    assert LETRAS == ("A", "B", "C", "D")


def test_main_sem_memoria_tem_todas_as_validas_e_h2_so_teste(tmp_path):
    from replica.arquivos import gravar_json, ler_json
    from replica.oab import main
    qs = [_q("2019-01_1", ano=2019, exam="2019-01"), _q("2022-01_1", ano=2022, exam="2022-01"),
          _q("2022-01_2", ano=2022, exam="2022-01", num=2, n_alt=3),
          _q("2024-01_1", ano=2024), _q("2025-01_1", ano=2025)]
    gravar_json(tmp_path / "atlas" / "questoes_oab.json", qs)
    gravar_json(tmp_path / "atlas" / "law_articles.json",
                [{"id": "1", "law_sigla": "CC", "content": "Art. 1º a"}])
    assert main(dados=tmp_path) == 0
    (sem,) = ler_json(tmp_path / "oab" / "sem_memoria.json")
    assert sem["metadata"]["qa_pair_ids"] == ["2019-01_1", "2022-01_1", "2024-01_1", "2025-01_1"]
    (h2,) = ler_json(tmp_path / "oab" / "h2.json")
    assert h2["metadata"]["qa_pair_ids"] == ["2024-01_1", "2025-01_1"]
    assert not (tmp_path / "oab" / "h1.json").exists()
