from types import SimpleNamespace

import pytest

from replica.clientes import Resposta
from replica.livro_caixa import LimiteDeGasto
from replica.oab import MATERIA_PARA_LEIS, formatar_questao
from replica.reclassificar import (
    MATERIAS,
    amostra_conferencia,
    interpretar,
    prompt_classificacao,
    reclassificar,
)

ROTULOS_DO_BANCO = (
    "Direito Civil", "Ética Profissional", "Direito Penal", "Direito Constitucional",
    "Direito Processual Civil", "Direito Administrativo", "Direito do Trabalho",
    "Direito Empresarial", "Direito Tributário", "Direito Processual do Trabalho",
    "Direito Processual Penal", "Direito da Criança e do Adolescente", "Direitos Humanos",
    "Direito do Consumidor", "Direito Internacional", "Filosofia do Direito",
    "Direito Ambiental", "Direito Eleitoral",
)


def _q(i, materia="Direito Civil", enunciado=None):
    return {"id": str(i), "exam_id": "2020-01", "ano": 2020, "materia": materia,
            "enunciado": enunciado or f"Enunciado {i}\ncom | barra",
            "alternativas": [{"letra": l, "texto": f"t{l}"} for l in "ABCD"], "gabarito": "A"}


class ClienteFalso:
    def __init__(self, textos, erro_em=()):
        self.textos = list(textos)
        self.erro_em = set(erro_em)
        self.chamadas = []

    def conversar(self, system, mensagem, max_tokens, prefixo_cacheavel=None):
        n = len(self.chamadas)
        self.chamadas.append((system, mensagem, max_tokens))
        erro = "falhou" if n in self.erro_em else None
        return Resposta(self.textos[n], 1, 0, 0, 1, 0.0, erro)


def test_materias_tem_20_e_cobre_rotulos_do_banco():
    assert len(MATERIAS) == 20 == len(set(MATERIAS))
    assert MATERIAS[:18] == tuple(MATERIA_PARA_LEIS)
    assert MATERIAS[18:] == ("Filosofia do Direito", "Direito Internacional")
    for r in ROTULOS_DO_BANCO:
        assert r in MATERIAS


def test_prompt_classificacao():
    q = _q(1)
    system, msg = prompt_classificacao(q)
    assert system == "Você classifica questões da prova da OAB pela matéria."
    assert msg.startswith("Classifique a questão abaixo em exatamente uma destas matérias:\n- Ética Profissional\n")
    assert "\n\nResponda só com o nome da matéria, exatamente como está na lista.\n\nQuestão:\n" in msg
    assert msg.endswith(formatar_questao(q))


def test_interpretar_aceita_so_a_lista():
    assert interpretar(" direito civil. ") == "Direito Civil"
    assert interpretar("ÉTICA PROFISSIONAL") == "Ética Profissional"
    assert interpretar("Direito Espacial") is None
    assert interpretar("") is None


def test_reclassificar_usa_o_cliente_e_mantem_rotulo_quando_invalido():
    qs = [_q(1, "Direito Civil"), _q(2, "Direito Civil"), _q(3, "Direito Tributário")]
    cliente = ClienteFalso(["Direito Penal", "???", "Direito Penal"], erro_em={2})
    mapa, sem = reclassificar(qs, cliente)
    assert mapa == {"1": "Direito Penal", "2": "Direito Civil", "3": "Direito Tributário"}
    assert sem == ["2", "3"]
    assert all(c[2] == 20 for c in cliente.chamadas) and len(cliente.chamadas) == 3
    assert cliente.chamadas[0][:2] == prompt_classificacao(qs[0])


def test_reclassificar_propaga_limite_de_gasto():
    class Estoura:
        def conversar(self, *a, **k):
            raise LimiteDeGasto("teto")

    with pytest.raises(LimiteDeGasto):
        reclassificar([_q(1)], Estoura())


def test_amostra_conferencia_tem_30_linhas_e_e_deterministica():
    qs = [_q(i) for i in range(100)]
    mapa = {str(i): "Direito Penal" for i in range(100)}
    a = amostra_conferencia(qs, mapa)
    assert a == amostra_conferencia(list(reversed(qs)), mapa)
    linhas = [l for l in a.splitlines() if l.startswith("|")]
    assert linhas[0] == "| id | trecho (150 caracteres) | rótulo do banco | rótulo novo | ok? |"
    assert len(linhas) == 2 + 30
    for l in linhas[2:]:
        assert l.endswith("|  |") and l.count("|") == 6  # a barra do enunciado foi trocada
        assert "Direito Civil" in l and "Direito Penal" in l
    assert "3 erros" in a
