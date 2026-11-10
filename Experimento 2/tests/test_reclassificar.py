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
from replica import reclassificar as rc

ROTULOS_DO_BANCO = (
    "Direito Civil", "Ética Profissional", "Direito Penal", "Direito Constitucional",
    "Direito Processual Civil", "Direito Administrativo", "Direito do Trabalho",
    "Direito Empresarial", "Direito Tributário", "Direito Processual do Trabalho",
    "Direito Processual Penal", "Direito da Criança e do Adolescente", "Direitos Humanos",
    "Direito do Consumidor", "Direito Internacional", "Filosofia do Direito",
    "Direito Ambiental", "Direito Eleitoral",
)


def _q(i, materia="Direito Civil", enunciado=None):
    return {"id": str(i), "numero_questao": i % 80 + 1, "exam_id": "2020-01", "ano": 2020, "materia": materia,
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


BLOCOS = (
    "Na prova objetiva da OAB aplicada pela FGV, as questões vêm em blocos por matéria, sempre nesta "
    "ordem de numeração: 1 a 8, Ética Profissional; 9 e 10, Filosofia do Direito; 11 a 17, Direito "
    "Constitucional; 18 e 19, Direitos Humanos; 20 e 21, Direito Internacional; 22 a 26, Direito "
    "Tributário; 27 a 32, Direito Administrativo; 33 e 34, Direito Ambiental; 35 a 41, Direito Civil; "
    "42 e 43, Direito da Criança e do Adolescente; 44 e 45, Direito do Consumidor; 46 a 50, Direito "
    "Empresarial; 51 a 57, Direito Processual Civil; 58 a 63, Direito Penal; 64 a 69, Direito "
    "Processual Penal; 70 a 75, Direito do Trabalho; 76 a 80, Direito Processual do Trabalho. A "
    "matéria da questão é a do bloco em que está o número dela. Só escolha outra matéria se o número "
    "estiver fora de 1 a 80 ou se o conteúdo não tiver relação nenhuma com a matéria do bloco."
)


def test_prompt_classificacao():
    q = {**_q(1), "numero_questao": 66}
    system, msg = prompt_classificacao(q)
    assert system == "Você classifica questões da prova da OAB pela matéria."
    esperado = (
        "Classifique a questão abaixo em exatamente uma destas matérias:\n"
        + "\n".join(f"- {m}" for m in MATERIAS)
        + "\n\n" + BLOCOS
        + "\n\nResponda só com o nome da matéria, exatamente como está na lista.\n\nQuestão número 66:\n"
        + formatar_questao(q)
    )
    assert msg == esperado


def test_prompt_traz_o_numero_da_questao():
    for n in (3, 77):
        _, msg = prompt_classificacao({**_q(1), "numero_questao": n})
        assert f"Questão número {n}:\n" in msg


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


class _LivroFalso:
    def __init__(self, *a, **k):
        pass

    def gasto(self, _):
        return 0.0


class _ClienteSempre:
    def __init__(self, *a, **k):
        pass

    def conversar(self, system, mensagem, max_tokens, prefixo_cacheavel=None):
        return Resposta("Direito Penal", 1, 0, 0, 1, 0.0, None)


def _preparar_main(monkeypatch, tmp_path):
    qs = [_q(i) for i in range(100)]
    monkeypatch.setattr(rc, "RAIZ", tmp_path)
    monkeypatch.setattr(rc, "DADOS", tmp_path / "dados")
    monkeypatch.setattr(rc, "LivroCaixa", _LivroFalso)
    monkeypatch.setattr(rc, "ClienteOpenAI", _ClienteSempre)
    monkeypatch.setattr(rc, "ler_json", lambda _: qs)
    monkeypatch.setattr(rc.oab, "questao_valida", lambda q: True)
    monkeypatch.setattr(rc.oab, "deduplicar", lambda x: x)
    (tmp_path / "dados").mkdir()
    return tmp_path / "docs" / "conferencia-materias.md"


def test_semente_muda_a_amostra():
    qs = [_q(i) for i in range(100)]
    mapa = {str(i): "Direito Penal" for i in range(100)}
    assert amostra_conferencia(qs, mapa, semente=1) != amostra_conferencia(qs, mapa, semente=2)


def test_main_aceita_semente(monkeypatch, tmp_path):
    arq = _preparar_main(monkeypatch, tmp_path)
    assert rc.main(["--semente", "43"]) == 0
    qs = [_q(i) for i in range(100)]
    mapa = {str(i): "Direito Penal" for i in range(100)}
    assert amostra_conferencia(qs, mapa, semente=43) in arq.read_text(encoding="utf-8")


def test_main_move_conferencia_preenchida(monkeypatch, tmp_path, capsys):
    arq = _preparar_main(monkeypatch, tmp_path)
    arq.parent.mkdir()
    preenchida = "| 1 | x | a | b | sim |\n"
    arq.write_text(preenchida, encoding="utf-8")
    assert rc.main([]) == 0
    anterior = arq.parent / "conferencia-materias-anterior.md"
    assert anterior.read_text(encoding="utf-8") == preenchida
    assert arq.read_text(encoding="utf-8") != preenchida
    assert "conferencia-materias-anterior.md" in capsys.readouterr().err


def test_main_sobrescreve_conferencia_vazia(monkeypatch, tmp_path):
    arq = _preparar_main(monkeypatch, tmp_path)
    arq.parent.mkdir()
    arq.write_text("| 1 | x | a | b |  |\n", encoding="utf-8")
    assert rc.main([]) == 0
    assert not (arq.parent / "conferencia-materias-anterior.md").exists()


def test_main_recusa_com_a_trava_do_rodador(monkeypatch, tmp_path, capsys):
    _preparar_main(monkeypatch, tmp_path)
    construidos = []
    monkeypatch.setattr(rc, "ClienteOpenAI", lambda *a, **k: construidos.append(1) or _ClienteSempre())
    trava = rc.TRAVA_RODADOR
    trava.parent.mkdir(parents=True, exist_ok=True)
    trava.write_text("pid 1", encoding="utf-8")
    assert rc.main([]) == 2
    assert "apague" in capsys.readouterr().err and construidos == [] and trava.exists()


def test_main_segura_a_trava_enquanto_classifica(monkeypatch, tmp_path):
    _preparar_main(monkeypatch, tmp_path)
    vista = []

    class ClienteQueOlhaATrava(_ClienteSempre):
        def conversar(self, *a, **k):
            vista.append(rc.TRAVA_RODADOR.exists())
            return super().conversar(*a, **k)

    monkeypatch.setattr(rc, "ClienteOpenAI", ClienteQueOlhaATrava)
    assert rc.main([]) == 0
    assert vista and all(vista) and not rc.TRAVA_RODADOR.exists()
