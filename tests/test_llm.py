"""Testes do wrapper do cliente (llm.py): o que é enviado, a contagem de tokens e o tratamento de erro."""

from types import SimpleNamespace

import pytest

from memoria.llm import LLM, texto
from tests.conftest import erro_api, resposta_recusa, resposta_texto


def test_responder_passa_modelo_system_e_mensagens(fake):
    cliente = fake([resposta_texto("oi")])
    llm = LLM(cliente)
    llm.responder("sys", [{"role": "user", "content": "olá"}])
    chamada = cliente.chamadas[0]
    assert chamada["model"] == "claude-sonnet-5-5"
    assert chamada["system"] == "sys"
    assert chamada["messages"] == [{"role": "user", "content": "olá"}]
    assert "tools" not in chamada and "output_config" not in chamada


def test_responder_envia_ferramentas_e_formato_quando_dados(fake):
    ferramentas = [{"name": "buscar_memoria", "description": "d", "input_schema": {"type": "object"}}]
    formato = {"type": "json_schema", "schema": {"type": "object"}}
    cliente = fake([resposta_texto("a"), resposta_texto("b")])
    llm = LLM(cliente)
    mensagens = [{"role": "user", "content": "olá"}]

    llm.responder("sys", mensagens, ferramentas=ferramentas)
    llm.responder("sys", mensagens, formato=formato)

    com_ferramentas, com_formato = cliente.chamadas
    assert com_ferramentas["tools"] == ferramentas
    assert "output_config" not in com_ferramentas
    assert com_formato["output_config"] == {"format": formato}
    assert "tools" not in com_formato


def test_responder_devolve_a_mensagem_da_api(fake):
    resposta = resposta_texto("oi")
    llm = LLM(fake([resposta]))
    assert llm.responder("sys", [{"role": "user", "content": "olá"}]) is resposta


def test_contador_soma_tokens_e_chamadas(fake):
    cliente = fake([resposta_texto("a", 100, 20), resposta_texto("b", 50, 5)])
    llm = LLM(cliente)
    mensagens = [{"role": "user", "content": "olá"}]
    llm.responder("sys", mensagens)
    llm.responder("sys", mensagens)

    c = llm.contador
    assert (c.chamadas, c.tokens_entrada, c.tokens_saida) == (2, 150, 25)
    assert c.custo_usd == pytest.approx(150 * 2.0 / 1e6 + 25 * 10.0 / 1e6)

    c.zerar()
    assert (c.chamadas, c.tokens_entrada, c.tokens_saida) == (0, 0, 0)
    assert c.custo_usd == 0


def test_responder_devolve_none_em_erro_de_api(fake):
    llm = LLM(fake([erro_api()]))
    assert llm.responder("s", [{"role": "user", "content": "olá"}]) is None
    assert llm.contador.chamadas == 0  # chamada que falhou não gerou resposta para contar


def test_texto_junta_blocos_de_texto():
    assert texto(resposta_texto("abc")) == "abc"


def test_texto_ignora_blocos_que_nao_sao_texto():
    msg = resposta_texto("")
    msg.content = [
        SimpleNamespace(type="thinking", thinking="pensando"),
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="tool_use", name="x", input={}, id="tu_1"),
        SimpleNamespace(type="text", text="b"),
    ]
    assert texto(msg) == "ab"


def test_texto_de_none_e_erro():
    assert texto(None) == "[erro: api]"


def test_texto_de_recusa_e_erro():
    assert texto(resposta_recusa()) == "[erro: recusa]"
