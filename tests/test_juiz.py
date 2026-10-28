"""Testes do juiz (juiz.py): o que ele envia ao modelo e como trata cada tipo de resposta."""

from memoria.avaliacao.juiz import FORMATO, PROMPT_JUIZ, julgar
from memoria.llm import LLM
from tests.conftest import erro_api, resposta_recusa, resposta_texto


def test_julgar_envia_formato_e_le_json(fake):
    cliente = fake([resposta_texto('{"correto": true, "justificativa": "bate"}')])
    r = julgar(LLM(cliente), "Como se chama meu cachorro?", "Thor", "Seu cachorro é o Thor.", "sessao_unica")
    assert r == {"correto": True, "justificativa": "bate"}
    ch = cliente.chamadas[0]
    assert ch["output_config"] == {"format": FORMATO}
    assert ch["system"] == PROMPT_JUIZ
    assert "Thor" in ch["messages"][0]["content"] and "Como se chama" in ch["messages"][0]["content"]


def test_julgar_mensagem_tem_categoria_pergunta_esperado_e_resposta(fake):
    cliente = fake([resposta_texto('{"correto": false, "justificativa": "errou"}')])
    julgar(LLM(cliente), "Onde moro?", "Jardim América", "Não sei.", "atualizacao")
    msg = cliente.chamadas[0]["messages"][0]
    assert msg["role"] == "user"
    assert "Categoria: atualizacao" in msg["content"]
    assert "Pergunta: Onde moro?" in msg["content"]
    assert "Resposta esperada: Jardim América" in msg["content"]
    assert "Resposta do assistente: Não sei." in msg["content"]


def test_julgar_usa_o_mesmo_prompt_em_toda_categoria(fake):
    cliente = fake([resposta_texto('{"correto": true, "justificativa": "ok"}')] * 2)
    julgar(LLM(cliente), "p", "e", "r", "temporal")
    julgar(LLM(cliente), "p", "e", "r", "multi_sessao")
    assert cliente.chamadas[0]["system"] == cliente.chamadas[1]["system"] == PROMPT_JUIZ


def test_julgar_abstencao_menciona_regra_no_prompt(fake):
    cliente = fake([resposta_texto('{"correto": true, "justificativa": "disse que não foi mencionado"}')])
    julgar(LLM(cliente), "Qual a raça do cachorro?", "Não foi dito: a raça nunca foi mencionada", "Não sei.", "abstencao")
    system = cliente.chamadas[0]["system"]
    assert "abstencao" in system  # a regra da categoria está no prompt
    assert "inventou" in system  # e diz o que torna a resposta incorreta
    assert "Categoria: abstencao" in cliente.chamadas[0]["messages"][0]["content"]


def test_julgar_erro_de_api_conta_como_incorreto(fake):
    cliente = fake([erro_api()])
    r = julgar(LLM(cliente), "p", "e", "r", "sessao_unica")
    assert r == {"correto": False, "justificativa": "[erro: api]"}


def test_julgar_recusa_conta_como_incorreto(fake):
    cliente = fake([resposta_recusa()])
    r = julgar(LLM(cliente), "p", "e", "r", "sessao_unica")
    assert r == {"correto": False, "justificativa": "[erro: recusa]"}


def test_julgar_json_invalido_conta_como_incorreto(fake):
    cliente = fake([resposta_texto("isso não é json")])
    r = julgar(LLM(cliente), "p", "e", "r", "sessao_unica")
    assert r == {"correto": False, "justificativa": "[erro: json inválido]"}


def test_julgar_json_sem_as_chaves_conta_como_incorreto(fake):
    cliente = fake([resposta_texto('{"correto": true}')])
    r = julgar(LLM(cliente), "p", "e", "r", "sessao_unica")
    assert r == {"correto": False, "justificativa": "[erro: json inválido]"}
