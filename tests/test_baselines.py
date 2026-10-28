"""Testes das baselines (baselines.py): o que cada uma enxerga em cada chamada."""

import pytest

from memoria.avaliacao.baselines import ContextoCompleto, SemMemoria
from memoria.config import K, SYSTEM_BASE
from memoria.llm import LLM
from memoria.store import Store
from tests.conftest import erro_api, resposta_texto


def marcador(data: str) -> list[dict]:
    """O par que ContextoCompleto anexa ao histórico quando começa uma sessão."""
    return [{"role": "user", "content": f"(Nova sessão. Hoje é {data}.)"}, {"role": "assistant", "content": "Ok."}]


def turno(i: int) -> list[dict]:
    """As duas mensagens do turno i dos testes: usuário `mensagem i` e assistente `resposta i`."""
    return [{"role": "user", "content": f"mensagem {i}"}, {"role": "assistant", "content": f"resposta {i}"}]


def test_sem_memoria_nunca_ve_sessao_anterior(fake):
    cliente = fake([resposta_texto("resposta 1"), resposta_texto("resposta 2"), resposta_texto("resposta 3")])
    b = SemMemoria(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")
    b.responder("mensagem 1")
    b.responder("mensagem 2")

    b.nova_sessao("2026-03-08")
    b.responder("mensagem 3")

    assert cliente.chamadas[2]["messages"] == [{"role": "user", "content": "mensagem 3"}]


def test_sem_memoria_respeita_k(fake):
    cliente = fake([resposta_texto(f"resposta {i}") for i in range(1, 6)])
    b = SemMemoria(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")

    for i in range(1, 6):
        b.responder(f"mensagem {i}")

    historico = turno(1) + turno(2) + turno(3) + turno(4)
    quinta = cliente.chamadas[4]["messages"]
    assert quinta == historico[-K:] + [{"role": "user", "content": "mensagem 5"}]
    assert len(quinta) == K + 1  # vale com K == 6: 7 mensagens
    assert quinta[0] == {"role": "user", "content": "mensagem 2"}


def test_contexto_completo_ve_tudo_com_marcadores_de_sessao(fake):
    cliente = fake([resposta_texto("resposta 1"), resposta_texto("resposta 2")])
    b = ContextoCompleto(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")
    b.responder("mensagem 1")

    b.nova_sessao("2026-03-08")
    b.responder("mensagem 2")

    assert cliente.chamadas[1]["messages"] == (
        marcador("2026-03-01") + turno(1) + marcador("2026-03-08") + [{"role": "user", "content": "mensagem 2"}]
    )


def test_contexto_completo_nao_tem_janela(fake):
    # Mais de K mensagens na mesma sessão: nada é cortado.
    cliente = fake([resposta_texto(f"resposta {i}") for i in range(1, 6)])
    b = ContextoCompleto(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")

    for i in range(1, 6):
        b.responder(f"mensagem {i}")

    historico = turno(1) + turno(2) + turno(3) + turno(4)
    quinta = cliente.chamadas[4]["messages"]
    assert quinta == marcador("2026-03-01") + historico + [{"role": "user", "content": "mensagem 5"}]
    assert len(quinta) > K + 1


def test_contexto_completo_duas_sessoes_vazias_seguidas_mantem_alternancia(fake):
    cliente = fake([resposta_texto("resposta 1")])
    b = ContextoCompleto(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")
    b.nova_sessao("2026-03-08")

    b.responder("mensagem 1")

    mensagens = cliente.chamadas[0]["messages"]
    assert [m["role"] for m in mensagens] == ["user", "assistant", "user", "assistant", "user"]
    assert mensagens == marcador("2026-03-01") + marcador("2026-03-08") + [{"role": "user", "content": "mensagem 1"}]


@pytest.mark.parametrize("classe", [SemMemoria, ContextoCompleto])
def test_system_tem_o_prompt_base_e_a_data_da_sessao_atual(fake, classe):
    cliente = fake([resposta_texto("resposta 1"), resposta_texto("resposta 2")])
    b = classe(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")
    b.responder("mensagem 1")

    b.nova_sessao("2026-03-08")
    b.responder("mensagem 2")

    # O system é o mesmo nas duas baselines e só carrega a data da sessão atual; as datas das
    # sessões antigas, no ContextoCompleto, ficam nos marcadores do histórico.
    assert cliente.chamadas[0]["system"] == f"{SYSTEM_BASE}\n\nHoje é 2026-03-01."
    assert cliente.chamadas[1]["system"] == f"{SYSTEM_BASE}\n\nHoje é 2026-03-08."


@pytest.mark.parametrize("classe", [SemMemoria, ContextoCompleto])
def test_turno_com_erro_entra_no_historico_como_resposta_do_assistente(fake, classe):
    cliente = fake([erro_api(), resposta_texto("Oi!")])
    b = classe(LLM(cliente), Store())
    b.nova_sessao("2026-03-01")

    assert b.responder("primeira") == "[erro: api]"
    b.responder("segunda")

    # A alternância usuário/assistente não quebra: o erro ocupa o lugar da resposta.
    segunda = cliente.chamadas[1]["messages"]
    assert segunda[-3:] == [
        {"role": "user", "content": "primeira"},
        {"role": "assistant", "content": "[erro: api]"},
        {"role": "user", "content": "segunda"},
    ]
