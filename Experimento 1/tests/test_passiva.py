"""Testes da memória passiva (passiva.py): o que vai no prompt, o que é gravado e o que acontece nos erros."""

from memoria.config import K
from memoria.llm import LLM
from memoria.passiva import PROMPT_RESUMIDOR, RESUMO_VAZIO, Passiva
from memoria.store import Store
from tests.conftest import erro_api, resposta_recusa, resposta_texto

RESUMO_ANA = "## Sobre o usuário\n- chama-se Ana"


def test_resumo_vazio_tem_as_quatro_secoes():
    for secao in ["## Sobre o usuário", "## Fatos e decisões", "## Pendências", "## Linha do tempo"]:
        assert f"{secao}\n(nada ainda)" in RESUMO_VAZIO


def test_prompt_do_resumidor_tem_os_marcadores_de_limite_e_data():
    assert "{limite}" in PROMPT_RESUMIDOR and "{data}" in PROMPT_RESUMIDOR


def test_primeiro_turno_injeta_resumo_vazio_e_faz_duas_chamadas(fake):
    cliente = fake([resposta_texto("Olá!"), resposta_texto(RESUMO_ANA)])
    p = Passiva(LLM(cliente), Store())
    p.nova_sessao("2026-03-01")

    assert p.responder("Oi, sou a Ana") == "Olá!"

    principal, resumidor = cliente.chamadas
    assert RESUMO_VAZIO in principal["system"] and "Hoje é 2026-03-01" in principal["system"]
    assert principal["messages"] == [{"role": "user", "content": "Oi, sou a Ana"}]
    # O resumidor recebe uma única mensagem com o resumo atual e o turno que acabou de acontecer.
    assert len(resumidor["messages"]) == 1
    conteudo = resumidor["messages"][0]["content"]
    assert RESUMO_VAZIO in conteudo and "Oi, sou a Ana" in conteudo and "Olá!" in conteudo
    # Os marcadores do prompt foram preenchidos com o limite de palavras e a data da sessão.
    assert "{" not in resumidor["system"] and "Hoje é 2026-03-01" in resumidor["system"]


def test_resumo_novo_e_gravado_e_usado_no_turno_seguinte(fake):
    cliente = fake([
        resposta_texto("Olá!"), resposta_texto(RESUMO_ANA),
        resposta_texto("Oi de novo, Ana."), resposta_texto("## Sobre o usuário\n- chama-se Ana\n- gosta de café"),
    ])
    store = Store()
    p = Passiva(LLM(cliente), store)
    p.nova_sessao("2026-03-01")

    p.responder("Oi, sou a Ana")
    assert store.ler_resumo() == RESUMO_ANA

    p.responder("Gosto de café")
    principal2, resumidor2 = cliente.chamadas[2], cliente.chamadas[3]
    assert RESUMO_ANA in principal2["system"] and RESUMO_VAZIO not in principal2["system"]
    assert RESUMO_ANA in resumidor2["messages"][0]["content"]
    assert store.ler_resumo() == "## Sobre o usuário\n- chama-se Ana\n- gosta de café"


def test_janela_envia_exatamente_as_ultimas_k_mensagens(fake):
    roteiro = []
    for i in range(1, 6):
        roteiro += [resposta_texto(f"resposta {i}"), resposta_texto(f"resumo {i}")]
    cliente = fake(roteiro)
    p = Passiva(LLM(cliente), Store())
    p.nova_sessao("2026-03-01")

    for i in range(1, 6):
        p.responder(f"mensagem {i}")

    historico = []
    for i in range(1, 5):
        historico += [{"role": "user", "content": f"mensagem {i}"}, {"role": "assistant", "content": f"resposta {i}"}]
    quinta_principal = cliente.chamadas[8]  # cada turno faz duas chamadas: a principal e a do resumidor
    assert quinta_principal["messages"] == historico[-K:] + [{"role": "user", "content": "mensagem 5"}]
    assert len(quinta_principal["messages"]) == K + 1
    assert quinta_principal["messages"][0] == {"role": "user", "content": "mensagem 2"}  # vale com K == 6


def test_nova_sessao_zera_janela_mas_mantem_resumo(fake):
    cliente = fake([resposta_texto("Olá!"), resposta_texto(RESUMO_ANA), resposta_texto("Oi, Ana."), resposta_texto("resumo 2")])
    p = Passiva(LLM(cliente), Store())
    p.nova_sessao("2026-03-01")
    p.responder("Oi, sou a Ana")

    p.nova_sessao("2026-03-08")
    p.responder("Lembra de mim?")

    principal2 = cliente.chamadas[2]
    assert principal2["messages"] == [{"role": "user", "content": "Lembra de mim?"}]
    assert RESUMO_ANA in principal2["system"] and "Hoje é 2026-03-08" in principal2["system"]


def test_erro_na_resposta_principal_nao_roda_resumidor(fake):
    cliente = fake([resposta_recusa()])
    store = Store()
    p = Passiva(LLM(cliente), store)
    p.nova_sessao("2026-03-01")

    assert p.responder("x") == "[erro: recusa]"

    assert len(cliente.chamadas) == 1
    assert store.ler_resumo() == ""


def test_turno_com_erro_entra_na_janela_como_resposta_do_assistente(fake):
    cliente = fake([erro_api(), resposta_texto("Oi!"), resposta_texto("resumo")])
    p = Passiva(LLM(cliente), Store())
    p.nova_sessao("2026-03-01")

    assert p.responder("primeira") == "[erro: api]"
    p.responder("segunda")

    assert cliente.chamadas[1]["messages"] == [
        {"role": "user", "content": "primeira"},
        {"role": "assistant", "content": "[erro: api]"},
        {"role": "user", "content": "segunda"},
    ]


def test_erro_no_resumidor_mantem_resumo_anterior(fake):
    cliente = fake([
        resposta_texto("Olá!"), resposta_texto(RESUMO_ANA),
        resposta_texto("Anotado."), erro_api(),  # o resumidor do segundo turno falha
    ])
    store = Store()
    p = Passiva(LLM(cliente), store)
    p.nova_sessao("2026-03-01")
    p.responder("Oi, sou a Ana")

    assert p.responder("Gosto de café") == "Anotado."  # o usuário ainda recebe a resposta

    assert store.ler_resumo() == RESUMO_ANA


def test_resumidor_vazio_mantem_resumo_anterior(fake):
    cliente = fake([
        resposta_texto("Olá!"), resposta_texto(RESUMO_ANA),
        resposta_texto("Anotado."), resposta_texto(""),  # ex.: saída cortada antes de escrever o resumo
    ])
    store = Store()
    p = Passiva(LLM(cliente), store)
    p.nova_sessao("2026-03-01")
    p.responder("Oi, sou a Ana")

    p.responder("Gosto de café")

    assert store.ler_resumo() == RESUMO_ANA
