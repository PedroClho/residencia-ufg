import numpy as np
from pytest import approx, raises

from tests.conftest import FakeAnthropic, FakeOpenAI
from replica.clientes import (
    ClienteAnthropic,
    ClienteEmbeddings,
    ClienteOpenAI,
    fabricar_cliente,
)
from replica.livro_caixa import LimiteDeGasto, custo_anthropic, custo_openai


def test_openai_manda_mensagem_como_o_autor(livro, fake_openai):
    c = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake_openai)
    c.conversar("SYS", "pergunta", 50, prefixo_cacheavel="CONTEXTO\n")
    kw = fake_openai.chamadas[0]
    assert kw["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "CONTEXTO\npergunta"},
    ]
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 50


def test_openai_registra_tokens_cacheados_e_custo(livro, fake_openai):  # prompt=1000, cached=800, completion=5
    r = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake_openai).conversar("S", "m", 20)
    assert r.tokens_cacheados == 800
    assert r.tokens_entrada == 1000 and r.tokens_saida == 5 and r.tokens_escritos_cache == 0
    assert livro.gasto("openai") == approx(custo_openai("gpt-4o-mini", 1000, 800, 5))


def test_anthropic_cacheia_o_prefixo_e_desliga_o_raciocinio(livro, fake_anthropic):
    ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake_anthropic).conversar(
        "SYS", "pergunta", 20, prefixo_cacheavel="CTX\n"
    )
    kw = fake_anthropic.chamadas[0]
    assert kw["messages"][0]["content"][0] == {
        "type": "text", "text": "CTX\n", "cache_control": {"type": "ephemeral"},
    }
    assert kw["messages"][0]["content"][1]["text"] == "pergunta"
    assert kw["thinking"] == {"type": "disabled"} and "temperature" not in kw and kw["system"] == "SYS"


def test_anthropic_sem_prefixo_manda_texto_simples(livro, fake_anthropic):
    ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake_anthropic).conversar("SYS", "pergunta", 20)
    assert fake_anthropic.chamadas[0]["messages"] == [{"role": "user", "content": "pergunta"}]


def test_anthropic_soma_tokens_e_registra_custo(livro):
    usage = dict(entrada=100, escrita=2000, leitura=3000, saida=7)
    fake = FakeAnthropic(responder=lambda kw: ("ok", usage))
    r = ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake).conversar("S", "m", 20)
    assert (r.tokens_entrada, r.tokens_cacheados, r.tokens_escritos_cache, r.tokens_saida) == (5100, 3000, 2000, 7)
    assert livro.gasto("anthropic") == approx(custo_anthropic("claude-haiku-5-5", 100, 2000, 3000, 7))


def test_teto_recusa_antes_de_chamar_a_api(livro, fake_openai):
    livro.registrar("openai", 19.5)  # acima do teto efetivo (19,5 menos a reserva de 4)
    with raises(LimiteDeGasto):
        ClienteOpenAI("gpt-4o-mini", livro, bruto=fake_openai).conversar("S", "m", 20)
    assert fake_openai.chamadas == []


def test_recusa_vira_erro(livro):
    fake = FakeAnthropic(stop_reason="refusal")
    r = ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake).conversar("S", "m", 20)
    assert r.erro == "[erro: recusa]" and r.texto == ""


def test_texto_vazio_vira_erro(livro):
    fake = FakeAnthropic(responder=lambda kw: ("", dict(entrada=10, escrita=0, leitura=0, saida=0)))
    r = ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake).conversar("S", "m", 20)
    assert r.erro == "[erro: resposta vazia]"


def test_excecao_vira_erro_de_api_sem_custo(livro, capsys):
    def responder(kw):
        raise RuntimeError("boom")

    fake = FakeOpenAI(responder=responder)
    r = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake).conversar("S", "m", 20)
    assert r.erro == "[erro: api]" and r.texto == ""
    assert livro.gasto("openai") == 0
    assert "boom" in capsys.readouterr().err


class ErroDeTaxa(Exception):
    pass


def test_limite_de_taxa_espera_e_tenta_de_novo(livro):
    falhas = []

    def responder(kw):
        if len(falhas) < 2:
            falhas.append(1)
            raise ErroDeTaxa("429")
        return "ok", dict(prompt=10, cached=0, completion=1)

    esperas = []
    c = ClienteOpenAI(
        "gpt-4o-mini", livro, bruto=FakeOpenAI(responder=responder),
        dormir=esperas.append, erros_de_taxa=(ErroDeTaxa,),
    )
    r = c.conversar("S", "m", 20)
    assert esperas == [60, 60] and r.erro is None and r.texto == "ok"


def test_limite_de_taxa_desiste_depois_de_cinco_tentativas(livro):
    def responder(kw):
        raise ErroDeTaxa("429")

    esperas = []
    fake = FakeOpenAI(responder=responder)
    c = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake, dormir=esperas.append, erros_de_taxa=(ErroDeTaxa,))
    r = c.conversar("S", "m", 20)
    assert r.erro == "[erro: api]" and len(esperas) == 5 and len(fake.chamadas) == 6


def test_embeddings_em_lotes_registram_custo(livro, fake_openai):
    textos = [f"texto numero {i}" for i in range(600)]
    m = ClienteEmbeddings(livro, bruto=fake_openai, lote=256).embutir(textos)
    assert len(fake_openai.chamadas_embeddings) == 3
    assert m.shape == (600, 8) and m.dtype in (np.float32, np.float64)
    assert livro.gasto("openai") > 0


def test_embeddings_fecham_o_lote_por_tokens_e_truncam(livro, fake_openai):
    longo = "palavra " * 40_000  # ~40 mil tokens cada
    ClienteEmbeddings(livro, bruto=fake_openai, lote=256).embutir([longo] * 40)
    # cada texto é truncado em 8.191 tokens; 250 mil tokens por requisição cabem 30 textos
    assert [len(kw["input"]) for kw in fake_openai.chamadas_embeddings] == [30, 10]
    assert all(len(t) < len(longo) for kw in fake_openai.chamadas_embeddings for t in kw["input"])


def test_fabricar_cliente_escolhe_provedor(livro):
    assert isinstance(fabricar_cliente("gpt-4o-mini", livro, 0.7), ClienteOpenAI)
    assert isinstance(fabricar_cliente("claude-haiku-5-5", livro, 0.7), ClienteAnthropic)
