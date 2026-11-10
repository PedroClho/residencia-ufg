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
    assert r.custo_usd > 0 and livro.gasto("anthropic") == approx(r.custo_usd)  # a recusa foi paga


def test_texto_vazio_vira_erro(livro):
    fake = FakeAnthropic(responder=lambda kw: ("", dict(entrada=10, escrita=0, leitura=0, saida=0)))
    r = ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake).conversar("S", "m", 20)
    assert r.erro == "[erro: resposta vazia]"
    assert r.custo_usd > 0 and livro.gasto("anthropic") == approx(r.custo_usd)  # a resposta vazia foi paga


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


def test_prefixo_identico_entre_chamadas(livro, fake_openai, fake_anthropic):
    prefixo = "CONTEXTO LONGO\n"
    co = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake_openai)
    ca = ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake_anthropic)
    for pergunta in ("primeira?", "segunda?"):
        co.conversar("SYS", pergunta, 20, prefixo_cacheavel=prefixo)
        ca.conversar("SYS", pergunta, 20, prefixo_cacheavel=prefixo)
    u1, u2 = (kw["messages"][1]["content"] for kw in fake_openai.chamadas)
    assert u1.startswith(prefixo) and u2.startswith(prefixo)
    a1, a2 = (kw["messages"][0]["content"][0] for kw in fake_anthropic.chamadas)
    assert a1 == a2 and a1["text"] == prefixo


def test_anthropic_prefixo_vazio_conta_como_sem_prefixo(livro, fake_anthropic):
    ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake_anthropic).conversar("S", "pergunta", 20, prefixo_cacheavel="")
    assert fake_anthropic.chamadas[0]["messages"] == [{"role": "user", "content": "pergunta"}]


def test_embeddings_texto_vazio_vira_espaco(livro, fake_openai):
    m = ClienteEmbeddings(livro, bruto=fake_openai).embutir(["a", "", "  \n", "b"])
    assert m.shape == (4, 8)
    assert fake_openai.chamadas_embeddings[0]["input"] == ["a", " ", " ", "b"]


def test_embeddings_lista_vazia_nao_chama_api(livro, fake_openai):
    m = ClienteEmbeddings(livro, bruto=fake_openai).embutir([])
    assert m.shape == (0, 0) and fake_openai.chamadas_embeddings == []


# --- custo registrado antes de ler a resposta, erro fatal do livro-caixa e limite do painel ---

from types import SimpleNamespace  # noqa: E402

from pytest import fixture  # noqa: E402

from replica.livro_caixa import ErroDoLivroCaixa, LimiteDoPainel, LivroCaixa  # noqa: E402


def test_openai_resposta_vazia_e_paga(livro):
    fake = FakeOpenAI(responder=lambda kw: ("", dict(prompt=100, cached=0, completion=0)))
    r = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake).conversar("S", "m", 20)
    assert r.erro == "[erro: resposta vazia]" and r.custo_usd > 0
    assert livro.gasto("openai") == approx(r.custo_usd)


def _bruto_openai(resposta):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resposta)))


def _bruto_anthropic(resposta):
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resposta))


def test_resposta_paga_ilegivel_vira_resposta_vazia_com_custo(livro):
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=5,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=0))
    r = ClienteOpenAI("gpt-4o-mini", livro, bruto=_bruto_openai(SimpleNamespace(choices=[], usage=usage))).conversar(
        "S", "m", 20)
    assert r.erro == "[erro: resposta vazia]" and r.texto == ""
    assert r.custo_usd == approx(custo_openai("gpt-4o-mini", 1000, 0, 5)) and r.tokens_entrada == 1000
    assert livro.gasto("openai") == approx(r.custo_usd)

    usage_a = SimpleNamespace(input_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                              output_tokens=5)
    quebrada = SimpleNamespace(content=None, stop_reason="end_turn", usage=usage_a)
    r = ClienteAnthropic("claude-haiku-5-5", livro, bruto=_bruto_anthropic(quebrada)).conversar("S", "m", 20)
    assert r.erro == "[erro: resposta vazia]" and r.custo_usd == approx(custo_anthropic("claude-haiku-5-5", 100, 0, 0, 5))
    assert livro.gasto("anthropic") == approx(r.custo_usd)


def test_usage_ilegivel_registra_o_custo_maximo_autorizado(livro, capsys):
    resposta = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="B"))], usage=None)
    r = ClienteOpenAI("gpt-4o-mini", livro, bruto=_bruto_openai(resposta)).conversar("S", "m", 20)
    assert r.texto == "B" and r.erro is None and r.custo_usd > 0
    assert livro.gasto("openai") == approx(r.custo_usd)
    assert "usage" in capsys.readouterr().err


class _LivroQueNaoGrava(LivroCaixa):
    def registrar(self, provedor, custo):
        raise ErroDoLivroCaixa(f"{provedor}: não gravei US$ {custo:.4f}")


@fixture
def livro_quebrado(tmp_path):
    return _LivroQueNaoGrava(tmp_path / "l.json")


def test_falha_do_livro_caixa_propaga_pelos_clientes(livro_quebrado, fake_openai, fake_anthropic):
    with raises(ErroDoLivroCaixa):
        ClienteOpenAI("gpt-4o-mini", livro_quebrado, bruto=fake_openai).conversar("S", "m", 20)
    with raises(ErroDoLivroCaixa):
        ClienteAnthropic("claude-haiku-5-5", livro_quebrado, bruto=fake_anthropic).conversar("S", "m", 20)
    with raises(ErroDoLivroCaixa):
        ClienteEmbeddings(livro_quebrado, bruto=fake_openai).embutir(["a"])


class ErroDaApi(Exception):
    def __init__(self, mensagem, code=None):
        super().__init__(mensagem)
        self.code = code


def test_cota_esgotada_da_openai_para_sem_esperar(livro):
    def responder(kw):
        raise ErroDaApi("Error code: 429 - You exceeded your current quota", code="insufficient_quota")

    esperas, fake = [], FakeOpenAI(responder=responder)
    c = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake, dormir=esperas.append, erros_de_taxa=(ErroDaApi,))
    with raises(LimiteDoPainel) as e:
        c.conversar("S", "m", 20)
    assert e.value.provedor == "openai" and isinstance(e.value, LimiteDeGasto)
    assert esperas == [] and len(fake.chamadas) == 1 and livro.gasto("openai") == 0


def test_saldo_ou_limite_de_uso_da_anthropic_para(livro):
    for mensagem in ("Your credit balance is too low to access the Anthropic API.",
                     "You have reached your specified workspace API usage limits.",
                     "Monthly spend limit reached"):
        def responder(kw, mensagem=mensagem):
            raise ErroDaApi(mensagem)

        with raises(LimiteDoPainel) as e:
            ClienteAnthropic("claude-haiku-5-5", livro, bruto=FakeAnthropic(responder=responder)).conversar("S", "m", 20)
        assert e.value.provedor == "anthropic"


def test_outro_erro_continua_erro_de_api(livro):
    def responder(kw):
        raise ErroDaApi("Error code: 429 - rate limit exceeded", code="rate_limit_exceeded")

    esperas = []
    c = ClienteOpenAI("gpt-4o-mini", livro, bruto=FakeOpenAI(responder=responder), dormir=esperas.append,
                      erros_de_taxa=(ErroDaApi,))
    assert c.conversar("S", "m", 20).erro == "[erro: api]" and len(esperas) == 5


def test_embeddings_com_cota_esgotada_param(livro):
    def criar(**kw):
        raise ErroDaApi("Error code: 429 - insufficient_quota", code="insufficient_quota")

    bruto = SimpleNamespace(embeddings=SimpleNamespace(create=criar))
    with raises(LimiteDoPainel) as e:
        ClienteEmbeddings(livro, bruto=bruto).embutir(["a"])
    assert e.value.provedor == "openai"


def test_embeddings_com_usage_ilegivel_registram_o_custo_estimado(livro):
    resposta = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])], usage=None)
    bruto = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: resposta))
    m = ClienteEmbeddings(livro, bruto=bruto).embutir(["texto qualquer"])
    assert m.shape == (1, 2) and livro.gasto("openai") > 0


def _erro_429_do_sdk(corpo):
    """RateLimitError montado como o SDK real monta, sem rede."""
    import httpx2
    import openai

    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx2.Response(429, json=corpo, request=req)
    return openai.OpenAI(api_key="x")._make_status_error_from_response(resp)


_MSG_SEM_CREDITO = ("You have no credits remaining. Add credits to continue using the API at "
                    "https://platform.openai.com/settings/organization/billing/.")


def _rodar_com_erro_do_sdk(livro, corpo):
    import openai

    erro = _erro_429_do_sdk(corpo)
    esperas, fake = [], FakeOpenAI(responder=lambda kw: (_ for _ in ()).throw(erro))
    c = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake, dormir=esperas.append, erros_de_taxa=(openai.RateLimitError,))
    return c, esperas, fake


def test_sem_credito_com_tipo_insufficient_quota_para(livro):
    c, esperas, fake = _rodar_com_erro_do_sdk(
        livro, {"error": {"message": _MSG_SEM_CREDITO, "type": "insufficient_quota"}})
    with raises(LimiteDoPainel):
        c.conversar("S", "m", 20)
    assert esperas == [] and len(fake.chamadas) == 1


def test_sem_credito_so_pela_mensagem_para(livro):
    c, esperas, fake = _rodar_com_erro_do_sdk(livro, {"error": {"message": _MSG_SEM_CREDITO}})
    with raises(LimiteDoPainel):
        c.conversar("S", "m", 20)
    assert esperas == [] and len(fake.chamadas) == 1


def test_limite_de_taxa_comum_continua_esperando(livro):
    msg = "Rate limit reached for gpt-4o-mini in organization org-x on tokens per min. Please try again in 20ms."
    c, esperas, fake = _rodar_com_erro_do_sdk(livro, {"error": {"message": msg, "type": "requests"}})
    assert c.conversar("S", "m", 20).erro == "[erro: api]" and len(esperas) == 5
