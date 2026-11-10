"""Rodador: prioridade, retomada, teto por provedor, portão do Haiku e piloto. Nenhum teste chama API."""
import io
import math
import os
import sys

import pytest
from pytest import approx

import replica.rodar as rodar
from replica.arquivos import anexar_jsonl, ler_jsonl
from replica.clientes import ClienteAnthropic, ClienteEmbeddings, ClienteOpenAI
from replica.config import FATOR_TOKENS_HAIKU, JANELA_HAIKU, PROVEDOR
from replica.conjuntos import Contexto, Pergunta
from replica.livro_caixa import ErroDoLivroCaixa, LivroCaixa
from replica.rodar import (
    CONFIGURACOES,
    agent_config,
    dataset_config,
    executar,
    main,
    portao_haiku_h2,
    projecao,
    rodar_configuracao,
    selecionar,
)
from replica.templates_oab import SYSTEM_SEM_MEMORIA, TEMPLATE_SEM_MEMORIA, obter_template
from tests.conftest import FakeAnthropic, FakeOpenAI

POR_NOME = {c.nome: c for c in CONFIGURACOES}
ORDEM = [
    "direta-mhqa-lc-4omini", "direta-mhqa-bm25-4omini", "direta-mcc-lc-4omini", "direta-mcc-bm25-4omini",
    "direta-mhqa-emb-4omini", "oab-sem-memoria-4omini", "oab-sem-memoria-haiku", "oab-h1-lc-4omini",
    "oab-h1-bm25-4omini", "oab-h2-lc-4omini", "oab-h2-bm25-4omini", "oab-h1-emb-4omini", "oab-h1-bm25-haiku",
    "oab-h1-emb-haiku", "oab-h1-bm25-4omini-k2", "oab-h1-bm25-4omini-k5", "oab-h1-bm25-4omini-k20",
    "oab-h1corpus-bm25-4omini", "oab-h2-bm25-haiku", "oab-h1-lc-haiku", "oab-h2-lc-haiku",
    "oab-h2-lc-haiku-600k", "oab-h2-emb-4omini", "oab-h2-emb-haiku", "direta-mcc-emb-4omini",
]
ESQUEMA = {"configuracao", "fonte", "contexto_id", "pergunta_id", "ano", "consulta", "gabarito", "saida", "acerto",
           "metricas", "tokens_entrada", "tokens_cacheados", "tokens_escritos_cache", "tokens_saida", "custo_usd",
           "erro"}


@pytest.fixture(autouse=True)
def nltk_sem_rede(monkeypatch):
    """O fatiador do autor chama `nltk.download` e `nltk.sent_tokenize`; troca os dois para rodar offline."""
    monkeypatch.setattr("nltk.download", lambda *a, **k: True)
    monkeypatch.setattr("nltk.sent_tokenize", lambda t: t.split(". "))


def contextos_ruler(n_contextos=2, n_perguntas=3):
    return [
        Contexto(id=i, fonte="ruler_qa2_421K",
                 texto=f"Documento {i}. A senha do cofre {i} é {100 + i}. Fim do documento {i}.",
                 perguntas=[Pergunta(id=f"ruler_qa2_421K:{i}:{j}",
                                     texto=f"Qual é a senha do cofre {i} (pergunta {j})?",
                                     gabarito=[str(100 + i)]) for j in range(n_perguntas)])
        for i in range(n_contextos)
    ]


def questao(j):
    return f"Questão de Ética {j}\n\nA) a\nB) b\nC) c\nD) d"


def contextos_sem_memoria(n=3, ano=2024):
    return [Contexto(id=0, fonte="oab_sem_memoria", texto="",
                     perguntas=[Pergunta(id=str(1000 + j), texto=questao(j), gabarito=["B"], ano=ano)
                                for j in range(n)])]


def contextos_oab(fonte, n=2):
    return [Contexto(id=0, fonte=fonte,
                     texto="Art. 1º O advogado é indispensável. Art. 2º É dever do advogado agir com ética. Fim.",
                     perguntas=[Pergunta(id=f"{fonte}-{j}", texto=questao(j), gabarito=["B"], ano=2024)
                                for j in range(n)])]


def fabrica(livro, fake_openai=None, fake_anthropic=None, registro=None):
    """`fabricar(modelo, temperatura)` com os clientes reais sobre os SDKs falsos."""
    def fabricar(modelo, temperatura):
        if registro is not None:
            registro.append((modelo, temperatura))
        if PROVEDOR[modelo] == "openai":
            return ClienteOpenAI(modelo, livro, temperatura=temperatura, bruto=fake_openai or FakeOpenAI())
        return ClienteAnthropic(modelo, livro, bruto=fake_anthropic or FakeAnthropic())
    return fabricar


def test_configuracoes_em_ordem_de_prioridade():
    assert [c.nome for c in CONFIGURACOES[:4]] == ["direta-mhqa-lc-4omini", "direta-mhqa-bm25-4omini", "direta-mcc-lc-4omini", "direta-mcc-bm25-4omini"]


def test_tabela_de_configuracoes():
    assert [c.nome for c in CONFIGURACOES] == ORDEM
    lc = POR_NOME["direta-mhqa-lc-4omini"]
    assert (lc.etapa, lc.hipotese, lc.papel, lc.agente, lc.modelo, lc.conjunto) == (
        "direta", "h1", "primária", "contexto_longo", "gpt-4o-mini", "ruler_qa2_421K")
    assert (lc.chunk_size, lc.retrieve_num, lc.input_length_limit, lc.buffer_length) == (4096, None, 128000, 4000)
    assert (lc.generation_max_length, lc.context_max_length, lc.temperatura) == (50, 524288, 0.7)
    k5 = POR_NOME["oab-h1-bm25-4omini-k5"]
    assert (k5.papel, k5.retrieve_num, k5.chunk_size, k5.input_length_limit, k5.buffer_length) == (
        "ablação", 5, 512, 10_000_000, 200)
    assert (k5.etapa, k5.hipotese, k5.context_max_length, k5.generation_max_length) == ("oab", "h1", 2_000_000, 20)
    v = POR_NOME["oab-h2-lc-haiku-600k"]
    assert (v.papel, v.buffer_length, v.temperatura, v.hipotese) == ("variante da 21", 670_000, None, "h2")
    sm = POR_NOME["oab-sem-memoria-haiku"]
    assert (sm.hipotese, sm.agente, sm.papel, sm.generation_max_length, sm.chunk_size) == (
        "linha_de_base", "sem_memoria", "linha de base", 20, None)
    emb = POR_NOME["direta-mcc-emb-4omini"]
    assert (emb.papel, emb.hipotese, emb.chunk_size, emb.input_length_limit, emb.buffer_length,
            emb.context_max_length) == ("reprodução", "h2", 4096, 300_000, 15_000, 131072)
    assert POR_NOME["oab-h1corpus-bm25-4omini"].conjunto == "oab_h1_corpus"


def test_limites_do_haiku_cabem_na_janela():  # Review Focus 4
    for c in CONFIGURACOES:
        if c.modelo == "claude-haiku-5-5" and c.agente == "contexto_longo":
            assert (c.input_length_limit - c.buffer_length) * FATOR_TOKENS_HAIKU + c.generation_max_length <= JANELA_HAIKU


def test_agent_config_e_dataset_config():
    lc = agent_config(POR_NOME["oab-h1-lc-haiku"])
    assert lc["agent_name"] == "Long_context_agent_claude-haiku-5-5" and lc["model"] == "claude-haiku-5-5"
    assert lc["temperature"] is None and (lc["input_length_limit"], lc["buffer_length"]) == (1_000_000, 480_000)
    assert "retrieve_num" not in lc and lc["output_dir"]
    bm25 = agent_config(POR_NOME["oab-h1-bm25-4omini-k20"])
    assert bm25["agent_name"] == "Simple_rag_bm25" and bm25["retrieve_num"] == 20 and bm25["temperature"] == 0.7
    assert (bm25["input_length_limit"], bm25["buffer_length"]) == (10_000_000, 200)
    emb = agent_config(POR_NOME["direta-mhqa-emb-4omini"])
    assert emb["agent_name"] == "Embedding_rag_text_embedding_3_small" and emb["retrieve_num"] == 10
    assert (emb["input_length_limit"], emb["buffer_length"]) == (300_000, 15_000)
    assert dataset_config(POR_NOME["direta-mhqa-bm25-4omini"], "ruler_qa2_421K") == {
        "sub_dataset": "ruler_qa2_421K", "chunk_size": 512, "generation_max_length": 50,
        "context_max_length": 524288, "dataset": "Accurate_Retrieval"}
    assert dataset_config(POR_NOME["direta-mcc-lc-4omini"], "icl_nlu_8296shot_balance")["dataset"] == "Test_Time_Learning"
    assert dataset_config(POR_NOME["oab-h1-bm25-4omini"], "oab_h1_03")["dataset"] == "Accurate_Retrieval"
    assert dataset_config(POR_NOME["oab-h1corpus-bm25-4omini"], "oab_h1_corpus")["dataset"] == "Accurate_Retrieval"
    assert dataset_config(POR_NOME["oab-h2-bm25-4omini"], "oab_h2")["dataset"] == "Test_Time_Learning"
    assert dataset_config(POR_NOME["oab-sem-memoria-4omini"], "oab_sem_memoria")["dataset"] == "Test_Time_Learning"


def test_selecao_sem_nomes_inclui_so_a_variante_escolhida():
    nomes = [c.nome for c in selecionar([], "1m")]
    assert "oab-h2-lc-haiku" in nomes and "oab-h2-lc-haiku-600k" not in nomes and len(nomes) == 24
    nomes = [c.nome for c in selecionar([], "600k")]
    assert "oab-h2-lc-haiku-600k" in nomes and "oab-h2-lc-haiku" not in nomes
    # com nomes: na ordem de prioridade, não na da linha de comando
    assert [c.nome for c in selecionar(["oab-h1-lc-4omini", "direta-mhqa-lc-4omini"])] == [
        "direta-mhqa-lc-4omini", "oab-h1-lc-4omini"]
    with pytest.raises(ValueError):
        selecionar(["nao-existe"])


def test_haiku_h2_explicito_nunca_roda_outra_variante():
    assert [c.nome for c in selecionar([], None)] == [c.nome for c in selecionar([], "1m")]  # padrão: 1m
    assert [c.nome for c in selecionar(["oab-h2-lc-haiku-600k"], "600k")] == ["oab-h2-lc-haiku-600k"]
    # com nomes e sem --haiku-h2, vale o que foi nomeado (até as duas variantes)
    assert len(selecionar(["oab-h2-lc-haiku", "oab-h2-lc-haiku-600k"], None)) == 2
    with pytest.raises(ValueError, match="600k conflita com oab-h2-lc-haiku nos nomes"):
        selecionar(["oab-h2-lc-haiku"], "600k")  # a 1m nomeada contra --haiku-h2 600k
    with pytest.raises(ValueError, match="1m conflita com oab-h2-lc-haiku-600k"):
        selecionar(["oab-h2-lc-haiku-600k", "oab-h1-lc-4omini"], "1m")
    with pytest.raises(ValueError, match="600k conflita"):
        selecionar(["oab-h2-lc-haiku", "oab-h2-lc-haiku-600k"], "600k")  # as duas nomeadas: a 1m não pode rodar
    with pytest.raises(ValueError, match="não teria efeito"):
        selecionar(["oab-h1-lc-4omini"], "600k")  # sem variante nomeada


def test_retomada_nao_repete_chamadas(tmp_path):  # Review Focus 3
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("100", dict(prompt=1000, cached=800, completion=5)))
    cfg = POR_NOME["direta-mhqa-lc-4omini"]
    ctxs = contextos_ruler(n_contextos=2, n_perguntas=4)
    caminho = tmp_path / f"{cfg.nome}.jsonl"

    r1 = rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path, limite_perguntas=3)
    assert len(ler_jsonl(caminho)) == 3 and len(fake.chamadas) == 3

    r2 = rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    linhas = ler_jsonl(caminho)
    assert len(linhas) == 8 and len(fake.chamadas) == 8
    assert sorted(l["pergunta_id"] for l in linhas) == sorted(p.id for c in ctxs for p in c.perguntas)
    assert livro.gasto("openai") == approx(sum(l["custo_usd"] for l in linhas))
    assert r1["custo_usd"] + r2["custo_usd"] == approx(livro.gasto("openai"))
    assert (r2["feitas"], r2["erros"], r2["status"]) == (8, 0, "completa")

    r3 = rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    assert len(fake.chamadas) == 8 and r3["custo_usd"] == 0 and r3["feitas"] == 8


def test_contexto_completo_nao_e_memorizado_de_novo(tmp_path, monkeypatch):
    fatiados = []
    original = rodar.chunk_text_into_sentences

    def contar(texto, chunk_size):
        fatiados.append(texto)
        return original(texto, chunk_size=chunk_size)

    monkeypatch.setattr(rodar, "chunk_text_into_sentences", contar)
    livro, fake = LivroCaixa(tmp_path / "livro.json"), FakeOpenAI()
    cfg = POR_NOME["direta-mhqa-bm25-4omini"]
    ctxs = contextos_ruler(n_contextos=2, n_perguntas=2)
    rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path, limite_perguntas=2)
    assert fatiados == [ctxs[0].texto]
    rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    assert fatiados == [ctxs[0].texto, ctxs[1].texto]  # o contexto 0, completo, não é fatiado nem memorizado
    assert len(fake.chamadas) == 4


def test_perguntas_do_mesmo_contexto_mandam_o_mesmo_prefixo(tmp_path, monkeypatch):
    relogio = iter(range(10_000))  # um carimbo novo a cada memorização: um agente refeito mudaria o prefixo
    monkeypatch.setattr("time.strftime", lambda *a: f"t{next(relogio)}")
    fake = FakeOpenAI()
    cfg = POR_NOME["direta-mhqa-lc-4omini"]
    rodar_configuracao(cfg, contextos_ruler(1, 3), fabrica(LivroCaixa(tmp_path / "l.json"), fake), None, tmp_path)
    linhas = ler_jsonl(tmp_path / f"{cfg.nome}.jsonl")
    prefixos = [kw["messages"][1]["content"].removesuffix(l["consulta"]) for kw, l in zip(fake.chamadas, linhas)]
    assert len(prefixos) == 3 and len(set(prefixos)) == 1
    assert prefixos[0].startswith("Dialogue between User and Assistant t0")


def test_erro_de_api_e_refeito_na_retomada(tmp_path):  # Review Focus 2
    livro = LivroCaixa(tmp_path / "livro.json")

    def falha_na_pergunta_1(kw):
        if "(pergunta 1)" in kw["messages"][-1]["content"]:
            raise RuntimeError("500 do servidor")
        return "101", dict(prompt=1000, cached=0, completion=3)

    cfg = POR_NOME["direta-mhqa-bm25-4omini"]
    ctxs = contextos_ruler(n_contextos=1, n_perguntas=3)
    caminho = tmp_path / f"{cfg.nome}.jsonl"
    falho = FakeOpenAI(falha_na_pergunta_1)
    r1 = rodar_configuracao(cfg, ctxs, fabrica(livro, falho), None, tmp_path)
    assert len(falho.chamadas) == 3  # a rodada segue depois do erro
    assert [l["erro"] for l in ler_jsonl(caminho)] == [None, "[erro: api]", None]
    assert (r1["feitas"], r1["erros"], r1["status"]) == (2, 1, "com erros de API")

    bom = FakeOpenAI(lambda kw: ("100", dict(prompt=1000, cached=0, completion=3)))
    r2 = rodar_configuracao(cfg, ctxs, fabrica(livro, bom), None, tmp_path)
    assert len(bom.chamadas) == 1 and "(pergunta 1)" in bom.chamadas[0]["messages"][-1]["content"]
    linhas = ler_jsonl(caminho)
    assert sorted(l["pergunta_id"] for l in linhas) == sorted(p.id for p in ctxs[0].perguntas)  # 1 linha por pergunta
    assert all(l["erro"] is None for l in linhas)
    assert (r2["feitas"], r2["erros"], r2["status"]) == (3, 0, "completa")


def test_linha_final_cortada_nao_derruba_a_retomada(tmp_path, capsys):
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    cfg = POR_NOME["oab-sem-memoria-4omini"]
    ctxs = contextos_sem_memoria(3)
    caminho = tmp_path / f"{cfg.nome}.jsonl"
    rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path, limite_perguntas=1)
    with open(caminho, "ab") as f:  # queda no meio da escrita, cortando um caractere de dois bytes
        f.write(b'{"configuracao": "oab-sem-memoria-4omini", "pergunta_id": "1001", "consulta": "Quest\xc3')
    r = rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    assert "cortada" in capsys.readouterr().err
    assert len(fake.chamadas) == 3 and r["feitas"] == 3
    assert sorted(l["pergunta_id"] for l in ler_jsonl(caminho)) == ["1000", "1001", "1002"]


def test_linha_sem_erro_nunca_e_trocada_por_uma_com_erro(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    cfg = POR_NOME["oab-sem-memoria-4omini"]
    ctxs = contextos_sem_memoria(2)
    caminho = tmp_path / f"{cfg.nome}.jsonl"
    rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    boa = ler_jsonl(caminho)[0]
    anexar_jsonl(caminho, {**boa, "saida": "", "acerto": False, "metricas": {}, "erro": "[erro: api]"})
    r = rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    assert len(fake.chamadas) == 2 and (r["feitas"], r["erros"]) == (2, 0)  # a paga não é refeita
    linhas = ler_jsonl(caminho)
    assert len(linhas) == 2 and linhas[0] == boa


def falha_api_na_2_e_vazia_na_1(kw):
    texto = kw["messages"][-1]["content"]
    if "Ética 2" in texto:
        raise RuntimeError("500 do servidor")
    if "Ética 1" in texto:
        return "", dict(prompt=100, cached=0, completion=0)  # resposta vazia, mas paga
    return "B", dict(prompt=100, cached=0, completion=1)


def test_recusa_e_resposta_vazia_sao_finais_e_so_erro_de_api_e_refeito(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")
    cfg = POR_NOME["oab-sem-memoria-4omini"]
    ctxs = contextos_sem_memoria(3)
    caminho = tmp_path / f"{cfg.nome}.jsonl"
    r1 = rodar_configuracao(cfg, ctxs, fabrica(livro, FakeOpenAI(falha_api_na_2_e_vazia_na_1)), None, tmp_path)
    assert (r1["feitas"], r1["erros"], r1["recusas"], r1["status"]) == (1, 1, 1, "com erros de API")

    bom = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    r2 = rodar_configuracao(cfg, ctxs, fabrica(livro, bom), None, tmp_path)
    assert len(bom.chamadas) == 1 and "Ética 2" in bom.chamadas[0]["messages"][-1]["content"]
    assert (r2["feitas"], r2["erros"], r2["recusas"], r2["status"]) == (2, 0, 1, "completa")
    vazia = next(l for l in ler_jsonl(caminho) if l["pergunta_id"] == "1001")
    assert vazia["erro"] == "[erro: resposta vazia]" and vazia["acerto"] is False and vazia["custo_usd"] > 0

    recusa = FakeAnthropic(stop_reason="refusal")
    haiku = POR_NOME["oab-sem-memoria-haiku"]
    for _ in range(2):
        r = rodar_configuracao(haiku, contextos_sem_memoria(1), fabrica(livro, fake_anthropic=recusa), None, tmp_path)
    assert len(recusa.chamadas) == 1 and (r["feitas"], r["erros"], r["recusas"]) == (0, 0, 1)


def test_falha_na_avaliacao_nao_perde_a_resposta_paga(tmp_path, monkeypatch):
    def avaliar_quebrado(fonte, saida, gabarito):
        raise KeyError("bug na avaliação")

    monkeypatch.setattr(rodar, "avaliar", avaliar_quebrado)
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    cfg = POR_NOME["oab-sem-memoria-4omini"]
    ctxs = contextos_sem_memoria(2)
    r = rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    linhas = ler_jsonl(tmp_path / f"{cfg.nome}.jsonl")
    assert len(linhas) == 2 and r["feitas"] == 2
    for l in linhas:
        assert l["saida"] == "B" and l["erro"] is None and l["acerto"] is False
        assert "KeyError" in l["metricas"]["erro_avaliacao"]
    rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    assert len(fake.chamadas) == 2  # a resposta paga não é refeita


def test_falha_numa_configuracao_nao_para_as_outras(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    (tmp_path / "oab-sem-memoria-haiku.jsonl").write_text(
        '{"pergunta_id": "1000"}\nlixo no meio\n{"pergunta_id": "1001"}\n', encoding="utf-8")

    def carregar(cfg):
        if cfg.conjunto == "oab_h1":
            raise FileNotFoundError("dados/oab/h1.json")
        return contextos_sem_memoria(2)

    cfgs = [POR_NOME["oab-sem-memoria-haiku"], POR_NOME["oab-h1-lc-4omini"], POR_NOME["oab-sem-memoria-4omini"]]
    rs = executar(cfgs, tmp_path, carregar, fabrica(livro, fake), None)
    assert rs[0]["status"].startswith("falhou:") and "oab-sem-memoria-haiku.jsonl" in rs[0]["status"]
    assert rs[1]["status"].startswith("falhou:") and "h1.json" in rs[1]["status"]
    assert rs[2]["status"] == "completa" and len(fake.chamadas) == 2


def test_falha_ao_compactar_no_fim_nao_perde_o_resultado(tmp_path, monkeypatch, capsys):
    livro = LivroCaixa(tmp_path / "livro.json")
    cfg = POR_NOME["oab-sem-memoria-4omini"]
    ctxs = contextos_sem_memoria(3)
    caminho = tmp_path / f"{cfg.nome}.jsonl"
    rodar_configuracao(cfg, ctxs, fabrica(livro, FakeOpenAI(falha_api_na_2_e_vazia_na_1)), None, tmp_path)
    original = rodar._regravar

    def regravar_bloqueado(caminho, linhas):
        raise PermissionError("arquivo aberto em outro programa")

    monkeypatch.setattr(rodar, "_regravar", regravar_bloqueado)
    bom = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    r = rodar_configuracao(cfg, ctxs, fabrica(livro, bom), None, tmp_path)
    assert (r["feitas"], r["erros"], r["recusas"], r["status"]) == (2, 0, 1, "completa")
    assert "compactar" in capsys.readouterr().err and len(ler_jsonl(caminho)) == 4  # a de erro velha ainda está lá
    monkeypatch.setattr(rodar, "_regravar", original)
    rodar_configuracao(cfg, ctxs, fabrica(livro, bom), None, tmp_path)
    assert len(ler_jsonl(caminho)) == 3 and len(bom.chamadas) == 1  # a próxima execução compacta, sem chamar nada


def test_sem_memoria_chama_o_cliente_direto(tmp_path):
    livro, registro = LivroCaixa(tmp_path / "livro.json"), []
    fake_openai, fake_anthropic = FakeOpenAI(), FakeAnthropic()
    for nome in ("oab-sem-memoria-4omini", "oab-sem-memoria-haiku"):
        rodar_configuracao(POR_NOME[nome], contextos_sem_memoria(1), fabrica(livro, fake_openai, fake_anthropic, registro),
                           None, tmp_path)
    assert registro == [("gpt-4o-mini", 0.7), ("claude-haiku-5-5", None)]
    kw = fake_openai.chamadas[0]
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 20
    assert kw["messages"] == [{"role": "system", "content": SYSTEM_SEM_MEMORIA},
                              {"role": "user", "content": TEMPLATE_SEM_MEMORIA.format(question=questao(0))}]
    kw = fake_anthropic.chamadas[0]
    assert "temperature" not in kw and kw["system"] == SYSTEM_SEM_MEMORIA


def test_teto_de_um_provedor_nao_para_o_outro(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")
    livro.registrar("anthropic", 29.5)
    fake_openai, fake_anthropic = FakeOpenAI(), FakeAnthropic()
    carregados = []

    def carregar(cfg):
        carregados.append(cfg.nome)
        return contextos_sem_memoria(3) if cfg.agente == "sem_memoria" else contextos_oab("oab_h1_00")

    cfgs = [POR_NOME["oab-sem-memoria-haiku"], POR_NOME["oab-sem-memoria-4omini"], POR_NOME["oab-h1-bm25-haiku"]]
    rs = executar(cfgs, tmp_path, carregar, fabrica(livro, fake_openai, fake_anthropic),
                  ClienteEmbeddings(livro, bruto=fake_openai))
    assert [r["status"] for r in rs] == ["parada: teto", "completa", "pulada: teto (anthropic)"]
    assert rs[0]["provedor_teto"] == "anthropic" and rs[1]["feitas"] == 3
    assert fake_anthropic.chamadas == [] and len(fake_openai.chamadas) == 3
    assert "oab-h1-bm25-haiku" not in carregados  # pulada antes de carregar os dados


def test_teto_do_embedding_conta_como_openai(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")
    livro.registrar("openai", 19.5)
    fake_openai, fake_anthropic = FakeOpenAI(), FakeAnthropic()
    cfgs = [POR_NOME["oab-h1-emb-haiku"], POR_NOME["oab-h2-bm25-haiku"], POR_NOME["oab-h2-emb-4omini"]]
    rs = executar(cfgs, tmp_path, lambda cfg: contextos_oab("oab_h1_00" if "h1" in cfg.nome else "oab_h2"),
                  fabrica(livro, fake_openai, fake_anthropic), ClienteEmbeddings(livro, bruto=fake_openai))
    assert (rs[0]["status"], rs[0]["provedor_teto"]) == ("parada: teto", "openai")
    assert rs[1]["status"] == "completa" and len(fake_anthropic.chamadas) == 2  # Haiku com BM25 não usa a OpenAI
    assert rs[2]["status"] == "pulada: teto (openai)"
    assert fake_openai.chamadas_embeddings == [] and fake_openai.chamadas == []


def gravar_linha_de_base(pasta, acertos, prefixo="", anos=(2024, 2025)):
    """Linhas do `oab-sem-memoria-haiku`: ids `{prefixo}{i}` de 2024-2025 com `acertos`, e cinco de 2023 erradas."""
    caminho = pasta / "oab-sem-memoria-haiku.jsonl"
    for i, a in enumerate(acertos):
        anexar_jsonl(caminho, {"pergunta_id": f"{prefixo}{i}", "ano": anos[i % 2], "acerto": a, "erro": None})
    for i in range(5):
        anexar_jsonl(caminho, {"pergunta_id": f"antiga-{i}", "ano": 2023, "acerto": False, "erro": None})


TESTE_H2 = {str(i) for i in range(10)}


def test_portao_haiku_h2(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    gravar_linha_de_base(a, [True] * 9 + [False])
    gravar_linha_de_base(b, [True] * 8 + [False] * 2)
    assert portao_haiku_h2(a, TESTE_H2) == (False, approx(0.9))
    assert portao_haiku_h2(b, TESTE_H2) == (True, approx(0.8))


def test_portao_exige_resposta_para_toda_questao_de_teste(tmp_path, capsys):
    libera, acuracia = portao_haiku_h2(tmp_path, TESTE_H2)  # sem arquivo
    assert not libera and math.isnan(acuracia)
    gravar_linha_de_base(tmp_path, [True] * 2 + [False] * 7)  # 9 de 10: falta a "9"
    libera, acuracia = portao_haiku_h2(tmp_path, TESTE_H2)
    assert not libera and math.isnan(acuracia)
    assert "faltam 1 de 10" in capsys.readouterr().out
    caminho = tmp_path / "oab-sem-memoria-haiku.jsonl"
    anexar_jsonl(caminho, {"pergunta_id": "9", "ano": 2025, "acerto": False, "erro": "[erro: api]"})
    assert portao_haiku_h2(tmp_path, TESTE_H2)[0] is False  # erro de API não é resposta
    anexar_jsonl(caminho, {"pergunta_id": "9", "ano": 2025, "acerto": False, "erro": "[erro: recusa]"})
    assert portao_haiku_h2(tmp_path, TESTE_H2) == (True, approx(0.2))  # recusa é resposta final, errada


def test_portao_no_executar_bloqueia_sem_linha_de_base_e_libera_abaixo_de_90(tmp_path, capsys):
    livro, fake_anthropic = LivroCaixa(tmp_path / "livro.json"), FakeAnthropic()
    cfg = POR_NOME["oab-h2-lc-haiku"]

    def h2(c):
        return contextos_oab("oab_h2", n=10)  # ids oab_h2-0 a oab_h2-9

    rs = executar([cfg], tmp_path, h2, fabrica(livro, fake_anthropic=fake_anthropic), None)
    assert rs[0]["status"] == "pulada: portão" and fake_anthropic.chamadas == []
    assert "portão" in capsys.readouterr().out

    gravar_linha_de_base(tmp_path, [True] * 4 + [False], prefixo="oab_h2-")  # 80%, mas só 5 das 10
    rs = executar([cfg], tmp_path, h2, fabrica(livro, fake_anthropic=fake_anthropic), None)
    assert rs[0]["status"] == "pulada: portão" and fake_anthropic.chamadas == []  # linha de base parcial: fechado

    (tmp_path / "oab-sem-memoria-haiku.jsonl").unlink()
    gravar_linha_de_base(tmp_path, [True] * 8 + [False] + [True], prefixo="oab_h2-")
    rs = executar([cfg], tmp_path, h2, fabrica(livro, fake_anthropic=fake_anthropic), None)
    assert rs[0]["status"] == "pulada: portão" and fake_anthropic.chamadas == []  # 90%: fechado

    (tmp_path / "oab-sem-memoria-haiku.jsonl").unlink()
    gravar_linha_de_base(tmp_path, [True] * 8 + [False] * 2, prefixo="oab_h2-")
    rs = executar([cfg], tmp_path, h2, fabrica(livro, fake_anthropic=fake_anthropic), None)
    assert rs[0]["status"] == "completa" and len(fake_anthropic.chamadas) == 10


def test_projecao_do_piloto():
    linhas = [{"custo_usd": 1.0}, {"custo_usd": 0.1}, {"custo_usd": 0.1}]
    assert projecao(linhas, n_contextos=2, n_perguntas=100) == approx(11.8)


def test_piloto_limita_as_perguntas_e_grava_a_projecao(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=1_000_000, cached=0, completion=0)))  # US$ 0,15 por chamada

    def carregar(cfg):
        return contextos_sem_memoria(5) if cfg.agente == "sem_memoria" else contextos_ruler(2, 4)

    rodada = tmp_path / "piloto"
    cfgs = [POR_NOME["direta-mhqa-lc-4omini"], POR_NOME["oab-sem-memoria-4omini"]]
    rs = executar(cfgs, rodada, carregar, fabrica(livro, fake), None, piloto=2)
    assert [r["feitas"] for r in rs] == [2, 5]  # a sem memória ignora o piloto e roda completa
    linhas_lc = ler_jsonl(rodada / "direta-mhqa-lc-4omini.jsonl")
    assert [l["pergunta_id"] for l in linhas_lc] == ["ruler_qa2_421K:0:0", "ruler_qa2_421K:0:1"]
    texto = (rodada / "projecao.md").read_text(encoding="utf-8")
    projetado = projecao(linhas_lc, n_contextos=2, n_perguntas=8)
    real_sm = sum(l["custo_usd"] for l in ler_jsonl(rodada / "oab-sem-memoria-4omini.jsonl"))
    assert projetado == approx(1.2) and real_sm == approx(0.75)
    assert "direta-mhqa-lc-4omini" in texto and "1.2000" in texto
    assert "oab-sem-memoria-4omini" in texto and "0.7500" in texto
    assert "1.9500" in texto  # total da OpenAI
    assert "do tamanho do primeiro" in texto  # aviso do viés da projeção


def test_jsonl_do_rodador_preserva_acentos(tmp_path):  # Review Focus 5
    livro = LivroCaixa(tmp_path / "livro.json")
    fake = FakeOpenAI(lambda kw: ("Gabarito: B, é a alternativa correta", dict(prompt=10, cached=0, completion=3)))
    cfg = POR_NOME["oab-h1-bm25-4omini"]
    ctxs = contextos_oab("oab_h1_00", n=1)
    rodar_configuracao(cfg, ctxs, fabrica(livro, fake), None, tmp_path)
    caminho = tmp_path / f"{cfg.nome}.jsonl"
    bruto = caminho.read_bytes()
    assert "Questão de Ética".encode("utf-8") in bruto and b"\\u00" not in bruto
    linha = ler_jsonl(caminho)[0]
    assert set(linha) == ESQUEMA
    assert linha["consulta"] == obter_template("oab_h1_00", "query", "Simple_rag_bm25").format(question=questao(0))
    assert "Questão de Ética" in linha["consulta"] and linha["saida"] == "Gabarito: B, é a alternativa correta"
    assert linha["acerto"] and linha["metricas"] == {"letra": "B", "letra_invalida": False}


def test_main_roda_pelo_nome_e_imprime_o_resumo(tmp_path, monkeypatch, capsys):
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=100, cached=0, completion=1)))
    monkeypatch.setattr(rodar, "LIVRO_CAIXA", tmp_path / "livro.json")
    monkeypatch.setattr(rodar, "carregar_conjunto", lambda cfg: contextos_sem_memoria(3))
    monkeypatch.setattr(rodar, "fabricar_cliente",
                        lambda modelo, livro, temperatura: ClienteOpenAI(modelo, livro, temperatura, bruto=fake))
    assert main(["oab-sem-memoria-4omini", "--rodada", str(tmp_path / "r")]) == 0
    saida = capsys.readouterr().out
    assert ("oab-sem-memoria-4omini: feitas=3, erros_api=0, recusas_vazias=0, custo=US$ " in saida
            and "status=completa" in saida)
    assert "OpenAI US$" in saida and "Anthropic US$" in saida  # gasto do livro-caixa no fim
    assert len(ler_jsonl(tmp_path / "r" / "oab-sem-memoria-4omini.jsonl")) == 3
    assert main(["nao-existe", "--rodada", str(tmp_path / "r")]) == 2
    assert main(["oab-h2-lc-haiku", "--haiku-h2", "600k", "--rodada", str(tmp_path / "r")]) == 2


def test_main_imprime_o_gasto_mesmo_quando_algo_quebra(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rodar, "LIVRO_CAIXA", tmp_path / "livro.json")

    def projecao_quebrada(*a, **k):
        raise OSError("disco cheio")

    monkeypatch.setattr(rodar, "carregar_conjunto", lambda cfg: contextos_sem_memoria(1))
    monkeypatch.setattr(rodar, "fabricar_cliente", lambda modelo, livro, temperatura: ClienteOpenAI(
        modelo, livro, temperatura, bruto=FakeOpenAI()))
    monkeypatch.setattr(rodar, "gravar_projecao", projecao_quebrada)
    with pytest.raises(OSError):
        main(["oab-sem-memoria-4omini", "--piloto", "1", "--rodada", str(tmp_path / "r")])
    assert "gasto no livro-caixa" in capsys.readouterr().out
    assert not rodar.TRAVA_RODADOR.exists()  # a trava sai mesmo quando algo quebra


# --- main: --rodada obrigatório e nomes ou --todas ---

@pytest.fixture
def executar_falso(monkeypatch):
    """Troca `executar` por um que só anota os nomes das configurações (e o que estava na trava)."""
    chamadas = []

    def falso(cfgs, *a, **k):
        trava = rodar.TRAVA_RODADOR
        chamadas.append({"nomes": [c.nome for c in cfgs],
                         "trava": trava.read_text(encoding="utf-8") if trava.exists() else None})
        return []

    monkeypatch.setattr(rodar, "executar", falso)
    return chamadas


def test_main_exige_rodada(executar_falso, capsys):
    with pytest.raises(SystemExit) as e:
        main(["oab-sem-memoria-4omini"])
    assert e.value.code == 2 and "--rodada" in capsys.readouterr().err and executar_falso == []


def test_main_sem_nomes_exige_todas(tmp_path, executar_falso, capsys):
    rodada = ["--rodada", str(tmp_path / "r")]
    assert main(rodada) == 2
    assert "--todas" in capsys.readouterr().err and executar_falso == []
    assert main(["oab-sem-memoria-4omini", "--todas", *rodada]) == 2  # os dois juntos é ambíguo
    assert executar_falso == []
    assert main(["--todas", *rodada]) == 0
    nomes = executar_falso[0]["nomes"]
    assert len(nomes) == 24 and "oab-h2-lc-haiku" in nomes and "oab-h2-lc-haiku-600k" not in nomes
    assert main(["--todas", "--haiku-h2", "600k", *rodada]) == 0
    nomes = executar_falso[1]["nomes"]
    assert "oab-h2-lc-haiku-600k" in nomes and "oab-h2-lc-haiku" not in nomes


# --- main: trava exclusiva ---

def test_main_recusa_com_a_trava_presente_e_nao_a_apaga(tmp_path, executar_falso, capsys):
    trava = rodar.TRAVA_RODADOR
    trava.parent.mkdir(parents=True, exist_ok=True)
    trava.write_text("pid 123", encoding="utf-8")
    assert main(["oab-sem-memoria-4omini", "--rodada", str(tmp_path / "r")]) == 2
    err = capsys.readouterr().err
    assert "apague" in err and str(trava) in err and executar_falso == []
    assert trava.read_text(encoding="utf-8") == "pid 123"


def test_main_pega_e_solta_a_trava(tmp_path, executar_falso):
    assert main(["oab-sem-memoria-4omini", "--rodada", str(tmp_path / "r")]) == 0
    assert str(os.getpid()) in executar_falso[0]["trava"]  # durante a rodada, a trava tem o pid
    assert not rodar.TRAVA_RODADOR.exists()


# --- main: livro-caixa conferido contra as linhas gravadas ---

def _linha_paga(nome, pid, custo):
    return {"configuracao": nome, "pergunta_id": pid, "custo_usd": custo, "erro": None}


def test_soma_das_linhas_conta_copia_uma_vez_e_ignora_desconhecidas(tmp_path):
    for pasta in ("piloto", "oab"):  # a linha de base reaproveitada: o mesmo arquivo em duas pastas
        anexar_jsonl(tmp_path / pasta / "oab-sem-memoria-haiku.jsonl", _linha_paga("oab-sem-memoria-haiku", "1", 0.25))
    anexar_jsonl(tmp_path / "oab" / "outra.jsonl", _linha_paga("nao-existe", "1", 9.0))
    for pid in ("1", "2"):
        anexar_jsonl(tmp_path / "oab" / "oab-h1-emb-haiku.jsonl", _linha_paga("oab-h1-emb-haiku", pid, 0.5))
        anexar_jsonl(tmp_path / "oab" / "oab-h1-lc-4omini.jsonl", _linha_paga("oab-h1-lc-4omini", pid, 0.125))
    assert rodar.soma_das_linhas(tmp_path) == approx({"openai": 0.25, "anthropic": 1.25})


def test_main_recusa_livro_caixa_abaixo_das_linhas(tmp_path, executar_falso, resultados_isolados, capsys):
    anexar_jsonl(resultados_isolados / "a" / "oab-sem-memoria-4omini.jsonl",
                 _linha_paga("oab-sem-memoria-4omini", "1", 0.30))
    anexar_jsonl(resultados_isolados / "a" / "oab-sem-memoria-haiku.jsonl",
                 _linha_paga("oab-sem-memoria-haiku", "1", 0.20))
    livro = LivroCaixa(rodar.LIVRO_CAIXA)
    livro.registrar("anthropic", 0.20)
    livro.registrar("openai", 0.10)  # revertido (git stash, checkout): menos que os 0,30 das linhas
    assert main(["oab-sem-memoria-4omini", "--rodada", str(tmp_path / "r")]) == 2
    err = capsys.readouterr().err
    assert "openai" in err and "0.3000" in err and "0.1000" in err and "anthropic" not in err
    assert executar_falso == [] and not rodar.TRAVA_RODADOR.exists()


def test_conferencia_do_livro_caixa_tem_tolerancia(tmp_path):
    anexar_jsonl(tmp_path / "a" / "oab-sem-memoria-4omini.jsonl", _linha_paga("oab-sem-memoria-4omini", "1", 0.3))
    livro = LivroCaixa(tmp_path / "livro.json")
    livro.registrar("openai", 0.3 - 5e-7)  # arredondamento: passa
    assert rodar.conferir_livro_caixa(livro, tmp_path) == []
    livro.registrar("openai", -2e-6)
    problemas = rodar.conferir_livro_caixa(livro, tmp_path)
    assert len(problemas) == 1 and "openai" in problemas[0]


def test_main_retoma_com_o_livro_caixa_igual_as_linhas(monkeypatch, resultados_isolados):
    fake = FakeOpenAI(lambda kw: ("B", dict(prompt=123, cached=7, completion=1)))
    monkeypatch.setattr(rodar, "carregar_conjunto", lambda cfg: contextos_sem_memoria(7))
    monkeypatch.setattr(rodar, "fabricar_cliente",
                        lambda modelo, livro, temperatura: ClienteOpenAI(modelo, livro, temperatura, bruto=fake))
    rodada = ["--rodada", str(resultados_isolados / "r")]  # a rodada dentro de resultados/, como na vida real
    assert main(["oab-sem-memoria-4omini", *rodada]) == 0
    assert main(["oab-sem-memoria-4omini", *rodada]) == 0  # o livro-caixa bate com as linhas: retoma
    assert len(fake.chamadas) == 7


# --- erro fatal do livro-caixa ---

class _LivroQueNaoGrava(LivroCaixa):
    def registrar(self, provedor, custo):
        raise ErroDoLivroCaixa(f"não gravei US$ {custo:.4f} da {provedor}")


def test_falha_do_livro_caixa_para_a_rodada_inteira(tmp_path):
    livro = _LivroQueNaoGrava(tmp_path / "livro.json")
    fake_openai, fake_anthropic = FakeOpenAI(), FakeAnthropic()
    cfgs = [POR_NOME["oab-sem-memoria-4omini"], POR_NOME["oab-sem-memoria-haiku"]]
    with pytest.raises(ErroDoLivroCaixa):
        executar(cfgs, tmp_path, lambda cfg: contextos_sem_memoria(3), fabrica(livro, fake_openai, fake_anthropic),
                 None)
    assert len(fake_openai.chamadas) == 1 and fake_anthropic.chamadas == []
    assert ler_jsonl(tmp_path / "oab-sem-memoria-4omini.jsonl") == []  # nada gravado como erro de API


def test_main_para_com_erro_do_livro_caixa(tmp_path, monkeypatch, capsys):
    def executar_que_quebra(*a, **k):
        raise ErroDoLivroCaixa("não gravei US$ 0.0100 da openai")

    monkeypatch.setattr(rodar, "executar", executar_que_quebra)
    assert main(["oab-sem-memoria-4omini", "--rodada", str(tmp_path / "r")]) == 1
    captura = capsys.readouterr()
    assert "não gravei" in captura.err and "gasto no livro-caixa" in captura.out
    assert not rodar.TRAVA_RODADOR.exists()


# --- limite do painel e teto só pelo provedor do leitor ---

def _carregar_misto(cfg):
    return contextos_sem_memoria(3) if cfg.agente == "sem_memoria" else contextos_oab("oab_h1_00")


def test_limite_do_painel_para_o_provedor_como_o_teto(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json")

    def cota(kw):
        raise RuntimeError("Error code: 429 - {'error': {'code': 'insufficient_quota'}}")

    fake_openai, fake_anthropic = FakeOpenAI(cota), FakeAnthropic()
    cfgs = [POR_NOME["oab-sem-memoria-4omini"], POR_NOME["oab-sem-memoria-haiku"], POR_NOME["oab-h1-bm25-4omini"]]
    rs = executar(cfgs, tmp_path, _carregar_misto, fabrica(livro, fake_openai, fake_anthropic), None)
    assert [r["status"] for r in rs] == ["parada: limite do painel", "completa", "pulada: limite do painel (openai)"]
    assert rs[0]["provedor_teto"] == "openai" and len(fake_openai.chamadas) == 1 and len(fake_anthropic.chamadas) == 3
    assert ler_jsonl(tmp_path / "oab-sem-memoria-4omini.jsonl") == []  # nada de [erro: api] para refazer


def test_teto_da_openai_nao_pula_o_haiku_com_embedding(tmp_path):
    livro = LivroCaixa(tmp_path / "livro.json", tetos={"openai": 1.0, "anthropic": 29.5},
                       reserva={"openai": 0.0, "anthropic": 0.0})
    livro.registrar("openai", 1.0 - 1e-5)  # cabem embeddings pequenos, não cabe uma chamada do GPT-4o-mini
    fake_openai, fake_anthropic = FakeOpenAI(), FakeAnthropic()
    cfgs = [POR_NOME["oab-sem-memoria-4omini"], POR_NOME["oab-h1-emb-haiku"], POR_NOME["oab-h1-bm25-4omini"]]
    rs = executar(cfgs, tmp_path, _carregar_misto, fabrica(livro, fake_openai, fake_anthropic),
                  ClienteEmbeddings(livro, bruto=fake_openai))
    assert [r["status"] for r in rs] == ["parada: teto", "completa", "pulada: teto (openai)"]
    assert fake_openai.chamadas == [] and fake_openai.chamadas_embeddings and len(fake_anthropic.chamadas) == 2


# --- saída que não derruba a rodada ---

def test_main_nao_morre_com_caractere_fora_do_cp1252(tmp_path, monkeypatch):
    saida = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    erros = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", saida)
    monkeypatch.setattr(sys, "stderr", erros)
    assert main(["nao-existe-✓", "--rodada", str(tmp_path / "r")]) == 2
    erros.flush()
    assert b"nao-existe-\\u2713" in erros.buffer.getvalue()
