"""Agentes do autor (agent.py@538026) sobre os clientes com teto. Nenhum teste chama API."""
from dataclasses import dataclass

import numpy as np
import pytest
import tiktoken

from replica.autor.agente import AgentWrapper
from replica.autor.templates import SYSTEM_MESSAGE, get_template
from replica.clientes import Resposta

T = "2026-01-01 00:00:00"
SUB = "ruler_qa2_421K"

CFG_LC = {"agent_name": "Long_context_agent_gpt-4o-mini", "model": "gpt-4o-mini", "temperature": 0.7,
          "input_length_limit": 128000, "buffer_length": 4000, "output_dir": "x"}
CFG_BM25 = {"agent_name": "Simple_rag_bm25", "model": "gpt-4o-mini", "temperature": 0.7,
            "input_length_limit": 10000000, "buffer_length": 200, "output_dir": "x", "retrieve_num": 2}
CFG_EMB = {"agent_name": "Embedding_rag_text_embedding_3_small", "model": "gpt-4o-mini", "temperature": 0.7,
           "input_length_limit": 300000, "buffer_length": 15000, "output_dir": "x", "retrieve_num": 2}
CFG_DADOS = {"sub_dataset": SUB, "context_max_length": 524288, "dataset": "Accurate_Retrieval",
             "generation_max_length": 50, "chunk_size": 4096}

# Português jurídico com acentos e caracteres de vários bytes (emoji, CJK, plano astral).
TEXTO_PT = ("Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza; "
            "é garantida a inviolabilidade do direito à vida, à liberdade e à propriedade. "
            "Ação rescisória, exceção de pré-executividade, ônus da prova, réu revel. "
            "⚖️ 📜 § 1º — “aspas curvas”; 中文 𠜎 ꙮ ẞ 🜁 𓀀.")

TOKENIZADOR = tiktoken.encoding_for_model("gpt-4o-mini")


def n_tokens(texto):
    return len(TOKENIZADOR.encode(texto, disallowed_special=()))


@dataclass
class Chamada:
    system: str | None
    mensagem: str
    max_tokens: int
    prefixo_cacheavel: str | None


class ClienteFalso:
    """Imita `ClienteOpenAI`/`ClienteAnthropic.conversar` e grava cada chamada."""

    def __init__(self):
        self.chamadas = []

    def conversar(self, system, mensagem, max_tokens, prefixo_cacheavel=None):
        self.chamadas.append(Chamada(system, mensagem, max_tokens, prefixo_cacheavel))
        return Resposta(texto="ok", tokens_entrada=123, tokens_cacheados=100, tokens_escritos_cache=7,
                        tokens_saida=2, custo_usd=0.001)


class EmbeddingsFalso:
    """Imita `ClienteEmbeddings.embutir`: o vetor do texto é o da primeira chave contida nele."""

    def __init__(self, vetores):
        self.vetores = vetores
        self.chamadas = []

    def embutir(self, textos):
        self.chamadas.append(list(textos))
        return np.asarray([next(v for k, v in self.vetores.items() if k in t) for t in textos], dtype=np.float32)


@pytest.fixture(autouse=True)
def relogio_fixo(monkeypatch):
    monkeypatch.setattr("time.strftime", lambda *a: T)


def memorizar(ag, trechos):
    for t in trechos:
        ag.send_message(t, memorizing=True)


def formatado(trecho, agent_name):
    return get_template(SUB, "memorize", agent_name).format(context=trecho, time_stamp=T)


def pergunta_rag(agent_name, questao):
    return get_template(SUB, "query", agent_name).format(question=questao)


# --- contexto longo ---------------------------------------------------------------------------

def test_contexto_longo_monta_a_mensagem_do_autor():
    cliente = ClienteFalso()
    ag = AgentWrapper(CFG_LC, CFG_DADOS, cliente)
    tpl = get_template(SUB, "memorize", "Long_context_agent_gpt-4o-mini")
    assert ag.send_message("c1", memorizing=True) == "Memorized"
    ag.send_message("c2", memorizing=True)
    ag.send_message("Q?")
    esperado = (tpl.format(context="c1", time_stamp=T) + "\n" + tpl.format(context="c2", time_stamp=T)).strip() + "\n"
    assert cliente.chamadas[0].prefixo_cacheavel == esperado
    assert cliente.chamadas[0].mensagem == "Q?"
    assert cliente.chamadas[0].system == SYSTEM_MESSAGE
    assert cliente.chamadas[0].max_tokens == 50


def test_contexto_longo_descarta_o_inicio():
    cfg = dict(CFG_LC, input_length_limit=150, buffer_length=0)  # limite = 150 - 0 - 50 = 100
    cliente = ClienteFalso()
    ag = AgentWrapper(cfg, CFG_DADOS, cliente)
    assert ag.input_length_limit == 100
    trechos = [f"palavra{i} " + "texto " * 10 for i in range(30)]
    memorizar(ag, trechos)
    assert n_tokens(ag.context) > 100
    ag.send_message("Q?")
    assert n_tokens(ag.context) <= 100
    assert ag.context.endswith(formatado(trechos[-1], cfg["agent_name"])[-60:])
    assert "palavra0 " not in ag.context
    assert cliente.chamadas[0].prefixo_cacheavel == ag.context + "\n"


@pytest.mark.parametrize("caso", ["sem_truncamento", "com_truncamento"])
def test_prefixo_identico_entre_perguntas(caso):  # Review Focus 1
    """Duas perguntas seguidas mandam o mesmo prefixo, byte a byte.

    Com truncamento, o corte do autor (decode dos últimos N tokens) roda a cada pergunta. Se decode e
    nova codificação não fossem idempotentes, a segunda pergunta truncaria de novo e o prefixo mudaria
    (cache perdido). O caso varre limites em que o corte cai no meio de um caractere de vários bytes e
    limites em que não cai.
    """
    trechos = [f"Trecho {i}: {TEXTO_PT}" for i in range(4)]
    limites = [None] if caso == "sem_truncamento" else range(20, 200)
    cortes_no_meio_de_caractere = 0
    for limite in limites:
        cfg = CFG_LC if limite is None else dict(CFG_LC, input_length_limit=limite + 50, buffer_length=0)
        cliente = ClienteFalso()
        ag = AgentWrapper(cfg, CFG_DADOS, cliente)
        memorizar(ag, trechos)
        memorizado = ag.context
        assert (n_tokens(memorizado) > limite) if limite else (n_tokens(memorizado) < ag.input_length_limit)
        ag.send_message("Q1?")
        ag.send_message("Q2?")
        p1, p2 = cliente.chamadas[0].prefixo_cacheavel, cliente.chamadas[1].prefixo_cacheavel
        assert p1.encode("utf-8") == p2.encode("utf-8"), f"limite {limite}"
        assert [c.mensagem for c in cliente.chamadas] == ["Q1?", "Q2?"]
        if limite is None:
            assert p1 == memorizado + "\n"
        else:
            assert n_tokens(p1[:-1]) <= limite
            cortes_no_meio_de_caractere += "�" in p1
    if caso == "com_truncamento":
        assert cortes_no_meio_de_caractere > 0  # o caso difícil foi exercitado


def test_haiku_recebe_o_system_e_usa_o_tokenizador_do_autor():  # adaptação 2
    cfg = dict(CFG_LC, agent_name="Long_context_agent_claude-haiku-5-5", model="claude-haiku-5-5")
    cliente = ClienteFalso()
    ag = AgentWrapper(cfg, CFG_DADOS, cliente)
    assert ag.tokenizer.name == "o200k_base"
    memorizar(ag, ["c1"])
    ag.send_message("Q?")
    assert cliente.chamadas[0].system == SYSTEM_MESSAGE
    assert cliente.chamadas[0].prefixo_cacheavel == formatado("c1", cfg["agent_name"]) + "\n"


def test_resposta_do_contexto_longo_tem_as_chaves_do_autor_e_da_contagem():
    ag = AgentWrapper(CFG_LC, CFG_DADOS, ClienteFalso())
    memorizar(ag, ["c1"])
    r = ag.send_message("Q?")
    assert r["output"] == "ok"
    assert (r["input_len"], r["output_len"], r["memory_construction_time"]) == (123, 2, 0)
    assert r["query_time_len"] >= 0
    assert (r["tokens_entrada"], r["tokens_saida"], r["tokens_cacheados"], r["tokens_escritos_cache"]) == (123, 2, 100, 7)
    assert r["custo_usd"] == 0.001 and r["erro"] is None


# --- RAG --------------------------------------------------------------------------------------

PALAVRAS = ["alfa", "beta", "gama", "delta", "epsilon"]


def test_bm25_recupera_k_trechos_deterministicamente():
    nome = CFG_BM25["agent_name"]
    pergunta = pergunta_rag(nome, "gama")
    mensagens = []
    for _ in range(2):
        cliente = ClienteFalso()
        ag = AgentWrapper(CFG_BM25, CFG_DADOS, cliente)
        memorizar(ag, [f"{p} {p}" for p in PALAVRAS])
        r = ag.send_message(pergunta, query_id=0, context_id=0)
        c = cliente.chamadas[0]
        assert c.system == SYSTEM_MESSAGE and c.prefixo_cacheavel is None and c.max_tokens == 50
        assert c.mensagem.startswith("Memory 1:\n" + formatado("gama gama", nome) + "\n\nMemory 2:\n")
        assert "Memory 3:" not in c.mensagem
        assert c.mensagem.endswith("\n" + pergunta)
        assert r["retrieval_context"][0] == formatado("gama gama", nome) + "\n"
        assert len(r["retrieval_context"]) == 2
        assert (r["input_len"], r["output_len"], r["custo_usd"]) == (123, 2, 0.001)
        mensagens.append(c.mensagem)
    assert mensagens[0] == mensagens[1]


def test_extracao_da_consulta():
    ag = AgentWrapper(CFG_BM25, CFG_DADOS, ClienteFalso())
    assert ag._extract_retrieval_query("x\n\n Now Answer the Question: Who?") == "Who?"
    assert ag._extract_retrieval_query("x\n\n Agora responda à questão: Quem?") == "Quem?"
    assert ag._extract_retrieval_query("sem padrão") == "sem padrão"


def test_embedding_ordena_por_cosseno():
    # Produto interno cru daria beta (3,0) e delta (0,8); o cosseno dá gama (0,995) e delta (0,8).
    vetores = {"consulta": [1.0, 0.0], "alfa": [0.0, 1.0], "beta": [3.0, 3.0], "gama": [0.5, 0.05],
               "delta": [0.8, 0.6], "epsilon": [-1.0, 0.0]}
    emb = EmbeddingsFalso(vetores)
    cliente = ClienteFalso()
    ag = AgentWrapper(CFG_EMB, CFG_DADOS, cliente, embeddings=emb)
    nome = CFG_EMB["agent_name"]
    memorizar(ag, [f"{p} {p}" for p in PALAVRAS])
    pergunta = pergunta_rag(nome, "consulta")
    r = ag.send_message(pergunta, query_id=0, context_id=0)
    esperado = ("Memory 1:\n" + formatado("gama gama", nome) + "\nMemory 2:\n" + formatado("delta delta", nome)
                + "\n" + pergunta)
    c = cliente.chamadas[0]
    assert c.mensagem == esperado
    assert c.system == SYSTEM_MESSAGE and c.prefixo_cacheavel is None and c.max_tokens == 50
    assert emb.chamadas[0] == ag.chunks
    assert emb.chamadas[1] == ["consulta"]  # a consulta extraída, não a mensagem inteira
    assert r["retrieval_context"] == ("Passage 1:\n" + formatado("gama gama", nome) + "\n\nPassage 2:\n"
                                      + formatado("delta delta", nome))
    assert r["input_len"] == n_tokens(r["retrieval_context"] + "\n" + pergunta)
    assert r["output_len"] == n_tokens("ok")
    assert (r["tokens_entrada"], r["custo_usd"], r["erro"]) == (123, 0.001, None)
    # mesma context_id: os trechos não são embutidos de novo
    ag.send_message(pergunta, query_id=1, context_id=0)
    assert len(emb.chamadas) == 3 and emb.chamadas[2] == ["consulta"]


def test_embedding_esquece_alem_do_limite_do_autor():
    cfg = dict(CFG_EMB, input_length_limit=1000, buffer_length=0)
    dados = dict(CFG_DADOS, generation_max_length=0, chunk_size=512)
    ag = AgentWrapper(cfg, dados, ClienteFalso(), embeddings=EmbeddingsFalso({}))
    memorizar(ag, ["um", "dois", "três"])
    assert ag.chunks == [formatado("três", cfg["agent_name"])]  # 1000 // 512 = 1
    assert ag.context_len == 512
