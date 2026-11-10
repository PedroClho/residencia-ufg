"""Constantes do experimento: modelos, preços, tetos, revisões, fontes e números do artigo."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
RESULTADOS = RAIZ / "resultados"
LIVRO_CAIXA = RESULTADOS / "livro_caixa.json"

MODELO_OPENAI = "gpt-4o-mini"
MODELO_ANTHROPIC = "claude-haiku-5-5"
MODELO_EMBEDDING = "text-embedding-3-small"

# Tetos do livro-caixa: US$ 0,50 antes da trava dos painéis (US$ 20 e US$ 30).
TETOS = {"openai": 19.5, "anthropic": 29.5}
# Reserva da OpenAI para depurar a etapa direta; liberada com usar_reserva=True.
RESERVA = {"openai": 4.0, "anthropic": 0.0}

# USD por milhão de tokens.
PRECOS = {
    "gpt-4o-mini": {"entrada": 0.15, "entrada_cache": 0.075, "saida": 0.60},
    "text-embedding-3-small": {"entrada": 0.02},
    "claude-haiku-5-5": {
        "entrada": 0.10,
        "saida": 0.50,
        "entrada_longa": 0.50,
        "saida_longa": 2.50,
        "limiar_longo": 100_000,
        "mult_escrita_cache": 1.25,
        "mult_leitura_cache": 0.10,
    },
}

PROVEDOR = {
    "gpt-4o-mini": "openai",
    "text-embedding-3-small": "openai",
    "claude-haiku-5-5": "anthropic",
}

# Margem sobre a razão medida de 1,71 (tokens do Haiku / tokens o200k).
FATOR_TOKENS_HAIKU = 1.85
JANELA_HAIKU = 1_000_000

AUTOR_COMMIT = "538026089d1a8a8eff05121d0db89b388f360eba"
HF_DATASET = "ai-hyz/MemoryAgentBench"
HF_REVISAO = "7ea066982b140a19337e17e60d45d4076e042faf"

FONTE_MHQA = "ruler_qa2_421K"
MCC_FONTES = (
    "icl_banking77_5900shot_balance",
    "icl_clinic150_7050shot_balance",
    "icl_nlu_8296shot_balance",
    "icl_trec_coarse_6600shot_balance",
    "icl_trec_fine_6400shot_balance",
)

# Números do artigo (arXiv 2507.05257v4). Chave: (fonte ou "mcc", agente).
NUMEROS_ARTIGO: dict[tuple[str, str], float] = {
    # Tabela 3: MH-QA e média das cinco MCC.
    (FONTE_MHQA, "contexto_longo"): 43.0,
    (FONTE_MHQA, "rag_bm25"): 56.0,
    (FONTE_MHQA, "rag_embedding"): 44.0,
    ("mcc", "contexto_longo"): 82.0,
    ("mcc", "rag_bm25"): 75.4,
    ("mcc", "rag_embedding"): 70.0,
    # Tabela 7: por dataset MCC.
    (MCC_FONTES[0], "contexto_longo"): 93.0,
    (MCC_FONTES[0], "rag_bm25"): 89.0,
    (MCC_FONTES[0], "rag_embedding"): 88.0,
    (MCC_FONTES[1], "contexto_longo"): 93.0,
    (MCC_FONTES[1], "rag_bm25"): 89.0,
    (MCC_FONTES[1], "rag_embedding"): 89.0,
    (MCC_FONTES[2], "contexto_longo"): 87.0,
    (MCC_FONTES[2], "rag_bm25"): 84.0,
    (MCC_FONTES[2], "rag_embedding"): 83.0,
    (MCC_FONTES[3], "contexto_longo"): 73.0,
    (MCC_FONTES[3], "rag_bm25"): 62.0,
    (MCC_FONTES[3], "rag_embedding"): 54.0,
    (MCC_FONTES[4], "contexto_longo"): 66.0,
    (MCC_FONTES[4], "rag_bm25"): 53.0,
    (MCC_FONTES[4], "rag_embedding"): 36.0,
}

SEMENTE = 42
PROJETO_ATLAS = "gqnejwbuloklbelsufnj"
