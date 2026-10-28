"""Constantes do projeto: modelo, tamanhos, limites, preços e o prompt base.

Ficam todas aqui, num só lugar, para quem lê o código ver de uma vez o que dá para ajustar.
As quatro configurações da comparação (sem memória, contexto completo, passiva, ativa) leem os
mesmos valores, então mudar uma constante muda as quatro ao mesmo tempo e a comparação segue justa.
"""

# Modelo único do projeto (Claude Sonnet 5.5). Só llm.py lê esta constante.
MODELO = "claude-sonnet-5-5"

# Janela recente: quantas mensagens (usuário + assistente) da sessão atual vão em cada chamada.
K = 6

# Teto de tokens gerados por chamada. O Sonnet 5.5 raciocina por padrão e esse raciocínio conta
# neste teto; um valor baixo cortaria a resposta no meio. Cobra-se só o que é gerado, então o teto
# alto não custa nada a mais.
MAX_TOKENS = 16000

# Memória passiva: o resumo da conversa nunca passa deste tamanho (o resumidor é instruído a isso).
LIMITE_PALAVRAS_RESUMO = 400

# Memória ativa: quantas notas a busca devolve, no máximo, por consulta.
TOP_N_NOTAS = 5

# Memória ativa: quantas rodadas de ferramenta o modelo pode fazer em um turno antes de desistirmos.
MAX_ITERACOES_FERRAMENTAS = 6

# Arquivo SQLite onde fica a memória (resumo da passiva, notas da ativa).
CAMINHO_BANCO = "memoria.db"

# Preços do modelo em dólares por milhão de tokens, para estimar o custo de cada configuração.
PRECO_ENTRADA_USD_POR_MILHAO = 2.0
PRECO_SAIDA_USD_POR_MILHAO = 10.0

# System prompt comum às quatro configurações. Cada uma acrescenta só o bloco que lhe diz respeito.
SYSTEM_BASE = """\
Você é um assistente pessoal. Responda em português do Brasil, de forma direta e curta.
Quando não tiver a informação para responder, diga claramente que não sabe ou que isso não foi mencionado, em vez de inventar."""
