# Código copiado do MemoryAgentBench

Origem: https://github.com/HUST-AI-HYZ/MemoryAgentBench (Hu, Wang e McAuley, ICLR 2026), licença MIT
(arquivo `LICENSE` desta pasta, copiado sem alteração).
Commit fixado: `538026089d1a8a8eff05121d0db89b388f360eba` (`replica.config.AUTOR_COMMIT`).

## Arquivos e funções copiados

| Aqui | Origem no commit | Situação |
|---|---|---|
| `LICENSE` | `LICENSE` | idêntico |
| `templates.py` | `utils/templates.py` | **idêntico byte a byte** (o teste confere o sha256, com finais de linha LF) |
| `metricas.py` | `utils/eval_other_utils.py` | só as funções abaixo, nomes e corpos do autor |
| `agente.py` | `agent.py` (classe `AgentWrapper`) e `methods/embedding_retriever.py` (`TextRetriever.retrieve`, `RAGSystem.answer_query`) | só os métodos abaixo, nomes e corpos do autor, com as adaptações listadas |

Funções de `metricas.py`: `normalize_answer`, `f1_score`, `drqa_exact_match_score`,
`substring_exact_match_score`, `drqa_metric_max_over_ground_truths`, `parse_output`,
`chunk_text_into_sentences`, `calculate_metrics`, `post_process`, `_process_icl_dataset`,
`_process_ruler_memory_merging_dataset`, `default_post_process`.

## Mudanças feitas (e por quê)

Em `metricas.py`:

1. `calculate_metrics` não calcula mais o ROUGE; devolve só `exact_match`, `f1` e
   `substring_exact_match`. O experimento não usa ROUGE, e isso evita as dependências
   `rouge_score` e `editdistance`.
2. `chunk_text_into_sentences` baixa também o `punkt_tab`, além do `punkt` (exigência do nltk 3.9 ou
   mais novo, que o autor não tinha).
3. A configuração de logging do módulo (`logging.basicConfig` e o logger) foi retirada; o pacote não
   deve mexer na configuração global de logging.
4. `post_process` tem só os ramos `icl` e `ruler`/`mermory_merging` (a grafia é a do autor), mais o
   padrão. Os ramos `eventqa`, `infbench`/`longmemeval` e `recsys` ficaram de fora porque o
   experimento não usa esses conjuntos.

Também ficaram de fora, por não serem usadas: `count_tokens`, `create_chunks_use_sent_tokenizer`,
as funções de filmes e recomendação, `metrics_summarization` e os outros `_process_*`. Os imports
foram reduzidos ao que as funções copiadas usam.

Em `templates.py`: nenhuma.

## Mudanças em `agente.py`

Métodos de `AgentWrapper` copiados de `agent.py`: `__init__`, `_initialize_agent_by_type`,
`_is_agent_type`, `_create_standard_response`, `_initialize_long_context_agent`,
`_initialize_rag_agent`, `send_message`, `_handle_long_context_agent`, `_query_long_context_agent`,
`_truncate_context_if_needed`, `_handle_rag_agent`, `_process_rag_query`, `_handle_bm25_rag`,
`_extract_retrieval_query` e `_handle_embedding_rag`. De `methods/embedding_retriever.py` vieram
`TextRetriever.retrieve` e `RAGSystem.answer_query`, que aqui viram métodos de `AgentWrapper` com o
mesmo nome (`retrieve`, `answer_query`). O único método novo é `_anexar_contagem`.

Adaptações (a lista fechada do plano, de 1 a 7):

1. **Toda chamada de LLM passa por `self.cliente.conversar(system, mensagem, max_tokens=self.max_tokens,
   prefixo_cacheavel=...)`**, o cliente com teto de gasto de `replica.clientes`. No contexto longo, o
   prefixo cacheável é `self.context + "\n"` e a mensagem é a pergunta; o texto enviado é o
   `full_message` do autor (`self.context + "\n" + message`). Nos RAG, o prefixo é `None` e a mensagem é
   o `ask_llm_message` do autor. Consequências:
   - some o `format_chat`: o cliente monta a lista de mensagens (system e user) do mesmo jeito;
   - `max_tokens` vai sempre (o BM25 do autor só mandava para modelos `gpt-4`, o que inclui o
     GPT-4o-mini; a Anthropic exige o parâmetro);
   - a temperatura fica no cliente (`ClienteOpenAI(temperatura=0.7)`, o valor das configurações do
     autor); o Haiku 5.5 não recebe temperatura, porque a rejeita. `self.temperature` continua lido
     da configuração, como no autor, mas não é usado;
   - onde o autor lia `response.usage` (contexto longo e BM25), `input_len` e `output_len` vêm de
     `Resposta.tokens_entrada` e `Resposta.tokens_saida`; no embedding, continuam contados pelo
     tokenizador, como o autor fazia;
   - o dicionário devolvido ganha, além das chaves do autor, `tokens_entrada`, `tokens_saida`,
     `tokens_cacheados`, `tokens_escritos_cache`, `custo_usd` e `erro`, tirados da `Resposta`
     (método novo `_anexar_contagem`). `answer_query` devolve também a `Resposta` na chave `resposta`.
2. **O Haiku recebe o `system_message`.** O caminho Claude do autor (`_query_claude`) omitia o system;
   o Haiku não está no artigo, e ficamos com o mesmo prompt dos agentes da OpenAI.
3. **BM25 sem LangChain:** `rank_bm25.BM25Okapi([c.split() for c in self.chunks])`, construído quando
   o `context_id` muda, e `get_top_n(consulta.split(), self.chunks, n=self.retrieve_num)`. É o que o
   `BM25Retriever` do LangChain faz por dentro (tokenização por `str.split()`, sem minúsculas).
4. **Embedding sem LangChain, FAISS nem Azure:** `self.embeddings.embutir(self.chunks)` quando o
   `context_id` muda, consulta com `embutir([consulta])` e top-k por produto interno entre vetores
   normalizados, com `numpy.argsort` descendente (estável). O FAISS L2 do autor sobre os vetores do
   `text-embedding-3-small`, que já vêm normalizados, dá a mesma ordem. O caminho do autor exigia
   Azure (`use_azure=True` fixo); aqui vai pelo `ClienteEmbeddings` normal da OpenAI. O modelo de
   embedding vem do cliente injetado (`config.MODELO_EMBEDDING`), por isso saiu o bloco que escolhia
   `embedding_model_name` pelo nome do agente.
5. **`_extract_retrieval_query` ganha o padrão `r"Agora responda à questão:\s*(.*)"`**, depois dos dois
   do autor, para a OAB. O caminho de embedding passou a usar esse método no lugar das regex repetidas
   dentro de `RAGSystem.answer_query` (que eram as mesmas duas, na mesma ordem).
6. **Templates por `self.obter_template(...)`** (padrão: o `get_template` do autor), para a OAB plugar
   os seus.
7. **Sem torch, transformers, LangChain, Azure, nem gravação em `./outputs/`.** O `_process_rag_query`
   não grava mais o `retrieval_context` em JSON nem o retira do dicionário: ele volta para quem chamou,
   e o rodador decide o que guardar.

Recortes que acompanham as adaptações:

- `__init__` recebe `cliente`, `embeddings` e `obter_template` no lugar de `load_agent_from`; o
  atributo `agent_save_to_folder`, que só servia ao Letta, saiu.
- `_initialize_agent_by_type`, `send_message` e `_process_rag_query` têm só os ramos do contexto longo,
  do `rag_bm25` e do `rag_text_embedding_3_small`. Os outros agentes do autor (Letta, Mem0, Cognee,
  Zep, GraphRAG, HippoRAG, RAPTOR, Self-RAG, MemoRAG, Contriever, Qwen3) não fazem parte do
  experimento; um nome de agente desses cai no `NotImplementedError` do próprio autor.
- `_initialize_long_context_agent` só zera o contexto: o cliente é injetado em vez de criado ali.
- Saíram os `print(...)` de depuração dos caminhos de BM25 e embedding (só ruído no log).
- `retrieve` e `answer_query` usam `self.embeddings`, `self.vetores_chunks`, `self.chunks` e
  `self.cliente` no lugar dos atributos de `TextRetriever` e `RAGSystem`.

Comportamentos do autor mantidos de propósito: o `time_stamp` de `time.strftime` no template de
memorização; o truncamento do contexto longo (decode dos últimos N tokens) a cada pergunta; o descarte
do trecho mais antigo no RAG quando `context_len` passa de `input_length_limit`, contando
`chunk_size` por trecho; o trecho vazio que o fatiador pode gerar, que não é filtrado.
