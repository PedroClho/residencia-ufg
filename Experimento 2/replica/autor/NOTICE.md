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

## Mudanças da Task 4 (`agente.py`)

Serão registradas aqui quando o arquivo existir.
