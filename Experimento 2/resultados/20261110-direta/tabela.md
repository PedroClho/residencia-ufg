# Resultados: 20261110-direta

## Primárias

Contexto longo × RAG BM25, GPT-4o-mini, perguntas pareadas por `pergunta_id`. Diferença = RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (IC de 95%); linhas com erro contam como erradas. H1 espera diferença positiva (RAG vence), H2 negativa (contexto longo vence). Comparação em que falta resposta final (linha ausente ou com erro de API) para alguma pergunta esperada sai "parcial (x de N)" e fica sem veredito.

Nas MCC, a acurácia é a leitura flexível de `label:` (ver Decisões do piloto no desenho): com o prompt do autor, a resposta obediente vem como "label: N", que o `exact_match` estrito do README conta como erro. O estrito medido está nas notas abaixo da tabela e na última coluna da seção de reprodução.

| Comparação | n | Contexto longo | RAG BM25 | Diferença | Veredito | Nota |
|---|---|---|---|---|---|---|
| H1 direta | 100 | 42.0% [32.0%; 52.0%] | 56.0% [46.0%; 66.0%] | +14.0 pp [+2.0; +26.0] | confirmada |  |
| H2 direta | 500 | 77.0% [73.4%; 80.4%] | 76.8% [73.4%; 80.2%] | -0.2 pp [-2.8; +2.4] | inconclusiva |  |
| H1 OAB | não rodada |  |  |  |  | sem: oab-h1-lc-4omini, oab-h1-bm25-4omini |
| H2 OAB | não rodada |  |  |  |  | sem: oab-h2-lc-4omini, oab-h2-bm25-4omini |

- H2 direta, `exact_match` estrito (literal do README), nas mesmas 500 perguntas: contexto longo 0.0%, RAG BM25 0.0%.

## Comparações secundárias

Contexto longo × RAG (embedding ou BM25), pareadas por `pergunta_id`, sem veredito: só a diferença com IC de 95%. Diferença = RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (nas MCC, estratificado por dataset); linhas com erro contam como erradas e nas MCC a acurácia é a leitura flexível de `label:`. Comparação com resposta final faltando sai "parcial (x de N)"; par com alguma configuração ausente da pasta sai "não rodada", e a nota diz qual falta.

| Comparação | n | Contexto longo | RAG | Diferença | Nota |
|---|---|---|---|---|---|
| H1 direta: contexto longo × embedding | 100 | 42.0% [32.0%; 52.0%] | 45.0% [35.0%; 55.0%] | +3.0 pp [-6.0; +13.0] |  |
| H2 direta: contexto longo × embedding | 266 | 87.6% [83.5%; 91.4%] | 88.7% [85.0%; 92.5%] | +1.1 pp [-1.5; +4.1] | parcial (266 de 500) |
| H1 OAB, GPT-4o-mini: contexto longo × embedding | não rodada |  |  |  | sem: oab-h1-lc-4omini, oab-h1-emb-4omini |
| H1 OAB, Haiku: contexto longo × BM25 | não rodada |  |  |  | sem: oab-h1-lc-haiku, oab-h1-bm25-haiku |
| H1 OAB, Haiku: contexto longo × embedding | não rodada |  |  |  | sem: oab-h1-lc-haiku, oab-h1-emb-haiku |
| H2 OAB, GPT-4o-mini: contexto longo × embedding | não rodada |  |  |  | sem: oab-h2-lc-4omini, oab-h2-emb-4omini |
| H2 OAB, Haiku: contexto longo × BM25 | não rodada |  |  |  | sem: oab-h2-lc-haiku, oab-h2-bm25-haiku |
| H2 OAB, Haiku: contexto longo × embedding | não rodada |  |  |  | sem: oab-h2-lc-haiku, oab-h2-emb-haiku |

- H2 direta: contexto longo × embedding: 234 pergunta(s) de `direta-mcc-lc-4omini` e 0 de `direta-mcc-emb-4omini` descartadas por não estarem nas duas configurações (ou fora das perguntas esperadas).

## Reprodução (etapa direta)

Número do artigo (em %) contra a nossa acurácia com IC de 95% (bootstrap). Reproduz se o número do artigo cai dentro do intervalo. Média MCC: bootstrap estratificado por dataset. Nas MCC, "Nossa acurácia" é a leitura flexível de `label:` e "Reproduz" a usa; a coluna do estrito é o `exact_match` literal do README (sem veredito). Fonte parcial (falta resposta final para alguma pergunta esperada) fica "sem teste"; a média MCC exige as cinco fontes completas.

| Configuração | Fonte | n | Nossa acurácia | Artigo | Reproduz | Estrito (literal do README) | Nota |
|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | ruler_qa2_421K | 100 | 42.0% [32.0%; 52.0%] | 43.0% | sim |  |  |
| direta-mhqa-bm25-4omini | ruler_qa2_421K | 100 | 56.0% [46.0%; 66.0%] | 56.0% | sim |  |  |
| direta-mcc-lc-4omini | icl_banking77_5900shot_balance | 100 | 89.0% [83.0%; 95.0%] | 93.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-lc-4omini | icl_clinic150_7050shot_balance | 100 | 88.0% [81.0%; 94.0%] | 93.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-lc-4omini | icl_nlu_8296shot_balance | 100 | 84.0% [77.0%; 91.0%] | 87.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-lc-4omini | icl_trec_coarse_6600shot_balance | 100 | 68.0% [59.0%; 77.0%] | 73.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-lc-4omini | icl_trec_fine_6400shot_balance | 100 | 56.0% [46.0%; 65.0%] | 66.0% | não | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-lc-4omini | média MCC | 500 | 77.0% [73.4%; 80.4%] | 82.0% | não | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-bm25-4omini | icl_banking77_5900shot_balance | 100 | 92.0% [86.0%; 97.0%] | 89.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-bm25-4omini | icl_clinic150_7050shot_balance | 100 | 87.0% [80.0%; 93.0%] | 89.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-bm25-4omini | icl_nlu_8296shot_balance | 100 | 83.0% [75.0%; 90.0%] | 84.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-bm25-4omini | icl_trec_coarse_6600shot_balance | 100 | 70.0% [61.0%; 79.0%] | 62.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-bm25-4omini | icl_trec_fine_6400shot_balance | 100 | 52.0% [42.0%; 62.0%] | 53.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-bm25-4omini | média MCC | 500 | 76.8% [73.4%; 80.2%] | 75.4% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mhqa-emb-4omini | ruler_qa2_421K | 100 | 45.0% [35.0%; 55.0%] | 44.0% | sim |  |  |
| direta-mcc-emb-4omini | icl_banking77_5900shot_balance | 100 | 91.0% [85.0%; 96.0%] | 88.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-emb-4omini | icl_clinic150_7050shot_balance | 100 | 89.0% [82.0%; 95.0%] | 89.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-emb-4omini | icl_nlu_8296shot_balance | 66 | 84.8% [75.8%; 92.4%] | 83.0% | sem teste | 0.0% [0.0%; 0.0%] | parcial (66 de 100) |
| direta-mcc-emb-4omini | icl_trec_coarse_6600shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-emb-4omini | icl_trec_fine_6400shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-emb-4omini | média MCC | 266 | 88.7% [85.0%; 92.5%] | 70.0% | sem teste | 0.0% [0.0%; 0.0%] | parcial (266 de 500) |

## Secundárias

Acurácia com IC de 95%, sem veredito. As configurações diretas têm a comparação com o artigo na seção de reprodução.

| Configuração | n | Acurácia | Nota |
|---|---|---|---|
| direta-mhqa-emb-4omini | 100 | 45.0% [35.0%; 55.0%] |  |
| direta-mcc-emb-4omini | 266 | 88.7% [85.0%; 92.5%] | parcial (266 de 500) |

## Custos

Por configuração. Erros de API e recusas/respostas vazias são contados à parte e valem como erro.

| Configuração | n | Tokens de entrada | % cacheado | Tokens de saída | US$ | Erros de API | Recusas/vazias | Nota |
|---|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | 100 | 12,402,749 | 96.9% | 986 | 0.9596 | 0 | 0 |  |
| direta-mhqa-bm25-4omini | 100 | 553,947 | 0.0% | 1,136 | 0.0838 | 0 | 0 |  |
| direta-mcc-lc-4omini | 500 | 53,259,848 | 97.2% | 2,000 | 4.1068 | 0 | 0 |  |
| direta-mcc-bm25-4omini | 500 | 21,131,867 | 26.7% | 2,000 | 2.7480 | 0 | 0 |  |
| direta-mhqa-emb-4omini | 100 | 553,033 | 0.0% | 1,568 | 0.0839 | 0 | 0 |  |
| direta-mcc-emb-4omini | 266 | 9,016,322 | 12.8% | 1,064 | 1.2667 | 0 | 0 | parcial (266 de 500) |

A coluna US$ soma o `custo_usd` das linhas, que não inclui os embeddings (centavos, só no livro-caixa). As configurações de embedding com o Haiku de leitor também gastaram na OpenAI, nos embeddings, e esse gasto não aparece aqui.
