# Resultados: 20261110-direta

## Primárias

Contexto longo × RAG BM25, GPT-4o-mini, perguntas pareadas por `pergunta_id`. Diferença = RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (IC de 95%); linhas com erro contam como erradas. H1 espera diferença positiva (RAG vence), H2 negativa (contexto longo vence). Comparação em que falta resposta final (linha ausente ou com erro de API) para alguma pergunta esperada sai "parcial (x de N)" e fica sem veredito.

Nas MCC, a acurácia é a leitura flexível de `label:` (ver Decisões do piloto no desenho): com o prompt do autor, a resposta obediente vem como "label: N", que o `exact_match` estrito do README conta como erro. O estrito medido está nas notas abaixo da tabela e na última coluna da seção de reprodução.

| Comparação | n | Contexto longo | RAG BM25 | Diferença | Veredito | Nota |
|---|---|---|---|---|---|---|
| H1 direta | 100 | 42.0% [32.0%; 52.0%] | 56.0% [46.0%; 66.0%] | +14.0 pp [+2.0; +26.0] | confirmada |  |
| H2 direta | 10 | 60.0% [30.0%; 90.0%] | 80.0% [50.0%; 100.0%] | +20.0 pp [+0.0; +50.0] | sem veredito | parcial (10 de 500) |
| H1 OAB | não rodada |  |  |  |  | sem: oab-h1-lc-4omini, oab-h1-bm25-4omini |
| H2 OAB | não rodada |  |  |  |  | sem: oab-h2-lc-4omini, oab-h2-bm25-4omini |

- H2 direta: 161 pergunta(s) de `direta-mcc-lc-4omini` e 0 de `direta-mcc-bm25-4omini` descartadas por não estarem nas duas configurações (ou fora das perguntas esperadas).
- H2 direta, `exact_match` estrito (literal do README), nas mesmas 10 perguntas: contexto longo 0.0%, RAG BM25 0.0%.

## Comparações secundárias

Contexto longo × RAG (embedding ou BM25), pareadas por `pergunta_id`, sem veredito: só a diferença com IC de 95%. Diferença = RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (nas MCC, estratificado por dataset); linhas com erro contam como erradas e nas MCC a acurácia é a leitura flexível de `label:`. Comparação com resposta final faltando sai "parcial (x de N)"; par sem nenhuma configuração na pasta sai "não rodada".

| Comparação | n | Contexto longo | RAG | Diferença | Nota |
|---|---|---|---|---|---|
| H1 direta: contexto longo × embedding | 10 | 30.0% [0.0%; 60.0%] | 50.0% [20.0%; 80.0%] | +20.0 pp [+0.0; +50.0] | parcial (10 de 100) |
| H2 direta: contexto longo × embedding | não rodada |  |  |  | sem: direta-mcc-emb-4omini |
| H1 OAB, GPT-4o-mini: contexto longo × embedding | não rodada |  |  |  | sem: oab-h1-lc-4omini, oab-h1-emb-4omini |
| H1 OAB, Haiku: contexto longo × BM25 | não rodada |  |  |  | sem: oab-h1-lc-haiku, oab-h1-bm25-haiku |
| H1 OAB, Haiku: contexto longo × embedding | não rodada |  |  |  | sem: oab-h1-lc-haiku, oab-h1-emb-haiku |
| H2 OAB, GPT-4o-mini: contexto longo × embedding | não rodada |  |  |  | sem: oab-h2-lc-4omini, oab-h2-emb-4omini |
| H2 OAB, Haiku: contexto longo × BM25 | não rodada |  |  |  | sem: oab-h2-lc-haiku, oab-h2-bm25-haiku |
| H2 OAB, Haiku: contexto longo × embedding | não rodada |  |  |  | sem: oab-h2-lc-haiku, oab-h2-emb-haiku |

- H1 direta: contexto longo × embedding: 90 pergunta(s) de `direta-mhqa-lc-4omini` e 0 de `direta-mhqa-emb-4omini` descartadas por não estarem nas duas configurações (ou fora das perguntas esperadas).

## Reprodução (etapa direta)

Número do artigo (em %) contra a nossa acurácia com IC de 95% (bootstrap). Reproduz se o número do artigo cai dentro do intervalo. Média MCC: bootstrap estratificado por dataset. Nas MCC, "Nossa acurácia" é a leitura flexível de `label:` e "Reproduz" a usa; a coluna do estrito é o `exact_match` literal do README (sem veredito). Fonte parcial (falta resposta final para alguma pergunta esperada) fica "sem teste"; a média MCC exige as cinco fontes completas.

| Configuração | Fonte | n | Nossa acurácia | Artigo | Reproduz | Estrito (literal do README) | Nota |
|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | ruler_qa2_421K | 100 | 42.0% [32.0%; 52.0%] | 43.0% | sim |  |  |
| direta-mhqa-bm25-4omini | ruler_qa2_421K | 100 | 56.0% [46.0%; 66.0%] | 56.0% | sim |  |  |
| direta-mcc-lc-4omini | icl_banking77_5900shot_balance | 100 | 89.0% [83.0%; 95.0%] | 93.0% | sim | 0.0% [0.0%; 0.0%] |  |
| direta-mcc-lc-4omini | icl_clinic150_7050shot_balance | 71 | 85.9% [77.5%; 93.0%] | 93.0% | sem teste | 0.0% [0.0%; 0.0%] | parcial (70 de 100) |
| direta-mcc-lc-4omini | icl_nlu_8296shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-lc-4omini | icl_trec_coarse_6600shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-lc-4omini | icl_trec_fine_6400shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-lc-4omini | média MCC | 171 | 87.7% [82.5%; 92.4%] | 82.0% | sem teste | 0.0% [0.0%; 0.0%] | parcial (170 de 500) |
| direta-mcc-bm25-4omini | icl_banking77_5900shot_balance | 10 | 80.0% [50.0%; 100.0%] | 89.0% | sem teste | 0.0% [0.0%; 0.0%] | parcial (10 de 100) |
| direta-mcc-bm25-4omini | icl_clinic150_7050shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-bm25-4omini | icl_nlu_8296shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-bm25-4omini | icl_trec_coarse_6600shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-bm25-4omini | icl_trec_fine_6400shot_balance | não rodada |  |  |  |  | parcial (0 de 100) |
| direta-mcc-bm25-4omini | média MCC | 10 | 80.0% [50.0%; 100.0%] | 75.4% | sem teste | 0.0% [0.0%; 0.0%] | parcial (10 de 500) |
| direta-mhqa-emb-4omini | ruler_qa2_421K | 10 | 50.0% [20.0%; 80.0%] | 44.0% | sem teste |  | parcial (10 de 100) |
| direta-mcc-emb-4omini | não rodada |  |  |  |  |  |  |

## Secundárias

Acurácia com IC de 95%, sem veredito. As configurações diretas têm a comparação com o artigo na seção de reprodução.

| Configuração | n | Acurácia | Nota |
|---|---|---|---|
| direta-mhqa-emb-4omini | 10 | 50.0% [20.0%; 80.0%] | parcial (10 de 100) |

## Custos

Por configuração. Erros de API e recusas/respostas vazias são contados à parte e valem como erro.

| Configuração | n | Tokens de entrada | % cacheado | Tokens de saída | US$ | Erros de API | Recusas/vazias | Nota |
|---|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | 100 | 12,402,749 | 96.9% | 986 | 0.9596 | 0 | 0 |  |
| direta-mhqa-bm25-4omini | 100 | 553,947 | 0.0% | 1,136 | 0.0838 | 0 | 0 |  |
| direta-mcc-lc-4omini | 171 | 18,186,236 | 96.9% | 680 | 1.4062 | 1 | 0 | parcial (170 de 500) |
| direta-mcc-bm25-4omini | 10 | 414,375 | 12.9% | 40 | 0.0582 | 0 | 0 | parcial (10 de 500) |
| direta-mhqa-emb-4omini | 10 | 54,743 | 0.0% | 147 | 0.0083 | 0 | 0 | parcial (10 de 100) |

A coluna US$ soma o `custo_usd` das linhas, que não inclui os embeddings (centavos, só no livro-caixa). As configurações de embedding com o Haiku de leitor também gastaram na OpenAI, nos embeddings, e esse gasto não aparece aqui.
