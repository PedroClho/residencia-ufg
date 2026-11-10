# Resultados: 20261110-oab

## Primárias

Contexto longo × RAG BM25, GPT-4o-mini, perguntas pareadas por `pergunta_id`. Diferença = RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (IC de 95%); linhas com erro contam como erradas. H1 espera diferença positiva (RAG vence), H2 negativa (contexto longo vence). Comparação em que falta resposta final (linha ausente ou com erro de API) para alguma pergunta esperada sai "parcial (x de N)" e fica sem veredito.

Nas MCC, a acurácia é a leitura flexível de `label:` (ver Decisões do piloto no desenho): com o prompt do autor, a resposta obediente vem como "label: N", que o `exact_match` estrito do README conta como erro. O estrito medido está nas notas abaixo da tabela e na última coluna da seção de reprodução.

| Comparação | n | Contexto longo | RAG BM25 | Diferença | Veredito | Nota |
|---|---|---|---|---|---|---|
| H1 direta | não rodada |  |  |  |  | sem: direta-mhqa-lc-4omini, direta-mhqa-bm25-4omini |
| H2 direta | não rodada |  |  |  |  | sem: direta-mcc-lc-4omini, direta-mcc-bm25-4omini |
| H1 OAB | 300 | 58.7% [53.0%; 64.3%] | 64.0% [58.3%; 69.3%] | +5.3 pp [+1.7; +9.0] | confirmada |  |
| H2 OAB | 224 | 65.2% [58.9%; 71.4%] | 66.1% [59.8%; 72.3%] | +0.9 pp [-3.6; +5.4] | inconclusiva |  |

## Comparações secundárias

Contexto longo × RAG (embedding ou BM25), pareadas por `pergunta_id`, sem veredito: só a diferença com IC de 95%. Diferença = RAG − contexto longo, bootstrap pareado de 10.000 reamostragens (nas MCC, estratificado por dataset); linhas com erro contam como erradas e nas MCC a acurácia é a leitura flexível de `label:`. Comparação com resposta final faltando sai "parcial (x de N)"; par com alguma configuração ausente da pasta sai "não rodada", e a nota diz qual falta.

| Comparação | n | Contexto longo | RAG | Diferença | Nota |
|---|---|---|---|---|---|
| H1 direta: contexto longo × embedding | não rodada |  |  |  | sem: direta-mhqa-lc-4omini, direta-mhqa-emb-4omini |
| H2 direta: contexto longo × embedding | não rodada |  |  |  | sem: direta-mcc-lc-4omini, direta-mcc-emb-4omini |
| H1 OAB, GPT-4o-mini: contexto longo × embedding | 300 | 58.7% [53.0%; 64.3%] | 64.0% [58.7%; 69.3%] | +5.3 pp [+1.3; +9.7] |  |
| H1 OAB, Haiku: contexto longo × BM25 | 300 | 73.7% [68.7%; 78.7%] | 74.3% [69.3%; 79.0%] | +0.7 pp [-2.7; +4.0] |  |
| H1 OAB, Haiku: contexto longo × embedding | 300 | 73.7% [68.7%; 78.7%] | 72.7% [67.3%; 77.7%] | -1.0 pp [-4.7; +2.7] |  |
| H2 OAB, GPT-4o-mini: contexto longo × embedding | 224 | 65.2% [58.9%; 71.4%] | 67.0% [60.7%; 73.2%] | +1.8 pp [-2.2; +5.8] |  |
| H2 OAB, Haiku: contexto longo × BM25 | 224 | 73.2% [67.4%; 79.0%] | 70.5% [64.3%; 76.3%] | -2.7 pp [-7.6; +2.2] |  |
| H2 OAB, Haiku: contexto longo × embedding | 224 | 73.2% [67.4%; 79.0%] | 75.9% [70.1%; 81.2%] | +2.7 pp [-1.3; +6.7] |  |

## Reprodução (etapa direta)

Número do artigo (em %) contra a nossa acurácia com IC de 95% (bootstrap). Reproduz se o número do artigo cai dentro do intervalo. Média MCC: bootstrap estratificado por dataset. Nas MCC, "Nossa acurácia" é a leitura flexível de `label:` e "Reproduz" a usa; a coluna do estrito é o `exact_match` literal do README (sem veredito). Fonte parcial (falta resposta final para alguma pergunta esperada) fica "sem teste"; a média MCC exige as cinco fontes completas.

| Configuração | Fonte | n | Nossa acurácia | Artigo | Reproduz | Estrito (literal do README) | Nota |
|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | não rodada |  |  |  |  |  |  |
| direta-mhqa-bm25-4omini | não rodada |  |  |  |  |  |  |
| direta-mcc-lc-4omini | não rodada |  |  |  |  |  |  |
| direta-mcc-bm25-4omini | não rodada |  |  |  |  |  |  |
| direta-mhqa-emb-4omini | não rodada |  |  |  |  |  |  |
| direta-mcc-emb-4omini | não rodada |  |  |  |  |  |  |

## Secundárias

Acurácia com IC de 95%, sem veredito. As configurações diretas têm a comparação com o artigo na seção de reprodução.

| Configuração | n | Acurácia | Nota |
|---|---|---|---|
| oab-sem-memoria-4omini | 3377 | 58.9% [57.3%; 60.6%] |  |
| oab-sem-memoria-haiku | 3377 | 70.7% [69.2%; 72.3%] |  |
| oab-h1-emb-4omini | 300 | 64.0% [58.7%; 69.3%] |  |
| oab-h1-bm25-haiku | 300 | 74.3% [69.3%; 79.0%] |  |
| oab-h1-emb-haiku | 300 | 72.7% [67.3%; 77.7%] |  |
| oab-h1-bm25-4omini-k2 | 300 | 61.7% [56.0%; 67.0%] |  |
| oab-h1-bm25-4omini-k5 | 300 | 63.3% [57.7%; 68.7%] |  |
| oab-h1-bm25-4omini-k20 | 300 | 62.0% [56.3%; 67.3%] |  |
| oab-h1corpus-bm25-4omini | 300 | 64.0% [58.3%; 69.3%] |  |
| oab-h2-bm25-haiku | 224 | 70.5% [64.3%; 76.3%] |  |
| oab-h1-lc-haiku | 300 | 73.7% [68.7%; 78.7%] |  |
| oab-h2-lc-haiku | 224 | 73.2% [67.4%; 79.0%] |  |
| oab-h2-emb-4omini | 224 | 67.0% [60.7%; 73.2%] |  |
| oab-h2-emb-haiku | 224 | 75.9% [70.1%; 81.2%] |  |

## Custos

Por configuração. Erros de API e recusas/respostas vazias são contados à parte e valem como erro.

| Configuração | n | Tokens de entrada | % cacheado | Tokens de saída | US$ | Erros de API | Recusas/vazias | Nota |
|---|---|---|---|---|---|---|---|---|
| oab-sem-memoria-4omini | 3377 | 1,076,206 | 0.0% | 3,429 | 0.1635 | 0 | 0 |  |
| oab-sem-memoria-haiku | 3377 | 1,803,542 | 0.0% | 10,131 | 0.1854 | 0 | 0 |  |
| oab-h1-lc-4omini | 300 | 27,403,864 | 94.2% | 580 | 2.1756 | 0 | 0 |  |
| oab-h1-bm25-4omini | 300 | 1,780,180 | 0.1% | 701 | 0.2673 | 0 | 0 |  |
| oab-h2-lc-4omini | 224 | 27,867,265 | 98.3% | 970 | 2.1269 | 0 | 0 |  |
| oab-h2-bm25-4omini | 224 | 9,357,047 | 1.9% | 1,054 | 1.3910 | 0 | 0 |  |
| oab-h1-emb-4omini | 300 | 1,728,215 | 0.7% | 792 | 0.2588 | 0 | 0 |  |
| oab-h1-bm25-haiku | 300 | 2,978,557 | 0.0% | 900 | 0.2983 | 0 | 0 |  |
| oab-h1-emb-haiku | 300 | 2,866,226 | 0.0% | 900 | 0.2871 | 0 | 0 |  |
| oab-h1-bm25-4omini-k2 | 300 | 447,093 | 0.5% | 1,178 | 0.0676 | 0 | 0 |  |
| oab-h1-bm25-4omini-k5 | 300 | 949,111 | 0.2% | 950 | 0.1428 | 0 | 0 |  |
| oab-h1-bm25-4omini-k20 | 300 | 3,426,659 | 0.0% | 689 | 0.5143 | 0 | 0 |  |
| oab-h1corpus-bm25-4omini | 300 | 1,843,856 | 0.1% | 814 | 0.2770 | 0 | 0 |  |
| oab-h2-bm25-haiku | 224 | 16,239,481 | 0.0% | 2,016 | 1.6250 | 0 | 0 |  |
| oab-h1-lc-haiku | 300 | 59,400,199 | 95.3% | 922 | 4.2309 | 0 | 0 |  |
| oab-h2-lc-haiku | 224 | 202,324,859 | 99.5% | 2,016 | 10.7107 | 0 | 0 |  |
| oab-h2-emb-4omini | 224 | 9,297,164 | 8.8% | 1,055 | 1.3341 | 0 | 0 |  |
| oab-h2-emb-haiku | 224 | 16,077,557 | 0.0% | 2,016 | 1.6088 | 0 | 0 |  |

A coluna US$ soma o `custo_usd` das linhas, que não inclui os embeddings (centavos, só no livro-caixa). As configurações de embedding com o Haiku de leitor também gastaram na OpenAI, nos embeddings, e esse gasto não aparece aqui.
