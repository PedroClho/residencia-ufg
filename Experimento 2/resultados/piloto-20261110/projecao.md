# Projeção de custo do piloto

Piloto com 10 pergunta(s) por configuração, as primeiras do primeiro contexto. As configurações sem memória rodam completas, e nelas o valor é o custo real. A projeção é `custo da 1ª linha × contextos + média das demais × (perguntas − contextos)`, com as respostas finais (sem erro de API) do provedor do leitor; os embeddings (centavos) ficam só no livro-caixa.

**A projeção supõe todos os contextos do tamanho do primeiro.** Na H1 da OAB o primeiro contexto (Ética) é menor que os do CPC, do CC e da CF, então as projeções do contexto longo na H1 saem baixas.

| configuração | provedor | linhas | contextos | perguntas | custo do piloto (US$) | projeção completa (US$) | status |
|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | openai | 10 | 1 | 100 | 0.1025 | 0.9411 | completa |
| direta-mhqa-bm25-4omini | openai | 10 | 1 | 100 | 0.0084 | 0.0846 | completa |
| direta-mcc-lc-4omini | openai | 10 | 5 | 500 | 0.0887 | 4.0731 | completa |
| direta-mcc-bm25-4omini | openai | 10 | 5 | 500 | 0.0582 | 2.8884 | completa |
| direta-mhqa-emb-4omini | openai | 10 | 1 | 100 | 0.0083 | 0.0830 | completa |
| oab-sem-memoria-4omini | openai | 3377 | 1 | 3377 | 0.1635 | 0.1635 | completa |
| oab-sem-memoria-haiku | anthropic | 3377 | 1 | 3377 | 0.1854 | 0.1854 | completa |
| oab-h1-lc-4omini | openai | 10 | 15 | 300 | 0.0688 | 1.9719 | completa |
| oab-h1-bm25-4omini | openai | 10 | 15 | 300 | 0.0086 | 0.2571 | completa |
| oab-h2-lc-4omini | openai | 10 | 1 | 224 | 0.1030 | 2.1080 | completa |
| oab-h2-bm25-4omini | openai | 10 | 1 | 224 | 0.0624 | 1.3971 | completa |
| oab-h1-emb-4omini | openai | 10 | 15 | 300 | 0.0085 | 0.2559 | completa |
| oab-h1-bm25-haiku | anthropic | 10 | 15 | 300 | 0.0096 | 0.2868 | completa |
| oab-h1-emb-haiku | anthropic | 10 | 15 | 300 | 0.0095 | 0.2850 | completa |
| oab-h1-bm25-4omini-k2 | openai | 10 | 15 | 300 | 0.0021 | 0.0641 | completa |
| oab-h1-bm25-4omini-k5 | openai | 10 | 15 | 300 | 0.0046 | 0.1366 | completa |
| oab-h1-bm25-4omini-k20 | openai | 10 | 15 | 300 | 0.0167 | 0.5022 | completa |
| oab-h1corpus-bm25-4omini | openai | 10 | 1 | 300 | 0.0099 | 0.2993 | completa |
| oab-h2-bm25-haiku | anthropic | 10 | 1 | 224 | 0.0741 | 1.6645 | completa |
| oab-h1-lc-haiku | anthropic | 10 | 15 | 300 | 0.1440 | 3.1863 | completa |
| oab-h2-lc-haiku | anthropic | 10 | 1 | 224 | 0.9738 | 10.7090 | completa |
| oab-h2-emb-4omini | openai | 10 | 1 | 224 | 0.0611 | 1.3650 | completa |
| oab-h2-emb-haiku | anthropic | 10 | 1 | 224 | 0.0735 | 1.6506 | completa |
| direta-mcc-emb-4omini | openai | 10 | 5 | 500 | 0.0607 | 3.0285 | completa |

## Total projetado por provedor

| provedor | projeção (US$) |
|---|---|
| openai | 19.6192 |
| anthropic | 17.9676 |
