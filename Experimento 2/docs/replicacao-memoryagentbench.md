# Replicação do MemoryAgentBench: RAG e contexto longo, nos dados do autor e na OAB

**Residência em IA (UFG) · Experimento 2 · Pedro Coelho**

**Versão parcial, de 2026-11-10.** A OpenAI ficou sem saldo pré-pago no meio da execução. Tudo o que depende do GPT-4o-mini e ainda não rodou está marcado como **pendente**, e a seção 9 lista os comandos que completam o experimento.

O Experimento 1 comparou memória passiva e ativa numa conversa sintética pequena, em que o contexto longo bastava. O Experimento 2 testa duas afirmações de um artigo revisado por pares, o MemoryAgentBench (Hu, Wang e McAuley, ICLR 2026), para saber se o autor estava certo:

- **H1.** O RAG vence o contexto longo quando é preciso achar um trecho preciso (achado 1 do artigo, §4.2).
- **H2.** O contexto longo vence o RAG quando é preciso aprender com exemplos (achado 2, §4.2).

Cada afirmação passa por duas etapas. A **direta** usa o código, os dados e o protocolo do autor, com o GPT-4o-mini, para ver se o número do artigo se repete. A **conceitual** leva a mesma afirmação para as questões da OAB, com o corpus de leis do projeto Atlas, para ver se ela vale fora dos dados do autor; nela entram o GPT-4o-mini e o Haiku 5.5. Rodar só na OAB não separaria "o autor errou" de "lei é diferente do que ele testou"; a etapa direta separa.

- Desenho, com as regras de decisão: [`superpowers/specs/2026-11-10-replicacao-memoryagentbench-design.md`](superpowers/specs/2026-11-10-replicacao-memoryagentbench-design.md)
- Código do autor e cada adaptação: [`../replica/autor/NOTICE.md`](../replica/autor/NOTICE.md)
- Resultados: [`../resultados/20261110-direta/tabela.md`](../resultados/20261110-direta/tabela.md), [`../resultados/20261110-oab/tabela.md`](../resultados/20261110-oab/tabela.md) e o piloto em [`../resultados/piloto-20261110/`](../resultados/piloto-20261110/)

**O resultado em poucas linhas.** A H1 direta está completa e foi **confirmada**: no MH-Doc QA, com o GPT-4o-mini, o RAG BM25 acertou 56,0% [46,0; 66,0] e o contexto longo 42,0% [32,0; 52,0], e a diferença pareada, +14,0 pp [+2,0; +26,0], exclui zero na direção do artigo. Os dois números também **reproduzem** o artigo: 43,0 e 56,0 caem dentro dos nossos intervalos. As outras três comparações primárias estão **pendentes**. A API da OpenAI passou a recusar as chamadas por falta de saldo pré-pago (HTTP 429, "You have no credits remaining") com US$ 3,29 gastos, bem abaixo da trava de US$ 20 do painel, e a rodada foi interrompida sem perder nenhuma resposta paga. Da H2 direta há só uma parte: o contexto longo reproduz o artigo no BANKING77 (89,0% [83,0; 95,0] contra 93,0), mas o BM25 tem 10 das 500 perguntas, e não há veredito. As duas primárias da OAB usam o GPT-4o-mini e não rodaram. Na OAB rodaram as linhas de base sem memória (GPT-4o-mini 58,9%, Haiku 70,7%, nas 3.377 questões) e as secundárias do Haiku, que não dão veredito. Nelas, o BM25 e o contexto longo ficaram a menos de 3 pontos um do outro, com intervalos que contêm zero nas duas afirmações (BM25 − contexto longo: H1 +0,7 pp [−2,7; +4,0]; H2 −2,7 pp [−7,6; +2,2]), e o contexto longo custou de 6,6 a 14 vezes mais. A memória acrescentou no máximo 5,0 pontos a um Haiku que, sem memória, já acerta 69,3% e 69,2% dessas questões; trocar o GPT-4o-mini pelo Haiku, sem memória, vale +11,8 pp [+10,1; +13,5].

## 1. O que foi replicado e o que não foi

Situação de cada comparação em 2026-11-10:

| Comparação | Configurações | Perguntas | Situação |
|---|---|---|---|
| H1 direta (primária) | contexto longo × RAG BM25, GPT-4o-mini, MH-Doc QA | 100 | **completa**, com veredito e teste de reprodução |
| H2 direta (primária) | contexto longo × RAG BM25, GPT-4o-mini, as cinco MCC | 500 | parcial: contexto longo 170 de 500 (mais 1 erro de API a refazer), BM25 10 de 500. **Pendente** |
| Reprodução do RAG por embedding no MH-QA | `text-embedding-3-small`, leitor GPT-4o-mini | 100 | 10 de 100. **Pendente** |
| Reprodução do RAG por embedding nas MCC | idem | 500 | 0 de 500. **Pendente** (a última da fila) |
| H1 OAB (primária) | contexto longo × RAG BM25, GPT-4o-mini | 300 | 0 de 300. **Pendente** |
| H2 OAB (primária) | idem | 224 | 0 de 224. **Pendente** |
| Linhas de base sem memória | GPT-4o-mini e Haiku 5.5 | 3.377 cada | completas |
| Secundárias do Haiku | contexto longo e RAG BM25, na H1 e na H2 da OAB | 300 e 224 | completas |
| Secundárias da OpenAI na OAB | RAG por embedding na H1 e na H2, ablação de k (2, 5 e 20), BM25 no corpus inteiro | 300 ou 224 | **pendentes** |
| Haiku com RAG por embedding | H1 e H2 da OAB | 300 e 224 | **pendentes** (os embeddings vêm da OpenAI) |

### Por que parou

A rodada da etapa direta começou pelas cinco configurações da OpenAI. Depois do MH-QA completo e de 170 perguntas das MCC no contexto longo, a API devolveu 429 com a mensagem "You have no credits remaining". O livro-caixa marcava US$ 3,29 (US$ 3,2867). O saldo pré-pago da conta acabou muito antes da trava de US$ 20 que o Pedro pôs no painel e do teto de US$ 19,50 do livro-caixa: a trava do painel limita o gasto, mas não garante que haja saldo. Cada resposta é gravada logo depois de chegar, e a retomada refaz só as linhas com erro de API, então tudo o que foi pago está nos arquivos.

O cliente do experimento reconhecia a falta de saldo pelo código `insufficient_quota`, e essa resposta não foi reconhecida; a rodada foi parada à mão. Desde o commit `fff61c6` ele reconhece também o texto da mensagem e para a configuração na hora (`LimiteDoPainel`), em vez de esperar 5 × 60 s por pergunta. Pôr crédito na conta é ação do Pedro.

Enquanto a OpenAI estava sem crédito, rodaram as quatro configurações que usam só a Anthropic: o BM25 e o contexto longo do Haiku, na H1 e na H2. Os tetos são por provedor, então isso não tirou orçamento de nenhuma primária. A decisão foi do Claude (seção 6.2). A Anthropic fechou em US$ 18,33 dos US$ 29,50 do livro-caixa. As configurações do Haiku com RAG por embedding esperam, porque os embeddings vêm da OpenAI.

### O que ficou fora por desenho

- **Cinco dos seis modelos de contexto longo do artigo.** O Claude 3.7 Sonnet (saiu em 2026-02-19) e o Gemini 2.0 Flash (2026-06-01) já não existem na API; o GPT-5-mini sai em 2026-12-11; o GPT-4o custaria sozinho mais que o teto; a configuração mais barata do GPT-4.1-mini passaria de US$ 4 (Correção 11 do desenho). Fica o GPT-4o-mini, que é também o leitor de todos os agentes RAG do artigo. O Haiku 5.5 entra só na OAB, como a extensão atual.
- **A metade "compreensão de longo alcance" do achado 2.** A tarefa de resumo depende do GPT-4o como juiz, e o Detective QA tem só 71 perguntas.
- **O achado 3, esquecimento seletivo**, por decisão do Pedro. Na OAB também não haveria como testá-lo: o corpus não guarda versões de lei.
- **Os agentes agênticos** (MemGPT, MIRIX, Self-RAG) **e os RAG com estrutura** (GraphRAG, HippoRAG, Mem0 e outros), pelo orçamento e porque as duas afirmações não dependem deles.
- **As ablações do artigo** (tamanho de chunk e top-k), por decisão do Pedro. A única ablação do experimento é a de k na OAB H1, que está **pendente**.

## 2. As regras de decisão, como foram registradas

O desenho registrou, antes de qualquer rodada, esta regra (seção 2 do desenho, "Regra de decisão (registrada antes de rodar)"):

> - **Uma comparação primária por afirmação e etapa**, quatro ao todo (H1 direta, H2 direta, H1 OAB, H2 OAB). A primária é sempre o par que o artigo reporta: **GPT-4o-mini com contexto longo × GPT-4o-mini com RAG BM25**. Só a primária dá veredito.
> - Na H2 direta, as cinco MCC entram juntas na primária: 500 perguntas, com o bootstrap reamostrando dentro de cada dataset (item 7 de "Correções depois da leitura do código do autor (2026-11-10)"). O resultado por dataset é secundário.
> - **Secundárias** (RAG por embedding × contexto longo, os pares do Haiku, cada dataset separado, a ablação de k) são reportadas com intervalo, sem veredito. Se divergirem da primária, a divergência vai para o documento e é discutida, mas não muda o veredito.
> - Sem correção para múltiplas comparações entre as quatro primárias: cada uma responde uma pergunta diferente, registrada antes.
> - Comparações sempre pareadas: as mesmas questões em todas as configurações.
> - Intervalo de 95% por bootstrap pareado (10.000 reamostragens) para a diferença de acurácia.
> - **Confirmada** se o intervalo exclui zero na direção do artigo. **Contrariada** se exclui zero na direção oposta. **Inconclusiva** se contém zero. Inconclusivo é resultado e vai para o documento como está.
> - Na etapa direta há um segundo teste: o número do artigo cai dentro do nosso intervalo de 95%? Se cai, a medida foi reproduzida.
> - Com 100 perguntas, o intervalo de uma acurácia isolada tem uns ±9 pontos; a diferença pareada é mais estreita. Diferenças de 2 a 3 pontos não são detectáveis neste orçamento, e o documento diz isso.

A ordem dos commits mostra que a regra veio antes dos dados:

| Commit | Hora (2026-11-10) | O que entrou |
|---|---|---|
| `6e8e707` | 01:36 | desenho original, já com a regra (a H2 direta tinha só BANKING77 e CLINC150) |
| `577fb2e` | 02:52 | desenho com as "Correções depois da leitura do código do autor"; a seção da regra fica com o texto atual |
| `bf903e9` | 04:41 | primeiro commit em `resultados/`: snapshot, reclassificação de matéria e livro-caixa, sem nenhuma resposta dos agentes |
| `dd3d6b2` | 06:34 | piloto e "Decisões do piloto" |
| `6003a15` | 07:22 | rodada direta parcial e linhas de base da OAB |
| `fc1b412` | 08:07 | rodada da OAB só com o Haiku |

A seção da regra tem o mesmo texto em `577fb2e` e na versão atual do desenho (conferido com `git diff`). Entre `6e8e707` e `577fb2e` mudou uma linha: a H2 direta passou de duas MCC para as cinco (Correção 7), porque nos dois datasets do desenho a diferença do artigo é de 4 pontos (93 × 89), que 200 perguntas não detectam com segurança; o Pedro aprovou. Depois do piloto houve uma decisão que muda como a H2 direta é medida, sem mexer na regra: a leitura flexível de `label:` como métrica das MCC (seção 5.2). Ela foi tomada pelo Claude, a partir do formato das respostas de 10 perguntas do piloto, antes da rodada completa.

## 3. Método

### 3.1 O código é o do autor

O código dos três agentes (contexto longo, RAG BM25 e RAG por embedding), os templates de prompt e as métricas vêm do repositório do autor, de licença MIT, no commit fixado `538026089d1a8a8eff05121d0db89b388f360eba`. Os dados vêm do Hugging Face `ai-hyz/MemoryAgentBench`, revisão `7ea066982b140a19337e17e60d45d4076e042faf`. O `templates.py` é idêntico byte a byte (um teste confere o sha256). O `metricas.py` e o `agente.py` têm só as funções e os métodos usados, com os nomes e os corpos do autor. As adaptações, todas descritas no `NOTICE.md`:

1. **Toda chamada de LLM passa pelo cliente com teto de gasto** (`replica.clientes`), que registra no livro-caixa os tokens normais, cacheados e escritos no cache. No contexto longo, o contexto vai como prefixo cacheável e a pergunta como mensagem; o texto enviado é o mesmo do autor. A temperatura fica no cliente (0,7 na OpenAI).
2. **O Haiku recebe a mensagem de sistema.** O caminho do Claude no código do autor a omitia; o Haiku não está no artigo, e fica com o mesmo prompt dos agentes da OpenAI.
3. **BM25 sem LangChain:** `rank_bm25.BM25Okapi` com tokenização por `str.split()`, sem minúsculas, que é o que o `BM25Retriever` do LangChain faz por dentro.
4. **Embedding sem LangChain, FAISS nem Azure:** produto interno entre vetores normalizados com numpy, que dá a mesma ordem do FAISS L2 do autor sobre os vetores do `text-embedding-3-small`. O caminho do autor exigia Azure; aqui vai pelo cliente normal da OpenAI.
5. **A extração da consulta do RAG ganha um padrão em português** (`Agora responda à questão:`), depois dos dois do autor, para a OAB.
6. **Templates por uma função injetável**, para a OAB plugar os seus; o padrão é o `get_template` do autor.
7. **Sem torch, transformers, LangChain, Azure nem gravação em `./outputs/`.** O trecho recuperado volta para o rodador, que o guarda na linha da resposta.

Nas métricas, o `calculate_metrics` deixou de calcular o ROUGE, que o experimento não usa, e o `post_process` ficou só com os ramos das tarefas usadas. Foram mantidos de propósito os comportamentos do autor que um leitor estranharia: o carimbo de hora no template de memorização, o truncamento do contexto longo a cada pergunta, o descarte do trecho mais antigo no RAG quando a memória passa do limite, e o trecho vazio que o fatiador pode gerar.

### 3.2 Parâmetros

| Parâmetro | Valor | Origem |
|---|---|---|
| top-k do RAG | 10 | autor |
| chunk do contexto longo | 4096 tokens | autor |
| chunk do RAG | 512 no MH-QA e na OAB H1; 4096 nas MCC e na OAB H2 | autor; na OAB, por analogia (Correção 8) |
| teto de saída | 50 tokens no MH-QA; 20 nas MCC e na OAB | autor (Correção 2) |
| `input_length_limit` / `buffer_length` | contexto longo do GPT-4o-mini 128.000 / 4.000; BM25 10.000.000 / 200; embedding 300.000 / 15.000; contexto longo do Haiku 1.000.000 / 480.000 | autor; o Haiku segue a convenção do autor para o Claude (Correção 12) |
| amostragem | temperatura 0,7 na OpenAI; o Haiku usa a padrão, porque a API recusa `temperature` | autor (Correção 1) |
| métrica | MH-QA: `substring_exact_match`. MCC: leitura flexível de `label:` (primária) e `exact_match` estrito (ao lado). OAB: letra extraída por regex | README do autor; Decisões do piloto |
| cache | ligado nas duas APIs | o artigo calcula os custos com cache (Apêndice I) |

O cache é o contrário da regra do Experimento 1, que o proibiu para a contagem de tokens ter um significado só. Aqui ele é obrigatório: sem cache, a Anthropic passaria de US$ 140. No piloto, as chamadas 2 a 10 das seis configurações de contexto longo vieram com 99% a 100% da entrada cacheada.

**Contexto longo não é contexto inteiro.** O agente de contexto longo do autor trunca a memória para caber em `input_length_limit − buffer_length` e guarda o fim. Com o GPT-4o-mini, isso dá uns 124 mil tokens: no MH-QA, cujo contexto tem ~421 mil, ele vê menos de um terço. As 100 perguntas do contexto longo somaram 12.402.749 tokens de entrada, ~124 mil por pergunta. O RAG BM25 indexa tudo. É o protocolo do artigo, e pesa na leitura da H1 (seção 5.1).

### 3.3 Adaptações para a OAB

- **Prompts em português**, que imitam os templates do autor para o `ruler_qa` (H1) e o `in_context_learning` (H2), copiados de [`../replica/templates_oab.py`](../replica/templates_oab.py):

```python
SYSTEM_OAB = "Você é um assistente prestativo que lê o contexto e o memoriza para consultas futuras."
INSTRUCAO_LETRA = "Responda apenas com a letra da alternativa correta (A, B, C ou D) e não escreva mais nada."

# H1, pergunta ao contexto longo (o RAG troca "Questão:" por "Agora responda à questão:")
"Responda à questão com base nos documentos memorizados. " + INSTRUCAO_LETRA + " \n\nQuestão: {question} \n\n Resposta:"

# H2, pergunta ao contexto longo (o RAG usa "Questão:{question}")
"Use os exemplos do contexto, questões da prova da OAB com o gabarito, para responder à nova questão. Responda apenas \"Gabarito: {{letra}}\" e nada mais. \n\n{question} \n\n Gabarito:"
```

- **A consulta do RAG.** Na H1, a consulta é o texto depois de "Agora responda à questão:", isto é, o enunciado com as alternativas. Na H2, como nas MCC do autor, nenhum padrão casa, e a consulta é a mensagem inteira.
- **Resposta por letra.** A letra é extraída por regex, e letra inválida conta como erro. No piloto, nenhuma configuração teve letra inválida.
- **Haiku sem raciocínio** (`thinking: {type: "disabled"}`), para ficar comparável ao GPT-4o-mini, que não raciocina.
- **H1: a lei da matéria.** 300 questões das provas de 2018 a 2023, sorteadas (semente 42) e estratificadas pela matéria reclassificada (seção 3.5), sem Filosofia e Internacional. Cada matéria aponta para uma ou mais leis (Civil → CC, Processo Civil → CPC, Ética → EAOAB, Regulamento Geral e CED, e assim por diante), o que dá 15 contextos; o maior tem 264.994 tokens `o200k` (o CPC). O contexto longo e o RAG recebem a mesma lei e diferem só em como a leem, que é o ponto do artigo. O contexto longo do Haiku vê cada lei inteira. O do GPT-4o-mini verá só o fim das leis que passam de ~124 mil tokens (CPC, CC, CF e CLT).
- **H1, variante "corpus inteiro":** o BM25 sobre todas as leis (1.755.201 tokens `o200k`), só com BM25, porque o agente de embedding do autor guardaria só os últimos ~285 mil tokens (Correção 9). **Pendente.**
- **H2: as provas antigas como exemplos.** A memória são 3.057 questões de até 2023, em ordem cronológica, cada uma seguida de "Gabarito: X" (775.322 tokens `o200k`). O teste são as 224 questões de 2024 e 2025 (229 antes dos filtros), depois do corte de treino do GPT-4o-mini (outubro de 2023, segundo fontes secundárias). Sai da memória toda questão antiga com enunciado quase igual ao de uma questão de teste (razão do `SequenceMatcher` acima de 0,8), e saem as duplicatas conhecidas (prova 2018-01 ≈ 2018-25, `2022-02-3` e `2022-02-3-fixed`). O contexto longo vê as questões mais recentes que cabem: no Haiku, ~520 mil tokens `o200k` (~890 mil do tokenizador dele), cerca de dois terços da memória; no GPT-4o-mini, ~124 mil, cerca de um sexto. O RAG usa chunks de 4096 tokens, e cada busca traz ~160 questões antigas (~41 mil tokens).
- **BM25 em português com a tokenização do autor.** O `str.split()` não passa o texto para minúsculas e deixa a pontuação grudada na palavra: "Lei", "lei" e "lei," são termos diferentes, e não há stemmer. O desenho (§5) sugeria um stemmer de português e deixava a escolha para quem construísse (§12); o plano manteve a tokenização do autor, por fidelidade. Isso pode prejudicar o RAG na OAB (seção 5.4).
- **Linhas de base sem memória:** as 3.377 questões válidas de todos os anos (as 3.395 do banco menos 18 com menos de quatro alternativas), nos dois modelos.

### 3.4 Vazamento de gabarito

A tabela `embeddings` do banco do Atlas tem 2.210 linhas de questões de 2010 a 2018 com o gabarito. Nada do experimento usa essas linhas: o RAG da H1 indexa só os artigos de lei, e os embeddings são gerados de novo, localmente. A revisão final do código conferiu, e as contas foram refeitas para este documento sobre os arquivos de `dados/oab/`:

- o par mais parecido entre a memória da H2 (3.057 questões) e as 224 questões de teste tem Jaccard de 4-gramas de palavras de 0,053;
- o par mais parecido dentro da amostra da H1 (300 questões) tem 0,052;
- nenhum dos 15 contextos da H1 contém o começo (120 caracteres) de nenhuma das 300 questões;
- as questões de teste da H2 estão todas na linha de base sem memória, e a H1 e a H2 não têm nenhuma questão em comum.

### 3.5 Reclassificação de matéria

Os rótulos de matéria do banco são ruins depois de 2018, e é o rótulo que decide qual lei o agente recebe na H1. As 953 questões de 2018 a 2023 que sobram depois dos filtros foram reclassificadas pelo GPT-4o-mini, com temperatura 0. A regra do plano: conferir à mão uma amostra de 30 e aprovar com até 3 erros.

- **Rodada 1, reprovada:** 4 erros em 30 ([`conferencia-materias-rodada1.md`](conferencia-materias-rodada1.md)). O modelo puxava para o ECA questões de Direito Civil que só têm um menor ou uma pessoa com deficiência na história, e tirava de Administrativo questões sobre servidor público e Lei Anticorrupção. Nas 30 linhas, a matéria certa coincidia com o bloco da prova da FGV pelo número da questão (1 a 8 Ética, 9 e 10 Filosofia, 11 a 17 Constitucional, e assim por diante).
- **Rodada 2, aprovada:** o prompt passou a informar o número da questão e a ordem dos blocos da FGV. Numa amostra nova (semente 43), 0 erro em 30 ([`conferencia-materias.md`](conferencia-materias.md)).

Restaram 65 das 953 questões (6,8%) com matéria fora do bloco da FGV, nenhuma na amostra da rodada 2, várias repetindo erros da rodada 1. Como o contexto longo e o RAG recebem a mesma lei em cada questão, esse ruído afeta os dois braços igualmente e não enviesa a comparação pareada; ele só a deixa mais ruidosa. **As duas conferências foram feitas pelo Claude, porque a sessão rodou sem o Pedro, e são provisórias até o Pedro reconferir.**

### 3.6 Análise

As comparações são pareadas por `pergunta_id`. A diferença é sempre RAG − contexto longo: a H1 espera diferença positiva, a H2 negativa. O intervalo de 95% vem de um bootstrap pareado com 10.000 reamostragens, que nas MCC reamostra dentro de cada dataset. Linhas com erro contam como erradas. Configuração em que falta resposta final para alguma pergunta esperada fica "parcial" e sem veredito nem teste de reprodução.

O comando `analise` gera as primárias, a reprodução, as acurácias, os custos e o gráfico, mas não os pares secundários. Os pares secundários do Haiku (seção 4.2) foram calculados com a mesma função (`replica.analise.bootstrap_pareado`, semente 42) sobre os JSONL da pasta da OAB; as contas foram refeitas para este documento e deram os mesmos números.

## 4. Resultados

### 4.1 Etapa direta

Rodada em `resultados/20261110-direta/`. As linhas da H1 e da H2 direta da tabela de primárias, coladas de `tabela.md` sem edição (as linhas da OAB dessa tabela dizem "não rodada"):

| Comparação | n | Contexto longo | RAG BM25 | Diferença | Veredito | Nota |
|---|---|---|---|---|---|---|
| H1 direta | 100 | 42.0% [32.0%; 52.0%] | 56.0% [46.0%; 66.0%] | +14.0 pp [+2.0; +26.0] | confirmada |  |
| H2 direta | 10 | 60.0% [30.0%; 90.0%] | 80.0% [50.0%; 100.0%] | +20.0 pp [+0.0; +50.0] | sem veredito | parcial (10 de 500) |

- H2 direta: 161 pergunta(s) de `direta-mcc-lc-4omini` e 0 de `direta-mcc-bm25-4omini` descartadas por não estarem nas duas configurações (ou fora das perguntas esperadas).
- H2 direta, `exact_match` estrito (literal do README), nas mesmas 10 perguntas: contexto longo 0.0%, RAG BM25 0.0%.

A seção de reprodução, colada sem edição. Nas MCC, "Nossa acurácia" é a leitura flexível de `label:`, e a coluna do estrito é o `exact_match` literal do README:

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

Custos, colados sem edição:

| Configuração | n | Tokens de entrada | % cacheado | Tokens de saída | US$ | Erros de API | Recusas/vazias | Nota |
|---|---|---|---|---|---|---|---|---|
| direta-mhqa-lc-4omini | 100 | 12,402,749 | 96.9% | 986 | 0.9596 | 0 | 0 |  |
| direta-mhqa-bm25-4omini | 100 | 553,947 | 0.0% | 1,136 | 0.0838 | 0 | 0 |  |
| direta-mcc-lc-4omini | 171 | 18,186,236 | 96.9% | 680 | 1.4062 | 1 | 0 | parcial (170 de 500) |
| direta-mcc-bm25-4omini | 10 | 414,375 | 12.9% | 40 | 0.0582 | 0 | 0 | parcial (10 de 500) |
| direta-mhqa-emb-4omini | 10 | 54,743 | 0.0% | 147 | 0.0083 | 0 | 0 | parcial (10 de 100) |

As 10 primeiras linhas de cada configuração direta vêm do piloto: mesmo código, mesmo modelo e mesmas perguntas; só o carimbo de hora do template de memorização difere.

### 4.2 OAB

Rodada em `resultados/20261110-oab/`. As primárias da H1 OAB e da H2 OAB (GPT-4o-mini, contexto longo × BM25) estão **pendentes**: na `tabela.md`, as quatro aparecem como "não rodada".

Acurácias das configurações que rodaram, coladas sem edição (seção "Secundárias" da `tabela.md`):

| Configuração | n | Acurácia | Nota |
|---|---|---|---|
| oab-sem-memoria-4omini | 3377 | 58.9% [57.3%; 60.6%] |  |
| oab-sem-memoria-haiku | 3377 | 70.7% [69.2%; 72.3%] |  |
| oab-h1-bm25-haiku | 300 | 74.3% [69.3%; 79.3%] |  |
| oab-h2-bm25-haiku | 224 | 70.5% [64.3%; 76.3%] |  |
| oab-h1-lc-haiku | 300 | 73.7% [68.7%; 78.3%] |  |
| oab-h2-lc-haiku | 224 | 73.2% [67.4%; 79.0%] |  |

Pares secundários, sem veredito. Cada linha compara a configuração a com a b nas mesmas questões; a coluna da direita é b − a com o intervalo de 95% do bootstrap pareado. Nas linhas "sem memória × ...", a linha de base é restrita às questões da H1 (300) ou do teste da H2 (224):

| Par (a × b) | n | a | b | b − a (IC 95%) |
|---|---|---|---|---|
| H1 OAB Haiku: contexto longo × BM25 | 300 | 73.7% | 74.3% | +0.7 pp [-2.7; +4.0] |
| H2 OAB Haiku: contexto longo × BM25 | 224 | 73.2% | 70.5% | -2.7 pp [-7.6; +2.2] |
| H1 OAB Haiku: sem memória × contexto longo | 300 | 69.3% | 73.7% | +4.3 pp [-0.3; +9.0] |
| H1 OAB Haiku: sem memória × BM25 | 300 | 69.3% | 74.3% | +5.0 pp [+0.3; +10.0] |
| H2 OAB Haiku: sem memória × contexto longo | 224 | 69.2% | 73.2% | +4.0 pp [-0.9; +8.9] |
| H2 OAB Haiku: sem memória × BM25 | 224 | 69.2% | 70.5% | +1.3 pp [-4.0; +6.2] |
| Sem memória: GPT-4o-mini × Haiku (3.377) | 3377 | 58.9% | 70.7% | +11.8 pp [+10.1; +13.5] |

Custos, colados sem edição. A coluna US$ não inclui os embeddings, que ficam só no livro-caixa:

| Configuração | n | Tokens de entrada | % cacheado | Tokens de saída | US$ | Erros de API | Recusas/vazias | Nota |
|---|---|---|---|---|---|---|---|---|
| oab-sem-memoria-4omini | 3377 | 1,076,206 | 0.0% | 3,429 | 0.1635 | 0 | 0 |  |
| oab-sem-memoria-haiku | 3377 | 1,803,542 | 0.0% | 10,131 | 0.1854 | 0 | 0 |  |
| oab-h1-bm25-haiku | 300 | 2,978,557 | 0.0% | 900 | 0.2983 | 0 | 0 |  |
| oab-h2-bm25-haiku | 224 | 16,239,481 | 0.0% | 2,016 | 1.6250 | 0 | 0 |  |
| oab-h1-lc-haiku | 300 | 59,400,199 | 95.3% | 922 | 4.2309 | 0 | 0 |  |
| oab-h2-lc-haiku | 224 | 202,324,859 | 99.5% | 2,016 | 10.7107 | 0 | 0 |  |

As linhas de base sem memória são as do piloto, copiadas para a pasta da rodada: mesma configuração, mesmo código e mesmas questões. O Haiku sem memória acertou 69,2% das 224 questões de teste da H2, abaixo dos 90% que fechariam o portão do contexto longo dele na H2 (acima disso, o modelo provavelmente já teria visto as questões). A projeção do piloto para a Anthropic ficou em ~US$ 18, abaixo dos US$ 24 que mandariam reduzir a memória do Haiku na H2 para 600 mil tokens, então a variante rodada é a de 1M.

O gráfico qualidade × tokens da OAB H1:

![OAB H1: acurácia do Haiku com contexto longo contra os tokens de entrada médios por pergunta](../resultados/20261110-oab/qualidade_tokens.png)

Por enquanto ele tem um ponto só, o contexto longo do Haiku. A curva que motivou o gráfico, o BM25 do GPT-4o-mini com k = 2, 5, 10 e 20, e o ponto do contexto longo do GPT-4o-mini estão **pendentes**. O eixo x conta os tokens de cada modelo: o tokenizador do Haiku dá 1,71× mais tokens que o `o200k_base` do GPT-4o-mini no mesmo texto jurídico.

## 5. Leitura honesta

### 5.1 H1 direta: confirmada e reproduzida

O RAG BM25 venceu o contexto longo por 14,0 pontos no MH-Doc QA, e o intervalo exclui zero: pela regra registrada, a H1 direta está **confirmada**. Por pouco, porém: o limite inferior é +2,0, e é uma rodada só, com temperatura 0,7. Os dois números ficaram a 1 ponto (42,0 × 43,0) e a 0 ponto (56,0 × 56,0) dos do artigo. O teste de reprodução é frouxo, porque com 100 perguntas cada intervalo tem 20 pontos de largura, mas a proximidade dos números diz mais que o teste.

Duas observações mudam a leitura do "RAG vence":

- **O contexto longo não viu o contexto inteiro.** Com a janela de 128K do GPT-4o-mini, o agente do autor guarda os últimos ~124 mil tokens de um contexto de ~421 mil (seção 3.2). A comparação da H1 direta é, então, a de um RAG que indexa tudo contra um contexto longo que vê menos de um terço. É exatamente o que o artigo rodou, e por isso a replicação é fiel; mas o achado diz "o RAG vence quando o contexto não cabe na janela" com mais segurança do que diz "o RAG lê melhor um trecho preciso".
- **O RAG custou 11 vezes menos.** US$ 0,0838 contra US$ 0,9596, com 96,9% da entrada do contexto longo cacheada. Sem cache, a distância seria maior.

### 5.2 H2 direta: sem veredito

A H2 direta tem 10 perguntas pareadas, as primeiras do BANKING77, e nelas o BM25 (80,0%) ficou acima do contexto longo (60,0%), contra o artigo. Isso não diz nada: o intervalo da diferença é [+0,0; +50,0], e nas 100 perguntas do BANKING77 o contexto longo acertou 89,0%, o que faz dessas 10 um começo difícil. O veredito sai quando as 500 perguntas existirem nos dois braços.

O que já dá para dizer:

- **O contexto longo reproduz o artigo no BANKING77:** 89,0% [83,0; 95,0] contra 93,0. No CLINC150 há 70 de 100 respostas (85,9%, sem teste). A média parcial do contexto longo (87,7%, 171 linhas) não se compara aos 82,0 do artigo: vem só dos dois datasets em que o artigo dá 93,0, e o TREC grosso e o fino, em que o artigo dá 73,0 e 66,0, ainda não rodaram.
- **O `exact_match` estrito deu 0,0% em todas as medições das MCC até agora** (100 perguntas do BANKING77 e 71 do CLINC150 no contexto longo, 10 no BM25). O prompt do próprio autor pede `Only output "label: {label}"`, o GPT-4o-mini obedece ("label: 68"), e o `exact_match` estrito do README conta isso como erro. O código do autor não normaliza "label:" em lugar nenhum, então o estrito não pode ter gerado os 82,0 e 93,0 do artigo, e deixaria a H2 direta degenerada (0 × 0). A primária passou a ser a leitura flexível, com o estrito ao lado em toda tabela. A decisão veio do formato da resposta, com os dois braços em zero, e não de quem vencia; as duas leituras estão gravadas em cada linha, e a análise pode ser refeita com a outra sem rodar nada.
- **Os 82,0 × 82,4 do artigo.** A Tabela 3 dá 82,0 para o contexto longo do GPT-4o-mini na média das MCC, mas a média dos cinco números da Tabela 7 (93,0, 93,0, 87,0, 73,0 e 66,0) é 82,4. Para o BM25 (75,4) e o embedding (70,0), as duas tabelas batem. O teste de reprodução usa os 82,0 da Tabela 3; quando a média do contexto longo existir, o documento vai dizer se o intervalo contém os dois números ou só um.

### 5.3 OAB: o que as secundárias do Haiku mostram

As primárias da OAB estão pendentes, então **não há veredito na OAB**. As secundárias do Haiku não dão veredito pela regra; o que segue é leitura dos intervalos.

- **H1 (achar o trecho na lei):** o BM25 ficou +0,7 pp acima do contexto longo, com intervalo [−2,7; +4,0]. Na direção do artigo, mas sem diferença detectável. O contexto longo do Haiku vê cada lei inteira (a maior tem 264.994 tokens `o200k`, dentro do que ele guarda), ao contrário do GPT-4o-mini no MH-QA. E custou 14 vezes mais: US$ 4,23 contra US$ 0,30.
- **H2 (aprender com as provas antigas):** o contexto longo ficou 2,7 pontos acima do BM25 (−2,7 pp [−7,6; +2,2] na convenção RAG − contexto longo). Também na direção do artigo, também sem diferença detectável. Custou 6,6 vezes mais: US$ 10,71 contra US$ 1,63.
- **A memória ajuda pouco um modelo que já sabe.** O Haiku sem memória acerta 69,3% das questões da H1 e 69,2% das da H2. A memória somou de +1,3 a +5,0 pontos, e só um dos quatro intervalos exclui zero: o BM25 na H1, +5,0 pp [+0,3; +10,0]. Trocar de modelo pesou mais: sem memória, o Haiku acerta 11,8 pontos [+10,1; +13,5] a mais que o GPT-4o-mini nas 3.377 questões. As duas grandezas vêm de conjuntos diferentes (3.377 questões de todos os anos contra 300 e 224), então a comparação é de ordem de grandeza.
- **Um limite que vale registrar:** não se sabe se o Haiku 5.5 viu as provas de 2024 e 2025 no treino. O portão de 90% existia para o caso extremo, e ele ficou em 69,2%, perto dos 69,3% das provas de 2018 a 2023.

Se essas duas diferenças pequenas vão se repetir no GPT-4o-mini, só as primárias dirão. Uma expectativa, escrita antes de elas rodarem: o GPT-4o-mini vê só o fim das leis que passam de ~124 mil tokens (CPC, CC, CF e CLT). Se ele perder para o BM25 na H1 da OAB enquanto o Haiku empata, a janela fica como a explicação mais provável do "RAG vence"; se empatar também, não.

### 5.4 Peculiaridades achadas

- **O agente de embedding do autor esquece.** Com `input_length_limit` 300.000 e `buffer_length` 15.000, ele descarta os trechos mais antigos quando a memória passa de ~285 mil tokens. No MH-QA, indexa só os últimos dois terços do contexto, enquanto o BM25, com limite de 10 milhões, indexa tudo. Isso ajuda a explicar o 44,0 do embedding contra o 56,0 do BM25 no artigo. É o que o artigo rodou, então ficou (Correção 5). A nossa reprodução do embedding no MH-QA tem só 10 de 100 perguntas (50,0%, sem teste). O mesmo limite tirou do experimento a variante "corpus inteiro" com embedding, que guardaria só ~285 mil dos ~1,9 milhão de tokens das leis.
- **Os 82,0 × 82,4 do artigo** e **o `exact_match` estrito nas MCC**: seção 5.2.
- **Os modelos saem do ar.** Dos seis modelos de contexto longo do artigo, o Claude 3.7 Sonnet e o Gemini 2.0 Flash já não existem na API, e o GPT-5-mini sai em 2026-12-11, antes do fim deste experimento. O GPT-4o-mini, o modelo central da replicação, sai em 2027-01-14, e por isso todas as rodadas com a OpenAI precisam terminar até 2026-12-20. Uma replicação fiel do artigo inteiro já não é possível, e a desta parte dele tem prazo.
- **BM25 em português com a tokenização de um artigo em inglês.** O `str.split()` do autor, sem minúsculas e com a pontuação grudada, foi mantido por fidelidade (seção 3.3). Mesmo assim, o BM25 do Haiku ficou a +0,7 pp do contexto longo na H1. Um BM25 com stemmer de português seria outra configuração, não rodada.
- **O saldo pré-pago acabou antes da trava.** A trava do painel e o teto do livro-caixa protegem contra gastar demais, não contra a falta de saldo. A falta de saldo não perdeu dinheiro nem respostas, mas parou as primárias.

## 6. Limitações

### 6.1 Tabela

| Limitação | O que se viu | O que falta ou estenderia |
|---|---|---|
| OpenAI sem saldo pré-pago | parou com US$ 3,29: H2 direta parcial; primárias da OAB, secundárias da OpenAI e Haiku com embedding sem rodar | crédito e os comandos da seção 9, até 2026-12-20 |
| Uma rodada por configuração, temperatura 0,7 na OpenAI | intervalos de ~±10 pontos com 100 perguntas; a H1 direta foi confirmada com limite inferior de +2,0 | repetir as rodadas; com este orçamento, diferenças de 2 a 3 pontos não são detectáveis |
| O contexto longo do GPT-4o-mini vê ~124 mil tokens | no MH-QA, menos de um terço do contexto | é o protocolo do artigo; o Haiku, que vê as leis inteiras na OAB H1, dá o contraste |
| BM25 com `str.split()` em português | sem minúsculas nem stemmer | BM25 com stemmer de português, como configuração separada |
| Matéria reclassificada | 65 de 953 questões fora do bloco da FGV, com a mesma lei nos dois braços | o Pedro reconferir; uma variante com a matéria imposta pelo bloco |
| O corpus de leis é uma foto | CPC de uma edição de 2016, CLT atual, Lei 14.133 no lugar da 8.666; o gabarito segue a lei vigente na data da prova (Pacote Anticrime de 2019, Lei 14.133 de 2021) | um corpus com as versões das leis |
| O CPC do corpus tem um "artigo" de 12.841 tokens | provável defeito do parser, que entra igual nos dois braços | corrigir o parser no Atlas |
| Não se sabe se o Haiku viu as provas de 2024 e 2025 | sem memória, 69,2% no teste da H2 | — |
| Tokenizador do Haiku | 1,71× mais tokens que o `o200k_base`; o eixo do gráfico mistura as duas contagens | — |
| Custos das tabelas sem embeddings | as configurações do Haiku com embedding gastam também na OpenAI | o livro-caixa tem o total |
| Deduplicação com `SequenceMatcher` | o `autojunk` padrão torna a razão dependente da ordem dos textos | — |
| Livro-caixa × painéis | comparação **pendente** (diferença de até 10% aceita) | o Pedro comparar `resultados/livro_caixa.json` com os painéis |

### 6.2 Decisões tomadas pelo Claude durante a execução sem o Pedro

O plano foi executado pelo Claude numa sessão sem o Pedro. As rodadas reais estavam pré-autorizadas, dentro dos tetos e na ordem do plano. As decisões abaixo foram tomadas nessa sessão; cada uma está registrada no desenho ("Decisões do piloto") ou neste documento para o Pedro rever:

1. **A métrica das MCC:** a leitura flexível de `label:` como primária, com o estrito ao lado (seção 5.2).
2. **A reclassificação de matéria:** a rodada 1 foi reprovada pela regra do plano, o prompt passou a informar o número da questão e a ordem dos blocos da FGV, e a rodada 2 foi aprovada numa amostra nova. **As duas conferências foram feitas pelo Claude** e são provisórias.
3. **Os rótulos das 65 questões fora do bloco** ficaram como o GPT-4o-mini os deu, em vez de impor o bloco, porque o método aprovado pela regra foi esse e o ruído não enviesa a comparação pareada.
4. **A variante do Haiku na H2:** 1M, porque a projeção da Anthropic ficou abaixo de US$ 24.
5. **Linhas reaproveitadas do piloto:** as linhas de base sem memória completas e as 10 primeiras linhas de cada configuração direta, em vez de pagá-las de novo.
6. **A ordem depois da falta de crédito:** rodar as quatro configurações só da Anthropic antes das primárias da OpenAI, que não podiam rodar. Dentro da Anthropic, a ordem de prioridade foi mantida.
7. **Este documento** foi escrito com as partes da OpenAI pendentes, para ser atualizado quando o crédito voltar.

### 6.3 Pendências do Pedro

- Pôr crédito na OpenAI e rodar a seção 9.
- Reconferir as duas amostras de matéria ([`conferencia-materias.md`](conferencia-materias.md) e [`conferencia-materias-rodada1.md`](conferencia-materias-rodada1.md)). Se a rodada 2 for reprovada, é preciso refazer a reclassificação e a OAB H1.
- Comparar o livro-caixa com os painéis da OpenAI e da Anthropic.
- Rever as decisões da seção 6.2.

## 7. O que cada afirmação diz sobre o mapa da residência

O mapa da residência é a tabela "Quando usar cada abordagem" da semana 7, que se lê de cima para baixo até a primeira linha que resolve o caso: só LLM, contexto longo, RAG, memória acoplada, agente gerindo a memória. Este experimento mexe só nas três primeiras linhas. O contexto longo e o RAG simples são memórias passivas: é o código, e não o modelo, que decide o que entra no contexto. Os agentes que gerem a própria memória ficaram fora do escopo, então **nada aqui fala de memória passiva × ativa**. O que o experimento faz é afinar a fronteira entre as linhas "contexto longo" e "RAG", e o que já dá para dizer é isto:

- **H1, achar um trecho preciso.** Com o modelo e os dados do artigo, num contexto três vezes maior que a janela, o RAG BM25 venceu o contexto longo por 14 pontos e custou 11 vezes menos. Isso sustenta a linha do mapa que manda passar do contexto longo ao RAG quando o histórico não cabe. No Experimento 1, com ~10 mil tokens por chamada, o contexto longo acertou tudo; aqui, com ~421 mil e uma janela de 128K, ele perde. Na OAB, com o Haiku e as leis inteiras na janela, não houve diferença detectável de acerto, mas o contexto longo custou 14 vezes mais. Se isso se mantiver nas primárias, o argumento do RAG, quando a lei cabe na janela, é o custo e não o acerto.
- **H2, aprender com exemplos.** Ainda não há veredito na etapa direta. Na OAB, com o Haiku, o contexto longo ficou à frente na direção do artigo, sem diferença detectável, e as provas antigas acrescentaram no máximo 4,0 pontos ao modelo sem memória. Uma explicação possível, não testada: nas MCC, o significado de cada rótulo só existe nos exemplos do contexto, e sem eles o modelo não tem como acertar; na OAB, a resposta depende de lei que o modelo já conhece, e as questões antigas são exemplos de outras questões, não do mesmo mapeamento.
- **A primeira linha do mapa pesou mais que as outras na OAB.** O Haiku sem memória acertou 70,7%, e trocar o GPT-4o-mini pelo Haiku valeu +11,8 pontos, mais que qualquer ganho de memória medido para o Haiku (no máximo +5,0). Na OAB, a escolha do modelo veio antes da escolha da memória.

## 8. Como rodar

Requisitos: Python 3.12 e [`uv`](https://docs.astral.sh/uv/). As chaves ficam em `.env` (fora do git): `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `OPENAI_API_KEY` e `ANTHROPIC_API_KEY`.

```
uv sync                                                      # ambiente
uv run pytest                                                # gate de testes (offline, sem API)
uv run python -c "import nltk; nltk.download('punkt_tab')"   # preparo único do fatiador do autor
uv run python -c "import tiktoken as t; t.get_encoding('o200k_base'); t.get_encoding('cl100k_base')"
                                                             # preparo único: aquece o cache do tiktoken

PYTHONUTF8=1 uv run python -m replica.snapshot               # banco do Atlas (só leitura) -> dados/     grátis
PYTHONUTF8=1 uv run python -m replica.reclassificar          # matéria 2018-2023         GASTA ~US$ 0,10 OpenAI
PYTHONUTF8=1 uv run python -m replica.oab                    # dados/oab/*.json a partir do snapshot     grátis
PYTHONUTF8=1 uv run python -m replica.rodar --todas --piloto 10 --rodada resultados/piloto-AAAAMMDD
                                                             # GASTA ~US$ 1,50 em cada provedor
PYTHONUTF8=1 uv run python -m replica.rodar NOMES... --rodada resultados/AAAAMMDD-ETAPA
                                                             # GASTA
PYTHONUTF8=1 uv run python -m replica.analise resultados/AAAAMMDD-ETAPA   # tabela.md e gráfico  grátis
```

- **Testes.** `uv run pytest` usa clientes falsos e nunca chama a API. Precisa do cache do tiktoken aquecido uma vez.
- **Toda rodada real gasta dinheiro.** O gasto acumulado fica em `resultados/livro_caixa.json`, que é versionado: nunca `git stash`, `git checkout -- .` nem `git reset --hard` com ele modificado.
- **Nomes ou `--todas`.** `--rodada` é obrigatório. Com nomes, roda só esses, sempre na ordem de prioridade. Sem nomes, o comando sai com erro; rodar todas exige `--todas` explícito, que numa pasta só da OAB refaria a etapa direta.
- **Nunca dois rodadores ao mesmo tempo.** O `rodar` e o `reclassificar` pegam a trava `resultados/.rodador.lock` e se recusam a começar se ela existe. Se nenhum processo estiver vivo (o processo caiu ou foi morto), apague o arquivo.
- **Retomada.** Rodar o mesmo comando de novo refaz só as linhas com erro de API. Repita até `erros_api=0`.
- **Teto e saldo.** O livro-caixa recusa a chamada que passaria do teto (US$ 19,50 na OpenAI, US$ 4 deles de reserva, liberados com `--usar-reserva`; US$ 29,50 na Anthropic). A falta de saldo ou o limite do painel param a configuração na hora.
- **Windows.** Com a saída redirecionada, os acentos saem trocados; defina `PYTHONUTF8=1` (no PowerShell, `$env:PYTHONUTF8 = "1"`).

## 9. Como completar

Quando a OpenAI tiver crédito, e antes de 2026-12-20:

0. **Antes.** Para que o teto do livro-caixa, e não o saldo, seja o limite, o saldo precisa cobrir pelo menos US$ 16,21 (o teto de US$ 19,50 com a reserva menos os US$ 3,29 já gastos). A projeção do piloto para todas as configurações da OpenAI foi de US$ 19,62, acima do teto efetivo: o que não couber sai do fim da fila, a reprodução das MCC por embedding primeiro, e o documento registra. Confira também que não existe `resultados/.rodador.lock` nem rodador vivo.
1. **Terminar a etapa direta.** As duas do MH-QA estão completas e são puladas; o comando refaz o erro de API do contexto longo das MCC e completa o resto. Repita até `erros_api=0`.

   ```
   PYTHONUTF8=1 uv run python -m replica.rodar direta-mhqa-lc-4omini direta-mhqa-bm25-4omini direta-mcc-lc-4omini direta-mcc-bm25-4omini direta-mhqa-emb-4omini --rodada resultados/20261110-direta
   PYTHONUTF8=1 uv run python -m replica.analise resultados/20261110-direta
   ```

   Aplique a regra de reprodução: se os números do artigo caírem dentro do intervalo nas quatro primárias (o MH-QA já reproduz; faltam as médias das MCC no contexto longo e no BM25), a reserva fica liberada. Se não, abre a janela de depuração da Task 13 do plano (até US$ 4, com `--usar-reserva`).
2. **As quatro primárias da OAB.**

   ```
   PYTHONUTF8=1 uv run python -m replica.rodar oab-h1-lc-4omini oab-h1-bm25-4omini oab-h2-lc-4omini oab-h2-bm25-4omini --rodada resultados/20261110-oab
   ```

3. **As secundárias da OpenAI e as do Haiku com embedding.** Acrescente `--usar-reserva` só se a etapa direta reproduziu ou se a janela de depuração fechou. Pela projeção do piloto, as duas do Haiku com embedding gastam ~US$ 1,94 na Anthropic (0,2850 + 1,6506), dentro do que resta.

   ```
   PYTHONUTF8=1 uv run python -m replica.rodar oab-h1-emb-4omini oab-h1-emb-haiku oab-h1-bm25-4omini-k2 oab-h1-bm25-4omini-k5 oab-h1-bm25-4omini-k20 oab-h1corpus-bm25-4omini oab-h2-emb-4omini oab-h2-emb-haiku --rodada resultados/20261110-oab
   ```

4. **A última da fila**, nas mesmas condições de `--usar-reserva`:

   ```
   PYTHONUTF8=1 uv run python -m replica.rodar direta-mcc-emb-4omini --rodada resultados/20261110-direta
   ```

5. **Analisar as duas pastas.**

   ```
   PYTHONUTF8=1 uv run python -m replica.analise resultados/20261110-direta
   PYTHONUTF8=1 uv run python -m replica.analise resultados/20261110-oab
   ```

6. **Recalcular os pares secundários** com `replica.analise.bootstrap_pareado` (seção 3.6), acrescentando os novos: embedding × contexto longo, a ablação de k e o corpus inteiro.
7. **Atualizar este documento:** trocar cada **pendente** pelos números das tabelas; dar os vereditos da H2 direta, da H1 OAB e da H2 OAB; fazer o teste de reprodução das MCC contra 82,0 e 82,4; reler a seção 7 com as primárias.
8. **Commitar por caminho, nunca com `git add -A`:** `git add resultados/20261110-direta resultados/20261110-oab resultados/livro_caixa.json docs/replicacao-memoryagentbench.md`.

## Referências

- Hu, Y., Wang, Y. e McAuley, J. (2026). Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench). ICLR. arXiv:2507.05257. Código: github.com/HUST-AI-HYZ/MemoryAgentBench (MIT); dados: Hugging Face `ai-hyz/MemoryAgentBench`.
- Experimento 1 da residência: [`../../Experimento 1/docs/semana-8/memoria-passiva-ativa.md`](../../Experimento%201/docs/semana-8/memoria-passiva-ativa.md).
