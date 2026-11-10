# Experimento 2: replicação do MemoryAgentBench, direta e na OAB

Data: 2026-11-10. Status: rascunho para aprovação do Pedro.

## 1. Intenção

A residência quer sair com um mapa de qual memória faz sentido para qual caso. O mapa já existe,
tirado das leituras: a tabela "Quando usar cada abordagem" da semana 7 e
`Experimento 1/RESIDENCIA/conteudos_gerados_por_llm/mapa-memoria-agente.pdf`. O que falta é
evidência própria. O Experimento 1 testou uma região do mapa (fatos de uma usuária que mudam entre
sessões). O Experimento 2 testa duas afirmações de um artigo revisado por pares para saber **se o
autor estava certo**, e depois se a afirmação vale num domínio novo: questões da OAB em português.

O artigo é o MemoryAgentBench: Hu, Wang e McAuley, "Evaluating Memory in LLM Agents via
Incremental Multi-Turn Interactions", ICLR 2026, arXiv 2507.05257. O PDF traduzido está em
`Experimento 1/artigos/MemoryAgentBench (1).pdf`. Código: github.com/HUST-AI-HYZ/MemoryAgentBench
(MIT). Dados: Hugging Face `ai-hyz/MemoryAgentBench`.

Por que esse artigo: as afirmações dele são do tipo "qual memória para qual caso", o código e os
dados são públicos (a replicação direta é possível), e as categorias de agente dele batem com as
quatro configurações do Experimento 1 (sem memória, contexto completo, passiva, ativa).

## 2. As afirmações e como cada uma é testada

Toda afirmação passa por duas etapas. Rodar só na OAB não separa "o autor errou" de "lei é
diferente do que ele testou"; a etapa direta resolve isso.

- **Etapa direta:** dados e protocolo do autor, com o modelo do autor, para ver se o número se
  repete.
- **Etapa conceitual:** a mesma afirmação nas questões da OAB, para ver se vale fora dos dados dele.

### H1. O RAG vence o contexto longo quando é preciso achar um trecho preciso

Achado (1) do artigo, §4.2: "superioridade dos métodos RAG na recuperação precisa".

- **Direta:** dataset MH-Doc QA (1 contexto, 100 perguntas, contexto médio de 421K tokens, métrica
  SubEM, sem juiz). No artigo, Tabela 3: GPT-4o-mini com contexto longo 43,0; BM25 56,0.
- **OAB:** 300 questões sorteadas das provas de 2018 a 2023, estratificadas por matéria (ver §5). O
  agente de contexto longo recebe a lei da matéria inteira (ou o que couber na janela). Os agentes RAG recuperam artigos dessa mesma
  lei. Assim os dois têm a mesma memória e diferem só em como a leem, que é o ponto do artigo.
  Variante secundária, barata: o RAG sobre o corpus inteiro, que é o uso realista.

### H2. O contexto longo vence o RAG quando é preciso aprender com exemplos

Achado (2) do artigo, §4.2: "superioridade dos modelos de contexto longo no aprendizado em tempo de
teste".

- **Direta:** as cinco tarefas de classificação multiclasse (MCC) do artigo (BANKING77, CLINC150,
  NLU, TREC grosso e TREC fino), com 100 perguntas cada, métrica `exact_match` estrito, sem juiz.
  No artigo, a média das cinco (Tabela 3): GPT-4o-mini com contexto longo 82,0; BM25 75,4; os
  números por dataset estão na Tabela 7. Mudou de BANKING77 e CLINC150 para as cinco: ver item 7 de
  "Correções depois da leitura do código do autor (2026-11-10)".
- **OAB:** a memória são as provas até 2023, com gabarito. As questões de teste são as de 2024 e
  2025 (229 antes de filtrar), que ficam depois do corte de treino do GPT-4o-mini (outubro de 2023,
  segundo fontes secundárias). O contexto longo recebe as questões antigas mais recentes que
  couberem na janela. O RAG indexa todas as questões antigas e recupera as k mais parecidas.

### O que não dá para replicar

- **Quatro dos seis modelos de contexto longo.** Claude 3.7 Sonnet e Gemini 2.0 Flash já saíram do
  ar, o GPT-5-mini sai em 2026-12-11, e o GPT-4o custaria sozinho mais que o teto. O GPT-4.1-mini
  fica opcional.
- **A metade "compreensão de longo alcance" do achado (2).** A tarefa de resumo depende do GPT-4o
  como juiz, que também sai do ar, e o Detective QA tem só 71 perguntas.
- **O achado (3), esquecimento seletivo** (no multi-hop, nenhum método passa de 28%). Fica fora por
  decisão do Pedro; na OAB também não haveria como testá-lo, porque o corpus não guarda versões de
  lei.
- **Os agentes agênticos e os RAG com estrutura** (MemGPT, MIRIX, Self-RAG, GraphRAG, HippoRAG, Mem0
  e outros). Ficam fora pelo orçamento e porque as duas afirmações não dependem deles.
- **As ablações do artigo** (tamanho de chunk e top-k, §4.3), por decisão do Pedro. A única ablação
  do experimento é a de k na OAB H1, que desenha a curva qualidade × tokens.
- **Diferenças pequenas por dataset.** Na H2 direta, a diferença média do artigo é de 6,6 pontos. Com
  as 500 perguntas das cinco MCC (item 7 das Correções, abaixo), isso é detectável; se der
  inconclusivo, o documento diz que o orçamento não tinha resolução para mais.

### Regra de decisão (registrada antes de rodar)

- **Uma comparação primária por afirmação e etapa**, quatro ao todo (H1 direta, H2 direta, H1 OAB,
  H2 OAB). A primária é sempre o par que o artigo reporta: **GPT-4o-mini com
  contexto longo × GPT-4o-mini com RAG BM25**. Só a primária dá veredito.
- Na H2 direta, as cinco MCC entram juntas na primária: 500 perguntas, com o bootstrap
  reamostrando dentro de cada dataset (item 7 de "Correções depois da leitura do código do autor (2026-11-10)"). O resultado por dataset é secundário.
- **Secundárias** (RAG por embedding × contexto longo, os pares do Haiku, cada dataset separado, a
  ablação de k) são reportadas com intervalo, sem veredito. Se divergirem da primária, a divergência
  vai para o documento e é discutida, mas não muda o veredito.
- Sem correção para múltiplas comparações entre as quatro primárias: cada uma responde uma pergunta
  diferente, registrada antes.
- Comparações sempre pareadas: as mesmas questões em todas as configurações.
- Intervalo de 95% por bootstrap pareado (10.000 reamostragens) para a diferença de acurácia.
- **Confirmada** se o intervalo exclui zero na direção do artigo. **Contrariada** se exclui zero na
  direção oposta. **Inconclusiva** se contém zero. Inconclusivo é resultado e vai para o documento
  como está.
- Na etapa direta há um segundo teste: o número do artigo cai dentro do nosso intervalo de 95%?
  Se cai, a medida foi reproduzida.
- Com 100 perguntas, o intervalo de uma acurácia isolada tem uns ±9 pontos; a diferença pareada é
  mais estreita. Diferenças de 2 a 3 pontos não são detectáveis neste orçamento, e o documento diz
  isso.

## 3. Configurações

| Configuração | Etapa direta | OAB H1 | OAB H2 |
|---|---|---|---|
| Sem memória (só a pergunta) | não | GPT-4o-mini, Haiku 5.5 (3.377 questões válidas, todos os anos) | idem |
| Contexto longo, GPT-4o-mini (janela 128K) | sim | sim | sim |
| Contexto longo, Haiku 5.5 (janela 1M) | não | sim | sim |
| RAG BM25, leitor GPT-4o-mini | sim | sim | sim |
| RAG `text-embedding-3-small`, leitor GPT-4o-mini | sim | sim | sim |
| RAG BM25 e embedding, leitor Haiku 5.5 | não | sim | sim |
| Contexto longo, GPT-4.1-mini (opcional) | só se sobrar orçamento | não | não |

- **Modelos.** O GPT-4o-mini é o modelo de todos os agentes RAG do artigo e o par da comparação
  central; é a replicação fiel mínima. O Haiku 5.5 é a extensão atual: substitui os modelos de
  contexto longo que saíram do ar e é da mesma classe do GPT-4o-mini. Ele não entra na etapa direta,
  que fica só com modelos do artigo.
- **Fora do escopo:** agentes agênticos (Self-RAG, MemGPT, MIRIX), RAG com estrutura (GraphRAG,
  HippoRAG, RAPTOR, MemoRAG, Mem0, Cognee, Zep), Contriever, Qwen3-Embedding,
  `text-embedding-3-large`, as competências de compreensão de longo alcance e esquecimento
  seletivo, Lost in the Middle, construção de trecho-ouro, juiz LLM.
- **Ablação de orçamento (barata):** na OAB H1, o RAG com k = 2, 5, 10 e 20. Isso dá a curva
  qualidade × tokens que motivou o experimento.

## 4. Protocolo

Seguir o artigo onde ele define algo; adaptar só o que a OAB obriga, e registrar cada adaptação.

- **O código é o do autor.** O Pedro pediu que a replicação reproduza o artigo, e o caminho mais
  fiel é rodar o próprio código. O repositório tem licença MIT, então as partes que os três agentes
  usam (montagem da memória, chunking, prompts, agente de contexto longo, agente BM25, agente de
  embedding) são copiadas para o pacote do experimento, com atribuição e o commit fixado. A única
  troca é o cliente de LLM, substituído por um que passa pelo livro-caixa de custo (§6). A OAB é
  convertida para o formato de dados do autor, para que o mesmo código rode as duas etapas.
  - Plano B, se as dependências não funcionarem no Windows nem no WSL: reimplementar esses agentes
    copiando literalmente do repositório os prompts, os tamanhos de chunk, o k e os tetos de saída.
    Nesse caso vale o teste de fidelidade da §9.
- **Parâmetros do autor:** top-k = 10. O contexto longo memoriza em chunks de 4096 tokens; o RAG usa
  chunks de 512 no MH-QA e de 4096 nas MCC. Teto de saída de 50 tokens no MH-QA e 20 nas MCC
  (configurações do autor; item 2 de "Correções depois da leitura do código do autor (2026-11-10)"). Os prompts são os do autor (Apêndice D), em inglês,
  sem tradução.
- **Agente de contexto longo:** a memória é a sequência de chunks no formato do autor. Se não cabe
  na janela, aplica a mesma política do código do autor (a confirmar no repositório; a expectativa é
  descartar os chunks mais antigos). Na OAB H1 isso significa que o GPT-4o-mini vê só parte das
  leis grandes (CPC, CC, CF, CLT passam de 128K), e o Haiku vê todas inteiras. Isso é uma
  limitação registrada, e também um contraste útil.
- **OAB:**
  - Prompts em português com acentos. O modelo responde só a letra; a resposta é extraída por
    regex; letra inválida conta como erro e fica registrada.
  - Teto de saída de 20 tokens.
  - Haiku 5.5 sem raciocínio (`thinking: {type: "disabled"}`, aceito com effort `high` ou abaixo),
    para ficar comparável ao GPT-4o-mini, que não raciocina. Quem constrói confirma na API.
- **Amostragem:** OpenAI com temperatura 0,7, como nas configurações do autor (item 1 de
  "Correções depois da leitura do código do autor (2026-11-10)"); só a reclassificação de matéria usa temperatura 0. Na Anthropic, `temperature`, `top_p` e `top_k` dão 400
  a partir da versão de API 2026-10-21; o Haiku roda com a amostragem padrão. Registrar isso.
- **Cache ligado nas duas APIs.** O artigo calcula custos com cache (Apêndice I). Na OpenAI o cache é
  automático para prefixos idênticos; na Anthropic precisa de `cache_control` no bloco de contexto
  compartilhado. As perguntas que usam o mesmo contexto rodam em sequência, sem intervalo, para o
  cache não expirar. A contagem registra tokens totais, cacheados e escritos no cache, por chamada.
  **Isto é o contrário da regra do Experimento 1** (lá o cache foi proibido para a contagem ter um
  significado só) e vai para o `CLAUDE.md` do Experimento 2.
- **Sem fallbacks** para outro modelo, sem `tool_choice` forçado, como no Experimento 1.

## 5. Dados da OAB

Fonte: o banco Supabase do projeto Atlas (projeto `gqnejwbuloklbelsufnj`), lido com a service key do
`Experimento 2/.env`. **Só leitura.** Verificado em 2026-11-10:

- `questoes_oab`: 3.395 questões, 2010 a 2025 (2024: 152; 2025: 77). Gabarito equilibrado (A 791,
  B 893, C 879, D 832); o acaso dá ~25% e "sempre B" dá 26,3%. 18 questões têm menos de quatro
  alternativas e saem. Nenhuma marcada como anulada.
- `law_articles`: 11.954 artigos de 44 leis, um artigo por chunk, ~1,91M tokens no tokenizador
  `o200k_base` (GPT-4o-mini). Maiores: CPC 265K, CC 168K, CF 166K, CLT 144K. O maior artigo isolado
  tem 12.841 tokens (CPC, provável defeito do parser).
- `embeddings`: 14.368 linhas: 11.954 `law_article`, 204 `document_chunk` e **2.210 `questao`**.
- **Vazamento:** as questões de 2010 a 2018 estão no mesmo índice das leis, e a busca devolve a
  questão com o gabarito. **Nada do Experimento 2 usa as linhas `source_type='questao'`.** O RAG da
  OAB H1 indexa só `law_articles`.
- Tokenizador do Haiku: 1,71× mais tokens que o `o200k_base` no mesmo texto jurídico (medido com
  `count_tokens` numa amostra de 60 mil caracteres).

Decisões de dados:

- **Snapshot local:** baixar uma vez para `Experimento 2/dados/` (fora do git), com um manifesto
  versionado (contagens e sha256). Toda rodada lê do snapshot, nunca do banco.
- **Embeddings re-gerados localmente** com `text-embedding-3-small` (~US$ 0,04 para o corpus inteiro)
  em vez de baixar os vetores do banco. O banco usa o mesmo modelo, então é equivalente e mais
  simples.
- **BM25 local** (por exemplo `bm25s` com stemmer de português), não a RPC do banco: a RPC usa
  `plainto_tsquery` com E lógico e, para uma consulta de sete palavras, devolveu um resultado só.
- **Anos da OAB H1: provas de 2018 a 2023** (decisão do Pedro, opção A). O corpus é uma foto das leis
  (o CPC de uma edição de 2016, a CLT atual, a Lei 14.133 no lugar da 8.666), e o gabarito segue a lei
  vigente na data da prova. A partir de 2018 a lei cobrada fica mais próxima do corpus (CPC de 2015,
  CLT depois da reforma de 2017), e não há sobreposição com o teste da H2. O ruído que sobra (Pacote
  Anticrime de 2019, Lei 14.133 de 2021) é registrado como limitação. Depois de filtrar e tirar
  duplicatas, sobram ~970 questões para sortear 300.
- **Matéria reclassificada** nas provas de 2018 a 2023, porque os rótulos do banco são ruins depois de
  2018 e é o rótulo que decide qual lei o agente recebe. Classificação pelo GPT-4o-mini (~US$ 0,06),
  com uma amostra de 30 questões conferida à mão antes da rodada.
- **Matéria → lei** para a OAB H1. Padrão (quem constrói pode ajustar e registra):
  Civil → CC; Processo Civil → CPC; Constitucional → CF; Trabalho e Processo do Trabalho → CLT;
  Penal → CP; Processo Penal → CPP; Ética → EAOAB + Regulamento Geral + CED; Tributário → CTN;
  ECA → ECA; Consumidor → CDC; Administrativo, Empresarial, Ambiental, Eleitoral e Direitos Humanos
  → todas as leis do eixo correspondente. Internacional e Filosofia ficam fora da H1.
- **OAB H2:** remover da memória qualquer questão antiga quase igual a uma questão de teste (por
  exemplo, similaridade de caracteres acima de 0,8), e as duplicatas conhecidas (prova 2018-01 ≈
  2018-25, `2022-02-3` e `2022-02-3-fixed`).

## 6. Orçamento

Tetos do Pedro: **US$ 20 na OpenAI e US$ 30 na Anthropic.** Preços oficiais de 2026-11-10, por
milhão de tokens: GPT-4o-mini US$ 0,15 de entrada (US$ 0,075 com cache) e US$ 0,60 de saída;
Haiku 5.5 US$ 0,10 e US$ 0,50 até 100K tokens por chamada, US$ 0,50 e US$ 2,50 acima, escrita no
cache ~1,25× e leitura ~0,1× da entrada; `text-embedding-3-small` US$ 0,02.

| | OpenAI | Anthropic |
|---|---|---|
| Primárias (direta + OAB) e linhas de base | ~US$ 14,80 | ~US$ 0,25 |
| Secundárias | ~US$ 5,90 | ~US$ 20,70 (com a H2 do Haiku em 1M; ~US$ 16,50 em 600K) |
| **Total se tudo rodar** | ~US$ 20,70 | ~US$ 21 |
| Teto do livro-caixa | US$ 19,50 (trava de US$ 20), US$ 4 deles de reserva | US$ 29,50 (trava de US$ 30) |

Tabela revista com a leitura do código do autor (item 13 de "Correções depois da leitura do código do autor (2026-11-10)"). Com a trava de US$ 20, as
primárias (~US$ 14,80) cabem nos US$ 15,50 efetivos. As secundárias da OpenAI dependem da reserva,
liberada (`--usar-reserva`) depois que a etapa direta reproduzir ou fechar a janela de depuração. O
que não couber sai do fim da fila, e o documento registra.

- Sem cache, a OpenAI daria ~US$ 19 (no limite) e a Anthropic passaria de US$ 140 (não cabe). **O
  cache é obrigatório**, e o piloto existe para confirmá-lo.
- O maior item é o contexto longo do Haiku na H2: até ~950K tokens por chamada (a memória inteira
  daria 1,37M tokens do Haiku e não cabe na janela). Se o piloto mostrar risco, a memória do Haiku na
  H2 cai para 600K tokens (~US$ 7,30).
- O experimento 1 gastou mais do que a estimativa (US$ 6,14 contra US$ 3 a 5). O livro-caixa e a
  ordem de prioridade garantem que o que ficar de fora seja secundário.

Travas obrigatórias:

1. **Teto no código.** Um livro-caixa persistente por provedor soma o custo de cada chamada a partir
   do `usage` devolvido pela API (tokens normais, cacheados e escritos no cache) e **recusa a próxima
   chamada** se ela puder passar do teto. O livro-caixa sobrevive entre rodadas.
2. **Piloto primeiro:** ~10 questões por configuração, para medir tokens reais e confirmar que o cache
   pega (`cached_tokens` na OpenAI, `cache_read_input_tokens` na Anthropic). Só depois a rodada
   completa.
3. **Ordem por prioridade:**
   1. As rodadas das comparações primárias (contexto longo e BM25 do GPT-4o-mini), primeiro na etapa
      direta e depois na OAB.
   2. Sem memória.
   3. As secundárias baratas (RAG por embedding, ablação de k, RAG do Haiku).
   4. O contexto longo do Haiku.
   5. O GPT-4.1-mini opcional.

   Se o livro-caixa cortar algo, corta do fim da lista.
4. **Portões do Haiku**, decididos antes de qualquer rodada cara dele:
   - O "sem memória" do Haiku roda antes do contexto longo do Haiku. Se ele acertar 90% ou mais das
     questões de teste da H2 (2024 e 2025), o modelo provavelmente já as viu e sobra pouco espaço
     para a memória ajudar. Nesse caso o contexto longo do Haiku na H2 não roda, e o documento diz
     por quê.
   - Depois do piloto, se a projeção de gasto na Anthropic passar de US$ 24 (80% do teto), o
     contexto longo do Haiku na H2 roda com 600K tokens de memória. Ele é registrado como outra
     configuração ("contexto longo Haiku 600K"), não como a mesma.

### Se a etapa direta não reproduzir

- A etapa direta roda e termina antes da OAB, porque a OAB usa o mesmo código.
- Um teste offline compara, texto por texto, os prompts montados pela reimplementação com os
  templates do repositório do autor.
- Se o número de uma primária ficar fora do nosso intervalo, há uma janela de depuração de até três
  dias úteis e US$ 4 reservados do teto da OpenAI. Nessa janela, se for viável, roda-se o código do
  autor num subconjunto para comparar.
- Se ainda assim não reproduzir, a OAB segue, e o documento registra: "não reproduzido; não foi
  possível separar o autor da reimplementação".

### Decisões do piloto (2026-11-10)

O piloto (`resultados/piloto-20261110/`, 10 perguntas por configuração; as sem memória completas)
gastou US$ 1,04 na OpenAI e US$ 1,47 na Anthropic, sem nenhum erro de API. Decisões tomadas pelo
Claude, que executou o plano sem o Pedro na sessão; cada uma está registrada para o Pedro rever.

- **Cache confirmado.** Nas seis configurações de contexto longo, as chamadas 2 a 10 vieram com 99% a
  100% dos tokens de entrada cacheados. A chamada do contexto longo do Haiku na H2 escreveu 902.537
  tokens no cache e coube na janela de 1M.
- **Letra inválida: 0%** em todas as configurações da OAB. A regex de `extrair_letra` fica como está.
- **Portão do Haiku aberto.** O Haiku sem memória acertou 69,2% das 224 questões de teste da H2
  (precisaria de 90% para fechar). O GPT-4o-mini sem memória acertou 58,9% de todas as 3.377.
- **Variante do Haiku na H2: 1M.** A projeção da Anthropic ficou em ~US$ 18 (~US$ 19 corrigindo o viés
  da H1, cujo primeiro contexto é o menor), abaixo de US$ 24.
- **Projeção da OpenAI:** primárias em ~US$ 14, dentro dos US$ 16 do plano; com as secundárias, ~US$
  19,60, acima do teto efetivo. O que não couber sai do fim da fila (a reprodução das MCC por
  embedding é a última).
- **Métrica das MCC: leitura flexível como primária.** Com o prompt do próprio autor ("Only output
  "label: {label}""), o GPT-4o-mini responde "label: 68", e o `exact_match` estrito do README conta
  isso como erro: 0 em 10 nos dois braços do piloto, contra 6 e 8 em 10 na leitura flexível. O código
  do autor não normaliza "label:" em lugar nenhum, então o estrito não pode ter gerado os 82,0 e 93,0
  do artigo, e deixaria a H2 direta degenerada (0 × 0). A acurácia primária das MCC (veredito,
  reprodução e secundárias) passa a ser `exact_match_flexivel`; o estrito aparece ao lado, como a
  métrica literal do README. A decisão veio do formato da resposta, antes de qualquer dado da rodada
  completa, e as duas leituras ficam gravadas em cada linha. Isto muda a Correção 3.
- **Reclassificação de matéria:** a rodada 1 teve 4 erros em 30 e foi reprovada pela regra (até 3). O
  prompt passou a informar o número da questão e a ordem dos blocos da prova da FGV; a rodada 2 teve 0
  erro numa amostra nova de 30 (`docs/conferencia-materias.md` e `-rodada1.md`). As duas conferências
  foram feitas pelo Claude e o Pedro reconfere. Restam 65 de 953 questões com matéria fora do bloco;
  como o contexto longo e o RAG recebem a mesma lei, isso não enviesa a comparação pareada.
- **Linha de base sem memória reaproveitada.** As 3.377 respostas de cada modelo sem memória do piloto
  são copiadas para a pasta da rodada da OAB, em vez de pagas de novo: mesma configuração, mesmo
  código, mesmas questões.
- **Pendente para o Pedro:** comparar `resultados/livro_caixa.json` com os painéis da OpenAI e da
  Anthropic (diferença de até 10% aceita). Os preços do Haiku 5.5 no código foram conferidos na
  documentação da Anthropic.

## Correções depois da leitura do código do autor (2026-11-10)

A leitura do código do autor (commit `538026089d1a8a8eff05121d0db89b388f360eba`) e dos dados dele (revisão `7ea066982b140a19337e17e60d45d4076e042faf` no Hugging Face) mudou estes pontos. Pela regra do próprio desenho ("seguir o artigo onde ele define algo"), os itens 1 a 6 não precisam de decisão nova. **O item 7 mudou o desenho; o Pedro aprovou em 2026-11-10.**

1. **Temperatura 0,7** nos agentes da OpenAI, como nas configurações do autor; o desenho dizia 0. Vale também para a OAB, porque o código e as configurações são os mesmos. Só a reclassificação de matéria usa temperatura 0, porque não é braço do experimento.
2. **Chunks e tetos de saída do autor.** O contexto longo memoriza em chunks de 4096 tokens. O RAG usa chunks de 512 no MH-QA (como o artigo e a lista de ablação do autor) e de 4096 nas MCC. O teto de saída do MH-QA é 50 tokens (configuração do autor; o desenho dizia 100).
3. **Métricas** (README do autor, "Clarification on Evaluation Metrics"): MH-QA usa `substring_exact_match`; MCC usa `exact_match` estrito, em que `"label: 43"` conta como erro. A versão com leitura flexível de `label:` entra como secundária.
4. **BM25 e embedding sem LangChain.** O `BM25Retriever` do LangChain é o `rank_bm25.BM25Okapi` com tokenização por `str.split()`, sem minúsculas; reproduzimos direto com `rank_bm25`. O FAISS L2 sobre embeddings normalizados ordena igual ao cosseno; reproduzimos com numpy. O caminho de embedding do autor exige Azure (`use_azure=True` fixo no código); trocamos pelo cliente normal da OpenAI.
5. **Peculiaridade do autor que fica:** o agente de embedding tem `input_length_limit` 300.000 e `buffer_length` 15.000. Com isso, ele descarta os chunks mais antigos quando a memória passa de ~285 mil tokens. No MH-QA (430 mil tokens), ele indexa só os últimos dois terços do contexto; o BM25 (limite de 10 milhões) indexa tudo. Isso ajuda a explicar o 44,0 do embedding contra o 56,0 do BM25. É o que o artigo rodou, então fica, e o documento final registra.
6. **Números do artigo para o teste de reprodução** (arXiv 2507.05257v4, conferidos verbatim):
   - Tabela 3, MH-QA: contexto longo 43,0; BM25 56,0; embedding 44,0. MCC (média das cinco): 82,0; 75,4; 70,0.
   - Tabela 7, por dataset (BANKING, CLINIC, NLU, TREC C, TREC F): contexto longo 93,0 / 93,0 / 87,0 / 73,0 / 66,0; BM25 89,0 / 89,0 / 84,0 / 62,0 / 53,0; embedding 88,0 / 89,0 / 83,0 / 54,0 / 36,0.
   - Inconsistência do artigo: a média da Tabela 7 do contexto longo dá 82,4, não os 82,0 da Tabela 3. O documento final registra.
7. **[Aprovado pelo Pedro] A H2 direta usa as cinco MCC em vez de só BANKING77 e CLINC150.**
   - Motivo: nesses dois datasets a diferença do artigo é de 4 pontos (93 × 89), que 200 perguntas não detectam com segurança. A célula MCC da Tabela 3 (82,0 × 75,4) é a média das cinco, e 500 perguntas detectam 6,6 pontos.
   - Custo: +US$ 4,30 na OpenAI. Se o Pedro preferir os dois datasets do desenho, basta reduzir `MCC_FONTES` em `config.py`.
8. **OAB H2 com chunk de 4096 no RAG**, análogo às MCC. Cada busca traz ~160 questões antigas (~41 mil tokens), mais caro do que o desenho estimou.
9. **Variante "corpus inteiro" da OAB H1 só com BM25.** Com o limite do autor, o agente de embedding guardaria só os últimos ~285 mil tokens do corpus de 1,9 milhão, e a variante não mediria nada.
10. **Ordem de prioridade.** As linhas de base sem memória sobem para antes das primárias da OAB, porque o portão do Haiku depende delas e custam ~US$ 0,50.
11. **GPT-4.1-mini fica fora.** A configuração mais barata dele custaria mais de US$ 4, e não sobra orçamento.
12. **Haiku no contexto longo** segue a convenção do autor para o Claude (janela menos uma folga, em tokens do tokenizador do autor): `input_length_limit` 1.000.000 com `buffer_length` 480.000 (memória de ~520 mil tokens `o200k`, ~890 mil do Haiku) ou 670.000 na variante 600K.
13. **Orçamento revisto, com cache:**

| | OpenAI | Anthropic |
|---|---|---|
| Primárias (direta + OAB) e linhas de base | ~US$ 14,80 | ~US$ 0,25 |
| Secundárias | ~US$ 5,90 | ~US$ 20,70 (com a H2 do Haiku em 1M; ~US$ 16,50 em 600K) |
| **Total se tudo rodar** | ~US$ 20,70 | ~US$ 21 |
| Teto do livro-caixa | US$ 19,50 (trava de US$ 20), US$ 4 deles de reserva | US$ 29,50 (trava de US$ 30) |

Com a trava de US$ 20, as primárias (~US$ 14,80) cabem nos US$ 15,50 efetivos. As secundárias da OpenAI dependem da reserva, liberada (`--usar-reserva`) depois que a etapa direta reproduzir ou fechar a janela de depuração. O que não couber sai do fim da fila, e o documento registra.

## 7. Prazo

O GPT-4o-mini, o GPT-4o e o GPT-4.1-mini saem da API da OpenAI em **2027-01-14**; o GPT-5-mini em
2026-12-11 (deprecations da OpenAI, consultado em 2026-11-10). O Claude 3.7 Sonnet saiu em
2026-02-19 e o Gemini 2.0 Flash em 2026-06-01. **Todas as rodadas com modelos da OpenAI terminam até
2026-12-20**, deixando três semanas para repetir o que falhar. A perda de modelos é ela mesma um
achado: menos de um ano depois da ICLR, dois dos seis modelos de contexto longo do artigo já não
existem, e o documento final registra isso.

## 8. Entregas

- `resultados/<AAAAMMDD-HHMMSS>/`: um JSON por configuração e etapa, com cada resposta, se acertou,
  tokens normais, cacheados e escritos no cache, e custo; e uma tabela-resumo em Markdown.
- Tabela por afirmação e etapa: acurácia com intervalo de 95%, diferença pareada com intervalo,
  veredito (confirmada, contrariada, inconclusiva), tokens e custo por configuração.
- Gráfico qualidade × tokens da OAB H1, com a ablação de k e os pontos de contexto longo.
- Documento da entrega em `docs/`, no estilo do documento da semana 8 do Experimento 1: hipótese
  escrita antes, resultado, leitura honesta, limitações, e o que cada afirmação diz sobre o mapa da
  residência.

## 9. Testes

Os testes nunca chamam API; usam clientes falsos, como no Experimento 1. Mínimo:

- O livro-caixa soma custos certos para usage com e sem cache e recusa a chamada que passaria do
  teto.
- A extração da letra aceita respostas válidas e marca as inválidas.
- A montagem do contexto longo respeita a janela e a política de descarte.
- O RAG devolve k itens, de forma determinística, e nunca devolve linha `questao` na OAB H1.
- A separação da H2 não deixa questão de teste nem quase-duplicata dela na memória.
- O bootstrap pareado devolve intervalo que contém a diferença observada.
- Fidelidade: os prompts montados para a etapa direta são iguais, texto por texto, aos templates do
  repositório do autor (obrigatório no plano B; no plano A, confirma que a cópia não alterou nada).
- A conversão da OAB para o formato do autor preserva questão, alternativas e gabarito.

As rodadas reais são eval: rodam, e o número é comparado; nunca têm assert de igualdade.

## 10. Restrições

- Tetos de US$ 20 (OpenAI) e US$ 30 (Anthropic); prazo de 2026-12-20 para a OpenAI.
- Windows; Python com `uv`; acentos exigem `PYTHONUTF8=1` em saída redirecionada.
- PT-BR com acentos em tudo que o modelo lê na OAB e em tudo que o professor lê. Na etapa direta, os
  prompts do autor ficam em inglês, sem tradução.
- O Experimento 1 não é alterado.

## 11. Anti-objetivos

- Passar dos tetos, mesmo "só para terminar uma configuração".
- Qualquer linha `questao` do banco dentro de um índice de RAG.
- Configurações comparadas sobre questões diferentes, ou com modelos diferentes dentro da mesma
  comparação.
- Afirmar mais do que o intervalo permite: inconclusivo é inconclusivo.
- Escrever no banco do Atlas.

## 12. O que fica com quem constrói

- Layout do pacote e nomes de módulos; seguir o espírito do Experimento 1 ("o mais simples que ainda
  ensina", um arquivo que se lê de cima para baixo por agente).
- Quais arquivos do repositório do autor copiar, e como isolar as dependências pesadas que os três
  agentes não usam (torch, hipporag). Windows nativo ou WSL.
- Tokenização de português para o BM25 na OAB (o artigo é em inglês), formato do snapshot, limiar de
  quase-duplicata.
- Número exato de questões antigas que cabem em cada janela.
