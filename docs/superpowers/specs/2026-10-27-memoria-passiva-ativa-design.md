# Memória passiva e memória ativa com LLM: dois exemplos comparáveis

Data: 2026-10-27
Contexto: Residência em IA (Bacharelado em IA, UFG). Entrega do gate da semana 8.
Status: aprovado pelo autor em 2026-10-27, após revisão de construtor e respostas às perguntas de §9.

## 0. Princípio: o mais simples que ainda ensina

Pedido explícito do autor: o repositório precisa ser fácil de revisar e de entender, com cada
exemplo legível de ponta a ponta, tudo documentado, e **sem código extra**. Isso vale mais do
que qualquer refinamento abaixo. Regras derivadas:

- Cada exemplo é **um arquivo** que se lê de cima para baixo. Nada de camadas de abstração entre
  o exemplo e a chamada à API além de um wrapper fino de cliente.
- Uma função faz uma coisa e tem docstring em PT-BR dizendo o que ela faz e por quê.
- Sem flags, opções ou caminhos alternativos que a entrega não usa. Se é "extensão futura",
  vai para o documento, não para o código.
- Dependências: só `anthropic`, `python-dotenv` e `pytest`. SQLite da biblioteca padrão.
- Se duas implementações funcionam, fica a mais curta e mais óbvia, mesmo que menos elegante.

## 1. Intenção e porquê

O gate da semana 7 fechou um framework em três passos para decidir se e como um sistema LLM
precisa de memória: **diagnóstico** (precisa mesmo de memória externa?), **desenho** (o que
guardar, quando escrever, como ler, como esquecer) e **verificação** (com qual benchmark, quais
baselines, quais métricas). Também separou duas formas de ligar memória a um LLM:

- **memória passiva (workflow):** o código decide quando ler e quando escrever. O modelo nunca
  sabe que existe memória.
- **memória ativa (agêntica):** o modelo recebe ferramentas de leitura e escrita e decide quando
  usá-las.

A entrega da semana 8 é **implementar e documentar um exemplo de cada**. Este documento descreve
os dois exemplos, a verificação que os compara e a documentação que vai para o professor.

O ponto pedagógico central: os dois exemplos compartilham tudo (mesma conversa, mesmo modelo,
mesmo banco, mesma janela de mensagens recentes) e diferem **só em quem controla a leitura e a
escrita**. Assim a comparação isola a variável que o framework da semana 7 destaca.

## 2. Objetivos, não objetivos e anti-objetivos

### Objetivos

1. Um exemplo de memória passiva por **sumarização incremental**, rodável com um comando.
2. Um exemplo de memória ativa **estilo MemGPT**, com ferramentas `buscar_memoria` e
   `gravar_memoria`, rodável com um comando.
3. Uma verificação em miniatura no formato do LongMemEval: conversa sintética de várias sessões,
   perguntas categorizadas, duas baselines (sem memória e contexto completo), juiz LLM, tabela de
   acertos e custo em tokens.
4. Um documento em PT-BR no padrão das semanas anteriores, estruturado pelo framework da semana 7
   (diagnóstico, desenho, verificação, conclusões).
5. Testes determinísticos (sem chamadas reais à API) cobrindo a lógica de cada pipeline.

### Não objetivos

- Não é um sistema de produção. Sem autenticação, sem concorrência, sem API HTTP.
- Não usa embeddings nem banco vetorial. A busca da memória ativa é textual (ver §5.2).
- Não replica o MemoryAgentBench nem roda benchmarks públicos. A verificação é própria e pequena.
- Não compara provedores nem modelos. Um único modelo em todas as configurações.
- Não implementa esquecimento explícito (ferramenta de apagar). Ver §5.2 para o comportamento
  esperado em atualização de conhecimento.

### Anti-objetivos (o que faria a entrega falhar mesmo funcionando)

- Os dois exemplos diferirem em algo além do controle da memória. Qualquer diferença extra
  contamina a comparação.
- Resultados sem baseline. Sem "sem memória" e "contexto completo", a tabela não diz nada.
- Chamar o exemplo passivo de RAG. É context compression (sumarização), e o documento usa o
  vocabulário dos artigos lidos.
- Código que só roda na máquina do autor. Setup documentado, chave via variável de ambiente.

## 3. Restrições

- **Modelo:** Claude Sonnet 5.5 pela API da Anthropic, ID `claude-sonnet-5-5`, SDK oficial
  Python (`anthropic`). Decisão do autor. ID em uma única constante de configuração. Nenhum
  recurso que muda entre gerações é usado: sem `tool_choice` forçado, sem thinking desligado,
  sem replay de blocos de thinking entre turnos.
- **A rodada paga faz parte da entrega.** O autor autorizou o uso da chave dele para a rodada de
  verificação, uma vez, no custo estimado em §3 Custo. O documento final leva os números reais.
- **Sem fallback de modelo.** O recurso de fallback server-side da API fica desligado. Se a API
  recusar uma chamada, o turno é registrado como erro e a rodada continua. Motivo: um fallback
  para outro modelo contaminaria a comparação.
- **Thinking:** adaptativo, padrão do modelo. Sem `budget_tokens`. Sem `tool_choice` forçado
  (retorna 400 nos modelos atuais). Esforço (`effort`) não configurado por padrão, exposto como
  constante.
- **Linguagem e ferramentas (escolha minha):** Python 3.12+, `uv` para ambiente e dependências,
  `pytest` para testes, SQLite da biblioteca padrão para persistência. Sem frameworks de agente
  (LangChain, Mem0 etc.), porque o objetivo é mostrar o mecanismo.
- **Idioma:** tudo que o modelo lê (system prompts, descrições de ferramentas, prompt do juiz) e
  tudo que o professor lê é em PT-BR com acentos. Nomes de código em PT-BR sem acento.
- **Chave:** `ANTHROPIC_API_KEY` por variável de ambiente ou `.env` (ignorado pelo git).
  `.env.example` versionado.
- **Custo:** a estimativa é de US$ 3 a 5 por rodada completa da verificação com o preço atual do
  Sonnet, contando thinking como saída. É uma meta, não um bloqueio. O runner imprime o total de
  tokens e o custo estimado ao final. Se passar muito disso, as alavancas permitidas, nesta
  ordem, são: rodar com `--config` só o que falta, e depois fixar `effort: "low"` **nas quatro
  configurações ao mesmo tempo**, anotando no documento. Nunca encolher o dataset depois de uma
  configuração já ter rodado, porque quebra a comparação. Escolha minha.

## 4. Estrutura do repositório

O repositório é o da residência inteira. A pasta `RESIDENCIA/` com os documentos das semanas já
existe e não muda. O resto é novo.

```
Residência-ufg/
├── RESIDENCIA/                      # documentos das semanas (existente, intocado)
├── memoria/                         # pacote Python, 6 arquivos
│   ├── __init__.py
│   ├── config.py                    # MODELO, K, limites, caminho do banco (só constantes)
│   ├── llm.py                       # único ponto que chama a API; soma tokens; aceita cliente fake
│   ├── store.py                     # SQLite: resumo por conversa, notas com busca FTS5
│   ├── passiva.py                   # EXEMPLO 1: sumarização incremental (+ chat no __main__)
│   ├── ativa.py                     # EXEMPLO 2: loop com ferramentas (+ chat no __main__)
│   └── avaliacao/
│       ├── dataset.json             # conversa sintética + perguntas (fixture estática)
│       ├── baselines.py             # sem memória, contexto completo (duas funções)
│       ├── juiz.py                  # juiz LLM (uma função)
│       └── rodar.py                 # roda as 4 configurações e gera a tabela
├── tests/                           # pytest, tudo com cliente fake
├── resultados/                      # saída da rodada final (versionada)
├── docs/
│   ├── semana-8/memoria-passiva-ativa.md    # documento da entrega
│   └── superpowers/{specs,plans}/
├── CLAUDE.md
├── .claude/settings.json            # habilita superpowers
├── pyproject.toml
├── .env.example
└── README.md                        # como rodar em 5 linhas
```

Comandos:

```
uv sync                                   # ambiente
uv run pytest                             # gate de testes (sem API)
uv run python -m memoria.passiva          # chat de terminal com o exemplo passivo
uv run python -m memoria.ativa            # chat de terminal com o exemplo ativo
uv run python -m memoria.avaliacao.rodar  # verificação completa (usa API, custa dinheiro)
```

## 5. Desenho

### 5.0 Conceitos compartilhados

- **Conversa** é o histórico completo de um usuário com o assistente, dividido em **sessões**.
  Cada sessão tem uma data e uma lista de turnos. **Um banco SQLite guarda uma conversa.** A
  memória atravessa sessões; a verificação abre um banco novo em memória para cada configuração.
  Isso é escolha minha: no desenho original a chave era `session_id`, mas as perguntas
  multi-sessão do LongMemEval só fazem sentido se a memória sobrevive à troca de sessão. Sem
  coluna de id de conversa, porque nada aqui precisa de duas conversas no mesmo banco.
- **Janela recente** são as últimas K mensagens **da sessão atual** (K configurável, padrão 6,
  contando mensagens de usuário e de assistente). A janela zera quando começa uma sessão nova.
  Consequência desejada: dentro de uma sessão, a janela ajuda; entre sessões, só a memória ajuda.
  Sem memória, passiva e ativa usam exatamente a mesma regra de janela. Contexto completo não
  tem janela por definição: vê tudo.
- **System prompt base** é o mesmo nas quatro configurações: assistente pessoal em PT-BR, responde
  de forma direta, diz "não sei" quando não tem a informação. Cada configuração acrescenta só o
  bloco que lhe diz respeito.
- **Data da sessão** entra no prompt de todas as configurações ("Hoje é 2026-03-14"). Sem isso,
  perguntas temporais são impossíveis para qualquer uma.
- **Contagem de tokens:** o wrapper `llm.py` soma tokens de entrada e saída e número de
  chamadas de cada resposta da API em um contador simples. O runner lê e zera o contador por
  configuração. É a fonte da coluna de custo da tabela. Sem tabela de log no banco.
- **O histórico guarda só texto.** Cada turno vira exatamente duas mensagens no histórico: o
  texto do usuário e o texto final do assistente. Blocos de thinking, `tool_use` e `tool_result`
  **não** entram no histórico entre turnos. Dentro de um mesmo turno da memória ativa, o conteúdo
  do assistente é reenviado inteiro e sem edição até o `end_turn`, como a API exige. Escolha
  minha, com dois efeitos desejados: a regra da janela K fica idêntica nas quatro configurações,
  e nenhum bloco de thinking é reenviado entre turnos, o que evita a verificação de preserved
  thinking sobre histórico editado nos modelos atuais. Consequência aceita: na ativa, o modelo
  não vê os resultados de busca do turno anterior e precisa buscar de novo. É assim no MemGPT.
- **Erro de API em um turno.** O SDK faz as 2 tentativas padrão. Se ainda falhar (ou se a API
  recusar), o turno recebe a resposta literal `[erro: <motivo>]`, que entra no histórico como
  mensagem do assistente para manter a alternância usuário/assistente. O resumidor da passiva
  não roda nesse turno. Uma pergunta da verificação com erro conta como incorreta e continua no
  N. A tabela tem uma coluna de erros por configuração. Escolha minha.
- **Sem prompt caching.** Mantém a contagem de tokens de entrada com um significado só. Fica
  registrado como extensão. Escolha minha.

### 5.1 Exemplo 1: memória passiva por sumarização incremental

É o pipeline desenhado pelo autor, com uma mitigação de colapso de contexto.

```
mensagem do usuário
   │
   ├─▶ 1. carrega o resumo atual S da conversa (SQLite)
   │
   ├─▶ 2. monta o prompt: system base + bloco "Resumo do que você sabe:" + S
   │       + últimas K mensagens da sessão + mensagem nova
   │       └─▶ modelo responde (sem ferramentas)
   │
   └─▶ 3. chamada ao resumidor: recebe S + (mensagem, resposta) + data da sessão
           └─▶ devolve S' e grava no SQLite, guardando também a versão anterior
```

Decisões:

- **S tem estrutura fixa**, em Markdown, com quatro seções: `Sobre o usuário`, `Fatos e
  decisões`, `Pendências` e `Linha do tempo` (eventos com data). O resumidor é instruído a
  **editar** as seções, não reescrever livre: manter o que continua válido, acrescentar o novo,
  substituir o que foi contradito e registrar na linha do tempo o que tem data. Isso é a
  mitigação do colapso de contexto documentado no ACE (semana 3). Escolha minha.
- **Limite de tamanho** de S em palavras (padrão 400). O resumidor é instruído a comprimir o que
  for menos importante quando passar do limite. Escolha minha.
- **Resumo a cada turno**, como no desenho original. Resumir só quando a janela estoura seria mais
  barato, mas muda o comportamento e vira outra variante. Fica registrado como extensão futura.
- **Evolução de S** é registrada pelo runner da verificação, que copia S ao fim de cada sessão
  para o JSON de saída. O banco guarda só o S atual. O documento final mostra o tamanho de S por
  sessão, que é a evidência direta de colapso ou de estabilidade.
- O modelo principal **não vê** que existe memória. Só recebe S como texto no prompt.

### 5.2 Exemplo 2: memória ativa estilo MemGPT

```
mensagem do usuário
   │
   ├─▶ monta o prompt: system base + bloco "Você tem memória externa..." 
   │     + últimas K mensagens da sessão + mensagem nova
   │     + ferramentas [buscar_memoria, gravar_memoria]
   │
   └─▶ loop:
         modelo responde
         ├─ stop_reason == tool_use → executa cada ferramenta, devolve tool_result, repete
         └─ stop_reason == end_turn → resposta final
```

Decisões:

- **Duas ferramentas**, definidas com `strict: true`:
  - `buscar_memoria(consulta: str) -> lista de notas` com id, data de gravação e conteúdo. Top-n
    (padrão 5).
  - `gravar_memoria(conteudo: str) -> id`. Nota curta, autocontida, carimbada com a data da
    sessão.
- **Busca textual com FTS5 do SQLite**, sem embeddings. Escolha minha: evita um segundo provedor
  (a Anthropic não oferece embeddings), é determinística e suficiente para o tamanho do teste.
  Limitação registrada: sem stemming em português, "viajei" não casa com "viagem". O documento
  cita isso como limitação e a busca por embeddings como extensão.
- **Notas são append-only.** Não há ferramenta de apagar ou editar. Quando um fato muda, o
  modelo grava uma nota nova e, na busca, recebe as duas com data. Resolver o conflito pela data
  é responsabilidade dele. Isso expõe de propósito o risco de staleness do mapa mental e a
  competência "selective forgetting" do MemoryAgentBench. O resultado na categoria "atualização
  de conhecimento" é o que se quer medir, não esconder.
- **Sem resumo automático.** Se o modelo não gravar, perdeu. O bloco do system prompt diz
  explicitamente: a janela de mensagens é curta, sessões anteriores não ficam visíveis, grave o
  que parecer durável e busque antes de responder sobre o passado.
- **Loop manual** (`while stop_reason == "tool_use"`), não o tool runner do SDK. Motivo: o
  documento precisa explicar o mecanismo, e o loop explícito é o que se explica. Escolha minha.
- **Limite de iterações** por turno (padrão 6) para evitar loop infinito. Ao estourar, a última
  resposta de texto vale como final.
- Tool results sempre devolvidos **todos em uma única mensagem de usuário** quando o modelo pede
  várias ferramentas em paralelo.
- **Contagem de ferramentas** por turno vai para o log. É uma coluna da tabela.

### 5.3 Baselines

- **Sem memória:** system base + últimas K mensagens da sessão + mensagem nova. Nada atravessa
  sessões. É o piso.
- **Contexto completo:** system base + todas as mensagens de todas as sessões anteriores + a
  sessão atual inteira + mensagem nova. É o teto de informação disponível e o mais caro. Equivale
  à linha "long-context" do LongMemEval.

### 5.4 Chat de terminal

Cada exemplo tem no fim do próprio arquivo um bloco `if __name__ == "__main__":` com um chat de
terminal mínimo: lê uma linha, responde, repete. Comando `/nova` zera a janela para simular
troca de sessão. O passivo imprime S' após cada turno; o ativo imprime cada chamada de
ferramenta conforme acontece. Serve para a demonstração no gate. Sem módulo separado.

## 6. Verificação

Aplica o passo 3 do framework da semana 7.

### 6.1 Dataset

Arquivo `memoria/avaliacao/dataset.json`, uma **fixture estática versionada**, escrita durante
a implementação e nunca gerada em tempo de execução, para que o gabarito seja controlado e
igual em toda rodada. Escolha minha. Ver §9 sobre revisão do autor antes da rodada paga.

- Um usuário fictício. Entre 4 e 6 sessões com datas distintas espaçadas por dias ou semanas.
- Entre 8 e 12 turnos de usuário por sessão. **Só as falas do usuário são fixas.** As respostas
  do assistente são geradas ao vivo por cada configuração, porque a memória precisa ser
  alimentada pelo pipeline real.
- As falas misturam informação durável (nome do cachorro, cidade, projeto em andamento, uma
  preferência) com conversa descartável, na proporção aproximada de 1 para 3. A informação
  relevante para cada pergunta aparece **uma única vez**.
- Pelo menos um fato é **atualizado** em sessão posterior (por exemplo, muda de emprego).
- **Sessão final de perguntas**, com data posterior à última sessão. Entre 15 e 20 perguntas, de
  3 a 4 por categoria:

| Categoria | O que testa | Exemplo |
|---|---|---|
| sessão única | fato dito em uma sessão anterior | "Como se chama meu cachorro?" |
| multi-sessão | juntar fatos de sessões diferentes | "Quais livros eu disse que comecei?" |
| temporal | raciocínio com datas | "Quantas semanas faz que entreguei o relatório?" |
| atualização de conhecimento | fato que mudou | "Onde eu trabalho hoje?" |
| abstenção | informação nunca dita | "Qual é o nome da minha irmã?" |

Cada pergunta tem: id, categoria, texto, resposta esperada, e para abstenção a resposta esperada
é "não sei / não foi dito".

As perguntas são feitas **em sequência na mesma sessão final**, cada uma como um turno normal
(a memória continua funcionando durante as perguntas, inclusive na ativa).

### 6.2 Juiz

Uma chamada por par (pergunta, resposta) com o mesmo modelo: recebe pergunta, resposta esperada
e resposta dada, devolve `correto` ou `incorreto` com justificativa de uma linha, via structured
output. Para abstenção, é correto se o assistente disse que não sabe ou que não foi mencionado.
Mesmo prompt de juiz para as quatro configurações.

### 6.3 Métricas e saída

Por configuração:

- acertos por categoria e total (N perguntas, portanto a tabela mostra `acertos/N`, não só %)
- tokens de entrada e de saída somados em toda a rodada (conversa + perguntas, sem contar o juiz)
- número de chamadas à API
- para a ativa: número de chamadas de ferramenta, separado por buscar e gravar
- para a passiva: tamanho de S em palavras ao fim de cada sessão

Saída em `resultados/<data-hora>/`: um JSON por configuração com todos os turnos, respostas,
veredictos e tokens, mais `tabela.md` com a tabela final. O runner aceita um único argumento
opcional, o nome de uma configuração, para rodar só ela. Sem outras opções.

O documento final registra que os números vêm de uma rodada e que saída de LLM não é
determinística. Nunca há assert de igualdade exata sobre saída de LLM. Repetir a rodada e tirar
média é extensão, não escopo.

### 6.4 Resultado esperado (hipótese a confirmar, não a forçar)

Baseado no que a residência leu: contexto completo deve ter o maior acerto e o maior custo;
sem memória deve errar tudo que atravessa sessão; passiva deve ir bem em sessão única e
atualização (o resumo sobrescreve) e pior em temporal (perde detalhe); ativa deve depender de
o modelo ter gravado, com custo intermediário e risco de staleness em atualização. Se o
resultado contrariar a hipótese, isso vai para o documento como está.

## 7. Documentação da entrega

`docs/semana-8/memoria-passiva-ativa.md`, em PT-BR, estrutura:

1. Diagnóstico: por que esse cenário precisa de memória (histórico não cabe na janela? cabe, mas
   custa; informação muda; sessões separadas).
2. Desenho do exemplo passivo, com o diagrama do pipeline, o prompt do resumidor e um exemplo
   real de S evoluindo.
3. Desenho do exemplo ativo, com o diagrama do loop, as definições das ferramentas, o bloco do
   system prompt e um trecho real de transcrição com chamadas de ferramenta.
4. Verificação: dataset, categorias, juiz, tabela de resultados, evolução do tamanho de S.
5. Conclusões: o que a comparação mostrou sobre passiva versus ativa, ligando com MemGPT, ACE,
   LongMemEval e MemoryAgentBench.
6. Limitações e extensões: embeddings, esquecimento explícito, resumo por limiar, repetições.
7. Como rodar.

O termo de aceite da semana 8 é escrito pelo autor, fora deste escopo.

## 8. Testes

Gate: `uv run pytest`, tudo offline. O cliente da API é injetado; os testes usam um cliente
fake que devolve respostas roteirizadas (inclusive `tool_use` para o loop ativo).

Cobertura mínima, uma task de teste por item no plano:

- `store`: criar e ler resumo por conversa; gravar e buscar notas por FTS5.
- `llm`: o contador soma tokens e chamadas; zerar funciona.
- `passiva`: o prompt enviado contém S e exatamente as últimas K mensagens da sessão; após o
  turno, S' é o que o fake resumidor devolveu; janela zera na troca de sessão.
- `ativa`: quando o fake devolve `tool_use`, a ferramenta roda, o `tool_result` volta no formato
  certo, e o loop termina no `end_turn`; limite de iterações respeitado; chamadas paralelas
  devolvidas em uma só mensagem; nota gravada aparece na busca seguinte.
- `baselines`: sem memória nunca inclui mensagens de outra sessão; contexto completo inclui todas.
- `juiz`: parse do structured output; abstenção avaliada pela regra.
- `rodar`: agrega acertos por categoria e tokens; grava os arquivos de saída; o argumento de
  configuração filtra.
- `dataset.json`: validação de esquema (cada pergunta tem categoria válida e resposta esperada;
  cada categoria tem entre 3 e 4 perguntas).

A rodada real da verificação é a "eval" do carve-out do CLAUDE.md: roda-se, compara-se o número,
não se faz assert.

## 9. Perguntas respondidas pelo autor (2026-10-27)

1. **ID do modelo:** `claude-sonnet-5-5`.
2. **Rodada paga:** faz parte da entrega. Chave do autor autorizada para uma rodada. A chave
   precisa estar em `.env` ou no ambiente antes da task de verificação; se não estiver, a
   execução para nessa task e pede ao autor.
3. **Revisão do dataset:** o autor lê depois. A validação de esquema nos testes basta para
   liberar a rodada.
4. **Material da semana 7:** está em `RESIDENCIA/Semana 7/Semana 7.docx`. Contém o modelo
   unificado dos quatro tipos de memória (CoALA e Du 2026), a tabela de políticas de controle
   (heurística, autocontrole por prompt, aprendida), as equivalências de nome entre artigos e a
   tabela "quando usar cada abordagem". O documento da entrega (§7) se encaixa nele: o exemplo
   passivo é a política **heurística** (resumir a cada n passos, como a fila FIFO do MemGPT) e
   o ativo é **autocontrole por prompt** (funções de memória do MemGPT). O ciclo
   escrever → gerenciar → ler de Du é o vocabulário para descrever os dois.

## 10. O que fica para o construtor

- Texto exato dos system prompts, do prompt do resumidor e do prompt do juiz (em PT-BR).
- Conteúdo do `dataset.json`, dentro das regras de §6.1.
- Esquema das tabelas SQLite.
- Formato de impressão do chat interativo.
- Valores padrão de K, top-n, limite de palavras e limite de iterações, desde que configuráveis.
- Organização interna dos módulos, desde que a árvore de §4 seja respeitada.
