# Experimento 2: replicação do MemoryAgentBench, direta e na OAB

Segundo experimento da residência de Pedro Coelho (Bacharelado em IA, UFG), cujo tema é a gestão
de memória de longo prazo para agentes LLM. O Experimento 1 (pasta irmã `../Experimento 1/`)
comparou memória passiva e ativa numa conversa sintética. Este replica duas afirmações do
MemoryAgentBench (Hu, Wang e McAuley, ICLR 2026) para ver se o autor estava certo:

- H1: o RAG vence o contexto longo quando é preciso achar um trecho preciso.
- H2: o contexto longo vence o RAG quando é preciso aprender com exemplos.

Cada afirmação passa por duas etapas: a direta (código, dados e protocolo do autor, com o
GPT-4o-mini) e a conceitual (questões da OAB e corpus de leis do projeto Atlas, com o GPT-4o-mini e
o Haiku 5.5). O código dos agentes é copiado do repositório do autor (MIT), não reescrito.

Design: `docs/superpowers/specs/2026-11-10-replicacao-memoryagentbench-design.md`.
Plano: `docs/superpowers/plans/2026-11-10-replicacao-memoryagentbench.md` (executar com
`superpowers:subagent-driven-development`).

## Arquitetura

Ainda não há código. O plano define o layout; o desenho pede estas peças:

```
dados/          snapshot local do banco do Atlas (fora do git) + manifesto versionado
<pacote>/       agentes (contexto longo, RAG BM25, RAG embedding), clientes de LLM com
                livro-caixa de custo, carga dos dados do autor e da OAB, avaliação e bootstrap
tests/          pytest, tudo offline com clientes falsos
resultados/     saída das rodadas reais (versionada)
docs/           documento da entrega e docs/superpowers (specs e plans)
```

Atualize esta seção no mesmo commit em que o layout real existir.

## Comandos

```
uv sync                 # ambiente
uv run pytest           # GATE DE TESTES (offline, sem API)
```

Os comandos de rodada entram aqui quando existirem. Toda rodada real **gasta dinheiro**.

Segredos em `.env` (ignorado pelo git, ver `.gitignore` desta pasta): `SUPABASE_URL`
(`https://gqnejwbuloklbelsufnj.supabase.co`), `SUPABASE_SERVICE_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`.

## Convenções

- Herdadas do Experimento 1: o mais simples que ainda ensina; PT-BR com acentos em tudo que o
  modelo lê na OAB e em tudo que o professor lê; identificadores sem acento; testes nunca chamam
  API; saída real de LLM nunca tem assert de igualdade; commits em PT-BR com prefixo, sem
  `Co-Authored-By`, autor `Pedro Coelho <coelho@discente.ufg.br>`.
- **Fidelidade ao artigo primeiro.** Onde o artigo define algo (top-k 10, chunk 512 ou 4096, teto de
  saída, prompts do Apêndice D), copie do repositório do autor. Toda adaptação para a OAB vai para o
  documento com o motivo.
- **Comparações pareadas:** todas as configurações respondem as mesmas questões, e um mesmo modelo
  dentro de cada comparação.
- **Cache ligado** nas duas APIs, ao contrário do Experimento 1: o artigo calcula custos com cache, e
  sem ele o Haiku estoura o orçamento. A contagem registra tokens normais, cacheados e escritos no
  cache.
- **Sem fallbacks** para outro modelo e sem `tool_choice` forçado.
- **Banco do Atlas só para leitura.** Toda rodada lê do snapshot em `dados/`, nunca do banco.

## Armadilhas conhecidas

- **Tetos de gasto:** o Pedro travou nos painéis US$ 20 na OpenAI e US$ 30 na Anthropic. O
  livro-caixa no código para US$ 0,50 antes de cada trava e guarda US$ 4 da OpenAI para depurar a
  etapa direta. Rodar o piloto antes de qualquer rodada completa.
- **Rodadas reais pré-autorizadas** pelo Pedro (2026-11-10): não pedir OK a cada rodada, desde que
  dentro dos tetos e na ordem do plano.
- **Nunca `git add -A`:** `../Experimento 1/RESIDENCIA/` e `../Experimento 1/artigos/` (PDFs) estão
  fora do git de propósito, e `-A` pega a árvore inteira, mesmo rodado daqui. Commite por caminho.
- **Limite da Anthropic:** 2 milhões de tokens de entrada por minuto (conferido em 2026-11-10); o
  contexto longo do Haiku (~950 mil tokens) cabe.
- **Os modelos da OpenAI saem do ar:** GPT-4o-mini, GPT-4o e GPT-4.1-mini em 2027-01-14; GPT-5-mini
  em 2026-12-11. Todas as rodadas com OpenAI terminam até 2026-12-20. Claude 3.7 Sonnet e Gemini 2.0
  Flash, usados no artigo, já não existem.
- **Vazamento de gabarito no banco:** a tabela `embeddings` tem 2.210 linhas `source_type='questao'`
  (questões de 2010 a 2018 com gabarito). Nada do experimento usa essas linhas; o RAG da OAB indexa
  só `law_articles`.
- **Tokenizador do Haiku:** 1,71× mais tokens que o `o200k_base` do GPT-4o-mini no texto jurídico em
  português. Acima de 100K tokens por chamada, o preço do Haiku quintuplica.
- **Haiku 5.5:** `temperature`, `top_p` e `top_k` dão 400 a partir da versão de API 2026-10-21, e o
  raciocínio vem ligado por padrão (desligar com `thinking: {type: "disabled"}` e effort `high` ou
  abaixo).
- **PostgREST do Supabase devolve no máximo 1.000 linhas por página:** paginar com o cabeçalho
  `Range`.
- **A RPC `search_embeddings_text` usa E lógico** (`plainto_tsquery`): consultas longas devolvem
  quase nada. O BM25 do experimento é local.
- **Dados da OAB:** 18 questões com menos de quatro alternativas; nenhuma anulada marcada; prova
  2018-01 quase igual à 2018-25; `2022-02-3` e `2022-02-3-fixed`; rótulos de matéria pouco confiáveis
  depois de 2018; o CPC do corpus vem de uma edição de 2016 e tem um "artigo" de 12.841 tokens.
- No Windows, saída redirecionada estraga os acentos; rode com `PYTHONUTF8=1`. `python -I` ignora as
  variáveis `PYTHON*`, inclusive essa. Os dados no banco estão íntegros: o "�" é só o console.

# superpowers

A metodologia padrão de todo trabalho aqui é o plugin **superpowers**. Ele não é opcional e não é
"para tarefas grandes" — é como se trabalha.

## A regra que faz tudo funcionar

> Se há **1% de chance** de uma skill se aplicar ao que você está fazendo, invoque a skill.
> Antes de responder, antes de perguntar, antes de abrir arquivo.

Red flags — se você pensar qualquer uma destas frases, está racionalizando para pular a skill:
*"isso é só uma pergunta simples"*, *"preciso de mais contexto antes"*, *"deixa eu olhar o código
rapidinho"*, *"a skill é overkill pra isso"*, *"eu lembro dessa skill"*.

## O fluxo padrão

```
brainstorm → spec aprovada → plano escrito → worktree → 1 subagent por task
   → review de spec + review de qualidade por task → gate de testes → ship
```

| Fase | Skill | Saída |
|---|---|---|
| 0 | `superpowers:brainstorming` | `docs/superpowers/specs/YYYY-MM-DD-<topico>-design.md` |
| 1 | `superpowers:writing-plans` | `docs/superpowers/plans/YYYY-MM-DD-<feature>.md` |
| 2 (opcional) | `gstack:plan-eng-review` · `plan-ceo-review` · `plan-design-review` · `autoplan` | plano editado, decisões travadas |
| 3 (opcional) | `superpowers:using-git-worktrees` | branch isolada (≥3 tasks ou arquivo compartilhado) |
| 4 | `superpowers:subagent-driven-development` (mesma sessão) ou `executing-plans` (sessão paralela) | código + reviews por task |
| 5 | `superpowers:verification-before-completion` | evidência real de test/build/lint |
| 6 | gate de testes → `gstack:ship` → `superpowers:finishing-a-development-branch` → `gstack:land-and-deploy` | PR / produção |

Na fase 4, por task e em sequência: dispatch de um implementer **novo** (contexto limpo) → TDD →
commit → self-review → reviewer de **spec** → reviewer de **qualidade** → fix loop para issues
Critical/Important → próxima task. Ao fim, um code review amplo sobre toda a branch.

## Catálogo de skills

| Skill | Quando usar |
|---|---|
| `brainstorming` | **Antes de qualquer trabalho criativo.** Feature nova, componente novo, mudança de comportamento. Termina em spec aprovada. |
| `writing-plans` | Tem a spec e vai mexer em várias etapas. Antes de tocar em código. |
| `subagent-driven-development` | Executar o plano na mesma sessão, 1 subagent por task, review por task. |
| `executing-plans` | Executar o plano em sessão separada (tasks acopladas, ou humano no loop). |
| `dispatching-parallel-agents` | Fan-out de trabalho genuinamente paralelo. |
| `test-driven-development` | RED-GREEN-REFACTOR. É o default de como se escreve código aqui. |
| `systematic-debugging` | Bug reportado, erro 500, stack trace, "funcionava ontem". Sem causa raiz, sem fix. |
| `verification-before-completion` | Antes de dizer "pronto/passou/corrigido", antes de commit e de PR. |
| `requesting-code-review` / `receiving-code-review` | Pedir e absorver review de subagent. |
| `using-git-worktrees` | Isolar a feature em worktree próprio. |
| `finishing-a-development-branch` | Fechar a branch: merge, limpeza, remoção do worktree. |
| `writing-skills` | Criar skill nova (quando um fluxo se repete 3x, vira skill). |
| `using-superpowers` | Meta: obriga a checar skills antes de qualquer resposta. |

## gstack — sob demanda, nunca proativo

O gstack (48 skills, em `~/.claude/skills/gstack`) é usado **quando pedido explicitamente**:
`/plan-eng-review`, `/autoplan`, `/office-hours`, `/review`, `/codex`, `/qa`, `/browse`,
`/investigate`, `/cso`, `/health`, `/ship`, `/land-and-deploy`, `/design-consultation`,
`/design-shotgun`, `/design-html`, `/design-review`, `/context-save`, `/context-restore`, `/retro`.

O padrão continua sendo superpowers. Não invoque gstack por conta própria.

## Regras não-negociáveis

1. **Design antes de código.** `brainstorming` tem hard gate: nenhuma implementação, nenhum
   scaffold antes da spec aprovada. Vale até para o que parece trivial.
2. **Testes vêm junto, não depois.** Toda feature que toca superfície testável (endpoint, tool,
   migration/RPC/RLS, lógica de negócio, fluxo de frontend) gera **≥1 task de teste no plano**.
   Sem essa task, o plano não sai da fase de planejamento.
   *Carve-out:* mudança puramente de prompt/modelo/chunking não gera teste determinístico — a
   obrigação vira **eval** (rodar antes/depois e comparar o número). Saída de LLM não é
   determinística; nunca peça assert de igualdade exata nela.
3. **Gate de testes antes do ship. Vermelho bloqueia.** Sem exceção, sem "só dessa vez".
4. **Evidência antes de afirmação.** Se você não rodou o comando de verificação **nesta mensagem**,
   você não pode afirmar que passa. Sinais de alarme: "deve", "provavelmente", "parece que";
   comemorar antes de verificar; confiar no relatório de sucesso de um subagent sem olhar o diff.
5. **Sem causa raiz, sem fix.** Bug entra por `systematic-debugging` / `/investigate`, nunca por
   tentativa e erro.
6. **Contexto fresco por task.** Cada implementer é um subagent novo, que recebe exatamente o
   contexto construído para ele — nunca o histórico da sessão.
7. **Um repo, um `CLAUDE.md` de verdade.** Não é README. Contém: arquitetura em 15 linhas, comandos
   comuns, convenções que um dev novo violaria sem saber, e a lista de armadilhas conhecidas.
   Quando uma decisão surpreendente for tomada, ela vira uma linha ali no mesmo commit.

## Ao criar ou revisar um `CLAUDE.md` de projeto

**Toda vez** que criar ou revisar um `CLAUDE.md` sob `C:\projetos\`, inclua esta seção
`# superpowers` completa no fim do arquivo. Se já existir desatualizada, atualize.

Checklist de projeto novo:

- [ ] `git init` + primeiro commit
- [ ] `<repo>/.claude/settings.json` habilitando `superpowers@claude-plugins-official`
- [ ] `<repo>/CLAUDE.md`: o que é o projeto · arquitetura · comandos · convenções · armadilhas
      · **+ a seção `# superpowers` no fim**
- [ ] Criar `docs/superpowers/specs/` e `docs/superpowers/plans/` (versionados, com `archive/`)
- [ ] Definir o comando do gate de teste (`pytest -m "not integration"`, `npm test`, …) e escrever
      no `CLAUDE.md`
- [ ] Só quando o projeto amadurecer: `.claude/agents/` especializados e uma skill de pipeline
      própria do projeto

## Referência rápida

| Situação | O que rodar |
|---|---|
| Ideia crua, não sei se vale construir | `/office-hours` |
| Feature nova | `superpowers:brainstorming` → `writing-plans` → `/plan-eng-review` → executar |
| Tenho plano e quero implementar | `superpowers:subagent-driven-development` |
| Bug / erro 500 / stack trace | `superpowers:systematic-debugging` ou `/investigate` |
| Antes de dizer "pronto" | `superpowers:verification-before-completion` |
| Segunda opinião de outro modelo | `/codex` |
| Pronto pra abrir PR | gate de testes → `/ship` |
| PR pronto pra ir a produção | `/land-and-deploy` |
| Fechando a branch | `superpowers:finishing-a-development-branch` |
| Achei que estava tudo certo e não estava | `/health`, `/review`, `/cso` |
