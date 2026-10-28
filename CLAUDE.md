# Memória passiva e memória ativa com LLM

Projeto didático da Residência em IA (UFG, semana 8): dois exemplos de memória para um LLM que diferem só em quem controla a leitura e a escrita.
Passiva = o código decide (sumarização incremental). Ativa = o modelo decide (ferramentas `buscar_memoria` e `gravar_memoria`, estilo MemGPT).
Uma verificação em miniatura compara os dois contra duas baselines (sem memória e contexto completo). Spec e plano em `docs/superpowers/`.

## Estrutura

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

## Comandos

```
uv sync                                   # ambiente
uv run pytest                             # gate de testes (sem API); vermelho bloqueia o ship
uv run python -m memoria.passiva          # chat de terminal com o exemplo passivo
uv run python -m memoria.ativa            # chat de terminal com o exemplo ativo
uv run python -m memoria.avaliacao.rodar  # verificação completa (usa API, custa dinheiro)
```

## Convenções

- **Um exemplo = um arquivo.** `passiva.py` e `ativa.py` se leem de cima para baixo, sem camadas entre o exemplo e a chamada à API além do wrapper fino de `llm.py`. Sem flags nem caminhos alternativos que a entrega não usa.
- **Idioma:** tudo que o modelo ou o professor lê (prompts, descrições de ferramentas, docs, docstrings) é PT-BR com acentos; nomes de código em PT-BR sem acento.
- **Dependências:** só `anthropic` e `python-dotenv` (e `pytest` em dev). SQLite da biblioteca padrão.
- **Testes** usam o cliente fake de `tests/conftest.py` (fixture `fake` e os construtores `resposta_texto`, `resposta_ferramenta`, `resposta_recusa`, `erro_api`).

## Armadilhas

- **Testes nunca chamam a API; a rodada real custa dinheiro.** Só `memoria.avaliacao.rodar` fala com a API de verdade, e a rodada paga acontece uma vez, de propósito.

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
