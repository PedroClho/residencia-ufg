# Experimento 2: replicação do MemoryAgentBench, plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicar as afirmações H1 e H2 do MemoryAgentBench com o código e os dados do autor (etapa direta) e nas questões da OAB (etapa conceitual), dentro de US$ 20 na OpenAI e US$ 30 na Anthropic.

**Architecture:** Um pacote Python pequeno, `replica/`, com o núcleo do código do autor copiado (licença MIT) em `replica/autor/`, trocando só o cliente de LLM por um que passa por um livro-caixa com teto. Os dados da OAB são convertidos para o formato de dados do autor, para que o mesmo código rode as duas etapas. Um rodador grava uma linha JSONL por pergunta, retoma de onde parou e respeita a ordem de prioridade. Uma análise faz o bootstrap pareado e escreve a tabela de vereditos.

**Tech Stack:** Python 3.12, `uv`, `openai`, `anthropic` (1.x), `tiktoken`, `nltk`, `rank-bm25`, `numpy`, `pyarrow`, `huggingface-hub`, `python-dotenv`, `matplotlib`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-10-replicacao-memoryagentbench-design.md`

Todos os caminhos são relativos a `Experimento 2/`. Rode todos os comandos de dentro dessa pasta. Trabalhe na branch `experimento-2`, no checkout atual: o `.env` e `dados/` são arquivos locais fora do git. Se usar um worktree, copie o `.env` para dentro dele.

## Global Constraints

- Tetos de gasto: o Pedro travou nos painéis US$ 14 na OpenAI e US$ 30 na Anthropic (2026-11-10). O livro-caixa para US$ 0,50 antes de cada trava: `TETOS = {"openai": 13.5, "anthropic": 29.5}`, sem reserva. Se a trava da OpenAI subir para US$ 20, passa a `TETOS["openai"] = 19.5` com `RESERVA["openai"] = 4.0` para depurar a etapa direta.
- Prazo: todas as rodadas com modelos da OpenAI terminam até 2026-12-20 (o GPT-4o-mini sai da API em 2027-01-14).
- Modelos: `gpt-4o-mini`, `claude-haiku-5-5`, `text-embedding-3-small`. Sem fallback para outro modelo, sem `tool_choice` forçado.
- Testes nunca chamam API; saída real de LLM nunca tem assert de igualdade exata.
- Banco do Atlas só para leitura. Toda rodada lê do snapshot em `dados/`. Nada usa as linhas `source_type='questao'`: o snapshot nem baixa a tabela `embeddings`.
- PT-BR com acentos em tudo que o modelo lê na OAB e em tudo que o professor lê; identificadores sem acento. Os prompts do autor ficam em inglês, literais.
- Windows + `uv`. JSON sempre gravado e lido em UTF-8 com `ensure_ascii=False`. Saída redirecionada com `PYTHONUTF8=1`.
- Cache ligado nas duas APIs (automático na OpenAI; `cache_control` no bloco de contexto na Anthropic).
- Haiku 5.5: nunca enviar `temperature`; raciocínio desligado com `thinking={"type": "disabled"}`.
- Agentes da OpenAI com `temperature=0.7`, como nas configurações do autor.
- Commits em PT-BR com prefixo (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), autor `Pedro Coelho <coelho@discente.ufg.br>`, sem `Co-Authored-By`.
- O Experimento 1 (`../Experimento 1/`) não é alterado.
- O Pedro autorizou em 2026-11-10 as rodadas reais sem novo OK a cada uma, dentro dos tetos. A trava dos painéis é a última barreira; o livro-caixa é a primeira.

## Review Focus

1. Duas perguntas seguidas sobre o mesmo contexto precisam mandar o prefixo de contexto **idêntico byte a byte**. Se não, o cache não pega e o custo do contexto longo dobra. Teste: `test_prefixo_identico_entre_perguntas` (Task 4).
2. Erro de API no meio da rodada (429, 500, 400 por janela estourada): a linha é gravada com `erro`, a rodada segue para a próxima pergunta, e a retomada refaz só essa pergunta. Num 429 que sobra depois dos retries do SDK, o cliente espera e tenta de novo antes de desistir. Testes: `test_limite_de_taxa_espera_e_tenta_de_novo` (Task 2) e `test_erro_de_api_e_refeito_na_retomada` (Task 9).
3. Rodada interrompida (teto, Ctrl+C, queda de rede): retomar não repete chamada já paga nem conta custo duas vezes. Teste: `test_retomada_nao_repete_chamadas` (Task 9).
4. Contexto do Haiku maior que a janela de 1M, porque o tokenizador dele gera 1,71× mais tokens que o do autor. Teste estático: `test_limites_do_haiku_cabem_na_janela` (Task 9).
5. Acentos no Windows: todo JSON e JSONL faz ida e volta em UTF-8 sem perder "ç" e "É". Testes: `test_json_utf8_ida_e_volta` (Task 5) e `test_jsonl_do_rodador_preserva_acentos` (Task 9).

## Correções ao desenho

A leitura do código do autor (commit `538026089d1a8a8eff05121d0db89b388f360eba`) e dos dados dele (revisão `7ea066982b140a19337e17e60d45d4076e042faf` no Hugging Face) mudou estes pontos. A Task 1 os registra no desenho. Pela regra do próprio desenho ("seguir o artigo onde ele define algo"), os itens 1 a 6 não precisam de decisão nova. **O item 7 mudou o desenho; o Pedro aprovou em 2026-11-10.**

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
| Teto do livro-caixa | US$ 13,50 (trava de US$ 14) | US$ 29,50 (trava de US$ 30) |

**Pendente:** com a trava atual da OpenAI, nem as primárias cabem (~US$ 14,80, e mais se o cache não pegar 100%). A recomendação é subir a trava para US$ 20. Aí `TETOS["openai"] = 19.5` com reserva de US$ 4: as primárias cabem nos US$ 15,50 efetivos, e as secundárias dependem da reserva, liberada (`--usar-reserva`) depois que a etapa direta reproduzir ou fechar a janela de depuração. Se a trava ficar em US$ 14, a H2 direta volta a BANKING77 e CLINC150 (primárias ~US$ 10,60) e as secundárias da OpenAI entram até o teto. Em qualquer caso, o que não couber sai do fim da fila, e o documento registra.

## Mapa de arquivos

```
pyproject.toml
replica/
  __init__.py
  config.py           constantes: modelos, preços, tetos, revisões, fontes, números do artigo
  livro_caixa.py      teto de gasto persistente, cálculo de custo por provedor
  clientes.py         clientes OpenAI, Anthropic e de embeddings, todos passando pelo livro-caixa
  arquivos.py         JSON/JSONL em UTF-8
  conjuntos.py        Pergunta, Contexto; carga dos dados do autor e da OAB
  autor/              cópia do código do autor (MIT)
    LICENSE  NOTICE.md  templates.py (literal)  metricas.py  agente.py
  templates_oab.py    prompts da OAB em PT-BR e despacho de templates
  snapshot.py         leitura paginada do banco do Atlas para dados/
  oab.py              questões e leis da OAB → linhas no formato do autor
  reclassificar.py    matéria das provas de 2018 a 2023 pelo GPT-4o-mini
  avaliacao.py        acerto por fonte (MH-QA, MCC, OAB)
  rodar.py            configurações em ordem de prioridade, rodador, piloto, portão
  analise.py          bootstrap pareado, vereditos, reprodução, tabela e gráfico
tests/                um arquivo por módulo, fakes em conftest.py
dados/                fora do git, exceto dados/manifesto.json
resultados/           versionado: rodadas e livro_caixa.json
docs/conferencia-materias.md, docs/replicacao-memoryagentbench.md
```

---

### Task 1: Andaime, configuração e livro-caixa

**Files:**
- Create: `pyproject.toml`, `replica/__init__.py`, `replica/config.py`, `replica/livro_caixa.py`, `tests/__init__.py`, `tests/test_livro_caixa.py`
- Modify: `.gitignore` (acrescentar `dados/*` e `!dados/manifesto.json`), `CLAUDE.md` (seção Arquitetura: o layout acima), `docs/superpowers/specs/2026-11-10-replicacao-memoryagentbench-design.md` (as "Correções ao desenho" deste plano)

**Interfaces:**
- Produces, em `replica/config.py`:
  - `RAIZ: Path`, `DADOS = RAIZ / "dados"`, `RESULTADOS = RAIZ / "resultados"`, `LIVRO_CAIXA = RESULTADOS / "livro_caixa.json"`
  - `MODELO_OPENAI = "gpt-4o-mini"`, `MODELO_ANTHROPIC = "claude-haiku-5-5"`, `MODELO_EMBEDDING = "text-embedding-3-small"`
  - `TETOS = {"openai": 13.5, "anthropic": 29.5}`, `RESERVA = {"openai": 0.0, "anthropic": 0.0}` (ver Global Constraints)
  - `PRECOS` em USD por milhão de tokens:
    - `"gpt-4o-mini": {"entrada": 0.15, "entrada_cache": 0.075, "saida": 0.60}`
    - `"text-embedding-3-small": {"entrada": 0.02}`
    - `"claude-haiku-5-5": {"entrada": 0.10, "saida": 0.50, "entrada_longa": 0.50, "saida_longa": 2.50, "limiar_longo": 100_000, "mult_escrita_cache": 1.25, "mult_leitura_cache": 0.10}`
  - `PROVEDOR = {"gpt-4o-mini": "openai", "text-embedding-3-small": "openai", "claude-haiku-5-5": "anthropic"}`
  - `FATOR_TOKENS_HAIKU = 1.85` (margem sobre a razão medida de 1,71) e `JANELA_HAIKU = 1_000_000`
  - `AUTOR_COMMIT = "538026089d1a8a8eff05121d0db89b388f360eba"`, `HF_DATASET = "ai-hyz/MemoryAgentBench"`, `HF_REVISAO = "7ea066982b140a19337e17e60d45d4076e042faf"`
  - `FONTE_MHQA = "ruler_qa2_421K"`
  - `MCC_FONTES = ("icl_banking77_5900shot_balance", "icl_clinic150_7050shot_balance", "icl_nlu_8296shot_balance", "icl_trec_coarse_6600shot_balance", "icl_trec_fine_6400shot_balance")`
  - `NUMEROS_ARTIGO: dict[tuple[str, str], float]`, com chave `(fonte ou "mcc", agente)` e os valores do item 6 das Correções
  - `SEMENTE = 42`, `PROJETO_ATLAS = "gqnejwbuloklbelsufnj"`
- Produces, em `replica/livro_caixa.py`:
  - `class LimiteDeGasto(Exception)`
  - `custo_openai(modelo: str, tokens_entrada: int, tokens_cacheados: int, tokens_saida: int) -> float`; `tokens_entrada` inclui os cacheados
  - `custo_anthropic(modelo: str, sem_cache: int, escrita_cache: int, leitura_cache: int, saida: int) -> float`; usa a faixa longa quando `sem_cache + escrita_cache + leitura_cache > limiar_longo`
  - `custo_embedding(modelo: str, tokens: int) -> float`
  - `class LivroCaixa(caminho: Path, tetos: dict[str, float] = TETOS, reserva: dict[str, float] = RESERVA, usar_reserva: bool = False)`, com:
    - `.gasto(provedor) -> float`
    - `.teto_efetivo(provedor) -> float`
    - `.autorizar(provedor, custo_maximo) -> None`, que levanta `LimiteDeGasto`
    - `.registrar(provedor, custo) -> None`, que persiste na hora com escrita atômica (`os.replace`)

- [ ] **Step 1: Criar `pyproject.toml` e o ambiente**

`[project] name = "experimento2"`, `requires-python = ">=3.12"`. Dependências: `openai`, `anthropic>=1.0`, `tiktoken`, `nltk>=3.9`, `rank-bm25`, `numpy`, `pyarrow`, `huggingface-hub`, `python-dotenv`, `matplotlib`. Em `[dependency-groups] dev`: `pytest`. Pacote: `replica`.

Run: `uv sync`
Expected: termina sem erro e cria `.venv/`.

- [ ] **Step 2: Escrever os testes que falham**

```python
from pytest import approx, raises
from replica.livro_caixa import LivroCaixa, LimiteDeGasto, custo_openai, custo_anthropic, custo_embedding

def test_custo_openai_sem_cache():
    assert custo_openai("gpt-4o-mini", 1_000_000, 0, 1_000_000) == approx(0.75)

def test_custo_openai_com_cache():
    assert custo_openai("gpt-4o-mini", 1_000_000, 1_000_000, 0) == approx(0.075)

def test_custo_anthropic_faixa_curta():
    assert custo_anthropic("claude-haiku-5-5", 50_000, 0, 0, 0) == approx(0.005)

def test_custo_anthropic_faixa_longa_com_leitura_de_cache():
    assert custo_anthropic("claude-haiku-5-5", 10_000, 0, 200_000, 1_000) == approx(0.0175)

def test_custo_anthropic_escrita_de_cache():
    assert custo_anthropic("claude-haiku-5-5", 1_000, 150_000, 0, 0) == approx(0.09425)

def test_custo_embedding():
    assert custo_embedding("text-embedding-3-small", 1_000_000) == approx(0.02)

def test_autorizar_recusa_o_que_passaria_do_teto(tmp_path):
    livro = LivroCaixa(tmp_path / "l.json", tetos={"openai": 1.0}, reserva={"openai": 0.0})
    livro.registrar("openai", 0.9)
    with raises(LimiteDeGasto):
        livro.autorizar("openai", 0.2)
    livro.autorizar("openai", 0.05)

def test_reserva_reduz_o_teto_ate_ser_liberada(tmp_path):
    tetos, reserva = {"openai": 20.0}, {"openai": 4.0}
    livro = LivroCaixa(tmp_path / "l.json", tetos=tetos, reserva=reserva)
    livro.registrar("openai", 15.5)
    with raises(LimiteDeGasto):
        livro.autorizar("openai", 1.0)
    LivroCaixa(tmp_path / "l.json", tetos=tetos, reserva=reserva, usar_reserva=True).autorizar("openai", 1.0)

def test_livro_persiste_entre_instancias(tmp_path):
    LivroCaixa(tmp_path / "l.json").registrar("anthropic", 1.25)
    assert LivroCaixa(tmp_path / "l.json").gasto("anthropic") == approx(1.25)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_livro_caixa.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'replica.livro_caixa'`

- [ ] **Step 4: Implementar `replica/config.py` e `replica/livro_caixa.py`** com as interfaces acima

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_livro_caixa.py -v`
Expected: 9 passed

- [ ] **Step 6: Registrar as Correções no desenho, atualizar o `CLAUDE.md` e o `.gitignore`**

No desenho:
- Uma seção nova "Correções depois da leitura do código do autor (2026-11-10)" com os itens 1 a 13 deste plano.
- Ajustar os trechos que elas contradizem: temperatura em §4; chunk e teto de saída em §4; datasets da H2 direta em §2; tabela de orçamento em §6.

No `CLAUDE.md`:
- Arquitetura com o mapa de arquivos.
- "Plano:" apontando para este arquivo.
- Uma armadilha nova: "o agente de embedding do autor esquece o que passa de ~285 mil tokens; é proposital (fidelidade)".

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock replica tests .gitignore CLAUDE.md docs/superpowers/specs
git commit -m "feat: andaime, configuração e livro-caixa com teto de gasto"
```

---

### Task 2: Clientes de LLM com teto

**Files:**
- Create: `replica/clientes.py`, `tests/conftest.py`, `tests/test_clientes.py`

**Interfaces:**
- Consumes: `LivroCaixa`, `LimiteDeGasto`, as funções `custo_*` e `config.PRECOS`/`PROVEDOR`/`FATOR_TOKENS_HAIKU` (Task 1)
- Produces:
  - `@dataclass class Resposta`, com `texto: str`, `tokens_entrada: int`, `tokens_cacheados: int`, `tokens_escritos_cache: int`, `tokens_saida: int`, `custo_usd: float` e `erro: str | None = None`.
  - `class ClienteOpenAI(modelo: str, livro: LivroCaixa, temperatura: float | None = 0.7, bruto=None, dormir=time.sleep)`
  - `class ClienteAnthropic(modelo: str, livro: LivroCaixa, bruto=None, dormir=time.sleep)`
  - Os dois expõem `conversar(system: str | None, mensagem: str, max_tokens: int, prefixo_cacheavel: str | None = None) -> Resposta`. O texto do usuário é sempre `(prefixo_cacheavel or "") + mensagem`.
  - `class ClienteEmbeddings(livro: LivroCaixa, modelo: str = MODELO_EMBEDDING, bruto=None, lote: int = 256)` com `embutir(textos: list[str]) -> numpy.ndarray`, de forma `(n, d)`.
  - `fabricar_cliente(modelo: str, livro: LivroCaixa, temperatura: float | None) -> ClienteOpenAI | ClienteAnthropic`
- Comportamento fixado:
  - **Antes de cada chamada:** estima os tokens de entrada com `tiktoken.get_encoding("o200k_base")` sobre `system + prefixo + mensagem` (×`FATOR_TOKENS_HAIKU` na Anthropic) e chama `livro.autorizar(provedor, custo_maximo)`. O custo máximo considera entrada sem cache mais `max_tokens` de saída.
  - **Depois:** calcula o custo pelo `usage` real e chama `livro.registrar`.
  - **OpenAI:** `bruto.chat.completions.create(model, messages=[{"role": "system", ...}, {"role": "user", ...}], temperature, max_tokens)`. Os tokens cacheados vêm de `usage.prompt_tokens_details.cached_tokens`, com `None` tratado como 0.
  - **Anthropic:** `bruto.messages.create(model, max_tokens, system, messages, thinking={"type": "disabled"})`, sem `temperature`. Com prefixo, o conteúdo é `[{"type": "text", "text": prefixo, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": mensagem}]`; sem prefixo, o conteúdo é a string `mensagem`. Os tokens vêm de `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` e `output_tokens`.
  - **Erros:** qualquer exceção, exceto `LimiteDeGasto`, vira `Resposta(texto="", erro="[erro: api]", ...zeros)`, e a `repr` vai para o stderr. Antes disso, um erro de limite de taxa (`openai.RateLimitError` ou `anthropic.RateLimitError`) espera 60 s com `dormir` e tenta de novo, até 5 vezes. `stop_reason == "refusal"` vira `erro="[erro: recusa]"`; texto vazio vira `erro="[erro: resposta vazia]"`.
  - Os clientes reais usam `max_retries=8`.

- [ ] **Step 1: Escrever os fakes em `tests/conftest.py`**

`FakeOpenAI` e `FakeAnthropic` imitam a forma das respostas dos SDKs com `types.SimpleNamespace`, guardam os kwargs de cada chamada em `.chamadas` e aceitam uma função `responder(kwargs) -> (texto, usage)` para controlar a saída. `FakeOpenAI` tem também `.embeddings.create(model, input)`, que devolve vetores determinísticos normalizados por texto e `usage.prompt_tokens`. Fixture `livro(tmp_path)` com um `LivroCaixa` em arquivo temporário.

- [ ] **Step 2: Escrever os testes que falham**

```python
def test_openai_manda_mensagem_como_o_autor(livro, fake_openai):
    c = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake_openai)
    c.conversar("SYS", "pergunta", 50, prefixo_cacheavel="CONTEXTO\n")
    kw = fake_openai.chamadas[0]
    assert kw["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "CONTEXTO\npergunta"}]
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 50

def test_openai_registra_tokens_cacheados_e_custo(livro, fake_openai):  # usage: prompt=1000, cached=800, completion=5
    r = ClienteOpenAI("gpt-4o-mini", livro, bruto=fake_openai).conversar("S", "m", 20)
    assert r.tokens_cacheados == 800
    assert livro.gasto("openai") == approx(custo_openai("gpt-4o-mini", 1000, 800, 5))

def test_anthropic_cacheia_o_prefixo_e_desliga_o_raciocinio(livro, fake_anthropic):
    ClienteAnthropic("claude-haiku-5-5", livro, bruto=fake_anthropic).conversar("SYS", "pergunta", 20, prefixo_cacheavel="CTX\n")
    kw = fake_anthropic.chamadas[0]
    assert kw["messages"][0]["content"][0] == {"type": "text", "text": "CTX\n", "cache_control": {"type": "ephemeral"}}
    assert kw["messages"][0]["content"][1]["text"] == "pergunta"
    assert kw["thinking"] == {"type": "disabled"} and "temperature" not in kw and kw["system"] == "SYS"

def test_anthropic_sem_prefixo_manda_texto_simples(livro, fake_anthropic): ...  # content == "pergunta"
def test_teto_recusa_antes_de_chamar_a_api(livro, fake_openai): ...  # livro quase no teto -> raises LimiteDeGasto; fake_openai.chamadas == []
def test_recusa_vira_erro(livro, fake_anthropic): ...  # stop_reason "refusal" -> r.erro == "[erro: recusa]" and r.texto == ""
def test_excecao_vira_erro_de_api_sem_custo(livro, fake_openai): ...  # responder levanta RuntimeError -> r.erro == "[erro: api]"; livro.gasto("openai") == 0
def test_limite_de_taxa_espera_e_tenta_de_novo(livro, fake_openai): ...  # 2 RateLimitError falsos e depois sucesso -> dormir chamado 2x com 60; r.erro is None
def test_embeddings_em_lotes_registram_custo(livro, fake_openai): ...  # 600 textos, lote 256 -> 3 chamadas; shape (600, d); gasto > 0
```

No teste de limite de taxa, injete a classe de erro: o cliente aceita `erros_de_taxa: tuple[type[Exception], ...]` no construtor, com padrão `(openai.RateLimitError,)` ou `(anthropic.RateLimitError,)`.

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_clientes.py -v`
Expected: FAIL com `ImportError` em `replica.clientes`

- [ ] **Step 4: Implementar `replica/clientes.py`** com as interfaces e o comportamento acima

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 6: Commit**

```bash
git add replica/clientes.py tests/conftest.py tests/test_clientes.py
git commit -m "feat: clientes OpenAI, Anthropic e de embeddings com teto e cache"
```

---

### Task 3: Cópia fiel do autor: templates e métricas

**Files:**
- Create: `replica/autor/__init__.py`, `replica/autor/LICENSE`, `replica/autor/NOTICE.md`, `replica/autor/templates.py`, `replica/autor/metricas.py`, `tests/test_autor_fidelidade.py`

**Interfaces:**
- Produces:
  - `replica.autor.templates` é a cópia **byte a byte** de `utils/templates.py`: `SYSTEM_MESSAGE`, `BASE_TEMPLATES`, `normalize_agent_name`, `get_template(sub_dataset, template_name, agent_name) -> str`.
  - `replica.autor.metricas` traz, copiados de `utils/eval_other_utils.py`: `normalize_answer`, `f1_score`, `drqa_exact_match_score`, `substring_exact_match_score`, `drqa_metric_max_over_ground_truths`, `parse_output`, `chunk_text_into_sentences(text, model_name="gpt-4o-mini", chunk_size=4096) -> list[str]`, `calculate_metrics(prediction, ground_truth_answers) -> dict`, `post_process(output: dict, answer, dataset_config: dict) -> tuple[dict, dict]` e os ramos `_process_icl_dataset`, `_process_ruler_memory_merging_dataset` e `default_post_process`.
- Mudanças permitidas em `metricas.py`, cada uma anotada no `NOTICE.md`:
  - Tirar o ROUGE de `calculate_metrics`, que passa a devolver só `exact_match`, `f1` e `substring_exact_match`.
  - Em `chunk_text_into_sentences`, baixar `punkt_tab` além de `punkt` (nltk ≥ 3.9).
  - Tirar a configuração de logging.
  - Em `post_process`, só os ramos `icl` e `ruler` mais o padrão.
- `NOTICE.md`: origem (URL do repositório, commit `AUTOR_COMMIT`), lista de arquivos e funções copiados, e toda mudança feita aqui e na Task 4.

- [ ] **Step 1: Clonar o repositório do autor fora do projeto e copiar**

```bash
git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench.git "$TEMP/mab_autor"
git -C "$TEMP/mab_autor" checkout 538026089d1a8a8eff05121d0db89b388f360eba
```
Copie `LICENSE` e `utils/templates.py` sem alterar nenhum byte. Copie as funções listadas para `metricas.py`.

- [ ] **Step 2: Escrever os testes que falham**

```python
SHA_TEMPLATES_AUTOR = "148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd"  # utils/templates.py @538026, finais de linha LF

def test_templates_sao_copia_literal():
    bruto = (Path(replica.autor.__file__).parent / "templates.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(bruto).hexdigest() == SHA_TEMPLATES_AUTOR

def test_metricas_do_autor():
    assert substring_exact_match_score("Yes, they were.", "yes")
    assert not drqa_exact_match_score("label: 43", "43")
    assert drqa_exact_match_score("43", "43")
    assert normalize_answer("The Chief of Protocol!") == "chief of protocol"
    assert parse_output("label: 43\noutra linha") == "label: 43"

def test_post_process_icl_e_ruler_como_no_autor():
    icl = {"sub_dataset": "icl_banking77_5900shot_balance"}
    assert post_process({"output": "43"}, ["43"], icl)[0]["exact_match"]
    assert not post_process({"output": "label: 43"}, ["43"], icl)[0]["exact_match"]
    assert post_process({"output": "It was yes."}, ["yes"], {"sub_dataset": "ruler_qa2_421K"})[0]["substring_exact_match"]

def test_chunker_respeita_o_tamanho(monkeypatch):  # sem rede: troca o sent_tokenize do nltk
    monkeypatch.setattr("nltk.sent_tokenize", lambda t: t.split(". "))
    monkeypatch.setattr("nltk.download", lambda *a, **k: True)
    pedacos = chunk_text_into_sentences(". ".join(["frase curta"] * 400), chunk_size=50)
    assert all(len(tiktoken.get_encoding("o200k_base").encode(p)) <= 50 for p in pedacos) and len(pedacos) > 1
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_autor_fidelidade.py -v`
Expected: FAIL até a cópia existir; o hash só bate com a cópia literal

- [ ] **Step 4: Completar a cópia e o `NOTICE.md`**

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 6: Commit**

```bash
git add replica/autor tests/test_autor_fidelidade.py
git commit -m "feat: cópia literal dos templates e das métricas do autor (MIT)"
```

---

### Task 4: Agentes do autor adaptados

**Files:**
- Create: `replica/autor/agente.py`, `tests/test_agente.py`
- Modify: `replica/autor/NOTICE.md` (listar as adaptações abaixo)

**Interfaces:**
- Consumes: `Resposta`, `ClienteOpenAI`/`ClienteAnthropic` (o método `conversar`), `ClienteEmbeddings.embutir` (Task 2); `replica.autor.templates.get_template` (Task 3)
- Produces: `class AgentWrapper(agent_config: dict, dataset_config: dict, cliente, embeddings=None, obter_template=get_template)`
  - `send_message(message: str, memorizing: bool = False, query_id=None, context_id=None) -> dict | str`
  - Ao perguntar, devolve o dicionário do autor (`output`, `input_len`, `output_len`, `memory_construction_time`, `query_time_len` e, nos RAG, `retrieval_context`) mais as chaves `tokens_cacheados`, `tokens_escritos_cache`, `custo_usd` e `erro`.
  - Tipos de agente pelo `agent_name`, como no autor: contém `Long_context_agent`; `rag_bm25`; `rag_text_embedding_3_small`.
- O que copiar de `agent.py@538026`, mantendo nomes e corpos:
  - `__init__` (cálculo de `input_length_limit`, tokenizador `tiktoken.encoding_for_model("gpt-4o-mini")`)
  - `_initialize_long_context_agent`
  - `_initialize_rag_agent`
  - `send_message`
  - `_handle_long_context_agent` (com o `time_stamp` de `time.strftime`)
  - `_query_long_context_agent`
  - `_truncate_context_if_needed`
  - `_handle_rag_agent`
  - `_process_rag_query` (só os dois handlers acima)
  - `_handle_bm25_rag`
  - `_extract_retrieval_query`
  - `_handle_embedding_rag`, com o prompt `"Memory {i+1}:\n{text}"` do `RAGSystem.answer_query`
- Adaptações, e só estas:
  1. Toda chamada de LLM vira `self.cliente.conversar(system_message, ..., max_tokens=self.max_tokens, prefixo_cacheavel=...)`. No contexto longo, `prefixo_cacheavel = self.context + "\n"` e a mensagem é a pergunta; o texto final é igual ao `full_message` do autor. No RAG, `prefixo_cacheavel=None` e a mensagem é o `ask_llm_message` do autor.
  2. O Haiku recebe o `system_message`. O caminho Claude do autor omitia o system; o Haiku não está no artigo, e ficamos com o mesmo prompt dos agentes da OpenAI.
  3. BM25: `rank_bm25.BM25Okapi([c.split() for c in self.chunks])` construído quando `context_id` muda, e `get_top_n(consulta.split(), self.chunks, n=self.retrieve_num)`.
  4. Embedding: `self.embeddings.embutir(self.chunks)` quando `context_id` muda, consulta com `embutir([consulta])`, e top-k por produto interno (vetores normalizados) com `numpy.argsort` descendente.
  5. `_extract_retrieval_query` ganha o padrão `r"Agora responda à questão:\s*(.*)"` além dos dois do autor. É usado também pelo caminho de embedding.
  6. Templates por `self.obter_template(...)`, para a OAB plugar os seus (Task 7).
  7. Sem torch, transformers, LangChain, Azure, nem gravação de arquivos em `./outputs/`.

- [ ] **Step 1: Escrever os testes que falham**

Use um `ClienteFalso` que grava `(system, mensagem, prefixo_cacheavel)` e devolve `Resposta(texto="ok", ...)`, e `monkeypatch.setattr("time.strftime", lambda *a: "2026-01-01 00:00:00")`.

```python
CFG_LC = {"agent_name": "Long_context_agent_gpt-4o-mini", "model": "gpt-4o-mini", "temperature": 0.7,
          "input_length_limit": 128000, "buffer_length": 4000, "output_dir": "x"}
CFG_DADOS = {"sub_dataset": "ruler_qa2_421K", "context_max_length": 524288, "dataset": "Accurate_Retrieval",
             "generation_max_length": 50, "chunk_size": 4096}

def test_contexto_longo_monta_a_mensagem_do_autor(...):
    tpl = get_template("ruler_qa2_421K", "memorize", "Long_context_agent_gpt-4o-mini")
    ag.send_message("c1", memorizing=True); ag.send_message("c2", memorizing=True); ag.send_message("Q?")
    esperado = (tpl.format(context="c1", time_stamp=T) + "\n" + tpl.format(context="c2", time_stamp=T)).strip() + "\n"
    assert cliente.chamadas[0].prefixo_cacheavel == esperado and cliente.chamadas[0].mensagem == "Q?"
    assert cliente.chamadas[0].system == SYSTEM_MESSAGE

def test_contexto_longo_descarta_o_inicio(...):  # input_length_limit=150, buffer 0, generation 50 -> limite 100
    # memoriza 30 chunks "palavraN ..." e pergunta; len(encode(ag.context)) <= 100 e ag.context termina com o fim do último chunk

def test_prefixo_identico_entre_perguntas(...):  # Review Focus 1
    ag.send_message("Q1?"); ag.send_message("Q2?")
    assert cliente.chamadas[0].prefixo_cacheavel == cliente.chamadas[1].prefixo_cacheavel

def test_bm25_recupera_k_trechos_deterministicamente(...):  # 5 chunks de palavras distintas, retrieve_num=2
    # a mensagem começa com "Memory 1:\n" + o chunk formatado que contém a palavra da pergunta; mesma saída em duas rodadas

def test_extracao_da_consulta():
    assert ag._extract_retrieval_query("x\n\n Now Answer the Question: Who?") == "Who?"
    assert ag._extract_retrieval_query("x\n\n Agora responda à questão: Quem?") == "Quem?"
    assert ag._extract_retrieval_query("sem padrão") == "sem padrão"

def test_embedding_ordena_por_cosseno(...):  # ClienteEmbeddings falso com vetores fixos; top-2 esperado
def test_embedding_esquece_alem_do_limite_do_autor(...):  # input_length_limit=1000, buffer 0, gen 0, chunk_size 512 -> ag.chunks guarda só os 1000//512 = 1 últimos chunks
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_agente.py -v`
Expected: FAIL com `ImportError` em `replica.autor.agente`

- [ ] **Step 3: Implementar `replica/autor/agente.py`** copiando os métodos listados e aplicando só as adaptações 1 a 7

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Commit**

```bash
git add replica/autor tests/test_agente.py
git commit -m "feat: agentes de contexto longo, BM25 e embedding do autor sobre os clientes com teto"
```

---

### Task 5: Arquivos UTF-8, conjuntos e dados do autor

**Files:**
- Create: `replica/arquivos.py`, `replica/conjuntos.py`, `tests/test_conjuntos.py`

**Interfaces:**
- Produces:
  - `arquivos.gravar_json(caminho: Path, obj) -> None` (UTF-8, `ensure_ascii=False`, `indent=1`, cria as pastas), `ler_json(caminho) -> object`, `anexar_jsonl(caminho, obj) -> None`, `ler_jsonl(caminho) -> list[dict]` (`[]` se o arquivo não existe).
  - `@dataclass class Pergunta`: `id: str`, `texto: str`, `gabarito: list[str]`, `ano: int | None = None`.
  - `@dataclass class Contexto`: `id: int`, `fonte: str`, `texto: str`, `perguntas: list[Pergunta]`.
  - `conjuntos.carregar_autor(fontes: Sequence[str], pasta: Path = DADOS / "autor", baixar=hf_hub_download) -> list[Contexto]`:
    - O split vem da fonte: `ruler*` → `Accurate_Retrieval`; `icl*` → `Test_Time_Learning`.
    - Arquivo `data/{split}-00000-of-00001.parquet`, baixado com `revision=HF_REVISAO`.
    - Uma linha por fonte, nesta ordem, com `id` = posição.
    - Perguntas `Pergunta(id=f"{fonte}:{i}", texto=questions[i], gabarito=list(answers[i]))`.
    - `assert len(context) > 2000`, como o autor.
  - `conjuntos.carregar_oab(nome: str, pasta: Path = DADOS / "oab") -> list[Contexto]`: lê `{nome}.json` (linhas no formato do autor, Task 7). `metadata.qa_pair_ids` vira `Pergunta.id`, `metadata.anos` vira `Pergunta.ano`, `answers[i]` vira `gabarito`.

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_json_utf8_ida_e_volta(tmp_path):  # Review Focus 5
    gravar_json(tmp_path / "a.json", {"matéria": "Ética Profissional", "x": "ção"})
    assert ler_json(tmp_path / "a.json") == {"matéria": "Ética Profissional", "x": "ção"}
    assert "Ética" in (tmp_path / "a.json").read_text(encoding="utf-8")

def test_carregar_autor_filtra_a_fonte_e_numera_perguntas(tmp_path):
    # grava um parquet mínimo com pyarrow: 2 linhas, metadata.source "ruler_qa2_421K" e "outra", context de 3000 chars
    # baixar=lambda *a, **k: str(caminho_do_parquet)
    ctx = carregar_autor(["ruler_qa2_421K"], pasta=tmp_path, baixar=falso)
    assert len(ctx) == 1 and ctx[0].perguntas[1].id == "ruler_qa2_421K:1" and ctx[0].perguntas[0].gabarito == ["yes"]

def test_carregar_varias_fontes_na_ordem(tmp_path): ...  # duas fontes icl_* -> ids 0 e 1 na ordem pedida
def test_carregar_oab_le_ids_e_anos(tmp_path): ...  # {nome}.json com metadata qa_pair_ids e anos -> Pergunta.id e .ano
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_conjuntos.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `replica/arquivos.py` e `replica/conjuntos.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Commit**

```bash
git add replica/arquivos.py replica/conjuntos.py tests/test_conjuntos.py
git commit -m "feat: conjuntos de dados e carga do dataset do autor na revisão fixada"
```

---

### Task 6: Snapshot do banco do Atlas

**Files:**
- Create: `replica/snapshot.py`, `tests/test_snapshot.py`

**Interfaces:**
- Consumes: `arquivos.gravar_json` (Task 5); `config.DADOS`, `PROJETO_ATLAS` (Task 1)
- Produces:
  - `TABELAS = {"questoes_oab": "id,exam_id,ano,numero_questao,materia,enunciado,alternativas,gabarito,anulada", "law_articles": "id,document_id,law_sigla,law_name,eixo,article_number,content"}`
  - `buscar_paginado(get_json, tabela: str, colunas: str, passo: int = 1000) -> list[dict]`. Chama `get_json(tabela, {"select": colunas, "order": "id.asc"}, f"{ini}-{ini+passo-1}")` até uma página vir com menos de `passo` linhas.
  - `get_json_supabase(url: str, chave: str) -> Callable`: `urllib` com os cabeçalhos `apikey`, `Authorization: Bearer` e `Range`; só GET.
  - `salvar_snapshot(pasta: Path, get_json) -> dict`. Grava `pasta/atlas/{tabela}.json` e `pasta/manifesto.json` = `{"projeto": PROJETO_ATLAS, "data": <ISO>, "tabelas": {tabela: {"linhas": n, "sha256": <do JSON gravado>}}}`.
  - `main() -> int`: lê `SUPABASE_URL` e `SUPABASE_SERVICE_KEY` do `.env` com `python-dotenv` e imprime as contagens, nunca a chave.

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_paginacao_junta_todas_as_paginas(): ...  # get_json falso com 2345 linhas -> 3 chamadas, 2345 linhas na ordem
def test_snapshot_grava_manifesto_e_nunca_pede_embeddings(tmp_path):
    pedidas = []
    m = salvar_snapshot(tmp_path, get_json=falso_que_registra(pedidas))
    assert set(pedidas) == {"questoes_oab", "law_articles"} and "embeddings" not in pedidas
    assert m["tabelas"]["questoes_oab"]["linhas"] == 3 and len(m["tabelas"]["law_articles"]["sha256"]) == 64
def test_snapshot_preserva_acentos(tmp_path): ...  # enunciado "Ética" volta igual de dados/atlas/questoes_oab.json
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_snapshot.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `replica/snapshot.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Commit**

```bash
git add replica/snapshot.py tests/test_snapshot.py
git commit -m "feat: snapshot só de leitura das questões e das leis do Atlas"
```

---

### Task 7: OAB: prompts, questões, leis e conjuntos

**Files:**
- Create: `replica/templates_oab.py`, `replica/oab.py`, `tests/test_oab.py`

**Interfaces:**
- Consumes: `arquivos` (Task 5); `autor.templates.get_template` e `normalize_agent_name` (Task 3); `config.SEMENTE` e `DADOS` (Task 1)
- Produces, em `replica/templates_oab.py` (texto exato, PT-BR):

```python
SYSTEM_OAB = "Você é um assistente prestativo que lê o contexto e o memoriza para consultas futuras."
SYSTEM_SEM_MEMORIA = "Você é um assistente prestativo."
INSTRUCAO_LETRA = "Responda apenas com a letra da alternativa correta (A, B, C ou D) e não escreva mais nada."
BASE_TEMPLATES_OAB = {
    "oab_h1": {
        "system": SYSTEM_OAB,
        "memorize": 'Diálogo entre Usuário e Assistente {time_stamp}\\n<Usuário> O contexto a seguir são os documentos que li: \n{context}\n <Assistente> Aprendi os documentos e vou responder à questão que você fizer.',
        "query": {
            "long_context_agent": "Responda à questão com base nos documentos memorizados. " + INSTRUCAO_LETRA + " \n\nQuestão: {question} \n\n Resposta:",
            "rag_agent": "Responda à questão com base nos documentos memorizados. " + INSTRUCAO_LETRA + " \n\n Agora responda à questão: {question}",
        },
    },
    "oab_h2": {
        "system": SYSTEM_OAB,
        "memorize": 'Diálogo entre Usuário e Assistente {time_stamp} \\n<Usuário> O contexto a seguir são os exemplos que estudei: \n{context}\n <Assistente> Estudei os exemplos e vou responder à questão que você fizer.',
        "query": {
            "long_context_agent": "Use os exemplos do contexto, questões da prova da OAB com o gabarito, para responder à nova questão. Responda apenas \"Gabarito: {{letra}}\" e nada mais. \n\n{question} \n\n Gabarito:",
            "rag_agent": "Use os exemplos do contexto, questões da prova da OAB com o gabarito, para responder à nova questão. Responda apenas \"Gabarito: {{letra}}\" e nada mais. \n\nQuestão:{question} \n\n Gabarito:",
        },
    },
}
TEMPLATE_SEM_MEMORIA = "Responda à questão da prova da OAB. " + INSTRUCAO_LETRA + " \n\nQuestão: {question} \n\n Resposta:"
def obter_template(sub_dataset: str, template_name: str, agent_name: str) -> str: ...
# "oab_h1*" -> BASE_TEMPLATES_OAB["oab_h1"]; "oab_h2*" -> ["oab_h2"]; qualquer outro -> autor.templates.get_template
```

Os templates imitam a estrutura dos do autor: `ruler_qa` para a H1 e `in_context_learning` para a H2, inclusive o `\\n` literal que o autor tem no `memorize`.

- Produces, em `replica/oab.py`:
  - `LETRAS = ("A", "B", "C", "D")`
  - `formatar_questao(q: dict) -> str` = `enunciado + "\n\n" + "\n".join(f"{a['letra']}) {a['texto']}")`
  - `extrair_letra(texto: str | None) -> str | None`. Padrões em ordem:
    1. `(?i)gabarito\s*:?\s*\(?\s*([abcd])\b`
    2. `(?i)\b(?:alternativa|letra|op[cç][aã]o)\s*\(?\s*([abcd])\b`
    3. `^\W*([ABCD])\s*(?:[).:\-]|\*\*|$)`
  - `questao_valida(q: dict) -> bool`: exatamente as alternativas A a D, gabarito em `LETRAS`, enunciado não vazio.
  - `deduplicar(qs: list[dict], limiar: float = 0.8) -> list[dict]`: tira a prova `2018-01` inteira (quase igual à `2018-25`); depois tira quase-duplicatas pelo `difflib.SequenceMatcher(None, a, b).ratio() > limiar` sobre o enunciado, ficando a de menor `id`.
  - `MATERIA_PARA_LEIS: dict[str, tuple[str, ...]]`:
    - Ética Profissional → `("EAOAB", "RG-EAOAB", "CED")`
    - Direito Constitucional → `("CF",)`
    - Direitos Humanos → `("CADH", "DUDH", "Lei 9.474")`
    - Direito Tributário → `("CTN",)`
    - Direito Administrativo → `("Lei 8.112", "Lei 14.133", "Lei 8.987", "Lei 8.429", "Lei 9.784", "Lei 12.016", "Lei 4.717")`
    - Direito Ambiental → `("Lei 12.651", "Lei 6.938", "Lei 9.605")`
    - Direito Civil → `("CC",)`
    - Direito da Criança e do Adolescente → `("ECA",)`
    - Direito do Consumidor → `("CDC",)`
    - Direito Empresarial → `("Lei 6.404", "Lei 11.101")`
    - Direito Processual Civil → `("CPC",)`
    - Direito Penal → `("CP",)`
    - Direito Processual Penal → `("CPP",)`
    - Direito do Trabalho e Direito Processual do Trabalho → `("CLT",)`
    - Direito Eleitoral → `("Lei 4.737", "Lei 9.504", "Lei 9.096", "LC 64")`
    - Direito Previdenciário → `("Lei 8.212", "Lei 8.213", "LOAS")`
    - Direito Financeiro → `("LRF", "Lei 4.320")`
  - `MATERIAS_FORA_H1 = ("Filosofia do Direito", "Direito Internacional")`
  - `ordenar_artigos(artigos: list[dict], siglas: tuple[str, ...]) -> list[dict]`. Chave: índice da sigla em `siglas`; depois o número lido de `^\s*Art\.?\s*(\d+(?:\.\d{3})*)\s*(?:º|°|o)?\s*(?:-\s*([A-Z]))?` no `content` (sem os pontos de milhar) e o sufixo; artigos sem número vão por último, por `id`.
  - `amostrar_h1(qs: list[dict], materias: dict[str, str], n: int = 300, semente: int = SEMENTE) -> list[dict]`:
    - Base: válidas, de 2018 a 2023, deduplicadas.
    - Matéria reclassificada (`materias[q["id"]]`), fora de `MATERIAS_FORA_H1` e presente em `MATERIA_PARA_LEIS`.
    - Alocação proporcional por matéria pelo maior resto.
    - Sorteio com `random.Random(semente)` sobre os ids ordenados.
  - `montar_h1(amostra, materias, artigos) -> list[dict]`:
    - Uma linha no formato do autor por tupla de leis, na ordem de `MATERIA_PARA_LEIS`.
    - `{"context": "\n\n".join(content dos artigos ordenados), "questions": [formatar_questao(q)], "answers": [[gabarito]], "metadata": {"source": f"oab_h1_{i:02d}", "qa_pair_ids": [ids], "anos": [anos], "leis": list(siglas)}}`
  - `montar_h1_corpus(amostra, artigos) -> list[dict]`: uma linha só, com todas as leis de `MATERIA_PARA_LEIS` e `source` `"oab_h1_corpus"`.
  - `separar_h2(qs: list[dict]) -> tuple[list[dict], list[dict]]`:
    - Teste: válidas de 2024 e 2025.
    - Memória: válidas até 2023, deduplicadas, sem nenhuma quase-duplicata (ratio > 0,8) de questão de teste, ordenadas por `(exam_id, numero_questao)`.
  - `montar_h2(memoria, teste) -> list[dict]`: uma linha com `context` = `"".join(f"{formatar_questao(q)}\nGabarito: {q['gabarito']}\n\n")` (o formato `"{frase}\nlabel: {rótulo}\n\n"` do autor) e `source` `"oab_h2"`.
  - `montar_sem_memoria(qs) -> list[dict]`: uma linha com `context` vazio, todas as válidas, `source` `"oab_sem_memoria"`.
  - `main() -> int`:
    - Lê `dados/atlas/*.json` e `dados/materias_2018_2023.json`.
    - Grava `dados/oab/{h1,h1_corpus,h2,sem_memoria}.json`.
    - Imprime o número de questões e contextos de cada conjunto e o tamanho de cada contexto em tokens `o200k_base`.

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_formatar_questao():
    q = {"enunciado": "Qual?", "alternativas": [{"letra": L, "texto": f"t{L}"} for L in "ABCD"]}
    assert formatar_questao(q) == "Qual?\n\nA) tA\nB) tB\nC) tC\nD) tD"

@pytest.mark.parametrize("saida,letra", [("B", "B"), ("Gabarito: c", "C"), ("A resposta correta é a alternativa D.", "D"),
    ("**A)** porque", "A"), ("Letra B", "B"), ("C.", "C"), ("", None), ("Não sei", None), ("E", None),
    ("A resposta correta é D", None), (None, None)])
def test_extrair_letra(saida, letra):
    assert extrair_letra(saida) == letra

def test_obter_template_despacha_oab_e_autor():
    assert obter_template("oab_h1_03", "query", "Simple_rag_bm25").endswith("Agora responda à questão: {question}")
    assert obter_template("ruler_qa2_421K", "system", "Simple_rag_bm25").startswith("You are a helpful assistant")

def test_questao_valida_rejeita_incompletas(): ...  # 3 alternativas, gabarito "E" ou enunciado vazio -> False
def test_deduplicar_tira_2018_01_e_quase_iguais(): ...
def test_ordenar_artigos(): ...  # "Art. 10.", "Art. 2º", "Art. 1.000.", "Art. 2º-A", "Parágrafo solto" -> 2º, 2º-A, 10, 1.000, solto
def test_amostra_h1_estratificada_e_deterministica(): ...  # 1000 falsas -> 300; contagens pelo maior resto; mesma saída duas vezes; nenhuma de MATERIAS_FORA_H1
def test_h1_agrupa_por_lei_e_nao_vaza(): ...  # Trabalho e Processual do Trabalho no mesmo contexto; enunciado[:80] de cada questão ausente do context; questions[i] == formatar_questao(q) e answers[i] == [q["gabarito"]]
def test_h2_sem_vazamento_e_em_ordem(): ...  # teste só 2024-2025; memória só até 2023, cronológica, com "Gabarito: X", sem quase-duplicata de teste
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_oab.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `replica/templates_oab.py` e `replica/oab.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Commit**

```bash
git add replica/templates_oab.py replica/oab.py tests/test_oab.py
git commit -m "feat: prompts da OAB e conversão das questões e leis para o formato do autor"
```

---

### Task 8: Reclassificação de matéria (2018 a 2023)

**Files:**
- Create: `replica/reclassificar.py`, `tests/test_reclassificar.py`

**Interfaces:**
- Consumes: `fabricar_cliente`, `ClienteOpenAI` (Task 2); `oab.questao_valida`, `oab.deduplicar`, `oab.formatar_questao` (Task 7); `arquivos` (Task 5)
- Produces:
  - `MATERIAS: tuple[str, ...]`: as 18 matérias de `MATERIA_PARA_LEIS` (contando Trabalho e Processual do Trabalho separadas), mais Filosofia do Direito e Direito Internacional, todas com acento e exatamente como no banco.
  - `prompt_classificacao(q: dict) -> tuple[str, str]` devolve `(system, mensagem)`:
    - `system = "Você classifica questões da prova da OAB pela matéria."`
    - `mensagem = "Classifique a questão abaixo em exatamente uma destas matérias:\n" + "\n".join(f"- {m}" for m in MATERIAS) + "\n\nResponda só com o nome da matéria, exatamente como está na lista.\n\nQuestão:\n" + formatar_questao(q)`
  - `interpretar(saida: str) -> str | None`: casa ignorando caixa, espaços e um ponto final; fora da lista vira `None`.
  - `reclassificar(qs: list[dict], cliente) -> tuple[dict[str, str], list[str]]` devolve `(mapa id → matéria, ids sem resposta válida)`. Quem fica sem resposta válida mantém o rótulo do banco. `max_tokens=20`.
  - `amostra_conferencia(qs, mapa, n: int = 30, semente: int = SEMENTE) -> str`: tabela Markdown `| id | trecho (150 caracteres) | rótulo do banco | rótulo novo | ok? |`, com a última coluna vazia para o Pedro.
  - `main() -> int`:
    - Questões válidas de 2018 a 2023, deduplicadas.
    - `ClienteOpenAI(MODELO_OPENAI, LivroCaixa(LIVRO_CAIXA), temperatura=0.0)`.
    - Grava `dados/materias_2018_2023.json` e `docs/conferencia-materias.md`.

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_interpretar_aceita_so_a_lista():
    assert interpretar(" direito civil. ") == "Direito Civil"
    assert interpretar("Direito Espacial") is None

def test_reclassificar_usa_o_cliente_e_mantem_rotulo_quando_invalido(): ...  # falso devolve "Direito Penal" e depois "???"
def test_amostra_conferencia_tem_30_linhas_e_e_deterministica(): ...
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_reclassificar.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `replica/reclassificar.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Commit**

```bash
git add replica/reclassificar.py tests/test_reclassificar.py
git commit -m "feat: reclassificação de matéria das provas de 2018 a 2023"
```

---

### Task 9: Avaliação e rodador

**Files:**
- Create: `replica/avaliacao.py`, `replica/rodar.py`, `tests/test_avaliacao.py`, `tests/test_rodar.py`
- Modify: `CLAUDE.md` (seção Comandos, com os comandos desta task e das Tasks 6 a 8 e 10)

**Interfaces:**
- Consumes: `AgentWrapper` (Task 4); `carregar_autor`, `carregar_oab`, `Contexto` (Task 5); `obter_template`, `TEMPLATE_SEM_MEMORIA`, `SYSTEM_SEM_MEMORIA`, `extrair_letra` (Task 7); `fabricar_cliente`, `ClienteEmbeddings`, `LivroCaixa`, `LimiteDeGasto` (Tasks 1 e 2); `autor.metricas.post_process` e `chunk_text_into_sentences` (Task 3)
- Produces, em `replica/avaliacao.py`: `avaliar(fonte: str, saida: str, gabarito: list[str]) -> tuple[bool, dict]`.
  - `oab*`: a letra extraída igual a `gabarito[0]`; métricas `{"letra", "letra_invalida"}`.
  - `icl*`: `post_process` do autor, com acerto = `exact_match`; métricas mais `"exact_match_flexivel"` (primeiro `-?\d+` depois de um `label:` opcional igual ao gabarito).
  - `ruler*`: `post_process` do autor, com acerto = `substring_exact_match`.
  - Outra fonte levanta `ValueError`.
- Produces, em `replica/rodar.py`:
  - `@dataclass(frozen=True) class Configuracao`, com os campos `nome`, `etapa`, `hipotese`, `papel`, `agente` (`"contexto_longo" | "rag_bm25" | "rag_embedding" | "sem_memoria"`), `modelo`, `conjunto`, `chunk_size`, `retrieve_num`, `input_length_limit`, `buffer_length`, `generation_max_length`, `context_max_length` e `temperatura`.
  - `CONFIGURACOES: list[Configuracao]`: a tabela abaixo, nesta ordem, que é a ordem de prioridade.
  - `agent_config(cfg) -> dict`, com os `agent_name` do autor: `f"Long_context_agent_{modelo}"`, `"Simple_rag_bm25"` e `"Embedding_rag_text_embedding_3_small"`.
  - `dataset_config(cfg, fonte) -> dict` (`sub_dataset=fonte`, `chunk_size`, `generation_max_length`, `context_max_length`, `dataset`).
  - `carregar_conjunto(cfg) -> list[Contexto]`: `ruler_qa2_421K` → `carregar_autor([FONTE_MHQA])`; `mcc` → `carregar_autor(MCC_FONTES)`; `oab_*` → `carregar_oab(cfg.conjunto.removeprefix("oab_"))`.
  - `rodar_configuracao(cfg, contextos, fabricar, embeddings, pasta: Path, limite_perguntas: int | None = None) -> dict`:
    - Devolve `{"nome", "feitas", "erros", "custo_usd", "status"}`.
    - Por contexto: cria um `AgentWrapper` novo, memoriza `chunk_text_into_sentences(texto, chunk_size=cfg.chunk_size)` com `send_message(c, memorizing=True)` e pergunta com `obter_template(fonte, "query", agent_name).format(question=p.texto)`.
    - `sem_memoria`: chama o cliente direto com `SYSTEM_SEM_MEMORIA` e `TEMPLATE_SEM_MEMORIA`.
    - Grava uma linha por pergunta em `pasta/{nome}.jsonl`.
    - Pula perguntas que já têm linha sem `erro`; linhas com `erro` são refeitas e substituídas.
    - `LimiteDeGasto` termina com `status="parada: teto"`.
  - Esquema da linha: `{"configuracao", "fonte", "contexto_id", "pergunta_id", "ano", "consulta", "gabarito", "saida", "acerto", "metricas", "tokens_entrada", "tokens_cacheados", "tokens_escritos_cache", "tokens_saida", "custo_usd", "erro"}`.
  - `portao_haiku_h2(pasta: Path) -> tuple[bool, float]`: acurácia do `oab-sem-memoria-haiku` nas perguntas de 2024 e 2025; libera se for menor que 0,90.
  - `projecao(linhas: list[dict], n_contextos: int, n_perguntas: int) -> float` = `custo da primeira linha × n_contextos + média das demais × (n_perguntas − n_contextos)`
  - `main(argv=None) -> int` com a CLI `python -m replica.rodar [NOMES ...] [--rodada DIR] [--piloto N] [--usar-reserva] [--haiku-h2 {1m,600k}]`:
    - Sem nomes, roda todas na ordem, incluindo só a variante do Haiku H2 escolhida (padrão `1m`) e aplicando o portão.
    - Teto de um provedor pula só as configurações daquele provedor.
    - `--piloto N` limita cada configuração a N perguntas (as primeiras do primeiro contexto) e grava `projecao.md` com o custo projetado por configuração e por provedor.
    - No fim, imprime o gasto do livro-caixa.

`temperatura` = 0.7 nas configurações da OpenAI e `None` nas do Haiku. `ctx_max` da OAB = 2.000.000. Limites de RAG: BM25 com `10_000_000/200`; embedding com `300_000/15_000`. `etapa` sai do prefixo do nome (`direta` ou `oab`). `hipotese`: `h1` para `mhqa` e `oab_h1*`, `h2` para `mcc` e `oab_h2`, `linha_de_base` para as sem memória. `papel` é a coluna da tabela.

| # | nome | papel | agente | modelo | conjunto | chunk | k | limite/buffer | saída | ctx_max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | direta-mhqa-lc-4omini | primária | contexto_longo | gpt-4o-mini | ruler_qa2_421K | 4096 | – | 128000/4000 | 50 | 524288 |
| 2 | direta-mhqa-bm25-4omini | primária | rag_bm25 | gpt-4o-mini | ruler_qa2_421K | 512 | 10 | BM25 | 50 | 524288 |
| 3 | direta-mcc-lc-4omini | primária | contexto_longo | gpt-4o-mini | mcc | 4096 | – | 128000/4000 | 20 | 131072 |
| 4 | direta-mcc-bm25-4omini | primária | rag_bm25 | gpt-4o-mini | mcc | 4096 | 10 | BM25 | 20 | 131072 |
| 5 | direta-mhqa-emb-4omini | reprodução | rag_embedding | gpt-4o-mini | ruler_qa2_421K | 512 | 10 | embedding | 50 | 524288 |
| 6 | oab-sem-memoria-4omini | linha de base | sem_memoria | gpt-4o-mini | oab_sem_memoria | – | – | – | 20 | – |
| 7 | oab-sem-memoria-haiku | linha de base | sem_memoria | claude-haiku-5-5 | oab_sem_memoria | – | – | – | 20 | – |
| 8 | oab-h1-lc-4omini | primária | contexto_longo | gpt-4o-mini | oab_h1 | 4096 | – | 128000/4000 | 20 | 2M |
| 9 | oab-h1-bm25-4omini | primária | rag_bm25 | gpt-4o-mini | oab_h1 | 512 | 10 | BM25 | 20 | 2M |
| 10 | oab-h2-lc-4omini | primária | contexto_longo | gpt-4o-mini | oab_h2 | 4096 | – | 128000/4000 | 20 | 2M |
| 11 | oab-h2-bm25-4omini | primária | rag_bm25 | gpt-4o-mini | oab_h2 | 4096 | 10 | BM25 | 20 | 2M |
| 12 | oab-h1-emb-4omini | secundária | rag_embedding | gpt-4o-mini | oab_h1 | 512 | 10 | embedding | 20 | 2M |
| 13 | oab-h1-bm25-haiku | secundária | rag_bm25 | claude-haiku-5-5 | oab_h1 | 512 | 10 | BM25 | 20 | 2M |
| 14 | oab-h1-emb-haiku | secundária | rag_embedding | claude-haiku-5-5 | oab_h1 | 512 | 10 | embedding | 20 | 2M |
| 15–17 | oab-h1-bm25-4omini-k2 / -k5 / -k20 | ablação | rag_bm25 | gpt-4o-mini | oab_h1 | 512 | 2 / 5 / 20 | BM25 | 20 | 2M |
| 18 | oab-h1corpus-bm25-4omini | secundária | rag_bm25 | gpt-4o-mini | oab_h1_corpus | 512 | 10 | BM25 | 20 | 2M |
| 19 | oab-h2-bm25-haiku | secundária | rag_bm25 | claude-haiku-5-5 | oab_h2 | 4096 | 10 | BM25 | 20 | 2M |
| 20 | oab-h1-lc-haiku | secundária | contexto_longo | claude-haiku-5-5 | oab_h1 | 4096 | – | 1000000/480000 | 20 | 2M |
| 21 | oab-h2-lc-haiku | secundária, com portão | contexto_longo | claude-haiku-5-5 | oab_h2 | 4096 | – | 1000000/480000 | 20 | 2M |
| 21' | oab-h2-lc-haiku-600k | variante da 21 | contexto_longo | claude-haiku-5-5 | oab_h2 | 4096 | – | 1000000/670000 | 20 | 2M |
| 22 | oab-h2-emb-4omini | secundária | rag_embedding | gpt-4o-mini | oab_h2 | 4096 | 10 | embedding | 20 | 2M |
| 23 | oab-h2-emb-haiku | secundária | rag_embedding | claude-haiku-5-5 | oab_h2 | 4096 | 10 | embedding | 20 | 2M |
| 24 | direta-mcc-emb-4omini | reprodução | rag_embedding | gpt-4o-mini | mcc | 4096 | 10 | embedding | 20 | 131072 |

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_avaliar_por_fonte():
    assert avaliar("oab_h1_02", "Gabarito: B", ["B"])[0]
    assert not avaliar("icl_banking77_5900shot_balance", "label: 43", ["43"])[0]
    assert avaliar("icl_banking77_5900shot_balance", "label: 43", ["43"])[1]["exact_match_flexivel"]
    assert avaliar("ruler_qa2_421K", "It was yes.", ["yes"])[0]

def test_configuracoes_em_ordem_de_prioridade():
    assert [c.nome for c in CONFIGURACOES[:4]] == ["direta-mhqa-lc-4omini", "direta-mhqa-bm25-4omini", "direta-mcc-lc-4omini", "direta-mcc-bm25-4omini"]

def test_limites_do_haiku_cabem_na_janela():  # Review Focus 4
    for c in CONFIGURACOES:
        if c.modelo == "claude-haiku-5-5" and c.agente == "contexto_longo":
            assert (c.input_length_limit - c.buffer_length) * FATOR_TOKENS_HAIKU + c.generation_max_length <= JANELA_HAIKU

def test_retomada_nao_repete_chamadas(tmp_path): ...  # Review Focus 3: limite 3 -> 3 linhas; rodar de novo sem limite -> N linhas; chamadas ao falso == N; gasto == soma das linhas
def test_erro_de_api_e_refeito_na_retomada(tmp_path): ...  # Review Focus 2: falso devolve erro na pergunta 2 da 1ª rodada; a 2ª rodada refaz só ela; 1 linha por pergunta
def test_teto_de_um_provedor_nao_para_o_outro(tmp_path): ...  # livro da Anthropic no teto; config haiku -> "parada: teto"; config openai completa
def test_portao_haiku_h2(tmp_path): ...  # 90% de acerto em 2024-2025 -> (False, 0.9); 80% -> (True, 0.8)
def test_projecao_do_piloto():
    linhas = [{"custo_usd": 1.0}, {"custo_usd": 0.1}, {"custo_usd": 0.1}]
    assert projecao(linhas, n_contextos=2, n_perguntas=100) == approx(11.8)
def test_jsonl_do_rodador_preserva_acentos(tmp_path): ...  # Review Focus 5: consulta "Questão de Ética" volta igual do .jsonl
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_avaliacao.py tests/test_rodar.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `replica/avaliacao.py` e `replica/rodar.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Atualizar a seção Comandos do `CLAUDE.md`**

Comandos: `snapshot`, `reclassificar`, `oab`, `rodar` (com `--piloto`, `--usar-reserva`, `--haiku-h2`) e `analise`. Marcar quais gastam dinheiro e incluir o comando `uv run python -c "import nltk; nltk.download('punkt_tab')"`, preparo único.

- [ ] **Step 6: Commit**

```bash
git add replica/avaliacao.py replica/rodar.py tests/test_avaliacao.py tests/test_rodar.py CLAUDE.md
git commit -m "feat: rodador com prioridade, retomada, piloto e portão do Haiku"
```

---

### Task 10: Análise

**Files:**
- Create: `replica/analise.py`, `tests/test_analise.py`

**Interfaces:**
- Consumes: `arquivos.ler_jsonl` (Task 5); `config.NUMEROS_ARTIGO`, `SEMENTE` (Task 1); `rodar.CONFIGURACOES` (Task 9), para mapear o nome de cada configuração ao seu `agente`. O número do artigo de uma configuração da etapa direta é `NUMEROS_ARTIGO[(fonte, cfg.agente)]` em cada fonte e `NUMEROS_ARTIGO[("mcc", cfg.agente)]` na média das cinco MCC.
- Produces:
  - `carregar_rodada(pasta: Path) -> dict[str, list[dict]]`
  - `intervalo_acuracia(acertos: list[bool], n: int = 10_000, semente: int = SEMENTE) -> tuple[float, float, float]` devolve `(acurácia, li, ls)`, por bootstrap de percentis 2,5 e 97,5.
  - `bootstrap_pareado(a: list[bool], b: list[bool], estratos: list[str] | None = None, n: int = 10_000, semente: int = SEMENTE) -> tuple[float, float, float]` devolve `(dif = média(b) − média(a), li, ls)`. Reamostra índices pareados e, com `estratos`, reamostra dentro de cada estrato.
  - `veredito(li: float, ls: float, sentido: int) -> str`:
    - `"confirmada"` se `sentido > 0 and li > 0` ou `sentido < 0 and ls < 0`
    - `"contrariada"` no caso oposto
    - `"inconclusiva"` no resto
  - `PRIMARIAS = [("H1 direta", "direta-mhqa-lc-4omini", "direta-mhqa-bm25-4omini", +1), ("H2 direta", "direta-mcc-lc-4omini", "direta-mcc-bm25-4omini", -1), ("H1 OAB", "oab-h1-lc-4omini", "oab-h1-bm25-4omini", +1), ("H2 OAB", "oab-h2-lc-4omini", "oab-h2-bm25-4omini", -1)]`. O pareamento é por `pergunta_id`; na H2 direta, `estratos` = `fonte`.
  - `reproduz(numero_artigo_pct: float, li: float, ls: float) -> bool`
  - `gerar_relatorio(pasta: Path) -> str` grava `pasta/tabela.md` com quatro seções:
    - Primárias: as duas acurácias com intervalo, a diferença com intervalo e o veredito.
    - Reprodução: o número do artigo, o nosso com intervalo e se reproduz, no MH-QA, na média MCC e em cada MCC.
    - Secundárias: acurácia com intervalo.
    - Custos: tokens de entrada, % cacheado, tokens de saída, US$ e erros, por configuração.
    - Linhas com `erro` contam como erro e são contadas à parte.
  - `grafico_qualidade_tokens(pasta: Path) -> Path` grava `pasta/qualidade_tokens.png`. Cada ponto é uma configuração da OAB H1 (k = 2, 5, 10, 20 do BM25 e os dois contextos longos), com x = tokens de entrada médios por pergunta e y = acurácia com barra de intervalo.
  - `main(argv=None) -> int` com a CLI `python -m replica.analise PASTA`

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_bootstrap_pareado_contem_a_diferenca():
    a = [True] * 60 + [False] * 40
    b = [True] * 75 + [False] * 25
    dif, li, ls = bootstrap_pareado(a, b)
    assert dif == approx(0.15) and li < 0.15 < ls
    assert bootstrap_pareado(a, b) == (dif, li, ls)  # determinístico

def test_veredito():
    assert veredito(0.02, 0.18, +1) == "confirmada"
    assert veredito(-0.18, -0.02, +1) == "contrariada"
    assert veredito(-0.01, 0.11, +1) == "inconclusiva"
    assert veredito(-0.12, -0.03, -1) == "confirmada"

def test_bootstrap_estratificado_preserva_estratos(): ...  # diferença constante dentro de cada estrato -> intervalo de largura ~0
def test_reproduz(): assert reproduz(56.0, 0.50, 0.62) and not reproduz(43.0, 0.50, 0.62)
def test_relatorio_tem_as_quatro_primarias(tmp_path): ...  # rodada falsa com JSONL das 8 configurações primárias
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_analise.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `replica/analise.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: tudo passa

- [ ] **Step 5: Commit**

```bash
git add replica/analise.py tests/test_analise.py
git commit -m "feat: análise com bootstrap pareado, vereditos e teste de reprodução"
```

---

### Task 11: Dados reais ⚠ gasta ~US$ 0,10 na OpenAI (autorizado)

**Files:**
- Create: `dados/manifesto.json` (versionado), `docs/conferencia-materias.md`
- Modify: nenhum código; se a conferência reprovar, `replica/reclassificar.py`

- [ ] **Step 1: Snapshot**

Run: `PYTHONUTF8=1 uv run python -m replica.snapshot`
Expected: `questoes_oab: 3395` e `law_articles: 11954`, e `dados/manifesto.json` criado. Se as contagens divergirem, parar e avisar o Pedro.

- [ ] **Step 2: Preparar o nltk e reclassificar**

Run: `uv run python -c "import nltk; nltk.download('punkt_tab')"` e depois `PYTHONUTF8=1 uv run python -m replica.reclassificar`
Expected: o mapa gravado, `docs/conferencia-materias.md` com 30 linhas e o gasto impresso abaixo de US$ 0,20.

- [ ] **Step 3: Conferência do Pedro**

O Pedro marca a coluna "ok?". Até 3 erros em 30 aprovam. Mais que isso: ajustar só o prompt de classificação, rodar o Step 2 de novo e conferir outra amostra.

- [ ] **Step 4: Montar os conjuntos da OAB**

Run: `PYTHONUTF8=1 uv run python -m replica.oab`
Expected:
- `h1`: 300 questões em uns 14 contextos.
- `h2`: memória com algo como 2.900 questões e teste com até 229.
- `sem_memoria`: ~3.377 questões.
- Nenhum contexto da H1 acima de ~265 mil tokens `o200k_base` (o CPC).

- [ ] **Step 5: Commit**

```bash
git add dados/manifesto.json docs/conferencia-materias.md
git commit -m "chore: snapshot do Atlas, matérias reclassificadas e conferidas"
```

---

### Task 12: Piloto ⚠ gasta ~US$ 1,50 em cada provedor (autorizado)

- [ ] **Step 1: Confirmar o pré-registro**

As regras de decisão e os números do artigo estão no desenho commitado (Task 1) antes de qualquer rodada.

Run: `git log --oneline -- docs/superpowers/specs`
Expected: o commit da Task 1 aparece antes de qualquer arquivo em `resultados/`.

- [ ] **Step 2: Conferir o tier das contas**

O Pedro olha os limites de taxa no console da Anthropic e da OpenAI. Com um limite de tokens de entrada por minuto abaixo de ~950 mil, a chamada de escrita do contexto longo do Haiku na H2 falha sempre. Nesse caso, as configurações 20, 21 e 21' ficam fora e o documento registra.

- [ ] **Step 3: Rodar o piloto**

Run: `PYTHONUTF8=1 uv run python -m replica.rodar --piloto 10 --rodada resultados/piloto-2026MMDD`
Expected:
- Uma linha por configuração com `feitas: 10` (as sem memória rodam completas; o portão precisa delas).
- `projecao.md` gravado.

- [ ] **Step 4: Ler o piloto e decidir**

- Cache: nas configurações de contexto longo, as chamadas 2 a 10 têm `tokens_cacheados ≥ 0,8 × tokens_entrada`. Se não tiverem, parar: sem cache o orçamento não fecha.
- `letra_invalida` abaixo de 10% nas configurações da OAB. Se passar, só a regex de `extrair_letra` pode mudar, nunca os prompts.
- Projeção da OpenAI: primárias até US$ 16. Projeção da Anthropic: até US$ 24 → `--haiku-h2 1m`; acima → `600k`.
- O Pedro compara o `resultados/livro_caixa.json` com os painéis da OpenAI e da Anthropic; diferença de até 10% aceita. Se o Haiku custar mais que o calculado, corrigir `PRECOS` antes de seguir.

- [ ] **Step 5: Registrar as decisões e commitar**

Escrever "Decisões do piloto" no fim da §6 do desenho: cache, variante do Haiku, conferência dos custos.

```bash
git add resultados/piloto-* resultados/livro_caixa.json docs/superpowers/specs
git commit -m "chore: piloto, projeção de custo e decisões registradas"
```

---

### Task 13: Rodada completa da etapa direta ⚠ ~US$ 8,20 (autorizado)

- [ ] **Step 1: Rodar as configurações 1 a 5**

Run: `PYTHONUTF8=1 uv run python -m replica.rodar direta-mhqa-lc-4omini direta-mhqa-bm25-4omini direta-mcc-lc-4omini direta-mcc-bm25-4omini direta-mhqa-emb-4omini --rodada resultados/2026MMDD-direta`
Expected: 100 linhas no MH-QA e 500 nas MCC por configuração, `erros: 0`. Se houver erros, rodar o mesmo comando de novo; só as perguntas com erro são refeitas.

- [ ] **Step 2: Analisar**

Run: `PYTHONUTF8=1 uv run python -m replica.analise resultados/2026MMDD-direta`
Expected: `tabela.md` com H1 direta e H2 direta e a seção de Reprodução preenchida.

- [ ] **Step 3: Aplicar a regra de reprodução**

Se o número do artigo cair dentro do intervalo nas quatro configurações primárias, a etapa reproduz, e a reserva pode ser liberada para as secundárias da OAB.

Se alguma primária não reproduzir, abre a janela de depuração: até 3 dias úteis e até US$ 4 com `--usar-reserva`.
- Conferir as diferenças de prompt contra o repositório do autor.
- Se for viável, rodar o código do autor num subconjunto.

Se ainda assim não reproduzir, registrar: "não reproduzido; não foi possível separar o autor da reimplementação".

- [ ] **Step 4: Commit**

```bash
git add resultados/2026MMDD-direta resultados/livro_caixa.json
git commit -m "chore: rodada da etapa direta e teste de reprodução"
```

---

### Task 14: Rodada completa da OAB ⚠ ~US$ 6,50 na OpenAI e até ~US$ 21 na Anthropic (autorizado)

- [ ] **Step 1: Rodar as primárias e as linhas de base**

Run: `PYTHONUTF8=1 uv run python -m replica.rodar oab-sem-memoria-4omini oab-sem-memoria-haiku oab-h1-lc-4omini oab-h1-bm25-4omini oab-h2-lc-4omini oab-h2-bm25-4omini --rodada resultados/2026MMDD-oab`
Expected: 300 linhas na H1, até 229 na H2 e ~3.377 nas sem memória, com `erros: 0` depois das retomadas.

- [ ] **Step 2: Rodar as secundárias**

Run: `PYTHONUTF8=1 uv run python -m replica.rodar --rodada resultados/2026MMDD-oab --haiku-h2 <decisão do piloto>`. Acrescente `--usar-reserva` só se a Task 13 reproduziu ou fechou a janela de depuração.
Expected: as configurações rodam até o teto de cada provedor. O que parar por teto aparece como `parada: teto` e vai para as limitações do documento.

- [ ] **Step 3: Analisar**

Run: `PYTHONUTF8=1 uv run python -m replica.analise resultados/2026MMDD-oab`
Expected: `tabela.md` com H1 OAB e H2 OAB, e `qualidade_tokens.png`.

- [ ] **Step 4: Commit**

```bash
git add resultados/2026MMDD-oab resultados/livro_caixa.json
git commit -m "chore: rodada da OAB"
```

---

### Task 15: Documento da entrega

**Files:**
- Create: `docs/replicacao-memoryagentbench.md`
- Modify: `CLAUDE.md` (armadilhas aprendidas nas rodadas)

- [ ] **Step 1: Escrever o documento**

No estilo do documento da semana 8 do Experimento 1:
- o resultado em poucas linhas;
- o que foi replicado e o que não foi;
- as regras de decisão como foram registradas (com o hash do commit);
- o método: fidelidade e cada adaptação do `NOTICE.md`;
- as tabelas de `tabela.md` e o gráfico;
- uma leitura honesta, com vereditos, reprodução e as peculiaridades achadas: o esquecimento do agente de embedding, os 82,0 × 82,4 do artigo, o `exact_match` estrito nas MCC e os modelos que saíram do ar;
- as limitações, incluindo o que parou por teto;
- o que cada afirmação diz sobre o mapa da residência;
- como rodar.

- [ ] **Step 2: Atualizar o `CLAUDE.md`** com as armadilhas novas, uma linha cada

- [ ] **Step 3: Verificar**

Run: `uv run pytest`
Expected: tudo passa

- [ ] **Step 4: Commit**

```bash
git add docs/replicacao-memoryagentbench.md CLAUDE.md
git commit -m "docs: documento da replicação do MemoryAgentBench"
```
