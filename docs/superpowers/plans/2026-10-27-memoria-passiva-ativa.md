# Memória passiva e ativa: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dois exemplos comparáveis de memória para LLM (passiva por sumarização incremental e ativa com ferramentas estilo MemGPT), uma verificação em miniatura com duas baselines, e o documento da entrega da semana 8 com números reais.

**Architecture:** Pacote Python `memoria/` com seis arquivos pequenos. `llm.py` é o único ponto que fala com a API e soma tokens. `store.py` é o SQLite (um resumo, notas com FTS5). Cada exemplo é uma classe de três métodos (`__init__`, `nova_sessao`, `responder`) em um arquivo que se lê de cima para baixo. A verificação em `memoria/avaliacao/` roda as quatro configurações sobre o mesmo `dataset.json` e gera a tabela. Testes todos offline com um cliente fake injetado.

**Tech Stack:** Python 3.12+, `uv`, `anthropic` (SDK oficial), `python-dotenv`, `pytest`, SQLite com FTS5 (biblioteca padrão).

**Spec:** `docs/superpowers/specs/2026-10-27-memoria-passiva-ativa-design.md`

## Global Constraints

- Princípio §0 da spec: o mais simples que ainda ensina. Cada exemplo em um arquivo. Sem flags, sem opções não usadas, sem camadas extras. Docstrings em PT-BR dizendo o quê e o porquê.
- Modelo: `claude-sonnet-5-5`, em uma única constante. Sem `tool_choice` forçado, sem `thinking` explícito, sem `fallbacks`, sem prompt caching, sem `budget_tokens`.
- Dependências de runtime: só `anthropic` e `python-dotenv`. Dev: `pytest`.
- Tudo que o modelo lê e tudo que o professor lê é em PT-BR com acentos. Identificadores de código em PT-BR sem acento (`resumo`, `nova_sessao`, `buscar_memoria`).
- Chave só por `ANTHROPIC_API_KEY` (ambiente ou `.env`). Nunca no código, nunca no git.
- Histórico entre turnos guarda só texto (usuário, assistente). Blocos de `tool_use`, `tool_result` e thinking não atravessam turnos.
- Erro de API ou recusa em um turno vira a resposta literal `[erro: <motivo>]`, que entra no histórico como assistente. Resumidor não roda nesse turno.
- Testes nunca chamam a API. Nunca há assert de igualdade exata sobre saída real de LLM.
- Commits em PT-BR com prefixo `feat:`, `fix:`, `chore:`, `docs:`, `test:`. Sem linha `Co-Authored-By`. Autor `Pedro Coelho <coelho@discente.ufg.br>`.
- Preços para a coluna de custo: entrada US$ 2,00 por milhão de tokens, saída US$ 10,00 por milhão.

## Review Focus

Entradas que a spec implica mas nenhuma task cobriria por padrão. Cada linha tem seu teste na task dona.

1. **Consulta de busca com pontuação ou operadores FTS5** ("onde eu moro?", "AND", aspas): o `MATCH` do FTS5 quebra com sintaxe inválida. Esperado: a busca nunca lança; a consulta é reduzida a palavras entre aspas unidas por OR. Teste na Task 2.
2. **Modelo pede ferramentas até estourar o limite de iterações** sem nunca dar `end_turn`: esperado que `responder` devolva `[erro: limite de ferramentas]` e não fique em loop. Teste na Task 4.
3. **Resposta com `stop_reason == "refusal"`**: esperado `[erro: recusa]`, e na passiva o resumidor não roda. Teste nas Tasks 1 e 3.
4. **Primeiro turno da conversa com banco vazio** (sem resumo, sem notas): a passiva injeta o esqueleto vazio de S; a ativa devolve "Nenhuma nota encontrada." na busca. Testes nas Tasks 2 e 3.
5. **Sessão nova sem turnos antes de outra sessão nova** no contexto completo: dois marcadores seguidos não podem quebrar a alternância usuário/assistente. Teste na Task 5.

---

### Task 0: Esqueleto do repositório

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `.claude/settings.json`, `README.md`, `CLAUDE.md`, `memoria/__init__.py`, `memoria/avaliacao/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: fixture `fake` em `tests/conftest.py`, usada por todas as tasks de teste.

- [ ] **Step 1: Criar `pyproject.toml`** com `name = "memoria"`, `requires-python = ">=3.12"`, `dependencies = ["anthropic", "python-dotenv"]`, grupo dev `pytest`, e `[tool.pytest.ini_options] testpaths = ["tests"]`. Build backend `hatchling` com `packages = ["memoria"]`.

- [ ] **Step 2: Rodar `uv sync` e confirmar que `uv run python -c "import anthropic, dotenv; print(anthropic.__version__)"` imprime uma versão.**

- [ ] **Step 3: Criar `.gitignore`** com `.venv/`, `.env`, `__pycache__/`, `*.db`, `.pytest_cache/`. Criar `.env.example` com uma linha `ANTHROPIC_API_KEY=` e um comentário em PT-BR.

- [ ] **Step 4: Criar `.claude/settings.json`** habilitando o plugin:

```json
{ "enabledPlugins": { "superpowers@claude-plugins-official": true } }
```

- [ ] **Step 5: Criar `tests/conftest.py`** com o cliente fake e os construtores de resposta. Interface exata:

```python
from types import SimpleNamespace

def resposta_texto(texto: str, entrada: int = 10, saida: int = 5) -> SimpleNamespace
    # .stop_reason="end_turn", .content=[SimpleNamespace(type="text", text=texto)],
    # .usage=SimpleNamespace(input_tokens=entrada, output_tokens=saida), .stop_details=None

def resposta_ferramenta(nome: str, entrada: dict, id: str = "tu_1") -> SimpleNamespace
    # .stop_reason="tool_use", .content=[SimpleNamespace(type="tool_use", name=nome, input=entrada, id=id)]

def resposta_recusa() -> SimpleNamespace
    # .stop_reason="refusal", .content=[], .stop_details=SimpleNamespace(category="x")

class FakeCliente:
    """Imita anthropic.Anthropic(): devolve respostas roteirizadas na ordem e guarda cada chamada."""
    def __init__(self, respostas: list): self.roteiro = list(respostas); self.chamadas = []
    # self.messages.create(**kwargs) -> guarda kwargs em self.chamadas, devolve self.roteiro.pop(0)
    # se o roteiro acabar, devolve resposta_texto("(sem roteiro)")

@pytest.fixture
def fake():
    """Fábrica: fake([r1, r2]) -> FakeCliente."""
    return FakeCliente
```

`resposta_ferramenta` deve aceitar também uma lista de tuplas para gerar vários `tool_use` no mesmo `content` (necessário para o teste de chamadas paralelas na Task 4).

- [ ] **Step 6: Criar `README.md`** com título, uma frase do que é, e os cinco comandos da spec §4. Criar `CLAUDE.md` com: o que é o projeto (3 linhas), a árvore de §4 da spec, os comandos, a convenção "um exemplo = um arquivo", a armadilha "testes nunca chamam a API; a rodada real custa dinheiro", e a seção `# superpowers` copiada do `~/.claude/CLAUDE.md` global do autor.

- [ ] **Step 7: Rodar `uv run pytest`** e confirmar `no tests ran` sem erro de import.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: esqueleto do projeto com uv, pytest e cliente fake"
```

---

### Task 1: `config.py` e `llm.py`

**Files:**
- Create: `memoria/config.py`, `memoria/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `config.MODELO = "claude-sonnet-5-5"`, `config.K = 6`, `config.MAX_TOKENS = 4096`, `config.LIMITE_PALAVRAS_RESUMO = 400`, `config.TOP_N_NOTAS = 5`, `config.MAX_ITERACOES_FERRAMENTAS = 6`, `config.CAMINHO_BANCO = "memoria.db"`, `config.PRECO_ENTRADA_USD_POR_MILHAO = 2.0`, `config.PRECO_SAIDA_USD_POR_MILHAO = 10.0`, `config.SYSTEM_BASE: str`.
  - `class llm.Contador`: campos `chamadas`, `tokens_entrada`, `tokens_saida` (int, iniciam em 0); método `zerar() -> None`; propriedade `custo_usd -> float`.
  - `class llm.LLM(cliente=None)`: atributos `cliente` (padrão `anthropic.Anthropic()`) e `contador: Contador`; método `responder(system: str, mensagens: list[dict], ferramentas: list[dict] | None = None, formato: dict | None = None) -> Message | None`.
  - `llm.texto(msg) -> str`.

- [ ] **Step 1: Escrever os testes em `tests/test_llm.py`**

```python
def test_responder_passa_modelo_system_e_mensagens(fake):
    cliente = fake([resposta_texto("oi")])
    llm = LLM(cliente)
    llm.responder("sys", [{"role": "user", "content": "olá"}])
    chamada = cliente.chamadas[0]
    assert chamada["model"] == "claude-sonnet-5-5"
    assert chamada["system"] == "sys"
    assert chamada["messages"] == [{"role": "user", "content": "olá"}]
    assert "tools" not in chamada and "output_config" not in chamada

def test_responder_envia_ferramentas_e_formato_quando_dados(fake): ...
    # com ferramentas=[{...}] -> chamada["tools"] é a lista; com formato={...} -> chamada["output_config"] == {"format": formato}

def test_contador_soma_tokens_e_chamadas(fake):
    cliente = fake([resposta_texto("a", 100, 20), resposta_texto("b", 50, 5)])
    ... duas chamadas ...
    assert (c.chamadas, c.tokens_entrada, c.tokens_saida) == (2, 150, 25)
    assert c.custo_usd == pytest.approx(150 * 2.0 / 1e6 + 25 * 10.0 / 1e6)
    c.zerar(); assert c.chamadas == 0

def test_responder_devolve_none_em_erro_de_api(fake):
    # cliente cujo messages.create lança anthropic.APIConnectionError(request=...)
    assert llm.responder("s", [...]) is None

def test_texto_junta_blocos_de_texto(): assert texto(resposta_texto("abc")) == "abc"
def test_texto_de_none_e_erro(): assert texto(None) == "[erro: api]"
def test_texto_de_recusa_e_erro(): assert texto(resposta_recusa()) == "[erro: recusa]"
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_llm.py -v`** e confirmar que falha por import.

- [ ] **Step 3: Implementar `memoria/config.py`** com as constantes acima e `SYSTEM_BASE` exatamente:

```
Você é um assistente pessoal. Responda em português do Brasil, de forma direta e curta.
Quando não tiver a informação para responder, diga claramente que não sabe ou que isso não foi mencionado, em vez de inventar.
```

- [ ] **Step 4: Implementar `memoria/llm.py`.** `LLM.__init__` chama `dotenv.load_dotenv()` antes de criar `anthropic.Anthropic()` quando `cliente` é `None`. `responder` monta `kwargs` com `model`, `max_tokens`, `system`, `messages`, acrescenta `tools` e `output_config={"format": formato}` só quando dados, chama `self.cliente.messages.create(**kwargs)` dentro de `try/except anthropic.APIError` (devolve `None` no except), soma `usage.input_tokens` e `usage.output_tokens` no contador e devolve a mensagem. `texto(msg)` devolve `"[erro: api]"` para `None`, `"[erro: recusa]"` quando `stop_reason == "refusal"`, senão a junção dos blocos `type == "text"`.

- [ ] **Step 5: Rodar `uv run pytest tests/test_llm.py -v`**, esperado tudo PASS.

- [ ] **Step 6: Commit** `feat: wrapper do cliente Anthropic com contador de tokens`

---

### Task 2: `store.py`

**Files:**
- Create: `memoria/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces `class store.Store(caminho: str = ":memory:")` com:
  - `ler_resumo() -> str` (`""` quando não há)
  - `gravar_resumo(resumo: str) -> None` (substitui)
  - `gravar_nota(data: str, conteudo: str) -> int` (id da nota)
  - `buscar_notas(consulta: str, n: int = TOP_N_NOTAS) -> list[dict]` com chaves `id`, `data`, `conteudo`, ordenadas por relevância
  - `fechar() -> None`

- [ ] **Step 1: Escrever `tests/test_store.py`**

```python
def test_resumo_comeca_vazio(): assert Store().ler_resumo() == ""
def test_gravar_resumo_substitui(): s.gravar_resumo("a"); s.gravar_resumo("b"); assert s.ler_resumo() == "b"
def test_gravar_nota_devolve_ids_crescentes(): assert s.gravar_nota("2026-01-01", "x") < s.gravar_nota("2026-01-02", "y")
def test_buscar_nota_por_palavra():
    s.gravar_nota("2026-01-01", "O cachorro da usuária se chama Thor")
    s.gravar_nota("2026-01-02", "Ela trabalha na empresa Alfa")
    [n] = s.buscar_notas("cachorro")
    assert n["conteudo"].startswith("O cachorro") and n["data"] == "2026-01-01" and isinstance(n["id"], int)
def test_buscar_sem_resultado_devolve_lista_vazia(): assert Store().buscar_notas("nada") == []
def test_buscar_respeita_n(): ... 7 notas com "projeto" ... assert len(s.buscar_notas("projeto", n=3)) == 3
def test_buscar_com_pontuacao_e_operadores_nao_quebra():
    s.gravar_nota("2026-01-01", "mora em Goiânia")
    assert s.buscar_notas('onde eu moro? "AND" (Goiânia) OR NOT *')  # não lança; retorna lista
def test_buscar_consulta_so_com_pontuacao_devolve_vazio(): assert s.buscar_notas("???") == []
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_store.py -v`**, esperado falha por import.

- [ ] **Step 3: Implementar `memoria/store.py`.** Duas tabelas criadas no `__init__`: `resumo(id INTEGER PRIMARY KEY CHECK (id = 1), texto TEXT)` e `CREATE VIRTUAL TABLE notas USING fts5(conteudo, data UNINDEXED)`. `gravar_resumo` faz `INSERT OR REPLACE` com `id = 1`. `buscar_notas` passa a consulta por uma função privada `_consulta_fts(consulta: str) -> str` que extrai só tokens `\w+` (unicode), envolve cada um em aspas duplas e une com `OR`; se não sobrar token, `buscar_notas` devolve `[]` sem consultar. Query: `SELECT rowid, data, conteudo FROM notas WHERE notas MATCH ? ORDER BY rank LIMIT ?`.

- [ ] **Step 4: Rodar `uv run pytest tests/test_store.py -v`**, esperado PASS.

- [ ] **Step 5: Commit** `feat: store SQLite com resumo e notas em FTS5`

---

### Task 3: Exemplo 1, `passiva.py`

**Files:**
- Create: `memoria/passiva.py`
- Test: `tests/test_passiva.py`

**Interfaces:**
- Consumes: `LLM`, `texto`, `Store`, `config.K`, `config.SYSTEM_BASE`, `config.LIMITE_PALAVRAS_RESUMO`.
- Produces: `class passiva.Passiva(llm: LLM, store: Store)` com `nova_sessao(data: str) -> None` e `responder(mensagem: str) -> str`; constantes `passiva.RESUMO_VAZIO: str` e `passiva.PROMPT_RESUMIDOR: str`.

- [ ] **Step 1: Escrever `tests/test_passiva.py`**

```python
def test_primeiro_turno_injeta_resumo_vazio_e_faz_duas_chamadas(fake):
    cliente = fake([resposta_texto("Olá!"), resposta_texto("## Sobre o usuário\n- chama-se Ana")])
    p = Passiva(LLM(cliente), Store()); p.nova_sessao("2026-03-01")
    assert p.responder("Oi, sou a Ana") == "Olá!"
    principal, resumidor = cliente.chamadas
    assert RESUMO_VAZIO in principal["system"] and "Hoje é 2026-03-01" in principal["system"]
    assert principal["messages"] == [{"role": "user", "content": "Oi, sou a Ana"}]
    assert "Oi, sou a Ana" in resumidor["messages"][0]["content"] and "Olá!" in resumidor["messages"][0]["content"]

def test_resumo_novo_e_gravado_e_usado_no_turno_seguinte(fake):
    ... após o primeiro turno, store.ler_resumo() == "## Sobre o usuário\n- chama-se Ana"
    ... no segundo turno, esse texto está no system da chamada principal

def test_janela_envia_exatamente_as_ultimas_k_mensagens(fake):
    # K == 6: após 4 turnos, a 5ª chamada principal tem 6 mensagens antigas + 1 nova = 7
    # e a primeira delas é a mensagem do usuário do turno 2

def test_nova_sessao_zera_janela_mas_mantem_resumo(fake):
    # após 1 turno, nova_sessao("2026-03-08"); próxima chamada principal tem só 1 mensagem e o resumo gravado no system

def test_erro_na_resposta_principal_nao_roda_resumidor(fake):
    cliente = fake([resposta_recusa()])
    ... assert p.responder("x") == "[erro: recusa]"; assert len(cliente.chamadas) == 1; assert store.ler_resumo() == ""

def test_erro_no_resumidor_mantem_resumo_anterior(fake):
    # cliente cujo segundo create lança APIConnectionError -> resumo gravado continua o anterior
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_passiva.py -v`**, esperado falha por import.

- [ ] **Step 3: Implementar `memoria/passiva.py`.** Docstring de módulo explicando em PT-BR que este é o exemplo de memória passiva (política heurística: o código sempre lê S, sempre injeta, sempre resume; o modelo nunca sabe que existe memória) e desenhando o pipeline de §5.1 da spec em ASCII. `RESUMO_VAZIO` é o esqueleto Markdown com as quatro seções `## Sobre o usuário`, `## Fatos e decisões`, `## Pendências`, `## Linha do tempo`, cada uma com `(nada ainda)`. `PROMPT_RESUMIDOR` é o system do resumidor, em PT-BR, e deve: dizer que ele mantém um resumo com essas quatro seções; instruir a **editar** (manter o que continua válido, acrescentar o novo, substituir o que foi contradito, registrar na linha do tempo o que tem data no formato `AAAA-MM-DD: evento`); impor o limite de `{limite}` palavras comprimindo o menos importante; informar `Hoje é {data}`; e exigir que a saída seja **só** o resumo em Markdown, sem comentários. Em `responder`: `resumo = store.ler_resumo() or RESUMO_VAZIO`; system = `SYSTEM_BASE` + linha `Hoje é {data}.` + bloco `Resumo do que você sabe sobre o usuário e a conversa até aqui:` + resumo; mensagens = `self.janela[-K:] + [nova]`; após a resposta, acrescenta usuário e assistente à `self.janela`; se a resposta **não** começa com `[erro:`, chama o resumidor com uma única mensagem de usuário contendo `RESUMO ATUAL:` + resumo + `NOVO TURNO:` + `Usuário: ...` + `Assistente: ...` e grava `texto(...)` só se o resumidor não devolver erro. O bloco `__main__` abre `Store(CAMINHO_BANCO)`, pede a data de hoje com `datetime.date.today().isoformat()`, lê linhas até EOF, trata `/nova` zerando a janela com a mesma data, e imprime a resposta e depois o resumo atual entre linhas `--- resumo ---`.

- [ ] **Step 4: Rodar `uv run pytest tests/test_passiva.py -v`**, esperado PASS.

- [ ] **Step 5: Commit** `feat: exemplo 1, memória passiva por sumarização incremental`

---

### Task 4: Exemplo 2, `ativa.py`

**Files:**
- Create: `memoria/ativa.py`
- Test: `tests/test_ativa.py`

**Interfaces:**
- Consumes: `LLM`, `texto`, `Store`, `config.K`, `config.SYSTEM_BASE`, `config.MAX_ITERACOES_FERRAMENTAS`.
- Produces: `class ativa.Ativa(llm: LLM, store: Store)` com `nova_sessao(data: str) -> None`, `responder(mensagem: str) -> str`, atributo `chamadas_ferramenta: dict[str, int]` com chaves `"buscar_memoria"` e `"gravar_memoria"`; constantes `ativa.FERRAMENTAS: list[dict]` e `ativa.BLOCO_MEMORIA: str`.

- [ ] **Step 1: Escrever `tests/test_ativa.py`**

```python
def test_sem_ferramenta_e_um_turno_simples(fake):
    cliente = fake([resposta_texto("Olá!")])
    a = Ativa(LLM(cliente), Store()); a.nova_sessao("2026-03-01")
    assert a.responder("oi") == "Olá!"
    ch = cliente.chamadas[0]
    assert [t["name"] for t in ch["tools"]] == ["buscar_memoria", "gravar_memoria"]
    assert all(t["strict"] is True for t in ch["tools"])
    assert BLOCO_MEMORIA in ch["system"] and "Hoje é 2026-03-01" in ch["system"]

def test_gravar_memoria_executa_e_loop_termina_no_end_turn(fake):
    cliente = fake([resposta_ferramenta("gravar_memoria", {"conteudo": "Cachorro se chama Thor"}, "tu_1"), resposta_texto("Anotado!")])
    ... assert a.responder("meu cachorro é o Thor") == "Anotado!"
    segunda = cliente.chamadas[1]["messages"]
    assert segunda[-2]["role"] == "assistant"
    assert segunda[-1] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "Nota 1 gravada."}]}
    assert store.buscar_notas("Thor")[0]["data"] == "2026-03-01"
    assert a.chamadas_ferramenta == {"buscar_memoria": 0, "gravar_memoria": 1}

def test_buscar_memoria_devolve_notas_formatadas(fake):
    store.gravar_nota("2026-02-01", "Cachorro se chama Thor")
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "cachorro"}), resposta_texto("Thor")])
    ... conteudo do tool_result == "[1] (2026-02-01) Cachorro se chama Thor"

def test_buscar_sem_notas_devolve_mensagem_padrao(fake): ... conteudo == "Nenhuma nota encontrada."

def test_chamadas_paralelas_voltam_em_uma_unica_mensagem(fake):
    # resposta_ferramenta com dois tool_use (gravar + buscar) -> a mensagem de user seguinte tem 2 tool_result, ids na mesma ordem

def test_janela_guarda_so_texto_entre_turnos(fake):
    # turno 1 com ferramenta; turno 2: a chamada tem exatamente 3 mensagens (user, assistant texto, user nova), nenhum tool_use

def test_limite_de_iteracoes(fake):
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "x"})] * 10)
    assert a.responder("?") == "[erro: limite de ferramentas]"
    assert len(cliente.chamadas) == 6

def test_erro_de_api_no_meio_do_loop(fake):
    # primeira resposta tool_use, segunda create lança APIConnectionError -> responder devolve "[erro: api]"
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_ativa.py -v`**, esperado falha por import.

- [ ] **Step 3: Implementar `memoria/ativa.py`.** Docstring de módulo: exemplo de memória ativa (política de autocontrole por prompt, como as funções de memória do MemGPT: o modelo decide quando ler e escrever; o código só executa o que ele pede), com o loop de §5.2 da spec em ASCII. `FERRAMENTAS` são duas definições com `strict: True`, `input_schema` com `additionalProperties: False` e `required`; descrições em PT-BR dizendo quando usar cada uma (`buscar_memoria`: antes de responder qualquer coisa que dependa do que o usuário disse em sessões anteriores; `gravar_memoria`: assim que o usuário disser algo durável sobre si, decisões, datas ou mudanças, uma nota curta e autocontida por fato). `BLOCO_MEMORIA` em PT-BR diz: a janela de mensagens é curta e sessões anteriores não ficam visíveis; o que não for gravado será perdido; grave fatos duráveis; busque antes de responder sobre o passado; notas têm data de gravação e, se duas se contradizem, vale a mais recente. `responder` segue o loop de §5.2: `for _ in range(MAX_ITERACOES_FERRAMENTAS)`, sai quando `msg is None` ou `stop_reason != "tool_use"`, senão anexa `{"role": "assistant", "content": msg.content}` e uma única mensagem de usuário com todos os `tool_result`. Ao sair, `resposta = texto(msg)`; se a saída foi por esgotar o `for` (último `stop_reason == "tool_use"`), `resposta = "[erro: limite de ferramentas]"`. `_executar(bloco) -> dict` incrementa `chamadas_ferramenta[bloco.name]`, e formata a busca como linhas `[{id}] ({data}) {conteudo}` ou `Nenhuma nota encontrada.`, e a gravação como `Nota {id} gravada.`. O `__main__` é igual ao da passiva, mas imprime cada chamada de ferramenta como `-> buscar_memoria(consulta='...')` / `-> gravar_memoria(conteudo='...')` no momento da execução (um `print` dentro de `_executar`, controlado por um atributo `verboso: bool = False` que o `__main__` liga).

- [ ] **Step 4: Rodar `uv run pytest tests/test_ativa.py -v`**, esperado PASS.

- [ ] **Step 5: Commit** `feat: exemplo 2, memória ativa com ferramentas estilo MemGPT`

---

### Task 5: Baselines

**Files:**
- Create: `memoria/avaliacao/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `LLM`, `texto`, `config.K`, `config.SYSTEM_BASE`.
- Produces: `class SemMemoria(llm, store)` e `class ContextoCompleto(llm, store)`, ambas com `nova_sessao(data)` e `responder(mensagem) -> str`. O parâmetro `store` existe só para a interface ser igual às dos exemplos e não é usado (docstring diz isso).

- [ ] **Step 1: Escrever `tests/test_baselines.py`**

```python
def test_sem_memoria_nunca_ve_sessao_anterior(fake):
    # 2 turnos na sessão 1, nova_sessao, 1 turno: a 3ª chamada tem exatamente 1 mensagem
def test_sem_memoria_respeita_k(fake): # como na passiva: após 4 turnos a 5ª chamada tem 7 mensagens
def test_contexto_completo_ve_tudo_com_marcadores_de_sessao(fake):
    # sessão "2026-03-01" com 1 turno, sessão "2026-03-08" com 1 turno: a 2ª chamada tem
    # [user "(Nova sessão. Hoje é 2026-03-01.)", assistant "Ok.", user m1, assistant r1,
    #  user "(Nova sessão. Hoje é 2026-03-08.)", assistant "Ok.", user m2]
def test_contexto_completo_duas_sessoes_vazias_seguidas_mantem_alternancia(fake):
    # nova_sessao, nova_sessao, responder: os papéis da lista alternam user/assistant sem repetição
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_baselines.py -v`**, esperado falha por import.

- [ ] **Step 3: Implementar `memoria/avaliacao/baselines.py`.** `SemMemoria` é a `Passiva` sem resumo: system = `SYSTEM_BASE` + `Hoje é {data}.`, mensagens = `janela[-K:] + [nova]`. `ContextoCompleto` mantém `self.historico` com tudo; `nova_sessao` anexa o par marcador `(Nova sessão. Hoje é {data}.)` / `Ok.`; `responder` envia `historico + [nova]` e depois anexa usuário e assistente ao histórico. Sem K.

- [ ] **Step 4: Rodar `uv run pytest tests/test_baselines.py -v`**, esperado PASS.

- [ ] **Step 5: Commit** `feat: baselines sem memória e contexto completo`

---

### Task 6: `dataset.json`

**Files:**
- Create: `memoria/avaliacao/dataset.json`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces o arquivo com o formato:

```json
{
  "usuario": "nome",
  "sessoes": [ { "data": "AAAA-MM-DD", "falas": ["...", "..."] } ],
  "perguntas": { "data": "AAAA-MM-DD", "itens": [
      { "id": "q01", "categoria": "sessao_unica", "pergunta": "...", "esperado": "..." } ] }
}
```

Categorias válidas, exatamente: `sessao_unica`, `multi_sessao`, `temporal`, `atualizacao`, `abstencao`.

- [ ] **Step 1: Escrever `tests/test_dataset.py`**

```python
CATEGORIAS = {"sessao_unica", "multi_sessao", "temporal", "atualizacao", "abstencao"}
def test_estrutura_basica(): ... 4 <= len(sessoes) <= 6; cada sessão tem 8 a 12 falas, todas strings não vazias; datas no formato ISO e estritamente crescentes; data das perguntas > última sessão
def test_perguntas(): ... 15 <= len(itens) <= 20; ids únicos; categoria em CATEGORIAS; cada categoria com 3 a 4 itens; pergunta e esperado não vazios
def test_abstencao_espera_nao_sei(): ... todo item de abstencao tem esperado começando com "Não foi dito"
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_dataset.py -v`**, esperado falha (arquivo não existe).

- [ ] **Step 3: Escrever `memoria/avaliacao/dataset.json`** seguindo §6.1 da spec: um usuário fictício; 5 sessões com datas espaçadas por 1 a 3 semanas; 8 a 12 falas por sessão, misturando fatos duráveis (nome de pet, cidade, trabalho, projeto, livro, uma viagem com data) com conversa descartável na proporção aproximada de 1 para 3; cada fato relevante aparece **uma única vez**; pelo menos dois fatos são atualizados em sessão posterior (um deles o emprego); as perguntas da categoria temporal devem ser respondíveis a partir das datas das sessões (por exemplo "quantas semanas faz que eu disse que entreguei o relatório?"); as de multi_sessao exigem juntar fatos de duas sessões; as de abstencao perguntam coisas nunca ditas, com `esperado` começando com `Não foi dito`. Cada `esperado` é curto e objetivo para o juiz. Texto em PT-BR natural.

- [ ] **Step 4: Rodar `uv run pytest tests/test_dataset.py -v`**, esperado PASS.

- [ ] **Step 5: Commit** `feat: dataset sintético de verificação com 5 categorias`

---

### Task 7: `juiz.py`

**Files:**
- Create: `memoria/avaliacao/juiz.py`
- Test: `tests/test_juiz.py`

**Interfaces:**
- Consumes: `LLM.responder(..., formato=...)`, `texto`.
- Produces: `juiz.julgar(llm: LLM, pergunta: str, esperado: str, resposta: str, categoria: str) -> dict` com chaves `correto: bool` e `justificativa: str`; constante `juiz.PROMPT_JUIZ: str`; constante `juiz.FORMATO: dict` (json_schema com `correto` boolean e `justificativa` string, `required` ambos, `additionalProperties: False`).

- [ ] **Step 1: Escrever `tests/test_juiz.py`**

```python
def test_julgar_envia_formato_e_le_json(fake):
    cliente = fake([resposta_texto('{"correto": true, "justificativa": "bate"}')])
    r = julgar(LLM(cliente), "Como se chama meu cachorro?", "Thor", "Seu cachorro é o Thor.", "sessao_unica")
    assert r == {"correto": True, "justificativa": "bate"}
    ch = cliente.chamadas[0]
    assert ch["output_config"] == {"format": FORMATO}
    assert "Thor" in ch["messages"][0]["content"] and "Como se chama" in ch["messages"][0]["content"]
def test_julgar_abstencao_menciona_regra_no_prompt(fake):
    # com categoria "abstencao", o system contém a palavra "abstenção" ou a instrução de aceitar "não sei"
def test_julgar_erro_de_api_conta_como_incorreto(fake):
    # create lança -> {"correto": False, "justificativa": "[erro: api]"}
def test_julgar_json_invalido_conta_como_incorreto(fake):
    # resposta_texto("isso não é json") -> correto False, justificativa começa com "[erro:"
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_juiz.py -v`**, esperado falha por import.

- [ ] **Step 3: Implementar `memoria/avaliacao/juiz.py`.** `PROMPT_JUIZ` em PT-BR: você compara a resposta de um assistente com a resposta esperada; `correto` é verdadeiro se a resposta contém a informação esperada, mesmo com palavras diferentes; para a categoria `abstencao`, é correto se o assistente disse que não sabe ou que não foi mencionado, e incorreto se inventou; justificativa de uma linha. `julgar` monta uma mensagem de usuário com `Categoria:`, `Pergunta:`, `Resposta esperada:`, `Resposta do assistente:`, chama `llm.responder(PROMPT_JUIZ, [msg], formato=FORMATO)` e faz `json.loads(texto(msg))` dentro de `try/except (json.JSONDecodeError, TypeError)`; qualquer erro vira `{"correto": False, "justificativa": "[erro: ...]"}`.

- [ ] **Step 4: Rodar `uv run pytest tests/test_juiz.py -v`**, esperado PASS.

- [ ] **Step 5: Commit** `feat: juiz LLM com saída estruturada`

---

### Task 8: `rodar.py`

**Files:**
- Create: `memoria/avaliacao/rodar.py`
- Test: `tests/test_rodar.py`

**Interfaces:**
- Consumes: as quatro classes de configuração (mesma interface `(llm, store)`, `nova_sessao`, `responder`), `Store`, `LLM`, `julgar`, `Contador`.
- Produces:
  - `rodar.CONFIGURACOES: dict[str, type]` com chaves, nesta ordem: `sem_memoria`, `contexto_completo`, `passiva`, `ativa`.
  - `rodar.rodar_configuracao(nome: str, dataset: dict, llm: LLM, llm_juiz: LLM) -> dict`
  - `rodar.tabela(resultados: list[dict]) -> str` (Markdown)
  - `rodar.main(argv: list[str]) -> None`

Formato do dicionário de resultado:

```python
{
  "configuracao": "passiva",
  "turnos": [{"sessao": "2026-03-01", "usuario": "...", "assistente": "..."}],
  "resumos_por_sessao": [{"sessao": "...", "palavras": 123, "resumo": "..."}],   # só passiva, senão []
  "ferramentas": {"buscar_memoria": 0, "gravar_memoria": 0},                      # só ativa, senão {}
  "respostas": [{"id", "categoria", "pergunta", "esperado", "resposta", "correto", "justificativa"}],
  "acertos_por_categoria": {"sessao_unica": [3, 4], ...},
  "acertos_total": [12, 18],
  "erros": 0,                 # respostas (conversa + perguntas) que começam com "[erro:"
  "chamadas": 0, "tokens_entrada": 0, "tokens_saida": 0, "custo_usd": 0.0   # sem o juiz
}
```

- [ ] **Step 1: Escrever `tests/test_rodar.py`** com um dataset mínimo inline (2 sessões de 1 fala, 2 perguntas de categorias diferentes):

```python
def test_rodar_configuracao_sem_memoria_agrega(fake):
    llm = LLM(fake([resposta_texto("r")] * 10)); juiz = LLM(fake([resposta_texto('{"correto": true, "justificativa": "x"}'), resposta_texto('{"correto": false, "justificativa": "y"}')]))
    r = rodar_configuracao("sem_memoria", DATASET, llm, juiz)
    assert r["acertos_total"] == [1, 2] and r["acertos_por_categoria"] == {"sessao_unica": [1, 1], "abstencao": [0, 1]}
    assert len(r["turnos"]) == 2 and r["chamadas"] == 4 and r["erros"] == 0
    assert r["resumos_por_sessao"] == [] and r["ferramentas"] == {}
def test_rodar_passiva_registra_resumo_por_sessao(fake): # len(resumos_por_sessao) == 2, "palavras" > 0
def test_rodar_ativa_registra_ferramentas(fake): # chaves buscar_memoria e gravar_memoria presentes
def test_rodar_conta_erros(fake): # uma resposta_recusa em uma pergunta -> erros == 1; o juiz ainda é chamado para ela (recebe "[erro: recusa]" como resposta) e a conta de perguntas N não muda
def test_tabela_tem_uma_linha_por_configuracao_e_colunas_fixas():
    md = tabela([r1, r2]); assert md.count("\n| ") >= 2 ... cabeçalho contém "sessao_unica", "total", "tokens entrada", "custo (USD)"
def test_main_grava_arquivos(tmp_path, monkeypatch, fake):
    # monkeypatch LLM para usar fakes, monkeypatch pasta de saída para tmp_path; main(["sem_memoria"]) cria sem_memoria.json e tabela.md
```

- [ ] **Step 2: Rodar `uv run pytest tests/test_rodar.py -v`**, esperado falha por import.

- [ ] **Step 3: Implementar `memoria/avaliacao/rodar.py`.** `rodar_configuracao`: zera `llm.contador`; cria `Store()` em memória; instancia a classe; para cada sessão, `nova_sessao(data)` e `responder` por fala, guardando turnos; ao fim de cada sessão, se a instância é `Passiva`, registra `store.ler_resumo()` e `len(resumo.split())`; depois `nova_sessao(perguntas.data)` e, para cada item, `responder` e `julgar(llm_juiz, ...)`; agrega. `tabela` produz colunas, nesta ordem: configuração, uma coluna por categoria na ordem da constante de categorias, total, erros, chamadas, tokens entrada, tokens saída, custo (USD), ferramentas (`buscar/gravar` ou `-`). Células de acerto no formato `3/4`. `main(argv)`: carrega `dataset.json` ao lado do arquivo; `nomes = argv[:1] or list(CONFIGURACOES)`; cria `resultados/<AAAAMMDD-HHMMSS>/`; para cada nome grava `<nome>.json` (`ensure_ascii=False`, `indent=2`); grava `tabela.md`; imprime a tabela e `Custo total estimado: US$ x.xx`. Pasta de saída definida por uma constante de módulo `PASTA_RESULTADOS = Path("resultados")` para o teste trocar via monkeypatch. `if __name__ == "__main__": main(sys.argv[1:])`.

- [ ] **Step 4: Rodar `uv run pytest -v`** (suíte inteira), esperado tudo PASS.

- [ ] **Step 5: Commit** `feat: runner da verificação com tabela de resultados`

---

### Task 9: Rodada real da verificação

**Files:**
- Create: `resultados/<data-hora>/*.json`, `resultados/<data-hora>/tabela.md`

**Interfaces:**
- Consumes: `python -m memoria.avaliacao.rodar`.

- [ ] **Step 1: Confirmar a chave.** Rodar `uv run python -c "import os, dotenv; dotenv.load_dotenv(); print('ok' if os.getenv('ANTHROPIC_API_KEY') else 'FALTA')"`. Se imprimir `FALTA`, **parar aqui** e pedir ao autor para criar `.env` com `ANTHROPIC_API_KEY=...`. Não prosseguir sem a chave.

- [ ] **Step 2: Fumaça com uma configuração barata.** Rodar `uv run python -m memoria.avaliacao.rodar sem_memoria`. Confirmar que terminou sem exceção, que a pasta foi criada, que a tabela imprime acertos plausíveis (abstenção alta, resto baixo) e que o custo ficou abaixo de US$ 0,50. Apagar essa pasta de resultados depois (é fumaça, não rodada final).

- [ ] **Step 3: Rodada completa.** Rodar `uv run python -m memoria.avaliacao.rodar`. Esperado: quatro JSONs e `tabela.md`, custo total entre US$ 2 e 6. Se alguma configuração tiver `erros > 2`, investigar o JSON antes de aceitar (motivo provável: rate limit; a correção é reexecutar só aquela configuração pelo argumento, copiando o JSON novo para a mesma pasta).

- [ ] **Step 4: Verificar a saída.** Abrir `tabela.md` e conferir: as quatro linhas existem; nenhuma célula vazia; `passiva.json` tem cinco entradas em `resumos_por_sessao` com `palavras` abaixo de 400; `ativa.json` tem `ferramentas` com valores acima de zero; o total de tokens é coerente com a hipótese (contexto completo é o maior).

- [ ] **Step 5: Commit** `feat: rodada final da verificação (resultados reais)`

---

### Task 10: Documento da entrega

**Files:**
- Create: `docs/semana-8/memoria-passiva-ativa.md`
- Modify: `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `resultados/<rodada final>/*`, `RESIDENCIA/Semana 7/Semana 7.docx` (vocabulário), spec §7.

- [ ] **Step 1: Ler as fontes.** Ler `tabela.md` e os quatro JSONs da rodada final, `memoria/passiva.py`, `memoria/ativa.py`, e extrair o texto de `RESIDENCIA/Semana 7/Semana 7.docx` (a tabela "Política de controle" e a tabela "Quando usar cada abordagem" são as âncoras).

- [ ] **Step 2: Escrever `docs/semana-8/memoria-passiva-ativa.md`** com as sete seções de §7 da spec, em PT-BR, nesta ordem e com este conteúdo mínimo:
  1. **Diagnóstico.** Aplicar os três passos da semana 7 ao cenário do dataset: o histórico cabe na janela (sim, são poucos milhares de tokens), então por que memória? Porque custa a cada turno, porque a informação muda, porque as sessões são separadas. Citar a tabela "quando usar" e dizer em qual linha cada exemplo cai.
  2. **Exemplo 1, memória passiva.** Diagrama ASCII do pipeline, o `PROMPT_RESUMIDOR` literal, e o S real de `passiva.json` após a sessão 1 e após a sessão 5. Nomear: política heurística; memória semântica (fatos) com linha do tempo (episódica comprimida); ciclo escrever → gerenciar → ler de Du. Dizer explicitamente que **não é RAG**: é context compression (LongMemEval).
  3. **Exemplo 2, memória ativa.** Diagrama ASCII do loop, as duas definições de ferramenta literais, o `BLOCO_MEMORIA` literal, e um trecho real de `ativa.json` com uma gravação e uma busca. Nomear: autocontrole por prompt; notas episódicas com data; MemGPT.
  4. **Verificação.** Descrever o dataset (sessões, categorias, exemplos de pergunta), o juiz, as baselines. Colar `tabela.md`. Tabela de `palavras` de S por sessão. Uma linha dizendo que os números vêm de uma rodada e que saída de LLM não é determinística.
  5. **Conclusões.** O que a tabela mostra sobre passiva versus ativa, categoria por categoria, confrontando com a hipótese de §6.4 da spec e com o que os artigos dizem (MemoryAgentBench sobre agentes que pulam a memória, LongMemEval sobre perda de detalhe em resumo, ACE sobre colapso). Honestidade: se a hipótese falhou, dizer.
  6. **Limitações e extensões.** Embeddings, ferramenta de apagar, resumo por limiar, repetições, FTS5 sem stemming.
  7. **Como rodar.** Os cinco comandos e o aviso de custo.
  Referências no formato das semanas anteriores.

- [ ] **Step 3: Atualizar `README.md`** com um link para o documento e `CLAUDE.md` com a linha de armadilha que a rodada real ensinou (por exemplo: limite de taxa, formatação do FTS5, o que for verdade).

- [ ] **Step 4: Verificação de completude.** Conferir que todo número citado no documento existe em `tabela.md` ou nos JSONs, que os blocos "literais" são cópias exatas do código, e que `uv run pytest` continua verde.

- [ ] **Step 5: Commit** `docs: documento da entrega da semana 8`
