# Memória passiva e memória ativa com LLM

Dois exemplos comparáveis de memória para um LLM (passiva, por sumarização incremental, e ativa, com ferramentas estilo MemGPT) e uma verificação em miniatura que os compara, para a Residência em IA (UFG).

O documento da entrega da semana 8, com o diagnóstico, o desenho dos dois exemplos, a verificação e as conclusões, está em [`docs/semana-8/memoria-passiva-ativa.md`](docs/semana-8/memoria-passiva-ativa.md).

```
uv sync                                   # ambiente
uv run pytest                             # gate de testes (sem API)
uv run python -m memoria.passiva          # chat de terminal com o exemplo passivo
uv run python -m memoria.ativa            # chat de terminal com o exemplo ativo
uv run python -m memoria.avaliacao.rodar  # verificação completa (usa API, custa dinheiro)
```

A chave da API vem de `ANTHROPIC_API_KEY`, por variável de ambiente ou por um arquivo `.env` (copie o `.env.example`).
