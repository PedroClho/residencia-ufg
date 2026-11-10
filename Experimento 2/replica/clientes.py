"""Clientes de LLM e de embeddings. Toda chamada passa pelo livro-caixa: autoriza antes, registra depois."""
import sys
import time
from dataclasses import dataclass

import numpy as np
import tiktoken

from replica import config
from replica.livro_caixa import (
    LimiteDeGasto,
    LivroCaixa,
    custo_anthropic,
    custo_embedding,
    custo_openai,
)

ESPERA_TAXA_S = 60
TENTATIVAS_TAXA = 5
LIMITE_TOKENS_EMBEDDING = 8_191
TOKENS_POR_REQUISICAO_EMBEDDING = 250_000  # a API recusa mais de 300 mil

_codificador_chat = None
_codificador_embedding = None


def _contar_chat(texto: str) -> int:
    """Estimativa de tokens com o o200k_base (o tokenizador do GPT-4o-mini)."""
    global _codificador_chat
    if _codificador_chat is None:
        _codificador_chat = tiktoken.get_encoding("o200k_base")
    return len(_codificador_chat.encode(texto, disallowed_special=()))


def _codificador_emb():
    global _codificador_embedding
    if _codificador_embedding is None:
        _codificador_embedding = tiktoken.get_encoding("cl100k_base")
    return _codificador_embedding


def _carregar_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(config.RAIZ / ".env")


@dataclass
class Resposta:
    texto: str
    tokens_entrada: int  # total processado (inclui cacheados e escritos no cache)
    tokens_cacheados: int  # leituras de cache
    tokens_escritos_cache: int
    tokens_saida: int
    custo_usd: float
    erro: str | None = None


def _resposta_de_erro(erro: str) -> Resposta:
    return Resposta("", 0, 0, 0, 0, 0.0, erro)


class _ClienteBase:
    provedor: str

    def __init__(self, modelo, livro: LivroCaixa, bruto, dormir, erros_de_taxa):
        self.modelo = modelo
        self.livro = livro
        self._bruto = bruto
        self._dormir = dormir
        self._erros_de_taxa = erros_de_taxa

    def _criar_bruto(self):
        raise NotImplementedError

    def _taxa_padrao(self) -> tuple[type[Exception], ...]:
        raise NotImplementedError

    @property
    def bruto(self):
        if self._bruto is None:
            self._bruto = self._criar_bruto()
        return self._bruto

    @property
    def erros_de_taxa(self):
        return self._taxa_padrao() if self._erros_de_taxa is None else self._erros_de_taxa

    def _chamar_com_espera(self, chamada):
        """Chama a API; em limite de taxa espera 60 s e tenta de novo, até 5 vezes."""
        tentativas = 0
        while True:
            try:
                return chamada()
            except self.erros_de_taxa:
                if tentativas >= TENTATIVAS_TAXA:
                    raise
                tentativas += 1
                self._dormir(ESPERA_TAXA_S)

    def conversar(self, system, mensagem, max_tokens, prefixo_cacheavel=None) -> Resposta:
        prefixo = prefixo_cacheavel or ""
        self.livro.autorizar(self.provedor, self._custo_maximo(system or "", prefixo, mensagem, max_tokens))
        try:
            return self._executar(system, prefixo, mensagem, max_tokens, bool(prefixo_cacheavel))
        except LimiteDeGasto:
            raise
        except Exception as e:
            print(f"[erro: api] {e!r}", file=sys.stderr)
            return _resposta_de_erro("[erro: api]")


class ClienteOpenAI(_ClienteBase):
    provedor = "openai"

    def __init__(self, modelo, livro, temperatura=0.7, bruto=None, dormir=time.sleep, erros_de_taxa=None):
        super().__init__(modelo, livro, bruto, dormir, erros_de_taxa)
        self.temperatura = temperatura

    def _criar_bruto(self):
        import openai

        _carregar_env()
        return openai.OpenAI(max_retries=8)

    def _taxa_padrao(self):
        import openai

        return (openai.RateLimitError,)

    def _custo_maximo(self, system, prefixo, mensagem, max_tokens):
        estimados = _contar_chat(system + prefixo + mensagem)
        return custo_openai(self.modelo, estimados, 0, max_tokens)

    def _executar(self, system, prefixo, mensagem, max_tokens, _com_prefixo):
        mensagens = [{"role": "user", "content": prefixo + mensagem}]
        if system is not None:
            mensagens.insert(0, {"role": "system", "content": system})
        kwargs = dict(model=self.modelo, messages=mensagens, max_tokens=max_tokens)
        if self.temperatura is not None:
            kwargs["temperature"] = self.temperatura
        r = self._chamar_com_espera(lambda: self.bruto.chat.completions.create(**kwargs))
        u = r.usage
        detalhes = getattr(u, "prompt_tokens_details", None)
        cacheados = (getattr(detalhes, "cached_tokens", 0) or 0) if detalhes else 0
        custo = custo_openai(self.modelo, u.prompt_tokens, cacheados, u.completion_tokens)
        self.livro.registrar(self.provedor, custo)
        texto = r.choices[0].message.content or ""
        erro = None if texto else "[erro: resposta vazia]"
        return Resposta(texto, u.prompt_tokens, cacheados, 0, u.completion_tokens, custo, erro)


class ClienteAnthropic(_ClienteBase):
    provedor = "anthropic"

    def __init__(self, modelo, livro, bruto=None, dormir=time.sleep, erros_de_taxa=None):
        super().__init__(modelo, livro, bruto, dormir, erros_de_taxa)

    def _criar_bruto(self):
        import anthropic

        _carregar_env()
        return anthropic.Anthropic(max_retries=8)

    def _taxa_padrao(self):
        import anthropic

        return (anthropic.RateLimitError,)

    def _custo_maximo(self, system, prefixo, mensagem, max_tokens):
        fator = config.FATOR_TOKENS_HAIKU
        est_prefixo = int(_contar_chat(prefixo) * fator)
        est_resto = int(_contar_chat(system + mensagem) * fator)
        return custo_anthropic(self.modelo, est_resto, est_prefixo, 0, max_tokens)

    def _executar(self, system, prefixo, mensagem, max_tokens, com_prefixo):
        if com_prefixo:
            conteudo = [
                {"type": "text", "text": prefixo, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": mensagem},
            ]
        else:
            conteudo = mensagem
        kwargs = dict(
            model=self.modelo,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": conteudo}],
            thinking={"type": "disabled"},
        )
        if system is not None:
            kwargs["system"] = system
        r = self._chamar_com_espera(lambda: self.bruto.messages.create(**kwargs))
        u = r.usage
        sem_cache = getattr(u, "input_tokens", 0) or 0
        escrita = getattr(u, "cache_creation_input_tokens", 0) or 0
        leitura = getattr(u, "cache_read_input_tokens", 0) or 0
        saida = getattr(u, "output_tokens", 0) or 0
        custo = custo_anthropic(self.modelo, sem_cache, escrita, leitura, saida)
        self.livro.registrar(self.provedor, custo)
        texto = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        if r.stop_reason == "refusal":
            erro, texto = "[erro: recusa]", ""
        elif not texto:
            erro = "[erro: resposta vazia]"
        else:
            erro = None
        return Resposta(texto, sem_cache + escrita + leitura, leitura, escrita, saida, custo, erro)


class ClienteEmbeddings:
    """Embeddings em lotes, fechados por quantidade e por tokens estimados."""

    def __init__(self, livro: LivroCaixa, modelo: str = config.MODELO_EMBEDDING, bruto=None, lote: int = 256):
        self.livro = livro
        self.modelo = modelo
        self._bruto = bruto
        self.lote = lote

    @property
    def bruto(self):
        if self._bruto is None:
            import openai

            _carregar_env()
            self._bruto = openai.OpenAI(max_retries=8)
        return self._bruto

    def _preparar(self, texto: str) -> tuple[str, int]:
        """Trunca em 8.191 tokens (limite do modelo) e devolve (texto, tokens).

        Texto vazio ou só com espaços vira um espaço (a API recusa entrada vazia).
        """
        if not texto.strip():
            print("[aviso] texto vazio trocado por um espaço no embedding", file=sys.stderr)
            return " ", 1
        enc = _codificador_emb()
        tokens = enc.encode(texto, disallowed_special=())
        if len(tokens) > LIMITE_TOKENS_EMBEDDING:
            print(
                f"[aviso] texto de {len(tokens)} tokens truncado para {LIMITE_TOKENS_EMBEDDING}",
                file=sys.stderr,
            )
            return enc.decode(tokens[:LIMITE_TOKENS_EMBEDDING]), LIMITE_TOKENS_EMBEDDING
        return texto, len(tokens)

    def _enviar(self, textos: list[str], estimados: int) -> list[list[float]]:
        self.livro.autorizar("openai", custo_embedding(self.modelo, estimados))
        r = self.bruto.embeddings.create(model=self.modelo, input=textos)
        self.livro.registrar("openai", custo_embedding(self.modelo, r.usage.prompt_tokens))
        return [d.embedding for d in r.data]

    def embutir(self, textos: list[str]) -> np.ndarray:
        if not textos:
            return np.zeros((0, 0), dtype=np.float32)
        vetores: list[list[float]] = []
        lote: list[str] = []
        tokens_lote = 0
        for texto in textos:
            texto, n = self._preparar(texto)
            if lote and (len(lote) >= self.lote or tokens_lote + n > TOKENS_POR_REQUISICAO_EMBEDDING):
                vetores += self._enviar(lote, tokens_lote)
                lote, tokens_lote = [], 0
            lote.append(texto)
            tokens_lote += n
        if lote:
            vetores += self._enviar(lote, tokens_lote)
        return np.asarray(vetores, dtype=np.float32)


def fabricar_cliente(modelo: str, livro: LivroCaixa, temperatura: float | None):
    """Cliente do provedor do modelo. A Anthropic ignora a temperatura (o Haiku 5.5 a rejeita)."""
    if config.PROVEDOR[modelo] == "openai":
        return ClienteOpenAI(modelo, livro, temperatura=temperatura)
    return ClienteAnthropic(modelo, livro)
