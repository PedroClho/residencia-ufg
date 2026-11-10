"""Clientes de LLM e de embeddings. Toda chamada passa pelo livro-caixa: autoriza antes, registra depois."""
import sys
import time
from dataclasses import dataclass

import numpy as np
import tiktoken

from replica import config
from replica.livro_caixa import (
    LimiteDeGasto,
    LimiteDoPainel,
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


def _texto_do_erro(e: Exception) -> str:
    return f"{e} {getattr(e, 'code', None) or ''} {getattr(e, 'body', None) or ''}".casefold()


_SINAIS_OPENAI = ("no credits remaining", "add credits", "exceeded your current quota", "insufficient_quota")


def _mensagens_do_corpo(e: Exception) -> str:
    corpo = getattr(e, "body", None)
    if not isinstance(corpo, dict):
        return ""
    interno = corpo.get("error")
    partes = [interno.get("message") if isinstance(interno, dict) else None, corpo.get("message")]
    return " ".join(p for p in partes if isinstance(p, str)).casefold()


def _cota_openai_esgotada(e: Exception) -> bool:
    """429 de cota/saldo: a trava do painel da OpenAI (ou falta de crédito). Esperar não adianta.

    Pelo código/tipo `insufficient_quota` ou pelo texto da mensagem, que nem sempre traz o código.
    Limite de taxa comum ("Rate limit reached ... try again in 20ms") não casa."""
    if getattr(e, "code", None) == "insufficient_quota":
        return True
    corpo = getattr(e, "body", None)
    interno = corpo.get("error") if isinstance(corpo, dict) else None
    if isinstance(interno, dict) and "insufficient_quota" in (interno.get("code"), interno.get("type")):
        return True
    texto = f"{_texto_do_erro(e)} {_mensagens_do_corpo(e)}"
    return any(s in texto for s in _SINAIS_OPENAI)


_SINAIS_ANTHROPIC = ("credit balance", "usage limit", "spend limit")


def _limite_anthropic(e: Exception) -> bool:
    """Saldo baixo ou limite de uso/gasto do painel da Anthropic. Esperar não adianta."""
    texto = _texto_do_erro(e)
    return any(s in texto for s in _SINAIS_ANTHROPIC)


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

    def _limite_do_painel(self, e: Exception) -> bool:
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
        """Chama a API; em limite de taxa espera 60 s e tenta de novo, até 5 vezes. Limite do painel (cota, saldo
        ou trava de gasto) levanta `LimiteDoPainel` na hora, sem esperar."""
        tentativas = 0
        while True:
            try:
                return chamada()
            except Exception as e:
                if self._limite_do_painel(e):
                    raise LimiteDoPainel(self.provedor, f"{self.provedor}: limite do painel ({e})") from e
                if not isinstance(e, self.erros_de_taxa) or tentativas >= TENTATIVAS_TAXA:
                    raise
                tentativas += 1
                self._dormir(ESPERA_TAXA_S)

    def conversar(self, system, mensagem, max_tokens, prefixo_cacheavel=None) -> Resposta:
        """Autoriza, chama e registra o custo.

        - Erro antes de a API responder vira `[erro: api]` sem custo (a retomada refaz).
        - Depois que a API respondeu, a chamada foi paga: o custo sai do `usage` e é registrado antes de ler o
          texto. Uma resposta ilegível vira `[erro: resposta vazia]` com o custo real (resposta final, não é
          refeita). Sem `usage` legível, registra o custo máximo autorizado.
        - `LimiteDeGasto` (inclusive `LimiteDoPainel`) e `ErroDoLivroCaixa` propagam.
        """
        prefixo = prefixo_cacheavel or ""
        custo_maximo = self._custo_maximo(system or "", prefixo, mensagem, max_tokens)
        self.livro.autorizar(self.provedor, custo_maximo)
        try:
            r = self._enviar(system, prefixo, mensagem, max_tokens, bool(prefixo_cacheavel))
        except LimiteDeGasto:
            raise
        except Exception as e:
            print(f"[erro: api] {e!r}", file=sys.stderr)
            return _resposta_de_erro("[erro: api]")
        try:
            entrada, cacheados, escritos, saida, custo = self._uso(r)
        except Exception as e:
            print(f"[aviso] usage ilegível na resposta paga ({e!r}); registro o custo máximo autorizado",
                  file=sys.stderr)
            entrada, cacheados, escritos, saida, custo = 0, 0, 0, 0, custo_maximo
        self.livro.registrar(self.provedor, custo)
        try:
            texto, erro = self._texto(r)
        except Exception as e:
            print(f"[erro: resposta vazia] resposta paga ilegível ({e!r})", file=sys.stderr)
            texto, erro = "", "[erro: resposta vazia]"
        return Resposta(texto, entrada, cacheados, escritos, saida, custo, erro)


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

    def _limite_do_painel(self, e):
        return _cota_openai_esgotada(e)

    def _custo_maximo(self, system, prefixo, mensagem, max_tokens):
        estimados = _contar_chat(system + prefixo + mensagem)
        return custo_openai(self.modelo, estimados, 0, max_tokens)

    def _enviar(self, system, prefixo, mensagem, max_tokens, _com_prefixo):
        mensagens = [{"role": "user", "content": prefixo + mensagem}]
        if system is not None:
            mensagens.insert(0, {"role": "system", "content": system})
        kwargs = dict(model=self.modelo, messages=mensagens, max_tokens=max_tokens)
        if self.temperatura is not None:
            kwargs["temperature"] = self.temperatura
        return self._chamar_com_espera(lambda: self.bruto.chat.completions.create(**kwargs))

    def _uso(self, r):
        u = r.usage
        detalhes = getattr(u, "prompt_tokens_details", None)
        cacheados = (getattr(detalhes, "cached_tokens", 0) or 0) if detalhes else 0
        custo = custo_openai(self.modelo, u.prompt_tokens, cacheados, u.completion_tokens)
        return u.prompt_tokens, cacheados, 0, u.completion_tokens, custo

    def _texto(self, r):
        texto = r.choices[0].message.content or ""
        return texto, (None if texto else "[erro: resposta vazia]")


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

    def _limite_do_painel(self, e):
        return _limite_anthropic(e)

    def _custo_maximo(self, system, prefixo, mensagem, max_tokens):
        fator = config.FATOR_TOKENS_HAIKU
        est_prefixo = int(_contar_chat(prefixo) * fator)
        est_resto = int(_contar_chat(system + mensagem) * fator)
        return custo_anthropic(self.modelo, est_resto, est_prefixo, 0, max_tokens)

    def _enviar(self, system, prefixo, mensagem, max_tokens, com_prefixo):
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
        return self._chamar_com_espera(lambda: self.bruto.messages.create(**kwargs))

    def _uso(self, r):
        u = r.usage
        sem_cache = getattr(u, "input_tokens", 0) or 0
        escrita = getattr(u, "cache_creation_input_tokens", 0) or 0
        leitura = getattr(u, "cache_read_input_tokens", 0) or 0
        saida = getattr(u, "output_tokens", 0) or 0
        custo = custo_anthropic(self.modelo, sem_cache, escrita, leitura, saida)
        return sem_cache + escrita + leitura, leitura, escrita, saida, custo

    def _texto(self, r):
        texto = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        if r.stop_reason == "refusal":
            return "", "[erro: recusa]"
        return texto, (None if texto else "[erro: resposta vazia]")


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
        """Um lote. Cota esgotada vira `LimiteDoPainel`; o custo é registrado antes de ler os vetores (sem `usage`
        legível, vale o custo estimado)."""
        custo_estimado = custo_embedding(self.modelo, estimados)
        self.livro.autorizar("openai", custo_estimado)
        try:
            r = self.bruto.embeddings.create(model=self.modelo, input=textos)
        except Exception as e:
            if _cota_openai_esgotada(e):
                raise LimiteDoPainel("openai", f"openai: limite do painel nos embeddings ({e})") from e
            raise
        try:
            custo = custo_embedding(self.modelo, r.usage.prompt_tokens)
        except Exception as e:
            print(f"[aviso] usage ilegível nos embeddings ({e!r}); registro o custo estimado", file=sys.stderr)
            custo = custo_estimado
        self.livro.registrar("openai", custo)
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
