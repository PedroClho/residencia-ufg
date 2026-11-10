"""Fakes dos SDKs e fixture do livro-caixa. Nenhum teste chama API."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from replica.livro_caixa import LivroCaixa

USAGE_OPENAI_PADRAO = dict(prompt=1000, cached=800, completion=5)
USAGE_ANTHROPIC_PADRAO = dict(entrada=100, escrita=0, leitura=0, saida=5)


def _usage_openai(prompt, cached, completion):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


class FakeOpenAI:
    """Imita `openai.OpenAI`: chat.completions.create e embeddings.create."""

    def __init__(self, responder=None):
        self.chamadas = []
        self.chamadas_embeddings = []
        self._responder = responder or (lambda kw: ("resposta", dict(USAGE_OPENAI_PADRAO)))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embeddings)

    def _chat(self, **kw):
        self.chamadas.append(kw)
        texto, usage = self._responder(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=texto), finish_reason="stop")],
            usage=_usage_openai(**usage),
        )

    def _embeddings(self, **kw):
        self.chamadas_embeddings.append(kw)
        dados = []
        for t in kw["input"]:
            semente = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "little")
            v = np.random.default_rng(semente).normal(size=8)
            dados.append(SimpleNamespace(embedding=(v / np.linalg.norm(v)).tolist()))
        return SimpleNamespace(
            data=dados, usage=SimpleNamespace(prompt_tokens=sum(max(1, len(t) // 4) for t in kw["input"]))
        )


class FakeAnthropic:
    """Imita `anthropic.Anthropic`: messages.create."""

    def __init__(self, responder=None, stop_reason="end_turn"):
        self.chamadas = []
        self.stop_reason = stop_reason
        self._responder = responder or (lambda kw: ("resposta", dict(USAGE_ANTHROPIC_PADRAO)))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.chamadas.append(kw)
        texto, u = self._responder(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=texto)] if texto else [],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=u["entrada"],
                cache_creation_input_tokens=u["escrita"],
                cache_read_input_tokens=u["leitura"],
                output_tokens=u["saida"],
            ),
        )


@pytest.fixture
def livro(tmp_path):
    return LivroCaixa(tmp_path / "livro_caixa.json")


@pytest.fixture
def fake_openai():
    return FakeOpenAI()


@pytest.fixture
def fake_anthropic():
    return FakeAnthropic()
