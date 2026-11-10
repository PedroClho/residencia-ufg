"""Cliente fake e construtores de resposta: os testes nunca chamam a API.

Os testes roteirizam o que o "modelo" responde, na ordem, e depois inspecionam o que o código
enviou. Uso típico:

    from tests.conftest import resposta_texto, resposta_ferramenta, resposta_recusa, erro_api

    def test_algo(fake):
        cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "x"}), resposta_texto("ok")])
"""

from types import SimpleNamespace

import anthropic
import httpx2  # dependência do SDK da anthropic; a classe de Request que ele espera
import pytest


def _resposta(stop_reason: str, content: list, entrada: int = 10, saida: int = 5, stop_details=None) -> SimpleNamespace:
    """Monta um objeto com o mesmo formato (os campos que o código lê) de uma resposta da API."""
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(input_tokens=entrada, output_tokens=saida),
        stop_details=stop_details,
    )


def resposta_texto(texto: str, entrada: int = 10, saida: int = 5) -> SimpleNamespace:
    """Resposta final do modelo: um bloco de texto e stop_reason "end_turn"."""
    return _resposta("end_turn", [SimpleNamespace(type="text", text=texto)], entrada, saida)


def resposta_ferramenta(nome: str | list[tuple[str, dict, str]], entrada: dict | None = None, id: str = "tu_1") -> SimpleNamespace:
    """Resposta em que o modelo pede ferramenta (stop_reason "tool_use").

    Uma chamada: resposta_ferramenta("gravar_memoria", {"conteudo": "x"}, id="tu_1").
    Várias no mesmo turno (chamadas paralelas): resposta_ferramenta([(nome, entrada, id), ...]).
    """
    chamadas = nome if isinstance(nome, list) else [(nome, entrada, id)]
    blocos = [SimpleNamespace(type="tool_use", name=n, input=e, id=i) for n, e, i in chamadas]
    return _resposta("tool_use", blocos)


def resposta_recusa() -> SimpleNamespace:
    """Resposta em que a API recusou o pedido: stop_reason "refusal", sem conteúdo."""
    return _resposta("refusal", [], stop_details=SimpleNamespace(category="x"))


def erro_api() -> anthropic.APIConnectionError:
    """Um erro de conexão da API, para colocar no roteiro: o FakeCliente o levanta em vez de devolver."""
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))


class FakeCliente:
    """Imita anthropic.Anthropic(): devolve respostas roteirizadas na ordem e guarda cada chamada."""

    def __init__(self, respostas: list):
        self.roteiro = list(respostas)
        self.chamadas = []
        self.messages = SimpleNamespace(create=self._criar)  # mesmo caminho de uso: cliente.messages.create(...)

    def _criar(self, **kwargs):
        """Guarda uma foto da chamada e devolve (ou levanta, se for exceção) o próximo item do roteiro.

        A cópia da lista `messages` evita que appends posteriores do chamador alterem o que foi
        registrado: chamadas[i] mostra exatamente o que foi enviado na chamada i.
        """
        self.chamadas.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.roteiro.pop(0) if self.roteiro else resposta_texto("(sem roteiro)")
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake():
    """Fábrica: fake([r1, r2]) -> FakeCliente."""
    return FakeCliente
