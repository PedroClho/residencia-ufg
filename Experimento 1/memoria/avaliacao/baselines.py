"""Baselines da comparação: o piso (sem memória) e o teto (contexto completo).

Servem de régua para as duas memórias. Sem memória mostra o que o modelo acerta sozinho, só com a
janela da sessão atual: tudo o que passiva e ativa acertarem acima disso vem da memória. Contexto
completo mostra o que o modelo acerta quando vê a conversa inteira: é o teto de informação
disponível e o mais caro, porque reenvia tudo a cada turno. Equivale à linha "long-context" do
LongMemEval.

As duas classes têm a mesma interface de Passiva e Ativa (`(llm, store)`, `nova_sessao(data)` e
`responder(mensagem)`), então o runner da avaliação trata as quatro configurações do mesmo jeito.
Nenhuma das duas usa o banco: o parâmetro `store` existe só para a assinatura ser igual. Cada
classe se lê de cima para baixo, sem depender de passiva.py, mesmo repetindo algumas linhas.
"""

from memoria.config import K, SYSTEM_BASE
from memoria.llm import LLM, texto
from memoria.store import Store


class SemMemoria:
    """Piso: o modelo só vê as últimas K mensagens da sessão atual. Nada atravessa sessões.

    É a Passiva sem o resumo: mesmo system base, mesma data, mesma regra de janela. O parâmetro
    `store` não é usado; existe só para a interface ser igual à dos exemplos.
    """

    def __init__(self, llm: LLM, store: Store):
        self.llm = llm
        self.janela: list[dict] = []  # mensagens da sessão atual, texto puro, usuário e assistente
        self.data = ""  # data da sessão atual; quem usa a classe chama nova_sessao antes de responder

    def nova_sessao(self, data: str) -> None:
        """Começa uma sessão nova: guarda a data de hoje (AAAA-MM-DD) e esvazia a janela.

        Como não há memória, o que foi dito nas sessões anteriores some de vez.
        """
        self.data = data
        self.janela = []

    def responder(self, mensagem: str) -> str:
        """Responde à mensagem do usuário só com a janela da sessão atual. Devolve a resposta.

        Se a API falhar ou recusar, a resposta é o texto `[erro: ...]` (ver `llm.texto`). Ela entra
        na janela como qualquer resposta, para manter a alternância usuário/assistente.
        """
        system = f"{SYSTEM_BASE}\n\nHoje é {self.data}."
        nova = {"role": "user", "content": mensagem}
        resposta = texto(self.llm.responder(system, self.janela[-K:] + [nova]))
        self.janela += [nova, {"role": "assistant", "content": resposta}]
        return resposta


class ContextoCompleto:
    """Teto: o modelo vê todas as mensagens de todas as sessões, sem janela e sem memória externa.

    O histórico só cresce. Cada sessão nova abre com um par de marcador (a data da sessão e um
    "Ok." do assistente), para o modelo saber onde uma sessão termina e a outra começa e em que
    dia cada coisa foi dita; o par também mantém a alternância usuário/assistente. O system tem só
    a data da sessão atual, igual ao de SemMemoria. O parâmetro `store` não é usado; existe só
    para a interface ser igual à dos exemplos.
    """

    def __init__(self, llm: LLM, store: Store):
        self.llm = llm
        self.historico: list[dict] = []  # todas as mensagens de todas as sessões, em ordem
        self.data = ""  # data da sessão atual; quem usa a classe chama nova_sessao antes de responder

    def nova_sessao(self, data: str) -> None:
        """Começa uma sessão nova: guarda a data de hoje (AAAA-MM-DD) e marca a virada no histórico.

        Nada é apagado. O marcador é um par usuário/assistente, e não uma linha solta, para duas
        sessões vazias seguidas continuarem alternando os papéis.
        """
        self.data = data
        self.historico += [
            {"role": "user", "content": f"(Nova sessão. Hoje é {data}.)"},
            {"role": "assistant", "content": "Ok."},
        ]

    def responder(self, mensagem: str) -> str:
        """Responde à mensagem do usuário vendo o histórico inteiro. Devolve a resposta.

        Se a API falhar ou recusar, a resposta é o texto `[erro: ...]` (ver `llm.texto`). Ela entra
        no histórico como qualquer resposta, para manter a alternância usuário/assistente.
        """
        system = f"{SYSTEM_BASE}\n\nHoje é {self.data}."
        nova = {"role": "user", "content": mensagem}
        resposta = texto(self.llm.responder(system, self.historico + [nova]))
        self.historico += [nova, {"role": "assistant", "content": resposta}]
        return resposta
