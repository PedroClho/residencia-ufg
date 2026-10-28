"""Exemplo 1: memória passiva, por sumarização incremental.

Aqui a memória é uma política do código, não uma escolha do modelo. A cada turno o código sempre
lê o resumo S da conversa, sempre o injeta no prompt e sempre o atualiza depois da resposta. O
modelo que conversa com o usuário nunca fica sabendo que existe memória: para ele, S é só mais um
trecho de texto do system prompt. (No exemplo 2, ativa.py, é o modelo quem decide quando ler e
quando gravar.)

Pipeline de um turno:

    mensagem do usuário
       |
       +--> 1. carrega o resumo atual S da conversa (SQLite)
       |
       +--> 2. monta o prompt: system base + data + bloco "Resumo do que você sabe..." + S
       |       + últimas K mensagens da sessão + mensagem nova
       |       +--> modelo responde (sem ferramentas)
       |
       +--> 3. chamada ao resumidor: recebe S + (mensagem, resposta) + data da sessão
               +--> devolve S' e grava no SQLite (S' passa a ser o S do próximo turno)

Decisões de desenho:

- S tem estrutura fixa, em Markdown, com quatro seções (ver RESUMO_VAZIO). O resumidor é instruído
  a editar essas seções, não a reescrever o resumo do zero: manter o que continua válido,
  acrescentar o novo, substituir o que foi contradito. É a mitigação do colapso de contexto
  descrito no ACE, em que reescritas livres vão apagando detalhes a cada passada.
- S tem limite de tamanho (LIMITE_PALAVRAS_RESUMO). Sem limite, S cresceria a cada turno e, no
  fim, custaria o mesmo que reenviar a conversa inteira.
- O resumo é refeito a cada turno. Resumir só quando a janela estoura sairia mais barato, mas
  seria outra variante, com outro comportamento.
- A janela recente são as últimas K mensagens da sessão atual e zera quando começa uma sessão
  nova. Dentro da sessão a janela ajuda; entre sessões, só S ajuda.
- O histórico guarda só texto. Um turno com erro (`[erro: ...]`) entra nele como resposta do
  assistente, e o resumidor não roda nesse turno.

Para conversar pelo terminal: `python -m memoria.passiva`.
"""

from datetime import date

from memoria.config import CAMINHO_BANCO, K, LIMITE_PALAVRAS_RESUMO, SYSTEM_BASE
from memoria.llm import LLM, texto
from memoria.store import Store

# O resumo antes de a conversa começar: as quatro seções, todas vazias. Serve de S no primeiro
# turno, para o modelo e o resumidor já verem a estrutura que o resumo vai ter.
RESUMO_VAZIO = """\
## Sobre o usuário
(nada ainda)

## Fatos e decisões
(nada ainda)

## Pendências
(nada ainda)

## Linha do tempo
(nada ainda)"""

# System prompt do resumidor, a chamada que atualiza S. Os marcadores {limite} e {data} são
# preenchidos a cada chamada. A instrução de editar (e não reescrever) é a mitigação do colapso
# de contexto; a de devolver só o resumo evita que um comentário do modelo vire parte de S.
PROMPT_RESUMIDOR = """\
Você mantém o resumo de uma conversa entre um usuário e um assistente. Esse resumo é tudo o que o \
assistente vai lembrar da conversa nos próximos turnos, então ele precisa guardar o que importa.

O resumo tem sempre estas quatro seções, nesta ordem:

## Sobre o usuário
## Fatos e decisões
## Pendências
## Linha do tempo

Você recebe o RESUMO ATUAL e o NOVO TURNO (a mensagem do usuário e a resposta do assistente). \
Devolva o resumo atualizado. EDITE o resumo atual, não o reescreva do zero:
- mantenha o que continua válido;
- acrescente o que o novo turno trouxe de novo;
- substitua o que o novo turno contradisse ou corrigiu;
- registre na seção Linha do tempo o que tem data, no formato `AAAA-MM-DD: evento`. Converta datas \
relativas ("ontem", "semana que vem") em datas completas a partir de hoje.
Seção sem conteúdo fica com `(nada ainda)`.

O resumo não pode passar de {limite} palavras. Se passar, comprima o que for menos importante.

Hoje é {data}.

Responda SOMENTE com o resumo em Markdown, sem comentários, sem explicações e sem cercas de código."""


class Passiva:
    """Assistente com memória passiva: o código lê, injeta e atualiza o resumo S a cada turno.

    O banco (`store`) é a memória: S é relido dele a cada turno. A instância guarda só o que é da
    sessão atual, que é a janela de mensagens recentes e a data de hoje.
    """

    def __init__(self, llm: LLM, store: Store):
        self.llm = llm
        self.store = store
        self.janela: list[dict] = []  # mensagens da sessão atual, texto puro, usuário e assistente
        self.data = ""  # data da sessão atual; quem usa a classe chama nova_sessao antes de responder

    def nova_sessao(self, data: str) -> None:
        """Começa uma sessão nova: guarda a data de hoje (AAAA-MM-DD) e esvazia a janela.

        O banco não é tocado: o resumo atravessa a troca de sessão, e é esse o ponto da memória.
        """
        self.data = data
        self.janela = []

    def responder(self, mensagem: str) -> str:
        """Responde à mensagem do usuário e, se deu certo, atualiza o resumo. Devolve a resposta.

        Se a API falhar ou recusar, a resposta é o texto `[erro: ...]` (ver `llm.texto`). Ela entra
        na janela como qualquer resposta, para manter a alternância usuário/assistente, mas o
        resumo não é atualizado: não há conteúdo real para resumir.
        """
        # Passo 1: o resumo atual vem do banco. No primeiro turno ainda não há nenhum.
        resumo = self.store.ler_resumo() or RESUMO_VAZIO

        # Passo 2: monta o prompt e pergunta ao modelo, que só vê S como texto no system.
        system = (
            f"{SYSTEM_BASE}\n\n"
            f"Hoje é {self.data}.\n\n"
            f"Resumo do que você sabe sobre o usuário e a conversa até aqui:\n{resumo}"
        )
        nova = {"role": "user", "content": mensagem}
        resposta = texto(self.llm.responder(system, self.janela[-K:] + [nova]))
        self.janela += [nova, {"role": "assistant", "content": resposta}]

        # Passo 3: o código atualiza S sozinho, sem o modelo principal saber.
        if not resposta.startswith("[erro:"):
            self._atualizar_resumo(resumo, mensagem, resposta)
        return resposta

    def _atualizar_resumo(self, resumo: str, mensagem: str, resposta: str) -> None:
        """Pede ao resumidor o resumo atualizado e o grava no banco, substituindo o anterior.

        O resumidor recebe S e o turno que acabou de acontecer, e devolve o S' inteiro. Só se grava
        S' se ele veio com texto e sem erro: uma falha (ou uma saída vazia, por exemplo cortada
        antes de escrever o resumo) apagaria a memória inteira. Com falha, S anterior fica e o
        próximo turno tenta de novo.
        """
        system = PROMPT_RESUMIDOR.format(limite=LIMITE_PALAVRAS_RESUMO, data=self.data)
        conteudo = (
            f"RESUMO ATUAL:\n{resumo}\n\n"
            f"NOVO TURNO:\nUsuário: {mensagem}\nAssistente: {resposta}"
        )
        novo = texto(self.llm.responder(system, [{"role": "user", "content": conteudo}]))
        if novo and not novo.startswith("[erro:"):
            self.store.gravar_resumo(novo)


if __name__ == "__main__":
    # Chat de terminal para a demonstração. A memória fica em CAMINHO_BANCO e sobrevive entre
    # execuções. /nova simula a troca de sessão: zera a janela, mas o resumo continua.
    store = Store(CAMINHO_BANCO)
    passiva = Passiva(LLM(), store)
    hoje = date.today().isoformat()
    passiva.nova_sessao(hoje)
    print("Chat com memória passiva. /nova começa uma sessão nova; Ctrl+D encerra (Ctrl+Z e Enter no Windows).")
    while True:
        try:
            linha = input("você> ").strip()
        except EOFError:
            break
        if not linha:  # mensagem vazia é recusada pela API e estragaria a janela
            continue
        if linha == "/nova":
            passiva.nova_sessao(hoje)
            print("(sessão nova: janela zerada, resumo mantido)")
            continue
        print(passiva.responder(linha))
        print("--- resumo ---")
        print(store.ler_resumo())
        print("--- resumo ---")
    store.fechar()
