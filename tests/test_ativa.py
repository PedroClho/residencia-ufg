"""Testes da memória ativa (ativa.py): o loop de ferramentas, o que volta ao modelo e o que fica na janela."""

from types import SimpleNamespace

from memoria.ativa import BLOCO_MEMORIA, FERRAMENTAS, Ativa
from memoria.config import K
from memoria.llm import LLM
from memoria.store import Store
from tests.conftest import erro_api, resposta_ferramenta, resposta_recusa, resposta_texto


def criar(cliente, store=None, data="2026-03-01"):
    """Uma Ativa com a sessão já começada, para os testes não repetirem a preparação."""
    ativa = Ativa(LLM(cliente), store or Store())
    ativa.nova_sessao(data)
    return ativa


def tool_result(id, conteudo):
    """O bloco que o código deve devolver ao modelo para um tool_use."""
    return {"type": "tool_result", "tool_use_id": id, "content": conteudo}


def test_ferramentas_tem_schema_estrito():
    argumento_de = {"buscar_memoria": "consulta", "gravar_memoria": "conteudo"}
    assert [t["name"] for t in FERRAMENTAS] == list(argumento_de)
    for t in FERRAMENTAS:
        assert t["strict"] is True and t["description"]
        schema = t["input_schema"]
        assert list(schema["properties"]) == [argumento_de[t["name"]]]
        assert schema["properties"][argumento_de[t["name"]]]["type"] == "string"
        assert schema["required"] == [argumento_de[t["name"]]]
        assert schema["additionalProperties"] is False


def test_bloco_memoria_diz_o_essencial_ao_modelo():
    assert "janela" in BLOCO_MEMORIA  # a janela é curta e sessões anteriores não aparecem
    assert "perdido" in BLOCO_MEMORIA  # o que não for gravado se perde
    assert "gravar_memoria" in BLOCO_MEMORIA and "buscar_memoria" in BLOCO_MEMORIA
    assert "mais recente" in BLOCO_MEMORIA  # notas que se contradizem: vale a mais recente


def test_sem_ferramenta_e_um_turno_simples(fake):
    cliente = fake([resposta_texto("Olá!")])
    a = criar(cliente)

    assert a.responder("oi") == "Olá!"

    assert len(cliente.chamadas) == 1
    chamada = cliente.chamadas[0]
    assert [t["name"] for t in chamada["tools"]] == ["buscar_memoria", "gravar_memoria"]
    assert all(t["strict"] is True for t in chamada["tools"])
    assert BLOCO_MEMORIA in chamada["system"] and "Hoje é 2026-03-01" in chamada["system"]
    assert chamada["messages"] == [{"role": "user", "content": "oi"}]
    assert a.chamadas_ferramenta == {"buscar_memoria": 0, "gravar_memoria": 0}


def test_gravar_memoria_executa_e_loop_termina_no_end_turn(fake):
    cliente = fake([
        resposta_ferramenta("gravar_memoria", {"conteudo": "Cachorro se chama Thor"}, "tu_1"),
        resposta_texto("Anotado!"),
    ])
    store = Store()
    a = criar(cliente, store)

    assert a.responder("meu cachorro é o Thor") == "Anotado!"

    assert len(cliente.chamadas) == 2
    segunda = cliente.chamadas[1]["messages"]
    assert segunda[-2]["role"] == "assistant"
    assert segunda[-1] == {"role": "user", "content": [tool_result("tu_1", "Nota 1 gravada.")]}
    assert store.buscar_notas("Thor")[0]["data"] == "2026-03-01"
    assert a.chamadas_ferramenta == {"buscar_memoria": 0, "gravar_memoria": 1}


def test_o_conteudo_do_assistente_volta_inteiro_e_so_tool_use_gera_resultado(fake):
    primeira = resposta_ferramenta("gravar_memoria", {"conteudo": "Mora em Goiânia"}, "tu_1")
    primeira.content = [
        SimpleNamespace(type="thinking", thinking="vou anotar"),
        SimpleNamespace(type="text", text="Anotando."),
        *primeira.content,
    ]
    cliente = fake([primeira, resposta_texto("Pronto!")])
    a = criar(cliente)

    a.responder("moro em Goiânia")

    segunda = cliente.chamadas[1]["messages"]
    assert segunda[-2] == {"role": "assistant", "content": primeira.content}  # com o thinking, sem editar
    assert segunda[-1] == {"role": "user", "content": [tool_result("tu_1", "Nota 1 gravada.")]}


def test_buscar_memoria_devolve_notas_formatadas(fake):
    store = Store()
    store.gravar_nota("2026-02-01", "Cachorro se chama Thor")
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "cachorro"}, "tu_1"), resposta_texto("Thor")])
    a = criar(cliente, store)

    assert a.responder("como se chama meu cachorro?") == "Thor"

    ultima = cliente.chamadas[1]["messages"][-1]
    assert ultima == {"role": "user", "content": [tool_result("tu_1", "[1] (2026-02-01) Cachorro se chama Thor")]}
    assert a.chamadas_ferramenta == {"buscar_memoria": 1, "gravar_memoria": 0}


def test_buscar_com_varias_notas_devolve_uma_por_linha(fake):
    store = Store()
    store.gravar_nota("2026-02-01", "Mora em Goiânia")
    store.gravar_nota("2026-03-01", "Mudou para Brasília")
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "Goiânia Brasília"}), resposta_texto("ok")])
    a = criar(cliente, store)

    a.responder("onde moro?")

    conteudo = cliente.chamadas[1]["messages"][-1]["content"][0]["content"]
    assert sorted(conteudo.split("\n")) == ["[1] (2026-02-01) Mora em Goiânia", "[2] (2026-03-01) Mudou para Brasília"]


def test_buscar_sem_notas_devolve_mensagem_padrao(fake):
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "cachorro"}, "tu_1"), resposta_texto("Não sei.")])
    a = criar(cliente)

    a.responder("como se chama meu cachorro?")

    ultima = cliente.chamadas[1]["messages"][-1]
    assert ultima == {"role": "user", "content": [tool_result("tu_1", "Nenhuma nota encontrada.")]}


def test_chamadas_paralelas_voltam_em_uma_unica_mensagem(fake):
    cliente = fake([
        resposta_ferramenta([
            ("gravar_memoria", {"conteudo": "Mora em Goiânia"}, "tu_a"),
            ("buscar_memoria", {"consulta": "cachorro"}, "tu_b"),
        ]),
        resposta_texto("Pronto."),
    ])
    a = criar(cliente)

    a.responder("moro em Goiânia; e meu cachorro?")

    segunda = cliente.chamadas[1]["messages"]
    assert segunda[-2]["role"] == "assistant"
    assert segunda[-1] == {
        "role": "user",
        "content": [tool_result("tu_a", "Nota 1 gravada."), tool_result("tu_b", "Nenhuma nota encontrada.")],
    }
    assert a.chamadas_ferramenta == {"buscar_memoria": 1, "gravar_memoria": 1}


def test_o_modelo_pode_encadear_varias_rodadas_de_ferramenta(fake):
    cliente = fake([
        resposta_ferramenta("buscar_memoria", {"consulta": "cachorro"}, "tu_1"),
        resposta_ferramenta("gravar_memoria", {"conteudo": "Cachorro se chama Thor"}, "tu_2"),
        resposta_texto("Feito."),
    ])
    a = criar(cliente)

    assert a.responder("meu cachorro é o Thor") == "Feito."

    terceira = cliente.chamadas[2]["messages"]
    # usuário, (assistente, resultados) da rodada 1, (assistente, resultados) da rodada 2
    assert [m["role"] for m in terceira] == ["user", "assistant", "user", "assistant", "user"]
    assert terceira[-1]["content"] == [tool_result("tu_2", "Nota 1 gravada.")]


def test_janela_guarda_so_texto_entre_turnos(fake):
    cliente = fake([
        resposta_ferramenta("gravar_memoria", {"conteudo": "Cachorro se chama Thor"}),
        resposta_texto("Anotado!"),
        resposta_texto("De nada."),
    ])
    a = criar(cliente)

    a.responder("meu cachorro é o Thor")
    a.responder("obrigado")

    # O turno 2 vê só texto: nenhum tool_use nem tool_result do turno 1 atravessa.
    assert cliente.chamadas[2]["messages"] == [
        {"role": "user", "content": "meu cachorro é o Thor"},
        {"role": "assistant", "content": "Anotado!"},
        {"role": "user", "content": "obrigado"},
    ]


def test_janela_envia_exatamente_as_ultimas_k_mensagens(fake):
    cliente = fake([resposta_texto(f"resposta {i}") for i in range(1, 6)])
    a = criar(cliente)

    for i in range(1, 6):
        a.responder(f"mensagem {i}")

    historico = []
    for i in range(1, 5):
        historico += [{"role": "user", "content": f"mensagem {i}"}, {"role": "assistant", "content": f"resposta {i}"}]
    quinta = cliente.chamadas[4]
    assert quinta["messages"] == historico[-K:] + [{"role": "user", "content": "mensagem 5"}]
    assert len(quinta["messages"]) == K + 1


def test_nova_sessao_zera_janela_mas_a_nota_sobrevive_com_a_data_de_quando_foi_gravada(fake):
    cliente = fake([
        resposta_ferramenta("gravar_memoria", {"conteudo": "Cachorro se chama Thor"}),
        resposta_texto("Anotado!"),
        resposta_ferramenta("buscar_memoria", {"consulta": "cachorro"}),
        resposta_texto("Thor"),
    ])
    a = criar(cliente)
    a.responder("meu cachorro é o Thor")

    a.nova_sessao("2026-03-08")
    assert a.responder("como se chama meu cachorro?") == "Thor"

    primeira_do_turno_2 = cliente.chamadas[2]
    assert primeira_do_turno_2["messages"] == [{"role": "user", "content": "como se chama meu cachorro?"}]
    assert "Hoje é 2026-03-08" in primeira_do_turno_2["system"]
    resultado = cliente.chamadas[3]["messages"][-1]["content"][0]["content"]
    assert resultado == "[1] (2026-03-01) Cachorro se chama Thor"


def test_nota_gravada_numa_sessao_nova_leva_a_data_dela(fake):
    cliente = fake([resposta_ferramenta("gravar_memoria", {"conteudo": "Mudou para Brasília"}), resposta_texto("ok")])
    store = Store()
    a = criar(cliente, store, data="2026-03-01")
    a.nova_sessao("2026-03-08")

    a.responder("mudei para Brasília")

    assert store.buscar_notas("Brasília")[0]["data"] == "2026-03-08"


def test_limite_de_iteracoes(fake):
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "x"})] * 10)
    a = criar(cliente)

    assert a.responder("?") == "[erro: limite de ferramentas]"

    assert len(cliente.chamadas) == 6


def test_turno_que_estourou_o_limite_entra_na_janela_como_erro(fake):
    cliente = fake([resposta_ferramenta("buscar_memoria", {"consulta": "x"})] * 6 + [resposta_texto("ok")])
    a = criar(cliente)
    a.responder("primeira")

    a.responder("segunda")

    assert cliente.chamadas[6]["messages"] == [
        {"role": "user", "content": "primeira"},
        {"role": "assistant", "content": "[erro: limite de ferramentas]"},
        {"role": "user", "content": "segunda"},
    ]


def test_erro_de_api_no_meio_do_loop(fake):
    cliente = fake([resposta_ferramenta("gravar_memoria", {"conteudo": "x"}), erro_api()])
    a = criar(cliente)

    assert a.responder("anote x") == "[erro: api]"

    assert len(cliente.chamadas) == 2


def test_turno_com_erro_entra_na_janela_como_resposta_do_assistente(fake):
    cliente = fake([erro_api(), resposta_texto("Oi!")])
    a = criar(cliente)

    assert a.responder("primeira") == "[erro: api]"
    a.responder("segunda")

    assert cliente.chamadas[1]["messages"] == [
        {"role": "user", "content": "primeira"},
        {"role": "assistant", "content": "[erro: api]"},
        {"role": "user", "content": "segunda"},
    ]


def test_recusa_vira_erro(fake):
    a = criar(fake([resposta_recusa()]))
    assert a.responder("x") == "[erro: recusa]"


def test_resposta_final_vazia_vira_erro(fake):
    # Texto vazio na janela faria a API rejeitar todos os turnos seguintes.
    a = criar(fake([resposta_texto("")]))
    assert a.responder("x") == "[erro: resposta vazia]"


def test_contador_de_ferramentas_acumula_entre_turnos_e_e_por_instancia(fake):
    cliente = fake([
        resposta_ferramenta("gravar_memoria", {"conteudo": "a"}), resposta_texto("ok"),
        resposta_ferramenta("gravar_memoria", {"conteudo": "b"}), resposta_texto("ok"),
    ])
    a = criar(cliente)
    outra = criar(fake([]))

    a.responder("um")
    a.responder("dois")

    assert a.chamadas_ferramenta == {"buscar_memoria": 0, "gravar_memoria": 2}
    assert outra.chamadas_ferramenta == {"buscar_memoria": 0, "gravar_memoria": 0}


def test_verboso_imprime_cada_chamada_de_ferramenta(fake, capsys):
    cliente = fake([
        resposta_ferramenta("gravar_memoria", {"conteudo": "Mora em Goiânia"}), resposta_texto("ok"),
        resposta_ferramenta([
            ("gravar_memoria", {"conteudo": "Cachorro se chama Thor"}, "tu_a"),
            ("buscar_memoria", {"consulta": "cachorro"}, "tu_b"),
        ]),
        resposta_texto("ok"),
    ])
    a = criar(cliente)

    a.responder("moro em Goiânia")
    assert capsys.readouterr().out == ""  # por padrão não imprime nada

    a.verboso = True
    a.responder("meu cachorro é o Thor")
    assert capsys.readouterr().out == (
        "-> gravar_memoria(conteudo='Cachorro se chama Thor')\n"
        "-> buscar_memoria(consulta='cachorro')\n"
    )
