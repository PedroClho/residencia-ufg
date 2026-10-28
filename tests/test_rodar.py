"""Testes do runner (rodar.py): agregação, contagem de erros, tabela e arquivos de saída.

Tudo com cliente fake. Os testes de `rodar_configuracao` usam um dataset mínimo (2 sessões de uma
fala, 2 perguntas de categorias diferentes); os de `main` usam o dataset.json de verdade, só para
conferir que o runner lê o arquivo real sem depender do conteúdo dele.
"""

import json
import re

import pytest

from memoria.avaliacao import rodar
from memoria.avaliacao.rodar import CATEGORIAS, CONFIGURACOES, rodar_configuracao, tabela
from memoria.llm import LLM
from tests.conftest import erro_api, resposta_ferramenta, resposta_recusa, resposta_texto

DATASET = {
    "usuario": "Larissa",
    "sessoes": [
        {"data": "2026-03-01", "falas": ["Meu cachorro se chama Paçoca."]},
        {"data": "2026-03-08", "falas": ["A semana foi corrida."]},
    ],
    "perguntas": {
        "data": "2026-05-11",
        "itens": [
            {"id": "q01", "categoria": "sessao_unica", "pergunta": "Como se chama o meu cachorro?", "esperado": "Paçoca"},
            {"id": "q02", "categoria": "abstencao", "pergunta": "Como se chama a minha irmã?", "esperado": "Não foi dito: nenhuma irmã foi mencionada"},
        ],
    },
}


def _veredito(correto: bool, entrada: int = 10, saida: int = 5):
    """Resposta roteirizada do juiz: o JSON que a saída estruturada devolveria."""
    return resposta_texto(json.dumps({"correto": correto, "justificativa": "x"}), entrada, saida)


def _resultado(nome: str, por_categoria: dict, total: list, ferramentas: dict | None = None) -> dict:
    """Um resultado já pronto, no formato de `rodar_configuracao`, para testar a tabela sozinha."""
    return {
        "configuracao": nome,
        "turnos": [],
        "resumos_por_sessao": [],
        "ferramentas": ferramentas or {},
        "respostas": [],
        "acertos_por_categoria": por_categoria,
        "acertos_total": total,
        "erros": 0,
        "chamadas": 4,
        "tokens_entrada": 1200,
        "tokens_saida": 300,
        "custo_usd": 0.0054,
    }


def test_configuracoes_e_categorias_na_ordem_certa():
    assert list(CONFIGURACOES) == ["sem_memoria", "contexto_completo", "passiva", "ativa"]
    assert CATEGORIAS == ["sessao_unica", "multi_sessao", "temporal", "atualizacao", "abstencao"]


def test_rodar_configuracao_sem_memoria_agrega(fake):
    cliente = fake([resposta_texto("r")] * 10)
    llm = LLM(cliente)
    juiz = LLM(fake([_veredito(True), _veredito(False)]))
    r = rodar_configuracao("sem_memoria", DATASET, llm, juiz)
    assert r["configuracao"] == "sem_memoria"
    assert r["acertos_total"] == [1, 2]
    assert r["acertos_por_categoria"] == {"sessao_unica": [1, 1], "abstencao": [0, 1]}
    assert len(r["turnos"]) == 2 and r["chamadas"] == 4 and r["erros"] == 0
    assert r["resumos_por_sessao"] == [] and r["ferramentas"] == {}
    # 2 falas + 2 perguntas, 10 tokens de entrada e 5 de saída cada; o juiz fica de fora.
    assert (r["tokens_entrada"], r["tokens_saida"]) == (40, 20)
    assert r["custo_usd"] == pytest.approx(0.00028)
    assert juiz.contador.chamadas == 2


def test_rodar_configuracao_registra_turnos_e_respostas(fake):
    juiz = LLM(fake([_veredito(True), _veredito(False)]))
    r = rodar_configuracao("sem_memoria", DATASET, LLM(fake([resposta_texto("r")] * 10)), juiz)
    assert r["turnos"][0] == {"sessao": "2026-03-01", "usuario": "Meu cachorro se chama Paçoca.", "assistente": "r"}
    assert r["turnos"][1]["sessao"] == "2026-03-08"
    assert r["respostas"][0] == {
        "id": "q01",
        "categoria": "sessao_unica",
        "pergunta": "Como se chama o meu cachorro?",
        "esperado": "Paçoca",
        "resposta": "r",
        "correto": True,
        "justificativa": "x",
    }
    assert r["respostas"][1]["id"] == "q02" and r["respostas"][1]["correto"] is False


def test_rodar_configuracao_pergunta_na_sessao_final(fake):
    """As perguntas vêm depois das falas, numa sessão nova com a data das perguntas."""
    cliente = fake([resposta_texto("r")] * 10)
    rodar_configuracao("sem_memoria", DATASET, LLM(cliente), LLM(fake([_veredito(True)] * 2)))
    assert "Hoje é 2026-03-01." in cliente.chamadas[0]["system"]
    assert "Hoje é 2026-03-08." in cliente.chamadas[1]["system"]
    assert "Hoje é 2026-05-11." in cliente.chamadas[2]["system"]
    assert cliente.chamadas[2]["messages"][-1]["content"] == "Como se chama o meu cachorro?"


def test_rodar_configuracao_manda_ao_juiz_pergunta_esperado_e_resposta(fake):
    cliente_juiz = fake([_veredito(True)] * 2)
    rodar_configuracao("sem_memoria", DATASET, LLM(fake([resposta_texto("Paçoca!")] * 10)), LLM(cliente_juiz))
    pedido = cliente_juiz.chamadas[0]["messages"][0]["content"]
    assert "Categoria: sessao_unica" in pedido
    assert "Pergunta: Como se chama o meu cachorro?" in pedido
    assert "Resposta esperada: Paçoca" in pedido
    assert "Resposta do assistente: Paçoca!" in pedido


def test_rodar_configuracao_zera_o_contador_antes_de_comecar(fake):
    llm = LLM(fake([resposta_texto("r")] * 10))
    llm.contador.chamadas = 99  # sobra da configuração anterior
    r = rodar_configuracao("sem_memoria", DATASET, llm, LLM(fake([_veredito(True)] * 2)))
    assert r["chamadas"] == 4


def test_rodar_passiva_registra_resumo_por_sessao(fake):
    # A passiva faz 2 chamadas por turno: a resposta e o resumidor.
    roteiro = [
        resposta_texto("a1"), resposta_texto("resumo da sessão um"),
        resposta_texto("a2"), resposta_texto("resumo das duas sessões até aqui"),
        resposta_texto("a3"), resposta_texto("resumo com a pergunta um"),
        resposta_texto("a4"), resposta_texto("resumo com a pergunta dois"),
    ]  # fmt: skip
    llm = LLM(fake(roteiro))
    r = rodar_configuracao("passiva", DATASET, llm, LLM(fake([_veredito(True)] * 2)))
    assert r["resumos_por_sessao"] == [
        {"sessao": "2026-03-01", "palavras": 4, "resumo": "resumo da sessão um"},
        {"sessao": "2026-03-08", "palavras": 6, "resumo": "resumo das duas sessões até aqui"},
    ]
    assert r["chamadas"] == 8 and r["ferramentas"] == {}
    assert [t["assistente"] for t in r["turnos"]] == ["a1", "a2"]  # o resumidor não vira turno
    assert [x["resposta"] for x in r["respostas"]] == ["a3", "a4"]


def test_rodar_ativa_registra_ferramentas(fake):
    roteiro = [
        resposta_ferramenta("gravar_memoria", {"conteudo": "O cachorro se chama Paçoca."}),
        resposta_texto("a1"),
        resposta_texto("a2"),
        resposta_ferramenta("buscar_memoria", {"consulta": "cachorro"}),
        resposta_texto("Paçoca"),
        resposta_texto("Não sei."),
    ]
    llm = LLM(fake(roteiro))
    r = rodar_configuracao("ativa", DATASET, llm, LLM(fake([_veredito(True)] * 2)))
    assert r["ferramentas"] == {"buscar_memoria": 1, "gravar_memoria": 1}
    assert r["resumos_por_sessao"] == []
    assert r["chamadas"] == 6
    assert [x["resposta"] for x in r["respostas"]] == ["Paçoca", "Não sei."]


def test_rodar_conta_erros(fake):
    # A segunda pergunta é recusada: vira "[erro: recusa]", o juiz ainda a julga e ela segue na conta de N.
    cliente_juiz = fake([_veredito(True), _veredito(False)])
    llm = LLM(fake([resposta_texto("r"), resposta_texto("r"), resposta_texto("Paçoca"), resposta_recusa()]))
    r = rodar_configuracao("sem_memoria", DATASET, llm, LLM(cliente_juiz))
    assert r["erros"] == 1
    assert r["respostas"][1]["resposta"] == "[erro: recusa]"
    assert len(cliente_juiz.chamadas) == 2
    assert "Resposta do assistente: [erro: recusa]" in cliente_juiz.chamadas[1]["messages"][0]["content"]
    assert r["acertos_total"] == [1, 2]
    assert len(r["turnos"]) == 2  # as perguntas não entram em turnos


def test_rodar_conta_erros_da_conversa_e_das_perguntas(fake):
    # A primeira fala falha na API ("[erro: api]") e a primeira pergunta é recusada.
    llm = LLM(fake([erro_api(), resposta_texto("r"), resposta_recusa(), resposta_texto("r")]))
    r = rodar_configuracao("sem_memoria", DATASET, llm, LLM(fake([_veredito(False)] * 2)))
    assert r["turnos"][0]["assistente"] == "[erro: api]"
    assert r["erros"] == 2
    assert r["chamadas"] == 3  # a chamada que falhou não chegou a ser contada


def test_acertos_por_categoria_so_tem_as_do_dataset_na_ordem_das_categorias(fake):
    r = rodar_configuracao(
        "sem_memoria", DATASET, LLM(fake([resposta_texto("r")] * 10)), LLM(fake([_veredito(True)] * 2))
    )
    assert list(r["acertos_por_categoria"]) == ["sessao_unica", "abstencao"]


def test_tabela_tem_uma_linha_por_configuracao_e_colunas_fixas():
    r1 = _resultado("sem_memoria", {"sessao_unica": [1, 3], "abstencao": [2, 4]}, [3, 7])
    r2 = _resultado("ativa", {"sessao_unica": [3, 3], "abstencao": [4, 4]}, [7, 7], {"buscar_memoria": 5, "gravar_memoria": 2})
    md = tabela([r1, r2])
    linhas = md.splitlines()
    assert md.count("\n| ") == 2  # uma linha por configuração, fora cabeçalho e separador
    assert linhas[0].startswith("| configuração |")
    assert "sessao_unica" in linhas[0] and "abstencao" in linhas[0]
    assert "total" in linhas[0] and "tokens entrada" in linhas[0] and "custo (USD)" in linhas[0]
    # as colunas de categoria seguem a ordem de CATEGORIAS, depois vêm total e as medidas
    cabecalho = [c.strip() for c in linhas[0].strip("|").split("|")]
    assert cabecalho[1:6] == CATEGORIAS
    assert cabecalho[6:] == [
        "total", "erros", "chamadas", "tokens entrada", "tokens saída", "custo (USD)", "ferramentas (buscar/gravar)",
    ]  # fmt: skip
    assert all(len(l.strip("|").split("|")) == len(cabecalho) for l in linhas)


def test_tabela_celulas_de_acerto_categoria_ausente_e_ferramentas():
    r1 = _resultado("sem_memoria", {"sessao_unica": [1, 3], "abstencao": [2, 4]}, [3, 7])
    r2 = _resultado("ativa", {"sessao_unica": [3, 3], "abstencao": [4, 4]}, [7, 7], {"buscar_memoria": 5, "gravar_memoria": 2})
    linha1, linha2 = (
        [c.strip() for c in l.strip("|").split("|")] for l in tabela([r1, r2]).splitlines()[2:]
    )
    assert linha1 == ["sem_memoria", "1/3", "-", "-", "-", "2/4", "3/7", "0", "4", "1200", "300", "0.0054", "-"]
    assert linha2[0] == "ativa" and linha2[6] == "7/7" and linha2[-1] == "5/2"


def _preparar_main(monkeypatch, tmp_path, fake, entrada=10):
    """Troca o LLM do runner por fakes (o primeiro é o da conversa, o segundo o do juiz) e a saída por tmp_path."""
    llm = LLM(fake([resposta_texto("r", entrada, 0)] * 400))
    juiz = LLM(fake([_veredito(True, entrada, 0)] * 100))
    fila = [llm, juiz]
    monkeypatch.setattr(rodar, "LLM", lambda: fila.pop(0))
    monkeypatch.setattr(rodar, "PASTA_RESULTADOS", tmp_path)
    return llm, juiz


def test_main_grava_arquivos(tmp_path, monkeypatch, fake, capsys):
    _preparar_main(monkeypatch, tmp_path, fake)
    rodar.main(["sem_memoria"])
    (pasta,) = tmp_path.iterdir()
    assert re.fullmatch(r"\d{8}-\d{6}", pasta.name)
    assert sorted(p.name for p in pasta.iterdir()) == ["sem_memoria.json", "tabela.md"]  # o argumento filtra
    bruto = (pasta / "sem_memoria.json").read_text(encoding="utf-8")
    resultado = json.loads(bruto)
    assert resultado["configuracao"] == "sem_memoria" and resultado["respostas"]
    assert "Paçoca" in bruto  # ensure_ascii=False: os acentos ficam legíveis no arquivo
    tabela_md = (pasta / "tabela.md").read_text(encoding="utf-8")
    assert "| sem_memoria |" in tabela_md
    saida = capsys.readouterr().out
    assert "Rodando sem_memoria..." in saida
    assert tabela_md.strip() in saida
    assert "Custo total estimado: US$ " in saida


def test_main_sem_argumento_roda_as_quatro_configuracoes(tmp_path, monkeypatch, fake, capsys):
    _preparar_main(monkeypatch, tmp_path, fake)
    rodar.main([])
    (pasta,) = tmp_path.iterdir()
    assert sorted(p.name for p in pasta.iterdir()) == [
        "ativa.json", "contexto_completo.json", "passiva.json", "sem_memoria.json", "tabela.md",
    ]  # fmt: skip
    assert (pasta / "tabela.md").read_text(encoding="utf-8").count("\n| ") == 4
    saida = capsys.readouterr().out
    for nome in CONFIGURACOES:
        assert f"Rodando {nome}..." in saida
    passiva = json.loads((pasta / "passiva.json").read_text(encoding="utf-8"))
    assert len(passiva["resumos_por_sessao"]) == len({t["sessao"] for t in passiva["turnos"]})  # uma por sessão


def test_main_custo_total_inclui_o_juiz_e_a_coluna_da_tabela_nao(tmp_path, monkeypatch, fake, capsys):
    # 500 mil tokens de entrada = US$ 1,00 por chamada, para a conta ser fácil de conferir.
    _preparar_main(monkeypatch, tmp_path, fake, entrada=500_000)
    rodar.main(["sem_memoria"])
    (pasta,) = tmp_path.iterdir()
    r = json.loads((pasta / "sem_memoria.json").read_text(encoding="utf-8"))
    assert r["custo_usd"] == pytest.approx(r["chamadas"] * 1.0)  # sem o juiz
    esperado = r["custo_usd"] + len(r["respostas"]) * 1.0  # com uma chamada do juiz por pergunta
    assert f"Custo total estimado: US$ {esperado:.2f}" in capsys.readouterr().out


def test_main_nome_desconhecido_sai_sem_criar_pasta(tmp_path, monkeypatch, fake):
    # Um nome digitado errado não pode deixar uma pasta vazia em resultados/ nem chamar o modelo.
    llm, juiz = _preparar_main(monkeypatch, tmp_path, fake)
    with pytest.raises(SystemExit, match="configuração desconhecida: passivo"):
        rodar.main(["passivo"])
    assert list(tmp_path.iterdir()) == []
    assert llm.contador.chamadas == 0 and juiz.contador.chamadas == 0
