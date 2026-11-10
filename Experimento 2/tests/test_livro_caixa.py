from pytest import approx, raises
from replica.livro_caixa import LivroCaixa, LimiteDeGasto, custo_openai, custo_anthropic, custo_embedding

def test_custo_openai_sem_cache():
    assert custo_openai("gpt-4o-mini", 1_000_000, 0, 1_000_000) == approx(0.75)

def test_custo_openai_com_cache():
    assert custo_openai("gpt-4o-mini", 1_000_000, 1_000_000, 0) == approx(0.075)

def test_custo_anthropic_faixa_curta():
    assert custo_anthropic("claude-haiku-5-5", 50_000, 0, 0, 0) == approx(0.005)

def test_custo_anthropic_faixa_longa_com_leitura_de_cache():
    assert custo_anthropic("claude-haiku-5-5", 10_000, 0, 200_000, 1_000) == approx(0.0175)

def test_custo_anthropic_escrita_de_cache():
    assert custo_anthropic("claude-haiku-5-5", 1_000, 150_000, 0, 0) == approx(0.09425)

def test_custo_embedding():
    assert custo_embedding("text-embedding-3-small", 1_000_000) == approx(0.02)

def test_autorizar_recusa_o_que_passaria_do_teto(tmp_path):
    livro = LivroCaixa(tmp_path / "l.json", tetos={"openai": 1.0}, reserva={"openai": 0.0})
    livro.registrar("openai", 0.9)
    with raises(LimiteDeGasto):
        livro.autorizar("openai", 0.2)
    livro.autorizar("openai", 0.05)

def test_reserva_reduz_o_teto_ate_ser_liberada(tmp_path):
    tetos, reserva = {"openai": 20.0}, {"openai": 4.0}
    livro = LivroCaixa(tmp_path / "l.json", tetos=tetos, reserva=reserva)
    livro.registrar("openai", 15.5)
    with raises(LimiteDeGasto):
        livro.autorizar("openai", 1.0)
    LivroCaixa(tmp_path / "l.json", tetos=tetos, reserva=reserva, usar_reserva=True).autorizar("openai", 1.0)

def test_livro_persiste_entre_instancias(tmp_path):
    LivroCaixa(tmp_path / "l.json").registrar("anthropic", 1.25)
    assert LivroCaixa(tmp_path / "l.json").gasto("anthropic") == approx(1.25)


# --- gravação com novas tentativas e erro fatal ---

import os  # noqa: E402

from replica import livro_caixa as modulo_livro  # noqa: E402
from replica.livro_caixa import ErroDoLivroCaixa  # noqa: E402


def test_registrar_tenta_de_novo_quando_o_arquivo_esta_preso(tmp_path, monkeypatch):
    real, falhas = os.replace, []

    def replace_preso(origem, destino):
        if len(falhas) < 2:
            falhas.append(1)
            raise PermissionError("arquivo aberto em outro programa")
        real(origem, destino)

    monkeypatch.setattr(modulo_livro.os, "replace", replace_preso)
    esperas = []
    livro = LivroCaixa(tmp_path / "l.json", dormir=esperas.append)
    livro.registrar("openai", 0.25)
    assert livro.gasto("openai") == approx(0.25) and esperas == [0.2, 0.2]
    assert list(tmp_path.glob("*.tmp")) == []


def test_registrar_que_nao_consegue_gravar_levanta_erro_fatal(tmp_path, monkeypatch):
    def replace_preso(origem, destino):
        raise PermissionError("arquivo aberto em outro programa")

    monkeypatch.setattr(modulo_livro.os, "replace", replace_preso)
    esperas = []
    livro = LivroCaixa(tmp_path / "l.json", dormir=esperas.append)
    with raises(ErroDoLivroCaixa, match="0.2500"):
        livro.registrar("openai", 0.25)
    assert esperas == [0.2] * 4  # cinco tentativas
    assert list(tmp_path.glob("*.tmp")) == [] and not (tmp_path / "l.json").exists()
    assert not issubclass(ErroDoLivroCaixa, LimiteDeGasto)  # não é teto: para a rodada inteira
