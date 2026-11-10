"""Acerto por fonte: MH-QA (substring), MCC (exact match estrito) e OAB (letra)."""
import pytest

from replica.avaliacao import avaliar


def test_avaliar_por_fonte():
    assert avaliar("oab_h1_02", "Gabarito: B", ["B"])[0]
    assert not avaliar("icl_banking77_5900shot_balance", "label: 43", ["43"])[0]
    assert avaliar("icl_banking77_5900shot_balance", "label: 43", ["43"])[1]["exact_match_flexivel"]
    assert avaliar("ruler_qa2_421K", "It was yes.", ["yes"])[0]


def test_metricas_da_oab_marcam_letra_invalida():
    assert avaliar("oab_h2", "Gabarito: C", ["B"]) == (False, {"letra": "C", "letra_invalida": False})
    assert avaliar("oab_sem_memoria", "Não sei.", ["B"]) == (False, {"letra": None, "letra_invalida": True})


def test_mcc_estrito_e_flexivel():
    acerto, metricas = avaliar("icl_trec_fine_6400shot_balance", "43", ["43"])
    assert acerto and metricas["exact_match"] and metricas["exact_match_flexivel"]
    acerto, metricas = avaliar("icl_trec_fine_6400shot_balance", "label: -7", ["43"])
    assert not acerto and not metricas["exact_match_flexivel"]


def test_fonte_desconhecida_levanta():
    with pytest.raises(ValueError):
        avaliar("longmemeval_s", "x", ["x"])
