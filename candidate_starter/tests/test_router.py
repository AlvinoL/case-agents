"""Testes do pré-processamento e do router (D1)."""
from candidate_starter.router import QueryRouter
from candidate_starter.text import normalize
from common.data_loader import load_router_training_data


def test_normalize_remove_acentos_e_maiusculas():
    assert normalize("Cartão BLOQUEADO, está?") == "cartao bloqueado, esta?"


def test_router_treinado_separa_saudacao_de_pedido():
    router = QueryRouter().fit(*load_router_training_data())

    assert router.predict("Boa tarde!").route == "FAST_PATH"
    result = router.predict("Quero bloquear meu cartão")
    assert result.route == "AGENT"
    assert 0.5 <= result.confidence <= 1.0
