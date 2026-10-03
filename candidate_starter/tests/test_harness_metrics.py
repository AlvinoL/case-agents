"""Testes unitários das métricas do harness: se a régua estiver errada, toda comparação também estará."""
import pytest

from candidate_starter.harness import (
    compute_cost_per_resolved,
    compute_latency_percentiles,
    compute_mrr,
    compute_precision_at_k,
    compute_router_metrics,
    compute_savings,
)

LABELS = ["FAST_PATH", "AGENT"]


def test_router_metrics_matriz_indexada_por_verdadeiro_e_previsto():
    y_true = ["AGENT", "AGENT", "FAST_PATH", "FAST_PATH"]
    y_pred = ["AGENT", "FAST_PATH", "FAST_PATH", "AGENT"]
    metrics = compute_router_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == 0.5
    # [verdadeiro][previsto]: o AGENT que foi para FAST_PATH é o erro grave.
    assert metrics["confusion_matrix"]["AGENT"] == {"FAST_PATH": 1, "AGENT": 1}
    assert metrics["confusion_matrix"]["FAST_PATH"] == {"FAST_PATH": 1, "AGENT": 1}


def test_precision_at_k_e_fracao_de_acertos():
    assert compute_precision_at_k([1, 0, 1, 1]) == 0.75
    assert compute_precision_at_k([]) == 0.0


def test_mrr_premia_posicao_da_tool_certa():
    # 1ª posição = 1, 2ª = 0,5, fora do top-k = 0.
    assert compute_mrr([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert compute_mrr([]) == 0.0


def test_percentis_de_latencia():
    result = compute_latency_percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
    assert result["p50"] == pytest.approx(30.0)
    assert result["p95"] == pytest.approx(48.0)  # interpolação linear entre 40 e 50
    assert compute_latency_percentiles([]) == {"p50": None, "p95": None}


def test_savings_positiva_negativa_e_baseline_zero():
    assert compute_savings(25.0, 40.0, 100.0, 100.0) == {
        "cost_savings_pct": 75.0,
        "latency_savings_pct": 60.0,
    }
    # Pipeline mais caro que o baseline gera economia negativa (não é truncada em zero).
    assert compute_savings(120.0, 100.0, 100.0, 100.0)["cost_savings_pct"] == pytest.approx(-20.0)
    assert compute_savings(1.0, 1.0, 0.0, 0.0) == {"cost_savings_pct": 0.0, "latency_savings_pct": 0.0}


def test_cost_per_resolved_e_sem_resolucao():
    result = compute_cost_per_resolved(0.30, 20, 0.90, 30)
    assert result["resolution_rate"] == pytest.approx(20 / 30)
    assert result["cost_per_resolved_usd"]["smart"] == pytest.approx(0.015)
    assert result["cost_per_resolved_usd"]["baseline"] == pytest.approx(0.03)

    # Nada resolvido: custo por resolvido é indefinido (None), não divisão por zero.
    assert compute_cost_per_resolved(0.30, 0, 0.90, 30)["cost_per_resolved_usd"]["smart"] is None


def test_sem_retriever_sempre_agent_e_identico_ao_baseline():
    """Sanidade da régua: sem router útil e sem retriever, o pipeline É o baseline."""
    from candidate_starter.baselines import AlwaysAgentRouter
    from candidate_starter.harness import run_harness

    eval_dataset = [
        {"query": "Quero saber meu saldo", "expected_route": "AGENT", "expected_tool": "consultar_saldo"},
        {"query": "Bom dia", "expected_route": "FAST_PATH", "expected_tool": None},
    ]
    report = run_harness(AlwaysAgentRouter(), None, [], eval_dataset)

    assert report["cost_savings_pct"] == pytest.approx(0.0, abs=0.01)  # só o custo ínfimo do router
    assert report["precision_at_k"] is None and report["hit_at_1"] is None
    assert report["resolution_rate"] == 1.0  # mesma premissa do baseline
