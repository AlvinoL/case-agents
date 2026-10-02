"""Pilar 3 — Evaluation Harness (Evals & Benchmarking).

A orquestração do pipeline (rodar o router, a seleção de tools e os mocks de LLM/tool) já
está feita em `run_harness`. O que falta implementar são as MÉTRICAS do relatório final:

  1. Acurácia do Router (+ matriz de confusão).
  2. Precision@K do Retriever de Tools (a tool certa estava no Top-K?).
  3. Economia de custo e latência do pipeline "inteligente" (router + seleção de tools)
     comparado ao baseline de mandar tudo para o LLM mais caro.

Preencha as funções marcadas com TODO. Respeite o formato de retorno pedido em cada
docstring, pois `run_harness` e `print_report` dependem dessas chaves.
"""
import time
from typing import Dict, List

from common.interfaces import BaseRouter, BaseToolRetriever
from common.mock_llm import (
    COST_RETRIEVAL_USD,
    COST_ROUTER_USD,
    fast_path_answer,
    mock_tool_execution,
    simulate_agent_llm_call,
    simulate_baseline_llm_call,
)
from common.schemas import Tool


def compute_router_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Acurácia e matriz de confusão do router.

    A matriz é indexada por [rótulo verdadeiro][rótulo previsto], ex.:
        {"FAST_PATH": {"FAST_PATH": 5, "AGENT": 1}, "AGENT": {"FAST_PATH": 0, "AGENT": 10}}
    """
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    correct = sum(matrix[label][label] for label in labels)
    accuracy = correct / len(y_true) if y_true else 0.0
    return {"accuracy": accuracy, "confusion_matrix": matrix}


def compute_precision_at_k(hits: List[int]) -> float:
    """Fração de queries em que a tool esperada estava no top-k.

    Com uma única tool correta por query, isso equivale a Hit Rate@K (Recall@K).
    """
    return sum(hits) / len(hits) if hits else 0.0


def compute_savings(
    smart_cost_usd: float,
    smart_latency_ms: float,
    baseline_cost_usd: float,
    baseline_latency_ms: float,
) -> Dict:
    """% de economia do pipeline inteligente em relação ao baseline: 100 * (1 - smart/baseline).

    Valor negativo indica que o pipeline inteligente saiu mais caro/lento que o baseline.
    """

    def pct(smart: float, baseline: float) -> float:
        return 100 * (1 - smart / baseline) if baseline else 0.0

    return {
        "cost_savings_pct": pct(smart_cost_usd, baseline_cost_usd),
        "latency_savings_pct": pct(smart_latency_ms, baseline_latency_ms),
    }


def run_harness(
    router: BaseRouter,
    retriever: BaseToolRetriever,
    tools: List[Tool],
    eval_dataset: List[dict],
    k: int = 2,
) -> dict:
    labels = ["FAST_PATH", "AGENT"]

    y_true: List[str] = []
    y_pred: List[str] = []
    precision_hits: List[int] = []

    smart_cost_total = 0.0
    smart_latency_ms_total = 0.0
    baseline_cost_total = 0.0
    baseline_latency_ms_total = 0.0

    rows = []

    for item in eval_dataset:
        query = item["query"]
        expected_route = item["expected_route"]
        expected_tool = item.get("expected_tool")

        route_result = router.predict(query)
        y_true.append(expected_route)
        y_pred.append(route_result.route)

        smart_cost = COST_ROUTER_USD
        smart_latency_ms = route_result.latency_ms

        row = {
            "query": query,
            "expected_route": expected_route,
            "predicted_route": route_result.route,
        }

        if route_result.route == "FAST_PATH":
            fast_path_answer(query)
        else:
            retrieval_result = retriever.search(query, k=k)
            smart_cost += COST_RETRIEVAL_USD
            smart_latency_ms += retrieval_result.latency_ms

            top_k_names = [m.name for m in retrieval_result.matches]
            if expected_tool:
                precision_hits.append(int(expected_tool in top_k_names))

            row["retrieved_tools"] = top_k_names
            row["expected_tool"] = expected_tool

            if top_k_names:
                mock_tool_execution(top_k_names[0], query)
                llm_result = simulate_agent_llm_call(query, top_k_names[0])
                smart_cost += llm_result["cost_usd"]

        smart_cost_total += smart_cost
        smart_latency_ms_total += smart_latency_ms

        baseline_start = time.perf_counter()
        baseline_result = simulate_baseline_llm_call(query)
        baseline_latency_ms_total += (time.perf_counter() - baseline_start) * 1000
        baseline_cost_total += baseline_result["cost_usd"]

        rows.append(row)

    router_metrics = compute_router_metrics(y_true, y_pred, labels)
    precision_at_k = compute_precision_at_k(precision_hits) if precision_hits else None
    savings = compute_savings(
        smart_cost_total, smart_latency_ms_total, baseline_cost_total, baseline_latency_ms_total
    )

    report = {
        "n_queries": len(eval_dataset),
        "router_accuracy": router_metrics["accuracy"],
        "confusion_matrix": router_metrics["confusion_matrix"],
        "precision_at_k": precision_at_k,
        "k": k,
        "smart_pipeline": {"total_cost_usd": smart_cost_total, "total_latency_ms": smart_latency_ms_total},
        "baseline_always_llm": {
            "total_cost_usd": baseline_cost_total,
            "total_latency_ms": baseline_latency_ms_total,
        },
        **savings,
        "rows": rows,
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("HARNESS DE AVALIAÇÃO - Router & Tool Retrieval")
    print("=" * 60)
    print(f"Queries avaliadas: {report['n_queries']}")
    print(f"Acurácia do Router: {report['router_accuracy']:.1%}")
    print(f"Matriz de confusão: {report['confusion_matrix']}")
    if report["precision_at_k"] is not None:
        print(f"Precision@{report['k']} do Retriever: {report['precision_at_k']:.1%}")
    print("-" * 60)
    print(f"Custo pipeline inteligente: ${report['smart_pipeline']['total_cost_usd']:.5f}")
    print(f"Custo baseline (tudo pro LLM): ${report['baseline_always_llm']['total_cost_usd']:.5f}")
    print(f"Economia de custo: {report.get('cost_savings_pct', 0):.1f}%")
    print(f"Latência pipeline inteligente: {report['smart_pipeline']['total_latency_ms']:.1f} ms")
    print(f"Latência baseline: {report['baseline_always_llm']['total_latency_ms']:.1f} ms")
    print(f"Economia de latência: {report.get('latency_savings_pct', 0):.1f}%")
    print("=" * 60)
