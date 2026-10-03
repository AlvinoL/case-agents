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
from typing import Dict, List, Optional

import numpy as np

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


def compute_mrr(reciprocal_ranks: List[float]) -> float:
    """Mean Reciprocal Rank: média de 1/posição da tool esperada (0 se fora do top-k)."""
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_latency_percentiles(latencies_ms: List[float]) -> Dict:
    """p50 (tempo típico) e p95 (cauda lenta) por query, com interpolação linear do numpy."""
    if not latencies_ms:
        return {"p50": None, "p95": None}
    p50, p95 = np.percentile(latencies_ms, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


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


def compute_cost_per_resolved(
    smart_cost_usd: float, n_resolved: int, baseline_cost_usd: float, n_queries: int
) -> Dict:
    """Custo por atendimento resolvido corretamente (métrica proposta, além do case).

    Premissa: o baseline (LLM caro com todas as tools) resolve 100% das queries. É otimista
    para o baseline e, portanto, conservadora para o pipeline inteligente.
    """
    return {
        "resolution_rate": n_resolved / n_queries if n_queries else 0.0,
        "cost_per_resolved_usd": {
            "smart": smart_cost_usd / n_resolved if n_resolved else None,
            "baseline": baseline_cost_usd / n_queries if n_queries else None,
        },
    }


def run_harness(
    router: BaseRouter,
    retriever: Optional[BaseToolRetriever],
    tools: List[Tool],
    eval_dataset: List[dict],
    k: int = 2,
) -> dict:
    """Roda o pipeline inteligente e o baseline em cada query e monta o relatório.

    `retriever=None` significa sem seleção de tools: queries AGENT vão ao LLM caro com o
    catálogo inteiro, e as métricas de retriever ficam None.
    """
    labels = ["FAST_PATH", "AGENT"]

    y_true: List[str] = []
    y_pred: List[str] = []
    precision_hits: List[int] = []
    hits_at_1: List[int] = []
    reciprocal_ranks: List[float] = []

    smart_cost_total = 0.0
    smart_latency_ms_total = 0.0
    baseline_cost_total = 0.0
    baseline_latency_ms_total = 0.0
    n_resolved = 0
    breakdown_total = {"router": 0.0, "retrieval": 0.0, "agent": 0.0}
    latencies_ms: Dict[str, List[float]] = {"smart": [], "baseline": [], "router": []}

    rows = []

    for item in eval_dataset:
        query = item["query"]
        expected_route = item["expected_route"]
        expected_tool = item.get("expected_tool")

        # Relógio de ponta a ponta, simétrico ao do baseline. Antes, a latência era a soma do que
        # cada componente declarava, e a chamada ao LLM do agente (que não declara) ficava de fora.
        smart_start = time.perf_counter()
        route_result = router.predict(query)
        y_true.append(expected_route)
        y_pred.append(route_result.route)

        smart_cost = COST_ROUTER_USD
        breakdown = {"router": route_result.latency_ms, "retrieval": 0.0, "agent": 0.0}

        row = {
            "query": query,
            "expected_route": expected_route,
            "predicted_route": route_result.route,
        }

        if route_result.route == "FAST_PATH":
            fast_path_answer(query)
            resolved = expected_route == "FAST_PATH"
        elif retriever is None:
            # Sem seleção de tools: o agente chama o LLM caro com o catálogo inteiro, igual ao
            # baseline. Isola o efeito do router (a economia do top-k ainda não existe).
            agent_start = time.perf_counter()
            llm_result = simulate_baseline_llm_call(query)
            breakdown["agent"] = (time.perf_counter() - agent_start) * 1000
            smart_cost += llm_result["cost_usd"]
            resolved = True  # mesma premissa do baseline: com todas as tools, o LLM resolve
        else:
            retrieval_result = retriever.search(query, k=k)
            smart_cost += COST_RETRIEVAL_USD
            breakdown["retrieval"] = retrieval_result.latency_ms

            top_k_names = [m.name for m in retrieval_result.matches]
            if expected_tool:
                precision_hits.append(int(expected_tool in top_k_names))
                # Mesmo conjunto de queries do Precision@K, olhando a posição da tool certa.
                hits_at_1.append(int(top_k_names[:1] == [expected_tool]))
                rank = top_k_names.index(expected_tool) + 1 if expected_tool in top_k_names else None
                reciprocal_ranks.append(1 / rank if rank else 0.0)

            row["retrieved_tools"] = top_k_names
            row["expected_tool"] = expected_tool

            if top_k_names:
                agent_start = time.perf_counter()
                mock_tool_execution(top_k_names[0], query)
                llm_result = simulate_agent_llm_call(query, top_k_names[0])
                breakdown["agent"] = (time.perf_counter() - agent_start) * 1000
                smart_cost += llm_result["cost_usd"]

            # Resolvido = a tool executada (top-1) é a esperada; FAST_PATH esperado nunca resolve aqui.
            resolved = bool(top_k_names) and top_k_names[0] == expected_tool

        smart_latency_ms = (time.perf_counter() - smart_start) * 1000
        row["latency_ms"] = smart_latency_ms
        row["latency_breakdown_ms"] = breakdown
        for component, ms in breakdown.items():
            breakdown_total[component] += ms
        latencies_ms["smart"].append(smart_latency_ms)
        latencies_ms["router"].append(route_result.latency_ms)

        row["resolved"] = resolved
        n_resolved += resolved
        smart_cost_total += smart_cost
        smart_latency_ms_total += smart_latency_ms

        baseline_start = time.perf_counter()
        baseline_result = simulate_baseline_llm_call(query)
        baseline_latency_ms = (time.perf_counter() - baseline_start) * 1000
        baseline_latency_ms_total += baseline_latency_ms
        latencies_ms["baseline"].append(baseline_latency_ms)
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
        "hit_at_1": compute_precision_at_k(hits_at_1) if hits_at_1 else None,
        "mrr_at_k": compute_mrr(reciprocal_ranks) if reciprocal_ranks else None,
        "k": k,
        "smart_pipeline": {
            "total_cost_usd": smart_cost_total,
            "total_latency_ms": smart_latency_ms_total,
            "latency_breakdown_ms": breakdown_total,
        },
        "baseline_always_llm": {
            "total_cost_usd": baseline_cost_total,
            "total_latency_ms": baseline_latency_ms_total,
        },
        **savings,
        "latency_percentiles_ms": {name: compute_latency_percentiles(v) for name, v in latencies_ms.items()},
        **compute_cost_per_resolved(smart_cost_total, n_resolved, baseline_cost_total, len(eval_dataset)),
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
        print(f"  Hit@1 (tool executada certa): {report['hit_at_1']:.1%} | MRR@{report['k']}: {report['mrr_at_k']:.3f}")
    print("-" * 60)
    print(f"Custo pipeline inteligente: ${report['smart_pipeline']['total_cost_usd']:.5f}")
    print(f"Custo baseline (tudo pro LLM): ${report['baseline_always_llm']['total_cost_usd']:.5f}")
    print(f"Economia de custo: {report.get('cost_savings_pct', 0):.1f}%")
    print(f"Latência pipeline inteligente: {report['smart_pipeline']['total_latency_ms']:.1f} ms")
    breakdown = report["smart_pipeline"]["latency_breakdown_ms"]
    print("  detalhamento: " + ", ".join(f"{c} {ms:.1f} ms" for c, ms in breakdown.items()))
    print(f"Latência baseline: {report['baseline_always_llm']['total_latency_ms']:.1f} ms")
    print(f"Economia de latência: {report.get('latency_savings_pct', 0):.1f}%")
    for name, p in report["latency_percentiles_ms"].items():
        print(f"  {name:<8} p50 {p['p50']:7.2f} ms | p95 {p['p95']:7.2f} ms (por query)")
    print("-" * 60)
    print(f"Taxa de resolução correta: {report['resolution_rate']:.1%}")
    cpr = report["cost_per_resolved_usd"]
    smart_cpr = f"${cpr['smart']:.5f}" if cpr["smart"] is not None else "n/a (nada resolvido)"
    print(f"Custo por atendimento resolvido: {smart_cpr} vs baseline ${cpr['baseline']:.5f}")
    print("=" * 60)
