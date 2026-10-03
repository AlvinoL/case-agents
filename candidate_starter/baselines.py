"""Baselines triviais para comparação no histórico de resultados.

Uso: python -m candidate_starter.baselines [--sem-retriever]
"""
import time
from typing import List

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class AlwaysAgentRouter(BaseRouter):
    """Manda toda query para o AGENT: nunca deixa cliente sem atendimento, mas não economiza."""

    def fit(self, texts: List[str], labels: List[str]) -> "AlwaysAgentRouter":
        return self

    def predict(self, query: str) -> RouteResult:
        start = time.perf_counter()
        return RouteResult(route="AGENT", latency_ms=(time.perf_counter() - start) * 1000, confidence=1.0)


if __name__ == "__main__":
    import sys

    from candidate_starter.run_case import main

    main(router=AlwaysAgentRouter(), with_retriever="--sem-retriever" not in sys.argv)
