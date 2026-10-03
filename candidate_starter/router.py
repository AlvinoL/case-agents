"""Pilar 1 — Router de Queries.

TF-IDF (palavras + n-gramas de caracteres) + Regressão Logística (decisão D1):
o texto vira uma tabela esparsa de features e a regressão logística atua como um
scorecard, devolvendo P(AGENT) e P(FAST_PATH). Barato, rápido e interpretável.
"""
import time
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from candidate_starter.text import build_vectorizer
from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self, C: float = 10.0) -> None:
        # C alto = regularização fraca: com 53 frases e milhares de features, o padrão (C=1)
        # deixa as probabilidades espremidas perto de 0,5. Valor provisório, revisado na D2.
        self._model = Pipeline([
            ("tfidf", build_vectorizer()),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ])
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Aprende o vocabulário/IDF e os pesos da regressão a partir das frases rotuladas."""
        self._model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, query: str) -> RouteResult:
        """Escolhe a rota mais provável; `confidence` é a probabilidade dessa rota."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()
        probas = self._model.predict_proba([query])[0]
        best = int(probas.argmax())
        route = str(self._model.classes_[best])
        latency_ms = (time.perf_counter() - start) * 1000
        return RouteResult(route=route, latency_ms=latency_ms, confidence=float(probas[best]))
