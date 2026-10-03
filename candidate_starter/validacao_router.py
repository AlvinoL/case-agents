"""Validação cruzada do router (D1) usando SOMENTE os dados de treino.

O eval do case é o conjunto de teste: não é usado aqui para nada. Este script estima a
generalização do router e compara variantes de features com K-fold estratificado (K=5),
repetido 10 vezes com seeds fixas.

Uso: python -m candidate_starter.validacao_router
"""
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

from candidate_starter.text import build_vectorizer, normalize
from common.data_loader import load_router_training_data

SAIDA = Path(__file__).resolve().parent.parent / "docs" / "resultados" / "cv_router_d1.json"


def variantes() -> dict:
    lr = LogisticRegression(C=10.0, max_iter=1000)
    return {
        "sempre AGENT (classe majoritária)": DummyClassifier(strategy="constant", constant="AGENT"),
        "só palavras": make_pipeline(TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2)), clone(lr)),
        "só caracteres": make_pipeline(
            TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5)), clone(lr)
        ),
        "D1: palavras + caracteres": make_pipeline(build_vectorizer(), clone(lr)),
    }


def avaliar(modelo, X: np.ndarray, y: np.ndarray, cv: RepeatedStratifiedKFold) -> dict:
    """Acurácia e erros graves (AGENT -> FAST_PATH) em cada fold."""
    acuracias, graves = [], 0
    for idx_treino, idx_val in cv.split(X, y):
        pred = clone(modelo).fit(X[idx_treino], y[idx_treino]).predict(X[idx_val])
        acuracias.append(float(np.mean(pred == y[idx_val])))
        graves += int(np.sum((y[idx_val] == "AGENT") & (pred == "FAST_PATH")))
    return {
        "acuracia_media": float(np.mean(acuracias)),
        "acuracia_desvio": float(np.std(acuracias)),
        "erros_graves_por_repeticao": graves / cv.n_repeats,  # cada repetição cobre os 53 exemplos
    }


def main() -> None:
    textos, rotulos = load_router_training_data()
    X, y = np.array(textos, dtype=object), np.array(rotulos)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    resultados = {nome: avaliar(modelo, X, y, cv) for nome, modelo in variantes().items()}
    print(f"K-fold estratificado K=5 x 10 repetições | {len(y)} exemplos de treino (eval NÃO usado)")
    for nome, r in resultados.items():
        print(f"  {nome:<34} acurácia {r['acuracia_media']:.1%} ± {r['acuracia_desvio']:.1%}"
              f" | erros graves por repetição {r['erros_graves_por_repeticao']:.1f}")

    SAIDA.write_text(json.dumps({"cv": "RepeatedStratifiedKFold(5, 10, seed=42)", "resultados": resultados},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {SAIDA}")


if __name__ == "__main__":
    main()
