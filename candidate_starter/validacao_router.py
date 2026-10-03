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
SAIDA_D2 = SAIDA.with_name("cv_router_d2.json")
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
VALORES_C = [1.0, 10.0, 100.0]


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


def probas_fast_fora_da_dobra(modelo, X: np.ndarray, y: np.ndarray, cv: RepeatedStratifiedKFold) -> np.ndarray:
    """P(FAST_PATH) de cada exemplo, prevista por um modelo que não o viu; uma linha por repetição."""
    n_splits = cv.get_n_splits() // cv.n_repeats
    probas = np.zeros((cv.n_repeats, len(y)))
    for i, (idx_treino, idx_val) in enumerate(cv.split(X, y)):
        ajustado = clone(modelo).fit(X[idx_treino], y[idx_treino])
        col_fast = list(ajustado.classes_).index("FAST_PATH")
        probas[i // n_splits, idx_val] = ajustado.predict_proba(X[idx_val])[:, col_fast]
    return probas


def varrer_thresholds(probas: np.ndarray, y: np.ndarray) -> list:
    """Para cada τ: FAST_PATH só se P(FAST_PATH) >= τ. Médias por repetição (53 exemplos)."""
    agent, fast = y == "AGENT", y == "FAST_PATH"
    linhas = []
    for tau in THRESHOLDS:
        vai_fast = probas >= tau
        linhas.append({
            "tau": tau,
            "acuracia": float(np.mean(vai_fast == fast)),
            "erros_graves": float(np.mean(np.sum(vai_fast & agent, axis=1))),
            "fast_perdidos": float(np.mean(np.sum(~vai_fast & fast, axis=1))),
        })
    return linhas


def main_d2(X: np.ndarray, y: np.ndarray, cv: RepeatedStratifiedKFold) -> None:
    """D2: trade-off entre erro grave e economia perdida, para cada C e cada τ."""
    print(f"\nD2 — threshold τ: FAST_PATH só se P(FAST_PATH) >= τ "
          f"(médias por repetição; {int(np.sum(y == 'AGENT'))} AGENT e {int(np.sum(y == 'FAST_PATH'))} FAST)")
    resultados = {}
    for C in VALORES_C:
        modelo = make_pipeline(build_vectorizer(), LogisticRegression(C=C, max_iter=1000))
        probas = probas_fast_fora_da_dobra(modelo, X, y, cv)
        resultados[f"C={C:g}"] = varrer_thresholds(probas, y)
        print(f"  C={C:g}  (P(FAST) média dos FAST {probas[:, y == 'FAST_PATH'].mean():.2f}"
              f" | dos AGENT {probas[:, y == 'AGENT'].mean():.2f})")
        for r in resultados[f"C={C:g}"]:
            print(f"    τ={r['tau']:.1f}  acurácia {r['acuracia']:.1%} | erros graves {r['erros_graves']:.1f}"
                  f" | FAST perdidos {r['fast_perdidos']:.1f}")
    SAIDA_D2.write_text(json.dumps({"cv": "RepeatedStratifiedKFold(5, 10, seed=42)", "resultados": resultados},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {SAIDA_D2}")


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
    main_d2(X, y, cv)


if __name__ == "__main__":
    main()
