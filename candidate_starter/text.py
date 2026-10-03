"""Pré-processamento de texto compartilhado pelo router e pelo retriever."""
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def normalize(text: str) -> str:
    """Minúsculas e sem acentos: 'Cartão' e 'cartao' viram o mesmo termo."""
    decomposed = unicodedata.normalize("NFKD", text.lower())  # 'ã' -> 'a' + til combinante
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def build_vectorizer() -> FeatureUnion:
    """TF-IDF de palavras (1-2 gramas) + de caracteres (2-5 gramas), lado a lado.

    Palavras capturam termos inteiros ("estornar"); caracteres toleram erro de digitação
    ("bloquar" ainda compartilha "blo", "loq", "quar"...). Stopwords NÃO são removidas:
    "meu" e "quero" são o principal sinal de AGENT nos dados de treino.
    """
    return FeatureUnion([
        ("palavras", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2))),
        ("caracteres", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5))),
    ])
