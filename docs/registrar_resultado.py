"""Registra o relatório atual do harness no histórico versionado de resultados.

Uso (após `python -m candidate_starter.run_case`):
    python docs/registrar_resultado.py <versao> "<descricao>"

Anexa uma linha a `docs/resultados/historico.csv` e copia o relatório completo para
`docs/resultados/<versao>.json`. Só usa a biblioteca padrão.
"""
import csv
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RELATORIO = RAIZ / "reports" / "candidate_report.json"
PASTA = RAIZ / "docs" / "resultados"
HISTORICO = PASTA / "historico.csv"
COLUNAS = [
    "versao", "data", "commit", "descricao", "router_accuracy", "precision_at_k",
    "cost_savings_pct", "latency_savings_pct", "erros_agent_para_fast",
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()


def main(versao: str, descricao: str) -> None:
    # O hash só identifica o código que gerou os números se não houver alteração pendente.
    if git("status", "--porcelain", "candidate_starter", "common"):
        sys.exit("Erro: há alterações sem commit no código. Faça o commit e rode o case de novo.")
    if (PASTA / f"{versao}.json").exists():
        sys.exit(f"Erro: a versão '{versao}' já foi registrada.")

    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    p_at_k = relatorio["precision_at_k"]
    linha = {
        "versao": versao,
        "data": date.today().isoformat(),
        "commit": git("rev-parse", "--short", "HEAD"),
        "descricao": descricao,
        "router_accuracy": round(relatorio["router_accuracy"], 4),
        "precision_at_k": None if p_at_k is None else round(p_at_k, 4),
        "cost_savings_pct": round(relatorio["cost_savings_pct"], 1),
        "latency_savings_pct": round(relatorio["latency_savings_pct"], 1),
        "erros_agent_para_fast": relatorio["confusion_matrix"]["AGENT"]["FAST_PATH"],
    }

    PASTA.mkdir(parents=True, exist_ok=True)
    novo = not HISTORICO.exists()
    with HISTORICO.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS)
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)
    shutil.copy(RELATORIO, PASTA / f"{versao}.json")
    print(f"Registrado: {linha}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
