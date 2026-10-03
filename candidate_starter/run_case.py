import json
import sys
from pathlib import Path
from typing import Optional

from common.data_loader import load_eval_dataset, load_router_training_data, load_tools
from common.interfaces import BaseRouter
from candidate_starter.harness import print_report, run_harness
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter


def main(router: Optional[BaseRouter] = None, with_retriever: bool = True) -> None:
    """Roda o harness; `router` permite avaliar alternativas (ex.: baselines triviais).

    `with_retriever=False` manda as queries AGENT ao LLM caro com todas as tools (isola o router).
    """
    tools = load_tools()
    train_texts, train_labels = load_router_training_data()
    eval_dataset = load_eval_dataset()

    router = (router or QueryRouter()).fit(train_texts, train_labels)
    retriever = ToolRetriever().fit(tools) if with_retriever else None

    report = run_harness(router, retriever, tools, eval_dataset)
    print_report(report)

    output_path = Path(__file__).resolve().parent.parent / "reports" / "candidate_report.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatório salvo em: {output_path}")


if __name__ == "__main__":
    main(with_retriever="--sem-retriever" not in sys.argv)
