"""Regenera o KANBAN.md (raiz) a partir do JSON embutido em docs/kanban.html.

O HTML é a fonte única dos dados; o Markdown existe para renderizar direto no GitHub
(checklists por frente e diagramas Mermaid). Só usa a biblioteca padrão.

Uso: python docs/sync_kanban.py
"""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
HTML = RAIZ / "docs" / "kanban.html"
SAIDA = RAIZ / "KANBAN.md"


def carregar_dados() -> dict:
    html = HTML.read_text(encoding="utf-8")
    bloco = re.search(r'<script id="kanban-data" type="application/json">(.*?)</script>', html, re.S)
    if not bloco:
        raise SystemExit("Bloco kanban-data não encontrado em docs/kanban.html.")
    return json.loads(bloco.group(1))


def grafo_dependencias(dados: dict) -> str:
    """Mesmo grafo do HTML: tarefas entregues agrupadas por frente, com setas de dependência."""
    feitas = [t for t in dados["tarefas"] if t["status"] == "feito"]
    linhas = ["flowchart TB"]
    for frente in dados["frentes"]:
        tarefas = [t for t in feitas if t["frente"] == frente["id"]]
        if not tarefas:
            continue
        linhas.append(f'  subgraph {frente["id"]}["{frente["nome"]}"]')
        linhas += [f'    {t["id"]}["{t["id"]} · {t["titulo"].replace(chr(34), chr(39))}"]' for t in tarefas]
        linhas.append("  end")
    linhas += [f"  {d} --> {t['id']}" for t in feitas for d in t["depende"]]
    return "\n".join(linhas)


def cartao(t: dict) -> str:
    marca = {"feito": "- [x]", "backlog": "- [ ]", "descartado": "- ✖"}[t["status"]]
    extras = [t["evidencia"]] if t["evidencia"] != "—" else []
    extras += [f"`{t['ref']}`"] if t["ref"] != "—" else []
    sufixo = f" — {' · '.join(extras)}" if extras else ""
    return f"{marca} **{t['id']}** {t['titulo']}{sufixo}"


def gerar(dados: dict) -> str:
    tarefas = dados["tarefas"]
    total = {s: sum(t["status"] == s for t in tarefas) for s in ("feito", "descartado", "backlog")}
    md = [
        f"# Kanban — {dados['titulo']}",
        "",
        "> Gerado por `python docs/sync_kanban.py` a partir de [`docs/kanban.html`](docs/kanban.html) "
        "(versão visual; abra no navegador). Não edite à mão.",
        "",
        f"**{total['feito']} entregues · {total['descartado']} testadas e descartadas · "
        f"{total['backlog']} no backlog · {len(dados['frentes'])} frentes de trabalho**",
        "",
        "## Arquitetura final",
        "",
        "```mermaid",
        dados["arquitetura"],
        "```",
        "",
        "## Melhoria contínua: uma mudança por versão",
        "",
        "| Versão | O que mudou | Acurácia router | Hit@1 retriever | Economia de custo | Resolução | Erros graves |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in dados["versoes"]:
        nome = f"**{v['v']}** (solução)" if v.get("final") else v["v"]
        md.append(f"| {nome} | {v['o_que']} | {v['acuracia']} | {v['hit1']} | {v['economia']} "
                  f"| {v['resolucao']} | {v['erros_graves']} |")
    # "\*" evita que o asterisco inicial vire item de lista no Markdown.
    md += ["", dados["nota_versoes"].replace("* ", "\\* ", 1), "", "## Frentes de trabalho", ""]

    legenda = {"feito": "Feito", "descartado": "Testado e descartado", "backlog": "Backlog (próximos passos)"}
    for frente in dados["frentes"]:
        da_frente = [t for t in tarefas if t["frente"] == frente["id"]]
        feitas = sum(t["status"] == "feito" for t in da_frente)
        planejadas = sum(t["status"] in ("feito", "backlog") for t in da_frente)
        md += [f"### {frente['id']} · {frente['nome']} ({feitas}/{planejadas})", "",
               f"*{frente['perfil']}* — {frente['objetivo']}", ""]
        for status, titulo in legenda.items():
            grupo = [t for t in da_frente if t["status"] == status]
            if grupo:
                md += [f"**{titulo}**", ""] + [cartao(t) for t in grupo] + [""]

    md += ["## Dependências entre tarefas entregues", "", "```mermaid", grafo_dependencias(dados), "```", ""]
    return "\n".join(md)


def main() -> None:
    SAIDA.write_text(gerar(carregar_dados()), encoding="utf-8")
    print(f"KANBAN.md regenerado: {SAIDA}")


if __name__ == "__main__":
    main()
