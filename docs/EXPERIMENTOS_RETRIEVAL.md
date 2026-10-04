# Experimentos de ranking do retriever (Pilar 2)

Registro do que foi testado para escolher o ranking de tools, inclusive o que **não** foi
adotado. Todos os números vêm de [`resultados/experimentos_retrieval.json`](resultados/experimentos_retrieval.json),
gerado por:

```bash
python -m candidate_starter.experimentos_retrieval   # e5 só entra com requirements-slm.txt instalado
```

## Como medimos

- **Conjunto:** as 20 queries AGENT do eval, com o retriever **isolado do router** (todas
  chegam ao retriever).
- **Métricas:** posição da tool esperada no ranking completo.
  - **Hit@1**: a tool executada (top-1) é a esperada. É o que resolve o cliente no harness.
  - **Hit@2**: a esperada está nas 2 primeiras. Equivale ao Precision@K do case com k=2.
  - **Hit@3**: a esperada está nas 3 primeiras.
  - **MRR**: média de 1/posição.
- **Ruído:** 1 query = 5 pp. Diferenças de 1–2 queries não provam superioridade.

## Por que o problema é difícil: o catálogo tem iscas

As 12 tools esperadas no eval são as **canônicas** (nomes curtos, descrições genéricas;
posições 0–66 do catálogo). Para cada uma existem variações que **parafraseiam as queries do
eval** (ex.: `parcelar_fatura_numero_vezes_escolhido`, "…como em 3 vezes";
`reclamar_cobranca_errada_cartao_dinheiro_volta`). Elas são lexical **e** semanticamente mais
parecidas com a query do que o gabarito, e todas estão na **mesma categoria** do gabarito.

## Resultados

| Variante | Hit@1 | Hit@2 | Hit@3 | MRR |
|---|---|---|---|---|
| documento, TF-IDF palavras+caracteres (v03) | 5% | 25% | 65% | 0,32 |
| documento, TF-IDF sublinear | 5% | 30% | 60% | 0,33 |
| documento, BM25 palavras (k1=1,2, b=0,75) | 0% | 15% | 40% | 0,25 |
| documento, BM25 palavras (k1=1,2, b=1,0) | 0% | 20% | 40% | 0,26 |
| documento, BM25 caracteres 3–5 | 10% | 15% | 50% | 0,31 |
| nome, TF-IDF palavras+caracteres | 20% | 40% | 50% | 0,39 |
| nome, BM25 palavras | 20% | 35% | 45% | 0,37 |
| **cascata nome → TF-IDF documento (v04, solução)** | **20%** | 40% | 45% | 0,39 |
| cascata nome → BM25 documento | 15% | 35% | 45% | 0,36 |
| documento, e5-small | 15% | **50%** | **70%** | **0,42** |
| cascata nome → e5-small, limiar 0,53 (v05) | 15% | 35% | 45% | 0,36 |
| cascata nome → e5-small, limiar 0,80 (v05.1) | 15% | 45% | 65% | 0,41 |

## O que aprendemos

**1. O nome da tool é o melhor sinal lexical.** O nome é o rótulo curto da intenção (verbo +
objeto). O cosseno normaliza query e documento juntos, então o rótulo que casa por inteiro
(`parcelar fatura`) vence variações longas com palavras a mais. Isso levou o Hit@1 de 5% para
20%.

**2. BM25 não ajudou, apesar de penalizar documentos longos.** O BM25 soma a contribuição de
cada termo da query encontrado no documento, e as iscas contêm mais termos da query. A
penalidade por tamanho (`b`) não compensa. O ganho do TF-IDF aqui vem da normalização do
cosseno, que o BM25 não tem da mesma forma. Em um catálogo sem duplicatas, o BM25 seria
um candidato natural.

**3. O e5-small melhora o recall, mas não o top-1.** O e5 puro coloca o gabarito nas 2
primeiras posições em 50% das queries (melhor Hit@2 e MRR da tabela). Mas, no top-1, também é
atraído pelas iscas, que são semanticamente mais próximas da query (ex.: "mudar o e-mail
**vinculado**" → `consultar_email_vinculado_conta` 0,902 vs `atualizar_email` 0,890). Custo:
~1 GB de dependências e ~9× a latência do retriever. Ver o trade-off abaixo.

**4. Filtro de categoria: teto zero.** Um filtro de categoria perfeito corrigiria 0 dos 16
erros de top-1 da v04, porque isca e gabarito estão sempre na mesma categoria. Não foi
implementado.

## O trade-off em aberto

| Critério | Melhor variante |
|---|---|
| Hit@1: a tool **executada** é a certa (resolução no harness) | v04: 20% contra 15% do e5 |
| Hit@2: a métrica **oficial** do case (Precision@K, k=2) | e5 puro: 50% contra 40% da v04 |
| Custo, latência e dependências | v04 (TF-IDF, sem torch) |

As diferenças são de 1 a 2 queries, dentro do ruído. A escolha depende de qual critério o
negócio prioriza.

## Limites

- 20 queries: todas as diferenças entre as variantes mais fortes estão dentro do ruído.
- As variantes foram exploradas olhando o eval. Os números são **otimistas** para a variante
  escolhida e devem ser recalibrados com dados de produção.
- O limite parece estar no **catálogo**, não no algoritmo: tools quase duplicadas exigem
  governança (deduplicação, uma tool canônica por intenção, descrições discriminativas).
