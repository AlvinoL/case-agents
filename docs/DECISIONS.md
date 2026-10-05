# Decisões técnicas (ADRs)

Registro curto de cada decisão: contexto, o que foi decidido, alternativas e por quê. Os números
vêm dos arquivos gerados em [`docs/resultados/`](resultados/); a evolução versão a versão está em
[`historico.csv`](resultados/historico.csv) (cada linha ligada ao commit que a gerou).

**Princípio geral:** o case pede o caminho mais barato e rápido. Preferimos o simples que
resolve, sem chave de API, 100% reprodutível (seeds fixas) e com dependências enxutas
(numpy, pandas, scikit-learn). Técnicas mais pesadas só entram se ganharem de forma mensurável.

| # | Decisão | Status |
|---|---|---|
| [D1](#d1--router-tf-idf-palavras--caracteres--regressão-logística) | Router: TF-IDF palavras + caracteres + Regressão Logística | Adotada |
| [D2](#d2--threshold-assimétrico-do-router) | Threshold assimétrico do router | Adiada: decisão de negócio |
| [D3](#d3--retriever-cascata-nome--descrição) | Retriever: cascata nome → descrição (TF-IDF) | Adotada (substitui o híbrido LSA + RRF) |
| [D4](#d4--variante-slm-e5-small) | Variante SLM: e5-small | Testada, não adotada |
| [D5](#d5--métricas-além-das-pedidas) | Métricas além das pedidas pelo case | Adotada (parcial, por escopo) |
| [D6](#d6--normalização-de-texto-sem-remover-stopwords) | Normalização de texto, sem remover stopwords | Adotada |
| [D7](#d7--latência-medida-de-ponta-a-ponta) | Latência medida de ponta a ponta | Adotada (correção do harness) |
| [D8](#d8--construção-incremental-melhoria-contínua) | Construção incremental (melhoria contínua), com retriever nulo | Adotada |
| [D9](#d9--protocolo-de-validação) | Protocolo de validação | Adotada |
| [D10](#d10--k--2-no-retriever) | k = 2 no retriever | Mantido |
| [D11](#d11--ideias-descartadas-com-evidência) | Ideias descartadas com evidência | Registro |

---

## D1 — Router: TF-IDF (palavras + caracteres) + Regressão Logística

**Contexto.** Classificar cada query em FAST_PATH ou AGENT com 53 frases de treino rotuladas,
barato e rápido.

**Decisão.** `TfidfVectorizer` de palavras (1–2 gramas) e de caracteres (2–5 gramas,
`char_wb`) lado a lado (`FeatureUnion`), seguidos de `LogisticRegression(C=10)`. A rota é a de
maior probabilidade, e `confidence` é essa probabilidade. Código: `candidate_starter/router.py`
e `candidate_starter/text.py` (compartilhado com o retriever).

**Alternativas.**
- Regras/palavras-chave: frágeis a paráfrase e com manutenção manual.
- Naive Bayes / SVM linear: desempenho parecido, mas o SVM não dá probabilidade nativa e o NB
  é mal calibrado, e o threshold (D2) precisa de probabilidade.
- Embeddings (e5) + LR: mais pesado, e o ganho precisaria ser provado.
- LLM classificando: contradiz o objetivo do router, que é evitar o LLM.
- Fine-tuning de BERT: 53 exemplos é pouco, e o modelo é pesado.

**Por quê.** O texto vira uma tabela de features, e a regressão logística atua como um
scorecard interpretável: os pesos mostram que possessivos e pedidos ("meu", "quero") puxam
para AGENT e perguntas sobre o banco ("vocês", "qual", "custa") puxam para FAST_PATH. Os
n-gramas de caracteres são uma aposta de robustez a erro de digitação.

**Evidência.** Validação cruzada K=5 estratificada × 10, só no treino
([`cv_router_d1.json`](resultados/cv_router_d1.json)):

| Variante | Acurácia | Erros graves (AGENT→FAST) por repetição |
|---|---|---|
| Sempre AGENT | 52,7% ± 2,2% | 0,0 |
| Só palavras | 86,0% ± 8,6% | 1,8 |
| Só caracteres | 88,1% ± 9,9% | 1,2 |
| **Palavras + caracteres** | **89,8% ± 8,6%** | **1,0** |

No eval (v02): 100% de acurácia. A estimativa honesta de generalização é a da validação
(~90%): o eval tem 30 queries, 3 delas aparecem idênticas no treino, e ele se mostrou mais fácil
que o treino.

**Limites.** O `C=10` é fixo (C=1 espreme as probabilidades perto de 0,5). Latência de ~5 ms
por query chamada uma a uma, que é overhead do scikit-learn: em lote, 30 queries levam ~3 ms.

## D2 — Threshold assimétrico do router

**Contexto.** Os erros têm custos diferentes. AGENT→FAST_PATH deixa o cliente sem atendimento
(grave). FAST_PATH→AGENT custa uma chamada de LLM a mais (US$ 0,03).

**Decisão.** Manter, por ora, a rota de maior probabilidade (τ = 0,5). O τ fica registrado como
**decisão de negócio**: a calibrar com a área, conforme o custo real de um cliente não atendido.

**Evidência.** Probabilidades fora da dobra (K=5 × 10, só treino;
[`cv_router_d2.json`](resultados/cv_router_d2.json)), C=10, por repetição de 53 frases:

| τ | Acurácia | Erros graves | FAST_PATH perdidos (de 25) |
|---|---|---|---|
| 0,5 | 89,8% | 1,0 | 4,4 |
| 0,6 | 83,8% | 0,6 | 8,0 |
| 0,7 | 75,1% | 0,2 | 13,0 |
| 0,8 | 67,4% | 0,0 | 17,3 |

Passar de τ=0,5 para 0,7 evita 0,8 erro grave e manda 8,6 FAST_PATH a mais ao LLM
(8,6 × US$ 0,03 ≈ US$ 0,26). **Vale se um cliente não atendido custar mais que ~US$ 0,32.**
É o mesmo raciocínio do ponto de corte de um scorecard de crédito.

**Achado.** `C` e τ não são independentes: o `C` muda a escala das probabilidades, mas a
fronteira do trade-off é quase a mesma. Para melhorar os dois lados, o caminho é mais dados,
não hiperparâmetro.

## D3 — Retriever: cascata nome → descrição

**Contexto.** Selecionar 2 tools entre 285 sem LLM e sem queries rotuladas com tool. O catálogo
tem **iscas**: as 12 tools esperadas no eval são as canônicas (nomes curtos, descrições
genéricas), e para cada uma há variações que parafraseiam as queries do eval, na mesma categoria.

**Decisão.** Cascata com o mesmo TF-IDF do router (`candidate_starter/retrieval.py`):
1. Cosseno da query com o **nome** da tool (`parcelar_fatura` → "parcelar fatura").
2. Se nenhum nome casa ≥ **0,53**, ranking pelo **documento completo** (nome + descrição +
   categoria).

O limiar 0,53 é o p90 da similaridade máxima nome × frase nas frases FAST_PATH do treino, ou
seja, o "ruído" de quem não pede tool, derivado sem rótulo de tool
(`candidate_starter/validacao_retrieval.py`).

**Por quê.** O nome é o rótulo curto da intenção (verbo + objeto). O cosseno normaliza query e
documento juntos, então o rótulo que casa por inteiro vence variações longas com palavras a
mais. A descrição cobre as queries que não usam as palavras do nome. Tool nova = linha nova no
índice, sem retreinar.

**Alternativa substituída.** A proposta inicial era um híbrido sparse (TF-IDF) + dense (LSA)
com fusão RRF. O LSA aprenderia só com o próprio catálogo, que é cheio de iscas, e o teste com
um dense semântico de verdade (e5, D4) não superou a cascata.

**Evidência (escada do harness, retriever é a única mudança; MRR@2 truncado no top-2):**

| Versão | Retriever | P@2 | Hit@1 | MRR@2 | Resolução | Custo/resolvido |
|---|---|---|---|---|---|---|
| v03 | TF-IDF no documento | 25% | 5% | 0,15 | 36,7% | US$ 0,0182 |
| **v04** | **cascata nome → descrição** | **40%** | **20%** | **0,30** | **46,7%** | **US$ 0,0143** |

Comparação com BM25, TF-IDF sublinear e e5 em
[`EXPERIMENTOS_RETRIEVAL.md`](EXPERIMENTOS_RETRIEVAL.md).

**Limites.** A ideia de usar o nome surgiu explorando os erros do eval; os números da v04 são
otimistas. Com 20 queries, 1 query = 5 pp.

## D4 — Variante SLM: e5-small

**Contexto.** Testar se um encoder semântico pequeno, open source, em CPU, supera o baseline
clássico, especialmente em paráfrases ("mudar o e-mail" ≈ "atualizar e-mail").

**Decisão.** `intfloat/multilingual-e5-small` (~118M parâmetros, **licença MIT**, revisão fixada
`614241f`), com prefixos `query:` e `passage:`, embeddings do catálogo em cache em disco e
fallback automático para TF-IDF sem torch. Dependência **opcional** (`requirements-slm.txt`).
Ativado com `run_case --slm`. **Não adotado como solução principal.**

**Por quê não.** Critério definido antes do teste: o SLM só entra se ganhar de forma mensurável.
Hit@k e MRR de [`experimentos_retrieval.json`](resultados/experimentos_retrieval.json);
latências de [`historico.csv`](resultados/historico.csv) (v04, v05, v05.1).

| Variante | Hit@1 | Hit@2 | MRR (ranking completo) | Latência do retriever (30 queries) |
|---|---|---|---|---|
| v04 (TF-IDF) | **20%** | 40% | 0,39 | ~63 ms |
| e5 puro | 15% | **50%** | **0,42** | — |
| cascata nome → e5 (v05) | 15% | 35% | 0,36 | ~179 ms |
| cascata nome → e5, limiar 0,80 (v05.1) | 15% | 45% | 0,41 | ~558 ms |

O e5 melhora o recall (Hit@2) e piora o top-1 (Hit@1), que é o que o sistema executa. As
diferenças são de 1–2 queries. Custo: ~1 GB de dependências e ~9× a latência do retriever.
O e5 também é atraído pelas iscas, que são semanticamente mais próximas da query.

**Colibri (`tardellirs/colibri-embed-ptbr`).** Não testado. Licença Gemma (termos de uso
próprios, não OSI) e base `google/embeddinggemma-300m` (~300M parâmetros). Ficou como opcional.

## D5 — Métricas além das pedidas

**Contexto.** O case pede acurácia + matriz de confusão, Precision@K e % de economia de custo
e latência. Essas métricas não capturam se o cliente foi atendido nem qual tool foi executada.

**Decisão.** Chaves **novas** no relatório, sem alterar as exigidas:
- `hit_at_1` e `mrr_at_k`: o mock executa a top-1, então o Precision@2 sozinho esconde a tool
  executada (um retriever com o gabarito sempre em 2º lugar tem P@2 = 100% e Hit@1 = 0%).
- `resolution_rate` e `cost_per_resolved_usd`: **custo por atendimento resolvido**. Premissa:
  o baseline (LLM caro com todas as tools) resolve 100%, o que favorece o baseline.
- `latency_percentiles_ms` (p50/p95) e `latency_breakdown_ms` (router, retrieval, agente).
- Testes unitários das métricas (`tests/test_harness_metrics.py`).

**Fora do escopo por decisão** (sugestões para próximas versões): retriever avaliado isolado
do router no harness, acurácia sem as 3 queries vazadas, IC 95% e recall de AGENT.

**Limite conhecido.** O custo por resolvido trata o cliente não atendido como custo zero.
Versão madura: custo esperado = custo de LLM + penalidade × não resolvidos, com a penalidade
definida pelo negócio.

## D6 — Normalização de texto, sem remover stopwords

**Decisão.** Minúsculas e remoção de acentos com `unicodedata` (biblioteca padrão), usadas pelo
router e pelo retriever. **Stopwords não são removidas.**

**Por quê.** "Cartão" e "cartao" precisam ser o mesmo termo, e clientes digitam sem acento.
"Meu" e "quero" são stopwords clássicas, mas aqui são o principal sinal de AGENT: removê-las
destruiria o sinal.

## D7 — Latência medida de ponta a ponta

**Contexto.** O `run_harness` original somava a latência que router e retriever declaravam.
A chamada ao LLM do agente (30–70 ms) não declara latência e ficava de fora, enquanto o
baseline era medido pelo relógio. Resultado: ~99,9% de economia de latência, um número
enganoso.

**Decisão.** Relógio de ponta a ponta por query, simétrico ao baseline, mais detalhamento por
componente. O ruído foi medido: um pipeline idêntico ao baseline (v00.3) deu −6,1% de
"economia" por sorteios diferentes de latência no mock, e por isso diferenças de latência
abaixo de ~10 pp entre versões não provam nada. O custo é determinístico no mock e é a métrica
confiável para comparar versões.

| Versão (router aleatório) | Economia de latência |
|---|---|
| v01.1 (soma declarada, com o bug) | 99,9% |
| v01.2 (ponta a ponta) | 60,6% |

## D8 — Construção incremental (melhoria contínua)

**Decisão.** Cada versão muda **uma** peça, registrada em `historico.csv`. O harness aceita
`retriever=None` (todas as tools ao LLM caro), para medir o router sem misturar com a economia
do top-k. O retriever aleatório ganhava de graça, no mock, a economia de prompt menor sem
entregar a qualidade correspondente.

| Versão | Router | Retriever | Acurácia | Econ. custo | Resolução | Erros graves |
|---|---|---|---|---|---|---|
| v00.3 | sempre AGENT | nulo | 66,7% | 0,0% | 100% | 0 |
| v01.4 | aleatório | nulo | 76,7%* | 30,0% | 90% | 3 |
| v02 | D1 | nulo | 100% | 33,3% | 100% | 0 |
| v03 | D1 | TF-IDF documento | 100% | 77,8% | 36,7% | 0 |
| **v04** | **D1** | **cascata (D3)** | **100%** | **77,8%** | **46,7%** | **0** |

\* A seed 42 do router aleatório foi o máximo de 1.000 seeds (média 49,9%).

**Leitura.** 33,3% é o teto de economia só com o router (10 FAST_PATH em 30 queries). A maior
parte da economia de custo vem do retriever reduzir o prompt. Um retriever ruim é pior que
nenhum: economiza, mas executa a tool errada (resolução de 100% para 36,7% na v03).

## D9 — Protocolo de validação

**Decisão.**
- **Router:** escolhas feitas com K-fold estratificado (K=5 × 10 repetições, seed 42) **só no
  treino**. O eval do case é o **teste**.
- **Retriever:** não há queries rotuladas com tool fora do eval, e optamos por **não** rotular
  dados novos, para não introduzir um viés nosso que não é o do case. O limiar da cascata foi
  derivado do treino sem rótulo de tool. As variantes foram exploradas olhando os erros do
  eval (análise de erro), e isso está **declarado**: os números do retriever são otimistas.

## D10 — k = 2 no retriever

**Decisão.** Manter k = 2, como pede o case.

**Por quê.** Com a v03, o gabarito estava nas 3 primeiras posições em 65% das queries contra
25% nas 2 primeiras. Aumentar k inflaria o Precision@K sem mudar a tool executada (o mock
executa a top-1) nem o custo (o mock cobra o mesmo por chamada, qualquer que seja o k). Fica
como **recomendação para produção**: com um LLM real escolhendo entre as top-k, k = 3 dá a ele
a chance de escolher a tool certa com mais frequência, a validar contra o custo em tokens.

## D11 — Ideias descartadas com evidência

Linhas marcadas com † vêm de explorações rápidas, sem script versionado (números indicativos).

| Ideia | Resultado | Por que não |
|---|---|---|
| Filtro por categoria da query (1º, 2º ou 3º estágio) † | **Teto zero**: um filtro perfeito corrigiria 0 dos 16 erros de top-1 da v04 | Isca e gabarito estão sempre na mesma categoria. Útil em escala (10 mil tools), não aqui |
| BM25 ([experimentos](EXPERIMENTOS_RETRIEVAL.md)) | Hit@1 15% na cascata (v04: 20%) | Soma a contribuição de cada termo, e as iscas contêm mais termos da query |
| TF-IDF sem IDF † | Hit@2 20% (v03: 25%) | Piorou |
| Canonicalização por agrupamento do catálogo † | Hit@2 entre 15% e 45% conforme o limiar | Instável, frágil para defender |
| Usar a posição da tool no catálogo como sinal | Não testado | Artefato do dataset: overfitting ao avaliador |
