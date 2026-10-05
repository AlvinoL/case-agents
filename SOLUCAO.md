# Solução — Router de Queries & Seleção de Tools

Resumo das escolhas técnicas e trade-offs. Detalhes de cada decisão em
[`docs/DECISIONS.md`](docs/DECISIONS.md); experimentos do retriever em
[`docs/EXPERIMENTOS_RETRIEVAL.md`](docs/EXPERIMENTOS_RETRIEVAL.md); evolução versão a versão em
[`docs/resultados/historico.csv`](docs/resultados/historico.csv); frentes de trabalho e tarefas em [`KANBAN.md`](KANBAN.md)
(versão visual: `docs/kanban.html`).

## Em uma frase

Duas peças clássicas, baratas e explicáveis, que compartilham a mesma representação de texto
(TF-IDF de palavras + n-gramas de caracteres): um **router** com regressão logística e um
**retriever em cascata** que compara a query primeiro com o **nome** das tools e, se o nome não
casar, com a descrição. Sem LLM, sem chave de API, 100% reprodutível.

## Resultado (v04, eval do case: 30 queries)

| Métrica | Pipeline | Baseline (tudo ao LLM caro) |
|---|---|---|
| Acurácia do router | 100% (0 erros AGENT→FAST_PATH) | — |
| Precision@2 do retriever | 40% | — |
| Hit@1 (tool executada certa) | 20% | — |
| Economia de custo | **77,8%** | — |
| Economia de latência | 59,6% | — |
| Custo por atendimento resolvido* | US$ 0,0143 | US$ 0,0300 |

\* Métrica proposta; premissa: o baseline resolve 100% das queries. A latência é simulada no mock e
varia ~±5 pp entre rodadas; custo e qualidade são determinísticos.

**Matriz de confusão do router (v04)**

| Verdadeiro \ Previsto | FAST_PATH | AGENT |
|---|---|---|
| **FAST_PATH** (10) | **10** | 0 |
| **AGENT** (20) | 0 (erro grave) | **20** |

O erro grave é AGENT → FAST_PATH: o cliente que precisava do agente recebe resposta local.

**Leitura honesta.** A estimativa de generalização do router é **~90%** (validação cruzada K=5
no treino), não 100%: o eval é pequeno, mais fácil que o treino e tem 3 queries idênticas ao
treino. O retriever é o ponto fraco: o catálogo tem tools quase duplicadas que parafraseiam as
queries do eval, e as alternativas testadas (BM25, e5-small, filtro por categoria) não superaram
a cascata no top-1.

## Como rodar

```bash
pip install -r requirements.txt
pytest candidate_starter/tests -v                  # 15 testes
python -m candidate_starter.run_case               # relatório em reports/candidate_report.json
python -m candidate_starter.validacao_router       # validação cruzada do router (só treino)
python -m candidate_starter.experimentos_retrieval # comparação de rankers do retriever
# Variante SLM (opcional): pip install -r requirements-slm.txt && python -m candidate_starter.run_case --slm
```

## Escolhas e trade-offs

| Escolha | Por quê | Trade-off aceito |
|---|---|---|
| Router TF-IDF + regressão logística | ~90% na validação, ~5 ms/query, pesos interpretáveis | Não entende sinônimo nunca visto |
| Não remover stopwords | "meu"/"quero" são o principal sinal de AGENT | Vocabulário maior |
| Rota de maior probabilidade (τ=0,5) | Ainda sem o custo real de um cliente não atendido | τ=0,7 cortaria ~80% dos erros graves ao custo de ~metade da economia: **decisão de negócio** |
| Retriever em cascata nome → descrição | Nome = rótulo da intenção; Hit@1 de 5% para 20% | Limiar calibrado com poucos dados |
| e5-small testado, não adotado | Pior no top-1, ~9× a latência, ~1 GB de dependências | Recall (Hit@2) 10 pp menor que o e5 puro |
| k = 2 | Pedido do case; o mock executa só a top-1 | Com k=3, a tool certa estaria na lista em 65% das queries (v03) |
| Métricas extras (Hit@1, MRR, p50/p95, custo por resolvido) | As do case não mostram a tool executada nem se o cliente foi atendido | Mais métricas para explicar |
| Latência de ponta a ponta (correção do harness) | A soma declarada omitia o LLM do agente (99,9% → 60,6%) | Ruído de ~±5 pp no mock |

## Limites

- 30 queries de eval (20 AGENT): 1 query = 3,3 pp no router e 5 pp no retriever.
- Custo e latência são mockados; só as proporções entre caminhos são defensáveis.
- As variantes do retriever foram exploradas olhando os erros do eval: números otimistas.
- O custo por resolvido trata cliente não atendido como custo zero (falta uma penalidade de negócio).

## Como levar para produção

**Arquitetura (AWS).** Router e retriever num container leve (scikit-learn, sem GPU) em ECS
Fargate ou Lambda, atrás de API Gateway. Artefatos de modelo e índice versionados no S3. O
catálogo de tools fica numa fonte única (ex.: DynamoDB), e o índice é reconstruído a cada
mudança por evento. O LLM do agente via Amazon Bedrock, mantendo os dados dentro da conta AWS.

**Observabilidade.** Traces por etapa (router, retrieval, LLM) com p50/p95; custo e **custo por
atendimento resolvido**; distribuição de rotas; taxa de baixa confiança do router; taxa de
fallback da cascata; erros AGENT→FAST_PATH via feedback (cliente repete a pergunta, abre
chamado).

**Drift e avaliação contínua.** Monitorar a distribuição de confiança e a taxa de termos fora do
vocabulário. Conjunto de avaliação maior e versionado, coletado de produção e rotulado por
humanos, priorizando as queries de baixa confiança (*active learning*). Novas versões em shadow
ou canary antes de 100% do tráfego, comparadas pelo mesmo harness.

**Guardrails.** τ como parâmetro de negócio (na dúvida, AGENT); tools com efeito colateral
(bloquear cartão, estornar) exigem confirmação do cliente; timeouts e fallback para atendimento
humano.

**Segurança e governança de dados.** Mascarar dados pessoais (CPF, número de cartão) antes de
logs e do LLM; criptografia e controle de acesso por menor privilégio; retenção alinhada à LGPD
e às políticas internas de risco de modelo (inventário, validação independente, documentação
como esta).

**Governança do catálogo (o maior ganho disponível).** Uma tool canônica por intenção, dono por
tool e descrições que diferenciam as tools. Um gate de CI pode usar o próprio retriever para
bloquear tools novas quase duplicadas das existentes.

**Evolução para agentes com LLM real.** Entregar as top-3 tools ao LLM e deixá-lo escolher via
function calling, medindo Hit@1 do LLM contra o custo em tokens; com 10 mil tools, roteamento
hierárquico (categoria → tool) para reduzir o espaço de busca.
