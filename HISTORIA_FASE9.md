# História da Fase 9 — de uma conversa de WhatsApp a três scripts com IC bootstrap

> Como surgiu a ideia, como foi executada, o que deu — para guardar o contexto
> que não cabe em commit message.

## Origem

Em maio/2026, durante uma revisão informal do projeto com um colega, ele
fez quatro sugestões em sequência:

1. Fazer regressões pra visualizar a distribuição dos dados.
2. Estudar os outliers depois de ver os grupos.
3. Comparar classificação via regressão vs nossa GNN.
4. Procurar "outliers de GNN" — tipo um nó com repost de repost de repost.

Três das quatro tinham mérito direto. A terceira (regressão vs GNN) já
estava feita no projeto e o colega não tinha visto — apontei pros scripts
10, 11, 13 e 14 que já cobrem o cruzamento textual×topológico.

As outras três viraram a Fase 9.

### Por que essas três sobreviveram à filtragem

- **"Regressão pra visualizar distribuição"** estava conceitualmente confuso
  (regressão não visualiza distribuição), mas o que ele queria de fato era
  uma curva contínua mostrando onde o modelo erra ao longo de uma métrica.
  Isso vira **LOWESS** (smoother por janela móvel) — proposta defensável.

- **"Outliers depois dos grupos"** era boa ordem metodológica. Definir
  outlier antes de ver os dados é viés de seleção; depois de ver é
  análise pós-hoc legítima desde que declarada como tal.

- **"Outliers de GNN, tipo repost de repost de repost"** apontava para o
  ponto fraco real do nosso TCC: a §6.3.a declara como limitação que "FNN
  tem cascatas estrela", o que sugere que GNNs não teriam topologia rica
  pra explorar. Mas isso nunca foi medido — só assumido. Valia testar.

## Decisão e escopo

Antes de codar, conversamos sobre escopo. Decidimos:

- **Onde rodar:** UPFD-GossipCop test (3826 grafos). É onde temos modelos
  persistidos (script 17) e onde o sinal estrutural é forte (Cohen's d=1.53).
  PolitiFact ficaria pra rodada futura porque exigiria re-treino.
- **Ordem:** primeiro ver a distribuição (script 28), só depois definir
  outliers a partir dela (script 29), e por último a análise de profundidade
  (script 30). Cascata lógica.
- **Princípio:** scripts 29 e 30 leem do CSV que o 28 salva — não
  re-rodam modelo. Isso garante reprodutibilidade barata e elimina
  variação de re-execução.

## Execução

### Script 28 — distribuição e estratificação

- Carrega UPFD-GossipCop test (3826 grafos).
- Calcula 6 métricas estruturais por grafo (num_nodes, num_edges,
  grau_root, depth_max, width_max, branching_avg).
- Roda os modelos persistidos (RF tabular + SAGE estrutural do script 17)
  e salva predição+acerto de cada um em `por_grafo.csv` — esse arquivo
  vira insumo dos scripts 29 e 30.
- Ajusta lognormal nas distribuições e roda KS pra testar fit.
- Faz LOWESS de P(acerto) vs cada métrica.
- Sumariza profundidade em barras (preparação pro script 30).

**Decisão técnica que valeu:** implementei o LOWESS como rolling-mean
caseiro em vez de importar statsmodels. Uma dependência a menos, output
idêntico pra essa escala de dados.

### Script 29 — outliers

- Lê `por_grafo.csv`.
- Define 6 subgrupos via percentil empírico (p5, p95):
  - `real_viral` — y=real ∩ grau_root > p95
  - `fake_contido` — y=fake ∩ grau_root ≤ p5
  - `cauda_alta_n` / `cauda_baixa_n` — num_nodes nos extremos
  - `deep_extreme` — depth ≥ 5
  - `wide_extreme` — branching no p95
- Reporta N, acurácia, F1m (quando mixed-class), delta vs baseline.

**Decisão metodológica:** usei percentil empírico em vez de z-score porque
as distribuições são heavy-tail (KS rejeita lognormal). Ver figura
`00_o_que_e_heavy_tail.png` na pasta do orientador — mostra os 4 panéis
com Gaussiana ajustada sobreposta deixando claro que a Gaussiana erra.

### Script 30 — cascatas profundas

- Lê `por_grafo.csv`.
- Calcula F1 do RF e do SAGE em subsets de profundidade crescente
  (`depth≥1`, `depth≥3`, `depth≥5`).
- Roda **bootstrap pareado** (2000 reamostragens) pra obter IC 95% da
  diferença F1(SAGE) − F1(RF). Permite afirmar significância estatística
  mesmo em subset definido pós-hoc (sem folds).
- Exporta 3 visualizações HTML das cascatas mais profundas (depth=8,9)
  via pyvis — pra defesa oral.

**Decisão metodológica:** bootstrap em vez de t-test porque os subsets
não são pareados entre folds — são definidos como filtros sobre o test
único. Bootstrap é a ferramenta canônica para CI de diferença em amostra
única.

## Resultado

### Achados quantitativos (UPFD-GossipCop test, N=3826)

**1. Inversão distributiva confirmada com número:**

| Subgrupo | N | acc SAGE | Interpretação |
|---|---|---|---|
| `fake_contido` | 31 | **0.000** | Inversão total |
| `real_viral` | 60 | 0.433 | Abaixo de chance |

Antes da Fase 9 a §4.6 do TCC só dizia que o classificador topológico
"se inverte" na discordância (F1=0.08 agregado). Agora tem N e acc por
subgrupo nominal.

**2. Gap da GNN cresce com profundidade (bootstrap pareado):**

| Subset | N | F1 RF | F1 SAGE | Diff | IC 95% |
|---|---|---|---|---|---|
| Todos | 3826 | 0.753 | 0.814 | +0.060 | [+0.047, +0.074] |
| depth ≥ 3 | 1143 | 0.657 | 0.791 | **+0.134** | [+0.103, +0.165] |
| depth ≥ 5 | 256 | 0.528 | 0.741 | **+0.213** | [+0.139, +0.287] |

Todos significantes (IC não cruza zero). Gap em depth≥5 é 3.5× o agregado.

### Aprendizado metodológico

**Heavy-tail é mais moderado do que eu esperava neste dataset.** O UPFD
parece ter cap interno de tamanho (max num_nodes = 199, só 4× a mediana).
Não é power-law dramático. Mas a KS rejeita o fit lognormal porque com
N=3826 o teste é hipersensível a desvios pequenos. **O uso de percentil
continua sendo a escolha certa** — não porque a cauda seja dramática, mas
porque é invariante à forma da distribuição (defensável independente
do que seja).

A figura `00_o_que_e_heavy_tail.png` é honesta nisso: mostra que a
Gaussiana erra (extende-se a valores negativos, subestima a cauda direita),
mas não inflama a história.

### Achado bônus: contradição interna do TCC

A §6.3.a do TCC declara como limitação que "FNN com cascatas estrela" é
um problema do dataset. **70% dos grafos UPFD-GossipCop têm depth=2
(estrela rasa), mas 30% têm depth ≥ 3, com cauda chegando a depth=9.** A
crítica de "estrela plana" só vale pro FakeNewsNet que **construímos**
(script 00) — não pro UPFD oficial.

Anotei como pendência pra refinar a redação do Cap 6.

## Onde isso entra no TCC

- **Cap 3 §3.8** (novo): metodologia de LOWESS, percentil empírico, bootstrap
- **Cap 4 §4.8** (novo): resultados em 4 subseções:
  - 4.8.1 distribuição heavy-tail
  - 4.8.2 LOWESS por métrica
  - 4.8.3 outliers e inversão (T12, F29, F30)
  - 4.8.4 cascatas profundas (T13, F31, F32)
- **Cap 6 §6.3.a**: refinar redação sobre "cascata estrela"

LaTeX já scaffolded com figuras, tabelas, e comentários `% ARGUMENTO:` /
`% RESUMO:` / `% ESCREVER (Cesar):`. Falta só a prosa.

## Custo

- Tempo de codar: ~2h de conversa + iteração
- Tempo de rodar os 3 scripts: ~3min (a maior parte é o 28 carregando o
  UPFD-GossipCop e rodando o SAGE em 3826 grafos)
- Linhas de código: ~700 (28 + 29 + 30 + helper de figura)
- Dependências novas: zero (tudo numpy / scipy / sklearn / pyvis já
  presentes)

## O que ficou de fora (rodadas futuras)

- **Estratificação em UPFD-PolitiFact** — exigiria re-treino porque o
  script 17 só persistiu modelo pra GossipCop. Vale fazer se Cohen's d
  baixo do PolitiFact criar suspeita de que o achado da Fase 9 é
  específico de GossipCop.
- **Estratificação em FNN nosso** — mesmo problema + o argumento de
  cascata profunda não se aplica (FNN é estrela por construção).
- **Visualização de outliers reais** — alguns dos `real_viral` e
  `fake_contido` viram bons exemplos pra figura ilustrativa do Cap 4,
  mas isso é "nice-to-have", não bloqueio.
- **Análise textual dos outliers** — seria interessante ver se há padrão
  no texto das `fake_contido` (talvez fake news que não dependem de
  viralização — boatos nichados, por exemplo). Fora de escopo do TCC.
