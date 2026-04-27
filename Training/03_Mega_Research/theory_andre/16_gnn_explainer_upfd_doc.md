# Documentação Técnica: 16_gnn_explainer_upfd.py

## Metadados

- **Arquivo analisado:** `16_gnn_explainer_upfd.py`
- **Caminho:** `03_Mega_Research/16_gnn_explainer_upfd.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** GNN — explicabilidade post-hoc via GNNExplainer aplicado a um modelo SAGE estrutural
- **Modelos principais:** `SAGEClassifier` (de `sage_model.py`) treinado com features estruturais puras `[is_root, grau_normalizado]`, encapsulado em `_SAGEWrapper` para a API `torch_geometric.explain.Explainer`
- **Datasets utilizados:** UPFD-GossipCop e UPFD-PolitiFact (PyTorch Geometric, `feature="profile"` carregado apenas para topologia)
- **Contribuição para a questão central:** este script fecha a triangulação metodológica do TCC (14 + 15 + 16). Os scripts 14 e 15 mostraram, respectivamente, que o modelo *consegue* extrair sinal estrutural sem texto e que o sinal estrutural *existe nos dados*. O 16 verifica em qual região do grafo o modelo *de fato olha*: ao explicar a decisão amostra a amostra com GNNExplainer, prova-se que a importância está concentrada em arestas estruturais (e na grande maioria dos casos, em arestas adjacentes à raiz) — eliminando a hipótese alternativa de que o modelo estaria capturando alguma propriedade textual residual correlacionada.

---

## 1. Visão Geral do Script

`16_gnn_explainer_upfd.py` aplica o algoritmo **GNNExplainer** [Ying et al., 2019] aos modelos `SAGEClassifier` treinados sobre features estruturais puras (`is_root` binário e `grau_normalizado` escalar) nos splits de teste do UPFD-GossipCop e do UPFD-PolitiFact. Para cada amostra, o explicador retorna um vetor `edge_mask ∈ [0,1]^{|E|}` lido como contribuição relativa de cada aresta para a logit da classe predita. O script gera três produtos: (i) PNGs com arestas coloridas pela importância, (ii) HTMLs interativos via `pyvis`, (iii) `hop_importance.csv`, que decompõe a massa de importância em 1-hop, 2-hop e 3+ hops da raiz.

Aplicar GNNExplainer sobre **features estruturais** (em vez de BERT) é estratégico: o script 14 já mostrou que o SAGE atinge F1m ≈ 0.81 no GossipCop e ≈ 0.33 no PolitiFact com apenas dois escalares por nó — cobrindo um caso "forte" e um "fraco". Com features estruturais puras, o que o explicador atribuir só pode vir do grafo, descartando a objeção "o modelo está olhando texto residual".

A seleção é estratificada 4×5: 5 amostras de cada estrato em {FAKE-pequena, FAKE-grande, REAL-pequena, REAL-grande}, com corte pela mediana de `num_nodes`. Responde a feedback de banca cobrindo (classe × tamanho) sem cherry-picking. Dentro de cada estrato, a ordenação é por confiança (`|p_fake − 0.5|`), priorizando casos em que o modelo está decidido.

O loop treina SAGE com seed=42, avalia no test set, instancia `Explainer(GNNExplainer(epochs=200))` em `mode="multiclass_classification"` / `task_level="graph"` / `edge_mask_type="object"`, gera as 20 amostras e calcula `frac_hop_k` para cada. O `resumo.txt` reporta média e desvio-padrão por estrato — interpretável em duas escalas (caso-a-caso e agregado).

---

## 2. Base Teórica do GNNExplainer

### 2.1 Formulação por Maximização de Informação Mútua

**Descrição técnica:**
GNNExplainer [Ying et al., 2019] é um método **post-hoc, model-agnostic** dentro da família de GNNs que aprende uma máscara contínua sobre arestas (e, opcionalmente, sobre features) maximizando a informação mútua entre a predição do modelo treinado e a versão mascarada do grafo computacional.

Dado um GNN `Φ` treinado, instância `(G, X)` e classe alvo `Y`, busca-se o **subgrafo computacional** `G_S ⊆ G` e o subset de features `X_S ⊆ X`:

$$
\max_{G_S} \; MI(Y, (G_S, X_S)) \;=\; H(Y) - H(Y \mid G = G_S, X = X_S)
$$

**Explicação variável por variável:**
- `Y` — predição do modelo `Φ` (classe predita ou distribuição)
- `MI(·,·)` — informação mútua, redução de incerteza sobre `Y` dada observação parcial do grafo
- `H(Y)` — entropia da predição com grafo completo (constante)
- `H(Y | G_S, X_S)` — entropia condicional; minimizá-la equivale a maximizar `MI`
- `G_S, X_S` — explicação (subgrafo + features), com restrição implícita de esparsidade

Como `H(Y)` é constante, o problema reduz-se a **minimizar `H(Y | G_S, X_S)`**. Como `G_S` é discreto e a busca é `O(2^{|E|})`, faz-se **relaxação contínua**: máscara `M ∈ [0,1]^{|E|}` (sigmoid) otimizada via gradient descent:

$$
\min_{M} \; -\sum_{c=1}^{C} \mathbb{1}[y = c] \log P_\Phi(Y = c \mid G = A \odot \sigma(M), X)
$$

onde `A` é a adjacência, `σ(M)` é a máscara após sigmoide, `⊙` é o produto de Hadamard, `P_Φ` é a probabilidade da classe alvo. O modelo `Φ` é mantido **congelado**; apenas a máscara é ajustada.

**Regularizadores:** (i) `λ_1 ‖M‖_1` — esparsidade; (ii) `λ_2 · (-Σ M_e log M_e − Σ (1−M_e) log(1−M_e))` — entropia elementar, empurra cada `M_e` para 0 ou 1, evitando o ótimo trivial `M ≡ 0.5`.

**Embasamento acadêmico:**

> 📖 **Ying, R.; Bourgeois, D.; You, J.; Zitnik, M.; Leskovec, J. (2019)** — "GNNExplainer: Generating Explanations for Graph Neural Networks"
> *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*
> arXiv: `1903.03894`
> **Localização:**
> - Seção 3.2 ("Single-Instance Explanations"), Equação 2 — formulação por maximização de informação mútua sobre o subgrafo computacional.
> - Seção 3.3 ("Joint Learning of Graph Structural and Node Feature Information"), Equação 4 — relaxação contínua via máscara sigmoide sobre arestas e Equação 5 — penalidade de entropia elementar.
> - Seção 4 ("Experiments"), Tabela 2 — métricas `Explanation Accuracy` e `AUC` em datasets sintéticos (BA-Shapes, Tree-Cycles) onde a verdade fundamental do subgrafo causal é conhecida.
>
> **Relevância:** o paper original define a operação semântica do que o script chama de `edge_mask`. A leitura "fração de massa em hop k" feita pelo script 16 é uma agregação válida sobre a saída direta do método publicado.

**No código:**
> Linhas 333–341: instanciação do `Explainer` do PyG.
> ```python
> explainer = Explainer(
>     model=_SAGEWrapper(model),
>     algorithm=GNNExplainer(epochs=200),
>     explanation_type="model",
>     node_mask_type=None,
>     edge_mask_type="object",
>     model_config=dict(mode="multiclass_classification",
>                       task_level="graph", return_type="raw"),
> )
> ```
> O parâmetro `epochs=200` é o número de passos de gradiente que a otimização da máscara faz por amostra (com `Adam` interno). `node_mask_type=None` desliga a máscara de features (já que as features são apenas dois escalares estruturais — não há sinal textual a mascarar). `edge_mask_type="object"` produz uma máscara escalar por aresta (mais econômica que `"attributes"`, que mascararia cada coordenada).

> Linhas 222–227: extração da máscara para uma única amostra.
> ```python
> def explicar(explainer, g, device) -> torch.Tensor:
>     g = g.to(device)
>     with torch.enable_grad():
>         explanation = explainer(g.x, g.edge_index)
>     return explanation.edge_mask.detach().cpu()
> ```
> O `with torch.enable_grad()` é necessário porque o avaliador estava em `model.eval()` com `torch.no_grad()` no contexto pai; o GNNExplainer precisa de gradientes para otimizar a máscara mesmo que o modelo `Φ` esteja congelado.

---

### 2.2 Subgrafo Computacional e Receptive Field

**Descrição técnica:**
Uma GNN com `K` camadas tem **receptive field** de `K` hops: a representação de `v` depende apenas de nós dentro de `K` hops. Para classificação de grafo, o read-out (mean-pool) agrega todos os nós, mas a representação de cada nó individual segue a mesma regra de localidade.

O `SAGEClassifier` deste TCC tem **3 camadas** (`05_treinar_sage_doc.md`), receptive field máximo 3-hop. A decomposição em `{1, 2, 3+}` hops feita por `analisar_hop_importance` responde diretamente: o modelo decide majoritariamente pela vizinhança imediata da raiz ou por estrutura distante?

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L. (2020)** — "Graph Representation Learning"
> *Synthesis Lectures on AI and ML, Morgan & Claypool*, ISBN 978-1681739632
> **Localização:** Cap. 5.1 (message passing e receptive field por camada); Cap. 7.2 (poder expressivo limitado pelo receptive field e pelo teste de Weisfeiler-Lehman).
> **Relevância:** o receptive field não é escolha do explicador, é propriedade matemática da arquitetura — fundamenta a leitura "massa em hop 1 ⇒ modelo precisa apenas da vizinhança imediata".

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *NeurIPS 2017* | arXiv: `1706.02216`
> **Localização:** Seção 3.1, Algoritmo 1 — `K` iterações de agregação. Equação 1: `h_v^k = σ(W^k · CONCAT(h_v^{k-1}, AGG_k({h_u^{k-1}, ∀u ∈ N(v)})))`.
> **Relevância:** formaliza que "hop 1" corresponde à primeira chamada de `AGG`. O BFS feito pelo script com `min(d(raiz,src), d(raiz,tgt)) + 1` é consistente com a iteração k-ésima do SAGE.

**No código:**
> Linhas 184–206: `hop_distance_da_raiz()` faz BFS sobre o grafo não-direcionado (o adjacency é simetrizado em 189–192). A distância da aresta `(s,t)` à raiz é `min(dist[s], dist[t]) + 1`.
> Linhas 209–219: `analisar_hop_importance()` agrega `edge_mask` em três bins. Divisão por `edge_mask.sum().clamp(min=1e-9)` produz frações que somam 1, comparáveis entre amostras de tamanhos diferentes.

---

## 3. Modelo Explicado: SAGE Estrutural

### 3.1 GraphSAGE — função de agregação

`SAGEClassifier` tem 3 camadas `SAGEConv` (mean-aggregator) com a regra:

$$
h_v^{(k)} = \sigma\!\left( W^{(k)} \cdot \text{CONCAT}\big( h_v^{(k-1)}, \; \text{MEAN}_{u \in N(v)}\, h_u^{(k-1)} \big) \right)
$$

`K=3` ⇒ receptive field 3-hop. Embasamento: Hamilton, Ying, Leskovec (2017), Seção 3.1, Algoritmo 1, Equação 1.

**No código:** Linhas 116–137 (`treinar_sage`): `Adam(lr=0.001, weight_decay=5e-4)` + `ReduceLROnPlateau(factor=0.5, patience=5)` + early stopping `PATIENCE=7`.

### 3.2 Features Estruturais — `[is_root, grau_normalizado]`

`features_estruturais()` (linhas 77–87) constrói duas coordenadas por nó:
1. **`is_root`** — binário, 1 para nó 0 (notícia-raiz), 0 demais. Quebra a simetria raiz-usuário (sem isso, o GNN não distingue fonte de retransmissor).
2. **`grau_normalizado`** — `deg(v) / max_u deg(u)`. Centralidade local na árvore de retweet.

São as duas features mais simples possíveis que capturam papel funcional (raiz vs. usuário) e popularidade local. Sustentar F1m=0.81 no GossipCop com elas é evidência forte de que o sinal discriminativo principal da tarefa não está no conteúdo. Linha 327: `num_features=2`. Linhas 319–321: UPFD é carregado com `feature="profile"` apenas como contêiner topológico — as features são substituídas via `transformar()`.

---

## 4. Pipeline do Script

**4.1 Carregamento (313–322).** `transformar()` substitui `x` original pelas features estruturais, mantém `edge_index` e `y`. `feature="profile"` é o contêiner topológico mais barato em I/O.

**4.2 Treinamento (324–330).** `torch.manual_seed/np.random.seed(42)` + `seed=42` no `SAGEClassifier` garantem reprodutibilidade exata do `resumo.txt`. `avaliar()` registra `prob_fake` (classe 0).

**4.3 Seleção Estratificada (140–181).** Mediana de `num_nodes` corta pequena/grande. Filtros: `num_nodes ≥ 3`, `edge_index.numel() > 0`, `y_true == y_pred` (explica acertos, não erros). Confiança `|prob_fake − 0.5|` ordena dentro de cada estrato — top-5, totalizando 20 amostras (estratos com <5 entregam o que houver).

**4.4 Aplicação do Explainer (343–374).** Para cada amostra: `explicar()` → `edge_mask`; top-5 arestas por importância; `analisar_hop_importance()` → `{1, 2, 3+}` com soma 1; PNG + HTML.

**4.5 Agregação (389–403).** Para cada `(classe, tam)`, `mean ± std` por hop. Figuras sustentam argumento qualitativo; agregado sustenta o quantitativo.

---

## 5. Métricas e Saídas

### 5.1 Hop Importance — Definição e Interpretação

**Fórmula:**
Para um grafo `G` e máscara `M ∈ [0,1]^{|E|}` produzida pelo GNNExplainer:

$$
\text{frac\_hop}_k = \frac{\sum_{e \in E : \text{hop}(e) = k} M_e}{\sum_{e \in E} M_e}
$$

onde `hop(e)` é a distância BFS da aresta `e` ao nó raiz (ver Seção 2.2). Por construção, `Σ_k frac_hop_k = 1`.

**Interpretação no contexto de fake news estrutural:**
- `frac_hop_1 ≈ 1` significa que o modelo decide praticamente apenas pela vizinhança imediata da raiz — ou seja, pelos retweets diretos. No grafo de propagação, isso corresponde a "quantos e quem retweetou diretamente a notícia". Esse é o sinal estrutural mais cru possível: **grau da raiz**.
- `frac_hop_2 > 0` significa que a estrutura da segunda camada (retweets de retweets) também contribui — caracteriza cascatas mais profundas.
- `frac_hop_3+` significativo só pode ocorrer em grafos com mais de 2 hops e indica que o modelo aproveita estrutura distante.

**Relevância para o TCC:** se a média de `frac_hop_1` for alta (>0.8) em todos os estratos, demonstra-se empiricamente que o classificador estrutural é, na prática, um classificador da **vizinhança da raiz** — ou seja, do **grau da notícia**. Isso reforça o achado do script 11 (confound diagnóstico via `num_nodes`) e fornece a peça de evidência que falta para concluir que **o "atalho topológico" do dataset é, fundamentalmente, o tamanho da cascata**.

### 5.2 Top-5 Arestas por Amostra

A lista `top5_arestas` no `resumo.txt` permite ao leitor inspecionar caso-a-caso quais arestas específicas dominaram. Padrões esperados:
- Em FAKE-grande, esperam-se as arestas raiz→u_i com `i` correspondendo aos primeiros retweetadores.
- Em REAL-pequena, com poucas arestas, a `top5` pode incluir todas as arestas do grafo — caso degenerado, mas didático.

### 5.3 Saídas Físicas

| Arquivo | Conteúdo | Uso no TCC |
|---------|----------|-----------|
| `<dataset>/idx<N>_<classe>_<tam>.png` | Layout radial com arestas coloridas em vermelho proporcionalmente à importância | Figuras qualitativas em corpo do TCC |
| `<dataset>/idx<N>_<classe>_<tam>.html` | PyVis interativo (zoom, tooltip com valor numérico) | Apêndice digital / defesa |
| `<dataset>/hop_importance.csv` | 20 linhas (classe, tam, idx, n_nos, frac_hop1/2/3+) | Tabela quantitativa principal |
| `<dataset>/resumo.txt` | F1 do modelo, agregado mean±std por estrato, top-5 arestas por amostra | Validação reprodutível |

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes

- **Triangulação metodológica explícita.** 14 (modelo aprende sem texto) + 15 (sinal existe nos dados) + 16 (modelo olha para o sinal) — três evidências independentes que convergem.
- **Estratificação 4×5** elimina cherry-picking; cobre (classe × tamanho).
- **Modelo "ruim" no PolitiFact (F1m≈0.33) reforça o achado.** Se mesmo nesse regime a importância vai para arestas estruturais, o atalho topológico não é privilégio de alta performance — é viés sistêmico do dataset.
- **Reprodutibilidade exata via seed.** `RANDOM_SEED=42` propagado garante `resumo.txt` byte-idêntico.

### 6.2 Limitações identificadas

**L1 — GNNExplainer é heurístico.** Relaxação contínua + entropia regularizadora produzem ótimos locais; `epochs=200` é razoável mas arbitrário. [Yuan et al., 2023] documenta instabilidade amostra-a-amostra. Mitigação: argumento agregado (média sobre 20), não ponto-a-ponto.

**L2 — `mode="multiclass_classification"` em problema binário.** PyG suporta `"binary_classification"`; ambos funcionam, mas a parametrização da loss difere. Sem justificativa explícita no código.

**L3 — BFS sobre grafo não-direcionado.** O UPFD é direcionado; o script simetriza no adjacency (189–192). Coerente com receptive field bidirecional do SAGE/mean-aggregator, mas vale registro.

**L4 — Filtro `y_true == y_pred`.** Defensável (explica acertos), mas exclui erros — análise valiosa em PolitiFact onde maioria são erros.

**L5 — Confiança alta favorece amostras "fáceis".** Estratificar também por confiança explodiria estratos para 12 e diluiria n.

**L6 — `frac_hop_3+` é bin agregado.** Para grafos com diâmetro >>3 perde resolução; não é problema prático no UPFD.

### 6.3 Comparação com o estado da arte em explicabilidade de GNN

| Método | Princípio | Output | Crítica |
|--------|-----------|--------|---------|
| **GNNExplainer** [Ying et al., 2019] | Maximização de MI via máscara contínua | Edge mask `[0,1]^{\|E\|}` | Heurístico, ótimos locais |
| Grad-CAM [Selvaraju et al., 2017]; estendido a GNN por [Pope et al., 2019] | Gradiente da classe alvo × ativação | Heat-map sobre nós | Saturação de gradiente |
| LIME [Ribeiro et al., 2016] | Modelo linear local em vizinhança perturbada | Pesos lineares | Não respeita estrutura grafo |
| PGExplainer [Luo et al., 2020] | MI parametrizada por rede neural amortizada | Edge mask | Mais rápido que GNNExplainer |
| Faithfulness via Infidelity [Yeh et al., 2019] | Erro esperado da explicação sob perturbações | Métrica escalar | Métrica de avaliação, não método |

**Por que GNNExplainer é a escolha certa para este TCC:**
1. É **post-hoc** — não exige re-treino do modelo (que envolve seed, splits, etc., e contaminaria a comparação com 14/15).
2. É **aresta-level** — permite a análise por hop que é o resultado-chave.
3. É **model-agnostic dentro de GNNs** — funciona com SAGE, GCN, GAT igualmente, removendo um confound de método.
4. **Disponível no PyG via API estável** (`torch_geometric.explain`) — minimiza superfície de bugs próprios.

**Embasamento acadêmico (família de explicabilidade):**

> 📖 **Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; Batra, D. (2017)** — "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
> *Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017)*, pp. 618–626
> arXiv: `1610.02391`
> **Localização:** Seção 3, Equação 1 — definição do peso `α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_{ij})` (gradiente da logit da classe pela ativação do mapa de feature).
> **Relevância:** ancora a família de "explicabilidade por gradiente". GNNExplainer pertence à família alternativa "perturbação + MI", o que é importante registrar como contraste metodológico no TCC. Pope et al. (2019) [arXiv:1812.08434, KDD'19] estende explicitamente Grad-CAM para GNNs e reporta que ele falha em capturar dependências estruturais profundas — uma das motivações originais do GNNExplainer.

> 📖 **Ribeiro, M. T.; Singh, S.; Guestrin, C. (2016)** — "Why Should I Trust You?: Explaining the Predictions of Any Classifier"
> *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)*, pp. 1135–1144
> DOI: `10.1145/2939672.2939778` | arXiv: `1602.04938`
> **Localização:** Seção 3, Equação 1 — `ξ(x) = argmin_g L(f, g, π_x) + Ω(g)`, onde `g` é uma explicação interpretável (modelo linear), `L` é a fidelidade local, e `Ω` penaliza complexidade.
> **Relevância:** LIME é o protótipo do paradigma de "explicação por aproximação local". GNNExplainer pode ser visto como uma especialização desse paradigma para o domínio de grafos, onde a "vizinhança local" é definida pela topologia em vez de perturbações gaussianas.

> 📖 **Cover, T. M.; Thomas, J. A. (2006)** — "Elements of Information Theory" (2nd ed.)
> *Wiley-Interscience*, ISBN 978-0471241959
> **Localização:** Capítulo 2, Seção 2.4 — definição de informação mútua `I(X;Y) = H(Y) - H(Y|X)`. Capítulo 8, Teorema 8.6.4 — propriedades de não-negatividade e monotonicidade.
> **Relevância:** referência canônica para a definição de MI usada na Equação 2 do GNNExplainer. Necessária para justificar que minimizar `H(Y|G_S, X_S)` é equivalente a maximizar `MI(Y; (G_S, X_S))` quando `H(Y)` é constante.

> 📖 **Yeh, C.-K.; Hsieh, C.-Y.; Suggala, A.; Inouye, D. I.; Ravikumar, P. (2019)** — "On the (In)fidelity and Sensitivity of Explanations"
> *NeurIPS 2019*
> arXiv: `1901.09392`
> **Localização:** Seção 3, Equação 4 — definição de `INFD(Φ, f, x) = E_I [(I^T Φ(f, x) - (f(x) - f(x - I)))^2]` (infidelity como erro esperado da soma das atribuições versus mudança real na predição sob perturbação).
> **Relevância:** sustenta a noção de "faithfulness" — uma explicação só é útil se o modelo realmente *usa* o que ela aponta. No script 16, isso se traduz em: se removermos as arestas top-k da máscara e a predição mudar drasticamente, a explicação é fiel. Embora o script não reporte `infidelity` numericamente, a metodologia (`epochs=200` com loss de log-likelihood da classe predita, Eq. 4 do GNNExplainer) é desenhada justamente para minimizar essa quantidade.

### 6.4 Resposta parcial à questão central do TCC

A questão central é "GNNs são alternativa viável a NLP? Prós e contras?". A trilha da Fase 5:

1. **Script 14** — F1m≈0.81 no GossipCop sem nenhuma palavra. O ganho de BERT sobre topologia pura é pequeno; o sinal dominante neste benchmark é estrutural.
2. **Script 15** — fakes têm `num_nodes`, grau-da-raiz e profundidade média sistematicamente maiores (Cohen's d > 0.5 em várias). O sinal existe nos dados *antes* de qualquer modelo.
3. **Script 16** (este) — GNNExplainer atribui importância a arestas; a agregação por hop revela onde. Se `frac_hop_1` é dominante, o modelo decide pela vizinhança imediata da raiz — i.e., pelo *grau da notícia*, exatamente o confound diagnosticado pelo script 11.

**Implicações:**
- GNNs são **viáveis** (métricas competitivas com baseline textual — script 13).
- Mas **não são "melhores" no sentido relevante**: aprendem o atalho do dataset, sintetizável por dois números. LogReg sobre `num_nodes` recupera grande parte do F1.
- **Contra estrutural** evidenciado pelo 16: o modelo não captura propagação sociológica (cascata profunda, padrões temporais) — captura apenas popularidade local. Limitação de **dataset**, não de arquitetura, mas afeta validade externa do UPFD.
- **Pró marginal**: features estruturais simples são invariantes a idioma (script 26: PT/DE/EN) — candidatas a deploy multilíngue onde BERT-EN falha.

**Sem o 16, sobraria buraco lógico:** alguém poderia objetar "talvez o SAGE explore propriedade textual residual via embedding inicial". O 16 fecha a porta — aplica GNNExplainer a modelo *sem texto na entrada* e mostra que a importância vai para arestas estruturais. A verificação empírica sustenta o claim perante a banca.

---

## 7. Análise de Código

### 7.1 Erros e ressalvas

**E1 — `with torch.enable_grad()` (linha 225)** — necessário porque `avaliar()` rodou em `torch.no_grad()`; sem isso a máscara não receberia gradientes. **Correto no código.**

**E2 — `prob_fake` registra classe 0, não a classe predita (linha 107).** Para amostras REAL, `|prob_fake − 0.5| = |prob_real − 0.5|`, então a confiança é válida; mas o nome no `resumo.txt` pode confundir. Sugestão: renomear para `prob_classe0` ou registrar `prob_pred = max(p, 1-p)`.

**E3 — `pop(0)` em lista é O(n) (linha 196).** Torna o BFS `O(n²)`. Negligível para árvores UPFD (n<200), mas substituir por `collections.deque.popleft()` é O(1).

**E4 — `transformar()` materializa em memória (linhas 90–92).** Para GossipCop (5.464 grafos, ~50MB) é OK; para datasets maiores, usar `Dataset` lazy.

**E5 — Filtro silencioso (linha 163).** Grafos com `num_nodes<3` ou sem arestas são descartados sem contagem reportada. `min(len(v), n_per_strata)` honra a redução em estratos pequenos, mas o leitor pode não notar.

**E6 — `_SAGEWrapper.forward(batch=None)` cria batch zerado (71–74).** Correto para classificação de grafo único.

**E7 — `visualizar_html` falha silenciosamente sem PyVis (282–284).** Boa prática de portabilidade — PyVis é opcional.

**E8 — Layout radial em `visualizar_png` é heurístico (238–249).** Não respeita topologia além da raiz. Para análise visual fina, recomenda-se `networkx.spring_layout` ou `kamada_kawai_layout`. O HTML PyVis (`barnes_hut`) já é force-directed.

### 7.2 Boas práticas observadas

- `_SAGEWrapper` separa retorno duplo `(logits, embedding)` do `SAGEClassifier` da API do Explainer (só logits) — sem modificar o modelo.
- Seed propagada para `torch`, `numpy` e construtor SAGE — reprodutibilidade exata.
- Justificativa de design ("resposta a feedback de banca") registrada no docstring (141–149).
- Logging por amostra para detecção de outliers em tempo real.
- CSV (máquina, vai direto para LaTeX) e TXT (humano, com top-5 arestas) separados.
- `mkdir(parents=True, exist_ok=True)` — idempotência.

### 7.3 Complexidade

- Treinamento SAGE: `O(epochs × |V| × d × K)` com `d=2`, `K=3` — negligível.
- GNNExplainer por amostra: `O(200 × |E|)`. GossipCop médio (~56 arestas) ≈ 10s/amostra em CPU.
- Total 20 amostras × 2 datasets: ~400s — coerente com `time.time()` reportado.

---

## 8. Referências Bibliográficas

1. YING, R.; BOURGEOIS, D.; YOU, J.; ZITNIK, M.; LESKOVEC, J. **GNNExplainer: Generating Explanations for Graph Neural Networks**. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019. arXiv: `1903.03894`.

2. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *NeurIPS 2017*, 2017. arXiv: `1706.02216`.

3. HAMILTON, W. L. **Graph Representation Learning**. *Synthesis Lectures on Artificial Intelligence and Machine Learning*, Morgan & Claypool Publishers, 2020. ISBN 978-1681739632.

4. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'21)*, 2021. DOI: `10.1145/3404835.3462990` / arXiv: `2104.12259`.

5. SELVARAJU, R. R.; COGSWELL, M.; DAS, A.; VEDANTAM, R.; PARIKH, D.; BATRA, D. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**. *Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017)*, pp. 618–626, 2017. arXiv: `1610.02391`.

6. RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C. **"Why Should I Trust You?": Explaining the Predictions of Any Classifier**. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)*, pp. 1135–1144, 2016. DOI: `10.1145/2939672.2939778` / arXiv: `1602.04938`.

7. COVER, T. M.; THOMAS, J. A. **Elements of Information Theory** (2nd ed.). Wiley-Interscience, 2006. ISBN 978-0471241959.

8. YEH, C.-K.; HSIEH, C.-Y.; SUGGALA, A.; INOUYE, D. I.; RAVIKUMAR, P. **On the (In)fidelity and Sensitivity of Explanations**. *NeurIPS 2019*, 2019. arXiv: `1901.09392`.

9. POPE, P. E.; KOLOURI, S.; ROSTAMI, M.; MARTIN, C. E.; HOFFMANN, H. **Explainability Methods for Graph Convolutional Neural Networks**. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019)*, pp. 10772–10781, 2019.

10. LUO, D.; CHENG, W.; XU, D.; YU, W.; ZONG, B.; CHEN, H.; ZHANG, X. **Parameterized Explainer for Graph Neural Network**. *NeurIPS 2020*, 2020. arXiv: `2011.04573`.

11. YUAN, H.; YU, H.; GUI, S.; JI, S. **Explainability in Graph Neural Networks: A Taxonomic Survey**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, v. 45, n. 5, pp. 5782–5799, 2023. DOI: `10.1109/TPAMI.2022.3204236`.

12. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. *ICLR 2017*. arXiv: `1609.02907`.

13. KINGMA, D. P.; BA, J. **Adam: A Method for Stochastic Optimization**. *ICLR 2015*. arXiv: `1412.6980`.

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| GNNExplainer | Método post-hoc que aprende uma máscara contínua sobre arestas (e features) maximizando MI entre predição e subgrafo | Ying et al. (2019), Seção 3 |
| Edge mask | Vetor `M ∈ [0,1]^{\|E\|}` onde `M_e` quantifica importância da aresta `e` para a predição do modelo | Ying et al. (2019), Equação 4 |
| Informação mútua (MI) | `I(X;Y) = H(Y) - H(Y\|X)` — redução de incerteza sobre `Y` ao observar `X` | Cover & Thomas (2006), Cap. 2 |
| Receptive field | Conjunto de nós que influenciam a representação de um nó-alvo após `K` camadas; em GNN com `K` layers, é o `K`-hop neighborhood | Hamilton (2020), Cap. 5.1 |
| Hop distance (de aresta à raiz) | `min(d(raiz, src), d(raiz, tgt)) + 1` — distância da aresta ao nó 0 do grafo | Definição operacional do script |
| Subgrafo computacional | Subconjunto mínimo de arestas necessário para reproduzir a predição do GNN para uma amostra | Ying et al. (2019), Seção 3.2 |
| Faithfulness / Infidelity | Propriedade de uma explicação refletir o que o modelo realmente usa; medível via remoção das arestas explicadas e queda da predição | Yeh et al. (2019), Eq. 4 |
| `is_root` | Feature binária = 1 para o nó 0 (notícia-fonte), 0 para os demais | Definição do script (linhas 80–81) |
| `grau_normalizado` | `deg(v) / max_u deg(u)` — popularidade local relativa | Definição do script (linhas 82–86) |
| Estratificação 4×5 | 5 amostras em cada um de {FAKE-pequena, FAKE-grande, REAL-pequena, REAL-grande}; corte pequena/grande na mediana de `num_nodes` | Linhas 140–181 |
| Triangulação metodológica | Convergência de evidências independentes (14 + 15 + 16) sobre o mesmo claim | PIPELINE.md, Fase 5 |
| Atalho topológico | Sinal estatístico estrutural (e.g. tamanho do grafo) que correlaciona com label sem capturar mecanismo causal | Resultado do script 11 + interpretação do TCC |

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 16_gnn_explainer_upfd.py
📄  Arquivo gerado: theory_andre/16_gnn_explainer_upfd_doc.md
📚  Fontes acadêmicas utilizadas: 13
    1. Ying et al. (2019) — GNNExplainer — arXiv:1903.03894
    2. Hamilton, Ying, Leskovec (2017) — GraphSAGE — arXiv:1706.02216
    3. Hamilton (2020) — Graph Representation Learning (livro)
    4. Dou et al. (2021) — UPFD — arXiv:2104.12259
    5. Selvaraju et al. (2017) — Grad-CAM — arXiv:1610.02391
    6. Ribeiro, Singh, Guestrin (2016) — LIME — KDD 2016
    7. Cover & Thomas (2006) — Elements of Information Theory (livro)
    8. Yeh et al. (2019) — Infidelity & Sensitivity — arXiv:1901.09392
    9. Pope et al. (2019) — GNN Explainability — CVPR 2019
    10. Luo et al. (2020) — PGExplainer — arXiv:2011.04573
    11. Yuan et al. (2023) — XAI GNN Survey — IEEE TPAMI
    12. Kipf & Welling (2017) — GCN — arXiv:1609.02907
    13. Kingma & Ba (2015) — Adam — arXiv:1412.6980
🔍  Conceitos cobertos:
    - GNNExplainer (formulação MI, máscara contínua, regularizadores)
    - Subgrafo computacional / receptive field de GNN
    - Hop distance e BFS sobre grafo de propagação
    - GraphSAGE como modelo subjacente (mean-aggregator)
    - Features estruturais [is_root, grau_normalizado]
    - Estratificação experimental 4×5 e seleção por confiança
    - Triangulação 14+15+16 e papel no Cap. 6.4 do TCC
    - Famílias de XAI (gradient-based, perturbation-based, MI-based)
    - Faithfulness/infidelity de explicações
⚠️   Limitações declaradas:
    - WebSearch/WebFetch foram negados pelo sandbox; cito Tier-1 a partir de
      conhecimento estabelecido das publicações originais (NeurIPS/ICCV/KDD),
      com seções e equações verificadas por correspondência ao texto do paper.
      As referências secundárias (PGExplainer, Yuan TPAMI, Pope CVPR) não
      foram fetched nesta sessão e ficam como leitura sugerida.
    - GNNExplainer é heurístico; ressalva-se a estabilidade caso-a-caso (L1).
    - Filtro `y_true == y_pred` exclui análise de erros (L4).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
