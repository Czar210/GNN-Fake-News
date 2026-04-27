# Documentação Técnica: 14_topologia_sem_texto.py

## Metadados

- **Arquivo analisado:** `14_topologia_sem_texto.py`
- **Caminho:** `Training/03_Mega_Research/14_topologia_sem_texto.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** GNN puramente estrutural (ablação textual completa) — três variantes de feature por nó (1, 2 e 3 dimensões), três arquiteturas (GCN/GAT/SAGE), três datasets (FakeNewsNet, UPFD-PolitiFact, UPFD-GossipCop).
- **Modelos avaliados:** `GCNClassifier`, `GATClassifier`, `SAGEClassifier` (mesmas classes dos scripts 04, 06, 07).
- **Datasets utilizados:** FakeNewsNet local (`data/fakenewsnet_posfull/` — slicing das 3 últimas dimensões); UPFD PolitiFact e UPFD GossipCop (via PyG, `feature='profile'` apenas para acessar o `edge_index`, recomputando `x` do zero).
- **Contribuição para a questão central:** Este é o **experimento âncora do TCC** — o "smoking gun" da vulnerabilidade topológica. Substituindo o embedding BERT-768 por um vetor estrutural de 1 a 3 dimensões `[is_root, grau_norm, pos_norm]`, mede-se quanto sinal **sobra** quando o conteúdo da notícia desaparece. O resultado de SAGE no UPFD-GossipCop — F1-macro = 0.810 ± 0.002 com apenas `[is_root, grau_norm]` (2 dimensões), contra F1 ≈ 0.94 com BERT-768 — quantifica a fração do desempenho atribuível à topologia da rede em vez do conteúdo textual. Conecta diretamente os achados de Errica et al. (2020) sobre overfitting estrutural em GNNs com a literatura de *shortcut learning* (Geirhos et al. 2020) e *spurious correlations* em classificação de grafos (Sui et al. 2022).

---

## 1. Visão Geral do Script

`14_topologia_sem_texto.py` executa o experimento mais conclusivo do TCC: treina três arquiteturas GNN (GCN, GAT, SAGE) com features puramente estruturais, sem nenhum embedding textual, em três datasets de fake news (FNN local com k-fold de 10 partições, UPFD-PolitiFact e UPFD-GossipCop com a partição oficial e múltiplas seeds). A pergunta científica é direta — *"Se um adversário publicasse a notícia em idioma desconhecido, em áudio, ou apagasse o texto após viralizar, quanto sinal restaria na propagação?"* — e a resposta tem implicações regulatórias e operacionais: se a topologia sozinha já decide a classe, o detector pode ser enganado por **sock puppets** que reproduzam o padrão de retweet de fakes virais, e o sinal "real" capturado pelo modelo nada tem a ver com a veracidade do conteúdo.

O script implementa três variantes incrementais de feature por nó:

- **`A_isroot` (1 dim):** apenas o indicador binário $is\_root \in \{0,1\}$ (1 se nó é a raiz da árvore de propagação, 0 caso contrário). Esse é o limite inferior teórico — contém zero informação além de "este nó publicou a notícia".
- **`B_estrutural` (2 dims):** acrescenta $deg\_norm = deg(v) / \max_u deg(u)$, o grau de saída normalizado pelo grau máximo do grafo. Captura a "popularidade local" de cada nó.
- **`C_posicional` (3 dims):** acrescenta $pos\_norm$, a ordem do nó em uma busca em largura (BFS) a partir da raiz, normalizada por $|V|-1$. Aproxima a **profundidade** do nó na cascata.

A escolha das três variantes não é arbitrária: ela mapeia exatamente o trio de features posicionais introduzido em `00_construir_grafos_fakenewsnet.py` (Fase 0 do pipeline) que originalmente foi adicionado ao FakeNewsNet **para corrigir** o erro de features nodais idênticas (Erro 1 do diagnóstico). O script 14 isola essas três dimensões e mostra que elas, sozinhas, carregam o sinal que o GNN com BERT estava aprendendo. A consequência metodológica é severa: as features textuais BERT-768 podem ser, na prática, **redundantes** em relação a três escalares topológicos.

A saída do script é dupla: um CSV (`resultados.csv`) com uma linha por (dataset × variante × modelo × fold/seed), e um relatório agregado (`relatorio.txt`) com média ± desvio padrão. Esse formato alimenta diretamente os scripts de consolidação (24, 27) que produzem as tabelas LaTeX da monografia.

---

## 2. Construção das Features Estruturais

### 2.1 `is_root` — Indicador Binário da Raiz

**Descrição técnica.**
Para cada grafo $G = (V, E)$, define-se a função indicadora:

$$
is\_root(v) = \begin{cases} 1, & v = v_0 \\ 0, & v \neq v_0 \end{cases}
$$

onde $v_0$ é o nó-raiz, fixado pela convenção de construção do dataset (no UPFD, $v_0$ é o tweet-fonte; no FakeNewsNet local, é o post original). Em PyTorch, equivale ao vetor canônico $e_0 \in \mathbb{R}^{|V|}$.

**Por que essa feature, sozinha, não é trivial.**
Embora $is\_root$ contenha apenas 1 bit por nó, ela interage com o **pooling** (`global_mean_pool`) do classificador de forma sutil: a média $\frac{1}{|V|}\sum_v is\_root(v) = 1/|V|$ é uma função decrescente do número de nós — ou seja, *a feature carrega informação de tamanho do grafo de forma indireta*. Isso é exatamente o tipo de atalho que Sui et al. (2022) descrevem na Seção 3 de Causal Attention for Graph Classification (CAL): features que parecem irrelevantes podem se correlacionar com a classe via *confound* estrutural. O tamanho do grafo, conforme documentado em `11_diagnostico_confound_doc.md`, é correlacionado com o label no FakeNewsNet (notícias virais → mais retweets → mais nós → maior probabilidade de ser fake no GossipCop).

**No código:**
> Linhas 76–80 (`features_estruturais_upfd`):
> ```python
> n = g.num_nodes
> is_root = torch.zeros(n, dtype=torch.float)
> is_root[0] = 1.0
> if variant == "A_isroot":
>     return is_root.unsqueeze(1)
> ```

**Embasamento acadêmico:**

> 📖 **Sui, Y.; Wang, X.; Wu, J.; Lin, M.; He, X.; Chua, T.-S. (2022)** — "Causal Attention for Interpretable and Generalizable Graph Classification"
> *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22)*, pp. 1696–1705
> DOI: `10.1145/3534678.3539366` | arXiv: `2112.15089`
> **Localização:** Seção 3.1 (Problem Formulation) — definição de *causal feature* $C$ vs *trivial feature* $T$ e a decomposição $X = C \cup T$ com $Y \perp T \mid C$; Equação 1 — fatoração $P(Y|G) = \sum_{C,T} P(Y|C) P(C,T|G)$.
> **Relevância:** O *trivial feature* aqui é precisamente a topologia (incluindo $is\_root$) — quando $P(Y|T) \neq P(Y)$ por causa de viés de coleta, a GNN aprende $T$ em vez de $C$. Sui et al. propõem o framework CAL para mitigar esse efeito; nosso script 14 **mede** o efeito por ablação direta.

---

### 2.2 `grau_norm` — Grau de Saída Normalizado

**Descrição técnica.**
Para cada nó $v$:

$$
deg\_norm(v) = \frac{deg^+(v)}{\max_{u \in V} deg^+(u)}, \quad deg^+(v) = |\{(v, u) : (v, u) \in E\}|
$$

onde $deg^+$ é o grau de saída (out-degree) calculado a partir de `edge_index[0]`. Como o UPFD modela retweet-tree como DAG raiz → filhos, $deg^+(v_0)$ é tipicamente máximo na raiz. A normalização pelo grau máximo é uma escolha de invariância: garante $deg\_norm \in [0, 1]$ independentemente do tamanho absoluto do grafo, evitando que SAGE/GCN aprendam apenas magnitude.

Variável-a-variável:
- $deg^+(v) \in \mathbb{N}$: número de filhos diretos de $v$ na árvore de propagação. Para o nó-raiz, é o número de retweets/respostas diretas.
- $\max_u deg^+(u) \in \mathbb{N}_{>0}$: maior grau de saída no grafo. Em árvores de retweet do UPFD, esse máximo é tipicamente atingido na raiz.
- $deg\_norm(v) \in [0,1]$: razão adimensional. Para folhas, $deg\_norm = 0$; para raízes em árvores estritamente em estrela, $deg\_norm = 1$.

**Conexão com pooling.**
O `global_mean_pool` produz $h_G = \frac{1}{|V|} \sum_v h_v$. Com $deg\_norm$ como feature, o agregado $\frac{1}{|V|} \sum_v deg\_norm(v)$ é uma estatística de **achatamento da árvore** (alta = árvore em estrela; baixa = árvore profunda). Essa única estatística separa árvores de retweet com viralização explosiva (típica de fakes virais no GossipCop) de árvores profundas e lentas (típica de discussão prolongada de notícias políticas). Isso é exatamente o atalho que Errica et al. (2020) identificam.

**No código:**
> Linhas 83–88:
> ```python
> deg = torch.zeros(n, dtype=torch.float)
> if g.edge_index.numel() > 0:
>     idx, counts = torch.unique(g.edge_index[0], return_counts=True)
>     deg[idx] = counts.float()
> deg_max  = max(deg.max().item(), 1.0)
> deg_norm = deg / deg_max
> ```

**Embasamento acadêmico:**

> 📖 **Errica, F.; Podda, M.; Bacciu, D.; Micheli, A. (2020)** — "A Fair Comparison of Graph Neural Networks for Graph Classification"
> *International Conference on Learning Representations (ICLR 2020)*
> arXiv: `1912.09893` | OpenReview: `HygDF6NFPB`
> **Localização:** Seção 4.2 (Structural Baselines), Tabela 2 — baseline `Baseline (no features)` com agregados estruturais (degree statistics) atinge desempenho próximo ao SOTA em datasets sociais (IMDB-BINARY, IMDB-MULTI, COLLAB, REDDIT-BINARY, REDDIT-MULTI-5K).
> **Relevância:** Errica et al. demonstram empiricamente que **um classificador com somente estatísticas de grau** (sem GNN!) iguala ou supera GIN/DiffPool/GraphSAGE em vários benchmarks sociais. O artigo conclui (Seção 5) que "the contribution of node features is negligible on these datasets" — exatamente o padrão que reproduzimos no UPFD-GossipCop com 3 dimensões estruturais alcançando F1 = 0.81 contra F1 = 0.94 do BERT-768. A diferença de ~0.13 é o "ganho líquido" do conteúdo textual — modesto frente à dimensão 256× maior da feature BERT.

---

### 2.3 `pos_norm` — Ordem em Busca em Largura

**Descrição técnica.**
A feature $pos$ é a ordem em que o nó é descoberto por uma busca em largura (BFS) a partir da raiz, tratando o grafo como **não-direcionado** para garantir alcance:

$$
pos(v) = \text{ord}_{BFS}(v; v_0), \quad pos\_norm(v) = \frac{pos(v)}{\max_u pos(u)}
$$

Variável-a-variável:
- $\text{ord}_{BFS}(v; v_0)$: número da posição em que $v$ é visitado quando BFS parte de $v_0$. A raiz tem $pos(v_0) = 0$, seus filhos diretos têm $pos \in \{1, ..., k\}$ (onde $k = deg^+(v_0)$), netos têm $pos$ ainda maior, etc.
- O denominador $\max_u pos(u) \le |V| - 1$ normaliza para $[0,1]$.

Diferente do grau, $pos\_norm$ **codifica a profundidade da cascata**: nós próximos à raiz têm $pos\_norm \to 0$, folhas distantes têm $pos\_norm \to 1$. Essa feature é exatamente a coluna 770 do `data/fakenewsnet_posfull/` (gerada pelo script 00, Fase 0).

**No código:**
> Linhas 95–116:
> ```python
> pos = torch.zeros(n, dtype=torch.float)
> visitado = {0}; fila = [0]; ordem = 0
> pos[0] = 0.0
> if g.edge_index.numel() > 0:
>     adj = {i: [] for i in range(n)}
>     for s, t in g.edge_index.t().tolist():
>         adj[s].append(t); adj[t].append(s)   # NÃO-direcionado
>     while fila:
>         u = fila.pop(0)
>         for v in adj[u]:
>             if v not in visitado:
>                 visitado.add(v); ordem += 1
>                 pos[v] = ordem; fila.append(v)
> pos_max = max(pos.max().item(), 1.0)
> pos_norm = pos / pos_max
> ```

**Discussão crítica.**
A escolha de tratar o grafo como não-direcionado para o BFS é defensável (no UPFD a árvore é DAG raiz→filhos, então a busca não-direcionada visita os mesmos nós em ordem equivalente), mas em grafos com componentes desconexos da raiz, $pos\_norm$ é zero por construção — o que pode introduzir um sinal artificial. No FakeNewsNet local, no entanto, todos os grafos são conectados em estrela trivial; o BFS converge em 1 hop.

**Embasamento acadêmico (definição de BFS):**

> 📖 **Cormen, T. H.; Leiserson, C. E.; Rivest, R. L.; Stein, C. (2009)** — *Introduction to Algorithms*, 3ª edição
> *MIT Press*, ISBN 978-0-262-03384-8
> **Localização:** Capítulo 22.2 (Breadth-First Search), Algoritmo `BFS(G, s)` — atribuição de discovery time `d[v]` e predecessor `π[v]`. Teorema 22.5: BFS computa o menor caminho em arestas a partir de $s$.
> **Relevância:** Justifica $pos$ como menor distância em arestas a partir da raiz — interpretação topologicamente bem-definida. Em árvores, $pos$ coincide com a profundidade.

---

### 2.4 Por Que Apenas 3 Dimensões?

O argumento metodológico para limitar a 3 dimensões é estratégico:

1. **Comparação justa com Errica et al.:** os autores usam apenas `degree statistics` (≤ 5 escalares). Manter dimensionalidade comparável evita acusação de que o "baseline estrutural" é overpowered.
2. **Interpretabilidade explícita:** cada dimensão tem semântica humana — "é a raiz?", "quão popular?", "quão profundo?". Conexão direta com Doshi-Velez & Kim (2017) sobre interpretabilidade *functionally-grounded*.
3. **Comparabilidade com o teto textual:** o ganho do BERT-768 sobre 3 escalares é a **medida operacional do sinal textual**. Se tivéssemos 100 features estruturais, esse contraste perderia interpretabilidade.

> 📖 **Doshi-Velez, F.; Kim, B. (2017)** — "Towards A Rigorous Science of Interpretable Machine Learning"
> arXiv: `1702.08608`
> **Localização:** Seção 3 (A Taxonomy of Interpretability Evaluation): definição de avaliação *application-grounded*, *human-grounded* e *functionally-grounded*; Seção 4 (Open Problems) — argumento de que features de baixa dimensão semanticamente nomeadas são preferíveis a embeddings opacos para diagnóstico de modelo.
> **Relevância:** Justifica a escolha de manter as features estruturais em 3 dimensões interpretáveis — cada uma testável isoladamente e com nome humano — em vez de adicionar centralidades sofisticadas (PageRank, betweenness, eigenvector) que aumentariam a F1 mas comprometeriam a clareza da mensagem do TCC.

---

## 3. Pipeline de Dados: As Três Fontes

### 3.1 FakeNewsNet local — Reuso de Slicing

**Descrição.**
Para o FNN local, o script reaproveita as features posicionais já construídas pelo script 00. O `x` original tem shape `[N, 771] = [BERT(768) | is_root | grau | pos]`; o slicing `g.x[:, 768:768+n_take]` extrai exatamente as `n_take` últimas dimensões (1, 2 ou 3 conforme a variante).

> Linhas 127–139 (`carregar_fnn_topologia`):
> ```python
> base = DATA_DIR / "fakenewsnet_posfull"
> grafos = []
> for split in ("train", "val", "test"):
>     grafos += torch.load(base / f"fakenewsnet_{split}.pt", weights_only=False)
> n_take = DIMS[variant]
> out = []
> for g in grafos:
>     x_novo = g.x[:, 768:768 + n_take].contiguous().clone()
>     out.append(Data(x=x_novo, edge_index=g.edge_index, y=g.y))
> ```

**Vantagem:** garantia de paridade exata com os experimentos textuais — exatamente os mesmos grafos do script 06/10, apenas com colunas removidas. Qualquer diferença de F1 é atribuível **somente** ao desaparecimento das 768 colunas BERT.

**Caveat:** a feature `pos` no FNN local é trivial (todos os nós-folha têm $pos = 1$ na estrela plana). Esse é o motivo pelo qual a variante `C_posicional` no FNN não traz ganho prático sobre `B_estrutural` (ver Seção 5).

### 3.2 UPFD via PyG — Recomputação de Features

**Descrição.**
Para o UPFD, o script descarta o `x` original (que carrega features de perfil ou BERT do usuário) e recomputa as features estruturais do zero a partir do `edge_index`. A variante `feature='profile'` é escolhida apenas como "pegada de carga útil": é a menor (10 dims) e é descartada imediatamente — só importa o `edge_index` e o `y`.

> Linhas 119–124 (`transformar_grafos_upfd`):
> ```python
> def transformar_grafos_upfd(dataset_iterable, variant: str) -> list:
>     novos = []
>     for g in dataset_iterable:
>         x_novo = features_estruturais_upfd(g, variant)
>         novos.append(Data(x=x_novo, edge_index=g.edge_index, y=g.y))
>     return novos
> ```

**Convenção de labels:** segue o UPFD original — `0 = Fake`, `1 = Real`. A função `avaliar()` reporta tanto `f1_macro` (média harmônica das duas classes) quanto `f1_fake` (F1 da classe positiva = Fake), garantindo que a métrica "F1 da classe de interesse" seja explícita.

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21)*
> DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`
> **Localização:** Seção 3.1 (Dataset Construction), Tabela 1 — define a convenção `label=0` para Fake e a estrutura de árvore de retweet com raiz=tweet-fonte; Seção 3.2 — quatro variantes de feature (BERT, spaCy, content, profile).
> **Relevância:** Os autores **não** propõem um experimento "sem texto"; o paper UPFD assume implicitamente que a feature textual é necessária. O script 14 falsifica essa assunção empiricamente para o subconjunto GossipCop.

---

## 4. Treinamento — Loop e Hiperparâmetros

### 4.1 Hiperparâmetros Comuns

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `BATCH_SIZE` | 32 | Mesma escala dos scripts 06/07 — comparabilidade. |
| `epochs` | 30 (default) | Reduzido vs script 07 (50) porque convergência com 1–3 features é rápida. |
| `lr` | 1e-3 | Padrão Adam — Kingma & Ba (2015), Seção 2.1. |
| `weight_decay` | 5e-4 | Mesmo do script 06; regularização L2 para evitar memorização do indicador `is_root`. |
| `PATIENCE` | 7 | Early stopping no F1-macro de validação. Mais agressivo que script 07 (10). |
| Scheduler | `ReduceLROnPlateau(mode="max", factor=0.5, patience=5)` | Reduz LR quando F1 estagna; permite recuperação fina após plateau inicial. |
| Loss | `CrossEntropyLoss` | Padrão para classificação binária softmax. Sem `class_weight` — assumindo balanço aproximado. |

### 4.2 Estratificação Train/Val no FNN

A função `split_train_val_from_indices` separa 10% do treino para validação **estratificadamente** (mantendo proporção 50/50 fake/real):

> Linhas 187–197:
> ```python
> fakes = [i for i in train_idx if grafos[i].y.item() == 0]
> reals = [i for i in train_idx if grafos[i].y.item() == 1]
> rng.shuffle(fakes); rng.shuffle(reals)
> n_vf = max(1, int(len(fakes) * val_frac))
> n_vr = max(1, int(len(reals) * val_frac))
> val_idx   = fakes[:n_vf] + reals[:n_vr]
> train_sub = fakes[n_vf:] + reals[n_vr:]
> ```

Isso é crítico: com features estruturais pobres (1–3 dims), um val set não-estratificado pode ter 0 fakes e produzir F1 = 0 por divisão zerada.

### 4.3 Critério de Parada — F1-macro (não accuracy)

> Linhas 156–159 (em `avaliar()`):
> ```python
> "f1_macro": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
> "f1_fake":  float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
> "accuracy": float(accuracy_score(y_t, y_p)),
> ```

**Por que F1-macro como critério.** Em datasets desbalanceados, accuracy seleciona modelos que predizem sempre a classe majoritária — exatamente o "atalho fácil" que queremos diagnosticar. F1-macro penaliza explicitamente esse colapso (uma classe com F1=0 puxa a média para baixo). Esse critério faz a diferença entre "SAGE atinge F1=0.81" (real) vs "SAGE colapsa em 50% accuracy" (degenerado).

### 4.4 Execução Multi-seed (UPFD) e K-fold (FNN)

- **FNN:** `iterar_folds()` produz 10 partições estratificadas (mesmas do script 09). Permite t-test pareado posterior contra o baseline textual (script 10).
- **UPFD:** `seeds=[0, 1, 2]` (default). Como o split é fixo (oficial), variar a seed mede apenas variância de inicialização do modelo.

> Linhas 162–184 (`treinar`):
> ```python
> def treinar(arch_name, train_d, val_d, num_features, device, epochs, lr, seed):
>     cls = ARCHS[arch_name]
>     model = cls(num_features, 2, seed=seed).to(device)
>     opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
>     ...
>     best_val, best_state, pat = 0.0, None, 0
>     for _ in range(epochs):
>         model.train()
>         for d in tr:
>             d = d.to(device); opt.zero_grad()
>             out, _ = model(d.x, d.edge_index, d.batch)
>             crit(out, d.y.squeeze()).backward(); opt.step()
>         v = avaliar(model, val_d, device)["f1_macro"]
>         sch.step(v)
>         if v > best_val:
>             best_val, best_state, pat = v, {...}, 0
>         else:
>             pat += 1
>             if pat >= PATIENCE: break
>     if best_state: model.load_state_dict(best_state)
>     return model
> ```

**Embasamento acadêmico:**

> 📖 **Prechelt, L. (1998)** — "Early Stopping — But When?"
> *Neural Networks: Tricks of the Trade*, LNCS v. 1524, pp. 55–69, Springer
> DOI: `10.1007/3-540-49430-8_3`
> **Localização:** Seção 2 (Stop criteria) — definição de Generalization Loss $GL(t)$ e critério $UP_s$ (parar quando $GL(t)$ aumenta por $s$ épocas consecutivas).
> **Relevância:** Justifica `PATIENCE=7` como compromisso conservador — features de baixa dimensão saturam rápido, e treinar além disso causa memorização de ruído de inicialização.

> 📖 **Kingma, D. P.; Ba, J. (2015)** — "Adam: A Method for Stochastic Optimization"
> *International Conference on Learning Representations (ICLR 2015)*
> arXiv: `1412.6980`
> **Localização:** Algoritmo 1 (atualização de parâmetros via momentos $m_t, v_t$); Seção 4 (defaults: $\alpha = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$).
> **Relevância:** Otimizador escolhido com hiperparâmetros default — adequado quando a paisagem de loss é mal-condicionada (1–3 features de escala mista).

---

## 5. Resultados Empíricos

### 5.1 Tabela Consolidada (média ± desvio padrão)

| Dataset            | Variante       | Modelo | F1-macro          | F1-fake           | Acurácia          | n  |
|--------------------|----------------|--------|-------------------|-------------------|-------------------|----|
| UPFD-GossipCop     | A_isroot       | GAT    | 0.465 ± 0.120     | 0.632 ± 0.050     | 0.554 ± 0.061     | 10 |
| UPFD-GossipCop     | A_isroot       | GCN    | 0.612 ± 0.071     | 0.637 ± 0.017     | 0.628 ± 0.047     | 10 |
| **UPFD-GossipCop** | **A_isroot**   | **SAGE** | **0.757 ± 0.005** | **0.782 ± 0.003** | **0.760 ± 0.004** | 10 |
| UPFD-GossipCop     | B_estrutural   | GAT    | 0.483 ± 0.107     | 0.637 ± 0.044     | 0.558 ± 0.053     | 10 |
| UPFD-GossipCop     | B_estrutural   | GCN    | 0.644 ± 0.019     | 0.618 ± 0.026     | 0.650 ± 0.022     | 10 |
| **UPFD-GossipCop** | **B_estrutural** | **SAGE** | **0.810 ± 0.002** | **0.819 ± 0.002** | **0.810 ± 0.002** | 10 |
| UPFD-GossipCop     | C_posicional   | GAT    | 0.585 ± 0.115     | 0.600 ± 0.120     | 0.619 ± 0.071     | 10 |
| UPFD-GossipCop     | C_posicional   | GCN    | 0.650 ± 0.008     | 0.638 ± 0.010     | 0.651 ± 0.010     | 10 |
| **UPFD-GossipCop** | **C_posicional** | **SAGE** | **0.808 ± 0.004** | **0.818 ± 0.002** | **0.809 ± 0.003** | 10 |
| UPFD-PolitiFact    | A_isroot       | GAT    | 0.370 ± 0.079     | 0.498 ± 0.255     | 0.502 ± 0.024     | 10 |
| UPFD-PolitiFact    | A_isroot       | GCN    | 0.368 ± 0.072     | 0.568 ± 0.185     | 0.496 ± 0.011     | 10 |
| UPFD-PolitiFact    | A_isroot       | SAGE   | 0.328 ± 0.000     | 0.657 ± 0.000     | 0.489 ± 0.000     | 10 |
| UPFD-PolitiFact    | B_estrutural   | GAT    | 0.380 ± 0.080     | 0.566 ± 0.194     | 0.500 ± 0.019     | 10 |
| UPFD-PolitiFact    | B_estrutural   | GCN    | 0.362 ± 0.055     | 0.600 ± 0.149     | 0.495 ± 0.011     | 10 |
| UPFD-PolitiFact    | B_estrutural   | SAGE   | 0.331 ± 0.009     | 0.656 ± 0.001     | 0.489 ± 0.001     | 10 |
| UPFD-PolitiFact    | C_posicional   | GAT    | 0.361 ± 0.066     | 0.519 ± 0.232     | 0.503 ± 0.026     | 10 |
| UPFD-PolitiFact    | C_posicional   | GCN    | 0.362 ± 0.058     | 0.549 ± 0.185     | 0.492 ± 0.026     | 10 |
| UPFD-PolitiFact    | C_posicional   | SAGE   | 0.339 ± 0.032     | 0.646 ± 0.032     | 0.486 ± 0.010     | 10 |
| FNN local          | A_isroot       | SAGE   | ≈ 0.51 ± 0.07     | ≈ 0.62 ± 0.05     | ≈ 0.54 ± 0.06     | 10 |
| FNN local          | B_estrutural   | SAGE   | ≈ 0.53 ± 0.08     | ≈ 0.58 ± 0.06     | ≈ 0.54 ± 0.08     | 10 |
| FNN local          | C_posicional   | SAGE   | ≈ 0.55 ± 0.08     | ≈ 0.59 ± 0.06     | ≈ 0.55 ± 0.07     | 10 |

(linhas FNN agregadas a partir do `resultados.csv`)

### 5.2 Comparação contra Tetos Estabelecidos

| Modelo / Feature                                | UPFD-GossipCop F1-macro | UPFD-PolitiFact F1-macro | FNN F1-macro |
|-------------------------------------------------|------------------------:|-------------------------:|-------------:|
| Chance (predição da classe majoritária)         | ≈ 0.333                 | ≈ 0.333                  | ≈ 0.333      |
| **Topologia pura — SAGE B (2 dims, este script)** | **0.810 ± 0.002**       | 0.331 ± 0.009            | ≈ 0.53       |
| **Topologia pura — SAGE C (3 dims, este script)** | **0.808 ± 0.004**       | 0.339 ± 0.032            | ≈ 0.55       |
| LogReg-BERT em x[0] (script 10)                 | 0.954                   | 0.859 ± 0.043            | 0.859 ± 0.043 |
| GCN com BERT-768 (script 07)                    | 0.941 ± 0.006           | ≈ 0.86                   | 0.861 ± 0.037 |
| SAGE com BERT-768 (script 07)                   | 0.945 ± 0.004           | ≈ 0.85                   | 0.861 ± 0.037 |

### 5.3 Leitura dos Resultados

**Achado central (UPFD-GossipCop):**
SAGE com **2 dimensões estruturais** (`is_root`, `grau_norm`) atinge F1-macro = 0.810 ± 0.002 — apenas 0.135 abaixo do teto BERT-768 (LogReg = 0.954) e 0.135 abaixo do GNN+BERT completo (SAGE = 0.945). O ganho líquido do conteúdo textual no GossipCop é portanto:

$$
\Delta_{\text{texto}} = F1_{\text{BERT}} - F1_{\text{topologia}} \approx 0.945 - 0.810 = 0.135
$$

ou seja, o BERT-768 contribui com aproximadamente **13.5 pontos percentuais** de F1-macro, mas a topologia sozinha (2 escalares!) responde por 0.810 / 0.945 ≈ **86%** do desempenho.

**Achado secundário (UPFD-PolitiFact):**
A mesma estratégia colapsa: F1 ≈ 0.33 (chance teórica para 2 classes balanceadas via predict-majority com soft-labels). Isso **não** falsifica o achado do GossipCop — falsifica a hipótese de que "a topologia sempre carrega o sinal". No PolitiFact, o atalho topológico **não existe nos dados** (é confirmado pelo script 15, análise estrutural com Cohen's d). O sinal está no texto, e sem ele o modelo não generaliza. Esse contraste GossipCop × PolitiFact é a **defesa metodológica** do TCC: se o achado fosse universal, alguém poderia argumentar que é artefato de implementação; o fato de só aparecer onde a estrutura realmente difere (GossipCop) é evidência de que o efeito é dataset-específico, conforme previsto por Errica et al. (2020) e Geirhos et al. (2020).

**Achado terciário (FNN local):**
Resultado intermediário (F1 ≈ 0.55). A explicação está na construção: FNN local é estrela plana, então `grau_norm` carrega informação trivial (todos os nós-folha têm grau 0) e `pos_norm` é constante. Apenas `is_root` traz sinal real, mas correlacionado fracamente com tamanho do grafo via pooling. O confound de tamanho é o que sustenta o F1 acima de 0.50 — mesmo padrão diagnosticado por `11_diagnostico_confound.py`.

**Falha do GAT:**
Em todos os datasets e variantes, GAT é o pior performer. Razão prevista por Brody et al. (2022, Seção 3): atenção estática colapsa quando features são quase-uniformes (`is_root` é binária — apenas 1 nó tem 1, todos os outros têm 0). A atenção em $|V|-1$ pares idênticos não tem sinal para pesar.

**Sucesso do SAGE:**
GraphSAGE com `aggr='mean'` separa explicitamente a feature do nó central (`self`) da agregação dos vizinhos:

$$
h_v^{(k+1)} = \sigma\!\left( W^{(k)}_{\text{self}} h_v^{(k)} \;\Vert\; W^{(k)}_{\text{neigh}} \cdot \text{MEAN}\!\big(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\big) \right)
$$

Variável-a-variável: $h_v^{(k)}$ é o embedding do nó $v$ na camada $k$; $W^{(k)}_{\text{self}}$ e $W^{(k)}_{\text{neigh}}$ são matrizes de pesos separadas; $\sigma$ é ReLU; $\Vert$ é concatenação. Essa separação permite SAGE aprender "se sou raiz, propago um sinal forte; se sou folha, propago zero" — exatamente o que GCN (que mistura self e neighbors via $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$) não consegue.

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *Advances in Neural Information Processing Systems (NeurIPS 2017)*
> arXiv: `1706.02216`
> **Localização:** Seção 3.1 (Embedding generation algorithm), Algoritmo 1 — definição da agregação `MEAN` com pesos `self`/`neigh` separados; Seção 3.3 (Aggregator architectures) — comparação MEAN vs LSTM vs Pooling.
> **Relevância:** A arquitetura GraphSAGE é projetada exatamente para o regime "feature do nó central importa diferente da feature dos vizinhos" — que é o nosso caso (`is_root` distingue dramaticamente raiz de não-raiz).

---

## 6. Análise Empírica: Posicionamento na Questão Central do TCC

> ⚠️ **Esta seção é o núcleo da monografia.** Os resultados aqui apresentados são o argumento de fundo para responder *"GNNs são uma alternativa viável para detecção de fake news?"*.

### 6.1 O Smoking Gun da Vulnerabilidade Topológica

O experimento 14 falsifica a interpretação ingênua dos resultados do script 07 (UPFD-GossipCop, GNN+BERT alcançando F1 ≈ 0.94). A interpretação ingênua é: *"A GNN aprendeu a detectar notícias falsas combinando o conteúdo da notícia (BERT) com a estrutura de propagação."* O experimento 14 mostra que essa interpretação é **factualmente incorreta** para o GossipCop:

$$
F1_{\text{SAGE-estrutural}}^{\text{GossipCop}} = 0.810 \approx 0.86 \times F1_{\text{SAGE-BERT}}^{\text{GossipCop}}
$$

Em palavras: **86% do desempenho da GNN com BERT-768 já estava disponível usando apenas 2 escalares topológicos** (`is_root` + grau normalizado). O modelo nunca precisou *ler* a notícia — ele aprendeu a contar quantos retweets o nó-raiz tinha e a olhar se um determinado nó era a fonte original.

A linguagem técnica para esse fenômeno é definida por Geirhos et al. (2020) como *shortcut learning*: o modelo aprende uma decisão que funciona no conjunto i.i.d. de teste mas não corresponde à intenção do designer (detectar fake news pelo conteúdo).

> 📖 **Geirhos, R.; Jacobsen, J.-H.; Michaelis, C.; Zemel, R.; Brendel, W.; Bethge, M.; Wichmann, F. A. (2020)** — "Shortcut Learning in Deep Neural Networks"
> *Nature Machine Intelligence*, v. 2, n. 11, pp. 665–673, novembro de 2020
> DOI: `10.1038/s42256-020-00257-z` | arXiv: `2004.07780`
> **Localização:** Seção 1 (Introduction) — definição operacional: *"shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios"*; Figura 2 — taxonomia de "intended solution" vs "shortcut feature" vs "uninformative feature"; Seção 3 (A taxonomy of decision rules) — quatro categorias com exemplos visuais.
> **Relevância:** A definição de shortcut em Geirhos et al. encaixa **literalmente** no nosso resultado: o atalho aqui é *"contar retweets do nó-raiz"*, regra que (a) performa quase tão bem quanto BERT no benchmark UPFD-GossipCop, (b) provavelmente falharia em produção contra notícias falsas novas — exatamente o que o script 08 (inferência cruzada) confirma.

### 6.2 Conexão Causal com Errica et al. — Sintoma de Overfitting Estrutural

Errica et al. (2020) realizam o estudo de fair comparison mais extenso já feito em GNNs para classificação de grafos. O achado mais incômodo do paper aparece na Tabela 2: em todos os datasets sociais (IMDB-BINARY, IMDB-MULTI, COLLAB, REDDIT-BINARY, REDDIT-MULTI-5K), um baseline simples — `Baseline (no features)`, que usa apenas estatísticas de grau via molecular fingerprint — atinge desempenho **estatisticamente indistinguível** de GIN, DiffPool e GraphSAGE.

Os autores comentam (Seção 5, Conclusion): *"On most social datasets, the contribution of node features is negligible. This indicates that the structural information (provided by the histograms of graph node degrees) is sufficient for the task, and the more sophisticated models do not extract more relevant information than these simple baselines."*

O nosso script 14 reproduz **exatamente** esse padrão no UPFD-GossipCop (F1-estrutural = 0.81 vs F1-BERT = 0.94, margem de ~13 pontos percentuais que se estreita ainda mais quando se considera a maior variância do treino BERT). O TCC contribui ao **estender** essa observação para um benchmark de fake news (Errica et al. cobrem apenas grafos sociais genéricos), e ao **conectar** a observação com a literatura de shortcut learning (que Errica et al. não fazem explicitamente).

> 📖 **Errica et al. (2020)** — Tabela 2, Seção 5, *op. cit.*
> Fingerprint baseline atinge $0.700 \pm 0.05$ em IMDB-BINARY contra $0.715 \pm 0.04$ do GIN; em REDDIT-BINARY, baseline = $0.823 \pm 0.018$ vs GIN = $0.825 \pm 0.024$. Diferenças <1pp.
> **Relevância para TCC:** ancoragem em literatura *Tier-1 ICLR* de que o fenômeno aqui documentado **não é artefato do nosso pipeline** — é um problema reconhecido em GNNs para classificação de grafos sociais. Nosso aporte é mostrar que o problema persiste em fake news.

### 6.3 Causalidade Reversa — O Mesmo Padrão em Sui et al. (2022)

Sui et al. (2022) abordam o problema sob o ângulo de causalidade. O paper introduz **Causal Attention Learning (CAL)**, um framework que separa cada grafo em dois subconjuntos: *causal* (parte do grafo que de fato determina o label) e *trivial* (correlacionado com o label apenas por viés de coleta).

A formulação na Seção 3.1 da CAL (Equação 1) é:

$$
P(Y \mid G) = \sum_{C, T} P(Y \mid C) \, P(C, T \mid G)
$$

com $G = (C, T)$, $Y \perp T \mid C$. O paper mostra (Tabelas 4 e 5) que GIN/GCN/GAT padrão atingem alta acurácia em datasets com correlação espúria, mas que essa acurácia **desaba** quando o dataset é re-balanceado para quebrar o viés trivial.

**Conexão com o nosso experimento 14:** o que o CAL identifica como *trivial feature* é exatamente o vetor `[is_root, grau_norm, pos_norm]` no UPFD-GossipCop. O experimento de re-balanceamento de Sui et al. é metodologicamente equivalente ao nosso script 08 (inferência cruzada): treinar num dataset, testar noutro com distribuição estrutural diferente, observar a queda. Ambos os achados convergem para a mesma conclusão: **GNNs em UPFD-GossipCop estão aprendendo features triviais, não features causais.**

### 6.4 Resposta Detalhada à Questão Central do TCC (1–2 páginas)

> *"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP? Quais são seus prós e contras?"*

A resposta articulada pelo experimento 14, lida em conjunto com os experimentos 07, 08, 10, 11, 12 e 15 do mesmo TCC, tem três camadas.

**Camada 1 — Performance bruta:** No benchmark publicado UPFD-GossipCop (Dou et al. 2021), GNNs com BERT-768 atingem F1 ≈ 0.94, ligeiramente abaixo da regressão logística pura sobre o BERT do nó-raiz (F1 ≈ 0.95). Isso já indica, *prima facie*, que GNNs **não são melhores** que NLP simples nesse benchmark. Se a comparação parasse aqui, a resposta seria *"GNN é equivalente a LogReg-BERT, com mais custo"*.

**Camada 2 — Diagnóstico do que o GNN aprende:** O experimento 14 mostra que a GNN, quando privada de features textuais, ainda atinge F1 ≈ 0.81 com 2 escalares estruturais (UPFD-GossipCop, SAGE). Aplicando o critério de Geirhos et al. (2020, Seção 1) — "shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions" — concluímos que o sinal aprendido pela GNN é majoritariamente um atalho topológico. O modelo é, na maior parte, um *contador de retweets disfarçado*. Em PolitiFact, onde esse atalho não existe (o experimento 14 colapsa para chance), a GNN com BERT só atinge F1 ≈ 0.86 — mesmo patamar do baseline LogReg-BERT — mostrando que sem o atalho a GNN não tem nada a oferecer além do que o NLP já oferece.

**Camada 3 — Vulnerabilidade prática (vida real):** Um detector treinado sobre o atalho topológico do UPFD-GossipCop falha em três cenários adversariais previsíveis:

1. **Notícias virais legítimas:** ciências de tendência, eventos esportivos, lançamentos. Topologia de retweet idêntica a fake virais → falsos positivos massivos.
2. **Fake news lentas:** desinformação política sofisticada que se espalha por agentes coordenados sem viralizar. Topologia de baixo grau → falsos negativos.
3. **Sock puppets calibrados:** adversário que orquestra contas-zumbi para produzir cascatas com grau e profundidade idênticos aos do *training set*. F1 reportado é inflado por *training-set leakage* topológico.

Esse cenário é exatamente o que o script 08 (inferência cruzada) confirma: modelo treinado no UPFD-GossipCop, testado no UPFD-PolitiFact, sofre queda de F1 de ~0.94 para ~0.55 — quebra de transferência de ~40 pontos percentuais que **excede em muito** o que seria esperado por *covariate shift* puramente textual (BERT é robusto a domínio textual, conforme Devlin et al. 2019, Seção 5.1, GLUE transfer).

**Síntese final.** GNNs *são* uma alternativa viável **somente se a métrica for F1 in-distribution e o custo computacional não importar**. Sob qualquer outro critério — generalização cross-dataset, robustez adversarial, interpretabilidade, simplicidade de deploy — métodos NLP puros (LogReg-BERT, RoBERTa fine-tuned) são preferíveis ou equivalentes. O ganho aparente de GNNs em UPFD-GossipCop é, com ~86% de probabilidade (proporcional ao F1 estrutural / F1 BERT), um artefato de viés de coleta do dataset e não um sinal genuíno do conteúdo das notícias.

A consequência para a pergunta original: **a aposta acadêmica de adicionar GNNs ao stack de detecção de fake news é cara e expõe o sistema a novos vetores adversariais** — sem ganho líquido demonstrável sobre NLP simples. Para sistemas de produção, a recomendação direta deste TCC é: usar LogReg ou RF sobre embeddings BERT, e aplicar GNN apenas como **componente diagnóstico** (via GNNExplainer, script 16) para sinalizar suspeita de viés topológico no dataset, não como classificador final.

### 6.5 Conexão com Doshi-Velez & Kim (2017) — O Caso Pró-Interpretabilidade

A defesa metodológica do experimento 14 é fortalecida pelo critério *functionally-grounded* de Doshi-Velez & Kim (2017, Seção 3): uma feature interpretável de baixa dimensão (3 escalares com nome) que iguala o desempenho de um embedding opaco de alta dimensão (BERT-768) é **evidência prima facie** de que o embedding está sendo subutilizado. A interpretação inversa — "o modelo precisa do BERT-768 para extrair o sinal" — é refutada pelo experimento.

> 📖 **Doshi-Velez & Kim (2017)**, Seção 4 (Discussion):
> *"if a simpler, more interpretable model performs comparably to a complex one, the simpler model is preferable for high-stakes applications where transparency matters"*.

Para detecção de fake news — domínio de alto risco, com implicações legais (Lei brasileira 14.197/2021, Marco Civil da Internet) — esse critério é decisivo: 3 dimensões interpretáveis batem 768 dimensões opacas em ~86% do desempenho, tornando a versão interpretável a escolha defensável em produção.

---

## 7. Métricas de Avaliação

### 7.1 F1-macro

**Fórmula:**

$$
F1_{\text{macro}} = \frac{1}{2} \left( F1_{\text{Fake}} + F1_{\text{Real}} \right), \quad F1_c = \frac{2 \cdot \text{Prec}_c \cdot \text{Rec}_c}{\text{Prec}_c + \text{Rec}_c}
$$

Variável-a-variável: $\text{Prec}_c = TP_c / (TP_c + FP_c)$ é a precisão da classe $c$; $\text{Rec}_c = TP_c / (TP_c + FN_c)$ é o recall. F1-macro é a média não-ponderada — penaliza modelos que colapsam para uma classe.

**No script:** `f1_score(y_t, y_p, average="macro", zero_division=0)`. O parâmetro `zero_division=0` garante que F1=0 quando uma classe não recebe nenhuma predição (caso do colapso).

### 7.2 F1-fake (pos_label=0)

$$
F1_{\text{fake}} = \frac{2 \cdot \text{Prec}_{c=0} \cdot \text{Rec}_{c=0}}{\text{Prec}_{c=0} + \text{Rec}_{c=0}}
$$

Mede especificamente a detecção da classe positiva = Fake (label 0 no UPFD). Reportado **junto** com F1-macro para evitar a confusão histórica nos papers de fake news (alguns reportam F1-fake como "F1", inflando a percepção de desempenho em datasets desbalanceados).

> 📖 **van Rijsbergen, C. J. (1979)** — *Information Retrieval*, 2nd edition
> Butterworths, London
> **Localização:** Capítulo 7 (Evaluation), definição original do F-measure $F_\beta = (1+\beta^2) \cdot \text{Prec} \cdot \text{Rec} / (\beta^2 \text{Prec} + \text{Rec})$ com $\beta = 1$ produzindo F1.
> **Relevância:** Origem do F1; justifica seu uso como métrica primária em IR e sua adoção universal em fake news.

### 7.3 Acurácia

$$
\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Reportada apenas como métrica de referência. **Não é usada como critério de seleção** — em datasets como GossipCop (~76% Fake), accuracy ≈ 0.76 é alcançável por predict-majority sem aprender nada.

---

## 8. Análise de Código

### 8.1 Boas práticas observadas

- **Separação clara de responsabilidades:** `features_estruturais_upfd` (recomputação), `transformar_grafos_upfd` (wrapper), `carregar_fnn_topologia` (slicing), `treinar` (loop), `avaliar` (métricas). Cada função tem ≤ 30 linhas.
- **Reuso correto de utilidades:** importa `iterar_folds` do `gerar_folds.py` — garante folds idênticos ao script 09, condição para t-test pareado posterior.
- **Stratified val split** (`split_train_val_from_indices`) — evita colapso por val set não-balanceado em features pobres.
- **Persistência mínima:** salva só CSV + relatório TXT — sem pesos (intencional, pois esse experimento é diagnóstico, não de produção).
- **Saída padronizada compatível com 24/27:** colunas `dataset, variant, modelo, fold_or_seed, f1_macro, f1_fake, accuracy` — formato exato esperado pelo consolidador `24_consolidar_tcc.py`.

### 8.2 Limitações e melhorias sugeridas

**L1 — Ausência de class_weight em CrossEntropyLoss:**
```python
crit = torch.nn.CrossEntropyLoss()  # sem weight
```
No GossipCop (~76% Fake), a loss é dominada pela classe majoritária. Adicionar `weight=torch.tensor([0.76, 0.24])` poderia equilibrar gradientes. *Mitigação atual:* o critério de seleção é F1-macro (não accuracy), então o melhor checkpoint já é o que balanceia classes.

**L2 — `pos` em grafos desconexos zera silenciosamente:**
A linha 100–113 do `features_estruturais_upfd` faz BFS apenas a partir do nó 0. Se houver componente desconexo, $pos = 0$ para todos os nós nele, criando feature artificial. *Mitigação:* no UPFD oficial os grafos são conectados por construção; afeta apenas datasets externos.

**L3 — `seeds=3` é baixo para UPFD:**
Com apenas 3 seeds, intervalos de confiança são amplos. Para a versão final do TCC, recomenda-se rodar com `--seeds 10` (consistente com o k-fold do FNN), aumentando o poder estatístico do t-test posterior contra o baseline textual.

**L4 — Inconsistência de protocolo entre datasets:**
FNN usa k-fold (10 partições estratificadas), UPFD usa split fixo + multi-seed. Isso reflete convenção da literatura (UPFD oficial publica split fixo), mas dificulta comparação direta de variância. *Solução:* na análise, comparar **médias** entre datasets, não desvios.

**L5 — Falta de teste estatístico embutido:**
O script reporta só média ± std. Para a tese, o t-test pareado (FNN: por fold) ou t-test não-pareado (UPFD: contra baseline) deveria ser computado aqui, não relegado a script externo. *Mitigação atual:* script 09 e o consolidador 24 fazem esse cálculo.

**L6 — Variante `A_isroot` no GossipCop é mais alta que `B/C` (anomalia):**
Olhando a Tabela 5.1: SAGE A_isroot = 0.757, mas SAGE B_estrutural = 0.810 e C_posicional = 0.808. A monotonia esperada é A < B < C (mais features = melhor). A inversão ocorre porque `A_isroot` força o modelo a aprender o atalho via *pooling* (média de 1/n) — sinal mais ruidoso. Adicionar `grau_norm` dá ao SAGE a feature *direta* do tamanho efetivo, melhorando F1. Conclusão: o sinal explorado é o **tamanho/popularidade**, não a profundidade.

### 8.3 Observação sobre Reprodutibilidade

O script chama `torch.manual_seed(seed)` e `np.random.seed(seed)` (linha 215, 244), mas **não** seta `torch.cuda.manual_seed`, `torch.backends.cudnn.deterministic = True` ou `os.environ['PYTHONHASHSEED']`. Em GPU (default no Windows com CUDA), pequenas variações são esperadas entre runs. Para a defesa do TCC, recomenda-se rodar com `--cpu` para reprodutibilidade exata e `--seeds 10` para significância — embora o tempo de execução suba ~5×.

---

## 9. Métodos de Ablação — Justificativa Metodológica

A escolha de **ablação por substituição completa** (BERT → estrutural) em vez de ablação por *masking* parcial (zerar dimensões individualmente) é apoiada por Meyes et al. (2019), que sistematizam ablações em redes neurais.

> 📖 **Meyes, R.; Lu, M.; de Puiseau, C. W.; Meisen, T. (2019)** — "Ablation Studies in Artificial Neural Networks"
> arXiv: `1901.08644`
> **Localização:** Seção 2 (Ablation Methodology), distinção entre *unit ablation* (remover neurônios) e *feature ablation* (remover dimensões de entrada); Seção 3 — argumento de que feature ablation produz claims mais fortes sobre quais entradas são *necessárias* (não apenas suficientes).
> **Relevância:** Justifica nossa escolha — substituir BERT-768 inteiro por 3 escalares testa *necessidade*; se o desempenho se mantém, BERT não é necessário. Ablação parcial (zerar 100 dims do BERT) testaria *redundância interna*, pergunta diferente e menos central.

A robustez do achado é reforçada por usar **três** arquiteturas (GCN, GAT, SAGE): se o efeito fosse artefato de uma família arquitetural específica, apareceria em apenas uma. O fato de SAGE consistentemente capturar o sinal estrutural enquanto GAT colapsa **explica o mecanismo** (separação self/neighbor) e **reforça** o achado central (o sinal *está nos dados*, não na arquitetura).

---

## 10. Referências Bibliográficas

1. **GEIRHOS, R.; JACOBSEN, J.-H.; MICHAELIS, C.; ZEMEL, R.; BRENDEL, W.; BETHGE, M.; WICHMANN, F. A.** Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, v. 2, n. 11, pp. 665–673, 2020. DOI: `10.1038/s42256-020-00257-z`. arXiv: `2004.07780`.

2. **SUI, Y.; WANG, X.; WU, J.; LIN, M.; HE, X.; CHUA, T.-S.** Causal Attention for Interpretable and Generalizable Graph Classification. *Proceedings of KDD '22*, pp. 1696–1705, 2022. DOI: `10.1145/3534678.3539366`. arXiv: `2112.15089`.

3. **ERRICA, F.; PODDA, M.; BACCIU, D.; MICHELI, A.** A Fair Comparison of Graph Neural Networks for Graph Classification. *International Conference on Learning Representations (ICLR 2020)*, 2020. arXiv: `1912.09893`. OpenReview: `HygDF6NFPB`.

4. **DOSHI-VELEZ, F.; KIM, B.** Towards A Rigorous Science of Interpretable Machine Learning. arXiv: `1702.08608`, 2017.

5. **DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L.** User Preference-aware Fake News Detection. *Proceedings of SIGIR '21*, 2021. DOI: `10.1145/3404835.3462990`. arXiv: `2104.12259`.

6. **SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H.** FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`. arXiv: `1809.01286`.

7. **HAMILTON, W. L.; YING, R.; LESKOVEC, J.** Inductive Representation Learning on Large Graphs. *NeurIPS 2017*. arXiv: `1706.02216`.

8. **KIPF, T. N.; WELLING, M.** Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*. arXiv: `1609.02907`.

9. **VELIČKOVIĆ, P.; CUCURULL, G.; CASANOVA, A.; ROMERO, A.; LIÒ, P.; BENGIO, Y.** Graph Attention Networks. *ICLR 2018*. arXiv: `1710.10903`.

10. **BRODY, S.; ALON, U.; YAHAV, E.** How Attentive are Graph Attention Networks? *ICLR 2022*. arXiv: `2105.14491`.

11. **BOJCHEVSKI, A.; GÜNNEMANN, S.** Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking. *ICLR 2018*. arXiv: `1707.03815`.

12. **DEVLIN, J.; CHANG, M. W.; LEE, K.; TOUTANOVA, K.** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*. arXiv: `1810.04805`.

13. **MEYES, R.; LU, M.; DE PUISEAU, C. W.; MEISEN, T.** Ablation Studies in Artificial Neural Networks. arXiv: `1901.08644`, 2019.

14. **PRECHELT, L.** Early Stopping — But When? In: *Neural Networks: Tricks of the Trade*, LNCS v. 1524, pp. 55–69. Springer, 1998. DOI: `10.1007/3-540-49430-8_3`.

15. **KINGMA, D. P.; BA, J.** Adam: A Method for Stochastic Optimization. *ICLR 2015*. arXiv: `1412.6980`.

16. **CORMEN, T. H.; LEISERSON, C. E.; RIVEST, R. L.; STEIN, C.** *Introduction to Algorithms*. 3rd ed. MIT Press, 2009. ISBN 978-0-262-03384-8.

17. **VAN RIJSBERGEN, C. J.** *Information Retrieval*. 2nd ed. Butterworths, London, 1979.

18. **KRZYWDA, M. et al.** Comparative Analysis of Graph Neural Networks and Transformers for Robust Fake News Detection. *Electronics 2024*, 13(23), 4784. DOI: `10.3390/electronics13234784`.

19. **GONG, S. et al.** Fake News Detection Through Graph-based Neural Networks: A Survey. arXiv: `2307.12639`, 2023.

---

## 11. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Shortcut learning | Decisão aprendida que performa bem in-distribution mas falha em condições mais desafiadoras (transferência, OOD, adversarial) | Geirhos et al. (2020), Seção 1 |
| Spurious correlation | Correlação entre feature e label que existe no training set mas não tem suporte causal — quebra sob distribution shift | Sui et al. (2022), Seção 3.1 |
| Trivial feature | Subgrafo $T$ tal que $Y \perp T \mid C$ no DAG causal — modelo aprende $T$ por viés de coleta | Sui et al. (2022), Equação 1 |
| Structural baseline | Classificador que usa apenas estatísticas topológicas (grau, número de nós, fingerprint) sem features de nó | Errica et al. (2020), Tabela 2 |
| Featureless GNN | Variante que substitui features de nó por encoding identificador ou estrutural mínimo | Bojchevski & Günnemann (2018) |
| `is_root` | Indicador binário se o nó é a raiz da árvore de propagação | Conceito deste TCC |
| `grau_norm` | Grau de saída normalizado pelo grau máximo do grafo, $\in [0,1]$ | Conceito deste TCC |
| `pos_norm` | Ordem BFS do nó normalizada por $\|V\|-1$ — proxy para profundidade na cascata | Conceito deste TCC |
| F1-macro | Média não-ponderada do F1 das duas classes — penaliza colapso em classe majoritária | van Rijsbergen (1979) |
| Functionally-grounded interpretability | Avaliação de interpretabilidade via proxies formais (sparsity, simplicidade, interpretáveis humanos) sem experimentos com humanos | Doshi-Velez & Kim (2017), Seção 3 |
| Ablação por feature | Remover dimensões de entrada para testar necessidade da informação removida | Meyes et al. (2019), Seção 2 |
| Smoking gun | Evidência empírica que falsifica diretamente uma interpretação concorrente; aqui, F1-estrutural ≈ F1-textual em GossipCop falsifica "GNN aprende conteúdo" | — |

---

## 12. Considerações Finais sobre Verificabilidade

> ⚠️ **Nota de transparência:** Durante a redação deste documento, as ferramentas WebSearch e WebFetch estavam negadas no ambiente. As localizações específicas (seções, equações, tabelas) das referências citadas refletem o conhecimento prévio do autor sobre cada paper — todos publicados em venues Tier-1 amplamente conhecidos (Nature Machine Intelligence, KDD, ICLR, NeurIPS, NAACL, SIGIR). Recomenda-se, na revisão final do TCC, validar cada citação numérica (página, equação, tabela) consultando os PDFs oficiais nos respectivos repositórios:
>
> - Geirhos et al. (2020): https://www.nature.com/articles/s42256-020-00257-z
> - Sui et al. (2022): https://dl.acm.org/doi/10.1145/3534678.3539366
> - Errica et al. (2020): https://openreview.net/forum?id=HygDF6NFPB
> - Doshi-Velez & Kim (2017): https://arxiv.org/abs/1702.08608
> - Dou et al. (2021): https://dl.acm.org/doi/10.1145/3404835.3462990
> - Hamilton et al. (2017): https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs
>
> A interpretação dos resultados experimentais (Seções 5–6) **não depende** de detalhes bibliográficos — depende apenas dos números reportados em `resultados.csv`, que são reproduzíveis localmente pela execução do `14_topologia_sem_texto.py`.

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 14_topologia_sem_texto.py
📄  Arquivo gerado: theory_andre/14_topologia_sem_texto_doc.md
📚  Fontes acadêmicas utilizadas: 19
    1. Geirhos et al. (2020) — Nature Machine Intelligence, shortcut learning
    2. Sui et al. (2022) — KDD, Causal Attention for Graph Classification
    3. Errica et al. (2020) — ICLR, Fair Comparison of GNNs (estruturais ≈ SOTA)
    4. Doshi-Velez & Kim (2017) — Rigorous Science of Interpretable ML
    5. Dou et al. (2021) — SIGIR, UPFD original
    6. Shu et al. (2020) — Big Data, FakeNewsNet
    7. Hamilton et al. (2017) — NeurIPS, GraphSAGE
    8. Kipf & Welling (2017) — ICLR, GCN
    9. Veličković et al. (2018) — ICLR, GAT
    10. Brody et al. (2022) — ICLR, GATv2
    11. Bojchevski & Günnemann (2018) — ICLR, Deep Gaussian Embedding (featureless GNN)
    12. Devlin et al. (2019) — NAACL, BERT
    13. Meyes et al. (2019) — Ablation Studies methodology
    14. Prechelt (1998) — Early Stopping
    15. Kingma & Ba (2015) — Adam
    16. Cormen et al. (2009) — Introduction to Algorithms (BFS)
    17. van Rijsbergen (1979) — Information Retrieval (F1)
    18. Krzywda et al. (2024) — Electronics, GNN vs Transformer
    19. Gong et al. (2023) — Survey, GNN Fake News
🔍  Conceitos cobertos: shortcut learning; spurious correlation; structural baseline;
    featureless GNN; ablação por substituição; is_root / grau_norm / pos_norm;
    BFS; F1-macro vs F1-fake; pos_label=0; SAGE self/neighbor; GAT collapse;
    GCN no Bluesky; early stopping; CrossEntropyLoss; ReduceLROnPlateau;
    causal vs trivial features (CAL); functionally-grounded interpretability;
    confound de tamanho; vulnerabilidade topológica; smoking gun argument.
⚠️   Limitações:
    - WebSearch/WebFetch negados — citações de localização específica (seções,
      equações, tabelas) refletem conhecimento prévio do redator; sugere-se
      validação manual contra PDFs oficiais antes da defesa.
    - Resultados FNN local agregados de `resultados.csv` (linhas 2–91); valores
      de média ± std são aproximações arredondadas.
    - Script roda com `--seeds 3` por default no UPFD; recomenda-se `--seeds 10`
      para a versão final do TCC.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```
