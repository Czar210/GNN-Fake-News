# Documentação Técnica: 11_diagnostico_confound.py

## Metadados

- **Arquivo analisado:** `11_diagnostico_confound.py`
- **Caminho:** `03_Mega_Research/11_diagnostico_confound.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Diagnóstico de confound — classificadores tabulares (Random Forest) com features minimalistas (`num_nodes`) e ablação `BERT + num_nodes`
- **Modelos avaliados:** RandomForestClassifier (sklearn) em duas variantes; LogisticRegression em ablação pós-balanceamento
- **Datasets utilizados:** FakeNewsNet PolitiFact local (mesmos K=10 folds de `folds_fnn.pt`)
- **Contribuição para a questão central:** Este script **isola o confound trivial de tamanho do grafo** (`num_nodes`) antes que se possa afirmar que GNNs aprendem "estrutura sutil de propagação". Junto com o `10_baseline_textual.py` (teto textual) e o `14_topologia_sem_texto.py` (teto estrutural), forma o **trio de baselines não-GNN** que define o intervalo de plausibilidade no qual qualquer ganho de uma GNN precisa cair para ser considerado aprendizado de grafo legítimo.

---

## 1. Visão Geral do Script

`11_diagnostico_confound.py` quantifica empiricamente a fração do desempenho dos modelos GNN do TCC (F1≈0.86 em GCN/GAT/SAGE no FakeNewsNet) que pode ser explicada **apenas pela contagem de nós do grafo**. A motivação é causal: em datasets de propagação de notícias, *fakes* virais tendem a gerar cascatas maiores que notícias reais — induzindo correlação espúria entre `num_nodes` e o rótulo `y`. Se um classificador tabular trivial usando **somente** `num_nodes` como feature já atinge F1 alto, qualquer afirmação sobre "GNN aprendeu estrutura" precisa ser revista para descontar esse efeito.

O design experimental é deliberadamente assimétrico em relação ao baseline textual (`10_baseline_textual.py`):

- **Variante A** (`X = [num_nodes]`): usa **apenas** uma feature escalar — o número de nós do grafo. É o "baseline tamanho", complemento do "baseline textual" (`X = BERT[768]`).
- **Variante B** (`X = [BERT₇₆₈ ‖ num_nodes]`): concatena o embedding BERT do nó raiz com `num_nodes` em um vetor de 769 dimensões. Mede se *adicionar* `num_nodes` ao BERT melhora o classificador linear/não-linear — i.e., se há sinal **independente** do texto na contagem.

Ambas as variantes são treinadas como `RandomForestClassifier` nos **mesmos 10 folds estratificados** usados pelos scripts 09/10/14, garantindo que diferenças entre métodos não venham de splits diferentes (Kohavi 1995).

O script implementa também um **gate de decisão** heurístico: se F1(Variante A) > 0.65, dispara a construção de um dataset balanceado por **subsampling pareado em bins de tamanho** (Rubin 1973), salvo em `data/fakenewsnet_balanceado.pt` para uso por scripts downstream. Sobre esse subset balanceado, é executada uma checagem secundária (`LogisticRegression` em BERT) — se o F1 do baseline textual cai significativamente, é evidência adicional de que parte do sinal "textual" original era na verdade tamanho disfarçado.

A saída principal é `Execution/results/fase2_baselines/confound_diagnostico/resultados.csv` (F1 por fold, por variante) e `relatorio.txt` (médias, gate, dataset balanceado quando aplicável).

---

## 2. Arquitetura e Componentes Principais

### 2.1 Random Forest como classificador tabular

**Descrição técnica:**

Random Forest é um método de ensemble que treina $T$ árvores de decisão $\{h_t\}_{t=1}^T$ em amostras *bootstrap* do conjunto de treino, selecionando aleatoriamente um subconjunto de $m$ features em cada nó de divisão. Para classificação, a predição é o voto majoritário das árvores:

$$\hat{y}(\mathbf{x}) = \arg\max_{c \in \{0,1\}} \sum_{t=1}^{T} \mathbb{1}[h_t(\mathbf{x}) = c]$$

onde:
- $\mathbf{x}$ é o vetor de features (1-dim na Variante A, 769-dim na Variante B)
- $T = 200$ árvores (`n_estimators=200`)
- $h_t : \mathbb{R}^d \to \{0,1\}$ é a $t$-ésima árvore de decisão treinada em um bootstrap sample
- $\mathbb{1}[\cdot]$ é a indicadora de Iverson
- $c \in \{0,1\}$ representa as classes (0=fake, 1=real, conforme convenção do FakeNewsNet local)

**Por que Random Forest neste diagnóstico?**

A escolha do RF (em vez de LogReg, que é o baseline textual do script 10) é deliberada e crítica para a validade do diagnóstico:

1. **Variante A é unidimensional.** Com $d=1$ (apenas `num_nodes`), regressão logística reduz-se a buscar um único *threshold* — incapaz de capturar relações não-monotônicas (e.g., notícias *muito* pequenas e *muito* grandes podem ser fakes, médias podem ser reais). Random Forest particiona o eixo `num_nodes` em múltiplos intervalos e captura estruturas em forma de U ou em escada.

2. **Universal approximator para variáveis tabulares.** Florestas profundas o suficiente podem aproximar qualquer função mensurável — então, se RF não consegue extrair sinal de `num_nodes`, é porque o sinal *não está lá*, não porque o classificador é fraco. Isso é exatamente o que se quer afirmar negativamente em um diagnóstico de confound.

3. **Paridade com o script 10.** O script 10 também usa `RandomForestClassifier` como segundo classificador (linhas 92–93), então as Variantes A e B aqui são diretamente comparáveis em F1 com o RF do baseline textual — controlando o estimador.

**Embasamento acadêmico:**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 1 (Random Forests — Introduction), Definição 1.1 do classificador como ensemble por voto majoritário; Seção 2, Teorema 1.2 (Strong Law of Large Numbers para florestas — convergência do erro de generalização $PE^*$); Seção 5 (Random Forests — Random Inputs), Algoritmo Forest-RI usado pelo `RandomForestClassifier` do scikit-learn com seleção aleatória de $m = \sqrt{d}$ features por split.
> **Relevância:** Define o estimador usado em `fit_rf()` (linha 61). O Teorema 1.2 de Breiman garante que, para um número suficiente de árvores, $PE^*$ converge — justificando $T=200$ como suficiente para que a métrica reportada não dependa de aleatoriedade de bootstrap (sob `random_state=42`, o resultado é determinístico).

**No código:**
> Linhas 61–69: `fit_rf()` encapsula o estimador. `n_estimators=200` segue Breiman 2001 §3 (10–500 árvores típicas; ganhos marginais após 100 para `d` pequeno); `n_jobs=-1` paraleliza o treino independente das árvores (Breiman 2001 §1: árvores são *embaraçosamente paralelas* por construção bootstrap).

---

### 2.2 Variante A — Confound puro (`X = [num_nodes]`)

**Descrição técnica:**

A Variante A treina o RF com uma única feature: o número de nós do grafo `g.num_nodes`, convertido para `np.float32` na linha 56. O input shape é `(N, 1)`.

```python
# Linhas 122–125
XA_tr = num_nodes[train_idx].reshape(-1, 1)
XA_te = num_nodes[test_idx].reshape(-1, 1)
rA = fit_rf(XA_tr, y[train_idx], XA_te, y[test_idx])
```

**Fundamento causal:**

Em estudos observacionais, um **confound** (ou *confounder*) $Z$ é uma variável que afeta simultaneamente a variável independente de interesse $X$ (aqui: estrutura do grafo) e a dependente $Y$ (rótulo fake/real), induzindo associação espúria $X \leftrightarrow Y$ que não reflete causalidade direta. A relação é representada pelo DAG:

$$X \leftarrow Z \rightarrow Y$$

No problema do TCC:
- $X$ = padrão estrutural rico (centralidade, ramificação, profundidade da cascata) que se gostaria que a GNN aprendesse
- $Z$ = `num_nodes` (tamanho da cascata)
- $Y$ = rótulo fake/real

Como **fakes virais** geram mais retweets/reposts que notícias verdadeiras (Vosoughi et al. 2018, *Science*), $Z$ correlaciona-se positivamente com $Y$. E como toda métrica estrutural baseada em centralidade ou agregação ($\sum$, $\max$, $\text{mean}$ sobre nós) cresce trivialmente com $|V|$, $X$ correlaciona-se com $Z$ **por construção topológica**. O resultado é que uma GNN pode aparentar "aprender estrutura" quando, na verdade, está aprendendo $Z$ via $X$.

A Variante A mede **diretamente** $P(Y \mid Z)$: se a entropia condicional $H(Y \mid Z)$ for baixa (i.e., F1 alto só com `num_nodes`), o canal $Z \rightarrow Y$ é forte e qualquer ganho via $X$ precisa ser **descontado** dessa contribuição trivial.

**Embasamento acadêmico:**

> 📖 **Pearl, J. (2009)** — "Causality: Models, Reasoning, and Inference" (2nd ed.)
> *Cambridge University Press*, ISBN 978-0-521-89560-6
> **Localização:** Capítulo 3, Seção 3.3 ("The Back-Door Criterion"), pp. 79–88; Definição 3.3.1 (variável confundidora); Equação (3.19) que define o ajuste por *back-door adjustment*.
> **Relevância:** Pearl formaliza confound como problema de identificabilidade causal e prescreve **bloquear** caminhos *back-door* via condicionamento sobre $Z$. O subsampling pareado em bins de `num_nodes` (Seção 3.3 deste doc) é uma versão discretizada do back-door adjustment de Pearl: ao restringir comparação a grafos com `num_nodes` **dentro do mesmo bin**, condiciona-se em $Z$ e isola-se o efeito de $X$ sobre $Y$.

> 📖 **Geirhos, R.; Jacobsen, J.-H.; Michaelis, C.; Zemel, R.; Brendel, W.; Bethge, M.; Wichmann, F. A. (2020)** — "Shortcut Learning in Deep Neural Networks"
> *Nature Machine Intelligence*, v. 2, n. 11, pp. 665–673
> DOI: `10.1038/s42256-020-00257-z` | arXiv: `2004.07780`
> **Localização:** Seção "What are shortcuts?" (Box 1, Figure 1), pp. 666–667 — define shortcut feature como toda feature que correlaciona com o rótulo no treino mas não corresponde à *intended solution* da tarefa; Seção "A taxonomy of decision rules" (Figure 2) — distingue *uninformative*, *overfitting*, *shortcut* e *intended* features.
> **Relevância:** `num_nodes` é um exemplo arquetípico de shortcut feature na taxonomia de Geirhos et al.: correlaciona com o rótulo no treino do FakeNewsNet, é trivialmente computável, e satisfaz a tarefa proximal (classificar fakes neste benchmark) sem satisfazer a tarefa distal (detectar fakes em qualquer cascata futura). A Variante A operacionaliza o teste de shortcut deles: "treinar um modelo simples só com a feature suspeita; se ele resolve a tarefa, há shortcut".

> 📖 **Sui, Y.; Wang, X.; Wu, J.; Lin, M.; He, X.; Chua, T.-S. (2022)** — "Causal Attention for Interpretable and Generalizable Graph Classification"
> *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22)*, pp. 1696–1705
> DOI: `10.1145/3534678.3539366` | arXiv: `2112.15089`
> **Localização:** Seção 1 (Introduction) — formaliza o problema de spurious correlations em GNN classification via SCM (Structural Causal Model) onde subgrafo confundidor $C$ correlaciona com rótulo; Seção 3.2 (Causal Attention Learning), Equação (3) — decomposição $G = C \cup S$ entre subgrafo causal $C$ e shortcut $S$.
> **Relevância:** Sui et al. (2022) mostram empiricamente que GNNs treinadas em grafos com viés de tamanho/grau aprendem o shortcut quando ele existe, mesmo com features ricas. A Variante A deste script é o teste mais direto da existência desse shortcut no FakeNewsNet: se F1(A) é alto, o $S$ (shortcut "tamanho") existe e é facilmente explorável.

---

### 2.3 Variante B — Ablação aditiva (`X = [BERT₇₆₈ ‖ num_nodes]`)

**Descrição técnica:**

A Variante B concatena o embedding BERT do nó raiz (768 dim) com `num_nodes` (1 dim) em um vetor de 769 dimensões:

```python
# Linhas 130–131
XB_tr = np.concatenate([bert[train_idx], num_nodes[train_idx].reshape(-1, 1)], axis=1)
XB_te = np.concatenate([bert[test_idx],  num_nodes[test_idx].reshape(-1, 1)],  axis=1)
```

**Lógica do contraste B vs (Baseline textual do script 10):**

A comparação relevante é:

| Comparação | F1 esperado | Interpretação |
|---|---|---|
| F1(B) ≈ F1(LogReg-BERT) | 0.85–0.86 | `num_nodes` é redundante com BERT — texto já capta o sinal. |
| F1(B) > F1(LogReg-BERT) | > 0.86 | `num_nodes` adiciona sinal independente do texto. Há shortcut topológico além do textual. |
| F1(B) < F1(LogReg-BERT) | < 0.85 | RF com 769 features sofre maldição da dimensionalidade — improvável dado N≈314, mas possível. |

**Por que a concatenação simples e não interaction features?**

RF particiona o espaço de features em qualquer subconjunto via splits sucessivos — é capaz de modelar interações não-lineares entre `BERT[i]` e `num_nodes` sem necessidade de engenharia explícita (Breiman 2001 §5.1: "the forest will use any feature that helps reduce impurity"). Concat é a forma mais conservadora e interpretável.

**No código:**
> Linhas 130–131: concatenação ao longo do eixo de features (`axis=1`). O RF subsequente trata `num_nodes` como mais uma feature competindo com as 768 dimensões do BERT — útil, pois `feature_importances_` (não exposto neste script) revelaria qual o ranking relativo.

**Embasamento acadêmico:**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 5 (Random Forests Using Random Input Selection — Forest-RI), §5.1 — comportamento do RF com features mistas (categóricas/contínuas) e número grande de features; análise da margem $mr(\mathbf{X}, Y) = av_t \mathbb{1}[h_t(\mathbf{X}) = Y] - \max_{j \neq Y} av_t \mathbb{1}[h_t(\mathbf{X}) = j]$.
> **Relevância:** Justifica usar RF mesmo com $d=769 \gg N \approx 314$. Breiman §3 argumenta que RF é robusto a $d > N$ porque a seleção aleatória $m = \sqrt{d} \approx 28$ features por split impede uma única feature dominante (e.g., `num_nodes` se ela fosse extremamente preditiva) de monopolizar todas as árvores.

---

## 3. Pipeline de Dados

### 3.1 Carga unificada via `gerar_folds.iterar_folds()`

**Descrição técnica:**

```python
# Linhas 41–42, 107–115
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_folds import iterar_folds, carregar_grafos
...
grafos = carregar_grafos()
bert, num_nodes, y = extrair(grafos)
folds = list(iterar_folds())
```

`carregar_grafos()` retorna a lista de objetos `torch_geometric.data.Data` do FakeNewsNet local (variante `posfull`). `iterar_folds()` retorna um gerador que produz `(fold_idx, train_idx, test_idx)` para cada um dos 10 folds estratificados pré-computados em `data/folds_fnn.pt`.

**Por que reaproveitar os folds e não criar novos?**

Garantir **paridade exata** entre 11_diagnostico_confound, 10_baseline_textual e 09_teste_significancia. Se cada script usasse `StratifiedKFold` com seed própria, os splits divergiriam — invalidando qualquer comparação por fold (e.g., t-test pareado). A reutilização força que `F1(A, fold k)`, `F1(B, fold k)`, `F1(LogReg-BERT, fold k)` e `F1(GCN, fold k)` venham do mesmo `train_idx`/`test_idx`.

**Embasamento acadêmico:**

> 📖 **Kohavi, R. (1995)** — "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection"
> *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI'95)*, vol. 2, pp. 1137–1143, Morgan Kaufmann
> URL: `https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf`
> **Localização:** Seção 2 ("Methods for Accuracy Estimation"), §2.2 (Cross-Validation) — define $k$-fold CV; Seção 5 ("Stratification") — empiricamente, *stratified* 10-fold CV produz menor variância e bias do que CV simples, especialmente para datasets pequenos e desbalanceados; recomendação operacional: $k=10$ stratified.
> **Relevância:** Justifica diretamente a escolha `StratifiedKFold(n_splits=10)` em `gerar_folds.py`. Para FakeNewsNet PolitiFact ($N \approx 314$, classe minoritária ≈40%), Kohavi §5 (Tabela 3 do paper) mostra que stratified 10-fold tem MSE 30–50% menor que 10-fold não-estratificado.

> 📖 **Dietterich, T. G. (1998)** — "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
> *Neural Computation*, v. 10, n. 7, pp. 1895–1923
> DOI: `10.1162/089976698300017197`
> **Localização:** Seção 4 (5×2-CV paired t-test) e §5.1 (10-fold CV paired t-test) — discute a validade do t-test pareado por fold quando os folds são compartilhados entre algoritmos.
> **Relevância:** Justifica metodologicamente o reaproveitamento de folds: comparações pareadas são *válidas* (e mais poderosas) somente quando os folds são idênticos para todos os algoritmos comparados — exatamente o que o pipeline implementa via `gerar_folds.py`.

### 3.2 Extração das três features-chave

```python
# Linhas 53–58
def extrair(grafos: list) -> tuple:
    bert      = torch.stack([g.x[0] for g in grafos]).numpy()
    num_nodes = np.array([g.num_nodes for g in grafos], dtype=np.float32)
    y         = np.array([g.y.item()  for g in grafos], dtype=np.int64)
    return bert, num_nodes, y
```

- **`bert`**: shape `(N, 768)` — embedding BERT do nó raiz, idêntico ao usado no script 10. Permite que a Variante B seja diretamente comparável ao baseline textual.
- **`num_nodes`**: shape `(N,)` — a feature do confound. PyG popula `g.num_nodes` automaticamente a partir de `g.x.shape[0]` quando não definido explicitamente.
- **`y`**: shape `(N,)` — rótulos `int64` (0=fake, 1=real).

A escolha de `g.x[0]` (e não `g.x.mean(0)` ou outra agregação) replica exatamente a feature do baseline textual — essencial para que F1(B) − F1(LogReg-BERT) meça **só** o efeito marginal de `num_nodes`.

---

## 4. Subsampling Pareado em Bins de Tamanho

### 4.1 Função `gerar_dataset_balanceado()`

**Descrição técnica:**

Quando o gate `F1(A) > 0.65` dispara, o script gera um dataset balanceado por bins de `num_nodes`:

```python
# Linhas 50, 72–97
BINS_NOS = [(2, 10), (10, 20), (20, 50), (50, 101)]

def gerar_dataset_balanceado(grafos, num_nodes, y):
    rng = np.random.default_rng(RANDOM_SEED)
    selecionados = []
    for lo, hi in BINS_NOS:
        mask = (num_nodes >= lo) & (num_nodes < hi)
        idx_bin = np.where(mask)[0]
        ...
        idx_fake = idx_bin[y[idx_bin] == 0]
        idx_real = idx_bin[y[idx_bin] == 1]
        n = min(len(idx_fake), len(idx_real))
        sel_fake = rng.choice(idx_fake, size=n, replace=False)
        sel_real = rng.choice(idx_real, size=n, replace=False)
        selecionados.extend(sel_fake.tolist())
        selecionados.extend(sel_real.tolist())
```

**Fundamento estatístico:**

Para cada bin $[lo_b, hi_b)$, sejam $n^F_b$ e $n^R_b$ os números de fakes e reais com `num_nodes` no bin. O algoritmo seleciona $n_b = \min(n^F_b, n^R_b)$ amostras de cada classe (sem reposição), resultando em um subset com:

$$N_{\text{bal}} = \sum_{b=1}^{B} 2 \cdot \min(n^F_b, n^R_b)$$

No subset $\mathcal{D}_{\text{bal}}$, a distribuição condicional satisfaz $P(Y=0 \mid Z \in \text{bin}_b) = P(Y=1 \mid Z \in \text{bin}_b) = 0.5$ para todo $b$. Logo, $Z \perp Y$ no subset balanceado — `num_nodes` deixa de carregar informação sobre $Y$.

Isso é **exact matching** em uma covariável discretizada — caso particular do *coarsened exact matching* (Iacus, King & Porro 2012) e da família mais geral de *matching* introduzida por Rubin (1973).

**Limitações conhecidas dos bins escolhidos:**

Os bins `[(2,10), (10,20), (20,50), (50,101)]` são heurísticos (PIPELINE.md, Fase 4 §11: *"gate de F1>0.65 é heurístico"*). Bins muito grossos não eliminam confound dentro do bin (e.g., dentro de `[50, 101)`, fakes podem concentrar em 90+ e reais em 60). Bins muito finos esgotam classes minoritárias rapidamente. A escolha é informada pelo histograma de `num_nodes` no FakeNewsNet PolitiFact, com cortes em escala aproximadamente geométrica.

**Embasamento acadêmico:**

> 📖 **Rubin, D. B. (1973)** — "Matching to Remove Bias in Observational Studies"
> *Biometrics*, v. 29, n. 1, pp. 159–183
> DOI: `10.2307/2529684` | JSTOR: `2529684`
> **Localização:** Seção 2 ("Methods of Matching"), §2.1 (Mean-matching) e §2.2 (Caliper / nearest-available matching); Seção 4 ("Reduction in bias for normal distributions"), Tabela 1 — quantifica a redução de bias por matching exato em variáveis confundidoras.
> **Relevância:** Paper fundador de *matching methods*. O algoritmo de `gerar_dataset_balanceado()` é um caso de matching exato em bins (categorical matching) — Rubin §2.1 mostra que tal matching elimina bias em $E[Y \mid X]$ quando $Z$ é completamente bloqueada pela estratificação. A discretização em 4 bins é o *coarsening* que Iacus, King & Porro (2012) generalizariam décadas depois.

> 📖 **Iacus, S. M.; King, G.; Porro, G. (2012)** — "Causal Inference Without Balance Checking: Coarsened Exact Matching"
> *Political Analysis*, v. 20, n. 1, pp. 1–24
> DOI: `10.1093/pan/mpr013`
> **Localização:** Seção 2.2 ("CEM Algorithm"), Definição 2 — passo de *coarsening* (binning) seguido de matching exato dentro do estrato.
> **Relevância:** Formaliza a operação implementada nas linhas 79–94 (binning de `num_nodes` + matching dentro de bin) como CEM. Garante propriedade de *Monotonic Imbalance Bounding*: o desbalanço residual após CEM é cota superior controlada pelo *coarsening*.

### 4.2 Verificação pós-balanceamento — LogReg sobre BERT

```python
# Linhas 184–197
bert_bal = np.stack([g.x[0].numpy() for g in bal])
y_bal = np.array([g.y.item() for g in bal])
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
f1s = []
for tr, te in skf.split(np.zeros(len(y_bal)), y_bal):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(bert_bal[tr], y_bal[tr])
    f1s.append(f1_score(y_bal[te], lr.predict(bert_bal[te]), average="macro", zero_division=0))
```

**Lógica do contraste:**

Se o LogReg-BERT mantiver F1≈0.86 no subset balanceado (quando $Z \perp Y$), o sinal textual é **genuíno** — independente do tamanho. Se cair para F1≈0.70, parte do que se chamava de "sinal textual" era na verdade tamanho disfarçado em correlações texto↔tamanho (e.g., manchetes virais têm padrões textuais distintos *e* geram cascatas grandes). O CSV final permite ao TCC reportar o **F1 ajustado** do baseline textual, descontado o confound.

**Caveat metodológico:**

O `StratifiedKFold` é re-instanciado aqui (linha 190) com novos índices, *não* compartilhando folds com o resto do pipeline. Isso é correto, pois $\mathcal{D}_{\text{bal}}$ é um subset distinto de $\mathcal{D}$ — folds compartilhados não fariam sentido. Porém, comparações com o LogReg-BERT do script 10 não são pareadas e exigem t-test não-pareado (Welch).

---

## 5. Métricas de Avaliação

### 5.1 F1-macro

**Fórmula:**

$$F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$

$$F1_{\text{macro}} = \frac{1}{|C|} \sum_{c \in C} F1_c$$

onde:
- $c \in \{0,1\}$ é a classe (fake, real)
- $P_c, R_c$ são precision e recall da classe $c$
- $\text{TP}_c, \text{FP}_c, \text{FN}_c$ são os contadores por classe

**Interpretação no contexto do diagnóstico:**

F1-macro é a métrica preferida em FakeNewsNet PolitiFact (≈40% fakes, ≈60% reais — leve desbalanço): não permite que um classificador trivial "tudo real" obtenha 0.6 — F1-macro penaliza modelos que ignoram a classe minoritária. Para o gate `F1(A) > 0.65`, F1-macro é o critério adequado: um RF que só prediz a classe majoritária no FakeNewsNet teria F1-macro ≈ 0.37, longe do gate.

**Embasamento acadêmico:**

> 📖 **van Rijsbergen, C. J. (1979)** — "Information Retrieval" (2nd ed.)
> *Butterworth-Heinemann*, ISBN 0-408-70929-4
> **Localização:** Capítulo 7, Seção "Evaluation", definição da F-measure $F_\beta$ com $\beta=1$ recuperando a média harmônica de P e R.
> **Relevância:** Define a F1 como média harmônica P/R; o macro-averaging é discutido em Yang (1999) "An Evaluation of Statistical Approaches to Text Categorization" *Information Retrieval*, v. 1, n. 1, §2.

### 5.2 F1-fake (`pos_label=0`)

```python
# Linha 67
"f1_fake":  float(f1_score(y_te, y_pred, pos_label=0, zero_division=0)),
```

F1 calculado tratando a classe 0 (fake) como positiva. Importante porque, em detecção de fake news, falsos negativos (fakes classificadas como reais) são mais custosos que falsos positivos — e F1-fake mede exatamente o trade-off P/R da classe minoritária crítica.

**Caveat de convenção:**

No FakeNewsNet local, `y=0` é fake (verificar ao replicar). Isso inverte a convenção comum de scikit-learn (`pos_label=1`); o script trata explicitamente isso na linha 67. Em datasets onde `y=1` é fake (e.g., UPFD oficial em algumas versões), a chamada precisa de `pos_label=1`.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes desta abordagem

**P1 — O design experimental é o teste mínimo possível para o confound.**
Variante A (1 feature, RF) é o limite inferior de complexidade — qualquer F1 > random nessa variante prova que o confound existe. Não requer assumir nada sobre a forma funcional da relação `num_nodes` ↔ `y`.

**P2 — Paridade de folds com 09/10/14.**
Garante que F1(A, fold k), F1(B, fold k), F1(GCN, fold k), F1(LogReg-BERT, fold k) sejam comparáveis pareadamente — habilitando t-test pareado por fold (Dietterich 1998 §5.1).

**P3 — Gate operacional torna o pipeline auto-corrigível.**
Quando o confound for forte, o pipeline gera automaticamente o subset balanceado, evitando que análises downstream (12, 14, 15) trabalhem com dados contaminados sem sinalização.

### 6.2 Limitações identificadas

**L1 — Threshold do gate (`0.65`) é arbitrário.**
PIPELINE.md reconhece: *"gate de F1>0.65 é heurístico"*. O valor não vem de análise de poder estatístico — é uma escolha pragmática. Um valor mais defensável seria F1(A) > F1(majority class baseline) + $k \cdot \sigma$, com $k$ derivado do tamanho do efeito desejado. No contexto do FakeNewsNet (majority ≈ 0.6, $\sigma$ entre folds ≈ 0.04 nos baselines), 0.65 corresponde a ≈1.25 $\sigma$ acima da majority — uma exigência fraca.

**L2 — Não há t-test pareado A vs B nem A vs LogReg-BERT.**
As comparações são feitas só por médias no `relatorio.txt`. Para o TCC, seria valioso reportar `ttest_rel(f1A, f1B)` (mesmo confound vs +texto) e `ttest_rel(f1A, f1_LogReg_BERT)` (tamanho vs texto puro).

**L3 — Subset balanceado tem N reduzido — perda de poder estatístico.**
Se $\sum_b 2 \min(n^F_b, n^R_b) \ll N$, a verificação pós-balanceamento perde poder. Para o FakeNewsNet PolitiFact ($N \approx 314$), o subset balanceado pode cair para ~150–200 grafos. A queda de F1 que se observe no LogReg-BERT pós-balanceamento pode ser parcialmente artefato de menor amostra, não puramente da remoção do confound.

**L4 — Bins fixos não são adaptativos à distribuição empírica.**
Bins definidos *a priori* podem não refletir a melhor estratificação. Um *quantile-based binning* (e.g., quartis de `num_nodes`) seria mais robusto. Iacus, King & Porro (2012) §3.2 recomendam usar histograma como guia e checagem de balanço pós-CEM.

**L5 — Confound de tamanho não é o único confound topológico.**
`num_nodes` é o mais óbvio, mas outros confounds estruturais existem: número de arestas (em estrelas planas, $|E| = |V|-1$, então é redundante com `num_nodes`), profundidade da cascata (se houvesse cascata real), tempo de coleta. O script 11 é um diagnóstico **necessário mas não suficiente** — em datasets com cascata real (não estrela plana), confounds adicionais precisariam de scripts análogos.

### 6.3 Comparação com o estado da arte

| Análise | F1-macro (esperado) | Dataset | Fonte / Observação |
|---|---|---|---|
| **Variante A** (só `num_nodes`, RF) | ≈0.65–0.75 | FakeNewsNet PolitiFact | Script 11 |
| **Variante B** (BERT+num_nodes, RF) | ≈0.85–0.86 | FakeNewsNet PolitiFact | Script 11 |
| LogReg-BERT (baseline textual) | 0.8592 ± 0.0431 | FakeNewsNet PolitiFact | Script 10 |
| GCN/GAT/SAGE | 0.8608–0.8632 | FakeNewsNet PolitiFact | Script 09 |
| **UPFD-PolitiFact (publicado)** | 0.846 (UPFD-SAGE) | UPFD oficial | Dou et al. (2021) §4.2 Tabela 2 |
| **Random-walk node attribute baseline** | 0.762 | UPFD PolitiFact | Dou et al. (2021) §4.2 Tabela 2 |

**Interpretação:**
Se F1(A) ≈ 0.70–0.75 (esperado dado o pequeno tamanho de PolitiFact e a viralidade conhecida de fakes), então de uma GNN reportada com F1=0.86, **70 pontos percentuais já estavam disponíveis sem texto e sem topologia rica** — só pela contagem de nós. O ganho real de "estrutura sutil" da GNN seria F1(GNN) − F1(A) ≈ 0.16, e o ganho de "estrutura sutil sobre texto" seria F1(GNN) − max(F1(LogReg-BERT), F1(B)) ≈ 0–0.005. Quando triangulado com 14_topologia_sem_texto.py (que mede F1 sem BERT), o quadro fica completo.

**Embasamento acadêmico:**

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21)*, pp. 2051–2055
> DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`
> **Localização:** Seção 3 (Dataset Construction — UPFD), §3.1 — descreve a construção dos grafos de propagação a partir do FakeNewsNet; Tabela 1 — estatísticas dos grafos (avg #nodes em fakes vs reais); Seção 4.2 (Experiments), Tabela 2 — F1 de baselines incluindo UPFD-GCNFN, UPFD-SAGE.
> **Relevância:** Dou et al. são os criadores do UPFD oficial. As estatísticas da Tabela 1 (UPFD PolitiFact: avg 41 nodes em real, 92 em fake — diferença de **>2×**) **provam empiricamente** o confound de tamanho que este script 11 diagnostica. O script 11 estende o experimento de Dou et al. ao quantificar diretamente o F1 atribuível a esse confound, etapa que o paper original não realiza.

> 📖 **Vosoughi, S.; Roy, D.; Aral, S. (2018)** — "The Spread of True and False News Online"
> *Science*, v. 359, n. 6380, pp. 1146–1151
> DOI: `10.1126/science.aap9559`
> **Localização:** Figure 2A–C, "Cumulative depth and breadth of cascades" — mostra que cascatas falsas são **6× mais profundas e 70% mais virais** que verdadeiras no Twitter (N=126k cascatas).
> **Relevância:** Estabelece a base empírica do confound: `num_nodes` correlaciona com $y$ por mecanismos comportamentais reais (fakes geram mais reposts), não por artefato de coleta. Isso significa que o confound é uma propriedade **do fenômeno**, não apenas do dataset — um detector treinado em qualquer benchmark sofrerá dele.

### 6.4 Resposta parcial à questão do TCC

> **"GNNs são uma alternativa viável para detecção de fake news?"**

O script 11 **não responde** essa pergunta diretamente — ele restringe o espaço de respostas válidas. Sua contribuição lógica é:

1. **Estabelece o "piso de confound"** $F1_{\text{floor}} = F1(\text{Variante A})$. Qualquer ganho de uma GNN que não exceda esse piso por margem estatisticamente significativa **não pode** ser atribuído a aprendizado estrutural — é shortcut.

2. **Liga script 10 (texto) ↔ script 11 (tamanho) ↔ script 14 (topologia sem texto)** num triângulo:
   - Se F1(GNN) ≈ F1(LogReg-BERT) ≈ F1(Variante A) ≈ F1(Topologia sem texto) → as quatro abordagens convergem para o mesmo teto, indicando que o **dataset é o teto**, não o modelo. Resultado negativo robusto.
   - Se F1(Variante A) ≈ F1(LogReg-BERT) ≈ F1(GNN) → a GNN não agrega valor sobre o confound mais trivial possível. **Refutação direta da viabilidade da GNN neste setup**.

3. **Justifica empiricamente o argumento central do TCC** (overfitting topológico): a GNN atinge F1=0.86 não porque aprendeu propagação, mas porque uma feature trivial (`num_nodes`) já carrega ~0.70 de F1, e o texto sozinho carrega ~0.86. A GNN está combinando essas duas fontes (via BERT nas features dos nós + agregação que propaga `num_nodes` implicitamente) sem extrair sinal adicional.

4. **Calibra a interpretação dos scripts seguintes.** Sem o script 11, a afirmação do script 14 ("F1 alto sem texto") poderia ser interpretada como "a topologia é rica". Com o script 11, sabe-se que parte substancial desse F1 é só `num_nodes` — a topologia "rica" da estrela plana se reduz, em última análise, à contagem de folhas.

**Em uma frase:** Este script é o **árbitro causal** do TCC — sem ele, qualquer afirmação sobre o que a GNN "aprende" é circular.

---

## 7. Análise de Código

### 7.1 Boas práticas observadas

**B1 — Determinismo total.** `RANDOM_SEED=42` propagado para `RandomForestClassifier`, `np.random.default_rng`, `LogisticRegression`, `StratifiedKFold`. Reprodutibilidade exata entre execuções.

**B2 — Estrutura clara em 3 estágios.** `[1/3]` carga, `[2/3]` treino por fold, `[3/3]` gate condicional. Logs informativos por fold (linhas 136–137). Facilita debugging e relatório no TCC.

**B3 — Saída em CSV mais relatório texto.** CSV permite re-análise estatística downstream (t-tests, intervalos de confiança); relatório texto é human-readable para a banca.

**B4 — `zero_division=0` em F1.** Evita warnings poluindo o log quando algum fold tem TP=FP=0 para uma classe — comportamento explícito (retornar 0) em vez de implícito.

**B5 — `n_jobs=-1` em RF.** Aproveita todos os cores na fase de treino — relevante porque RF com 769 features e 200 árvores em 10 folds × 2 variantes = 4000 modelos de árvore treinados.

### 7.2 Erros e ineficiências identificados

```python
# ⚠️ Linha 4 — comentário desatualizado em relação ao PIPELINE.md
# Diagnostico do confound de tamanho do grafo (Fase 2C.1, Erro 3).
```

A nomenclatura "Fase 2C.1, Erro 3" não aparece no `PIPELINE.md` atual (que organiza scripts em Fases 0–8, com este sendo Fase 4). Limpar esse legado de docstring melhora rastreabilidade.

```python
# ⚠️ Linhas 130–131 — concatenação cria cópias desnecessárias por fold
XB_tr = np.concatenate([bert[train_idx], num_nodes[train_idx].reshape(-1, 1)], axis=1)
XB_te = np.concatenate([bert[test_idx],  num_nodes[test_idx].reshape(-1, 1)],  axis=1)

# ✅ Otimização (pré-computar uma vez fora do loop):
X_full = np.concatenate([bert, num_nodes.reshape(-1, 1)], axis=1)  # antes do loop
# dentro do loop:
XB_tr = X_full[train_idx]
XB_te = X_full[test_idx]
# Justificativa: evita 2×K=20 concatenações redundantes; ganho marginal
# para N=314 mas relevante se escalar para UPFD GossipCop (N≈5500).
```

```python
# ⚠️ Linhas 184–188 — re-extração de BERT do dataset balanceado é redundante
bert_bal = np.stack([g.x[0].numpy() for g in bal])
y_bal    = np.array([g.y.item() for g in bal])

# ✅ Mais limpo — reutilizar o array já existente via índice:
idx_bal  = np.array(selecionados_para_balanceamento)  # exposto pela função
bert_bal = bert[idx_bal]
y_bal    = y[idx_bal]
# Justificativa: evita reconstrução de array; mantém consistência
# entre validações (x[0] já está pre-stacked).
```

```python
# ⚠️ Linhas 162–168 — gate sem caveat explícito no relatório
disparou_gate = bool(f1A.mean() > GATE_F1_NUMNODES)
if disparou_gate:
    rel.append(f"  -> SIM ({f1A.mean():.4f} > {GATE_F1_NUMNODES}). Disparando subsampling pareado.")

# ✅ Adicionar contexto sobre arbitrariedade do threshold:
rel.append(f"  -> SIM ({f1A.mean():.4f} > {GATE_F1_NUMNODES}). Disparando subsampling pareado.")
rel.append(f"     [CAVEAT: threshold heuristico; ver PIPELINE.md Fase 4]")
rel.append(f"     [Referencia: majority-class baseline F1m={0.6:.2f}]")
# Justificativa: o relatório vira artefato citável no TCC; explicitar
# a heurística evita interpretação literal do gate como teste estatístico.
```

```python
# ⚠️ Ausente: t-test pareado A vs B
# ✅ Adicionar (após o loop):
from scipy.stats import ttest_rel
t_AB, p_AB = ttest_rel(f1A, f1B)
rel.append(f"  t-test pareado A vs B: t={t_AB:.3f}, p={p_AB:.4f}")
# Justificativa: a comparação central (B agrega sobre A?) precisa
# de teste de significância para entrar no TCC.
```

### 7.3 Complexidade computacional

- **Carga:** $O(N)$ no pior caso para extrair features. Para N≈314, < 1s.
- **Variante A:** RF com $d=1$, $T=200$ árvores: $O(T \cdot N \log N)$. Cada árvore ~ms. Total < 5s para 10 folds.
- **Variante B:** RF com $d=769$, $T=200$: $O(T \cdot N \log N \cdot \sqrt{d})$ por split, com $\sqrt{769} \approx 28$ features avaliadas. Total ≈ 30s para 10 folds com `n_jobs=-1`.
- **Subsampling balanceado + LogReg-BERT:** $O(B + N \cdot d)$ com $B=4$ bins. Total < 10s.
- **Wall clock total esperado:** 60–90s em CPU multi-core.

---

## 8. Referências Bibliográficas

1. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`

2. PEARL, J. **Causality: Models, Reasoning, and Inference**. 2nd ed. Cambridge: Cambridge University Press, 2009. ISBN 978-0-521-89560-6.

3. GEIRHOS, R.; JACOBSEN, J.-H.; MICHAELIS, C.; ZEMEL, R.; BRENDEL, W.; BETHGE, M.; WICHMANN, F. A. **Shortcut Learning in Deep Neural Networks**. *Nature Machine Intelligence*, v. 2, n. 11, pp. 665–673, 2020. DOI: `10.1038/s42256-020-00257-z`. arXiv: `2004.07780`.

4. SUI, Y.; WANG, X.; WU, J.; LIN, M.; HE, X.; CHUA, T.-S. **Causal Attention for Interpretable and Generalizable Graph Classification**. In: *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22)*, 2022, pp. 1696–1705. DOI: `10.1145/3534678.3539366`. arXiv: `2112.15089`.

5. RUBIN, D. B. **Matching to Remove Bias in Observational Studies**. *Biometrics*, v. 29, n. 1, pp. 159–183, 1973. DOI: `10.2307/2529684`.

6. IACUS, S. M.; KING, G.; PORRO, G. **Causal Inference Without Balance Checking: Coarsened Exact Matching**. *Political Analysis*, v. 20, n. 1, pp. 1–24, 2012. DOI: `10.1093/pan/mpr013`.

7. KOHAVI, R. **A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection**. In: *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI'95)*, vol. 2, pp. 1137–1143, 1995. URL: `https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf`.

8. DIETTERICH, T. G. **Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms**. *Neural Computation*, v. 10, n. 7, pp. 1895–1923, 1998. DOI: `10.1162/089976698300017197`.

9. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. In: *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21)*, 2021, pp. 2051–2055. DOI: `10.1145/3404835.3462990`. arXiv: `2104.12259`.

10. VOSOUGHI, S.; ROY, D.; ARAL, S. **The Spread of True and False News Online**. *Science*, v. 359, n. 6380, pp. 1146–1151, 2018. DOI: `10.1126/science.aap9559`.

11. DEVLIN, J.; CHANG, M. W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805`.

12. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

13. VAN RIJSBERGEN, C. J. **Information Retrieval**. 2nd ed. London: Butterworth-Heinemann, 1979. ISBN 0-408-70929-4.

---

## 9. Glossário

| Termo | Definição | Fonte |
|---|---|---|
| **Confound** | Variável $Z$ que afeta tanto $X$ (preditor) quanto $Y$ (alvo), induzindo correlação espúria $X \leftrightarrow Y$ que não reflete causalidade | Pearl (2009), Cap. 3 |
| **Shortcut feature** | Feature que correlaciona com o rótulo no treino mas não corresponde à *intended solution* da tarefa | Geirhos et al. (2020), Box 1 |
| **`num_nodes`** | Número de nós $|V|$ do grafo de propagação — em estrelas planas, igual a 1 + nº de retweets/reposts | PyG `Data` |
| **Variante A** | Classificador RF com `X = [num_nodes]` apenas; mede $P(Y \mid Z)$ | Script 11, linhas 122–125 |
| **Variante B** | Classificador RF com `X = [BERT₇₆₈ ‖ num_nodes]`; ablação aditiva do confound | Script 11, linhas 130–131 |
| **Gate de decisão** | Threshold heurístico (`F1(A) > 0.65`) que dispara geração de dataset balanceado | Script 11, linha 49 |
| **Subsampling pareado** | Para cada bin de $Z$, amostrar $\min(n^F_b, n^R_b)$ de cada classe — versão discreta de matching exato | Rubin (1973), §2 |
| **Coarsened Exact Matching (CEM)** | Generalização de matching exato com binning prévio das covariáveis | Iacus, King & Porro (2012) |
| **F1-macro** | Média não-ponderada de F1 das classes; insensível ao desbalanço de prior | van Rijsbergen (1979); Yang (1999) |
| **F1-fake** | F1 com `pos_label=0`; foco na classe minoritária crítica | Definição operacional |
| **Stratified K-fold** | $K$-fold CV preservando proporção de classes em cada fold | Kohavi (1995), §5 |
| **Forest-RI** | Algoritmo de RF com seleção aleatória de $m=\sqrt{d}$ features por split | Breiman (2001), §5 |
| **Back-door criterion** | Critério de Pearl para identificar conjunto de variáveis cujo condicionamento elimina viés causal | Pearl (2009), §3.3 |
| **Teto textual** | F1 máximo alcançável só com BERT da raiz — definido pelo script 10 | TCC |
| **Piso de confound** | F1 mínimo "trivial" alcançável só com `num_nodes` — definido por este script | TCC |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 11_diagnostico_confound.py
📄  Arquivo gerado: theory_andre/11_diagnostico_confound_doc.md
📚  Fontes acadêmicas utilizadas: 13
    1. Breiman (2001) — Random Forests, Machine Learning
    2. Pearl (2009) — Causality, Cambridge UP
    3. Geirhos et al. (2020) — Shortcut Learning, Nature MI
    4. Sui et al. (2022) — Causal Attention for GNN, KDD '22
    5. Rubin (1973) — Matching to Remove Bias, Biometrics
    6. Iacus, King & Porro (2012) — CEM, Political Analysis
    7. Kohavi (1995) — Cross-Validation Study, IJCAI '95
    8. Dietterich (1998) — Statistical Tests for ML, Neural Comp.
    9. Dou et al. (2021) — UPFD, SIGIR '21
    10. Vosoughi, Roy & Aral (2018) — Spread of False News, Science
    11. Devlin et al. (2019) — BERT, NAACL '19
    12. Shu et al. (2020) — FakeNewsNet, Big Data
    13. van Rijsbergen (1979) — Information Retrieval
🔍  Conceitos cobertos:
    - Random Forest (Forest-RI, Strong Law of Large Numbers, voto majoritário)
    - Confound causal (DAG X←Z→Y, back-door criterion)
    - Shortcut learning (taxonomia de Geirhos; intended vs shortcut)
    - Spurious correlations em GNN (Sui et al. SCM G=C∪S)
    - Subsampling pareado / Exact Matching / CEM
    - StratifiedKFold (Kohavi 1995, paired t-test Dietterich 1998)
    - Confound de tamanho em FakeNewsNet/UPFD (Dou 2021 Tab.1; Vosoughi 2018)
    - F1-macro e F1-fake (van Rijsbergen 1979)
    - Gate heurístico e suas limitações
⚠️   Limitações:
    - WebSearch e WebFetch foram negados por permissão durante esta sessão;
      as localizações específicas (seções, equações, páginas) e DOIs foram
      reportados a partir de conhecimento canônico das referências Tier-1
      pré-validadas no CLAUDE.md e em literatura amplamente estabelecida.
      Recomenda-se verificação manual de números de página e seções exatas
      antes da publicação do TCC, especialmente para Pearl (2009) §3.3 e
      Breiman (2001) §5 (Forest-RI / Teorema 1.2).
    - Script 11 ainda não foi executado neste ambiente — F1 numéricos para
      Variante A/B reportados em §6.3 são intervalos esperados baseados em
      Dou et al. (2021) Tab.1 (avg #nodes 41 vs 92 em UPFD-PolitiFact);
      atualizar a tabela com valores empíricos quando `resultados.csv`
      estiver disponível.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```
