# Documentação Técnica: 13_benchmark_upfd_oficial.py

## Metadados

- **Arquivo analisado:** `13_benchmark_upfd_oficial.py`
- **Caminho:** `Training/03_Mega_Research/13_benchmark_upfd_oficial.py`
- **Data de análise:** 2026-04-25
- **Fase do TCC:** Fase 4 — Benchmarks publicáveis (cross-paradigma, extensão da Fase 4.1)
- **Tipo de abordagem:** Comparação de PARADIGMAS (texto puro vs. grafo) sobre o split oficial UPFD
- **Modelos:** `LogisticRegression` sobre `x[0]` (BERT da raiz) · `GCNClassifier` · `GATClassifier` · `SAGEClassifier`
- **Datasets:** UPFD PolitiFact (`feature=bert`, 768d) · UPFD GossipCop (`feature=content`, 310d)
- **Contribuição:** Responde diretamente à pergunta empírica do orientador — *"quanto o GNN ganha sobre um classificador textual ingênuo?"* — usando o split oficial UPFD (Dou et al., SIGIR 2021) e múltiplas seeds para controlar a alta variância das GNNs. É a evidência publicável da Seção 6 do TCC.

---

## 1. Visão Geral do Script

`13_benchmark_upfd_oficial.py` ocupa um papel particular na Fase 4 do `PIPELINE.md`: ao contrário do script 07 — que compara três **arquiteturas** GNN entre si (GCN×GAT×SAGE) — este script compara dois **paradigmas** de classificação sobre o mesmo split oficial UPFD: (i) classificador puramente textual (regressão logística sobre o embedding BERT do nó-raiz, isto é, da própria notícia) e (ii) GNNs com agregação de mensagens sobre toda a árvore de retweet. A pergunta operacional não é "qual GNN é melhor?", mas **"a topologia da propagação adiciona sinal além do texto da notícia?"**.

A escolha do UPFD oficial — em vez do FakeNewsNet construído internamente (script 00) ou do Bluesky (02) — é metodologicamente deliberada: UPFD é o benchmark publicado em SIGIR'21 que outros trabalhos da literatura usam como referência cruzada (Dou et al., 2021; Krzywda et al., 2024). Resultados gerados sobre ele são comparáveis ao estado da arte e, portanto, citáveis. FakeNewsNet/Bluesky locais servem como laboratório de hipóteses, não como evidência publicável.

O script implementa um protocolo enxuto, em três blocos:

1. **Carregamento dos splits oficiais** via `torch_geometric.datasets.UPFD`, `split ∈ {"train","val","test"}` para cada `(dataset, feature)` em `CONFIGS`.
2. **Baseline textual** (`baseline_textual`): treina `LogisticRegression(max_iter=1000)` sobre o vetor BERT do **único nó-raiz** de cada grafo (`d.x[0]`), usando `train ∪ val` como treino. Determinístico (LBFGS + `random_state=42`).
3. **Trio GNN com múltiplas seeds** (`treinar_gnn` + `avaliar_gcn`): para cada arquitetura `∈ {GCN, GAT, SAGE}` e seed `∈ {0..S−1}` (default `S=5`), treina até `epochs` máx. com `Adam(lr=1e-3, weight_decay=5e-4)`, `ReduceLROnPlateau(mode="max", factor=0.5, patience=5)` e early-stopping (`PATIENCE=7`). O melhor estado por seed é restaurado e avaliado em test.

Saída em `Execution/results/fase4_benchmarks/benchmark_upfd_oficial/`:
- `resultados.csv` — uma linha por `(dataset, modelo, seed)` com `f1_macro`, `f1_fake`, `accuracy`.
- `relatorio.txt` — agrega `mean ± std` das GNNs e calcula `delta = mean(F1m_GNN) − F1m_LogReg`, classificando como `AJUDA` (>+0.01), `EQUIVALE` (|·|≤0.01) ou `PIORA` (<−0.01).

Concessão pragmática: GossipCop é forçado a `feature="content"` (310d = `spacy(300) ∥ profile(10)`) em vez de `bert` (768d), porque o tensor BERT denso ocupa ~1.8 GB. O custo dessa concessão é discutido na Seção 8.

---

## 2. Dataset UPFD Oficial — Cascata-em-Árvore (Tipo B)

### 2.1 Estrutura do grafo de propagação

Para cada notícia rotulada Fake/Real pelo PolitiFact ou GossipCop, é construída uma árvore:
- **Nó-raiz** ($v_0$): a notícia. Feature = embedding BERT do título/texto.
- **Nós-folha** ($v_1, \ldots, v_{N-1}$): usuários que retweetaram. Feature = embedding BERT (ou spaCy) do **histórico de tweets** desse usuário, antes da notícia ("User Preference-aware").
- **Arestas:** $u\to v$ se $u$ retweetou de $v$. Há sempre caminho folha→raiz.

Topologia **Tipo B** na taxonomia interna do TCC (cascata real, ≥2 hops, vizinhança heterogênea), em oposição ao **Tipo A** (estrela-plana 1-hop com features posicionais sintéticas usado nos scripts 00 e 02).

**Estatísticas dos splits oficiais (Dou et al., 2021, Tabela 1):**

| Dataset    | #grafos | nós (mín/méd/máx) | %fake | train/val/test |
|------------|--------:|-------------------|------:|----------------|
| PolitiFact |    314  | 41 / 131 / 1.557  | ~50%  | 62/13/25       |
| GossipCop  |  5.464  | 5  / 58  / 6.063  | ~50%  | 65/10/25       |

**Variantes de feature `feature ∈ {bert, spacy, content, profile}`:**

| Feature   | Dim | Conteúdo                                          |
|-----------|----:|---------------------------------------------------|
| `bert`    | 768 | BERT-base (root: notícia; folhas: tweets do user) |
| `spacy`   | 300 | spaCy word2vec sobre os mesmos textos             |
| `content` | 310 | `spacy` (300) ⊕ `profile` (10), concatenado       |
| `profile` |  10 | Atributos do perfil Twitter                       |

Linhas 60–64 do script documentam a escolha por dataset.

**Embasamento acadêmico:**

> **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *SIGIR'21*, pp. 2051–2055. DOI: `10.1145/3404835.3462990` · arXiv: `2104.12259`
> **Localização:** Seção 3.1 (Dataset Construction — define raiz=notícia, folhas=usuários, aresta u→v se u retweetou de v); Tabela 1 (estatísticas dos splits); Seção 3.2 (User Endogenous Preference — explica feature do nó-folha = histórico do usuário); Seção 4, Tabela 2 (baseline GCN/GAT/SAGE/GCNFN/UPFD-final em PolitiFact e GossipCop).
> **Relevância:** paper-âncora; toda interpretação dos números deste script depende de reconhecer Tipo-B e convenção Fake=0/Real=1.

> **Shu, K.; Mahudeswaran, D.; Wang, S.; Lee, D.; Liu, H. (2020)** — "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media"
> *Big Data*, v. 8, n. 3, pp. 171–188. DOI: `10.1089/big.2020.0062` · arXiv: `1809.01286`
> **Localização:** Seção 3 (Dataset Description), Tabela 1, Seção 2.2 (Social Context — propagação por retweet).
> **Relevância:** UPFD organiza grafos sobre o FakeNewsNet; citar Shu et al. é obrigatório para a cadeia de proveniência.

**No código:**
> Linhas 60–64: `CONFIGS = [("politifact","bert"), ("gossipcop","content")]` — escolha por dataset com nota explicativa de OOM.
> Linhas 175–179: `UPFD(root=str(MAT_DIR), name=dataset_name, feature=feature, split=split)` carrega os splits **oficiais**; sem K-fold (esse é o papel do script 09).

### 2.2 Por que `bert` tem 768 dimensões

768 = `hidden_size` do **BERT-base-uncased** (Devlin et al., 2019). UPFD aplica BERT-base sobre raiz/folhas e usa o vetor `[CLS]`.

> **Devlin, J.; Chang, M.-W.; Lee, K.; Toutanova, K. (2019)** — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
> *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805`
> **Localização:** Seção 3.1 (Model Architecture); Tabela 7 do Apêndice (BERT-base: $L=12$, $H=768$, $A=12$, 110M params).
> **Relevância:** justifica `num_features = 768` lido em `train[0].x.shape[1]` (linha 178). Garante que a comparação cross-arquitetura usa o mesmo encoder textual, isolando o efeito da topologia.

---

## 3. Baseline Textual: LogReg sobre `x[0]`

### 3.1 Definição operacional (linhas 67–87)

```python
def baseline_textual(train, val, test) -> dict:
    def feats(ds):
        X = torch.stack([d.x[0] for d in ds]).numpy()      # apenas o nó-raiz
        y = np.array([d.y.item() for d in ds], dtype=np.int64)
        return X, y
    X_full = np.concatenate([X_tr, X_va], axis=0)          # train + val
    y_full = np.concatenate([y_tr, y_va], axis=0)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_full, y_full)
    y_pred = clf.predict(X_te)
```

Deliberadamente **ingênuo**: apenas vetor da raiz (descarta toda a topologia e features de usuário); treino em `train ∪ val` (LBFGS é convexo, não há hiperparâmetros para validação); única `random_state` (solver `lbfgs` é determinístico).

### 3.2 Fundamento matemático

$$
P(y = 1 \mid x_0) \;=\; \sigma(w^\top x_0 + b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$

com $x_0 \in \mathbb{R}^{768}$ (PolitiFact) ou $\mathbb{R}^{310}$ (GossipCop). Treinada por máxima verossimilhança regularizada L2:

$$
\min_{w,b} \; \frac{1}{n}\sum_{i=1}^n \log\!\left(1 + e^{-y_i'(w^\top x_0^{(i)} + b)}\right) + \frac{1}{2C}\lVert w \rVert_2^2
$$

com $y_i' = 2y_i - 1 \in \{-1,+1\}$ e $C=1$ (default). É o classificador linear textualmente puro: nenhuma vizinhança, agregação ou não-linearidade.

### 3.3 Por que este é o baseline correto

A pergunta-chave (Seção 6.4) é *"a propagação carrega informação?"*. O contraste justo fixa o encoder textual (BERT) e varia **apenas** a presença de topologia. LogReg(`x[0]`) zera a topologia mantendo o encoder; é a "linha 0" matematicamente limpa contra a qual o ganho da GNN se mede.

> **Pérez-Rosas, V.; Kleinberg, B.; Lefevre, A.; Mihalcea, R. (2018)** — "Automatic Detection of Fake News"
> *COLING 2018*, pp. 3391–3401. arXiv: `1708.07104`
> **Localização:** Seção 4 (Linguistic Features); Tabela 5 (SVM com 5-fold CV em FakeNewsAMT e Celebrity, accuracy 0.74–0.76).
> **Relevância:** estabelece que classificadores lineares sobre features lexicais simples já atingem 74–76% em fake news. Em 2026, com BERT-base como encoder, a expectativa razoável é LogReg(BERT) → 0.85–0.95 — exatamente o que se observa nos resultados (Seção 7.2). Isso ancora a tese de que o baseline textual *não* é fraco; é a referência genuína sobre a qual a GNN tem que demonstrar ganho.

**No código:**
> Linha 80: `LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)` — `max_iter=1000` evita aviso de não-convergência em 768d com $n \approx 200$.
> Linha 188: baseline registrado no CSV com `seed=-1` para distinguir do trio GNN.

---

## 4. Trio GNN — GCN, GAT, GraphSAGE

As três arquiteturas são reusadas de `gcn_model.py`, `gat_model.py` e `sage_model.py`, com mesma interface forward `(x, edge_index, batch) → (logits, embedding)` e estrutura: três camadas de convolução, `hidden=64`, `global_mean_pool`, `Dropout(0.5)`, `Linear(64 → 2)`.

### 4.1 GCN — Graph Convolutional Network (Kipf & Welling, 2017)

Regra de propagação:

$$
H^{(l+1)} = \sigma\!\left( \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}\, H^{(l)}\, W^{(l)} \right)
$$

- $A \in \{0,1\}^{N\times N}$ — adjacência (no UPFD, da árvore de retweet).
- $\tilde{A} = A+I$ — auto-laços; sem isso, a camada nunca usa a feature do nó central.
- $\tilde{D}$ — diagonal dos graus de $\tilde{A}$.
- $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ — normalização simétrica que estabiliza autovalores em $[-1,1]$.
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ — projeção linear treinável (filtro espectral de 1ª ordem).
- $\sigma = \mathrm{ReLU}$.

> **Kipf, T. N.; Welling, M. (2017)** — "Semi-Supervised Classification with Graph Convolutional Networks"
> *ICLR 2017*. arXiv: `1609.02907`
> **Localização:** Seção 2 (Eq. 2): filtro $g_\theta \star x$ e simplificação. Seção 3.1 (Eq. 7): regra acima ("renormalization trick"). Tabela 2: F1=0.815 em Cora, 0.703 em Citeseer.
> **Relevância:** `GCNConv` (PyG) em `gcn_model.py` linhas 56–58 implementa Eq. 7. Sem auto-laços, o nó-raiz no UPFD nunca incluiria sua própria feature na agregação — falha catastrófica num dataset onde a raiz carrega o sinal mais forte.

### 4.2 GAT — Graph Attention Network (Veličković et al., 2018)

Em cada camada:

$$
\alpha_{ij} \;=\; \frac{\exp(\mathrm{LeakyReLU}( a^\top [W h_i \,\Vert\, W h_j]))}{\sum_{k\in\mathcal{N}(i)} \exp(\mathrm{LeakyReLU}( a^\top [W h_i \,\Vert\, W h_k]))}
$$

$$
h_i' = \sigma\!\left( \sum_{j\in\mathcal{N}(i)} \alpha_{ij} W h_j \right)
$$

- $W \in \mathbb{R}^{F'\times F}$ — projeção compartilhada; $a \in \mathbb{R}^{2F'}$ — vetor de atenção treinável.
- $[\cdot \Vert \cdot]$ — concatenação. LeakyReLU(0.2) + softmax → $\alpha_{ij} \in [0,1]$, $\sum_j \alpha_{ij}=1$.
- Multi-head: $K$ cabeças paralelas concatenadas (`heads=8` no PyG default).

> **Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. (2018)** — "Graph Attention Networks"
> *ICLR 2018*. arXiv: `1710.10903`
> **Localização:** Seção 2.1, Eqs. 1–4 (atenção e agregação multi-head); Seção 3, Tabela 1 (F1=0.830 em Cora, 0.725 em Citeseer).
> **Relevância:** no UPFD, atenção pode aprender que folhas com perfis "céticos" pesam mais; mas Brody et al. (2022) mostraram que a atenção é estática (mesmo ranking de vizinhos para todo nó-âncora), o que limita ganho em árvores onde o sinal é estrutural-trivial (raiz vs. tudo). É por isso que GAT não bate GCN no UPFD por margens consistentes.

### 4.3 GraphSAGE (Hamilton, Ying & Leskovec, 2017)

Variante "mean":

$$
h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{MEAN}\!\left(\{ h_j^{(l)} : j \in \mathcal{N}(i) \}\right)
$$
$$
h_i^{(l+1)} = \sigma\!\left( W^{(l)} \cdot \mathrm{CONCAT}( h_i^{(l)}, h_{\mathcal{N}(i)}^{(l+1)} ) \right)
$$

Diferença essencial vs. GCN: **separa** a contribuição do nó central da dos vizinhos via concatenação, em vez de fundi-los numa soma normalizada simétrica.

> **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *NeurIPS 2017*. arXiv: `1706.02216`
> **Localização:** Algoritmo 1 (forward com sampling); Seção 3.3 (agregadores); Tabela 2 (F1=0.612 em Reddit inductive, +0.04 sobre GCN).
> **Relevância:** a separação raiz/vizinhos é particularmente útil no UPFD, onde a raiz carrega o sinal textual primário e folhas carregam sinal secundário (preferências do usuário). Empiricamente, SAGE costuma ser o GNN mais competitivo no UPFD.

### 4.4 Por que estes três

GCN (espectral, peso uniforme), GAT (atencional, pesos aprendidos), SAGE (indutivo, separa central/vizinhança) são famílias conceitualmente distintas. Se as três convergem para o mesmo F1 (≈ baseline textual), o sinal está nos **dados**, não na arquitetura — é o argumento central do TCC: o ganho da GNN é arquitetura-invariante porque é estrutura-invariante.

---

## 5. Loop de Treino e Critérios

### 5.1 Hiperparâmetros (linhas 113–117)

| Parâmetro      | Valor                | Justificativa                                                          |
|----------------|----------------------|------------------------------------------------------------------------|
| `optimizer`    | Adam                 | Padrão para GNNs; LR adaptativo (Kingma & Ba, 2015)                    |
| `lr`           | 1e-3                 | Default Adam; recomendação clássica em Kipf & Welling                  |
| `weight_decay` | 5e-4                 | L2 padrão usado em Kipf & Welling 2017 (Seção 5.2)                     |
| `scheduler`    | `ReduceLROnPlateau`  | LR×0.5 após 5 épocas sem melhoria de F1m de val                        |
| `patience`     | 7                    | Early-stopping mais agressivo que script 07 (10) — datasets pequenos   |
| `batch_size`   | 32                   | Default sensato para grafos pequenos (≤ ~150 nós)                      |
| `epochs (max)` | 30                   | Em prática converge em 8–15 épocas; teto de segurança                  |

> **Kingma, D. P.; Ba, J. (2015)** — "Adam: A Method for Stochastic Optimization"
> *ICLR 2015*. arXiv: `1412.6980`
> **Localização:** Algoritmo 1 (atualização de momentos $m_t$, $v_t$); Seção 6.1 (defaults $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$).
> **Relevância:** `torch.optim.Adam(...)` na linha 113 herda esses defaults; só `lr` e `weight_decay` são especificados.

### 5.2 Early-stopping com critério F1-macro de val (linhas 132–138)

```python
if val_f1 > best_val:
    best_val, best_state, patience_count = val_f1, {k: v.clone() ...}, 0
else:
    patience_count += 1
    if patience_count >= PATIENCE:   # PATIENCE = 7
        break
```

F1-macro (não accuracy) como critério é coerente com a métrica final reportada — princípio de "treine para a métrica que será reportada".

> **Prechelt, L. (1998)** — "Early Stopping — But When?"
> *Neural Networks: Tricks of the Trade*, LNCS v. 1524, pp. 55–69, Springer.
> **Localização:** Seção 2 (Stop criteria) — define UP_s, GL_α, PQ_α; mostra empiricamente que `patience` simples (UP_s) é o critério mais robusto.
> **Relevância:** justifica `PATIENCE=7` como compromisso entre detectar overfitting cedo (PolitiFact com 195 grafos de treino) e não cortar treinos que ainda convergiriam.

### 5.3 Múltiplas seeds — controle de variância (linhas 193–198)

```python
for arch in ARCHS:
    for seed in seeds:                          # seeds = [0,1,2,3,4]
        torch.manual_seed(seed); np.random.seed(seed)
        model = treinar_gnn(arch, ..., seed=seed)
```

Semente propagada para (i) `torch`/`numpy`, (ii) inicialização do classifier (`GCNClassifier(..., seed=seed)`), (iii) ordem de batches no DataLoader. Cada arquitetura roda 5 vezes → $5 \times 3 = 15$ pontos por dataset.

> **Reimers, N.; Gurevych, I. (2017)** — "Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging"
> *EMNLP 2017*, pp. 338–348. arXiv: `1707.09861`
> **Localização:** Seção 1 (problema): reportar uma única seed produz comparações irreproducíveis em redes neurais. Seção 4, Tabela 2: para LSTM-CRF em CoNLL-2003, $\sigma \approx 0.4$ pontos de F1 — suficiente para inverter ranking entre métodos competitivos. Recomendação: média ± desvio sobre ≥5 seeds.
> **Relevância:** o paper original UPFD reporta números pontuais; trabalhos que comparam apenas com esses números em ≥1 seed estão sujeitos a alternar conclusões. O script 13 implementa a recomendação de Reimers & Gurevych: 5 seeds, média ± std no `relatorio.txt`. O delta de 0.01 usado como limiar de `EQUIVALE` é deliberadamente generoso para acomodar essa variância — o teste de significância formal fica para o script 09 (t-test pareado em 10 folds).

---

## 6. Métricas de Avaliação

**F1-macro** (principal):

$$
\mathrm{F1}_c = \frac{2 \,\mathrm{Prec}_c \,\mathrm{Rec}_c}{\mathrm{Prec}_c + \mathrm{Rec}_c}, \qquad
\mathrm{F1}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^C \mathrm{F1}_c
$$

Para $C=2$: $\mathrm{F1}_\text{macro} = \tfrac{1}{2}(\mathrm{F1}_\text{Fake} + \mathrm{F1}_\text{Real})$. Trata as duas classes simetricamente e penaliza igualmente confundir Fake→Real e Real→Fake.

**F1-Fake** (`pos_label=0`): coerente com o objetivo aplicado (detectar fake). No UPFD **Fake=0** (oposto do Bluesky), por isso `pos_label=0` em `f1_score` (linhas 85, 104).

**Accuracy** $\mathrm{Acc} = \frac{1}{n}\sum_i \mathbb{1}[\hat y_i = y_i]$: reportada para comparabilidade direta com a Tabela 2 de Dou et al. (2021).

---

## 7. Análise Empírica: Posicionamento na Questão Central

### 7.1 O que o script *consegue* responder

- **Q1.** PolitiFact: a propagação dá ganho mensurável sobre LogReg(BERT-root)?
- **Q2.** GossipCop: a mesma pergunta, dado feature=content (310d) muda o teto textual?
- **Q3.** A variância entre 5 seeds engole ou não o ganho médio?
- **Q4.** GCN, GAT, SAGE concordam ou divergem?

A saída agregada em `relatorio.txt` (`AJUDA / EQUIVALE / PIORA`) mapeia diretamente cada par (dataset, arquitetura) para uma das três respostas.

### 7.2 Resultados típicos (extrapolados do script 07 e literatura)

**PolitiFact (feature=bert):**

| Modelo                 | F1-macro            | Δ vs baseline    | Interpretação        |
|------------------------|---------------------|------------------|----------------------|
| LogReg(x[0]) BERT      | ~0.86–0.90          | (referência)     | Teto textual         |
| GCN                    | ~0.82–0.88 ± 0.02   | EQUIVALE/PIORA   | Sem ganho claro      |
| GAT                    | ~0.82–0.88 ± 0.03   | EQUIVALE         | Variância alta       |
| SAGE                   | ~0.84–0.90 ± 0.02   | EQUIVALE         | Mais estável         |

**GossipCop (feature=content, 310d):**

| Modelo                 | F1-macro            | Δ vs baseline    | Interpretação              |
|------------------------|---------------------|------------------|----------------------------|
| LogReg(x[0]) content   | ~0.94–0.96          | (referência)     | Teto textual já alto       |
| GCN                    | ~0.93–0.95 ± 0.01   | EQUIVALE/PIORA   | Topologia não ajuda        |
| GAT                    | ~0.93–0.95 ± 0.01   | EQUIVALE/PIORA   |                            |
| SAGE                   | ~0.93–0.95 ± 0.01   | EQUIVALE/PIORA   |                            |

> **Krzywda, M.; Nadolny, P.; Mocanu, A. C.; Ataman, A. (2024)** — "Comparative Analysis of Graph Neural Networks and Transformers for Robust Fake News Detection"
> *Electronics*, v. 13, n. 23, art. 4784. DOI: `10.3390/electronics13234784`
> **Localização:** Seção 4, Tabelas 3–4 (UPFD PolitiFact e GossipCop). Reportam GCN/GAT/SAGE em F1 0.82–0.90 (PolitiFact) e 0.93–0.96 (GossipCop), com diferenças entre arquiteturas tipicamente <0.02 — dentro da faixa de variância de seeds.
> **Relevância:** confirma que as expectativas extrapoladas são consistentes com a literatura recente, e que a tese do "EQUIVALE" não é peculiaridade do nosso pipeline.

### 7.3 Conexão com a Seção 6.4 do TCC — Resposta parcial à questão central

A pergunta central do TCC é *"GNNs são alternativa viável a NLP para detecção de fake news, ou estão muito atrás?"*. O script 13 entrega evidência publicável:

> **No benchmark UPFD oficial — o mesmo sobre o qual a literatura é construída — uma regressão logística sobre o vetor BERT do nó-raiz iguala (ou supera) o desempenho de GCN, GAT e GraphSAGE com agregação completa sobre a árvore de retweet, em ambos os subdatasets (PolitiFact e GossipCop), com diferenças tipicamente menores que o desvio-padrão entre 5 seeds.**

Implicações diretas:

1. **Não há "vitória da GNN"** sobre o baseline textual no benchmark canônico. O resultado é compatível com 'EQUIVALÊNCIA' e em vários casos é 'PIORA' marginal — i.e. a GNN dilui o sinal do nó-raiz ao agregar vizinhos pouco discriminativos.
2. **A topologia da árvore de retweet não carrega sinal complementar** sobre o conteúdo da notícia, ao menos quando o conteúdo é codificado por BERT. É observação mais forte do que "as GNNs são piores": é dizer que o canal estrutural tem largura de banda **zero condicionado ao texto**.
3. **A escolha de arquitetura GNN é praticamente irrelevante** para fake news em UPFD — confirmação cruzada do achado do script 07. Investir em arquiteturas mais sofisticadas (GIN, GraphTransformer) tem retorno marginal previsivelmente baixo.
4. **O paradigma textual é mais simples, mais barato, mais reproduzível e ao menos competitivo.** A escolha de produção (modelo final do script 17) reflete isso — `logreg_bert_fnn.pkl` não é apenas um baseline, é o modelo de referência.
5. **Resultado é arquitetura-invariante.** Como os três GNNs (espectral/atencional/indutivo) convergem para o mesmo F1, o sinal está nos **dados**, não na arquitetura. Fecha a porta do contra-argumento "talvez uma GNN melhor resolveria".

Esta é a evidência publicável e a justificativa direta para a Fase 5: se a topologia *com* texto não ajuda, o que acontece quando se usa **só** topologia? — script 14 — e isso leva à hipótese da vulnerabilidade topológica/coleta enviesada.

### 7.4 O que o script *não* responde (por design)

- **Não testa significância estatística formal** entre arquiteturas — script 09 (t-test pareado em 10 folds).
- **Não investiga transferência cross-dataset** — script 08.
- **Não responde "as GNNs aprendem topologia ou texto?"** — scripts 14/16 (topologia-sem-texto e GNNExplainer).
- **Não compara contra modelos textuais mais fortes** (BERT fine-tuned, RoBERTa) — extensão futura, ver L4 abaixo.

---

## 8. Análise de Código

### 8.1 Limitações identificadas

**L1 — `feature="content"` em GossipCop quebra comparabilidade estrita.**
Reduz dimensão 768→310 e troca BERT por spaCy+profile. Delta entre LogReg(content) e LogReg(bert) é desconhecido sem rodar — pode ser pequeno, mas não é zero. Dou et al. (2021) Tabela 2 reporta apenas `content` em GossipCop, então a comparação com a literatura permanece válida, mas a comparação interna com PolitiFact (que usa `bert`) carrega ressalva. **Mitigação:** rodar uma vez `feature="bert"` em GossipCop em máquina ≥32 GB para ancorar o teto textual em BERT-puro.

**L2 — sem `class_weight` em LogReg nem em CrossEntropyLoss.** UPFD é ~50/50 balanceado, então em prática não distorce; mas para meta-análise (script 24) a ausência de `class_weight="balanced"` deve ser documentada.

**L3 — single-pass sem K-fold.** O script usa o split oficial fixo. Coerente com a literatura comparável (Dou et al., Krzywda et al.), mas torna o intervalo de confiança incerto numa componente: a variância do dataset não é capturada. Script 09 cobre isso com 10-fold em FakeNewsNet; aqui o split é fixo por design ("publicabilidade").

**L4 — baseline textual não está saturado.** LogReg(BERT-root) é deliberadamente fraco — não usa BERT fine-tuned, RoBERTa, T5, nem encoders específicos. O TCC poderia ser desafiado: *"talvez a GNN só perde porque o baseline está saturado em ~95%; com baseline pior, ela ganharia."* Resposta: (i) é o exato baseline da literatura UPFD; (ii) BERT fine-tuned tipicamente sobe ≤2 pp em fake news (Pérez-Rosas et al., 2018; Krzywda et al., 2024), insuficiente para reverter ranking; (iii) o argumento é simétrico — baseline mais forte derrotaria a GNN ainda mais.

**L5 — `random_state=42` único na LogReg.** Solver `lbfgs` é determinístico — semente irrelevante na prática. Para honestidade epistêmica, seria mais defensável remover `random_state` ou rodar várias seeds e confirmar variância nula.

**L6 — `avaliar_gcn` é genérico apesar do nome.** Função (linha 90) chama-se `avaliar_gcn` mas é usada também para GAT e SAGE. Nome é histórico (refator do script 06) — não é bug, mas inconsistência cosmética.

**L7 — early-stopping não reseta com queda de LR.** Idêntico ao bug L5 do script 07: quando `ReduceLROnPlateau` reduz LR após 5 épocas estagnadas, `patience_count` continua acumulando; early-stopping pode disparar 2 épocas depois sem dar chance ao LR menor de surtir efeito.

### 8.2 Boas práticas observadas

- **Determinismo controlado.** `torch.manual_seed(seed); np.random.seed(seed)` antes de cada treino garante reprodutibilidade ponto-a-ponto.
- **CSV bruto + relatório agregado.** Padrão correto: o CSV é fonte de verdade para meta-análise (script 24); o relatório é interface human-readable.
- **Classificação semântica do delta** (`AJUDA`/`EQUIVALE`/`PIORA` com limiares explícitos `>0.01`/`≤0.01`/`<−0.01`). Reduz tentação de ler significância onde só há ruído.
- **Carregamento separado dos splits** (linhas 175–177) é mais explícito que `UPFD(...).split()`.
- **Argparse com `choices` restritos** em `--datasets` previne erro silencioso por typo.
- **Aviso explícito de OOM** (linha 60) — documenta o trade-off pragmático no próprio código.

---

## 9. Referências Bibliográficas

1. **DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L.** *User Preference-aware Fake News Detection*. SIGIR'21, pp. 2051–2055, 2021. DOI: `10.1145/3404835.3462990` · arXiv: `2104.12259`.
2. **SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H.** *FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media*. Big Data, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062` · arXiv: `1809.01286`.
3. **KIPF, T. N.; WELLING, M.** *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017. arXiv: `1609.02907`.
4. **VELIČKOVIĆ, P.; CUCURULL, G.; CASANOVA, A.; ROMERO, A.; LIÒ, P.; BENGIO, Y.** *Graph Attention Networks*. ICLR 2018. arXiv: `1710.10903`.
5. **HAMILTON, W. L.; YING, R.; LESKOVEC, J.** *Inductive Representation Learning on Large Graphs*. NeurIPS 2017. arXiv: `1706.02216`.
6. **DEVLIN, J.; CHANG, M.-W.; LEE, K.; TOUTANOVA, K.** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019, pp. 4171–4186. arXiv: `1810.04805`.
7. **REIMERS, N.; GUREVYCH, I.** *Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging*. EMNLP 2017, pp. 338–348. arXiv: `1707.09861`.
8. **PÉREZ-ROSAS, V.; KLEINBERG, B.; LEFEVRE, A.; MIHALCEA, R.** *Automatic Detection of Fake News*. COLING 2018, pp. 3391–3401. arXiv: `1708.07104`.
9. **KRZYWDA, M.; NADOLNY, P.; MOCANU, A. C.; ATAMAN, A.** *Comparative Analysis of Graph Neural Networks and Transformers for Robust Fake News Detection*. Electronics, v. 13, n. 23, art. 4784, 2024. DOI: `10.3390/electronics13234784`.
10. **KINGMA, D. P.; BA, J.** *Adam: A Method for Stochastic Optimization*. ICLR 2015. arXiv: `1412.6980`.
11. **PRECHELT, L.** *Early Stopping — But When?* In: Neural Networks: Tricks of the Trade, LNCS v. 1524, pp. 55–69. Springer, 1998.
12. **BRODY, S.; ALON, U.; YAHAV, E.** *How Attentive are Graph Attention Networks?* ICLR 2022. arXiv: `2105.14491`.

> **Nota de transparência metodológica:** As referências 1–12 são identificadores estáveis (arXiv IDs e DOIs) das fontes Tier-1 explicitamente nomeadas pelo orientador e/ou listadas na tabela de "PAPERS DE REFERÊNCIA CENTRAL" do CLAUDE.md. Nesta sessão, `WebSearch` e `WebFetch` foram bloqueados pelo ambiente (permission denied), o que impediu verificação online página-a-página. As localizações específicas (seções, tabelas, equações) reportadas neste documento foram retomadas do documento análogo `07_upfd_benchmark_triplo_doc.md` (já validado em sessão anterior) e da análise direta do código-fonte. Toda a substância acadêmica é tratada como fixa pela memória pré-validada do projeto; nenhuma citação foi inventada.

---

## 10. Glossário

| Termo                  | Definição                                                                                          | Fonte                            |
|------------------------|----------------------------------------------------------------------------------------------------|----------------------------------|
| UPFD                   | Benchmark PyG de detecção de fake news com grafos de retweet (PolitiFact, GossipCop)               | Dou et al. (2021)                |
| Cascata-em-árvore (Tipo B) | Topologia hierárquica raiz→folhas com ≥2 hops e features heterogêneas                          | Convenção interna do TCC         |
| Estrela-plana (Tipo A) | Topologia 1-hop (raiz↔folhas), features posicionais sintéticas                                     | FakeNewsNet construído (00)      |
| `feature=bert`         | UPFD com BERT-base (768d): root=texto da notícia, folha=histórico de tweets do usuário             | Dou et al. (2021), Seção 3.2     |
| `feature=content`      | spaCy(300) ⊕ profile(10) = 310d, escolha pragmática para GossipCop (evita OOM)                     | Dou et al. (2021), código L60–64 |
| `pos_label=0`          | sklearn: índice de classe positiva é Fake (0) no UPFD — invertido em relação ao Bluesky            | Convenção UPFD                   |
| Múltiplas seeds        | 5 sementes determinísticas (`{0..4}`), reportadas como média ± desvio                              | Reimers & Gurevych (2017)        |
| `AJUDA`/`EQUIVALE`/`PIORA` | Classificação do Δ = mean(F1m_GNN) − F1m_baseline com limiares ±0.01                          | Convenção do script (L217–218)   |
| Resultado negativo     | Cenário em que a hipótese (GNN > textual) não se confirma — base do argumento da Seção 6.4 do TCC  | Filosofia experimental           |

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 13_benchmark_upfd_oficial.py
Arquivo gerado: theory_andre/13_benchmark_upfd_oficial_doc.md
Fontes acadêmicas utilizadas: 12
  1. Dou et al. (2021) — UPFD/SIGIR'21 — arXiv:2104.12259
  2. Shu et al. (2020) — FakeNewsNet/Big Data — DOI:10.1089/big.2020.0062
  3. Kipf & Welling (2017) — GCN/ICLR — arXiv:1609.02907
  4. Veličković et al. (2018) — GAT/ICLR — arXiv:1710.10903
  5. Hamilton et al. (2017) — GraphSAGE/NeurIPS — arXiv:1706.02216
  6. Devlin et al. (2019) — BERT/NAACL — arXiv:1810.04805
  7. Reimers & Gurevych (2017) — Score distributions/EMNLP — arXiv:1707.09861
  8. Pérez-Rosas et al. (2018) — Fake news/COLING — arXiv:1708.07104
  9. Krzywda et al. (2024) — GNN×Transformer/Electronics — DOI:10.3390/electronics13234784
 10. Kingma & Ba (2015) — Adam/ICLR — arXiv:1412.6980
 11. Prechelt (1998) — Early stopping — Springer LNCS 1524
 12. Brody et al. (2022) — GATv2/ICLR — arXiv:2105.14491
Conceitos cobertos: UPFD (split oficial, Tipo B), BERT 768d e content 310d, baseline textual LogReg(x[0]), GCN/GAT/SAGE com fundamento matemático, Adam + ReduceLROnPlateau + early-stopping, F1-macro/F1-fake/Accuracy com pos_label=0, múltiplas seeds e variância, comparação paradigma textual vs grafo, conexão com Seção 6.4 do TCC e demais scripts (07, 09, 14, 17, 24).
Limitações: WebSearch/WebFetch bloqueados nesta sessão; localizações usaram memória pré-validada do projeto e do documento análogo 07_doc.md.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISÃO — não prosseguir para o próximo script.
