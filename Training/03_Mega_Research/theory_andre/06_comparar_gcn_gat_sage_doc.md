# Documentação Técnica: 06_comparar_gcn_gat_sage.py + sage_model.py

## Metadados

- **Arquivos analisados:** `06_comparar_gcn_gat_sage.py`, `sage_model.py` (com referência cruzada a `gcn_model.py` e `gat_model.py`)
- **Caminho:** `03_Mega_Research/06_comparar_gcn_gat_sage.py`, `03_Mega_Research/sage_model.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** GNN — benchmark triplo controlado GCN vs GAT vs GraphSAGE
- **Modelos principais:** GCNClassifier (3×GCNConv) · GATClassifier (3×GATConv) · SAGEClassifier (3×SAGEConv)
- **Datasets utilizados:** Bluesky (HuggingFace: `Zaras210/bluesky-fake-news-dataset`), grafos `.pt` gerados pelo script 02
- **Contribuição para a questão central:** Este é o benchmark mais completo do TCC até o momento — três arquiteturas GNN treinadas nas mesmas condições permitem isolar o efeito do mecanismo de agregação. O resultado (SAGE > GCN > GAT, com GAT colapsando) tem duas implicações diretas: (1) mecanismos de atenção softmax falham em grafos homogêneos/rasos de propagação; (2) a separação de matrizes W_l/W_r do SAGE traz ganho mensurável sem o custo de atenção multi-head — evidência concreta para a discussão custo-benefício GNN×NLP.

---

## 1. Visão Geral dos Scripts

`06_comparar_gcn_gat_sage.py` estende o benchmark anterior (script 04, GCN×GAT) adicionando **GraphSAGE** como terceiro concorrente. O protocolo experimental é rigorosamente controlado: mesmos dados de treino/validação/teste, mesma função de perda (`CrossEntropyLoss` sem class weights), mesma LR (0.005), mesmo número de épocas (30), mesmo batch size (64), mesmo seed de inicialização. Diferenças nas métricas finais são, portanto, atribuíveis exclusivamente à arquitetura — especificamente ao mecanismo de agregação de vizinhança.

`sage_model.py` implementa o `SAGEClassifier`: três camadas `SAGEConv` (768→64→64→64) com ReLU nas duas primeiras, seguidas de `global_mean_pool`, `Dropout(0.5)` e classificador linear. A interface é idêntica ao GCN e ao GAT — `out, h = model(x, edge_index, batch)` — possibilitando substituição direta no loop unificado de treinamento.

O loop de treino (`treinar()`) inclui: otimizador Adam com weight decay (5e-4), scheduler `ReduceLROnPlateau` monitorando F1-macro de validação (fator 0.5, patience 5), e seleção do melhor modelo pelo critério de melhor F1-macro de validação. A avaliação final é feita no conjunto de teste com o estado de modelo que teve melhor validação (early stopping implícito).

As três saídas são: matrizes de confusão lado a lado (`matrizes_confusao.png`), barras agrupadas de métricas e tempo por época (`metricas_barras.png`), e relatório textual com Δ vs GCN e conclusão automática para monografia (`relatorio.txt`).

---

## 2. Arquitetura e Componentes Principais

### 2.1 GraphSAGE — Inductive Representation Learning

**Descrição técnica:**
GraphSAGE (SAmple and aggreGatE) é um framework indutivo que aprende funções de agregação de vizinhança em vez de embeddings fixos por nó. A diferença fundamental em relação ao GCN é que cada nó mantém sua própria representação separada da representação agradada dos vizinhos, conectadas por duas matrizes de peso distintas — $W_l$ (self) e $W_r$ (vizinhos). Isso torna o método **indutivo**: pode generalizar para grafos ou nós nunca vistos durante o treino.

**Fundamento matemático (Algorithm 1 de Hamilton et al., 2017):**

Para cada camada $k$ de $1$ a $K$, e para cada nó $v$ no grafo:

**Passo 1 — Agregação de vizinhança:**
$$\mathbf{h}_{\mathcal{N}(v)}^k = \text{AGG}_k\!\left(\left\{\mathbf{h}_u^{k-1} : u \in \mathcal{N}(v)\right\}\right)$$

**Passo 2 — Concatenação e transformação linear:**
$$\mathbf{h}_v^k = \sigma\!\left(\mathbf{W}^k \cdot \text{CONCAT}\!\left(\mathbf{h}_v^{k-1},\; \mathbf{h}_{\mathcal{N}(v)}^k\right)\right)$$

onde:
- $\mathbf{h}_v^k \in \mathbb{R}^{d_k}$ — embedding do nó $v$ na camada $k$; $\mathbf{h}_v^0 = \mathbf{x}_v$ (feature BERT-768)
- $\mathcal{N}(v)$ — conjunto de vizinhos de $v$ (com amostragem no algoritmo original)
- $\mathbf{W}^k \in \mathbb{R}^{d_k \times 2d_{k-1}}$ — matriz de peso aprendível da camada $k$
- $\sigma(\cdot)$ — função de ativação (ReLU no script)
- $\text{AGG}_k$ — função de agregação (detalhada em 2.2)

**Formulação equivalente do SAGEConv do PyG (implementação do script):**

A implementação PyTorch Geometric desdobra a concatenação em duas matrizes separadas:
$$\mathbf{h}_v^k = \sigma\!\left(\mathbf{W}_l^k\, \mathbf{h}_v^{k-1} + \mathbf{W}_r^k\, \underbrace{\frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{k-1}}_{\text{mean aggregation}}\right)$$

onde $\mathbf{W}_l^k$ trata o **self-loop** e $\mathbf{W}_r^k$ trata os **vizinhos** — matrizes completamente independentes, sem o acoplamento do GCN.

**Contagem de parâmetros por camada (SAGEConv):**
- Camada 1: $W_l^1 \in \mathbb{R}^{64 \times 768}$ + $W_r^1 \in \mathbb{R}^{64 \times 768}$ + bias = $2 \times (64 \times 768) + 64 = 98,368$
- Camada 2: $W_l^2, W_r^2 \in \mathbb{R}^{64 \times 64}$ = $2 \times 4,096 + 64 = 8,256$
- Camada 3: idem = $8,256$
- Classificador linear: $64 \times 2 + 2 = 130$
- **Total: 115,010 parâmetros** (≈ 2× GCN, ≈ 0.53× GAT)

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*
> arXiv: `1706.02216`
> **Localização:** Seção 3 (Proposed Method), Algorithm 1 (Mini-batch Forward Pass), Equação de update na p.3; Seção 3.3 (Aggregator Architectures) para variantes mean/max/LSTM
> **Relevância:** Paper original do GraphSAGE; define as equações implementadas pelo `SAGEConv` do PyG e usadas em `sage_model.py`. A distinção indutivo/transdutivo argumentada na Seção 1 justifica por que o SAGE generaliza melhor do que o GCN para grafos novos.

**No código (`sage_model.py`):**
> Linhas 78–96: `self.conv1/2/3 = SAGEConv(...)` — as três camadas SAGE com `aggr='mean'`.
> Linhas 118–128: forward pass — `conv1 → relu → conv2 → relu → conv3`.
> Linha 129: `h = global_mean_pool(x, batch)` — readout idêntico ao GCN e GAT.

---

### 2.2 Variantes de Agregador do GraphSAGE

**Descrição técnica:**
Hamilton et al. propõem três famílias de agregadores, cada uma com trade-offs diferentes entre expressividade, simetria (invariância a permutações) e custo computacional. O script expõe todas via `--aggr mean|max|lstm`.

**Mean Aggregator (padrão no script):**
$$\mathbf{h}_{\mathcal{N}(v)}^k = \text{MEAN}\!\left(\left\{\mathbf{h}_u^{k-1} : u \in \mathcal{N}(v)\right\}\right)$$
- Simples, simétrico, eficiente. Equivalente a uma versão simplificada do GCN sem normalização por grau.

**Max-Pooling Aggregator:**
$$\mathbf{h}_{\mathcal{N}(v)}^k = \max\!\left(\left\{\sigma\!\left(\mathbf{W}_{\text{pool}}\, \mathbf{h}_u^{k-1} + \mathbf{b}\right) : u \in \mathcal{N}(v)\right\}\right)$$
- Captura a **presença** da feature mais proeminente entre os vizinhos (max element-wise).
- Mais expressivo que a média para vizinhanças heterogêneas.

**LSTM Aggregator:**
$$\mathbf{h}_{\mathcal{N}(v)}^k = \text{LSTM}\!\left(\left[\mathbf{h}_u^{k-1} : u \in \pi(\mathcal{N}(v))\right]\right)$$
- $\pi$ é uma permutação aleatória dos vizinhos (necessária pois LSTM é sensível a ordem).
- Maior capacidade expressiva, mas não é formalmente invariante a permutações.
- Custo computacional mais elevado.

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *NeurIPS 2017* | arXiv: `1706.02216`
> **Localização:** Seção 3.3 (Aggregator Architectures), pp. 4–5; resultados comparativos dos três agregadores na Tabela 1 do paper
> **Relevância:** Fundamenta as opções `mean`, `max` e `lstm` do argumento `--aggr`. O paper mostra que o max-pooling tende a superar a média em benchmarks de classificação de nós, mas para grafos pequenos/rasos como os do TCC o ganho é marginal.

**No código (`06_comparar_gcn_gat_sage.py`):**
> Linhas 355–358: `parser.add_argument("--aggr", ...)` — expõe os três agregadores como hiperparâmetro.
> Linha 383: `sage = SAGEClassifier(768, 2, aggr=args.aggr).to(device)` — propagado para o modelo.

---

### 2.3 Graph Convolutional Network (GCN) — Baseline

**Descrição técnica:**
Documentado em detalhe em `04_comparar_gcn_gat_doc.md`. Resumo para referência cruzada: usa uma única matriz $W^{(l)}$ aplicada com normalização simétrica por grau. **Diferença crítica** vs SAGE: não separa a transformação do nó raiz da transformação dos vizinhos — ambos passam pela mesma $W$.

**Regra de propagação (Kipf & Welling, 2017, Equação 2):**
$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2}\, H^{(l)}\, W^{(l)}\right)$$

**Embasamento acadêmico:**

> 📖 **Kipf, T. N.; Welling, M. (2017)** — "Semi-Supervised Classification with Graph Convolutional Networks"
> *ICLR 2017* | arXiv: `1609.02907`
> **Localização:** Seção 2 (Fast Approximate Convolutions on Graphs), Equação 2

**No código (`gcn_model.py`):**
> Linhas 56–58: `self.conv1/2/3 = GCNConv(...)` — três camadas GCN com única matriz $W$.
> 57,666 parâmetros totais.

---

### 2.4 Graph Attention Network (GAT) e o Colapso de Atenção

**Descrição técnica:**
GAT substitui a normalização estática do GCN por coeficientes de atenção aprendidos entre pares de nós. Em grafos profundos e heterogêneos isso confere vantagem. Porém, no dataset Bluesky (grafos estrela rasos, 1–50 nós, alta homofilia de features BERT), o GAT apresentou **colapso completo** (F1=0.0000, Precision=0.0000).

**Mecanismo de atenção (GAT original, Veličković et al., 2018, Equação 3):**
$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^T\!\left[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j\right]\right), \quad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

onde $\mathbf{a} \in \mathbb{R}^{2F'}$ é o vetor de atenção, $\mathbf{W} \in \mathbb{R}^{F' \times F}$ a projeção linear.

**Por que o GAT colapsa em grafos homogêneos/rasos — fundamento acadêmico:**

Brody, Alon & Yahav (ICLR 2022) demonstram formalmente que o GAT original computa **atenção estática**: o ranking dos coeficientes $\alpha_{ij}$ entre os vizinhos de $i$ é **independente** de $\mathbf{h}_i$ (o nó consultante). Em grafos onde todos os vizinhos têm features semelhantes (alta homofilia, como em grafos estrela com embeddings BERT de textos do mesmo tópico), os coeficientes de atenção colapsam para valores uniformes — equivalendo a uma média simples sem gradiente informativo. O softmax amplifica pequenas diferenças numéricas em distribuições degenradas, levando a instabilidade no treinamento.

> 📖 **Brody, S.; Alon, U.; Yahav, E. (2022)** — "How Attentive are Graph Attention Networks?"
> *10th International Conference on Learning Representations (ICLR 2022)*
> arXiv: `2105.14491`
> **Localização:** Seção 3 (Static vs. Dynamic Attention), Proposição 1 (prova de que GAT é estático), Seção 4 (experimentos mostrando que GAT falha em fitting even the training data)
> **Relevância:** Fornece a explicação teórica rigorosa para o resultado empírico F1=0.0000 do GAT nos experimentos 04 e 06 do TCC. O paper propõe GATv2 (atenção dinâmica) como solução, mas o script usa GAT v1.

> 📖 **Veličković, P. et al. (2018)** — "Graph Attention Networks"
> *6th International Conference on Learning Representations (ICLR 2018)*
> arXiv: `1710.10903`
> **Localização:** Seção 2.1 (Graph Attentional Layer), Equações 1–3 (mecanismo de atenção)

**No código (`06_comparar_gcn_gat_sage.py`):**
> Linhas 329–336: `relatorio.txt` documenta automaticamente o colapso do GAT e apresenta a hipótese de homofilia/grafos rasos.
> Linha 422: `gat, _, _, tempos["GAT"] = treinar(...)` — GAT treinado nas mesmas condições.

---

### 2.5 Loop de Treinamento Unificado

**Descrição técnica:**
O mesmo `treinar()` é aplicado aos três modelos sequencialmente — condição *ceteris paribus* do benchmark. Usa Adam com weight decay como regularizador implícito e `ReduceLROnPlateau` para adaptação da LR.

**Otimizador Adam (Kingma & Ba, 2015):**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\,\hat{m}_t, \quad
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

onde $m_t$ e $v_t$ são estimativas de 1ª e 2ª ordem do gradiente.

**ReduceLROnPlateau:**
Monitora o F1-macro de validação a cada época; se não melhorar por `patience=5` épocas, multiplica a LR por `factor=0.5`. Comportamento de **early stopping soft** — sem interromper o treino, mas reduzindo o passo de atualização quando o modelo estagna.

**Seleção do melhor estado:**
> Linhas 123–125: `best_state = {k: v.clone() for k, v in model.state_dict().items()}` — clona os pesos sempre que `val_f1 > best_val`. No final, carrega o melhor estado (linha 132).
> Isso implementa early stopping implícito via "checkpoint do melhor modelo".

**Função de perda — CrossEntropyLoss:**
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=0}^{C-1} y_{i,c}\,\log\!\left(\hat{p}_{i,c}\right)$$

onde $\hat{p}_{i,c} = \text{softmax}(\text{logits})_c$. Sem `class_weight` → penaliza igualmente classe "Real" (majoritária) e "Fake" (minoritária).

**Embasamento acadêmico:**

> 📖 **Kingma, D. P.; Ba, J. (2015)** — "Adam: A Method for Stochastic Optimization"
> *3rd International Conference on Learning Representations (ICLR 2015)*
> arXiv: `1412.6980`
> **Localização:** Seção 2 (Algorithm 1), Equações 3–7 (bias correction e update)

---

### 2.6 Global Mean Pooling (Readout)

**Descrição técnica:**
Para classificar grafos inteiros, é necessário agregar os embeddings dos nós em um único vetor de tamanho fixo. Todos os três modelos usam `global_mean_pool`.

**Formulação:**
$$\mathbf{h}_G = \frac{1}{|V|}\sum_{v \in V} \mathbf{h}_v^K$$

onde $K$ é a última camada, $V$ é o conjunto de nós do grafo, e $|V|$ varia entre grafos no mesmo batch.

**Alternativas e trade-offs:**

| Readout | Fórmula | Sensível a $|V|$ | Captura outliers |
|---------|---------|------------------|-----------------|
| Mean pool | média elemento a elemento | Não | Fraca |
| Max pool | máximo elemento a elemento | Não | Forte |
| Sum pool | soma | Sim | Média |
| Attention | $\sum \alpha_v h_v$ (aprendido) | Não | Controlada |

Para grafos estrela (1 raiz + $n$ folhas), a média é dominada pelas folhas quando $n$ cresce — limitação relevante neste dataset.

**Embasamento acadêmico:**

> 📖 **Gilmer, J. et al. (2017)** — "Neural Message Passing for Quantum Chemistry"
> *34th International Conference on Machine Learning (ICML 2017)*
> arXiv: `1704.01212`
> **Localização:** Seção 2 (Message Passing Neural Networks), Equação de readout $\hat{y} = R(\{h_v^T : v \in G\})$; discute variantes de readout como passo distinto do message passing
> **Relevância:** Framework MPNN que formaliza o readout como operação separada e permutation-invariant — justificativa teórica para o uso de `global_mean_pool`.

**No código:**
> `sage_model.py` linha 129: `h = global_mean_pool(x, batch)` (idêntico em `gcn_model.py:65` e `gat_model.py`).

---

## 3. Pipeline de Dados e Pré-processamento

O carregamento de dados (`carregar_dados()`, linhas 51–68) usa os grafos `.pt` gerados pelo script 02. Cada grafo representa a **árvore de propagação** de uma notícia no Bluesky:

| Componente | Descrição |
|-----------|-----------|
| `data.x` | Embeddings BERT-768 por nó (texto do post/reply) |
| `data.edge_index` | Arestas direcionadas do grafo de propagação |
| `data.y` | Label binário: 0=Real, 1=Fake |
| `data.batch` | Vetor de mapeamento nó→grafo para o DataLoader |

**DataLoader com batch collation:**
`DataLoader` do PyG realiza *sparse batching*: empilha matrizes de adjacência diagonalmente em uma única matriz esparsa grande, e o vetor `batch` rastreia a qual grafo pertence cada nó. Isso permite processar múltiplos grafos de tamanhos diferentes em uma única passagem da GPU.

---

## 4. Construção do Grafo

Documentada em `02_construir_grafos_bluesky_doc.md`. Resumo:
- **Nó raiz:** post original da notícia no Bluesky
- **Nós folha:** respostas/reposts diretos e em cadeia
- **Arestas:** da resposta em direção ao post que responde (topologia estrela ou árvore)
- **Feature:** embedding do texto via modelo BERT multilingual do HuggingFace

A predominância de grafos tipo estrela (baixa profundidade, ≤ 50 nós) é a causa estrutural do colapso do GAT e da vantagem marginal do SAGE sobre o GCN — em grafos profundos com múltiplos hops, as diferenças entre os mecanismos de agregação seriam mais pronunciadas.

---

## 5. Métricas de Avaliação

### 5.1 F1-Score (critério principal)

**Fórmulas:**
$$\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}, \quad
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

onde $TP$ = verdadeiros positivos (Fake detectado corretamente), $FP$ = falso alarme, $FN$ = fake não detectado.

**Resultados obtidos:**

| Modelo | Accuracy | Precision | Recall | F1-Score | Params | Tempo/ép. |
|--------|----------|-----------|--------|----------|--------|-----------|
| GCN    | 0.9947   | 0.5678    | 0.3454 | 0.4295   | 57,666 | 8.24 s   |
| GAT    | 0.9942   | 0.0000    | 0.0000 | 0.0000   | 218,562| 11.35 s  |
| SAGE   | **0.9946** | 0.5547  | **0.3918** | **0.4592** | 115,010 | **6.67 s** |
| Δ (SAGE−GCN) | −0.0001 | −0.0131 | +0.0464 | **+0.0297** | +57,344 | −1.57 s |

**Interpretação no contexto de fake news:**

A accuracy ~0.9945 para todos os modelos é **enganosa**: reflete o desbalanceamento de classes — se o modelo predisse sempre "Real" (majoritária), obteria accuracy próxima ao baseline. A métrica relevante é o **F1-Score da classe Fake**, pois um falso negativo (fake classificada como real) é custo maior para a sociedade do que um falso positivo.

O F1=0.4592 do SAGE indica desempenho **moderado** na detecção de fake news — consistente com a hipótese do TCC de que GNNs com apenas features de texto estrutural estão significativamente abaixo do estado da arte em NLP (F1 > 0.85).

**Embasamento acadêmico:**

> 📖 **Gong, S. et al. (2023)** — "Fake News Detection Through Graph-based Neural Networks: A Survey"
> arXiv: `2307.12639`
> **Localização:** Seção 4 (Evaluation Metrics), discussão sobre por que F1-macro é preferido a accuracy em datasets desbalanceados de fake news
> **Relevância:** Justifica o uso de F1 como critério de ranking e a necessidade de reportar Precision e Recall separadamente para análise de erros.

### 5.2 Matriz de Confusão

Cada célula da matriz reporta contagens absolutas (não normalizadas), permitindo visualizar assimetria de erros:
- **Falsos Negativos (FN):** fake classificadas como real — o erro mais grave
- **Falsos Positivos (FP):** real classificadas como fake — custo menor mas afeta credibilidade

O GAT com toda a coluna "Fake" predita como "Real" (Recall=0) é o caso limite: o modelo degenera em classificador constante — comportamento confirmado pela Precision=0.0000.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes desta abordagem

**Controle experimental rigoroso:** O design controlado (mesmos dados, LR, epochs, seed) é metodologicamente correto para benchmark comparativo — elimina variáveis confundidoras e isola o efeito da arquitetura. Isso é exatamente o que a literatura recomenda para comparações justas.

**GraphSAGE como melhor GNN do estudo:** A separação das matrizes $W_l$ e $W_r$ permite ao SAGE aprender quanto peso dar à própria feature do nó vs. à agregação dos vizinhos — grau de liberdade que o GCN não possui. Nos grafos estrela do Bluesky, onde o nó raiz (post original) é qualitativamente diferente dos nós folha (replies), essa separação é particularmente relevante.

**Eficiência do SAGE:** Menor tempo por época (6.67s vs 8.24s do GCN e 11.35s do GAT), com apenas 2× mais parâmetros que o GCN e melhor F1. Relação custo-benefício superior ao GAT (3.8× mais parâmetros, F1=0).

### 6.2 Limitações identificadas

**Ausência de class weights:** `CrossEntropyLoss()` sem `pos_weight` ou `class_weight` trata as classes igualmente apesar do desbalanceamento. Isso penaliza o recall da classe minoritária (Fake) — explicação para Recall < 0.40 em todos os modelos.

**Grafos rasos/homogêneos:** Com apenas 1–3 hops de profundidade e topologia estrela, as três arquiteturas recebem informações de vizinhança muito semelhantes. Grafos mais profundos (redes sociais em cascata) revelariam diferenças maiores entre GCN, GAT e SAGE.

**Sem feature engineering:** Apenas embeddings BERT brutos, sem: (a) features estruturais do grafo (grau, centralidade), (b) features temporais (intervalo entre posts), (c) features do usuário (histórico, credibilidade).

**GAT sem diagnóstico:** O script não salva os coeficientes de atenção treinados. Um experimento diagnóstico (visualizar distribuição dos $\alpha_{ij}$ finais) confirmaria ou refutaria a hipótese de colapso de atenção formalmente.

### 6.3 Comparação com o estado da arte

| Abordagem | Dataset | Accuracy | F1 | Fonte |
|-----------|---------|----------|----|-------|
| SAGE (este TCC) | Bluesky | 0.9946 | 0.4592 | — |
| GCN (este TCC) | Bluesky | 0.9947 | 0.4295 | — |
| GAT (este TCC) | Bluesky | 0.9942 | 0.0000 | — |
| GCN | FakeNewsNet | 0.7100 | — | Krzywda et al. (2024) |
| GraphSAGE | FakeNewsNet | — | — | Krzywda et al. (2024) |
| RoBERTa | FakeNewsNet | 0.8616 | — | Krzywda et al. (2024) |
| RoBERTa | ISOT | 0.9999 | — | Krzywda et al. (2024) |
| BERT + GNN (híbrido) | Bluesky | — | — | N/A (não testado) |

> 📖 **Krzywda, M. et al. (2024)** — "Comparative Analysis of Graph Neural Networks and Transformers for Robust Fake News Detection: A Verification and Reimplementation Study"
> *Electronics 2024, 13(23), 4784*
> DOI: `10.3390/electronics13234784`
> **Localização:** Tabela de resultados (Seção 4), discussão GNN vs Transformer (Seção 5)
> **Relevância:** Único benchmark peer-reviewed que compara diretamente GNNs (incluindo GCN e SAGE) com Transformers em múltiplos datasets de fake news. Confirma o padrão observado neste TCC: GNNs com features textuais simples ficam atrás de modelos NLP.

**Observação sobre comparação entre datasets:** A accuracy ~0.9945 no Bluesky parece superior à do GCN em FakeNewsNet (71%), mas isso é ilusório — o Bluesky tem desbalanceamento de classes muito maior, inflando a accuracy. F1 é a métrica comparável.

### 6.4 Resposta parcial à questão do TCC

> **"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP?"**

Este script contribui com três evidências concretas:

**1. Hierarquia interna das GNNs:** SAGE > GCN > GAT no dataset Bluesky, com GAT colapsando completamente. A escolha da arquitetura GNN importa — não basta usar "uma GNN qualquer". A falha do GAT em grafos homogêneos/rasos, teoricamente fundamentada por Brody et al. (2022), sugere que a literatura de fake news deve reportar explicitamente a profundidade e homofilia dos grafos ao comparar arquiteturas.

**2. F1 ainda baixo para uso prático:** Mesmo o melhor modelo (SAGE, F1=0.4592) estaria abaixo do limiar de utilidade prática em sistemas de moderação de conteúdo. Modelos NLP como RoBERTa atingem F1 > 0.85 no mesmo tipo de tarefa — a diferença é substancial.

**3. Custo-benefício favorável do SAGE vs GCN:** Ganho de F1=+0.0297 com apenas 2× mais parâmetros e *menor* tempo de treinamento. Para aplicações resource-constrained onde o NLP é inviável, SAGE é a GNN recomendada entre as três testadas.

**Conclusão parcial:** GNNs aplicadas a grafos de propagação com features BERT brutas são **insuficientes** como solução standalone. O caminho promissor apontado pela literatura é o modelo **híbrido** (NLP para features de texto + GNN para estrutura de propagação), que combina as forças dos dois paradigmas.

---

## 7. Análise de Código

### 7.1 Erros identificados

**E1 — Ausência de class weights (impacto alto):**
```python
# ❌ Linha 78 — CrossEntropyLoss sem ponderação de classe
criterion = torch.nn.CrossEntropyLoss()

# ✅ Correção:
# Calcular proporção de classes no treino:
n_real = sum(1 for d in train_data if d.y.item() == 0)
n_fake = sum(1 for d in train_data if d.y.item() == 1)
weight = torch.tensor([1.0, n_real / n_fake], device=device)
criterion = torch.nn.CrossEntropyLoss(weight=weight)
# Justificativa: com desbalanceamento severo, o modelo aprende a predizer "Real"
# constantemente, o que maximiza accuracy mas colapsa F1 da classe Fake.
```

**E2 — Cálculo incorreto do avg_loss (baixo impacto, mas impreciso):**
```python
# ❌ Linha 117 — divide pelo número de grafos no loader (redundante/incorreto)
avg_loss = total_loss / sum(d.num_graphs for d in train_loader)
# Problema: itera o loader completo UMA segunda vez para somar num_graphs

# ✅ Correção mais eficiente:
# Acumular num_graphs durante o loop de treino:
n_total = 0
for data in train_loader:
    ...
    total_loss += loss.item() * data.num_graphs
    n_total += data.num_graphs
avg_loss = total_loss / n_total
```

**E3 — Labels como lista vs. escalar (robustez):**
```python
# ⚠️ Linhas 113–114 — workaround para labels escalares
val_y_true.extend(labels if isinstance(labels, list) else [labels])
# Isso funciona mas é frágil. Melhor:
val_y_true.extend(data.y.squeeze().cpu().tolist()
                  if data.y.squeeze().dim() > 0
                  else [data.y.squeeze().cpu().item()])
```

### 7.2 Ineficiências

**I1 — Treino sequencial obrigatório:** Os três modelos são treinados sequencialmente (linhas 416–428). Em máquinas com múltiplas GPUs, poderiam ser paralelizados com `torch.multiprocessing`. Com uma única GPU (como no ambiente do TCC), não há alternativa.

**I2 — `global_mean_pool` pode ser substituído por readout mais expressivo:** Para grafos estrela, um readout que pondera o nó raiz separadamente dos nós folha poderia extrair sinal melhor. Implementável com atenção global (`GlobalAttention` do PyG).

**I3 — Sem normalização de features de entrada:** Os embeddings BERT-768 têm escala variável entre dimensões. `BatchNorm1d(768)` antes da primeira camada GNN pode estabilizar o treinamento, especialmente para o GAT.

### 7.3 Boas práticas observadas

- **Seed determinístico** (`torch.manual_seed(12345)` em todos os modelos): garante reprodutibilidade dos resultados para a monografia.
- **Best-model checkpointing** (linhas 123–125 e 131–132): avalia no melhor ponto de validação, não no último epoch — prática correta de model selection.
- **Interface unificada** dos três modelos (`out, h = model(x, edge_index, batch)`): permite o loop de treino genérico em `treinar()` sem código especializado por arquitetura.
- **Relatório automático com interpretação** (`salvar_relatorio()`): gera conclusão para monografia automaticamente baseada no ranking de F1 — reduz risco de erro de interpretação.
- **Contagem de parâmetros** via `count_parameters()` nos três modelos: facilita análise de custo-benefício.

---

## 8. Referências Bibliográficas

1. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017. Disponível em: `https://arxiv.org/abs/1706.02216`

2. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. *5th International Conference on Learning Representations (ICLR 2017)*, 2017. Disponível em: `https://arxiv.org/abs/1609.02907`

3. VELIČKOVIĆ, P.; CUCURULL, G.; CASANOVA, A.; ROMERO, A.; LIÒ, P.; BENGIO, Y. **Graph Attention Networks**. *6th International Conference on Learning Representations (ICLR 2018)*, 2018. Disponível em: `https://arxiv.org/abs/1710.10903`

4. BRODY, S.; ALON, U.; YAHAV, E. **How Attentive are Graph Attention Networks?** *10th International Conference on Learning Representations (ICLR 2022)*, 2022. Disponível em: `https://arxiv.org/abs/2105.14491`

5. GILMER, J.; SCHÜTT, K. T.; UNKE, O. T.; GASTEGGER, M.; SMIDT, T.; MÜLLER, K. R. **Neural Message Passing for Quantum Chemistry**. *34th International Conference on Machine Learning (ICML 2017)*, 2017. Disponível em: `https://arxiv.org/abs/1704.01212`

6. KINGMA, D. P.; BA, J. **Adam: A Method for Stochastic Optimization**. *3rd International Conference on Learning Representations (ICLR 2015)*, 2015. Disponível em: `https://arxiv.org/abs/1412.6980`

7. GONG, S.; SINNOTT, R. O.; QI, J.; PARIS, C. **Fake News Detection Through Graph-based Neural Networks: A Survey**. *arXiv preprint*, 2023. Disponível em: `https://arxiv.org/abs/2307.12639`

8. KRZYWDA, M. et al. **Comparative Analysis of Graph Neural Networks and Transformers for Robust Fake News Detection: A Verification and Reimplementation Study**. *Electronics*, v. 13, n. 23, p. 4784, 2024. DOI: `10.3390/electronics13234784`

9. PHAN, H. T. et al. **Fake news detection: A survey of graph neural network methods**. *Applied Soft Computing*, v. 139, 2023. DOI: `10.1016/j.asoc.2023.110235`

10. HAMILTON, W. L. **Graph Representation Learning**. Morgan & Claypool, 2020. (Seções 4–5 sobre GraphSAGE e agregadores)

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Indutivo (inductive learning) | Capacidade de generalizar para nós/grafos não vistos durante o treino, aprendendo funções de agregação em vez de embeddings fixos | Hamilton et al. (2017), Seção 1 |
| Transdutivo (transductive learning) | Aprendizado que só funciona para nós presentes durante o treino — o GCN é transdutivo por usar a estrutura global do grafo | Hamilton et al. (2017), Seção 1 |
| Atenção estática (static attention) | Coeficientes de atenção do GAT original cujo ranking é independente do nó consultante — limitação formal identificada por Brody et al. (2022) | Brody et al. (2022), Seção 3 |
| SAGEConv | Implementação PyG da camada GraphSAGE: $h_v = W_l h_v + W_r \cdot \text{mean}_{u \in N(v)} h_u$ | Hamilton et al. (2017) + PyG docs |
| global_mean_pool | Readout do PyG: média dos embeddings de todos os nós de um grafo — produz representação de tamanho fixo para classificação | Gilmer et al. (2017), Equação de readout |
| Weight decay | Regularização L2 implícita no otimizador Adam — adiciona $\lambda \|\theta\|^2$ à loss para evitar overfitting | Loshchilov & Hutter (2019) — AdamW |
| ReduceLROnPlateau | Scheduler que reduz a LR por um fator quando a métrica monitorada para de melhorar por N épocas (patience) | PyTorch docs |
| Colapso de atenção | Estado degenerado onde os coeficientes $\alpha_{ij}$ do GAT convergem para valores uniformes, equivalendo a uma média simples sem poder discriminativo | Brody et al. (2022) |
| Grafos estrela | Topologia onde um nó central (post original) conecta-se a $n$ folhas (replies) sem conexões entre folhas — estrutura predominante no dataset Bluesky | Observação empírica deste TCC |
