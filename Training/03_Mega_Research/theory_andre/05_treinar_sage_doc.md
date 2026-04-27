# Documentação Técnica: 05_treinar_sage.py + sage_model.py

## Metadados

- **Arquivos analisados:** `05_treinar_sage.py`, `sage_model.py`
- **Caminho:** `03_Mega_Research/05_treinar_sage.py`, `03_Mega_Research/sage_model.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** GNN — classificação de grafos com GraphSAGE (indutivo)
- **Modelos principais:** SAGEClassifier (3 × SAGEConv, ~115,010 parâmetros)
- **Datasets utilizados:** Bluesky (grafos `.pt` do script 02)
- **Contribuição para a questão central:** Este script introduz o GraphSAGE como terceira arquitetura GNN no benchmark do TCC. Diferentemente de GCN e GAT, o SAGE é **indutivo** — aprende funções de agregação generalizáveis a nós nunca vistos durante o treino. As 4 variantes com class weights progressivos ([1.0,1.0], [1.0,4.0], [1.0,6.0]) permitem avaliar se penalizar a classe "fake" melhora recall sem destruir precision — evidência direta sobre viabilidade de GNNs em cenários de detecção com desequilíbrio de classes.

---

## 1. Visão Geral dos Scripts

`05_treinar_sage.py` treina 4 variantes do `SAGEClassifier` nos mesmos grafos Bluesky usados pelos scripts anteriores (GCN e GAT). As variantes diferem exclusivamente nos pesos da função de perda `CrossEntropyLoss`: duas usam pesos balanceados [1.0, 1.0] e duas usam pesos crescentemente enviesados para a classe fake ([1.0, 4.0] e [1.0, 6.0]). O script incorpora **early stopping** (PATIENCE=7) além do `ReduceLROnPlateau`, tornando-o mais robusto contra overfitting que os scripts anteriores. Cada variante salva seus pesos em `Execution/weights/` e reporta métricas completas no conjunto de teste.

`sage_model.py` implementa o `SAGEClassifier`: três camadas `SAGEConv` com ReLU (a terceira sem ativação), seguidas de `global_mean_pool`, Dropout(0.5) e um classificador linear. O agregador é parametrizável (`mean`, `max`, `lstm`), com `mean` como padrão — equivalente ao SAGE original de Hamilton et al. (2017). A interface `out, h = model(x, edge_index, batch)` é idêntica ao GCN e GAT, mantendo a intercambiabilidade entre arquiteturas.

**Comparação arquitetural completa das três GNNs do TCC:**

| Aspecto | GCNClassifier | GATClassifier | SAGEClassifier |
|---------|--------------|--------------|----------------|
| Mecanismo de agregação | Normalização por grau (fixa) | Atenção aprendida | Média + self (W_l e W_r separados) |
| Indutivo | Não | Não | Sim |
| Multi-head | Não | 4 heads | Não |
| Ativação | ReLU | ELU | ReLU |
| Dropout entre camadas | Não | 0.3 | Não |
| Dropout pré-classificador | 0.5 | 0.5 | 0.5 |
| Parâmetros | ~57,666 | ~218,562 | ~115,010 |
| Class weights | Não | Não | Sim (4 variantes) |
| Early stopping | Não | Não | Sim (PATIENCE=7) |

---

## 2. Arquitetura e Componentes Principais

### 2.1 GraphSAGE (SAGEConv)

**Descrição técnica:**
GraphSAGE (Graph Sample and Aggregate) aprende funções de agregação de vizinhança em vez de embeddings transductivos por nó. Em vez de aprender uma representação para cada nó individualmente (como o GCN em modo semi-supervisionado), o SAGE aprende como agregar features de vizinhos — permitindo generalizar para nós não vistos durante o treino.

**Fundamento matemático:**

**Algoritmo de propagação (Hamilton et al. 2017, Algoritmo 1, linhas 4–5):**

$$h_{\mathcal{N}(v)}^{(l)} = \text{AGGREGATE}^{(l)}\!\left(\left\{h_u^{(l-1)},\, \forall\, u \in \mathcal{N}(v)\right\}\right)$$

$$h_v^{(l)} = \sigma\!\left(W^{(l)} \cdot \text{CONCAT}\!\left(h_v^{(l-1)},\, h_{\mathcal{N}(v)}^{(l)}\right)\right)$$

onde:
- $h_v^{(l)}$ — embedding do nó $v$ na camada $l$; $h_v^{(0)} = x_v$ (feature original)
- $\mathcal{N}(v)$ — conjunto de vizinhos de $v$ (com sample no paper original; sem sample no script, usa toda a vizinhança)
- $\text{AGGREGATE}^{(l)}$ — função de agregação da camada $l$ (mean, max ou LSTM)
- $W^{(l)} \in \mathbb{R}^{2d \times d'}$ — matriz de pesos que projeta a concatenação de self e vizinhos
- $\sigma(\cdot)$ — ReLU no script

**Implementação PyG (SAGEConv) — duas matrizes separadas:**

Na implementação do PyTorch Geometric, a concatenação e projeção são equivalentemente reformuladas com duas matrizes separadas (forma "add" em vez de concatenação explícita):

$$h_v^{(l)} = \sigma\!\left(W_l^{(l)}\, h_v^{(l-1)} + W_r^{(l)}\, \frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} h_u^{(l-1)}\right)$$

onde:
- $W_l^{(l)} \in \mathbb{R}^{d \times d'}$ — matriz de transformação do self-embedding (self-loop)
- $W_r^{(l)} \in \mathbb{R}^{d \times d'}$ — matriz de transformação dos vizinhos agregados
- O bias não é incluído por padrão no SAGEConv do PyG

Esta formulação é algebraicamente equivalente à concatenação com projeção única quando a concatenação é seguida por uma linear layer não compartilhada.

**Por que SAGE usa duas matrizes ($W_l$ e $W_r$) em vez de uma ($W$ do GCN):**
O GCN trata o self-loop como um vizinho adicional com peso fixo ($1/\sqrt{\tilde{d}_i \tilde{d}_j}$) e usa uma única matriz. O SAGE aprende pesos **diferentes** para a contribuição do nó consigo mesmo ($W_l$) e para a contribuição dos vizinhos ($W_r$), dando ao modelo mais capacidade expressiva para balancear informação local vs. de vizinhança — ao custo de ~2× os parâmetros por camada.

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *31st Conference on Neural Information Processing Systems (NeurIPS 2017)*
> arXiv: `1706.02216`
> **Localização:** Seção 3 (Proposed Method: GraphSAGE), Algoritmo 1 (linhas 4–5), Seção 3.1 (Aggregator Architectures), Equações 1–2
> **Relevância:** Paper original que define o algoritmo GraphSAGE. O Algoritmo 1 corresponde diretamente ao forward pass de `sage_model.py` (linhas 118–129). As Equações 1–2 definem $h_v^{(l)}$ conforme implementado pelo `SAGEConv`. A Seção 3.1 descreve os três agregadores (mean, max, LSTM) disponíveis via `--aggr` no script 05.

**No código (`sage_model.py`):**
> Linhas 78–82: `self.conv1 = SAGEConv(768, 64, aggr=aggr)` — primeira camada SAGE.
> Linhas 85–96: `self.conv2`, `self.conv3` — camadas adicionais de agregação.
> Linhas 118–129: forward pass com `conv1 → relu → conv2 → relu → conv3 → global_mean_pool`.

---

### 2.2 Contagem de Parâmetros do SAGEClassifier

**Derivação exata para aggr='mean':**

| Camada | Operação | Cálculo | Parâmetros |
|--------|----------|---------|-----------|
| conv1 | $W_l \in \mathbb{R}^{768 \times 64}$ (self) | $768 \times 64$ | 49,152 |
| conv1 | $W_r \in \mathbb{R}^{768 \times 64}$ (vizinhos) | $768 \times 64$ | 49,152 |
| conv1 | bias $\in \mathbb{R}^{64}$ | 64 | 64 |
| conv2 | $W_l \in \mathbb{R}^{64 \times 64}$ | $64 \times 64$ | 4,096 |
| conv2 | $W_r \in \mathbb{R}^{64 \times 64}$ | $64 \times 64$ | 4,096 |
| conv2 | bias $\in \mathbb{R}^{64}$ | 64 | 64 |
| conv3 | (idem conv2) | — | 8,256 |
| lin | $W \in \mathbb{R}^{64 \times 2}$, bias $\in \mathbb{R}^2$ | $64 \times 2 + 2$ | 130 |
| **Total** | | | **115,010** |

O SAGEClassifier tem ~2× mais parâmetros que o GCN e ~0.53× do GAT. No espectro de complexidade GCN < SAGE < GAT, ocupa uma posição intermediária.

---

### 2.3 Propriedade Indutiva do GraphSAGE

**Descrição técnica:**
Uma arquitetura é **indutiva** quando pode produzir embeddings para nós não vistos durante o treinamento sem re-treinar o modelo. O GCN em seu modo semi-supervisionado original é **transdutivo** — os embeddings dependem da estrutura total do grafo, incluindo nós de teste, durante o treino.

**Por que GraphSAGE é indutivo:**
O modelo aprende *funções de agregação* $W_l^{(l)}$ e $W_r^{(l)}$ em vez de embeddings por nó. Para um nó novo $v'$ com features $x_{v'}$ e vizinhos $\mathcal{N}(v')$, o forward pass computa $h_{v'}^{(l)}$ usando as mesmas matrizes treinadas — sem re-otimização.

**Relevância para grafos de notícias Bluesky:**
No pipeline atual do TCC, cada notícia é um grafo separado (grafo de propagação de repost). Os conjuntos train/val/test contêm grafos completamente distintos — portanto, todos os modelos (GCN, GAT, SAGE) operam de forma indutiva na prática (classificam grafos novos, não nós novos em um grafo existente). A vantagem indutiva do SAGE se tornaria relevante em um cenário de streaming em produção onde novos usuários/posts aparecem em um grafo contínuo.

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *NeurIPS 2017*, arXiv: `1706.02216`
> **Localização:** Seção 1 (Introduction), Seção 3 (parágrafo de abertura: "inductive setting")
> **Relevância:** O próprio título do paper enfatiza "inductive". A Seção 1 explica explicitamente que métodos transductivos como o GCN original não generalizam a nós ausentes durante o treino.

> 📖 **Kipf, T. N.; Welling, M. (2017)** — "Semi-Supervised Classification with Graph Convolutional Networks"
> *ICLR 2017*, arXiv: `1609.02907`
> **Localização:** Seção 2.1 (nota de rodapé / discussão sobre limitação transdutiva)
> **Relevância:** Confirma que o GCN original opera em modo transdutivo (todos os nós presentes durante o treino), contrastando com o SAGE.

---

### 2.4 Agregadores do GraphSAGE

O script expõe três agregadores via `--aggr`:

#### Mean Aggregator (padrão)

$$h_{\mathcal{N}(v)}^{\text{mean}} = \frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} h_u^{(l-1)}$$

Invariante à permutação dos vizinhos. Equivalente a uma convolução espectral de primeira ordem. É o agregador padrão e o descrito na Equação 2 do paper SAGE.

#### Max-Pooling Aggregator

$$h_{\mathcal{N}(v)}^{\text{max}} = \max_{u \in \mathcal{N}(v)}\!\left(\sigma\!\left(W_{\text{pool}}\, h_u^{(l-1)} + b\right)\right)$$

Capta a feature mais proeminente entre os vizinhos. Útil quando a presença de um único vizinho com feature extrema é informativa (ex: um post viral específico na propagação).

#### LSTM Aggregator

$$h_{\mathcal{N}(v)}^{\text{lstm}} = \text{LSTM}\!\left(\left[h_{u_1}^{(l-1)}, h_{u_2}^{(l-1)}, \ldots\right]\right)$$

Opera sobre sequência de vizinhos em ordem arbitrária — **não é invariante à permutação**. Para grafos estrela (estrutura dos grafos Bluesky), a ordem dos filhos é determinística pela construção do grafo (não há garantia de ordem semântica), tornando este agregador potencialmente instável.

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *NeurIPS 2017*, arXiv: `1706.02216`
> **Localização:** Seção 3.1 (Aggregator Architectures), Equações 3–5
> **Relevância:** Define os três agregadores implementados no script. A Seção 3.1 discute as propriedades de invariância à permutação ("a good aggregator should be symmetric"), o que embasa a limitação do LSTM em grafos sem ordem semântica definida.

---

### 2.5 Global Mean Pooling (Readout)

**Descrição técnica:**
Após o message passing, os embeddings de nós individuais precisam ser agregados em um embedding único por grafo para a tarefa de classificação de grafos. O `global_mean_pool` calcula a média dos embeddings de todos os nós do grafo:

$$h_G = \frac{1}{|V|}\sum_{v \in V} h_v^{(L)}$$

onde $L$ é a última camada de message passing e $|V|$ é o número de nós no grafo.

**Embasamento acadêmico:**

> 📖 **Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. (2019)** — "How Powerful are Graph Neural Networks?"
> *7th International Conference on Learning Representations (ICLR 2019)*
> arXiv: `1810.00826`
> **Localização:** Seção 2 (Preliminaries), Seção 4.2 (Graph-Level Readout)
> **Relevância:** Discute as limitações de expressividade do mean pooling para readout de grafos — em particular, a Seção 4.2 demonstra que mean pooling não distingue grafos com a mesma distribuição de features mas estruturas diferentes. No contexto de grafos estrela com features homogêneas (todos os filhos com o mesmo BERT), essa limitação é diretamente relevante.

**No código:**
> Linha 129 (`sage_model.py`): `h = global_mean_pool(x, batch)` — readout idêntico ao GCN e GAT.

---

## 3. Pipeline de Treinamento

### 3.1 Fluxo Completo

```
1. Carregar train/val/test .pt (script 02)
2. Para cada variante em {baseline, ctrl, cetico, ex_cetico}:
   a. Instanciar SAGEClassifier(768, 2, aggr=args.aggr)
   b. CrossEntropyLoss(weight=peso_ce)
   c. Adam(lr, weight_decay=5e-4)
   d. ReduceLROnPlateau(mode='max', factor=0.5, patience=5)
   e. Loop de treino:
      - treinar_epoca → (loss, train_acc)
      - avaliar val_loader → val_f1 (macro)
      - scheduler.step(val_f1)
      - early stopping se epochs_sem_melhora >= 7
      - best_state = model.state_dict() quando val_f1 melhora
   f. model.load_state_dict(best_state)
   g. avaliar_teste → {acc, prec, rec, f1-binary}
   h. torch.save(model.state_dict(), WEIGHTS_DIR / config["arquivo"])
3. Imprimir tabela comparativa das 4 variantes
```

### 3.2 Variantes e Class Weights

| Nome | Arquivo salvo | Pesos CE | Efeito esperado |
|------|--------------|----------|-----------------|
| baseline | `pesos_sage.pth` | [1.0, 1.0] | Aprendizado neutro |
| ctrl | `pesos_sage_bs_ctrl.pth` | [1.0, 1.0] | Idêntico ao baseline (ver Seção 7.1) |
| cetico | `pesos_sage_bs_cetico.pth` | [1.0, 4.0] | Penaliza 4× erros em fake → recall↑, precision↓ |
| ex_cetico | `pesos_sage_bs_ex_cetico.pth` | [1.0, 6.0] | Penaliza 6× erros em fake → recall↑↑, precision↓↓ |

---

## 4. Métricas de Avaliação

### 4.1 F1-Macro (critério de seleção do modelo)

$$F1_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} F1_c = \frac{1}{2}\left(F1_{\text{real}} + F1_{\text{fake}}\right)$$

Trata as duas classes igualmente, independente do tamanho. Usado como critério de validação e early stopping (linha 112 do script de treino).

### 4.2 F1 Binary (métrica de teste)

$$F1_{\text{binary}} = \frac{2 \cdot \text{Precision}_{\text{fake}} \cdot \text{Recall}_{\text{fake}}}{\text{Precision}_{\text{fake}} + \text{Recall}_{\text{fake}}}$$

Mede apenas a classe positiva (fake = label 1). Padrão do `sklearn.f1_score` sem `average=`. Usado no relatório final (linha 221 de `avaliar_teste`).

**Inconsistência entre critério de seleção e métrica de reporte:** O modelo é selecionado por F1-macro na validação mas reportado por F1-binary no teste. Para datasets balanceados os valores convergem; para datasets desbalanceados podem divergir. Esta limitação é herdada dos scripts anteriores.

**Embasamento acadêmico:**

> 📖 **Sokolova, M.; Lapalme, G. (2009)** — "A systematic analysis of performance measures for classification tasks"
> *Information Processing & Management*, v. 45, n. 4, pp. 427–437
> DOI: `10.1016/j.ipm.2009.03.002`
> **Localização:** Seção 3 (Performance Measures for Binary Classification), Tabela 2
> **Relevância:** Define formalmente F1-macro e F1-binary, documentando a diferença de comportamento em datasets desbalanceados.

### 4.3 Aprendizado Sensível ao Custo (CrossEntropyLoss com pesos)

**Descrição técnica:**
A `CrossEntropyLoss` com pesos de classe pondera a contribuição de cada amostra à perda pelo peso da sua classe verdadeira:

$$\mathcal{L} = -\sum_{i=1}^{N} w_{y_i} \log\!\left(\frac{e^{z_{y_i}}}{\sum_k e^{z_k}}\right)$$

onde:
- $w_c$ — peso da classe $c$ (ex: $w_0 = 1.0$, $w_1 = 4.0$ para a variante "cético")
- $z_k$ — logit da classe $k$
- $y_i$ — label verdadeiro da amostra $i$

**Efeito nos gradientes:** Erros na classe "fake" ($y_i = 1$) com $w_1 = 4.0$ contribuem 4× mais ao gradiente que erros na classe "real". O modelo aprende a ser mais conservador com fake — diminui falsos negativos (aumenta recall) ao custo de mais falsos positivos (diminui precision).

**Embasamento acadêmico:**

> 📖 **Elkan, C. (2001)** — "The Foundations of Cost-Sensitive Learning"
> *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001)*, pp. 973–978
> Disponível em: https://dl.acm.org/doi/10.5555/1642194.1642224
> **Localização:** Seção 2 (Cost-Sensitive Classifiers), Teorema 1
> **Relevância:** Fundamenta teoricamente a abordagem de class weights como forma de aprendizado sensível ao custo. O Teorema 1 demonstra que pesar amostras por $w_c$ é equivalente a otimizar para um limiar de decisão ajustado — evidenciando que as variantes "cético" e "ex-cético" estão implicitamente ajustando o ponto de operação no espaço precision-recall.

> 📖 **King, G.; Zeng, L. (2001)** — "Logistic Regression in Rare Events Data"
> *Political Analysis*, v. 9, n. 2, pp. 137–163
> DOI: `10.1093/pan/9.2.137`
> **Localização:** Seção 3 (Correcting for Prior Imbalance)
> **Relevância:** Demonstra empiricamente que modelos treinados sem correção para desbalanceamento de classes subrepresentam a classe minoritária. No contexto do TCC, se fake news for a classe minoritária no dataset Bluesky, as variantes com pesos [1.0, 4.0] e [1.0, 6.0] são metodologicamente justificadas.

---

## 5. Early Stopping

**Descrição técnica:**
Early stopping interrompe o treinamento quando a métrica de validação não melhora por `PATIENCE` épocas consecutivas, restaurando os pesos do melhor checkpoint:

```
if val_f1 > best_val_f1:
    best_state = model.state_dict()  # snapshot
    epochs_sem_melhora = 0
else:
    epochs_sem_melhora += 1
    if epochs_sem_melhora >= PATIENCE:
        break  # para o loop
model.load_state_dict(best_state)  # restaura melhor
```

**Interação com ReduceLROnPlateau (PATIENCE=5 do scheduler vs. PATIENCE=7 do early stopping):**
O scheduler reduz o LR pela metade na época 6 sem melhoria; o early stopping para na época 7 sem melhoria. Há apenas 1 época de "segunda chance" após a redução de LR antes do stop. Em prática, se a redução de LR não trouxer melhoria imediata em 1 época, o treino para — o scheduler tem pouco tempo para ter efeito.

**Embasamento acadêmico:**

> 📖 **Prechelt, L. (1998)** — "Early Stopping — But When?"
> In: **Orr, G. B.; Müller, K.-R.** (eds.) *Neural Networks: Tricks of the Trade*. Lecture Notes in Computer Science, vol. 1524. Springer, Berlin, pp. 55–69
> DOI: `10.1007/3-540-49430-8_3`
> **Localização:** Seção 2 (The Early Stopping Criteria), Critério GL (Generalization Loss), Seção 3 (Experimental Results)
> **Relevância:** Define formalmente o critério de early stopping por plateau de validação (equivalente ao implementado no script). A Seção 3 demonstra empiricamente que a restauração do melhor estado (implementada no script via `best_state`) é essencial — parar o treino sem restaurar pode resultar em um modelo com pior generalização do que o checkpoint guardado.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos Fortes desta Abordagem

**F1 — Early stopping + melhor checkpoint:** O script preserva o modelo com melhor F1-macro na validação, não o modelo da última época. Isso é metodologicamente mais robusto que os scripts anteriores (GCN e GAT), que também faziam early-stopping via best_state mas sem o mecanismo de contagem de patience explícito.

**F2 — Variantes de class weights como ablation study parcial:** As 4 variantes permitem visualizar o tradeoff precision-recall conforme o peso da classe fake cresce de 1.0 para 6.0 — análise útil para a discussão de custo operacional (custo de falso negativo em fake news é alto).

**F3 — Agregador configurável (`--aggr`):** Permite comparar mean vs. max no mesmo script sem alterar código — possibilita uma micro-ablation adicional sobre o mecanismo de agregação dentro do SAGE.

**F4 — Relatório final em tabela:** A saída estruturada no terminal facilita a transcrição direta para tabelas do TCC.

### 6.2 Limitações Identificadas

**L1 — Variantes "baseline" e "ctrl" são funcionalmente idênticas:**
Ambas usam `peso_ce = [1.0, 1.0]` e o mesmo modelo, LR e arquitetura (ver Seção 7.1). Os resultados serão diferentes apenas por estocasticidade no DataLoader shuffle e na inicialização de pesos se a seed não for diferente — e de fato não é, pois o `torch.manual_seed(12345)` em `SAGEClassifier.__init__` é chamado toda vez que um novo modelo é instanciado, gerando a mesma inicialização. A diferença entre baseline e ctrl será apenas o shuffle do DataLoader (sem seed fixada).

**L2 — Caminhos hardcoded (mesma limitação dos scripts 03 e 04):**
`DATA_DIR / "grafos_bluesky_*.pt"` pressupõe que o script 02 salvou na mesma pasta (`03_Mega_Research/data/`). Se o script 02 usar `--data-suffix`, o path não corresponde.

**L3 — F1 binary no teste vs F1-macro na validação:**
Inconsistência herdada dos scripts anteriores — o critério de seleção (macro) difere da métrica reportada (binary).

**L4 — DataLoader sem seed:**
`DataLoader(train_data, shuffle=True)` sem `generator=torch.Generator().manual_seed(42)` — resultados não são exatamente reprodutíveis entre runs.

**L5 — LSTM aggregator instável em grafos estrela sem ordem de vizinhos definida:**
Os grafos Bluesky são estrelas onde a ordem dos filhos depende da ordem de inserção durante a construção (script 02). A ordem não tem significado semântico, tornando o agregador LSTM potencialmente prejudicial.

**L6 — Import dentro de função (`avaliar_teste`, linha 203):**
`from sklearn.metrics import ...` dentro da função é ineficiente — reimporta o módulo a cada chamada. Funcional mas viola convenções PEP8.

### 6.3 Comparação com Estado da Arte

| Modelo | Acc | F1 | Dataset | Notas |
|--------|-----|-----|---------|-------|
| GCN (UPFD, bert) | 83.26% | 83.14% | PolitiFact | Dou et al. (2021) |
| GraphSAGE (UPFD, bert) | 82.17% | 82.02% | PolitiFact | Dou et al. (2021) |
| GAT (UPFD, spacy) | 81.17% | 81.10% | PolitiFact | Dou et al. (2021) |
| SAGEClassifier (este script) | — | — | Bluesky | — |

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *SIGIR'21: Proceedings of the 44th International ACM SIGIR Conference*, pp. 2051–2055
> DOI: `10.1145/3404835.3462990`
> **Localização:** Seção 4.2 (Baselines), Tabela 3

**Nota importante:** Os dados acima usam o dataset PolitiFact do FakeNewsNet com features spaCy/BERT, que é diferente do dataset Bluesky usado neste TCC. A comparação direta de números não é válida — serve apenas como referência de ordem de grandeza para o desempenho de GraphSAGE com features BERT em grafos de fake news.

### 6.4 Resposta Parcial à Questão do TCC

Este script contribui de duas formas específicas para a questão "GNNs são viáveis para detecção de fake news?":

1. **Posicionamento do GraphSAGE no espectro de complexidade:** Com ~115k parâmetros (2× GCN, 0.53× GAT) e a propriedade indutiva, o SAGE representa um meio-termo custo-benefício. Se o SAGE atingir performance similar ao GAT (4× mais parâmetros), argumenta-se que a complexidade adicional do GAT não se justifica.

2. **Evidência sobre class weights em GNNs para fake news:** A progressão de pesos [1.0,1.0] → [1.0,6.0] mostrará como o recall para fake news responde à penalização crescente. Se recall baixo for a limitação principal (falsos negativos = fake news não detectada), as variantes "cético" e "ex-cético" oferecem um mecanismo de ajuste operacional sem mudar a arquitetura — relevante para a discussão de limitações práticas de GNNs.

---

## 7. Análise de Código

### 7.1 Erros Identificados

```python
# ❌ Linhas 51–56 — "baseline" e "ctrl" são funcionalmente idênticas
VARIANTES = {
    "baseline":  {"peso_ce": [1.0, 1.0], "arquivo": "pesos_sage.pth"},
    "ctrl":      {"peso_ce": [1.0, 1.0], "arquivo": "pesos_sage_bs_ctrl.pth"},
    ...
}
# Ambas têm peso_ce=[1.0,1.0] e o mesmo modelo.
# O torch.manual_seed(12345) em SAGEClassifier.__init__ garante
# MESMA inicialização para ambas. A única diferença será o shuffle
# aleatório do DataLoader (sem seed). Os resultados divergirão
# apenas por estocasticidade não controlada — não por design.

# ✅ Se a intenção era replicar a estrutura de 4 variantes do ensemble
#    GCN/GAT com algo diferenciado em "ctrl", uma opção seria:
"ctrl": {"peso_ce": [1.0, 1.0], "arquivo": "pesos_sage_bs_ctrl.pth",
         "seed": 99999},  # semente diferente → inicialização diferente
# E passar o seed para SAGEClassifier(..., seed=config.get("seed", 12345))
```

```python
# ❌ Linhas 63–67 — caminhos hardcoded (mesmo problema dos scripts 03 e 04)
DATA_DIR / "grafos_bluesky_train.pt"
# Script 02 salva em data/bluesky_<suffix>/ quando --data-suffix é passado.

# ✅ Mesma correção já documentada nos scripts anteriores:
parser.add_argument("--data-suffix", type=str, default="",
    help="Sufixo do subdiretório de dados (ex: '_v2')")
DATA_DIR = Path(__file__).resolve().parent / "data" / f"bluesky{args.data_suffix}"
```

```python
# ❌ Linha 221 — f1_score sem average= explícito → F1 binary (default sklearn)
"f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
# Inconsistente com val_f1 macro usado no early stopping (linha 128).

# ✅ Reportar ambos explicitamente:
"f1_macro":  round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
"f1_binary": round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4),
```

```python
# ❌ Linhas 83–84 e 85–86 — DataLoaders sem seed de shuffle
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ✅ Correção:
g = torch.Generator().manual_seed(42)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          generator=g, num_workers=0)
```

```python
# ❌ Linha 203 — import dentro de função (ineficiente, anti-padrão PEP8)
def avaliar_teste(model, test_data: list, device: torch.device) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ...
# Reimporta o módulo a cada chamada de avaliar_teste.

# ✅ Mover para o topo do arquivo junto com os outros imports sklearn (linha 31).
```

```python
# ⚠️  Linhas 158–159 — scheduler PATIENCE=5 vs early stopping PATIENCE=7
# A janela entre redução de LR e stop é de apenas 1 época.
# Se a redução de LR não surtir efeito imediato (épocas 6→7), o treino para.
# Para dar ao LR reduzido mais tempo de convergir, sugerir:
PATIENCE_LR   = 5   # scheduler (atual)
PATIENCE_STOP = 12  # early stopping → pelo menos 7 épocas após LR reduction
```

### 7.2 Boas Práticas Observadas

**B1 — Acumulação de loss correta no loop de treino (linhas 107–112):**
`total_loss += loss.item() * data.num_graphs` seguido de `total_loss / total` — correto. Evita o bug do script 04 que recorria o `train_loader` para somar `num_graphs`.

**B2 — Early stopping com restauração do melhor checkpoint (linhas 173–193):**
`best_state = {k: v.clone() for k, v in model.state_dict().items()}` garante uma cópia profunda dos pesos no momento do melhor val_f1 — não uma referência mutável. Correto e seguro.

**B3 — `torch.manual_seed(seed)` no construtor do modelo (linha 72 do `sage_model.py`):**
Garante inicialização reprodutível quando o mesmo seed é usado. Permite comparações entre runs com a mesma semente.

**B4 — Agregador como parâmetro do construtor e da CLI:**
`SAGEClassifier(..., aggr=aggr)` e `parser.add_argument("--aggr", choices=["mean","max","lstm"])` — permite ablation do agregador sem alterar código.

**B5 — Variante selecionável via `--variante` (linhas 260–263):**
`{args.variante: VARIANTES[args.variante]}` permite treinar apenas uma variante — útil para retomar experimentos parciais sem re-treinar todas as 4.

**B6 — Interface drop-in com GCN e GAT:**
`out, h = model(x, edge_index, batch)` idêntico — permite trocar SAGE por GCN/GAT no loop `treinar_variante` sem modificação de código.

---

## 8. Referências Bibliográficas

1. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. In: *31st Conference on Neural Information Processing Systems (NeurIPS 2017)*. Disponível em: https://arxiv.org/abs/1706.02216

2. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. In: *5th International Conference on Learning Representations (ICLR 2017)*. Disponível em: https://arxiv.org/abs/1609.02907

3. XU, K.; HU, W.; LESKOVEC, J.; JEGELKA, S. **How Powerful are Graph Neural Networks?** In: *7th International Conference on Learning Representations (ICLR 2019)*. Disponível em: https://arxiv.org/abs/1810.00826

4. ELKAN, C. **The Foundations of Cost-Sensitive Learning**. In: *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001)*, pp. 973–978. Disponível em: https://dl.acm.org/doi/10.5555/1642194.1642224

5. KING, G.; ZENG, L. **Logistic Regression in Rare Events Data**. *Political Analysis*, v. 9, n. 2, pp. 137–163, 2001. DOI: `10.1093/pan/9.2.137`

6. SOKOLOVA, M.; LAPALME, G. **A systematic analysis of performance measures for classification tasks**. *Information Processing & Management*, v. 45, n. 4, pp. 427–437, 2009. DOI: `10.1016/j.ipm.2009.03.002`

7. PRECHELT, L. **Early Stopping — But When?** In: ORR, G. B.; MÜLLER, K.-R. (eds.). *Neural Networks: Tricks of the Trade*. Lecture Notes in Computer Science, vol. 1524. Springer, Berlin, pp. 55–69, 1998. DOI: `10.1007/3-540-49430-8_3`

8. SHCHUR, O.; MUMME, M.; BOJCHEVSKI, A.; GÜNNEMANN, S. **Pitfalls of Graph Neural Network Evaluation**. *Workshop on Relational Representation Learning, NeurIPS 2018*. Disponível em: https://arxiv.org/abs/1811.05868

9. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. In: *SIGIR'21: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 2051–2055, 2021. DOI: `10.1145/3404835.3462990`

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| GraphSAGE | Graph Sample and Aggregate — aprende funções de agregação de vizinhança generalizáveis a nós não vistos | Hamilton et al. (2017), Seção 3 |
| Indutivo | Capacidade de gerar embeddings para nós/grafos ausentes durante o treino sem re-otimização | Hamilton et al. (2017), Seção 1 |
| Transdutivo | Aprende embeddings somente para nós presentes durante o treino (ex: GCN original) | Kipf & Welling (2017), Seção 2.1 |
| $W_l$ e $W_r$ | Matrizes de transformação separadas do SAGEConv para self-embedding ($W_l$) e vizinhos ($W_r$) | Hamilton et al. (2017), Seção 3 |
| Mean Aggregator | Agrega features de vizinhos pela média aritmética — invariante à permutação | Hamilton et al. (2017), Seção 3.1 |
| Max-pooling Aggregator | Captura feature máxima entre vizinhos após transformação linear — invariante à permutação | Hamilton et al. (2017), Seção 3.1 |
| LSTM Aggregator | Processa vizinhos como sequência via LSTM — não invariante à permutação | Hamilton et al. (2017), Seção 3.1 |
| Early stopping | Interrompe o treino quando a métrica de validação não melhora por `PATIENCE` épocas | Prechelt (1998), Seção 2 |
| Aprendizado sensível ao custo | Técnica que atribui pesos diferentes a erros de classes distintas na função de perda | Elkan (2001), Seção 2 |
| Class weight | Fator $w_c$ que pondera a contribuição de amostras da classe $c$ na CrossEntropyLoss | Elkan (2001), Teorema 1 |
| F1-macro | Média simples do F1 de cada classe — trata classes desbalanceadas igualmente | Sokolova & Lapalme (2009), Tabela 2 |
| F1-binary | F1-Score calculado somente para a classe positiva (fake = 1) — padrão do sklearn | Sokolova & Lapalme (2009), Tabela 2 |
| ReduceLROnPlateau | Scheduler que reduz o LR por um fator quando a métrica de validação para de melhorar | — |
| Drop-in replacement | Componente substituível sem alterar o código circundante — GCN, GAT e SAGE partilham a interface `model(x, edge_index, batch)` | — |
