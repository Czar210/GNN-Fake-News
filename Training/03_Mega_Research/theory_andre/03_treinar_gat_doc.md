# Documentação Técnica: 03_treinar_gat.py + gat_model.py

## Metadados

- **Arquivos analisados:** `03_treinar_gat.py`, `gat_model.py`
- **Caminho:** `03_Mega_Research/03_treinar_gat.py`, `03_Mega_Research/gat_model.py`
- **Data de análise:** 2026-04-24
- **Tipo de abordagem:** GNN — Graph Attention Network para classificação de grafos
- **Modelos principais:** GATClassifier (3 × GATConv + global_mean_pool + Linear)
- **Datasets utilizados:** Bluesky (grafos `.pt` do script 02)
- **Contribuição para a questão central:** Implementa o modelo GNN com mecanismo de atenção no coração do TCC. As 4 variantes com diferentes pesos de classe permitem medir a sensibilidade do GAT ao desbalanceamento dos labels heurísticos do Bluesky. Os resultados deste script, confrontados com o baseline NLP (script 10), são a evidência direta para responder "GNNs são viáveis?".

---

## 1. Visão Geral do Script

`03_treinar_gat.py` orquestra o treinamento de 4 variantes do `GATClassifier` definido em `gat_model.py`. As variantes diferem apenas nos pesos da `CrossEntropyLoss`: duas usam pesos uniformes `[1.0, 1.0]` e duas penalizam a classe "fake" com pesos `[1.0, 4.0]` e `[1.0, 6.0]`, tentando compensar o desbalanceamento de classes esperado nos dados Bluesky com labels heurísticos.

O loop de treinamento usa Adam com weight decay, um scheduler `ReduceLROnPlateau` que reduz o LR pela metade quando o F1 de validação para de melhorar, e early stopping com paciência de 7 épocas. O critério de seleção do melhor modelo é o F1 macro no conjunto de validação — escolha adequada para datasets desbalanceados.

`gat_model.py` implementa a arquitetura GATClassifier com 3 camadas `GATConv`: as duas primeiras usam 4 cabeças de atenção com `concat=False` (média das heads, mantendo dimensão 64), e a terceira usa 1 cabeça como consolidação. O readout é `global_mean_pool`, que produz um vetor de grafo agregando todos os nós. Um classificador linear final projeta para 2 logits.

---

## 2. Arquitetura e Componentes Principais

### 2.1 Graph Attention Network (GAT)

**Descrição técnica:**
GAT introduz o mecanismo de atenção sobre vizinhanças de grafos: em vez de agregar features dos vizinhos com pesos fixos (como GCN), GAT *aprende* quanto cada vizinho deve contribuir para a atualização de um nó. Os coeficientes de atenção são calculados com base nas features dos nós envolvidos e normalizados por softmax.

**Fundamento matemático:**

**Etapa 1 — Score de atenção não-normalizado:**
$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^T \left[\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j\right]\right)$$

onde:
- $\mathbf{h}_i, \mathbf{h}_j \in \mathbb{R}^{F}$: features dos nós $i$ e $j$
- $\mathbf{W} \in \mathbb{R}^{F' \times F}$: matriz de transformação linear (aprendida)
- $\mathbf{a} \in \mathbb{R}^{2F'}$: vetor de atenção (aprendido)
- $\|$: concatenação; $\text{LeakyReLU}$ com slope negativo $\alpha = 0.2$

**Etapa 2 — Normalização (softmax sobre vizinhança):**
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

**Etapa 3 — Atualização do nó:**
$$\mathbf{h}'_i = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, \mathbf{W}\mathbf{h}_j\right)$$

**Etapa 4 — Multi-head, modo `concat=False` (média — Equação 6):**
$$\mathbf{h}'_i = \sigma\!\left(\frac{1}{K}\sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k\, \mathbf{W}^k\mathbf{h}_j\right)$$

onde $K$ é o número de cabeças. Com `concat=False`, a dimensão de saída permanece $F'$ (não $K \cdot F'$).

**Comparação com `concat=True` (Equação 5 — concatenação, para camadas intermediárias):**
$$\mathbf{h}'_i = \Big\|_{k=1}^{K} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k\, \mathbf{W}^k\mathbf{h}_j\right) \in \mathbb{R}^{K \cdot F'}$$

**Embasamento acadêmico:**

> 📖 **Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. (2018)** — "Graph Attention Networks"
> *6th International Conference on Learning Representations (ICLR 2018)*
> arXiv: `1710.10903` | OpenReview: `rJXMpikCZ`
> **Localização:** Seção 2.1 (Graph Attentional Layer), Equações 3–6
> **Relevância:** Paper original que define as equações acima implementadas por `GATConv` do PyG. As Equações 5 e 6 correspondem às opções `concat=True` e `concat=False` do script.
>
> ⚠️ **Nota crítica:** O paper GAT original avalia **exclusivamente node classification** (Cora, Citeseer, PubMed, PPI). A Seção 4 (Conclusões) cita "extending to graph classification" como trabalho futuro. Portanto, o uso de `global_mean_pool` + classificador linear para transformar o GAT em classificador de grafos é uma extensão empírica não validada no paper original — justificada pela prática estabelecida no benchmark UPFD (Dou et al., 2021).

**No código (`gat_model.py`):**
> Linhas 69–75: `conv1 = GATConv(768, 64, heads=4, concat=False, dropout=0.3)` — Equação 6.
> Linhas 78–84: `conv2 = GATConv(64, 64, heads=4, concat=False, dropout=0.3)` — Equação 6.
> Linhas 87–93: `conv3 = GATConv(64, 64, heads=1, concat=True)` — com 1 head, `concat=True` é equivalente a `concat=False`.
> Linhas 115–122: forward pass com ELU e dropout entre camadas.

---

### 2.2 Ativação ELU

**Descrição técnica:**
ELU (*Exponential Linear Unit*) é a ativação usada nas camadas intermediárias do GAT, conforme especificado no paper original. Diferentemente de ReLU, ELU produz valores negativos para entradas negativas, empurrando a média das ativações em direção a zero sem BatchNorm.

**Fundamento matemático:**

$$\text{ELU}(x) = \begin{cases} x & \text{se } x > 0 \\ \alpha(e^x - 1) & \text{se } x \leq 0 \end{cases}$$

onde $\alpha > 0$ (tipicamente $\alpha = 1.0$). O PyG/PyTorch usa $\alpha = 1.0$ por padrão.

Propriedades:
- **Gradiente não-nulo para $x < 0$**: atenua o problema do *dying ReLU*
- **Média de ativações ≈ 0**: acelera o aprendizado sem BatchNorm
- **Contínua e diferenciável em $x = 0$**: $\text{ELU}(0) = 0$, $\text{ELU}'(0^-) = \alpha = 1$

**Embasamento acadêmico:**

> 📖 **Clevert, D.-A.; Unterthiner, T.; Hochreiter, S. (2016)** — "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
> *4th International Conference on Learning Representations (ICLR 2016)*
> arXiv: `1511.07289`
> **Localização:** Seção 2 (Exponential Linear Units), Definição 1 (ELU)
> **Relevância:** Demonstra que ELUs levam a ativações com média mais próxima de zero e aceleraram convergência. Veličković et al. (2018) adotaram ELU explicitamente no GAT (Seção 3.3), tornando-a parte do design original da arquitetura.

**No código (`gat_model.py`):**
> Linhas 116, 121: `x = F.elu(x)` — entre conv1→conv2 e conv2→conv3.

---

### 2.3 Global Mean Pooling (Readout)

**Descrição técnica:**
`global_mean_pool` agrega as features de todos os nós de um grafo em um único vetor, calculando a média elementar:

$$\mathbf{h}_G = \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v \in \mathbb{R}^{d}$$

Esta operação produz um embedding de nível de grafo com a mesma dimensão das features nodais pós-convolucionais ($d = 64$), independentemente do número de nós.

**Comportamento em estrelas planas:**
Em grafos estrela onde todos os filhos têm features similares (mesmo embedding BERT, features posicionais variando em `pos`), o mean pool produz essencialmente a média do embedding da raiz com os embeddings dos filhos. Como todos compartilham o mesmo BERT de 768 dims, o mean pool é dominado pelo embedding textual compartilhado — a informação posicional tem peso $3/771 \approx 0.4\%$ no vetor final.

**Limitação teórica:**

> 📖 **Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. (2019)** — "How Powerful are Graph Neural Networks?"
> *7th International Conference on Learning Representations (ICLR 2019)*
> arXiv: `1810.00826`
> **Localização:** Seção 4.2 (Graph-level readout), Lema 5
> **Relevância:** Demonstra que `mean pooling` não é injetivo sobre multiconjuntos de nós — dois grafos com distribuições de features diferentes mas mesma média produzirão o mesmo embedding de grafo. O `sum pooling` (mais expressivo) é menos sensível ao número de nós. Para grafos estrela de fake news com nós homogêneos (mesmo BERT), essa limitação é relevante.

**No código (`gat_model.py`):**
> Linha 128: `h = global_mean_pool(x, batch)` — `batch` é o tensor de mapeamento nó→grafo injetado pelo `DataLoader` do PyG.

---

### 2.4 CrossEntropyLoss com Class Weights

**Descrição técnica:**
As variantes cético e ex-cético penalizam erros na classe "fake" com pesos `[1.0, 4.0]` e `[1.0, 6.0]` respectivamente. Com labels Bluesky heurísticos (onde a classe fake pode ser rara ou deficiente), esses pesos empurram o modelo a priorizar o recall da classe minoritária.

**Fundamento matemático:**

$$\mathcal{L}_{\text{CE}} = -\sum_{c \in \{0,1\}} w_c \cdot y_c \cdot \log\hat{p}_c$$

onde $w_c$ é o peso da classe $c$, $y_c \in \{0,1\}$ é o rótulo one-hot, e $\hat{p}_c$ é a probabilidade predita.

**Efeito nas variantes:**

| Variante | $w_0$ (real) | $w_1$ (fake) | Comportamento |
|----------|-------------|-------------|---------------|
| baseline / ctrl | 1.0 | 1.0 | Neutro — segue distribuição dos dados |
| cético | 1.0 | 4.0 | Erra fake custa 4× mais → maior recall para fake |
| ex-cético | 1.0 | 6.0 | Erra fake custa 6× mais → recall ainda maior, precision menor |

**Embasamento acadêmico:**

> 📖 **King, G.; Zeng, L. (2001)** — "Logistic Regression in Rare Events Data"
> *Political Analysis*, v. 9, n. 2, pp. 137–163. DOI: `10.1093/pan/9.2.137`
> **Localização:** Seção 3 (Correcting for Choice-Based Sampling)
> **Relevância:** Fundamenta o ajuste de pesos de classe como correção de desbalanceamento em classificação binária — princípio estatístico clássico que motiva as variantes cético e ex-cético.

> 📖 **Dou et al. (2021)**, UPFD, Tabela 3:
> **Relevância:** Benchmark que treina GNNs (incluindo GAT) em PolitiFact com dados equilibrados (~50% fake). O dataset Bluesky com labels heurísticos pode ter distribuição muito diferente, justificando a experimentação com class weights.

**No código (`03_treinar_gat.py`):**
> Linhas 51–56: dicionário `VARIANTES` com `peso_ce`.
> Linhas 151–152: `peso_tensor = torch.tensor(config["peso_ce"], ...)` → `criterion = CrossEntropyLoss(weight=peso_tensor)`.

---

### 2.5 Otimizador Adam

**Fundamento matemático:**

Adam mantém estimativas dos primeiro e segundo momentos dos gradientes:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(média móvel dos gradientes)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(variância móvel dos gradientes)}$$

Com correção de bias e atualização:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Embasamento acadêmico:**

> 📖 **Kingma, D. P.; Ba, J. (2015)** — "Adam: A Method for Stochastic Optimization"
> *3rd International Conference on Learning Representations (ICLR 2015)*
> arXiv: `1412.6980`
> **Localização:** Seção 2 (Algorithm 1), Seção 3 (Initialization Bias Correction)
> **Relevância:** Algoritmo de otimização usado diretamente no script. Parâmetros: `lr=0.005`, `weight_decay=5e-4` (L2 regularization equivalente a AdamW-lite).

**No código:**
> Linha 153: `optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)`.

---

### 2.6 ReduceLROnPlateau + Early Stopping

**Descrição técnica:**
O scheduler reduz o LR por fator 0.5 se o F1 de validação não melhora em 5 épocas consecutivas (`mode="max", patience=5`). O early stopping para o treinamento após 7 épocas sem melhoria (`PATIENCE=7`). A hierarquia é: redução de LR primeiro (épocas 5+), depois parada (épocas 7+).

**No código:**
> Linhas 156–158: `ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)`.
> Linha 169: `scheduler.step(val_f1)` — modo max (monitora F1, não loss).
> Linhas 183–185: early stopping após `PATIENCE=7` épocas sem melhora.
> Linha 174: `best_state = {k: v.clone() ...}` — clona pesos para evitar referências mutáveis.

---

### 2.7 F1 Macro como Critério de Seleção

**Fundamento matemático:**

$$F1_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

onde $P_c$ e $R_c$ são precisão e recall da classe $c$, calculados por classe e depois mediados. Para $C=2$: média simples do F1 das classes fake e real.

**Vantagem sobre accuracy para datasets desbalanceados:** Se 94% dos exemplos são "real", um classificador que prediz sempre "real" tem accuracy 94% mas F1 macro ≈ 0.47.

**No código:**
> Linha 128 (`avaliar()`): `f1_score(y_true, y_pred, average="macro", zero_division=0)`.
> Linha 220 (`avaliar_teste()`): `f1_score(y_true, y_pred, zero_division=0)` — sem `average=` → default `"binary"`. **Inconsistência documentada na Seção 7.1**.

---

## 3. Pipeline de Treinamento

### 3.1 DataLoader com Mini-batching PyG

O `DataLoader` do PyG (`torch_geometric.loader.DataLoader`) cria mini-batches de grafos concatenando-os em um único grafo desconexo, com um tensor `batch` indicando a qual grafo original cada nó pertence. Isso habilita o `global_mean_pool` a agregar nós por grafo.

**No código:**
> Linhas 83–86: `DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)`.

---

### 3.2 Forward Pass Completo

```
Entrada: x [N_total, 768], edge_index [2, E_total], batch [N_total]
  ↓
GATConv(768→64, heads=4, concat=False)  → ELU → Dropout(0.3)  → [N, 64]
  ↓
GATConv(64→64,  heads=4, concat=False)  → ELU → Dropout(0.3)  → [N, 64]
  ↓
GATConv(64→64,  heads=1, concat=True)                          → [N, 64]
  ↓
global_mean_pool(x, batch)                                     → [B, 64]
  ↓
Dropout(0.5)
  ↓
Linear(64→2)                                                   → [B, 2] (logits)
```

onde $N_{\text{total}} = \sum_{g \in \text{batch}} |V_g|$ e $B$ é o tamanho do batch em grafos.

---

## 4. Análise Empírica: Posicionamento na Questão Central

### 4.1 Pontos Fortes desta Abordagem

**Mecanismo de atenção sobre vizinhos:** Em grafos de propagação com topologia variada, o GAT pode aprender a dar mais peso a certas arestas. Para grafos estrela, isso se traduz em pesar diferentemente cada filho (repost) em relação à raiz — potencialmente capturando padrões de "quem repostou" representados pelas features posicionais.

**Ablation por class weights:** As 4 variantes permitem medir o impacto do desbalanceamento nos resultados, contribuindo para a validade interna dos experimentos.

**Interface padronizada:** `out, h = model(x, edge_index, batch)` — o retorno do embedding `h` permite uso posterior para análise (e.g., t-SNE, transferência de aprendizado).

### 4.2 Limitações Identificadas

**L1 — GAT foi concebido para node classification, não graph classification:**
O paper original (Veličković et al., 2018, Seção 4) não avalia em graph classification. A extensão via `global_mean_pool` é empírica e não tem garantias teóricas equivalentes às do GIN para graph classification (Xu et al., 2019).

**L2 — `global_mean_pool` com nós homogêneos:**
Em grafos estrela onde todos os nós compartilham o mesmo BERT de 768 dims (apenas 3 dims posicionais diferem), o mean pool produz essencialmente o embedding textual médio. A atenção do GAT opera sobre os 771 dims totais, mas a contribuição posicional é de $3/771 \approx 0.4\%$ — o mecanismo de atenção está funcionando quase exclusivamente sobre o espaço BERT.

**L3 — Caminhos de dados hardcoded incompatíveis com script 02:**
Ver Seção 7.1, E2.

**L4 — `num_node_features=768` hardcoded incompatível com variantes do script 02:**
Ver Seção 7.1, E1.

### 4.3 Comparação com Estado da Arte

| Modelo | Acc | F1 | Dataset | Fonte |
|--------|-----|-----|---------|-------|
| GAT + BERT (UPFD) | ~84% | ~84% | PolitiFact | Dou et al. (2021) |
| GAT baseline (este script) | — | — | Bluesky (heurístico) | — |
| GAT cético [1,4] | — | — | Bluesky | — |
| GAT ex-cético [1,6] | — | — | Bluesky | — |

> **Nota:** Resultados deste script não são comparáveis com UPFD devido a diferenças de dataset, topologia de grafo, e qualidade dos labels.

### 4.4 Resposta Parcial à Questão do TCC

O GAT com 4 cabeças de atenção é a arquitetura GNN mais expressiva implementada neste pipeline. Se os resultados forem fracos mesmo com class weights calibrados, isso indica que a limitação não é de arquitetura (GAT vs. GCN) mas dos dados: grafos estrela com labels ruidosos não fornecem sinal suficiente para que o mecanismo de atenção seja útil. Isso apoia diretamente a hipótese de que NLP puro (script 10) é mais adequado para este problema.

---

## 5. Análise de Código

### 5.1 Erros Identificados

```python
# ❌ Linha 148 — num_node_features=768 hardcoded
model = GATClassifier(num_node_features=768, num_classes=2).to(device)
# Script 02 pode gerar grafos com 3 dims (notext) ou 771 dims (full).
# Esse valor fixo causará RuntimeError no primeiro forward pass se o
# grafo tiver dimensão diferente.

# ✅ Correção: inferir da dimensão real dos dados
dim_entrada = train_loader.dataset[0].x.shape[1]
model = GATClassifier(num_node_features=dim_entrada, num_classes=2).to(device)
```

```python
# ❌ Linhas 63–67 — caminhos hardcoded não correspondem à saída do script 02
DATA_DIR / "grafos_bluesky_train.pt"
# Script 02 salva em: data/bluesky_<suffix>/bluesky_train.pt
# Esses arquivos nunca existem no caminho esperado por este script.

# ✅ Correção: adicionar argumento --data-suffix
parser.add_argument("--data-suffix", type=str, default=None)
# e construir o caminho dinamicamente:
subdir = f"bluesky_{args.data_suffix}" if args.data_suffix else "bluesky"
arquivos = {"train": DATA_DIR / subdir / "bluesky_train.pt", ...}
```

```python
# ❌ Linha 220 — avaliar_teste() usa F1 binary, avaliar() usa F1 macro
"f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
# Sem average= explícito → default "binary" (F1 da classe positiva)
# Inconsistência: o modelo é selecionado por macro-F1 mas reportado por binary-F1.

# ✅ Correção: usar average="macro" consistentemente
"f1_macro": round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
"f1_binary": round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4),
```

```python
# ❌ Linhas 51–56 — variantes "baseline" e "ctrl" são idênticas em hiperparâmetros
"baseline":   {"peso_ce": [1.0, 1.0], "arquivo": "pesos_gat.pth"},
"ctrl":       {"peso_ce": [1.0, 1.0], "arquivo": "pesos_gat_bs_ctrl.pth"},
# Mesmos pesos CE, mesmo código, mesmos dados → resultados diferem apenas
# por estocasticidade (shuffle sem seed). Sem propósito experimental claro.

# ✅ Sugestão: diferenciar ctrl com weight_decay ou lr distintos,
# ou documentar explicitamente que o objetivo é medir variância entre runs.
```

```python
# ❌ DataLoader sem generator com seed fixo (linha 83)
DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
# A ordem dos batches muda entre execuções → resultados não reprodutíveis.

# ✅ Correção:
g = torch.Generator()
g.manual_seed(RANDOM_SEED)
DataLoader(train_data, batch_size=64, shuffle=True,
           generator=g, num_workers=0)
```

### 5.2 Ineficiências

**I1 — `avaliar()` percorre o loader duas vezes por época:**
`avaliar(model, val_loader)` é chamada dentro do loop de treinamento, depois `scheduler.step(val_f1)`. O loader percorre o conjunto de validação completo a cada época. Para ~200 grafos de val, o overhead é insignificante.

**I2 — `best_state` clona todos os tensores:**
`{k: v.clone() for k, v in model.state_dict().items()}` clona ~200k parâmetros a cada época de melhoria. Para o tamanho do modelo (~200k params × 4 bytes ≈ 800 KB), o custo é negligenciável.

### 5.3 Boas Práticas Observadas

**B1 — `v.clone()` no best_state:** Essencial — sem `.clone()`, `best_state` apontaria para os tensores do modelo que continuariam mudando no treinamento.

**B2 — `model.eval()` / `model.train()` corretamente alternados:** Garante que dropout está ativo no treino e inativo na avaliação.

**B3 — `@torch.no_grad()`** em `avaliar()` e `avaliar_teste()`: Evita acúmulo de gradientes na memória durante inferência.

**B4 — `count_parameters()` em `gat_model.py`:** Facilita diagnóstico do tamanho do modelo.

**B5 — `torch.manual_seed(seed)` no `__init__` do modelo:** Garante que os pesos iniciais do modelo sejam reprodutíveis entre instâncias com o mesmo seed — útil para comparar as variantes que devem diferir apenas nos pesos de CE.

**B6 — Early stopping com restauração dos melhores pesos:** Protege contra overfitting ao restaurar o modelo do ponto de menor perda de generalização.

---

## 6. Referências Bibliográficas

1. VELIČKOVIĆ, P.; CUCURULL, G.; CASANOVA, A.; ROMERO, A.; LIÒ, P.; BENGIO, Y. **Graph Attention Networks**. In: *6th International Conference on Learning Representations (ICLR 2018)*. Disponível em: https://arxiv.org/abs/1710.10903

2. CLEVERT, D.-A.; UNTERTHINER, T.; HOCHREITER, S. **Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)**. In: *4th International Conference on Learning Representations (ICLR 2016)*. Disponível em: https://arxiv.org/abs/1511.07289

3. KINGMA, D. P.; BA, J. **Adam: A Method for Stochastic Optimization**. In: *3rd International Conference on Learning Representations (ICLR 2015)*. Disponível em: https://arxiv.org/abs/1412.6980

4. XU, K.; HU, W.; LESKOVEC, J.; JEGELKA, S. **How Powerful are Graph Neural Networks?** In: *7th International Conference on Learning Representations (ICLR 2019)*. Disponível em: https://arxiv.org/abs/1810.00826

5. FEY, M.; LENSSEN, J. E. **Fast Graph Representation Learning with PyTorch Geometric**. *ICLR 2019 Workshop*. Disponível em: https://arxiv.org/abs/1903.02428

6. DOU, Y. et al. **User Preference-aware Fake News Detection (UPFD)**. *SIGIR'21*, 2021. DOI: `10.1145/3404835.3462990`. Disponível em: https://arxiv.org/abs/2104.12259

7. KING, G.; ZENG, L. **Logistic Regression in Rare Events Data**. *Political Analysis*, v. 9, n. 2, pp. 137–163, 2001. DOI: `10.1093/pan/9.2.137`

---

## 7. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| GATConv | Camada de convolução com atenção do PyG. Implementa as Equações 3–6 de Veličković et al. (2018). | Veličković et al. (2018) |
| `concat=False` | Modo de agregação multi-head onde as heads são **mediadas** em vez de concatenadas. Dimensão de saída = `out_channels`. | Veličković et al. (2018), Eq. 6 |
| `concat=True` | Modo padrão: heads são concatenadas. Dimensão de saída = `heads × out_channels`. | Veličković et al. (2018), Eq. 5 |
| ELU | *Exponential Linear Unit* — ativação com valores negativos que empurram a média das ativações a zero. | Clevert et al. (2016) |
| `global_mean_pool` | Operação de readout: média das features de todos os nós de um grafo → vetor de grafo único. | Xu et al. (2019), Seção 4.2 |
| Class weights | Pesos aplicados à CrossEntropyLoss por classe, compensando desbalanceamento. `weight=[1,4]` faz errar classe 1 custar 4× mais. | King & Zeng (2001) |
| Early stopping | Parada antecipada do treinamento quando a métrica de validação para de melhorar por `PATIENCE` épocas. | Prática estabelecida |
| ReduceLROnPlateau | Scheduler que reduz o LR quando uma métrica monitorada para de melhorar. `mode="max"` monitora F1. | PyTorch docs |
| F1 Macro | Média simples do F1 por classe. Não é ponderado pelo suporte de cada classe — trata classes desbalanceadas igualmente. | — |
| `data.batch` | Tensor `[N]` do PyG mapeando cada nó ao índice do grafo no mini-batch. Injetado automaticamente pelo `DataLoader`. | Fey & Lenssen (2019) |
