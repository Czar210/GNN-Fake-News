# Documentação Técnica: 04_comparar_gcn_gat.py + gcn_model.py

## Metadados

- **Arquivos analisados:** `04_comparar_gcn_gat.py`, `gcn_model.py` (com referência cruzada a `gat_model.py`)
- **Caminho:** `03_Mega_Research/04_comparar_gcn_gat.py`, `03_Mega_Research/gcn_model.py`
- **Data de análise:** 2026-04-24
- **Tipo de abordagem:** GNN — benchmark comparativo GCN vs GAT para classificação de grafos
- **Modelos principais:** GCNClassifier (3 × GCNConv) vs GATClassifier (3 × GATConv, já documentado)
- **Datasets utilizados:** Bluesky (grafos `.pt` do script 02)
- **Contribuição para a questão central:** Este é o primeiro benchmark controlado do TCC: mede se o mecanismo de atenção do GAT oferece vantagem sobre a convolução espectral do GCN em grafos de propagação de fake news. O resultado informa se a complexidade adicional (3.8× mais parâmetros, dropout entre camadas) do GAT é justificada neste domínio — evidência concreta para a discussão de custo-benefício de arquiteturas GNN.

---

## 1. Visão Geral dos Scripts

`04_comparar_gcn_gat.py` implementa um benchmark controlado: treina GCN e GAT com os **mesmos** dados, LR, número de épocas e função de perda (sem class weights), garantindo que as diferenças observadas sejam atribuíveis à arquitetura. Produz três artefatos: matrizes de confusão lado a lado, gráfico de barras das métricas, e relatório de texto com interpretação automática.

`gcn_model.py` implementa o `GCNClassifier`: três camadas `GCNConv` com ReLU (a terceira sem ativação), seguidas de `global_mean_pool`, Dropout(0.5) e um classificador linear. A interface é idêntica ao `GATClassifier` — `out, h = model(x, edge_index, batch)` — permitindo trocar os modelos no mesmo loop de treinamento.

**Comparação arquitetural direta:**

| Aspecto | GCNClassifier | GATClassifier |
|---------|--------------|--------------|
| Mecanismo de agregação | Normalização simétrica por grau (fixa) | Atenção aprendida por pares de nós |
| Multi-head | Não | 4 heads (média, camadas 1–2) |
| Ativação | ReLU | ELU |
| Dropout entre camadas | Não | Sim (0.3 entre cada GATConv) |
| Dropout pré-classificador | 0.5 | 0.5 |
| Parâmetros (approx.) | ~58k | ~218k (~3.8×) |

---

## 2. Arquitetura e Componentes Principais

### 2.1 Graph Convolutional Network (GCN)

**Descrição técnica:**
GCN aplica uma convolução espectral aproximada sobre grafos. Cada camada propaga features dos nós através da estrutura do grafo usando uma normalização simétrica do grau, adicionando auto-loops para que cada nó se inclua na sua própria agregação.

**Fundamento matemático:**

**Regra de propagação (Equação 2, Kipf & Welling 2017):**

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2}\, H^{(l)}\, W^{(l)}\right)$$

onde:
- $\tilde{A} = A + I_N$ — matriz de adjacência com auto-loops adicionados
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ — grau de cada nó em $\tilde{A}$
- $H^{(l)} \in \mathbb{R}^{N \times d_l}$ — features dos nós na camada $l$; $H^{(0)} = X$
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ — pesos treináveis da camada $l$
- $\sigma(\cdot)$ — função de ativação (ReLU no script)

**Interpretação como agregação normalizada:**

Para um nó $i$, a mensagem recebida de um vizinho $j$ é escalada por $\frac{1}{\sqrt{\tilde{d}_i \cdot \tilde{d}_j}}$:

$$h_i^{(l+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}}\, W^{(l)}\, h_j^{(l)}\right)$$

Nós de alto grau recebem mensagens com pesos menores — a normalização previne instabilidade quando alguns nós têm muitas conexões. Nós de grau 1 (raiz de uma estrela com 1 filho) recebem peso $1/\sqrt{2 \cdot 2} = 0.5$.

**Motivação espectral:**
A regra deriva de uma aproximação de primeira ordem de convolução espectral usando polinômios de Chebyshev (Seções 2.1–2.2 de Kipf & Welling). A renormalização com $\tilde{A}$ (Equação 8 do paper) garante que os autovalores de $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ estejam em $[0, 2]$, estabilizando o treinamento com múltiplas camadas.

**Embasamento acadêmico:**

> 📖 **Kipf, T. N.; Welling, M. (2017)** — "Semi-Supervised Classification with Graph Convolutional Networks"
> *5th International Conference on Learning Representations (ICLR 2017)*
> arXiv: `1609.02907`
> **Localização:** Seção 2 (Fast Approximate Convolutions on Graphs), Equação 2 (regra de propagação), Equação 8 (normalização renormalizada), Seção 2.1 (motivação espectral)
> **Relevância:** Paper original que define a operação `GCNConv` usada em `gcn_model.py`. Todos os três `GCNConv` do script implementam diretamente a Equação 2.

**No código (`gcn_model.py`):**
> Linhas 56–58: `self.conv1/2/3 = GCNConv(...)` — três camadas GCN.
> Linhas 62–64: `x = self.conv1(x, edge_index).relu()` etc. — forward pass.
> Linha 65: `h = global_mean_pool(x, batch)` — readout idêntico ao GAT.

**Contagem exata de parâmetros (GCN):**

| Camada | Operação | Parâmetros |
|--------|----------|-----------|
| conv1 | $W_1 \in \mathbb{R}^{768 \times 64}$, bias $\in \mathbb{R}^{64}$ | 49,216 |
| conv2 | $W_2 \in \mathbb{R}^{64 \times 64}$, bias $\in \mathbb{R}^{64}$ | 4,160 |
| conv3 | $W_3 \in \mathbb{R}^{64 \times 64}$, bias $\in \mathbb{R}^{64}$ | 4,160 |
| lin | $W \in \mathbb{R}^{64 \times 2}$, bias $\in \mathbb{R}^2$ | 130 |
| **Total** | | **~57,666** |

---

### 2.2 GCN vs GAT: Diferença Fundamental de Agregação

**GCN — pesos fixos pelo grau:**
$$\alpha_{ij}^{\text{GCN}} = \frac{1}{\sqrt{\tilde{d}_i \cdot \tilde{d}_j}}$$

Os pesos de agregação dependem apenas da estrutura topológica (grau dos nós) e são calculados uma vez, antes do treinamento. Em grafos estrela onde a raiz tem grau $N$ e os filhos têm grau 1, a raiz recebe mensagem de cada filho ponderada por $1/\sqrt{(N+1) \cdot 2}$ — valor que diminui conforme o grafo cresce.

**GAT — pesos aprendidos pelas features:**
$$\alpha_{ij}^{\text{GAT}} = \text{softmax}_j\!\left(\text{LeakyReLU}\!\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)\right)$$

O modelo *aprende* quanto cada vizinho contribui com base nas features dos nós. Em teoria, o GAT pode aprender a ignorar filhos com features "não informativas" — mas em grafos estrela onde todos os filhos têm o mesmo BERT de 768 dims, essa capacidade de discriminação é limitada às features posicionais (3 dims).

**Implicação para grafos estrela de fake news:**
Quando todos os nós filhos compartilham o mesmo embedding textual (título do artigo), os coeficientes de atenção $\alpha_{ij}^{\text{GAT}}$ serão muito similares entre si — a atenção não consegue diferenciar filhos com base em texto porque não há texto diferente. A única diferença provém das features posicionais (`is_root=0`, `grau_norm=0`, `pos=i/N`), que têm peso $3/771 \approx 0.4\%$ no vetor de entrada.

**Embasamento acadêmico:**

> 📖 **Veličković et al. (2018)** — "Graph Attention Networks", arXiv:`1710.10903`, ICLR 2018
> **Localização:** Seção 2.1 (Equações 1–4)

> 📖 **Karn, I.; Jensen, D. (2025)** — "The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News"
> arXiv: `2512.06638`
> **Localização:** Seção de resultados empíricos
> **Relevância:** Demonstra que para grafos de fake news rasos (>75% dos nós a 1 hop da raiz), a estrutura do grafo contribui marginalmente — tanto GCN quanto GAT tendem a convergir para desempenho similar ao MLP com as mesmas features.

---

### 2.3 Benchmark Controlado: Design Experimental

**Descrição técnica:**
As condições de treinamento são intencionalmente igualadas para que diferenças nos resultados sejam atribuíveis apenas à arquitetura:
- Mesmo LR (`0.005`), weight decay (`5e-4`), batch size (`64`)
- Mesma função de perda (`CrossEntropyLoss` sem pesos)
- Mesmo scheduler (`ReduceLROnPlateau`, mode="max", patience=5)
- Mesmos dados (train/val/test)

**Condição não controlada — confound presente:**
O GAT tem dropout entre camadas (`dropout_attn=0.3`) enquanto o GCN não tem — só o dropout final de 0.5 é compartilhado. Isso significa que os dois modelos diferem em:
1. Mecanismo de agregação (fixo vs. aprendido)
2. Ativação (ReLU vs. ELU)
3. Regularização durante propagação (sem dropout vs. dropout 0.3)

Portanto, qualquer diferença de desempenho **não pode ser atribuída exclusivamente** ao mecanismo de atenção — pode ser efeito do dropout ou da ativação ELU.

**Embasamento acadêmico:**

> 📖 **Shchur, O.; Mumme, M.; Bojchevski, A.; Günnemann, S. (2018)** — "Pitfalls of Graph Neural Network Evaluation"
> *Workshop on Relational Representation Learning, NeurIPS 2018*
> arXiv: `1811.05868`
> **Localização:** Seção 3 (Experimental Design Issues), Seção 3.1 (Training Procedure)
> **Relevância:** Demonstra que diferenças no procedimento de treinamento (dropout, early stopping, hyperparameters) podem causar rankings de modelos "dramaticamente diferentes". Qualquer benchmark que não equalize **todos** os hiperparâmetros corre o risco de atribuir diferenças de desempenho à arquitetura quando a causa é a regularização.

---

### 2.4 Matriz de Confusão

**Descrição técnica:**
A matriz de confusão é uma tabela $2 \times 2$ que decompõe os erros e acertos de um classificador binário:

$$\text{CM} = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

onde:
- $TP$ — verdadeiros positivos (fake predito como fake)
- $TN$ — verdadeiros negativos (real predito como real)
- $FP$ — falsos positivos (real predito como fake)
- $FN$ — falsos negativos (fake predito como real)

As métricas derivam diretamente:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$
$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Embasamento acadêmico:**

> 📖 **Sokolova, M.; Lapalme, G. (2009)** — "A systematic analysis of performance measures for classification tasks"
> *Information Processing & Management*, v. 45, n. 4, pp. 427–437
> DOI: `10.1016/j.ipm.2009.03.002`
> **Localização:** Seção 3 (Performance Measures for Binary Classification), Tabela 2
> **Relevância:** Referência seminal para uso de Precision, Recall, F-measure em classificação binária. A Tabela 2 apresenta as definições exatas em termos de TP/TN/FP/FN usadas nas linhas 152–156 do script.

**No código:**
> Linhas 152–157: cálculo de accuracy, precision, recall, f1, confusion_matrix.
> Linhas 162–188: `plot_matrizes()` — visualização com anotação numérica em cada célula.

---

## 3. Pipeline de Avaliação

### 3.1 Fluxo Completo

```
1. Treinar GCN (30 épocas, Adam, CrossEntropy, ReduceLROnPlateau)
   ↓ melhor modelo por val F1-macro
2. Treinar GAT (mesmas condições)
   ↓ melhor modelo por val F1-macro
3. Avaliar ambos no conjunto de teste
   ↓ accuracy, precision, recall, f1-binary
4. Gerar matrizes_confusao.png
5. Gerar metricas_barras.png
6. Gerar relatorio.txt com conclusão automática
```

### 3.2 Interpretação Automática (`salvar_relatorio`)

O script gera uma conclusão textual baseada no F1 do teste:

```python
if res_gat["f1"] >= res_gcn["f1"]:
    "O mecanismo de atenção diferenciou efetivamente nós influentes na propagação."
else:
    "GCN superou o GAT. Possível causa: grafos pequenos/rasos não se beneficiam..."
```

Esta interpretação automática contém um problema de validade interna: ela infere causalidade mecanística ("a atenção diferenciou nós influentes") de uma comparação de F1 sem replicações. O texto da segunda branch (GCN > GAT) é mais conservador e metodologicamente correto ao mencionar "possível causa" em vez de afirmar causalidade.

---

## 4. Análise Empírica: Posicionamento na Questão Central

### 4.1 Pontos Fortes desta Abordagem

**Benchmark nas mesmas condições:** LR, epochs, loss e dados são idênticos — qualquer diferença é arquitetural (embora com o confound do dropout documentado).

**Relatório automático de texto:** Facilita diretamente a redação do TCC ao gerar frases de interpretação já formatadas.

**Métricas e tempo combinados:** O gráfico de barras inclui tempo por época — permite avaliar o tradeoff desempenho/custo computacional do GAT (3.8× mais parâmetros).

### 4.2 Limitações Identificadas

**L1 — Confound dropout/ativação:**
O GAT tem regularização extra (dropout 0.3 entre camadas) e ativação diferente (ELU vs. ReLU). Uma comparação verdadeiramente controlada precisaria ou de um GAT com as mesmas ativações e regularização do GCN, ou de um GCN com o mesmo dropout entre camadas.

**L2 — Uma única run por modelo:**
Com SGD estocástico e DataLoader sem seed, duas runs do mesmo modelo podem diferir ±1–3% em F1. Shchur et al. (2018) demonstram que rankings de modelos com diferenças pequenas são instáveis. O script 09 (`teste_significancia`) provavelmente aborda isso, mas aqui não há múltiplas runs.

**L3 — F1 binary na avaliação final vs F1 macro no critério de seleção:**
O melhor modelo é selecionado por F1 macro (validação), mas reportado por F1 binary (teste). Para datasets desbalanceados, esses valores podem divergir significativamente.

**L4 — Conclusão automática infere causalidade:**
"O mecanismo de atenção diferenciou efetivamente nós influentes" é uma afirmação causal não suportada por uma única comparação sem replicação.

### 4.3 Comparação com Estado da Arte

| Modelo | Acc | F1 | Dataset | Notas |
|--------|-----|-----|---------|-------|
| GCN (UPFD, bert) | 83.26% | 83.14% | PolitiFact | Dou et al. (2021) |
| GAT (UPFD, bert) | — | — | PolitiFact | Não reportado no UPFD |
| GCNClassifier (este script) | — | — | Bluesky | — |
| GATClassifier (este script) | — | — | Bluesky | — |

> 📖 **Fonte UPFD:** Dou et al. (2021), DOI: `10.1145/3404835.3462990`, Tabela 3.

### 4.4 Resposta Parcial à Questão do TCC

O resultado deste benchmark fornece a primeira evidência direta de dois ângulos:
1. **Se GAT ≈ GCN:** o mecanismo de atenção não agrega valor em grafos estrela com features homogêneas — custo computacional extra não se justifica.
2. **Se GAT > GCN:** a atenção captura algo além do grau nos grafos Bluesky — mas o confound com dropout/ELU impede conclusão definitiva sem o ablation controlado.

Em ambos os casos, a comparação com o baseline NLP (script 10) é necessária para responder "GNNs são viáveis?" — este script apenas posiciona GCN e GAT entre si dentro da família GNN.

---

## 5. Análise de Código

### 5.1 Erros Identificados

```python
# ❌ Linhas 50–58 — caminhos hardcoded não correspondem à saída do script 02
DATA_DIR / "grafos_bluesky_train.pt"
# Script 02 salva em data/bluesky_<suffix>/bluesky_train.pt
# (mesmo problema dos scripts 03 e 04)

# ✅ Correção: argumento --data-suffix (idêntico ao sugerido no script 03)
```

```python
# ❌ Linha 155 — F1 binary na avaliação final, inconsistente com F1 macro no treino
"f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
# default average="binary" → mede apenas a classe positiva (fake)

# ✅ Correção: reportar ambos explicitamente
"f1_macro":  round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
"f1_binary": round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4),
```

```python
# ❌ Linha 113 — recorre train_loader para calcular n_total_grafos
avg_loss = total_loss / sum(d.num_graphs for d in train_loader)
# Isso inicia um novo loop sobre o dataset de treino só para somar num_graphs
# — ineficiente (O(N) desnecessário) e potencialmente com shuffle diferente.

# ✅ Correção: acumular n_grafos no próprio loop de treino
total_grafos = 0
for data in train_loader:
    ...
    total_grafos += data.num_graphs
avg_loss = total_loss / total_grafos
```

```python
# ❌ Linhas 275–286 — conclusão automática infere causalidade de resultado único
if res_gat["f1"] >= res_gcn["f1"]:
    linhas.append("  O mecanismo de atenção diferenciou efetivamente nós influentes...")
# Afirmação causal sem suporte estatístico.

# ✅ Sugestão: usar linguagem hedgeada
"  GAT superou GCN em F1 (+{delta:.4f}). Hipótese: mecanismo de atenção "
"  pode ter discriminado padrões de propagação — verificar com teste t (script 09)."
```

```python
# ❌ DataLoader sem seed no shuffle (mesma limitação do script 03)
DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ✅ Correção:
g = torch.Generator().manual_seed(42)
DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)
```

### 5.2 Boas Práticas Observadas

**B1 — Interface drop-in `out, h = model(x, edge_index, batch)`:** GCN e GAT partilham a mesma interface, permitindo trocar os modelos no mesmo loop `treinar()` sem nenhuma alteração — boa prática de polimorfismo.

**B2 — `--usar-pesos-salvos`:** Permite reutilizar modelos já treinados pelo script 03, evitando re-treinamento desnecessário — economiza ~10 minutos de GPU para comparação rápida.

**B3 — Artefatos de saída em diretório dedicado (`Execution/results/comparativo_gcn_gat/`):** Separação limpa dos resultados facilita a coleta para o TCC.

**B4 — `plot_matrizes()` com anotação numérica e contraste automático:** `color="white" if cm[i,j] > cm.max()/2 else "black"` garante legibilidade em células claras e escuras.

**B5 — Relatório `Δ` por métrica:** A coluna de diferença (`Δ = GAT - GCN`) no relatório quantifica a margem de forma direta.

**B6 — `torch.manual_seed(seed)` nos modelos:** Garante pesos iniciais idênticos entre instâncias com o mesmo seed — condição importante para reprodutibilidade da comparação.

---

## 6. Referências Bibliográficas

1. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. In: *5th International Conference on Learning Representations (ICLR 2017)*. Disponível em: https://arxiv.org/abs/1609.02907

2. VELIČKOVIĆ, P. et al. **Graph Attention Networks**. In: *ICLR 2018*. Disponível em: https://arxiv.org/abs/1710.10903

3. SHCHUR, O.; MUMME, M.; BOJCHEVSKI, A.; GÜNNEMANN, S. **Pitfalls of Graph Neural Network Evaluation**. *Workshop on Relational Representation Learning, NeurIPS 2018*. Disponível em: https://arxiv.org/abs/1811.05868

4. SOKOLOVA, M.; LAPALME, G. **A systematic analysis of performance measures for classification tasks**. *Information Processing & Management*, v. 45, n. 4, pp. 427–437, 2009. DOI: `10.1016/j.ipm.2009.03.002`

5. KARN, I.; JENSEN, D. **The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News**. arXiv, 2025. Disponível em: https://arxiv.org/abs/2512.06638

6. DOU, Y. et al. **User Preference-aware Fake News Detection (UPFD)**. *SIGIR'21*, 2021. DOI: `10.1145/3404835.3462990`

7. XU, K. et al. **How Powerful are Graph Neural Networks?** *ICLR 2019*. Disponível em: https://arxiv.org/abs/1810.00826

---

## 7. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| $\tilde{A} = A + I_N$ | Matriz de adjacência com auto-loops — cada nó se inclui na sua própria agregação. | Kipf & Welling (2017), Seção 2 |
| Normalização simétrica | $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ — pondera cada mensagem pelo inverso da raiz quadrada do produto dos graus. Estabiliza autovalores em $[0,2]$. | Kipf & Welling (2017), Eq. 8 |
| Confound | Variável que difere entre os grupos comparados além da variável de interesse, impedindo inferência causal. Ex: GAT tem dropout entre camadas e GCN não. | Shchur et al. (2018) |
| F1 binary | F1-Score calculado apenas para a classe positiva (label 1 = fake). Default do `sklearn.f1_score` sem `average=`. | Sokolova & Lapalme (2009) |
| F1 macro | Média simples do F1 de cada classe. Trata classes desbalanceadas igualmente. | Sokolova & Lapalme (2009) |
| Benchmark controlado | Experimento onde todas as variáveis exceto a de interesse são equalizadas entre os grupos comparados. | Shchur et al. (2018) |
| Drop-in replacement | Componente que pode substituir outro sem modificar o código circundante — GCN e GAT compartilham a mesma interface `model(x, edge_index, batch)`. | — |
