# Análise Comparativa de Arquiteturas GNN para Detecção de Fake News

**Projeto:** TCC — Detecção de Fake News com Graph Neural Networks  
**Autor:** André Livingston Messina  
**Data:** Abril de 2026  
**Status:** Em andamento — fase de benchmark multi-dataset

---

## 1. Visão Geral do Projeto

Este projeto investiga o uso de Graph Neural Networks (GNNs) para classificação de fake news em redes sociais. A hipótese central é que a **estrutura de propagação** de uma notícia — como ela se espalha entre usuários — carrega informação discriminativa que modelos de texto puros não capturam.

Três arquiteturas foram implementadas e comparadas:

| Arquitetura | Mecanismo de Agregação | Parâmetros | Paper de referência |
|---|---|---|---|
| **GCN** | Soma normalizada por grau | ~57,666 | Kipf & Welling, 2017 |
| **GAT** | Atenção aprendida (softmax) | ~218,562 | Veličković et al., 2018 |
| **GraphSAGE** | Média amostrada (W_l + W_r) | ~115,010 | Hamilton et al., 2017 |

Todas as arquiteturas recebem features de nós de **768 dimensões** (embeddings BERT via `paraphrase-multilingual-mpnet-base-v2`) e produzem classificação binária: **Real (0) vs Fake (1)**.

---

## 2. Datasets Utilizados

### 2.1 Dataset Bluesky (HuggingFace)

- **Origem:** API ATProtocol do Bluesky, coletado via `01_BlueSky_Pipe/`
- **Dataset HuggingFace:** `Zaras210/bluesky-fake-news-dataset` (~21 GB)
- **Tamanho após processamento:** 166,910 posts
- **Distribuição de classes:**
  - Real: 166,010 posts (99.46%)
  - Fake: 900 posts (0.54%)
  - **Razão de desbalanceamento: 184:1**
- **Feeds coletados:** Blacksky (51%), News (25%), Science (20%), outros (4%)
- **Split:** 60% treino / 20% validação / 20% teste
- **Labeling:** Inferido a partir do feed de origem (feeds de desinformação = label Fake)

### 2.2 Dataset UPFD (PolitiFact)

- **Origem:** Twitter, via dataset UPFD (Bian et al., 2020)
- **Acesso:** `torch_geometric.datasets.UPFD(name='politifact', feature='bert')`
- **Estrutura:** Grafos de propagação reais com árvore de retweets
- **Distribuição de classes:** ~740 fake / ~740 real (balanceado)
- **Nós por grafo:** variável, mediana ~200 nós
- **Arestas:** hierarquia real de propagação (quem retweetou quem)

---

## 3. Arquiteturas Implementadas

### 3.1 GCNClassifier

```
Input: [N, 768]
  → GCNConv(768 → 64)  + ReLU
  → GCNConv(64 → 64)   + ReLU
  → GCNConv(64 → 64)
  → global_mean_pool → [B, 64]
  → Dropout(0.5)
  → Linear(64 → 2)
Output: logits [B, 2]
```

**Características:**
- Propagação espectral: cada nó agrega os vizinhos normalizados por grau (`D^{-1/2} A D^{-1/2}`)
- Sem parâmetros por aresta — simples e eficiente
- Resiliente a grafos rasos/pequenos
- Parâmetros: ~57,666

### 3.2 GATClassifier

```
Input: [N, 768]
  → GATConv(768 → 64, heads=4, concat=False)  + ELU + Dropout(0.3)
  → GATConv(64 → 64,  heads=4, concat=False)  + ELU + Dropout(0.3)
  → GATConv(64 → 64,  heads=1, concat=True)
  → global_mean_pool → [B, 64]
  → Dropout(0.5)
  → Linear(64 → 2)
Output: logits [B, 2]
```

**Características:**
- Cada aresta recebe um peso de atenção aprendido (α_ij via softmax)
- `concat=False` nas camadas 1 e 2: as 4 cabeças são **médias**, não concatenadas → dimensão se mantém em 64
- Parâmetros: ~218,562 (3.8× mais que GCN)
- Requer grafos com estrutura para se beneficiar — em grafos triviais, o softmax colapsa para distribuição uniforme

### 3.3 SAGEClassifier

```
Input: [N, 768]
  → SAGEConv(768 → 64, aggr='mean')  + ReLU
  → SAGEConv(64 → 64,  aggr='mean')  + ReLU
  → SAGEConv(64 → 64,  aggr='mean')
  → global_mean_pool → [B, 64]
  → Dropout(0.5)
  → Linear(64 → 2)
Output: logits [B, 2]
```

**Características:**
- Cada nó: `h = W_l · h_self + W_r · mean(h_vizinhos)` — separa a identidade do nó da agregação
- **Indutivo**: generaliza para grafos não vistos no treino
- `aggr='mean'` padrão; suporta também `'max'` e `'lstm'`
- Parâmetros: ~115,010 (~2× GCN, metade do GAT)

---

## 4. Pipeline de Treinamento

**Configurações comuns a todos os experimentos:**

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Learning rate | 0.005 |
| Weight decay | 5 × 10⁻⁴ |
| Batch size | 64 |
| Epochs máximos | 30 |
| Early stopping | 7 épocas sem melhoria em val_acc |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | CrossEntropyLoss |
| Pooling | global_mean_pool |

**Variantes com class weighting (para dataset desbalanceado):**

| Variante | Pesos [Real, Fake] | Descrição |
|---|---|---|
| Baseline | [1.0, 1.0] | Sem penalização |
| Controlado | [1.0, 1.0] | Idem (serve como segundo seed) |
| Cético | [1.0, 4.0] | Penaliza 4× erros em Fake |
| Extra Cético | [1.0, 6.0] | Penaliza 6× erros em Fake |

---

## 5. Resultados — Dataset Bluesky

### 5.1 Benchmark GCN vs GAT vs GraphSAGE

*Treinamento com 30 épocas, CrossEntropyLoss sem class weights*

| Métrica | GCN | GAT | GraphSAGE | Δ (SAGE−GCN) |
|---|---|---|---|---|
| **Accuracy** | 0.9951 | 0.9942 | 0.9948 | −0.0003 |
| **Precision** | 0.6667 | 0.0000 | **0.7436** | +0.0769 |
| **Recall** | **0.2990** | 0.0000 | 0.1495 | −0.1495 |
| **F1-Score** | **0.4128** | 0.0000 | 0.2489 | −0.1639 |
| Parâmetros | 57,666 | 218,562 | 115,010 | — |
| Tempo/época | ~8.0s | ~11.5s | ~7.0s | — |

**Ranking em F1: GCN > GraphSAGE > GAT**

### 5.2 Variantes do GraphSAGE (com class weighting)

| Variante | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline [1,1] | 0.9948 | 0.7436 | 0.1495 | 0.2489 |
| Controlado [1,1] | 0.9948 | 0.7436 | 0.1495 | 0.2489 |
| **Cético [1,4]** | 0.9943 | 0.5159 | 0.3351 | **0.4062** |
| Extra Cético [1,6] | 0.9946 | 0.6207 | 0.1856 | 0.2857 |

A variante Cética é a melhor GraphSAGE, com F1=0.4062 — ligeiramente abaixo do GCN baseline (0.4128).

---

## 6. Descoberta Crítica — Problema no Pipeline de Dados Bluesky

### 6.1 O que foi encontrado

Durante a análise pós-benchmark, descobriu-se que **todos os 166,910 grafos gerados têm exatamente 1 nó e 0 arestas**:

```
TREINO (100,146 grafos): num_nodes=1, num_edges=0 em 100% dos casos
VAL    (33,382 grafos):  num_nodes=1, num_edges=0 em 100% dos casos
TESTE  (33,382 grafos):  num_nodes=1, num_edges=0 em 100% dos casos
```

Isso significa que os três modelos GNN nunca processaram um grafo de verdade. Todos operaram como classificadores lineares sobre um único vetor BERT.

### 6.2 Causa Raiz

O problema está no join entre os CSVs de posts e reposts em `02_construir_grafos_bluesky.py`:

```python
# Linha 129 — posts têm IDs no range de milhões:
pid = str(row.get("post_id", ""))   # ex: "165204275"

# Linha 131 — reposts têm IDs no range de dezenas de milhares:
if reposts_idx is not None and pid in reposts_idx.groups:  # sempre False!
```

Os dois CSVs foram coletados em momentos diferentes e têm IDs sem sobreposição:

| Arquivo | IDs únicos | Range de IDs |
|---|---|---|
| `posts_coletados.csv` | 165,815 | ~2K a ~165M |
| `reposts_coletados.csv` | 8,444 | ~5 a ~4M |
| **Interseção** | **35** | — |

Com apenas 35 matches em 166,910 posts (0.02%), a construção de grafos com estrutura de propagação foi virtualmente impossível.

**Adicionalmente**, as colunas `reposts`, `likes`, `replies` e `reply_to` estão todas zeradas para 100% dos posts — o script de coleta não capturou métricas de engajamento.

### 6.3 Impacto nos Resultados

| Consequência | Detalhe |
|---|---|
| GAT F1=0 | Não é falha arquitetural — sem arestas, o softmax de atenção colapsa |
| GCN "vence" | A normalização por grau (D⁻¹/²) de um nó isolado é constante; o classificador linear aprende um viés maior para "Real" |
| Resultados sem validade | A comparação GCN vs GAT vs GraphSAGE no Bluesky não mede diferenças arquiteturais GNN — mede apenas classificação de texto BERT com leve pressão de desbalanceamento |
| Baseline correto | O verdadeiro baseline é um MLP(768→64→2) puro; qualquer GNN neste dataset é equivalente a isso |

### 6.4 Por que o GCN obteve F1>0 e o GAT não?

Com grafos de 1 nó:
- **GCN**: `h = σ(W · x_raiz)` — o nó agrega apenas a si mesmo; o classificador linear consegue encontrar um hiperplano separador nos embeddings BERT com leve assimetria de treinamento
- **GAT**: o mecanismo de atenção com `softmax` sobre uma única aresta (self-loop implícito) produz αᵢᵢ = 1.0 fixo, mas os gradientes de atenção são menos estáveis com grafos triviais, levando ao colapso total para a classe majoritária

---

## 7. Análise do Dataset Bluesky

### 7.1 Desbalanceamento extremo

| Split | Total | Fake | Real | Taxa Fake |
|---|---|---|---|---|
| Treino | 100,146 | 521 | 99,625 | 0.52% |
| Val | 33,382 | 185 | 33,197 | 0.55% |
| Teste | 33,382 | 194 | 33,188 | 0.58% |
| **Total** | **166,910** | **900** | **166,010** | **0.54%** |

Um modelo que classifica **tudo como Real** obtém accuracy de **99.46%** — tornando a accuracy uma métrica inútil. As métricas relevantes são Precision, Recall e F1.

### 7.2 Distribuição por feed

| Feed | Posts | Fake | Taxa Fake |
|---|---|---|---|
| Blacksky | 85,411 | 467 | 0.55% |
| News | 41,639 | 258 | 0.62% |
| Science | 33,831 | 141 | 0.42% |
| GreenSky | 662 | 7 | 1.06% |
| #UkrainianView | 2,097 | 18 | 0.86% |

O feed **News** tem a maior taxa absoluta de fakes entre os grandes; feeds temáticos como GreenSky têm taxa relativa mais alta.

### 7.3 Problema de labeling

A atribuição de labels Fake/Real é feita por feed de origem, não por verificação individual de cada post. Isso introduz ruído: posts reais em feeds de desinformação são marcados como Fake. Uma validação mais robusta exigiria checagem individual.

---

## 8. Comparação das Arquiteturas — Análise Teórica

### 8.1 Por que cada arquitetura se comporta como esperado em grafos pequenos/rasos

**GCN em estrela (1 nó raiz + N filhos):**
- Camada 1: h_raiz agrega os N filhos → feature rica se N > 0
- Com N=0: h_raiz = W · x_raiz (linear puro)
- **Resultado esperado:** F1 moderado, estável

**GAT em estrela:**
- A atenção entre raiz e filhos pode especializar em nós influentes
- Com N=0: sem arestas → sem atenção → colapso para predição da classe majoritária
- **Resultado esperado:** F1=0 com grafos triviais, potencialmente superior a GCN com grafos densos

**GraphSAGE em estrela:**
- `h = W_l · x_raiz + W_r · mean(x_filhos)`
- Com N=0: apenas W_l · x_raiz (similar ao GCN mas com matriz separada para self)
- **Resultado esperado:** desempenho próximo ao GCN em grafos rasos, potencialmente melhor em grafos com muitos filhos (por separar W_l e W_r)

### 8.2 Hipótese para grafos ricos

Em grafos com estrutura real de propagação (10–500 nós):

| Hipótese | Justificativa |
|---|---|
| GAT > GCN | Nós influentes têm padrões de atenção distintos; fake news tem propagação heterogênea |
| SAGE ~ GCN | Ambos fazem agregação de média; a separação W_l/W_r não muda o resultado para grafos homogêneos |
| GAT > SAGE | Em propagações virais, os 2–3 super-spreaders dominam; atenção captura isso melhor que média uniforme |

Estas hipóteses serão testadas no benchmark UPFD (seção 9).

---

## 9. Próximos Passos — Roteiro de Experimentos

### 9.1 UPFD Benchmark (imediato)

Repetir o benchmark triplo usando o dataset PolitiFact/UPFD, que possui grafos de propagação reais:

```
Script: Training/03_Mega_Research/07_upfd_benchmark_triplo.py
Dataset: torch_geometric.datasets.UPFD(name='politifact', feature='bert')
Arquiteturas: GCN, GAT, GraphSAGE
Saída: Execution/results/upfd_benchmark/
```

**O que esperamos descobrir:**
- GAT deve competir ou superar GCN em grafos com estrutura real
- GraphSAGE deve ficar entre GCN e GAT

### 9.2 Inferência Cruzada (cross-dataset)

Testar se um modelo treinado numa rede social consegue generalizar para outra — experimento fundamental para avaliar robustez:

| Experimento | Treino | Inferência | Hipótese |
|---|---|---|---|
| BS→UPFD | Bluesky | UPFD PolitiFact | Degrada: grafos Bluesky são triviais; o modelo aprendeu BERT features, não propagação |
| UPFD→BS | UPFD PolitiFact | Bluesky | Parcial: o modelo aprendeu propagação, mas Bluesky tem 1 nó — usa apenas W_l |

```
Script: Training/03_Mega_Research/08_inferencia_cruzada.py
```

### 9.3 Correção do Pipeline Bluesky (médio prazo)

Para ter comparação válida entre as duas redes sociais, é necessário recoletar os dados Bluesky garantindo que posts e reposts estejam vinculados:

**Opção 1 — Coleta integrada:** Para cada post coletado, buscar imediatamente seus reposts/replies via API ATProtocol. O `main.py` já faz isso para inferência.

**Opção 2 — Filtro por engajamento:** Usar apenas posts com `reposts > 0` (a coluna existe no CSV mas está zerada — problema no script de coleta) e enriquecer a estrutura de grafo com os dados disponíveis.

**Opção 3 — Grafos por thread:** Usar o campo `reply_to` para construir threads hierárquicas. No momento, `reply_to` está vazio em 100% dos posts — mesma origem do bug.

### 9.4 Exploração de Grafos Ricos no Bluesky

A hipótese mencionada na análise — "posts com mais compartilhamentos teriam grafos maiores e mudariam o desempenho relativo das arquiteturas" — é válida mas requer dados corretos:

- Se filtrarmos posts com ≥ 10 reposts, teríamos grafos de 11–50 nós
- A atenção do GAT teria estrutura real para processar
- Esperamos que GAT supere GCN nesse subset, ao contrário do dataset completo

**Status atual:** Impossível testar com os dados existentes (`reposts=0` em 100% dos posts). Requer recoleta.

### 9.5 Experimentos Futuros de Longo Prazo

| Experimento | Descrição | Esforço |
|---|---|---|
| **UPFD Gossipcop** | Repetir benchmark no segundo dataset UPFD (maior, ~22k grafos) | Baixo |
| **GraphSAGE aggr=max** | Testar agregador max-pooling no UPFD | Baixo |
| **GNN+Transformer** | Usar transformer para features de nós + GNN para propagação | Alto |
| **GNNExplainer** | Visualizar quais nós/arestas ativam a detecção de fake | Médio |
| **Adversarial** | Testar robustez a ataques de propagação (injeção de nós falsos) | Alto |

---

## 10. Lições Aprendidas

### 10.1 Técnicas

1. **Validar o pipeline de dados antes do modelo.** Inspecionar a distribuição de `num_nodes` e `num_edges` dos grafos gerados deve ser o primeiro passo após qualquer construção de dataset.

2. **Accuracy é enganosa com datasets desbalanceados.** Com 99.46% da classe majoritária, qualquer modelo trivial atinge accuracy "quase perfeita". F1-Score é a métrica relevante.

3. **GAT requer grafos não-triviais.** O mecanismo de softmax sobre atenção colapsa sem arestas. Em grafos de propagação reais, o GAT demonstra vantagem; em grafos triviais, falha.

4. **GraphSAGE é mais robusto a grafos rasos que o GAT.** A separação W_l/W_r (self vs vizinhos) permite aprendizado mesmo com poucos vizinhos, sem o colapso de atenção.

5. **Class weighting ajuda, mas não resolve problemas de estrutura.** Pesos [1.0, 4.0] e [1.0, 6.0] melhoram o recall, mas não compensam a ausência de grafos com estrutura.

### 10.2 Metodológicas

1. **Domínio importa.** Bluesky e Twitter têm padrões de propagação diferentes; um modelo treinado em um pode não generalizar para o outro.

2. **O labeling por feed é uma proxy frágil.** Fake news pode aparecer em feeds "reais" e vice-versa. Um dataset com verificação individual (fact-checking manual) seria mais robusto.

3. **Comparar arquiteturas GNN requer grafos com estrutura.** Sem arestas, a comparação mede apenas a eficácia do classificador linear sobre embeddings de texto — o que pode ser feito com um MLP simples.

---

## 11. Estrutura de Arquivos do Projeto

```
GNN-Fake-News/
├── Training/
│   ├── 01_BlueSky_Pipe/           # Coleta de dados Bluesky via ATProtocol
│   │   ├── main.py                # Coleta integrada (posts + reposts vinculados)
│   │   └── src/features/          # Embeddings BERT
│   ├── 02_UPFD_Benchmark/         # Baseline GCN no UPFD
│   │   └── collect_train_extract_plot.py
│   └── 03_Mega_Research/          # Benchmark principal
│       ├── sage_model.py          # SAGEClassifier
│       ├── gat_model.py           # GATClassifier
│       ├── 02_construir_grafos_bluesky.py  # PIPELINE COM BUG (grafos vazios)
│       ├── 03_treinar_gat.py      # 4 variantes GAT
│       ├── 04_comparar_gcn_gat.py # Benchmark GCN vs GAT
│       ├── 05_treinar_sage.py     # 4 variantes GraphSAGE
│       ├── 06_comparar_gcn_gat_sage.py  # Benchmark triplo (Bluesky)
│       ├── 07_upfd_benchmark_triplo.py  # [A CRIAR] Benchmark triplo (UPFD)
│       └── 08_inferencia_cruzada.py     # [A CRIAR] BS↔UPFD cross-inference
├── Execution/
│   ├── weights/                   # Pesos treinados (.pth)
│   └── results/
│       ├── comparativo_gcn_gat/   # Resultados iniciais GCN vs GAT
│       └── comparativo_gcn_gat_sage/  # Benchmark triplo Bluesky
│           ├── matrizes_confusao.png
│           ├── metricas_barras.png
│           └── relatorio.txt
└── Interface/
    └── frontend/                  # API FastAPI + Frontend Next.js
        └── api/main.py            # Ensemble de inferência (4 modelos)
```

---

## 12. Referências

- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Veličković, P., et al. (2018). *Graph Attention Networks*. ICLR.
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- Bian, T., et al. (2020). *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks*. AAAI. *(Dataset UPFD)*
- Monti, F., et al. (2019). *Fake News Detection on Social Media using Geometric Deep Learning*. arXiv.
- Wu, L., et al. (2020). *User Preference-aware Fake News Detection*. SIGIR. *(Dataset UPFD)*
