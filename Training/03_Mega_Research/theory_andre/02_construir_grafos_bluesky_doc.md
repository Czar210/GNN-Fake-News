# Documentação Técnica: 02_construir_grafos_bluesky.py

## Metadados

- **Arquivo analisado:** `02_construir_grafos_bluesky.py`
- **Caminho:** `03_Mega_Research/02_construir_grafos_bluesky.py`
- **Data de análise:** 2026-04-24
- **Tipo de abordagem:** Pré-processamento / construção de grafo (pipeline para GNN)
- **Modelos principais:** SentenceTransformer `paraphrase-multilingual-mpnet-base-v2` (BERT multilingual)
- **Datasets utilizados:** `Zaras210/bluesky-fake-news-dataset` (via CSVs do script 01)
- **Contribuição para a questão central:** Este script é o equivalente Bluesky do script 00 (FakeNewsNet). A variante `notext` — que remove todos os embeddings textuais e usa apenas 3 features posicionais — é um experimento ablation direto sobre a questão "o que as GNNs aprendem: texto ou estrutura?". Se a GNN com `notext` atingir accuracy próxima à versão com BERT, isso sugere que a estrutura topológica do grafo é o sinal dominante — evidência central para o debate GNN vs. NLP.

---

## 1. Visão Geral do Script

O script transforma os CSVs gerados por `01_baixar_dataset_hf.py` em objetos `torch_geometric.data.Data` prontos para treinamento com GNNs. A estrutura de grafo produzida é idêntica ao script 00: estrela plana com o post raiz no nó 0 e cada repost como nó filho. As features nodais combinam embeddings BERT de 768 dimensões com 3 features posicionais manuais.

A inovação principal em relação ao script 00 é a **filtragem agressiva antes da geração de embeddings** (linhas 282–289): o dataset Bluesky tem ~166.000 posts, mas apenas ~994 têm reposts suficientes para formar grafos válidos. Gerar embeddings BERT para todos os 166k posts causaria estouro de memória (*OOM*) no ambiente Windows utilizado. A filtragem prévia resolve isso reduzindo o conjunto de encode para ~0,6% do total.

Uma quarta variante de features, `notext`, é introduzida: sem BERT, o nó é representado apenas pelos 3 scalares posicionais. Isso transforma a GNN em um classificador puramente topológico, separando o sinal de estrutura do sinal de texto — experimento fundamental para a questão central do TCC.

---

## 2. Arquitetura e Componentes Principais

### 2.1 PyTorch Geometric — Objeto `Data`

**Descrição técnica:**
`torch_geometric.data.Data` é o container fundamental do PyG para um único grafo. Armazena as features nodais (`x`), a conectividade em formato COO (`edge_index`) e atributos opcionais como o rótulo (`y`). O formato COO representa a matriz de adjacência como dois vetores de índices: `edge_index[0]` (nós fonte) e `edge_index[1]` (nós destino).

**Fundamento matemático:**

Para um grafo $G = (V, E)$, o formato COO representa:
$$\text{edge\_index} = \begin{bmatrix} u_1 & u_2 & \cdots & u_{|E|} \\ v_1 & v_2 & \cdots & v_{|E|} \end{bmatrix} \in \mathbb{Z}^{2 \times |E|}$$

onde cada coluna $(u_i, v_i)$ representa uma aresta $u_i \to v_i$.

O mini-batching de grafos de tamanhos diferentes — necessário para treinar GNNs em conjuntos de grafos heterogêneos — é realizado empilhando os `edge_index` com offset de nó: se $G_1$ tem $n_1$ nós e $G_2$ tem $n_2$ nós, o batch $G_1 \cup G_2$ tem $n_1 + n_2$ nós e os índices de $G_2$ são somados de $n_1$.

**Embasamento acadêmico:**

> 📖 **Fey, M.; Lenssen, J. E. (2019)** — "Fast Graph Representation Learning with PyTorch Geometric"
> *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*
> arXiv: `1903.02428`
> **Localização:** Seção 3 (Data Handling of Graphs), Seção 3.1 (The Data Object), Seção 3.2 (Mini-batches)
> **Relevância:** Define a estrutura `Data` e o formato COO (`edge_index`) usados diretamente nas linhas 123–136 do script. A Seção 3.2 descreve como o PyG lida com grafos de tamanhos diferentes em batches — mecanismo ativado implicitamente ao usar `DataLoader` nos scripts de treinamento.

**No código:**
> Linha 136: `return Data(x=x, edge_index=edge_index)` — construção sem label (y é adicionado externamente).
> Linha 171: `grafo.y = torch.tensor([label], dtype=torch.long)` — atribuição dinâmica de atributo ao objeto `Data`.

---

### 2.2 Variante `notext` — GNN Puramente Topológica

**Descrição técnica:**
Quando `--feature-variant notext`, o campo `inclui_bert` é `False` e cada nó é representado apenas pelos 3 scalares posicionais: `[is_root, grau_norm, pos]`. A dimensão resultante por nó é 3 em vez de 771.

```
Raiz:    x₀ = [1.0,  grau_norm,  0.0]  ∈ ℝ³
Filho i: xᵢ = [0.0,  0.0,        i/N]  ∈ ℝ³
```

Esta configuração isola completamente a componente estrutural da GNN, removendo qualquer sinal semântico de texto. Se um modelo treinado com `notext` atingir accuracy comparável ao modelo com BERT completo (`full`), isso indica que a topologia em estrela plana já é suficientemente informativa — questionando a necessidade da componente GNN propriamente dita para este problema.

**Embasamento acadêmico:**

> 📖 **Karn, I.; Jensen, D. (2025)** — "The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News"
> arXiv: `2512.06638`
> **Localização:** Seção principal de resultados empíricos
> **Relevância:** Demonstra que em benchmarks padrão de fake news, MLPs combinam ou superam GNNs usando as mesmas features nodais. Crucialmente: "performance collapses under feature shuffling" mas permanece estável com arestas aleatorizadas — provando que features nodais (texto), e não estrutura, impulsionam o desempenho. A variante `notext` testa exatamente o lado oposto: estrutura pura sem texto.

> 📖 **Karn & Jensen (2025)**, mesma referência acima:
> "Over 75% of nodes are only one hop from the root" em grafos de fake news — confirma que a topologia em estrela plana (usada neste script) é a norma nos benchmarks, limitando a capacidade expressiva do *message passing*.

> 📖 **Bian, T.; Xiao, X.; Xu, T.; Zhao, P.; Huang, W.; Rong, Y.; Huang, J. (2020)** — "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks"
> *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-20)*, pp. 549–556
> arXiv: `2001.06362` | AAAI DOI: `10.1609/aaai.v34i01.5393`
> **Localização:** Seção 1 (Introduction), Seção 3 (Methodology)
> **Relevância:** Motivação oposta: demonstra que grafos de propagação *ricos* (árvore bidirecional com histórico temporal) contêm sinal topológico discriminativo para detecção de rumores. Evidência de que a limitação não é da GNN, mas da qualidade da topologia — reforçando por contraste a limitação da estrela plana deste script.

**No código:**
> Linhas 102–110: `inclui_bert = feature_variant != "notext"`. Se `notext`, `x_raiz = posicionais_raiz` (shape `[3]`).
> Linhas 323–325: para `notext`, embeddings são zeros placeholder `[[0.0]*768]*N` — não são usados na construção.
> Linha 249: `esperado_dim = 3 if args.feature_variant == "notext" else 771`.

---

### 2.3 Filtragem Agressiva Antes do BERT

**Descrição técnica:**
O dataset Bluesky tem ~166.000 posts, mas apenas ~994 têm pelo menos 2 reposts. Gerar embeddings BERT para 166k posts exigiria ~166k × 768 × 4 bytes ≈ **510 MB** de memória para os vetores, mais overhead do modelo (~420 MB), totalizando potencialmente >1 GB. Em ambiente Windows com GPU limitada, isso causaria OOM.

A solução (linhas 285–289) é computar o conjunto `posts_qualificados_ids` primeiro, filtrar `df_posts` para apenas esses posts, e então gerar BERT somente para os ~994 qualificados.

**Fundamento matemático:**

Seja $P$ o conjunto de todos os posts e $Q \subseteq P$ o subconjunto com $|reposts| \geq$ `min_reposts`. O custo de encoding BERT é:

$$C_{\text{BERT}} = O(|Q| \cdot d_{\text{BERT}} \cdot L_{\text{max}}) \quad \text{onde } |Q| \ll |P|$$

Para $|P| = 166\,000$ e $|Q| \approx 994$: redução de $166\times$ no custo de encoding.

**No código:**
> Linha 285: `posts_qualificados_ids = set(reposts_por_post[reposts_por_post >= args.min_reposts].index.astype(str))`
> Linha 288: `df_posts = df_posts[df_posts["post_id"].isin(posts_qualificados_ids)].reset_index(drop=True)`

**Importância do `reset_index(drop=True)`:** Após filtrar `df_posts`, os índices numéricos do DataFrame são remapeados de 0 a $|Q|-1$. Isso é **crítico** porque a lista `embeddings` é indexada por posição — `embeddings[i]` deve corresponder à i-ésima linha de `df_posts`. Sem `reset_index`, os embeddings seriam associados às linhas erradas.

---

### 2.4 Cache por Configuração

**Descrição técnica:**
O cache de embeddings BERT é nomeado `_bert_bluesky_cache_min{args.min_reposts}.pt`, incluindo o valor de `min_reposts` no nome. Isso evita conflito entre execuções com diferentes limiares de filtragem, pois o conjunto de posts qualificados (e portanto os embeddings gerados) muda com `min_reposts`.

**Comparação com script 00:** O script 00 usa um único cache `_bert_titulos_cache.pt` sem parâmetro, pois o conjunto de posts nunca muda (todos os artigos do FakeNewsNet recebem embedding, independente de filtros). Aqui, o conjunto muda com `min_reposts`, justificando o cache parametrizado.

**No código:**
> Linha 302: `cache_path = OUTPUT_DIR / f"_bert_bluesky_cache_min{args.min_reposts}.pt"`

---

## 3. Pipeline de Dados e Pré-processamento

### 3.1 Carregamento e Normalização dos CSVs

O script carrega os dois CSVs do script 01 e aplica normalização defensiva:
- `df_posts["texto"].fillna("").astype(str)` — previne erro de encoding BERT em valores nulos
- `df_posts["label"].fillna(0).astype(int)` — garante tipo inteiro consistente

**No código:**
> Linhas 267–270: carregamento e normalização.

---

### 3.2 Cálculo de N_max_global

**Descrição técnica:**
`N_max_global` é o denominador da feature `grau_norm` e representa o número máximo de reposts entre os posts qualificados:

$$N_{\max} = \min\left(\max_{j \in \mathcal{D}_{\text{qualificados}}} |\text{reposts}_j|,\ \text{max\_nos} - 1\right)$$

**Diferença crítica em relação ao script 00:**
O script 00 calcula `N_max_global` apenas sobre os índices de treino (`df_train = df_filtrado.iloc[train_idx]`), evitando *normalization leakage*. Este script calcula sobre **todos** os posts qualificados, incluindo validação e teste. Isso viola o princípio documentado por Shchur et al. (2018) e implementado corretamente no script 00.

**Impacto prático:** Para a feature `grau_norm`, o valor calculado para posts de validação/teste usa informação do conjunto completo $\mathcal{D}_{\text{qualificados}}$. Se um post de teste tem o maior número de reposts de todo o dataset, ele "vaza" essa informação para a normalização de todos os outros posts.

**No código:**
> Linhas 328–330: `N_max_global` calculado sobre todos os qualificados (não apenas treino).

---

### 3.3 Convencão de Label

**Atenção crítica:** Este script herda os labels do `posts_coletados.csv` gerado pelo script 01, onde `_label()` retorna `1=fake`, `0=real`. O script 00 (FakeNewsNet) usa a convenção UPFD inversa: `0=fake`, `1=real`.

| Script | Dataset | Label fake | Label real |
|--------|---------|-----------|-----------|
| 00 | FakeNewsNet | 0 | 1 |
| 02 | Bluesky | 1 | 0 |

Esta inconsistência não causa erro interno em cada script isolado, mas torna os modelos **não-diretamente-comparáveis**: um modelo treinado em Bluesky com threshold de decisão em 0.5 para classe 1 está classificando a classe oposta em relação ao mesmo threshold para FakeNewsNet. Ao reportar resultados no TCC, a convenção de cada dataset deve ser explicitada.

**No código:**
> Linha 168: `label = int(row.get("label", 0))` — passagem direta do CSV.
> Linha 171: `grafo.y = torch.tensor([label], dtype=torch.long)`

---

## 4. Construção do Grafo

### 4.1 Topologia em Estrela e Limitações

A topologia é idêntica ao script 00: estrela plana com raiz → filhos. Ver documentação detalhada em `00_construir_grafos_fakenewsnet_doc.md`, Seções 2.2 e 6.2. A limitação principal (perda do sinal de cascata de propagação) aplica-se igualmente aqui.

### 4.2 Separação Raiz–Label

Nota arquitetural: `_construir_grafo_posicional()` retorna `Data(x, edge_index)` **sem o atributo `y`**. O label é adicionado em `construir_grafos()` após a chamada. Esta separação é uma boa prática de design — a função de construção topológica não precisa conhecer o label, que é um metadado externo ao grafo em si.

**No código:**
> Linha 136: `return Data(x=x, edge_index=edge_index)` — sem `y`.
> Linha 171: `grafo.y = ...` — atribuição posterior.

---

## 5. Métricas de Avaliação

Este script não executa treinamento. As métricas reportadas são de diagnóstico:

| Métrica | Fórmula | Propósito |
|---------|---------|-----------|
| Taxa de fake | $N_{\text{fake}} / N_{\text{total}}$ | Verificar desbalanceamento induzido pela heurística |
| Nós médios por split | $\bar{n} = \frac{1}{N}\sum\|V_i\|$ | Caracterizar tamanho dos grafos por split |
| Arestas médias | $\bar{e} = \bar{n} - 1$ (estrela plana) | Confirmar topologia correta |

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos Fortes desta Abordagem

**Eficiência de memória:** A filtragem antes do BERT é uma solução pragmática que viabiliza o experimento em hardware doméstico, sem comprometer a validade dos grafos construídos.

**Variante `notext` como experimento ablation:** É a contribuição mais relevante deste script para o TCC. Permite medir diretamente quanto do desempenho da GNN vem da estrutura topológica versus do texto — separação que a maioria dos papers do estado da arte não realiza explicitamente.

**Cache parametrizado:** Permite explorar diferentes valores de `min_reposts` sem regenerar embeddings desnecessariamente.

### 6.2 Limitações Identificadas

**L1 — N_max_global com data leakage (vs. script 00):**
Diferentemente do script 00 que computa `N_max_global` apenas no treino, este script usa todos os qualificados. Para o TCC, esta inconsistência deve ser reportada como limitação metodológica — ou o script 02 deve ser corrigido para paridade com o script 00.

**L2 — Inconsistência de convenção de label entre datasets:**
FakeNewsNet usa `0=fake`, Bluesky usa `1=fake`. Qualquer análise conjunta ou transferência de modelos entre datasets deve converter a convenção explicitamente.

**L3 — Placeholder de zeros para `notext`:**
Para a variante `notext`, a lista `embeddings` é preenchida com zeros mas não é usada. Isso é inofensivo funcionalmente mas ocupa $|Q| \times 768 \times 8$ bytes ≈ 5,8 MB desnecessariamente. Poderia ser `None` com verificação antes do uso.

**L4 — `--max-posts` antes da filtragem:**
O argumento `--max-posts N` trunca `df_posts.head(N)` antes de filtrar por reposts. Se os primeiros N posts do CSV não tiverem reposts, o script termina com "[ERRO] Nenhum grafo construido" — comportamento confuso para debug.

**L5 — Labels herdados da heurística fraca do script 01:**
Ver `01_baixar_dataset_hf_doc.md`, Seção 7.1 — a qualidade dos labels é a principal ameaça à validade dos experimentos Bluesky.

### 6.3 Comparação com Estado da Arte

| Configuração | Features | Dim | Resultado esperado |
|---|---|---|---|
| `full` | BERT + pos | 771 | Maior accuracy (sinal textual dominante) |
| `notext` | apenas posicionais | 3 | Menor accuracy se texto > estrutura |
| `pos-min` | BERT + is_root + pos | 771 | Ablation de grau_norm |
| `pos-grau` | BERT + is_root + grau_norm | 771 | Ablation de pos |

> 📖 **Karn & Jensen (2025)**, arXiv:2512.06638: "performance collapses under feature shuffling" mas é estável com arestas aleatorizadas — predição: `notext` deve ter desempenho significativamente inferior ao `full`, confirmando que texto domina sobre estrutura neste tipo de grafo raso.

### 6.4 Resposta Parcial à Questão do TCC

Se `notext` (3 dims posicionais) produz accuracy próxima ao `full` (771 dims com BERT), isso sugere que a topologia de estrela plana carrega sinal suficiente para a GNN funcionar sem texto — potencialmente mais forte do que NLP puro precisaria. Se `notext` falhar (próximo a baseline aleatório) e `full` funcionar, o resultado reforça que o BERT é o componente decisivo, e a GNN é apenas um wrapper do embedding textual. Ambos os cenários geram conclusões relevantes para a questão central do TCC.

---

## 7. Análise de Código

### 7.1 Erros Identificados

```python
# ❌ Linhas 328–330 — N_max_global calculado sobre todos os qualificados
#    (inclui validação e teste) — inconsistência com script 00
qualificados = reposts_por_post[reposts_por_post >= args.min_reposts]
N_max_global = int(min(qualificados.max() if len(qualificados) > 0 else 1,
                       args.max_nos - 1))

# ✅ Correção para paridade com script 00:
# Calcular N_max_global somente sobre o subconjunto de treino:
import random
rng = random.Random(RANDOM_SEED)
all_ids = list(qualificados.index)
rng.shuffle(all_ids)
n_tr = int(len(all_ids) * 0.60)
train_ids = all_ids[:n_tr]
N_max_global = int(min(
    qualificados.loc[qualificados.index.isin(train_ids)].max(),
    args.max_nos - 1
))
# Justificativa: evita normalization leakage na feature grau_norm
# (Shchur et al., 2018, arXiv:1811.05868)
```

```python
# ❌ Linhas 323–325 — Zeros placeholder para notext ocupa memória
#    sem necessidade
embeddings = [[0.0] * 768 for _ in range(len(df_posts))]

# ✅ Correção: usar None e verificar antes do uso
embeddings = None  # sinal explícito de "não usado"
# Em construir_grafos(), checar: x_bert = torch.zeros(768) if embeddings is None else ...
```

```python
# ❌ Linhas 271–272 — --max-posts trunca antes da filtragem por reposts
if args.max_posts:
    df_posts = df_posts.head(args.max_posts)
# Se os primeiros max_posts posts não têm reposts -> zero grafos

# ✅ Correção: aplicar max_posts APÓS a filtragem por qualificados:
# (mover a linha 271-272 para depois da linha 288)
df_posts = df_posts[df_posts["post_id"].isin(posts_qualificados_ids)].reset_index(drop=True)
if args.max_posts:
    df_posts = df_posts.head(args.max_posts)  # <- aqui
```

### 7.2 Ineficiências

**I1 — `df.iterrows()` para construção de grafos (linha 154):**
Mesmo problema do script 00. Para ~994 grafos, o impacto é negligenciável.

**I2 — `reposts_idx.get_group(pid)` sem verificação de `pid` vazio:**
Se algum post tem `post_id=""` (linha 156), ele pode causar match indevido no agrupamento.

### 7.3 Boas Práticas Observadas

**B1 — `reset_index(drop=True)` após filtro:** Crítico para alinhar `embeddings[i]` com `df_posts.iloc[i]`. Corretamente implementado na linha 288.

**B2 — `max(N_max_global, 1)` no denominador (linha 104):** Previne divisão por zero se `N_max_global=0` — defensivo e correto.

**B3 — Cache parametrizado por `min_reposts`:** Evita conflito de cache entre configurações diferentes.

**B4 — `random.Random(RANDOM_SEED)` local (linha 182):** Consistente com script 00, evita modificar estado global do RNG — contrasta com a má prática do script 01.

**B5 — `verificar_grafos()` pós-construção:** Verifica dimensões dos primeiros 5 grafos e imprime estatísticas de balanceamento — diagnóstico útil para detectar regressões.

**B6 — Separação `_construir_grafo_posicional()` / `construir_grafos()`:** O label (`y`) é responsabilidade do chamador, não do construtor topológico — boa separação de concerns.

---

## 8. Referências Bibliográficas

1. FEY, M.; LENSSEN, J. E. **Fast Graph Representation Learning with PyTorch Geometric**. In: *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*, 2019. Disponível em: https://arxiv.org/abs/1903.02428

2. KARN, I.; JENSEN, D. **The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News**. arXiv, 2025. Disponível em: https://arxiv.org/abs/2512.06638

3. BIAN, T.; XIAO, X.; XU, T.; ZHAO, P.; HUANG, W.; RONG, Y.; HUANG, J. **Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks**. In: *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-20)*, 2020, pp. 549–556. DOI: `10.1609/aaai.v34i01.5393`. Disponível em: https://arxiv.org/abs/2001.06362

4. REIMERS, N.; GUREVYCH, I. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. In: *EMNLP-IJCNLP 2019*, Hong Kong, pp. 3982–3992. Disponível em: https://arxiv.org/abs/1908.10084

5. SHCHUR, O.; MUMME, M.; BOJCHEVSKI, A.; GÜNNEMANN, S. **Pitfalls of Graph Neural Network Evaluation**. *Workshop on Relational Representation Learning, NeurIPS 2018*. Disponível em: https://arxiv.org/abs/1811.05868

6. DOU, Y. et al. **User Preference-aware Fake News Detection (UPFD)**. *SIGIR'21*, 2021. DOI: `10.1145/3404835.3462990`. Disponível em: https://arxiv.org/abs/2104.12259

7. ZARAS210. **bluesky-fake-news-dataset** [Dataset]. HuggingFace Hub, 2023–2024. Disponível em: https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| COO (*Coordinate Format*) | Formato esparso para matrizes de adjacência: dois vetores (linha, coluna) listam entradas não-nulas. Formato nativo do PyTorch Geometric. | Fey & Lenssen (2019), Seção 3.1 |
| Mini-batching em grafos | Técnica do PyG para processar grafos de tamanhos diferentes em lote: empilha grafos em um grafo desconexo maior com offsets de índice de nó. | Fey & Lenssen (2019), Seção 3.2 |
| Variante `notext` | Configuração de features onde BERT é removido e apenas as 3 features posicionais são usadas. Produz grafos com 3 dims/nó. | Definido neste script |
| OOM (*Out Of Memory*) | Erro de estouro de memória GPU/RAM. Ocorre quando tensores alocados excedem a memória disponível. | — |
| Normalization Leakage | Uso de estatísticas calculadas sobre o conjunto completo (treino + teste) para normalizar features — forma de data leakage que infla métricas de avaliação. | Shchur et al. (2018) |
| Propagation Graph | Grafo que representa como uma notícia se espalha em uma rede social. Nós = usuários/posts; arestas = ações de compartilhamento. | Bian et al. (2020), Seção 1 |
| `reset_index(drop=True)` | Operação pandas que remapeia os índices de um DataFrame filtrado para [0, n-1] contíguo. Essencial para alinhar índices com listas Python externas. | — |
