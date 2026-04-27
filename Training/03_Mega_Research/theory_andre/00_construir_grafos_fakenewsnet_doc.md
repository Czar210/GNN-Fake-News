# Documentação Técnica: 00_construir_grafos_fakenewsnet.py

## Metadados

- **Arquivo analisado:** `00_construir_grafos_fakenewsnet.py`
- **Caminho:** `03_Mega_Research/00_construir_grafos_fakenewsnet.py`
- **Data de análise:** 2026-04-24
- **Tipo de abordagem:** Pré-processamento / construção de grafo (pipeline para GNN)
- **Modelos principais:** SentenceTransformer `paraphrase-multilingual-mpnet-base-v2` (BERT multilingual)
- **Datasets utilizados:** FakeNewsNet PolitiFact (`politifact_fake.csv`, `politifact_real.csv`)
- **Contribuição para a questão central:** Este script é a etapa de preparação dos dados para experimentos GNN. A qualidade dos grafos construídos aqui determina diretamente o teto de desempenho dos modelos GNN avaliados. A escolha de uma topologia em estrela plana (em vez do grafo de propagação completo do UPFD) é uma simplificação que pode limitar a capacidade das GNNs de capturar padrões de difusão, afetando o debate central sobre a viabilidade das GNNs versus NLP puro.

---

## 1. Visão Geral do Script

O script constrói grafos de propagação compatíveis com o formato UPFD (User Preference-aware Fake News Detection) a partir dos CSVs públicos do FakeNewsNet PolitiFact, hospedados no repositório GitHub de Shu et al. Cada grafo representa uma notícia com seu conjunto de tweets/retweets associados, onde o nó raiz codifica o artigo e os nós filhos representam os usuários que compartilharam a notícia.

O problema resolvido é a conversão de dados tabulares do FakeNewsNet (artigos + lista de tweet IDs) em objetos `torch_geometric.data.Data` prontos para treinamento com GNNs. O script resolve ainda um problema clássico de design de features para grafos de propagação: como diferenciar o nó raiz (artigo) dos nós filhos (usuários) quando todos compartilham o mesmo embedding de texto do título.

A metodologia combina embeddings densos de 768 dimensões do modelo `paraphrase-multilingual-mpnet-base-v2` (Sentence-BERT multilingual) com três features posicionais manuais: flag `is_root`, grau normalizado da raiz (`grau_norm`) e posição relativa do filho no grafo (`pos`). Três variantes desta combinação são suportadas via `--feature-variant`, permitindo ablation study das contribuições individuais de cada feature posicional.

A saída são três arquivos `.pt` (train/val/test) com listas de objetos `Data`, particionados em proporção 60/20/20 com semente aleatória fixada. Uma proteção contra vazamento de dados (*data leakage*) é implementada explicitamente: o valor `N_max_global` (denominador da normalização de grau) é calculado exclusivamente sobre os exemplos de treino.

---

## 2. Arquitetura e Componentes Principais

### 2.1 SentenceTransformer — `paraphrase-multilingual-mpnet-base-v2`

**Descrição técnica:**
Modelo de embedding de sentenças baseado em XLM-RoBERTa, treinado com dados de paráfrases para produzir representações semânticas de 768 dimensões em 50+ idiomas. É utilizado para encodar o título de cada artigo de notícia em um vetor denso.

**Fundamento matemático:**

O modelo aplica um encoder Transformer seguido de mean pooling sobre os token embeddings:

$$\mathbf{e} = \text{MeanPool}\left(\text{Transformer}(\mathbf{w}_1, \dots, \mathbf{w}_T)\right) \in \mathbb{R}^{768}$$

onde $\mathbf{w}_1, \dots, \mathbf{w}_T$ são os tokens da sentença de entrada.

O treinamento usa uma rede siamesa com *objective* de similaridade de cosseno entre pares de paráfrases:

$$\text{loss} = 1 - \cos(\mathbf{e}_1, \mathbf{e}_2) = 1 - \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \cdot \|\mathbf{e}_2\|}$$

**Embasamento acadêmico:**

> 📖 **Reimers, N.; Gurevych, I. (2019)** — "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
> *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*, Hong Kong. pp. 3982–3992
> arXiv: `1908.10084` | ACL Anthology: `D19-1410`
> **Localização:** Seção 3 (Model), Equação 1 (Objective Function)
> **Relevância:** Define a arquitetura siamesa e o mean pooling que produz os vetores de 768 dimensões usados como features BERT no script. O modelo multilingual empregado é uma extensão direta desta arquitetura para 50+ idiomas.

> 📖 **Devlin, J.; Chang, M.-W.; Lee, K.; Toutanova, K. (2019)** — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
> *Proceedings of NAACL-HLT 2019*, Minneapolis. pp. 4171–4186
> arXiv: `1810.04805` | ACL Anthology: `N19-1423`
> **Localização:** Seção 3 (Model Architecture), Seção 3.1 (Pre-training BERT)
> **Relevância:** Base do encoder Transformer utilizado. O modelo `paraphrase-multilingual-mpnet-base-v2` usa a arquitetura Transformer do BERT como backbone.

**No código:**
> Linhas 99–110: `gerar_embeddings()` — instancia `SentenceTransformer`, chama `.encode()` com `convert_to_numpy=True` e retorna lista de vetores de 768 dims.
> Linhas 309–328: cache em disco (`_bert_titulos_cache.pt`) é verificado antes de gerar embeddings, permitindo reuso entre variantes sem re-computar.

---

### 2.2 Grafo de Propagação em Estrela

**Descrição técnica:**
Cada artigo de notícia é representado como um grafo em estrela (*star graph*): um nó raiz (o artigo) conectado diretamente a $N$ nós filhos (os tweet IDs associados). Todas as arestas são direcionadas da raiz para os filhos — não há arestas entre filhos nem entre filhos e raiz no sentido inverso.

**Fundamento matemático:**

Para um artigo com $N$ tweets, o grafo $G = (V, E)$ tem:
$$V = \{v_0, v_1, \dots, v_N\}, \quad |V| = N+1$$
$$E = \{(v_0, v_i) \mid i = 1, \dots, N\}, \quad |E| = N$$

O `edge_index` em formato COO (Coordinate format) do PyG é:
$$\text{edge\_index} = \begin{bmatrix} 0 & 0 & \cdots & 0 \\ 1 & 2 & \cdots & N \end{bmatrix} \in \mathbb{Z}^{2 \times N}$$

**Distinção em relação ao UPFD original:**

O paper UPFD (Dou et al., 2021) constrói uma árvore de propagação mais rica: as arestas entre usuários são determinadas por timestamps e relações de *follow*, resultando em estruturas DAG com, em média, 131 nós e uma topologia hierárquica. O presente script simplifica para uma estrela plana, o que:
- Perde informação de cascata de retweet
- Elimina a dependência de dados de timeline de usuário
- Reduz o risco de *data leakage* por informação de rede social

**Embasamento acadêmico:**

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'21)*, pp. 2051–2055
> DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`
> **Localização:** Seção 3.1 (Propagation Graph Construction), Tabela 1 (Dataset Statistics)
> **Relevância:** Define o formato UPFD com o qual este script é declarado compatível. A Tabela 1 reporta 314 grafos PolitiFact (157 fake, 157 real) com 131 nós/grafo em média — base de comparação para os grafos gerados por este script.

**No código:**
> Linhas 157–162: construção do `edge_index`. `src = zeros(num_filhos)` (todos partem da raiz), `dst = arange(1, N+1)`.

---

### 2.3 Features Nodais com Componente Posicional

**Descrição técnica:**
Como todos os nós de um mesmo grafo compartilham o mesmo embedding BERT (do título do artigo), as features nodais seriam idênticas sem diferenciação adicional — o Erro 1 descrito no docstring. A solução é concatenar 3 features posicionais manuais ao vetor BERT.

**Fundamento matemático:**

Para o nó raiz $v_0$ (artigo):
$$\mathbf{x}_0 = \left[\mathbf{e} \mid 1.0 \mid \frac{N}{N_{\max}} \mid 0.0\right] \in \mathbb{R}^{771}$$

Para o filho $v_i$ ($i = 1, \dots, N$):
$$\mathbf{x}_i = \left[\mathbf{e} \mid 0.0 \mid 0.0 \mid \frac{i}{N}\right] \in \mathbb{R}^{771}$$

onde:
- $\mathbf{e} \in \mathbb{R}^{768}$: embedding BERT do título do artigo
- $\text{is\_root} \in \{0, 1\}$: flag que identifica a raiz
- $\text{grau\_norm} = N / N_{\max}$: grau da raiz normalizado pelo máximo **do conjunto de treino**
- $\text{pos} = i / N$: posição relativa do filho (ordem de aparição no CSV)

**Variantes (ablation):**

| Variante | is_root | grau_norm | pos |
|----------|---------|-----------|-----|
| `full`     | ✓       | ✓         | ✓   |
| `pos-min`  | ✓       | ✗ (zerado) | ✓   |
| `pos-grau` | ✓       | ✓         | ✗ (zerado) |

**No código:**
> Linhas 139–153: `usa_grau` e `usa_pos` controlam quais features são zeradas.
> Linha 144: `grau_raiz = (num_filhos / N_max_global) if usa_grau else 0.0`
> Linhas 165–172: asserção de sanidade — verifica que há pelo menos `min(num_nos, 3)` vetores únicos em `x`, prevenindo regressão ao Erro 1.

---

### 2.4 Normalização de Grau Anti-Leakage

**Descrição técnica:**
O valor `N_max_global` (denominador de `grau_norm`) é calculado **somente** sobre os artigos do conjunto de treino, após aplicar o filtro `min_tweets`. Isso evita que informação do conjunto de validação/teste vaze para as features nodais.

**Fundamento matemático:**

$$N_{\max} = \max_{j \in \mathcal{D}_{\text{train}}} \left|\text{tweets}_j\right|, \quad \text{capped em } (\text{max\_nos} - 1)$$

Sem este cuidado, usar o máximo global seria uma forma de *normalization leakage*: um estatístico calculado sobre toda a distribuição $\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}$ codificaria informação dos conjuntos de avaliação.

**Embasamento acadêmico:**

> 📖 **Shchur, O.; Mumme, M.; Bojchevski, A.; Günnemann, S. (2018)** — "Pitfalls of Graph Neural Network Evaluation"
> *Workshop on Relational Representation Learning, NeurIPS 2018*
> arXiv: `1811.05868`
> **Localização:** Seção 3 (Experimental Design Issues), Seção 3.2 (Data Splits)
> **Relevância:** Documenta que a presença de qualquer informação dos conjuntos de teste/validação no processo de preparação dos dados pode introduzir bias otimista significativo nos resultados reportados.

**No código:**
> Linhas 296–304: `split_indices` é chamado sobre `df_filtrado`, e `N_max_global` é calculado apenas em `df_train.iloc[train_idx]`.

---

## 3. Pipeline de Dados e Pré-processamento

### 3.1 Download dos CSVs FakeNewsNet

**Descrição técnica:**
Os CSVs `politifact_fake.csv` e `politifact_real.csv` são baixados diretamente do repositório público no GitHub. O script desabilita verificação SSL (`CERT_NONE`) para contornar problemas de certificado — solução pragmática porém não recomendada para produção.

**Embasamento acadêmico:**

> 📖 **Shu, K.; Mahudeswaran, D.; Wang, S.; Lee, D.; Liu, H. (2020)** — "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media"
> *Big Data*, vol. 8, n.º 3, pp. 171–188
> DOI: `10.1089/big.2020.0062` | arXiv: `1809.01286`
> **Localização:** Seção 3 (Data Collection), Tabela 2 (Dataset Statistics), Seção 5 (Repository Structure)
> **Relevância:** Define o esquema do CSV. A Tabela 2 reporta 432 artigos fake + 624 artigos real no PolitiFact; a Seção 5 descreve a estrutura do repositório incluindo o campo `tweet_ids`.

**No código:**
> Linhas 72–78: `_fetch_csv()` — SSL desabilitado via `ctx.verify_mode = ssl.CERT_NONE`.
> Linhas 81–94: `carregar_fakenewsnet()` — concatena fake e real com labels `0` e `1` respectivamente.

---

### 3.2 Campo `tweet_ids` e Ordem de Aparição

**Descrição técnica:**
O campo `tweet_ids` no CSV contém IDs de tweets separados por tabulação (`\t`). O script usa essa ordem como proxy de posição relativa (`pos = i/N`). O docstring documenta explicitamente que esta ordem **não** é cronológica documentada — é a "ordem de aparição no CSV".

**CAVEAT crítico:** O dataset FakeNewsNet armazena tweets como arquivos separados na pasta `tweets/` com timestamps individuais, mas o CSV consolidado (`politifact_fake.csv`) não garante que a ordem dos IDs em `tweet_ids` reflete ordem temporal. Portanto, a feature `pos` captura posição relativa no CSV, não sequência temporal de propagação. Isso deve ser reportado nas conclusões (Fase 5A.5 do TCC) como limitação.

**Embasamento acadêmico:**

> 📖 **Shu et al. (2020)**, Seção 5 (Repository Structure):
> "tweets folder contains the metadata of the list of tweets associated with the news article collected as separate files for each tweet. Temporal information indicates that we record the timestamps of user engagements."
> **Relevância:** Confirma que timestamps existem nos arquivos individuais de tweets, mas não no campo `tweet_ids` do CSV consolidado — sustentando o CAVEAT do script.

---

### 3.3 Filtro por Número Mínimo de Tweets

**Descrição técnica:**
Grafos com menos de `min_tweets=2` tweets são descartados (`construir_todos()`, linha 189–191). Isso previne grafos degenerados (apenas raiz sem filhos) onde a GNN teria informação estrutural nula.

**No código:**
> Linha 189: `if len(ids) < min_tweets: descartados += 1; continue`

---

### 3.4 Particionamento 60/20/20

**Descrição técnica:**
O split é feito com `random.Random(42).shuffle(indices)` — determinístico e independente do PyTorch, garantindo reprodutibilidade sem depender de estado global de RNG.

**No código:**
> Linhas 204–213: `split_indices()` retorna índices embaralhados particionados em 60/20/20.

---

## 4. Construção do Grafo

### 4.1 Objeto `torch_geometric.data.Data`

**Descrição técnica:**
O grafo é armazenado como objeto `Data` com três atributos:
- `x`: tensor `[num_nodes, 771]` — features nodais
- `edge_index`: tensor `[2, num_edges]` em formato COO — conectividade
- `y`: tensor `[1]` — label da notícia (0=fake, 1=real)

**No código:**
> Linhas 174–178: construção do objeto `Data`.

---

### 4.2 Cap de Nós por Grafo

**Descrição técnica:**
`num_filhos = min(len(ids), max_nos - 1)` limita o grafo a `max_nos=100` nós por padrão. Isso garante grafos de tamanho uniforme-máximo e controla o custo computacional.

**No código:**
> Linha 136: `num_filhos = min(len(ids), max_nos - 1)`

---

## 5. Métricas de Avaliação

Este script não executa treinamento nem avaliação — ele apenas constrói e salva os grafos. As métricas de avaliação são aplicadas nos scripts subsequentes (e.g., `07_upfd_benchmark_triplo.py`).

No entanto, o script reporta **métricas de diagnóstico** dos grafos:

| Métrica | Fórmula | Propósito |
|---------|---------|-----------|
| Balanceamento | $\frac{N_{\text{fake}}}{N_{\text{total}}} \times 100\%$ | Detectar desbalanceamento que geraria accuracy enganosa |
| Nós médios | $\frac{1}{N}\sum_i \|V_i\|$ | Caracterizar tamanho médio dos grafos |
| Arestas médias | $\frac{1}{N}\sum_i \|E_i\|$ | Equivale a nós médios − 1 (estrela plana) |

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos Fortes desta Abordagem

**Normalização anti-leakage**: A computação de `N_max_global` exclusivamente sobre o treino é uma prática metodologicamente sólida, evitando o bias otimista documentado por Shchur et al. (2018).

**Cache de embeddings**: O cache em disco dos embeddings BERT permite testar múltiplas variantes de features sem recomputar o encoding, o que é eficiente para ablation studies.

**Compatibilidade com benchmarks**: A compatibilidade declarada com o formato UPFD permite comparação direta com os resultados de Dou et al. (2021), onde GraphSAGE+BERT atinge 84.62% de accuracy no PolitiFact.

**Ablation embutida**: As três variantes (`full`, `pos-min`, `pos-grau`) permitem isolar a contribuição de cada feature posicional, seguindo princípios de design experimental controlado.

### 6.2 Limitações Identificadas

**L1 — Topologia em estrela vs. grafo de propagação completo:**
O UPFD constrói árvores de propagação com 131 nós/grafo em média e arestas baseadas em timestamps e relações de *follow*. Este script usa estrelas planas, perdendo toda a informação de cascata de difusão. GNNs como GCN e GAT exploram a topologia do grafo via *message passing*; uma estrela plana com todos os filhos conectados apenas à raiz significa que, após um passo de agregação, cada filho recebe somente a informação da raiz — equivalente a uma transformação linear aplicada individualmente. A utilidade do *message passing* em grafos estrela é, portanto, mínima.

**L2 — Todos os filhos têm o mesmo embedding de texto:**
O nó raiz e todos os filhos compartilham o embedding BERT do *título do artigo*. Na realidade, cada usuário teria um embedding diferente (do seu tweet). Isso homogeniza artificialmente as features e faz com que a diferenciação entre filhos dependa *inteiramente* das features posicionais manuais.

**L3 — `pos` não é cronológica:**
A feature de posição relativa captura ordem de aparição no CSV, não sequência temporal de propagação. Pesquisas mostram que o padrão temporal de difusão (rapidez de propagação, pico de compartilhamento) é um sinal importante para detecção de fake news (Shu et al., 2020, Seção 2).

**L4 — Informação de usuário ausente:**
O UPFD utiliza histórico de tweets dos usuários (perfil, 200 posts recentes) como features de nó. Este script usa apenas o título do artigo para todos os nós, eliminando o sinal de *confirmation bias* do usuário que o UPFD foi desenhado para capturar.

### 6.3 Comparação com Estado da Arte

| Abordagem | Accuracy | F1 | Dataset | Fonte |
|-----------|----------|-----|---------|-------|
| GraphSAGE + BERT (UPFD) | 84.62% | 84.65% | PolitiFact (UPFD) | Dou et al. (2021), Tabela 3 |
| GCN (GCNFN + BERT) | 83.26% | 83.14% | PolitiFact (UPFD) | Dou et al. (2021), Tabela 3 |
| Este script (GNN a definir) | — | — | PolitiFact (FNN CSV, estrela) | — |

> 📖 **Fonte da comparação:** Dou et al. (2021), DOI: `10.1145/3404835.3462990`, Tabela 3.

**Nota importante sobre comparabilidade:** Os grafos do UPFD oficial foram construídos com árvores de propagação ricas (usuários reais, timestamps, relações de *follow*), enquanto este script usa estrelas planas derivadas apenas dos IDs de tweets no CSV. Os resultados obtidos com este script **não são diretamente comparáveis** com os do UPFD original, mesmo usando o mesmo conjunto PolitiFact.

### 6.4 Resposta Parcial à Questão do TCC

Este script posiciona o TCC em um cenário **mais desfavorável para as GNNs** do que o UPFD original: a topologia em estrela plana e a ausência de features de usuário removem as principais vantagens estruturais que justificam o uso de GNNs. Se as GNNs performarem bem mesmo com esses grafos simplificados, isso sugere que o sinal vem principalmente do texto (BERT) — o que favorece a hipótese de que **NLP é o fator determinante**, e a topologia do grafo contribui marginalmente. Se performarem mal, isso pode indicar que a qualidade do grafo (não as GNNs em si) é o gargalo. Em ambos os casos, este pipeline fornece evidência relevante para a questão central do TCC.

---

## 7. Análise de Código

### 7.1 Erros Identificados

```python
# ❌ Linha 76–77 — SSL verification desabilitado
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# ✅ Correção sugerida (para uso em produção):
ctx = ssl.create_default_context()
# (sem modificar ctx — usar verificação padrão)
# Justificativa: CERT_NONE torna o script vulnerável a ataques MITM. Para
# datasets públicos de pesquisa, o risco prático é baixo, mas deve ser
# documentado como escolha consciente, não como descuido.
```

```python
# ❌ Linhas 296–302 — N_max_global calculado sobre df_filtrado,
#    mas split_indices() usa o mesmo RANDOM_SEED=42 que será usado
#    depois em particionar_e_salvar(). Se o usuário modificar o seed
#    em uma das chamadas, os índices divergem silenciosamente.
train_idx, _, _ = split_indices(len(df_filtrado))
df_train = df_filtrado.iloc[train_idx]
N_max_global = int(max(...for s in df_train["tweet_ids"]))

# ✅ Sugestão: passar seed explicitamente ou documentar a dependência
# como invariante crítico no comentário.
```

```python
# ❌ Linhas 168–172 — assert com condição complexa pode falhar em casos
#    legítimos na variante "pos-grau" com num_nos==2 (1 raiz + 1 filho).
#    Se a raiz e o único filho têm o mesmo BERT embedding, is_root diferente
#    mas pos==0 (pois usa_pos=False), n_unique=2 e n_esperado=2 → OK.
#    Mas se num_nos==1 (deveria ser filtrado por min_tweets), n_unique=1
#    e a assert lança exceção com mensagem confusa.
assert n_unique >= min(n_esperado, 3) or num_nos < 2, (...)

# ✅ O filtro min_tweets=2 previne num_nos==1, então o assert está correto
#    na prática. Porém, a lógica seria mais clara como:
if num_nos >= 2:
    assert n_unique >= min(n_esperado, 3), (...)
```

### 7.2 Ineficiências

**I1 — Embedding idêntico para todos os nós do grafo:**
A função `construir_grafo()` recebe um único `embedding: list` (do título) e o replica para todos os nós (`x_bert` é o mesmo tensor para raiz e todos os filhos). Isso gera um tensor `x` de shape `[N+1, 771]` onde as primeiras 768 colunas são idênticas em todas as linhas. Do ponto de vista de compressão, seria mais eficiente armazenar o embedding uma vez e reconstruir na coleta (*batching*), mas o formato `Data` do PyG não suporta isso nativamente. Custo: $O(N \cdot 768)$ memória por grafo em vez de $O(768 + N \cdot 3)$.

**I2 — `df.iterrows()` com `tqdm`:**
`iterrows()` é notoriamente lento em pandas para DataFrames grandes. Para o tamanho do FakeNewsNet (~1000 artigos), o impacto é negligenciável, mas poderia ser substituído por `df.itertuples()` ou operação vetorizada se o dataset crescesse.

### 7.3 Boas Práticas Observadas

**B1 — Semente aleatória isolada:** O uso de `random.Random(RANDOM_SEED)` (objeto local) em vez de `random.seed()` (estado global) evita interferências com outros módulos que usam `random` — boa prática de engenharia de software.

**B2 — Cache de embeddings BERT:** A verificação de tamanho do cache (`len(cache) == len(df)`) previne uso silencioso de cache desatualizado após mudança no dataset.

**B3 — `N_max_global` apenas no treino:** Conforme documentado na Seção 2.4, esta é a prática correta para evitar *normalization leakage*.

**B4 — Verificação de reprocessamento:** A pergunta interativa antes de sobrescrever arquivos existentes previne perda acidental de experimentos.

**B5 — Asserção de dimensão:** `assert dim == 771` na linha 342 fornece *fail-fast* explícito se a dimensão mudar por qualquer refatoração futura.

---

## 8. Referências Bibliográficas

1. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, p. 171–188, 2020. DOI: `10.1089/big.2020.0062`. Disponível em: https://arxiv.org/abs/1809.01286

2. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. In: *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'21)*, 2021, p. 2051–2055. DOI: `10.1145/3404835.3462990`. Disponível em: https://arxiv.org/abs/2104.12259

3. REIMERS, N.; GUREVYCH, I. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. In: *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*, Hong Kong, 2019, p. 3982–3992. Disponível em: https://arxiv.org/abs/1908.10084 | ACL Anthology: `D19-1410`

4. DEVLIN, J.; CHANG, M.-W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. In: *Proceedings of NAACL-HLT 2019*, Minneapolis, 2019, p. 4171–4186. Disponível em: https://arxiv.org/abs/1810.04805 | ACL Anthology: `N19-1423`

5. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. In: *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*. Disponível em: https://arxiv.org/abs/1609.02907

6. SHCHUR, O.; MUMME, M.; BOJCHEVSKI, A.; GÜNNEMANN, S. **Pitfalls of Graph Neural Network Evaluation**. *Workshop on Relational Representation Learning, NeurIPS 2018*. Disponível em: https://arxiv.org/abs/1811.05868

7. SENTENCE-TRANSFORMERS. **paraphrase-multilingual-mpnet-base-v2 — Model Card**. HuggingFace Hub. Disponível em: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Grafo em Estrela (*Star Graph*) | Grafo onde um único nó central (raiz) está conectado a todos os outros nós (folhas), sem arestas entre folhas. $K_{1,N}$ na nomenclatura padrão. | Definição padrão de teoria de grafos |
| UPFD | *User Preference-aware Fake News Detection* — framework de detecção de fake news via GNN proposto por Dou et al. (2021); também nome do benchmark e formato de dataset associado. | Dou et al. (2021) |
| Mean Pooling | Operação que agrega vetores de tokens somando e dividindo pelo número de tokens: $\bar{\mathbf{e}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{h}_t$. | Reimers & Gurevych (2019), Seção 3 |
| COO (*Coordinate Format*) | Formato esparso para representar matrizes de adjacência: dois vetores (linha, coluna) listam as coordenadas das entradas não-nulas. Formato nativo do PyTorch Geometric. | Documentação PyG |
| Data Leakage / Normalization Leakage | Contaminação de dados de teste por informação do conjunto de treinamento durante pré-processamento (e.g., normalização com estatísticas globais). | Shchur et al. (2018) |
| Confirmation Bias | Tendência de usuários a compartilhar informações que confirmam suas crenças preexistentes — principal motivação teórica do UPFD para usar histórico do usuário como feature. | Dou et al. (2021), Seção 1 |
| `tweet_ids` | Campo do CSV FakeNewsNet contendo IDs de tweets relacionados ao artigo, separados por tabulação. A ordem é de aparição no CSV, não cronológica documentada. | Shu et al. (2020), Seção 5 |
| `grau_norm` | Feature do nó raiz: número de filhos do grafo dividido pelo máximo de filhos no conjunto de treino. Indica a "viralidade" relativa da notícia. | Definido neste script |
| `pos` | Feature do nó filho: posição relativa $i/N$ dentro dos filhos do grafo. Proxy de ordem de aparição no CSV. | Definido neste script |
