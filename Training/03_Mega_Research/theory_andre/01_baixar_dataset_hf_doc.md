# Documentação Técnica: 01_baixar_dataset_hf.py

## Metadados

- **Arquivo analisado:** `01_baixar_dataset_hf.py`
- **Caminho:** `03_Mega_Research/01_baixar_dataset_hf.py`
- **Data de análise:** 2026-04-24
- **Tipo de abordagem:** Ingestão de dados / pré-processamento (pipeline ETL)
- **Modelos principais:** Nenhum (script de coleta de dados)
- **Datasets utilizados:** `Zaras210/bluesky-fake-news-dataset` (HuggingFace Hub)
- **Contribuição para a questão central:** Fornece os dados brutos do Bluesky que alimentam o pipeline de GNN. A qualidade da rotulagem heurística aqui definida (`_label()`) impacta diretamente a validade de qualquer comparação GNN vs. NLP feita downstream — labels ruidosos distorcem métricas de avaliação de ambas as abordagens igualmente.

---

## 1. Visão Geral do Script

O script realiza o download seletivo do dataset `Zaras210/bluesky-fake-news-dataset` do HuggingFace Hub, evitando intencionalmente o uso de `datasets.load_dataset()` por causa de incompatibilidade de colunas entre os CSVs do repositório (schema heterogêneo). A estratégia é: baixar os arquivos individualmente via `hf_hub_download`, fazer o parse manual de cada formato, e produzir dois CSVs padronizados.

O problema resolvido é a conversão de dados heterogêneos do Bluesky — posts em formato JSONL comprimido e dados de propagação em TAR aninhado — para um esquema tabular fixo compatível com o pipeline de construção de grafos (`02_construir_grafos_bluesky.py`). O script lida com múltiplos formatos de arquivo (`.jsonl.gz`, `.csv.gz`, `.jsonl` puro) com hierarquia de prioridade e fallback.

A saída são dois CSVs: `posts_coletados.csv` com metadados e rótulos de cada post, e `reposts_coletados.csv` com as arestas do grafo de propagação (quem repostou quem). Um Reservoir Sampling limita os reposts a um máximo configurável (`MAX_REPOSTS_DEFAULT = 300_000`) sem carregar o arquivo inteiro em memória.

---

## 2. Arquitetura e Componentes Principais

### 2.1 HuggingFace Hub — Download Seletivo

**Descrição técnica:**
O script usa `huggingface_hub.hf_hub_download()` para baixar arquivos individuais do repositório de datasets do HuggingFace, com cache local automático em `Material/dados_bluesky/hf_cache`. O motivo para não usar `datasets.load_dataset()` está documentado no docstring: o dataset `Zaras210/bluesky-fake-news-dataset` tem colunas heterogêneas entre os CSVs, causando `DatasetGenerationCastError`.

**Embasamento — Dataset utilizado:**

> 📖 **Zaras210 (2023–2024)** — `Zaras210/bluesky-fake-news-dataset`
> *HuggingFace Datasets Hub*. Disponível em: https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset
> **Localização:** Dataset card e estrutura de arquivos do repositório
> **Relevância:** Fonte primária dos dados Bluesky utilizados no TCC. O dataset contém posts de múltiplos feeds temáticos do Bluesky (Science, News, Political Science, etc.) e dados de reposts. Tamanho: 2,54 GB. Sem paper acadêmico associado — é um dataset comunitário.
> **Limitação de citação (Tier 3):** Sem paper peer-reviewed. O dataset não possui publicação acadêmica formal. Deve ser citado como "dataset de repositório público" nas referências do TCC.

**Embasamento — Plataforma Bluesky:**

> 📖 **Kleppmann, M.; Frazee, P.; Gold, J.; Graber, J.; Holmgren, D.; Ivy, D.; Johnson, J.; Newbold, B.; Volpert, J. (2024)** — "Bluesky and the AT Protocol: Usable Decentralized Social Media"
> *Proceedings of the ACM CoNEXT-2024 Workshop on the Decentralization of the Internet (DIN '24)*, Los Angeles, CA, EUA. 9–12 dez. 2024.
> DOI: `10.1145/3694809.3700740` | arXiv: `2402.03239`
> **Localização:** Abstract e Seção de introdução
> **Relevância:** Descreve a arquitetura do Bluesky e do AT Protocol, incluindo o sistema de *labels* de moderação (`graphic-media`, `porn`, `spam`, `misleading`) que é usado como sinal de rotulagem no script. Reporta crescimento a >10 milhões de usuários em outubro de 2024.

**No código:**
> Linhas 78–93: `baixar_arquivo()` — wrapper sobre `hf_hub_download()` com parâmetros de cache e force-download.
> Linhas 64–73: `listar_arquivos_repo()` — usa `list_repo_files()` para obter a lista de arquivos antes de decidir o que baixar.

---

### 2.2 Heurística de Rotulagem (`_label()`)

**Descrição técnica:**
A função `_label()` atribui rótulo binário (`1=fake`, `0=real`) combinando dois sinais:
1. **Labels da API AT Protocol** — campos estruturados como `"misleading"`, `"spam"`, `"!hide"`, `"graphic-media"`, `"porn"` são interpretados como fake/problemático (`label=1`)
2. **Busca por keywords** — lista de 13 termos (`"fake"`, `"conspiracy"`, `"hoax"`, etc.) no texto do post

**Fundamento conceitual (Weak Supervision):**

Esta abordagem implementa *distant supervision* / *weak labeling* — uso de fontes de sinal imperfeitas como proxy de rótulos de treino, em ausência de anotação humana especializada.

> 📖 **Ratner, A.; De Sa, C.; Wu, S.; Selsam, D.; Ré, C. (2016)** — "Data Programming: Creating Large Training Sets, Quickly"
> *Advances in Neural Information Processing Systems (NeurIPS) 29*, Barcelona, 2016.
> arXiv: `1605.07723`
> **Localização:** Seção 1 (Introduction), Seção 3 (The Data Programming Paradigm)
> **Relevância:** Fundamenta o paradigma de usar *labeling functions* imperfeitas combinadas para gerar rótulos de treino em larga escala. A função `_label()` deste script é exatamente uma *labeling function* (LF) no vocabulário de Ratner et al. — com a diferença que usa apenas uma LF combinada em vez de um conjunto com modelo de denoising.

> 📖 **Ratner, A.; Bach, S.H.; Ehrenberg, H.; Fries, J.; Wu, S.; Ré, C. (2017)** — "Snorkel: Rapid Training Data Creation with Weak Supervision"
> *Proceedings of the VLDB Endowment*, v. 11, n. 3, pp. 269–282, 2018.
> DOI: `10.14778/3157794.3157797` | arXiv: `1711.10160`
> **Localização:** Seção 2 (The Snorkel System), Seção 3 (Generative Model)
> **Relevância:** Documenta que *labeling functions* baseadas em keywords têm altas taxas de erro em dados de misinformação — o denoising probabilístico do Snorkel existe precisamente porque cada LF isolada é insuficiente.

**Problemas identificados (ver Seção 7.1):**

A heurística confunde categorias de conteúdo problemático com *fake news* especificamente:
- `"graphic-media"`, `"porn"`, `"spam"` → `label=1` (fake): conteúdo ofensivo ≠ desinformação
- Keywords como `"fake"`, `"conspiracy"` no texto → `label=1`: posts *desmentindo* fake news ("this conspiracy theory is false...") serão classificados como fake
- `"misleading"` do AT Protocol pode incluir conteúdo satírico, hipérbole ou opinativo

**No código:**
> Linhas 45–59: definição de `KEYWORDS_FAKE` e `_label()`.

---

### 2.3 Reservoir Sampling

**Descrição técnica:**
O Algoritmo R de Vitter é implementado na extração de reposts para amostrar uniformemente até `max_linhas` registros de um stream de tamanho total desconhecido, sem carregar o arquivo inteiro em memória. O arquivo `graphs.tar.gz` pode conter dezenas de milhões de linhas; o Reservoir Sampling garante que qualquer subconjunto de `max_linhas` linhas tenha probabilidade igual de ser selecionado.

**Fundamento matemático:**

Para um stream de elementos $x_1, x_2, \dots, x_N$ (com $N$ desconhecido) e reservoir de tamanho $k$:

**Inicialização:** Para $i \leq k$: inserir $x_i$ diretamente no reservoir.

**Fase de substituição:** Para $i > k$:
$$j \sim \text{Uniform}(0, i)$$
$$\text{Se } j < k: \text{ reservoir}[j] \leftarrow x_i$$

**Invariante:** Após processar $i$ elementos, cada um dos $i$ elementos tem probabilidade $k/i$ de estar no reservoir.

**Prova:** Por indução. Após processar $i$ elementos, cada $x_j$ ($j \leq i$) tem prob. $k/i$ no reservoir. Para $x_{i+1}$: prob. de entrar = $k/(i+1)$. Para cada $x_j$ já no reservoir: prob. de sobreviver = $1 - \frac{k}{i+1} \cdot \frac{1}{k} = \frac{i}{i+1}$. Prob. de estar no reservoir após $i+1$ passos = $\frac{k}{i} \cdot \frac{i}{i+1} = \frac{k}{i+1}$. $\square$

**Complexidade:** $O(N)$ tempo, $O(k)$ memória.

**Embasamento acadêmico:**

> 📖 **Vitter, J. S. (1985)** — "Random Sampling with a Reservoir"
> *ACM Transactions on Mathematical Software (TOMS)*, v. 11, n. 1, pp. 37–57.
> DOI: `10.1145/3147.3165`
> **Localização:** Seção 2 (Algorithm R), Teorema 1 (prova de uniformidade)
> **Relevância:** Paper original que define o Algoritmo R implementado nas linhas 283–289 do script. A implementação é uma transcrição direta do algoritmo descrito na Seção 2.

**No código:**
> Linhas 264–289: loop sobre as linhas do CSV de reposts.
> Linha 283: `if i < max_linhas: reposts.append(row)` — fase de inicialização.
> Linhas 285–289: `j = random.randint(0, i); if j < max_linhas: reposts[j] = row` — fase de substituição.

**Divergência em relação ao Algoritmo R original:**
O Vitter usa `randint(0, i)` inclusive em ambos os lados (intervalo $[0, i]$, tamanho $i+1$), o que é consistente com `random.randint(0, i)` do Python (que também é inclusivo). A implementação está **correta**.

---

### 2.4 Extração de TAR Aninhado

**Descrição técnica:**
O arquivo `graphs.tar.gz` é lido como um `gzip` wrappando um `tar`, acessando o membro interno `graphs/reposts.csv` sem extração para disco. A cadeia é: `gzip.open(local, "rb")` → `tarfile.open(fileobj=gz)` → `tar.extractfile(membro)` → iteração linha a linha.

**No código:**
> Linhas 247–248: abertura encadeada `gzip` + `tarfile`.
> Linhas 250–254: busca do membro `reposts.csv` por nome dentro do tar.
> Linhas 264–289: leitura streaming + reservoir sampling.

---

## 3. Pipeline de Dados e Pré-processamento

### 3.1 Hierarquia de Formatos de Posts

O script prioriza `.jsonl.gz` (formato rico com todos os metadados), faz fallback para `.csv.gz` (formato degradado sem texto real — ver limitação L3), e suporta `.jsonl` puro como terceiro nível.

| Formato | Texto extraído | Label inferível | Metadados |
|---------|----------------|-----------------|-----------|
| `.jsonl.gz` | ✓ (campo `text`/`texto`) | ✓ (keywords + API labels) | ✓ completos |
| `.csv.gz` | ✗ (placeholder `[post bluesky ID]`) | ✗ (sempre 0) | Parcial |
| `.jsonl` | ✓ (campo `text`) | ✓ (keywords) | Parcial |

**No código:**
> Linhas 170–172: filtro de `feed_files` por sufixo.
> Linhas 189–214: despacho por tipo de arquivo.
> Linhas 131–162: `_parse_feed_csv_gz()` — texto placeholder.

---

### 3.2 Normalização de Schema (`_parse_jsonl_gz`)

**Descrição técnica:**
O parser JSONL suporta dois esquemas de campo por aliases: `"text"` / `"texto"`, `"post_id"` / `"id"`, `"user_id"` / `"author"`, `"date"` / `"created_at"`. Isso indica que os arquivos do dataset têm nomes de campo inconsistentes entre feeds.

**No código:**
> Linhas 111–118: `obj.get("text", obj.get("texto", ""))` — pattern de alias duplo.

---

### 3.3 Estrutura do CSV de Reposts

O CSV `graphs/reposts.csv` tem formato: `post_id, reposter_id, data_yyyymmdd` (sem cabeçalho). O script mapeia para:

```
source  = reposter_id   (quem repostou)
target  = post_id       (o post original)
```

**Atenção semântica:** Em terminologia de grafo de propagação, a notícia *flui de* `post_id` *para* `reposter_id` — a propagação vai do post original ao reposter, não o contrário. A convenção `source=reposter, target=post_id` inverte a direção de propagação em relação ao significado usual de `(source → target)`. O script downstream (`02_construir_grafos_bluesky.py`) deve ser verificado para confirmar como esses campos são interpretados.

**No código:**
> Linhas 275–281: mapeamento de campos do CSV para o dict de repost.

---

## 4. Construção do Grafo (não aplicável neste script)

Este script não constrói grafos — produz os dados tabulares que alimentam `02_construir_grafos_bluesky.py`. A estrutura do grafo é definida no script seguinte.

---

## 5. Métricas de Avaliação

Este script reporta **métricas de diagnóstico** da extração:

| Métrica | Fórmula | Propósito |
|---------|---------|-----------|
| Taxa de fake | $N_{\text{fake}} / N_{\text{total}}$ | Detectar desbalanceamento induzido pela heurística |
| Posts por feed | $\text{contagem}(\text{feed}_i)$ | Identificar dominância de um feed específico |

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos Fortes desta Abordagem

**Novidade do dataset:** O Bluesky é uma plataforma descentralizada com modelo de moderação diferente do Twitter/X, onde os *labels* de conteúdo são públicos via AT Protocol. Isso oferece potencial de sinais de supervisão mais transparentes do que o FakeNewsNet, que usa fact-checkers externos.

**Escalabilidade:** O Reservoir Sampling permite processar arquivos de dezenas de milhões de linhas com memória $O(k)$ constante — viabilizando uso de datasets grandes sem infraestrutura especial.

**Flexibilidade de formato:** O fallback entre JSONL.gz → CSV.gz → JSONL garante robustez contra mudanças no schema do dataset.

### 6.2 Limitações Identificadas

**L1 — Rotulagem heurística com alta taxa de erro esperada:**
A função `_label()` combina dois sinais fracos sem calibração estatística. Keywords como `"fake"` e `"conspiracy"` têm alta prevalência em posts que *desmentem* desinformação (e.g., "this fake claim about vaccines is false"). Sem gold labels para validação, a taxa de erro real da heurística é desconhecida. Ratner et al. (2016, 2017) documentam que *labeling functions* individuais tipicamente têm precisões entre 60–80% — menor que o que modelos de classificação bem treinados atingem.

**L2 — Confusão entre categorias de conteúdo problemático:**
AT Protocol labels como `"spam"`, `"porn"`, `"graphic-media"` são mapeados para `label=1 (fake)`. Esses conteúdos não são fake news — são conteúdo impróprio. Misturar categorias compromete o conceito de `label=1` no dataset resultante.

**L3 — Posts CSV sem texto real:**
Para feeds em formato `.csv.gz`, o campo `"texto"` é preenchido com `f"[post bluesky {post_id}]"`. Esses posts terão embeddings BERT inúteis no pipeline downstream e `label=0` sempre (a heurística de keyword não encontrará nada). Se houver muitos feeds em CSV, a qualidade do dataset será comprometida silenciosamente.

**L4 — `random.seed()` modifica estado global:**
O uso de `random.seed(RANDOM_SEED)` (linha 244) em vez de `random.Random(RANDOM_SEED)` modifica o estado global do RNG do Python. Isso quebra reprodutibilidade se outros módulos importarem `random` no mesmo processo — inconsistência com a abordagem do script `00` que usa objeto `Random` local.

**L5 — Sem deduplicação por `post_id`:**
`todos_posts.extend(novos)` não verifica se um `post_id` já foi adicionado por outro feed. Posts presentes em múltiplos feeds (e.g., um post de ciência republicado em feed de notícias) serão duplicados, inflando o dataset e potencialmente vazando informação entre splits.

**L6 — `source`/`target` potencialmente invertidos:**
Ver Seção 3.3. Dependendo de como `02_construir_grafos_bluesky.py` interpreta esses campos, as arestas do grafo podem apontar na direção oposta à propagação real da notícia.

### 6.3 Comparação com o estado da arte

Este script não implementa modelos de classificação, portanto não há métricas de comparação diretas. A relevância para a questão central está na **qualidade dos dados** que alimentam os experimentos:

| Aspecto | Este dataset (Bluesky) | FakeNewsNet PolitiFact | Diferença |
|---------|------------------------|------------------------|-----------|
| Labels | Heurística (noisy) | Fact-checkers externos (gold) | FNN >> Bluesky |
| Texto | Tweets/posts curtos | Artigos completos + tweets | FNN mais rico |
| Propagação | Reposts diretos | Retweets com timestamps | Similar |
| Tamanho | >> 300k reposts | ~1000 artigos | Bluesky maior |
| Plataforma | Bluesky (nova) | Twitter (descontinuado) | Bluesky mais atual |

> 📖 **Fonte da comparação FakeNewsNet:** Shu et al. (2020), DOI: `10.1089/big.2020.0062`, Tabela 2.

### 6.4 Resposta Parcial à Questão do TCC

A qualidade dos labels gerados por `_label()` é a principal ameaça à validade dos experimentos com o dataset Bluesky. Se os rótulos têm alta taxa de ruído, as GNNs e os modelos NLP serão avaliados sobre um alvo incorreto — e qualquer diferença de desempenho entre as abordagens pode refletir diferença em *robustez a ruído* em vez de diferença em capacidade de detectar fake news. Isso deve ser reportado explicitamente como limitação na Seção de Ameaças à Validade do TCC.

---

## 7. Análise de Código

### 7.1 Erros Identificados

```python
# ❌ Linhas 52–59 — Confusão de categorias em _label()
for lbl in labels_api:
    val = lbl.get("val", "") if isinstance(lbl, dict) else str(lbl)
    if val in ("graphic-media", "porn", "spam", "!hide", "misleading"):
        return 1   # Rotula conteúdo adulto/spam como "fake news"

# ✅ Sugestão: separar "misleading" (relevante para fake news) das
#    demais categorias, ou criar label multi-classe:
FAKE_NEWS_LABELS = {"misleading"}
SPAM_LABELS = {"graphic-media", "porn", "spam", "!hide"}
# label=1 apenas para "misleading"; criar campo separado para spam
# Justificativa: spam e conteúdo adulto não são desinformação —
# misturá-los distorce a tarefa de classificação.
```

```python
# ❌ Linha 57–58 — Keywords no texto causam falsos positivos
t = texto.lower()
return 1 if any(k in t for k in KEYWORDS_FAKE) else 0
# Exemplo: "This conspiracy theory is fake — here's the truth"
# -> label=1 (ERRADO — este post desmete fake news)

# ✅ Sugestão: usar contexto de negação ou limitar a hashtags:
HASHTAGS_FAKE = {"#fakenews", "#misinformation", "#disinformation"}
return 1 if any(h in t for h in HASHTAGS_FAKE) else 0
# Justificativa: hashtags de conteúdo (não de debunking) são sinais
# mais precisos do que palavras isoladas.
```

```python
# ❌ Linha 244 — random.seed() modifica estado global
random.seed(RANDOM_SEED)

# ✅ Correção (consistente com script 00):
rng = random.Random(RANDOM_SEED)
# ...
j = rng.randint(0, i)     # linha 287
# Justificativa: isola o RNG do reservoir sampling de outros módulos
# que possam usar random no mesmo processo.
```

```python
# ❌ Sem deduplicação após extend (linhas 220, 280)
todos_posts.extend(novos)

# ✅ Sugestão:
seen_ids = set()
for p in novos:
    if p["post_id"] and p["post_id"] not in seen_ids:
        seen_ids.add(p["post_id"])
        todos_posts.append(p)
# Justificativa: posts presentes em múltiplos feeds serão duplicados,
# podendo causar data leakage entre train/val/test splits.
```

### 7.2 Ineficiências

**I1 — Carga de lista de arquivos completa antes do download:**
`listar_arquivos_repo()` retorna todos os arquivos do repositório para depois filtrar por `feed_files = [f for f in arquivos_repo if ...]`. Para repositórios grandes, isso é ineficiente. Alternativa: usar `list_repo_files(..., path_in_repo="feed_posts/")` se suportado.

**I2 — `feeds: dict = {}` com `feeds.get()` manual:**
`relatorio()` usa dict com `.get()` manual para contagem. `collections.Counter(p.get("feed", "??") for p in posts)` seria mais idiomático e legível.

### 7.3 Boas Práticas Observadas

**B1 — Lazy imports de `huggingface_hub`:**
`from huggingface_hub import hf_hub_download` dentro das funções evita crash na importação se a biblioteca não estiver instalada — o erro é tratado explicitamente no `main()` com mensagem clara.

**B2 — `extrasaction="ignore"` no `DictWriter`:**
Previne `ValueError` se algum dict tiver campos extras não listados em `campos` — defensivo e correto.

**B3 — Streaming do TAR sem extração:**
`tar.extractfile(membro)` lê o membro diretamente da memória sem extrair para disco — essencial dado o tamanho do arquivo (potencialmente >1 GB descomprimido).

**B4 — `--dry-run` para inspeção prévia:**
Opção de listar arquivos sem baixar é boa prática de UX para scripts de ingestão de dados.

**B5 — Guard de reprocessamento:**
Verificação `if OUTPUT_POSTS.exists() and OUTPUT_POSTS.stat().st_size > 1000` previne reprocessamento acidental. O threshold de 1000 bytes é uma heurística razoável para detectar arquivo não-vazio.

---

## 8. Referências Bibliográficas

1. KLEPPMANN, M. et al. **Bluesky and the AT Protocol: Usable Decentralized Social Media**. In: *Proceedings of the ACM CoNEXT-2024 Workshop on the Decentralization of the Internet (DIN '24)*, Los Angeles, EUA, 2024. DOI: `10.1145/3694809.3700740`. Disponível em: https://arxiv.org/abs/2402.03239

2. VITTER, J. S. **Random Sampling with a Reservoir**. *ACM Transactions on Mathematical Software (TOMS)*, v. 11, n. 1, pp. 37–57, 1985. DOI: `10.1145/3147.3165`

3. RATNER, A.; DE SA, C.; WU, S.; SELSAM, D.; RÉ, C. **Data Programming: Creating Large Training Sets, Quickly**. In: *Advances in Neural Information Processing Systems (NeurIPS) 29*, Barcelona, 2016. Disponível em: https://arxiv.org/abs/1605.07723

4. RATNER, A.; BACH, S. H.; EHRENBERG, H.; FRIES, J.; WU, S.; RÉ, C. **Snorkel: Rapid Training Data Creation with Weak Supervision**. *Proceedings of the VLDB Endowment*, v. 11, n. 3, pp. 269–282, 2018. DOI: `10.14778/3157794.3157797`. Disponível em: https://arxiv.org/abs/1711.10160

5. JEONG, U.; JIANG, B.; TAN, Z.; BERNARD, H. R.; LIU, H. **BlueTempNet: A Temporal Multi-network Dataset of Social Interactions in Bluesky Social**. *IEEE Data Descriptions*, 2024. Disponível em: https://arxiv.org/abs/2407.17451

6. SHU, K. et al. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`

7. ZARAS210. **bluesky-fake-news-dataset** [Dataset]. HuggingFace Hub, 2023–2024. Disponível em: https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| AT Protocol | *Authenticated Transfer Protocol* — protocolo descentralizado para redes sociais, base do Bluesky. Define o sistema de *labels* de moderação pública. | Kleppmann et al. (2024) |
| Reservoir Sampling | Algoritmo que seleciona uniformemente $k$ amostras de um stream de tamanho desconhecido usando $O(k)$ memória. | Vitter (1985) |
| Weak Supervision | Paradigma de treinamento que usa fontes de sinal imperfeitas (keywords, heurísticas, regras) como proxy de rótulos de treino na ausência de anotação humana. | Ratner et al. (2016) |
| Labeling Function (LF) | Função que atribui rótulos a instâncias com base em regras heurísticas. Pode ter precisão imperfeita e cobertura parcial. | Ratner et al. (2017), Seção 2 |
| Noisy Labels | Rótulos de treinamento com taxa de erro superior a zero, gerados por fontes imperfeitas (LFs, distant supervision, crowdsourcing). | Ratner et al. (2016) |
| Firehose | Stream em tempo real de todos os posts públicos do Bluesky, disponível via AT Protocol. | Kleppmann et al. (2024) |
| JSONL | *JSON Lines* — formato de arquivo onde cada linha é um objeto JSON válido e independente. Facilita processamento streaming linha a linha. | — |
| TAR aninhado | Arquivo `.tar.gz` = arquivo `.tar` comprimido com gzip. O script lê como `gzip.open` → `tarfile.open` encadeados. | — |
