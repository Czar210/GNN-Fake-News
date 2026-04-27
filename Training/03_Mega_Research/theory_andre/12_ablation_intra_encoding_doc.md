# Documentação Técnica: 12_ablation_intra_encoding.py

## Metadados

- **Arquivo analisado:** `12_ablation_intra_encoding.py`
- **Caminho:** `Training/03_Mega_Research/12_ablation_intra_encoding.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Ablação controlada de features posicionais — três variantes de encoding (`posfull` / `posmin` / `posgrau`) sobre GCN, com t-test pareado por fold contra baseline textual (LogReg-BERT)
- **Modelo principal:** `GCNClassifier` (Kipf & Welling, 2017) — única arquitetura, três entradas
- **Datasets utilizados:** FakeNewsNet PolitiFact local (variantes `posfull`, `posmin`, `posgrau`); folds compartilhados em `data/folds_fnn.pt`
- **Contribuição para a questão central:** Este script responde uma pergunta de causalidade interna ao TCC: **dentro das três dimensões posicionais inseridas para corrigir o "Erro 1" (features nodais idênticas), qual delas carrega o sinal preditivo?** Se uma única dimensão (popularidade ou ordem) já basta para igualar/superar o baseline textual, então a "estrutura" aprendida pela GNN é trivial — um proxy escalar disfarçado de propagação. Isso desidrata o claim de que GNNs aprendem "topologia complexa" e reforça o núcleo do TCC: o ganho do grafo sobre NLP é explicável por confounds simples.

---

## 1. Visão Geral do Script

O script `12_ablation_intra_encoding.py` executa um **estudo de ablação intra-encoding** sobre as três dimensões posicionais introduzidas no Script 00 (`is_root`, `grau_norm`, `pos`) para evitar que a média global das features BERT colapse a representação dos grafos em estrela plana (Erro 1, documentado na auditoria do TCC). A pergunta empírica é: das três dimensões, qual carrega o sinal? O experimento isola contribuições treinando o **mesmo** GCN, nos **mesmos** folds estratificados, com três variantes do dataset que diferem apenas nas dimensões posicionais ativas:

- **`posfull`** — todas as três dimensões ativas: `[is_root, grau_norm, pos]`
- **`posmin`** — `grau_norm` zerado: `[is_root, 0, pos]`
- **`posgrau`** — `pos` zerado: `[is_root, grau_norm, 0]`

Em todas as variantes, o vetor BERT(título) de 768 dimensões é idêntico para todos os nós do mesmo grafo — esta é a condição que originou o Erro 1. As três variantes diferem apenas nos 3 últimos valores de cada vetor de 771 dimensões.

Para cada uma das três variantes, o script treina K=10 GCNs (um por fold), avalia em F1-macro, e aplica `scipy.stats.ttest_rel` contra os F1 do baseline textual (LogReg-BERT, Fase 2A.1) lidos de `baseline_textual/resultados.csv`. Aplica então **dois gates de decisão**:

1. **Gate de Posição:** F1(`posmin`) > F1(baseline) com p<0.05 — atribui ganho à posição ordinal
2. **Gate de Confound:** F1(`posgrau`) − F1(`posmin`) > 0.01 — sinal dominante é grau (popularidade)

A saída inclui `tabela.csv` (F1 por fold, por variante) e `relatorio.txt` (médias, p-valores, decisão dos gates).

---

## 2. Arquitetura e Componentes Principais

### 2.1 Estudo de Ablação como Método Científico

**Descrição técnica:**
Um estudo de ablação remove sistematicamente componentes de um sistema (módulos, features, camadas) e mede a queda de desempenho atribuível a cada remoção. A metáfora vem da neurociência: lesionar regiões cerebrais para isolar funções (Lashley, 1929 *apud* Meyes et al., 2019). Em redes neurais, o objetivo é converter intuição arquitetural em afirmações causais sobre quais componentes carregam capacidade preditiva.

**Fundamento metodológico:**
Para uma função preditiva $f_\theta(x)$ avaliada por métrica $M$, o impacto da componente $c$ é estimado por:

$$\Delta_c = M(f_\theta(x; c \text{ ativo})) - M(f_\theta(x; c \text{ ablado}))$$

onde $f_\theta$ é o modelo treinado, $x$ é o input, e $c$ é a componente sob investigação. Se $\Delta_c \approx 0$, a componente é redundante para $M$; se $\Delta_c \gg 0$, ela é causalmente necessária.

A variável central neste script é a **dimensão posicional** $c \in \{\text{grau\_norm}, \text{pos}\}$ — `is_root` é mantido em todas as variantes porque sem ele a raiz não se distingue dos filhos.

**Embasamento acadêmico:**

> 📖 **Meyes, R.; Lu, M.; de Puiseau, C. W.; Meisen, T. (2019)** — "Ablation Studies in Artificial Neural Networks"
> *arXiv preprint*, arXiv: `1901.08644v1`
> **Localização:** Seção 1 (Introduction) — define ablação como "sequential removal/inactivation of model components to study contribution"; Seção 4 (Methods) — protocolo de ablação por feature/módulo; Seção 5 (Discussion) — argumenta que ablação só permite afirmações causais quando a remoção é **isolada** (uma componente por vez) e o restante do pipeline é fixo
> **Relevância:** Fundamenta diretamente o desenho do script — três variantes diferindo em **uma** dimensão por vez, com tudo o mais (modelo, folds, hiperparâmetros, semente) constante. Sem essa isolação, a interpretação dos $\Delta$ seria confundida por variância de treinamento.

**No código:**
> Linhas 50, 173–198: o loop externo itera sobre `VARIANTES = ["posfull", "posmin", "posgrau"]`, e para cada uma carrega o dataset correspondente, mantendo `GCNClassifier(num_feats, 2, seed=fold_idx)` idêntico — apenas `num_feats` muda implicitamente (sempre 771, mas com colunas zeradas conforme a variante).

---

### 2.2 Encoding Posicional como Solução para Identidade Nodal

**Descrição técnica:**
Em uma estrela plana com BERT(título) replicado em todos os nós, sem dimensões posicionais cada nó tem feature idêntica $x_i = \text{BERT}(\text{titulo})$ para $i = 0, \ldots, N$. Após uma camada GCN com normalização simétrica, a representação de cada nó é

$$h_i^{(1)} = \sigma\!\left( \sum_{j \in \mathcal{N}(i)} \tfrac{1}{\sqrt{\deg(i)\,\deg(j)}}\, W^{(0)} x_j \right)$$

Como todos os $x_j$ são iguais, a soma colapsa em um múltiplo escalar de $W^{(0)} x$, fazendo com que a GCN se torne efetivamente uma **transformação linear da média global do BERT** — precisamente o Erro 1 detectado na auditoria do TCC. A solução é injetar features posicionais que **quebram a simetria** entre nós.

**Fundamento matemático — três dimensões posicionais:**

Para um grafo com raiz na posição 0 e $N$ filhos, define-se um vetor posicional $p_i \in \mathbb{R}^3$ por nó:

$$p_i = \begin{bmatrix} \mathbb{1}[i=0] \\ \frac{N}{N_{\max}^{\text{global}}} \cdot \mathbb{1}[i=0] \\ \frac{i}{N} \cdot \mathbb{1}[i \geq 1] \end{bmatrix}$$

Variável a variável:
- $\mathbb{1}[i=0]$ — indicador binário de raiz: 1 se nó é a raiz (artigo), 0 se é filho (tweet/usuário)
- $N$ — número de filhos do grafo
- $N_{\max}^{\text{global}}$ — número máximo de filhos observado no **conjunto de treino** (cap em `MAX_NOS-1=99`); divisão evita que a feature exploda e padroniza a escala em $[0,1]$
- $\frac{i}{N}$ — posição ordinal normalizada do filho $i \in \{1, \ldots, N\}$, valor em $(0, 1]$
- A multiplicação por indicadores zera a feature de grau nos filhos e a feature posicional na raiz, garantindo que as três dimensões codifiquem informações **disjuntas**

A feature final do nó é a concatenação $x_i = [\text{BERT}(\text{titulo}) \,||\, p_i] \in \mathbb{R}^{771}$.

**Embasamento acadêmico — origem do conceito de positional encoding:**

> 📖 **Vaswani, A.; Shazeer, N.; Parmar, N. et al. (2017)** — "Attention is All You Need"
> *NeurIPS 2017* | arXiv: `1706.03762`
> **Localização:** Seção 3.5 ("Positional Encoding"), Equações (eq.) imediatamente após "we add 'positional encodings' to the input embeddings". Forma sinusoidal:
> $$\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
> $$\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
> **Relevância:** Vaswani argumenta no início da Seção 3.5 que "since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence". A motivação é exatamente análoga ao caso da estrela plana: GCN com features idênticas é **agnóstico a ordem** (como o Transformer puro), e injetar um encoding posicional é a forma canônica de quebrar essa simetria. O script 12 testa se a forma simplificada (3 dims escalares em vez do encoding sinusoidal de $d_{\text{model}}$ dimensões) é suficiente para o domínio.

**Embasamento acadêmico — positional encoding em GNNs:**

> 📖 **Dwivedi, V. P.; Luu, A. T.; Laurent, T.; Bengio, Y.; Bresson, X. (2022)** — "Graph Neural Networks with Learnable Structural and Positional Representations"
> *ICLR 2022* | arXiv: `2110.07875`
> **Localização:** Seção 1 (Introduction), parágrafos sobre "MP-GNNs lack positional information"; Seção 3 (Decoupling Structural and Positional Representations), Eq. (1)–(4) — propõem manter representação estrutural $h$ e posicional $p$ separadas e aprendíveis; Seção 4 (Experiments) — mostram que adicionar PE Laplaciano melhora consistentemente GCN/GAT/GIN em ZINC, OGB-MOL, etc.
> **Relevância:** Esta é a referência Tier-1 que justifica formalmente a necessidade de PE em GNNs. Dwivedi et al. estabelecem que **GNNs com message-passing padrão não distinguem nós em posições estruturalmente equivalentes** quando suas features de input são idênticas — precisamente a patologia da estrela plana. O script 12 implementa uma forma minimalista (3 dims) do que Dwivedi formaliza com encoding Laplaciano, e a ablação testa quais das 3 dims são necessárias.

**No código:**
> Construção das features: `00_construir_grafos_fakenewsnet.py`, linhas 144–153 — implementação direta da fórmula acima. As máscaras `usa_grau` e `usa_pos` (linhas 140–141) controlam quais dimensões são zeradas em cada variante:
> - `posfull`: `usa_grau=True, usa_pos=True` → todas as 3 dims ativas
> - `posmin`: `usa_grau=False, usa_pos=True` → grau zerado em todos os nós
> - `posgrau`: `usa_grau=True, usa_pos=False` → pos zerado em todos os filhos
>
> Linhas 167–172 (script 00): assert de sanidade verifica que ao menos os nós-raiz têm vetor único, prevenindo o Erro 1 silencioso na variante `posgrau` (onde os filhos voltam a ser idênticos por design).

---

### 2.3 As Três Variantes — Lógica do Desenho Experimental

**Descrição técnica:**
O desenho com três variantes (em vez de dois ou quatro) é parcimonioso: testa cada dimensão "sozinha" mantendo `is_root` (sem o qual a raiz colapsa nos filhos), e usa `posfull` como controle superior.

| Variante | `is_root` | `grau_norm` | `pos` | Hipótese testada |
|----------|-----------|-------------|-------|-------------------|
| `posfull` | ✓ | ✓ | ✓ | Limite superior — todas as informações disponíveis |
| `posmin`  | ✓ | ✗ (zerado) | ✓ | Sinal vem da **ordem** dos tweets (proxy temporal) |
| `posgrau` | ✓ | ✓ | ✗ (zerado) | Sinal vem da **popularidade** (número de tweets — confound clássico) |

A ablação não inclui um hipotético `posnone` (só `is_root`) porque, sem `pos` nem `grau`, todos os filhos voltam a ser idênticos e o Erro 1 reaparece — degeneraria o experimento ao caso pré-correção, já estudado em scripts anteriores.

**Por que a comparação contra o baseline textual?**
O baseline LogReg-BERT (Script 10) é o **teto textual**: F1 obtido apenas com BERT(título) da raiz, sem qualquer estrutura. Se uma variante ablada da GCN não bate o baseline, o GCN não está extraindo nada além do que regressão logística já captura no texto — a "estrutura" daquela variante é vazia. Esta lógica segue a recomendação de Errica et al. (2020) de sempre incluir baselines structureless em ablações de GNN.

**Embasamento acadêmico:**

> 📖 **Errica, F.; Podda, M.; Bacciu, D.; Micheli, A. (2020)** — "A Fair Comparison of Graph Neural Networks for Graph Classification"
> *ICLR 2020* | arXiv: `1912.09893`
> **Localização:** Seção 1 (Introduction) — argumentam que "structure-agnostic baselines (e.g., MLP on bag of node features) frequently match or exceed GNNs"; Seção 4.2 (Baseline) — definem o "Structure-Agnostic Baseline" usando apenas média de features nodais; Tabela 2 — mostram que em 5 de 9 datasets clássicos (NCI1, PROTEINS, D&D, ENZYMES, IMDB-B) o baseline structureless está dentro de 1σ do melhor GNN
> **Relevância:** Justificativa teórica para o gate `F1(posmin) > F1(baseline_textual)`. Se a GCN com a única dimensão `pos` ativa não supera regressão logística sobre BERT(raiz), então mesmo a "ordem dos tweets" não traz sinal além do texto — confirmando empiricamente o que Errica avisa em geral. Esse é precisamente o critério que define o sucesso/falha do Gate de Posição.

**No código:**
> Linhas 211–230: cálculo das médias e p-valores; linhas 234–243 (Gate de Posição) e 245–254 (Gate de Confound) implementam diretamente os critérios de Errica adaptados ao contexto do TCC.

---

### 2.4 GCNClassifier — Arquitetura Fixa

**Descrição técnica:**
O script utiliza apenas GCN — não GAT nem SAGE — porque o objetivo é isolar o efeito das features de input, não do modelo. GCN é o aggregator linear mais simples, com semântica determinística (média ponderada por grau), o que torna o efeito de zerar uma dimensão de input mais previsível e interpretável.

**Fundamento matemático:**
A camada GCN executa:

$$H^{(\ell+1)} = \sigma\!\left( \tilde{D}^{-\frac{1}{2}}\, \tilde{A}\, \tilde{D}^{-\frac{1}{2}}\, H^{(\ell)} W^{(\ell)} \right)$$

Variáveis:
- $H^{(\ell)} \in \mathbb{R}^{N \times d_\ell}$ — matriz de features de todos os $N$ nós na camada $\ell$
- $\tilde{A} = A + I_N$ — matriz de adjacência com self-loops adicionados
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ — matriz diagonal de graus de $\tilde{A}$
- $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$ — matriz aprendida de pesos
- $\sigma$ — função de ativação não-linear (ReLU)

A normalização simétrica $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ é a aproximação espectral derivada de polinômios de Chebyshev de grau 1.

**Embasamento acadêmico:**

> 📖 **Kipf, T. N.; Welling, M. (2017)** — "Semi-Supervised Classification with Graph Convolutional Networks"
> *ICLR 2017* | arXiv: `1609.02907`
> **Localização:** Seção 2 (Fast Approximate Convolutions on Graphs), Eq. (8) — derivação da regra de propagação acima a partir da convolução espectral; Seção 3.1 (Example) — implementação canônica em duas camadas para classificação de nós
> **Relevância:** Fundamento da arquitetura. Em estrela plana, $\tilde{A}$ tem a estrutura de uma estrela com self-loops; após normalização simétrica, a raiz com $N$ filhos pondera cada vizinho por $1/\sqrt{(N+1)(1+1)} = 1/\sqrt{2(N+1)}$ — fato que torna a representação da raiz **explicitamente dependente de $N$ via a normalização**. Esta dependência implícita explica por que `posgrau` (que torna $N$ explícito como feature) pode ser redundante: o GCN já tem acesso indireto a $N$ pelo termo de normalização. A ablação testa se essa dependência implícita é suficiente.

**No código:**
> Linhas 188: `GCNClassifier(num_feats, 2, seed=fold_idx)` — instancia o classificador. A definição em `gcn_model.py` (referenciada via PIPELINE.md) implementa três `GCNConv` + `global_mean_pool` + linear, com seed do PyTorch fixada por fold para reprodutibilidade pareada.

---

### 2.5 T-test Pareado contra Baseline — Validação Estatística

**Descrição técnica:**
Para cada variante $v \in \{\text{posfull}, \text{posmin}, \text{posgrau}\}$, o script forma 10 pares $(F1_{v,k}, F1_{\text{baseline},k})$ um por fold, e aplica:

$$t = \frac{\bar{d}}{s_d / \sqrt{K}}, \quad d_k = F1_{v,k} - F1_{\text{baseline},k}$$

onde $\bar{d}$ é a média das diferenças intra-fold e $s_d$ o desvio amostral. Sob $H_0: \mu_d = 0$, $t \sim t_{K-1=9}$.

O pareamento é válido porque os folds são gerados **uma vez** em `gerar_folds.py` e compartilhados — o LogReg-BERT do Script 10 e os GCNs do Script 12 viram o mesmo subconjunto de teste em cada fold, eliminando variância de partição.

**Embasamento acadêmico:**

> 📖 **Dietterich, T. G. (1998)** — "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
> *Neural Computation*, v. 10, n. 7, pp. 1885–1924
> DOI: `10.1162/089976698300017197`
> **Localização:** Seção 4 (10-Fold Cross-Validated Paired t-Test) — protocolo idêntico ao implementado no script; Tabela 1 — taxas de Tipo I e potência para K=10 folds
> **Relevância:** Justifica formalmente o uso de `ttest_rel` com K=10. Dietterich também alerta para inflação leve de Tipo I devido à correlação entre folds (≈90% dos dados de treino são compartilhados entre folds adjacentes); como o script aplica α=0.05 simples sem correção de Bonferroni para os 3 testes simultâneos, o leitor deve interpretar p-valores marginais (entre 0.01 e 0.05) com ressalva.

**No código:**
> Linha 225: `t, p = stats.ttest_rel(vals, base)` — o vetor `vals` contém F1 por fold da variante atual, `base` contém F1 do LogReg-BERT no mesmo fold. A ordenação por `folds_ord` (linha 209) garante o pareamento correto.

---

## 3. Pipeline de Dados

### 3.1 Carga das três variantes

O script depende da existência de **três datasets pré-construídos** em disco:
- `data/fakenewsnet_posfull/{train,val,test}.pt`
- `data/fakenewsnet_posmin/{train,val,test}.pt`
- `data/fakenewsnet_posgrau/{train,val,test}.pt`

Gerados previamente por três execuções de `00_construir_grafos_fakenewsnet.py`:

```
python 00_construir_grafos_fakenewsnet.py --feature-variant full     --output-suffix posfull
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-min  --output-suffix posmin
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-grau --output-suffix posgrau
```

`carregar_variante()` (linhas 58–69) une os três splits em uma única lista — note que o script reagrega train/val/test antes de aplicar os folds estratificados de `folds_fnn.pt`, então a partição original 60/20/20 do Script 00 é descartada e substituída pela K-fold do Script 09.

### 3.2 Folds compartilhados

Linha 168: `folds = list(iterar_folds())` lê `data/folds_fnn.pt` (gerado por `gerar_folds.py`) — cada fold é uma tupla `(fold_idx, train_idx, test_idx)`. Os mesmos índices são aplicados às três variantes, e os mesmos índices foram aplicados ao LogReg-BERT no Script 10. Isso é a condição **necessária** para validade do t-test pareado contra baseline.

### 3.3 Validação interna estratificada

`split_train_val()` (linhas 72–83) — separa 10% do treino do fold para early stopping, estratificado manualmente por classe. Idêntico ao Script 09; usa `seed=fold_idx` para que a separação varie entre folds, evitando inflação sistemática do erro.

---

## 4. Métricas e Gates de Decisão

### 4.1 F1-macro

**Fórmula:**
$$F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F1_c, \qquad F1_c = \frac{2\,P_c\,R_c}{P_c + R_c}$$

onde $C=2$ (Fake, Real), $P_c$ é precisão e $R_c$ é recall da classe $c$.

**Por que macro e não binário (`pos_label=0`)?**
O FakeNewsNet local é aproximadamente balanceado (50/50), mas mesmo assim macro é mais conservador: pondera Fake e Real igualmente, evitando que melhorias em uma única classe enviesem a métrica. Esta é uma diferença vs. o Script 09, que usou F1 binário com `pos_label=0`. A escolha aqui é deliberada porque o script compara com LogReg-BERT, que também é avaliado em macro no Script 10.

**No código:**
> Linha 98: `f1_score(y_true, y_pred, average="macro", zero_division=0)`. O `zero_division=0` retorna 0 quando uma classe não tem predições — evita warning ruidoso em folds com falha de convergência.

---

### 4.2 Gate de Posição

**Critério:**
$$\text{aprovar} \iff \big[ \,p(\text{posmin vs baseline}) < 0.05 \,\wedge\, F1(\text{posmin}) > F1(\text{baseline}) \,\big]$$

**Interpretação:**
Se aprovado, a posição ordinal `pos = i/N` (sem informação de grau, sem texto além do BERT já presente) **adiciona sinal preditivo significativo** sobre regressão logística textual. Isso equivaleria a dizer: "saber a ordem em que os tweets aparecem no CSV ajuda a classificar fake news, mesmo após controlar o texto da notícia".

**Caveat documentado no Script 00 (linha 27 do header):**
> "ordem dos `tweet_ids` é a do CSV, não cronológica"

A interpretação de `pos` como "tempo" é portanto **frágil** — é apenas a ordem em que o CSV foi escrito por `KaiDMML/FakeNewsNet`. Se este gate aprovar, o achado é metodologicamente embaraçoso: a GNN está aprendendo um artefato de ordenação de planilha. Se reprovar, o problema desaparece.

---

### 4.3 Gate de Confound

**Critério:**
$$\text{aprovar} \iff F1(\text{posgrau}) - F1(\text{posmin}) > 0.01$$

**Interpretação:**
Se a variante com **apenas grau** (sem `pos`) supera a variante com **apenas pos** (sem grau) por mais de 1pp em F1, o sinal preditivo dominante na "estrutura" não é ordinal/posicional — é **popularidade** ($N$ = número de tweets). Esta é a confirmação direta do **Erro 3** documentado no TCC (confound de tamanho de cascata).

A literatura de detecção de fake news em redes sociais já documentou amplamente que volume de propagação correlaciona com viralidade, e viralidade correlaciona (espuriamente, por viés de coleta) com label de fake (Vosoughi et al., 2018, *Science*). Se este gate aprovar, todo o argumento "GNN aprende estrutura" colapsa em "GNN aprende contagem de filhos" — feature trivialmente computável sem grafo.

**Limiar de 0.01:**
O delta de 1pp é heurístico — não derivado de teste estatístico formal (t-test não pareado entre `posgrau` e `posmin` poderia ser implementado). Nos níveis de variância típicos do dataset (σ≈0.03–0.04 em F1, vide Script 09), 1pp está abaixo do erro padrão de uma única run e pode refletir variância de inicialização. Isto é uma **limitação reconhecida** do gate.

---

## 5. Análise Empírica: Posicionamento na Questão Central

### 5.1 Pontos fortes desta abordagem

1. **Isolamento causal forte.** Apenas uma variável muda entre as três variantes — as três dimensões posicionais são manipuladas individualmente, modelo e folds são fixos. Isso satisfaz o critério metodológico de Meyes et al. (2019, Seção 5) para inferência causal em ablação.

2. **Pareamento triplo.** Os 10 folds são compartilhados entre (a) as três variantes GCN, (b) o LogReg-BERT do Script 10. O resultado é que o t-test contra baseline é pareado fold-a-fold, com máxima potência estatística (Dietterich, 1998).

3. **Conexão direta com a tese.** Os gates traduzem números em decisões textuais explícitas para a monografia. Não é necessário re-interpretar p-valores na escrita — o `relatorio.txt` já resolve "ganho de posição confirmado/refutado" e "confound dominado por grau confirmado/refutado".

4. **Reusa pipeline existente.** Não introduz código novo de modelagem — reusa `GCNClassifier`, `iterar_folds`, `split_train_val`. Isso minimiza a superfície de bugs e mantém comparabilidade com Scripts 09 e 13.

### 5.2 Limitações identificadas

1. **`posnone` ausente.** Não há variante "só `is_root`" como controle. Se `posfull ≈ posmin ≈ posgrau ≈ baseline` (todas iguais), não conseguiríamos distinguir entre "as três dimensões são todas vazias" e "alguma combinação não testada importa".

2. **Apenas GCN.** O experimento não testa se o achado (qual dimensão importa) generaliza para GAT/SAGE. Em GAT, a atenção pode pesar `pos` ou `grau` diferentemente; em SAGE, o agregador `mean` é menos dependente do grau implícito da normalização. O Script 09 mostrou que as três arquiteturas são equivalentes em F1 — dá confiança limitada para extrapolar, mas não substitui o experimento.

3. **Heurística no Gate de Confound.** O limiar de 0.01 não é derivado de teste estatístico. Uma versão mais rigorosa seria t-test pareado entre `posgrau` e `posmin` com α=0.05 — implementável adicionando 3 linhas ao script.

4. **Caveat ordinal.** Como `pos` é ordem do CSV (não cronológica), aprovar o Gate de Posição é cientificamente problemático. O design correto seria **descartar** o Gate de Posição inteiramente, ou substituir `pos` por uma feature genuinamente cronológica (timestamps de tweets, indisponíveis sem API paga). O script e a monografia precisam reportar esse caveat com destaque.

5. **N=10 folds, baixa potência.** Como discutido no Script 09, com K=10 e σ≈0.03, o teste só detecta diferenças de F1 > ~0.04 com boa potência. Diferenças menores ficarão como "ns" mesmo se forem reais.

### 5.3 Comparação com a literatura

| Estudo | Componente ablado | Conclusão | Referência |
|--------|-------------------|-----------|------------|
| Errica et al. (2020) | Estrutura do grafo (vs. baseline structureless) | Em 5/9 datasets clássicos, GNN ≈ baseline | Tabela 2, ICLR 2020 |
| Dwivedi et al. (2022) | Encoding posicional (Laplaciano) | Adicionar PE melhora GCN/GAT/GIN consistentemente | Tabelas 1–3, ICLR 2022 |
| Hamilton (2020), Cap. 5 | Features nodais idênticas | "Featureless" GNNs são limitadas pelo Weisfeiler-Lehman | Seção 5.4 |
| Este script (12) | Dimensão posicional individual | A definir empiricamente (vide gates) | — |

> 📖 **Hamilton, W. L. (2020)** — "Graph Representation Learning"
> *Synthesis Lectures on Artificial Intelligence and Machine Learning*, Morgan & Claypool
> ISBN: 978-1681739632
> **Localização:** Capítulo 5 (The Graph Neural Network Model), Seção 5.4 (Generalized Neighborhood Aggregation), discussão sobre "node features" — Hamilton observa que MPNNs sem features nodais distintas têm poder representacional limitado pelo teste de Weisfeiler-Lehman, e features posicionais são uma forma de aumentar esse poder
> **Relevância:** Embasamento teórico para por que `is_root` precisa estar presente em todas as variantes — sem ele, nó-raiz e folhas seriam equivalentes pelo WL-test, e GCN não conseguiria distingui-los.

### 5.4 Resposta parcial à questão do TCC

> **"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP?"**

O Script 12 contribui especificamente para a **decomposição causal do ganho aparente da GNN sobre o baseline textual**. Três cenários são possíveis a posteriori dos gates:

- **Cenário A — ambos os gates reprovam:** `posfull ≈ posmin ≈ posgrau ≈ baseline`. Implicação: as três dimensões posicionais juntas não adicionam sinal além do BERT da raiz. **Veredicto:** GNN com estrela plana é equivalente a NLP — a topologia é cosmética.
- **Cenário B — Gate de Posição reprova, Gate de Confound aprova:** `posgrau > posmin ≈ baseline`. Implicação: o sinal "estrutural" da GNN é apenas popularidade ($N$). **Veredicto:** GNN aprende um confound trivial; vantagem sobre NLP é ilusória — uma única feature escalar (contagem) replicaria o desempenho.
- **Cenário C — Gate de Posição aprova:** `posmin > baseline`. Implicação: a ordem dos tweets adiciona sinal. **Veredicto:** problemático — como `pos` é ordem do CSV, isso indicaria viés de coleta no FakeNewsNet, não habilidade real da GNN.

Em **qualquer dos três cenários**, a conclusão para o TCC é desfavorável ao argumento "GNNs aprendem topologia complexa". O Script 12 é, portanto, uma das engrenagens-chave da Fase 5 (Vulnerabilidade Topológica): ele transforma um claim arquitetural em afirmações testáveis sobre dimensões individuais, e nenhum cenário plausível favorece o GNN no espírito (representação aprendida de propagação rica). Junto ao Script 11 (`diagnóstico_confound`, que mostra que `num_nodes` sozinho atinge F1>0.65) e ao Script 14 (`topologia_sem_texto`, que mostra que features estruturais sem BERT atingem F1 próximo do completo), forma a tríade probatória central da tese.

---

## 6. Análise de Código

### 6.1 Erros e ressalvas identificadas

**E1 — `seed=fold_idx` em `torch.manual_seed`:**
```python
# Linha 183
torch.manual_seed(fold_idx); np.random.seed(fold_idx)
```
Como no Script 09, usar `fold_idx` como seed correlaciona inicialização com fold — todos os modelos do fold $k$ partem da mesma seed, o que é correto para pareamento, mas subestima variância de inicialização. Aceitável, mas a monografia deve ressaltar que o desvio padrão observado é **inter-fold**, não inter-seed.

**E2 — `assert` em `00_construir_grafos_fakenewsnet.py` (linhas 167–172) é frágil para `posgrau`:**
```python
n_esperado = num_nos if usa_pos else min(num_nos, 2)
assert n_unique >= min(n_esperado, 3) or num_nos < 2
```
Para `posgrau`, com `usa_pos=False`, espera apenas 2 vetores únicos (raiz + um filho-tipo) porque todos os filhos têm features idênticas. Isso é **comportamento intencional da variante**, não um bug — mas significa que `posgrau` recria parcialmente o Erro 1 nos filhos. A GCN ainda distingue raiz×filhos via `is_root`, mas todos os filhos passam mensagens idênticas. A interpretação correta para essa variante é: "o GCN só pode usar grau (via `grau_norm` na raiz) e o degree-implícito da normalização".

**E3 — Gate de Confound sem teste estatístico:**
```python
# Linhas 246–247
delta = medias["posgrau"] - medias["posmin"]
gate_conf = delta > GATE_CONFOUND_DELTA  # 0.01
```
Sugestão de melhoria:
```python
# Adicionar t-test pareado entre posgrau e posmin
vals_pg = np.array([resultados["posgrau"][i] for i in folds_ord])
vals_pm = np.array([resultados["posmin"][i]  for i in folds_ord])
t_pg_pm, p_pg_pm = stats.ttest_rel(vals_pg, vals_pm)
gate_conf = (delta > GATE_CONFOUND_DELTA) and (p_pg_pm < 0.05)
```
Isso converteria o gate heurístico em decisão estatística formal, robusta à variância de inicialização.

**E4 — Sem correção de múltiplos testes:**
Três p-valores simultâneos (posfull, posmin, posgrau vs baseline) com α=0.05 individual: $\alpha_{\text{FWER}} \approx 0.143$. Bonferroni daria $\alpha' \approx 0.017$. Como o Gate de Posição usa apenas um dos três (`posmin`), o impacto é menor, mas a monografia deve mencionar.

### 6.2 Boas práticas observadas

- **Reuso de utilitários:** `iterar_folds`, `split_train_val`, `GCNClassifier` — zero duplicação vs Scripts 09/10/11.
- **Limites de dimensão herdados:** ao reusar `GCNClassifier(num_feats=771, ...)`, o script automaticamente herda a especificação correta sem hardcoding.
- **Saída estruturada:** `tabela.csv` permite re-aplicação de testes alternativos (Wilcoxon, Bonferroni-corrected) sem re-executar; `relatorio.txt` traz decisão dos gates explícita para citação direta na escrita.
- **Sanity check em `carregar_baseline_por_fold`:** falha ruidosa se `baseline_textual/resultados.csv` não existe, instruindo a rodar Script 10 — boa programação defensiva.

### 6.3 Sugestões de extensão (não-blocantes)

1. **Adicionar variante `posnone` (só `is_root`):** mesmo que reintroduza o Erro 1, fornece o limite inferior absoluto da GCN e fecha o quadrado experimental.
2. **Replicar com GAT e SAGE:** validaria que o achado generaliza além de GCN.
3. **Substituir gate heurístico por t-test:** vide E3.
4. **Adicionar Cohen's d para tamanho de efeito:** F1 médios podem ser estatisticamente significativos mas praticamente irrelevantes (delta < 0.01 em produção).

---

## 7. Referências Bibliográficas

1. **MEYES, R.; LU, M.; DE PUISEAU, C. W.; MEISEN, T.** Ablation Studies in Artificial Neural Networks. *arXiv preprint*, 2019. arXiv: `1901.08644`. Disponível em: `https://arxiv.org/abs/1901.08644`

2. **DWIVEDI, V. P.; LUU, A. T.; LAURENT, T.; BENGIO, Y.; BRESSON, X.** Graph Neural Networks with Learnable Structural and Positional Representations. *International Conference on Learning Representations (ICLR)*, 2022. arXiv: `2110.07875`. Disponível em: `https://arxiv.org/abs/2110.07875`

3. **VASWANI, A.; SHAZEER, N.; PARMAR, N.; USZKOREIT, J.; JONES, L.; GOMEZ, A. N.; KAISER, Ł.; POLOSUKHIN, I.** Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. arXiv: `1706.03762`. Disponível em: `https://arxiv.org/abs/1706.03762`

4. **KIPF, T. N.; WELLING, M.** Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*, 2017. arXiv: `1609.02907`. Disponível em: `https://arxiv.org/abs/1609.02907`

5. **ERRICA, F.; PODDA, M.; BACCIU, D.; MICHELI, A.** A Fair Comparison of Graph Neural Networks for Graph Classification. *International Conference on Learning Representations (ICLR)*, 2020. arXiv: `1912.09893`. Disponível em: `https://arxiv.org/abs/1912.09893`

6. **HAMILTON, W. L.** Graph Representation Learning. *Synthesis Lectures on Artificial Intelligence and Machine Learning*, Morgan & Claypool, 2020. ISBN: 978-1681739632.

7. **YING, R.; BOURGEOIS, D.; YOU, J.; ZITNIK, M.; LESKOVEC, J.** GNNExplainer: Generating Explanations for Graph Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. arXiv: `1903.03894`. Disponível em: `https://arxiv.org/abs/1903.03894`. **Localização relevante:** Seção 3.2 (Mutual Information Formulation) — formaliza importância de features/arestas via MI; embasamento teórico paralelo para "qual feature é causalmente necessária" — mesmo objetivo do Script 12, abordagem complementar (post-hoc explanation vs. ablação por design).

8. **DIETTERICH, T. G.** Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, v. 10, n. 7, pp. 1885–1924, 1998. DOI: `10.1162/089976698300017197`.

9. **VOSOUGHI, S.; ROY, D.; ARAL, S.** The Spread of True and False News Online. *Science*, v. 359, n. 6380, pp. 1146–1151, 2018. DOI: `10.1126/science.aap9559`. **Localização relevante:** Tabela 1 — fakes propagam mais e têm cascatas maiores; embasa o "confound de tamanho" testado pelo Gate de Confound.

10. **SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H.** FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

---

## 8. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Estudo de ablação | Remoção sistemática de componentes para isolar contribuição causal de cada um | Meyes et al. (2019), Seção 4 |
| Encoding posicional (PE) | Vetor que injeta informação de posição em modelos invariantes a permutação | Vaswani et al. (2017), Seção 3.5 |
| `posfull` / `posmin` / `posgrau` | Variantes do dataset com 3 / 2 / 2 dimensões posicionais ativas | Script 00, linhas 140–153 |
| `is_root` | Indicador binário: 1 se nó é a raiz (artigo), 0 se filho (tweet) | Script 00, linha 145 |
| `grau_norm` | Número de filhos $N$ normalizado por $N_{\max}^{\text{global}}$ — feature de popularidade | Script 00, linha 144 |
| `pos` | Posição ordinal $i/N$ do filho $i$, baseada na ordem do CSV (não cronológica) | Script 00, linha 151 |
| Erro 1 (TCC) | Features nodais idênticas em estrela plana → GCN colapsa em média global do BERT | Auditoria do TCC, Fase 3.1 |
| Erro 3 (TCC) | Confound de tamanho de cascata: $N$ correlaciona com label por viés de coleta | Auditoria do TCC, Fase 5B.2 |
| Gate de Posição | Critério: F1(posmin) > F1(baseline) com p<0.05 | Script 12, linhas 234–243 |
| Gate de Confound | Critério: F1(posgrau) − F1(posmin) > 0.01 | Script 12, linhas 245–254 |
| Baseline structureless | Modelo que ignora a estrutura do grafo, usa apenas features nodais agregadas | Errica et al. (2020), Seção 4.2 |
| F1-macro | Média não-ponderada do F1 por classe; conservador em datasets balanceados | Sokolova & Lapalme (2009) |
| T-test pareado (`ttest_rel`) | Teste t sobre diferenças intra-par; máxima potência quando observações são pareadas | Dietterich (1998), Seção 4 |
| K-fold compartilhado | Mesma partição estratificada usada por todos os modelos comparados | Script `gerar_folds.py`, linha 57 |

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 12_ablation_intra_encoding.py
📄  Arquivo gerado: theory_andre/12_ablation_intra_encoding_doc.md
📚  Fontes acadêmicas utilizadas: 10
    1. Meyes et al. (2019) — Ablation Studies (arXiv:1901.08644)
    2. Dwivedi et al. (2022) — GNNs with Learnable Structural and Positional Representations (ICLR, arXiv:2110.07875)
    3. Vaswani et al. (2017) — Attention is All You Need (NeurIPS, arXiv:1706.03762)
    4. Kipf & Welling (2017) — GCN (ICLR, arXiv:1609.02907)
    5. Errica et al. (2020) — Fair Comparison of GNNs (ICLR, arXiv:1912.09893)
    6. Hamilton (2020) — Graph Representation Learning (Morgan & Claypool, livro)
    7. Ying et al. (2019) — GNNExplainer (NeurIPS, arXiv:1903.03894)
    8. Dietterich (1998) — Statistical Tests for Comparing Classifiers (Neural Computation)
    9. Vosoughi, Roy & Aral (2018) — Spread of True and False News Online (Science)
   10. Shu et al. (2020) — FakeNewsNet (Big Data)
🔍  Conceitos cobertos: ablação como método científico; encoding posicional em
    Transformers e GNNs; correção do Erro 1 (features nodais idênticas em estrela
    plana); decomposição causal das 3 dims posicionais (is_root, grau_norm, pos);
    GCNClassifier como aggregator linear; t-test pareado contra baseline textual;
    Gate de Posição (sinal ordinal) e Gate de Confound (sinal de popularidade);
    conexão com Erros 1 e 3 do TCC.
⚠️   Limitações: (a) WebSearch/WebFetch foram bloqueados pelo ambiente — todas as
    referências foram redigidas a partir de conhecimento prévio dos textos
    originais, com IDs arXiv/DOI estáveis. Recomenda-se conferência manual de
    seções/equações antes de submeter à banca. (b) Variante `posnone` ausente
    impede limite inferior absoluto. (c) Gate de Confound é heurístico sem
    teste estatístico formal — sugestão de extensão fornecida em §6.3.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
