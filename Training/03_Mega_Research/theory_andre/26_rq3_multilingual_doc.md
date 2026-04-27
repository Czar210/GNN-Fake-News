# Documentação Técnica: 26_rq3_multilingual.py

## Metadados

- **Arquivo analisado:** `26_rq3_multilingual.py`
- **Caminho:** `Training/03_Mega_Research/26_rq3_multilingual.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Validação estatística cross-lingual de classificador estrutural (RandomForest treinado em GossipCop) — Kolmogorov-Smirnov 2-sample, Mann-Whitney U e Cohen's d sobre distribuições de score em posts monolingues PT/DE/EN do Bluesky.
- **Modelo principal:** `rf_struct_gossipcop.pkl` (RandomForestClassifier com features `[num_nodes, grau_root]`).
- **Datasets utilizados:** `dados_bluesky/feed_posts/*.jsonl` filtrado por `langs == ('eng',)`, `('deu',)`, `('por',)` exclusivamente.
- **Contribuição para a questão central:** este script responde **diretamente a RQ3** do TCC (Seção 6.4): se o classificador estrutural for *language-agnostic*, a distribuição de scores deve ser estatisticamente indistinguível entre PT, DE e EN. Distribuições iguais validam o claim "o modelo decide pela ESTRUTURA, não pelo idioma"; distribuições diferentes refutam-no e indicam que algum sinal lingüístico latente vaza para as features estruturais (e.g., comunidades específicas têm padrões de propagação distintos por idioma). É a peça final do argumento de **vulnerabilidade topológica** do TCC, agora estendida à dimensão multilíngue.

---

## 1. Visão Geral do Script

`26_rq3_multilingual.py` aplica o RandomForest estrutural treinado em GossipCop (`Execution/weights/rf_struct_gossipcop.pkl`) sobre posts do Bluesky particionados por idioma nativo declarado no campo `langs` do AT Protocol. O fluxo é: (i) carregar o RF, (ii) varrer `feed_posts/*.jsonl` filtrando exclusivamente posts **monolingues** (com `len(langs) == 1`) cujo código ISO 639-3 esteja em `{eng, deu, por}`, (iii) extrair as duas únicas features estruturais (`num_nodes = 1 + reply + repost + quotes`, `grau_root = reply + repost + quotes`), (iv) calcular a probabilidade de classe 0 (fake) via `predict_proba`, e (v) testar pareadamente as três distribuições.

A análise estatística usa três testes complementares por par de idiomas: **Kolmogorov-Smirnov 2-sample** (Massey, 1951) testa se as distribuições são iguais em formato; **Mann-Whitney U** (Mann & Whitney, 1947) testa se há deslocamento de mediana sem assumir normalidade; e **Cohen's d** (Cohen, 1988) quantifica magnitude do efeito (não significância). A combinação é deliberada: KS/MW retornam p-valores que com `n` grande (Bluesky tem dezenas de milhares de posts por idioma) tornam-se sensíveis a desvios triviais; Cohen's d corrige isso indicando se o efeito é **praticamente** relevante (d < 0.2 ⇒ "desprezível").

A saída inclui `scores_por_lang.csv` (uma linha por post), `comparacoes.csv` (estatísticas pareadas), `fig_distribuicao.png` (3 histogramas sobrepostos), `fig_qq.png` (Q-Q PT×EN e DE×EN) e `relatorio.txt` (interpretação textual). A visualização Q-Q é particularmente útil porque, sob H₀ ("idiomas equivalentes"), os pontos devem cair sobre `y = x` — desvios sistemáticos da diagonal sinalizam viés cross-lingual mesmo quando o p-valor agregado é dominado pelo tamanho amostral.

O script difere dos outros experimentos do TCC por **não treinar nada**: é puramente uma análise *post-hoc* de invariância. A hipótese H₀ é "RF estrutural é language-agnostic"; rejeição significa que o modelo absorveu sinal lingüístico via correlação espúria entre comunidade e padrão de propagação no GossipCop.

---

## 2. Arquitetura e Componentes Principais

### 2.1 RandomForestClassifier (modelo carregado, treinado em script anterior)

**Descrição técnica:**
RandomForest é um *ensemble* de árvores de decisão treinadas com **bagging** (bootstrap aggregating) e amostragem aleatória de features em cada split. A predição final é a média (regressão) ou voto majoritário (classificação) das árvores. Para classificação probabilística, `predict_proba(X)[i, c]` é a fração de árvores que classificam `X[i]` como classe `c`. Neste script o RF tem 2 features apenas (`num_nodes`, `grau_root`), tornando-o um classificador estrutural mínimo.

**Fundamento matemático:**

Seja $T_b$ a $b$-ésima árvore treinada em uma amostra bootstrap $\mathcal{D}_b$ do dataset $\mathcal{D}$. A probabilidade estimada da classe $c$ para entrada $\mathbf{x}$ é:

$$\hat{p}(c \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}\{T_b(\mathbf{x}) = c\}$$

onde $B$ é o número de árvores (`n_estimators`) e $\mathbb{1}$ é a função indicadora. Em cada nó, a árvore escolhe o split que maximiza redução de Gini ou entropia; em cada split, apenas $\sqrt{p}$ features (de $p$ totais) são candidatas — no nosso caso, $p=2$ e $\sqrt{p} \approx 1.4$, então há aleatoriedade trivial entre features.

**Embasamento acadêmico:**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 1 (definição formal de Random Forest), Seção 2 (taxa de erro de generalização e teorema da convergência), Seção 4 (justificativa do bootstrap + feature subsampling).
> **Relevância:** Estabelece o RF como ensemble que converge em lei para uma taxa de erro limitada superiormente (Teorema 2.3) por uma função da força das árvores individuais e da correlação entre elas. Para o RQ3, o ponto chave é que o RF aprende **partições no espaço de features** — se as duas features (`num_nodes`, `grau_root`) forem invariantes a idioma, o score de saída também será.

**No código:**
> Linhas 78–81: carregamento do `rf_struct_gossipcop.pkl` via `pickle.load`.
> Linha 122–124: `rf.predict_proba(X)[:, idx_fake]` retorna a probabilidade da classe 0 (fake) para cada post — esta é a variável aleatória cuja distribuição é comparada entre idiomas.

---

### 2.2 Filtro Monolingue via Campo `langs` do AT Protocol

**Descrição técnica:**
O Bluesky usa o AT Protocol, que armazena um campo opcional `langs` em cada post — uma lista de códigos ISO 639-3 declarada pelo autor. O script aceita apenas posts com **exatamente um** código (`len(langs) == 1`), descartando posts multilingues (que confundiriam a atribuição de idioma) e posts sem declaração (que poderiam ser detectados via langid mas introduziriam ruído de classificação).

**Justificativa estatística:**
A H₀ do experimento ("o classificador é invariante a idioma") só é testável se a partição por idioma for limpa. Posts multilingues têm distribuição de score mista (não atribuível a um único idioma) e violariam a independência das amostras requerida pelo KS-test e pelo Mann-Whitney U. O custo é amostral: posts monolingues são minoria em comunidades cosmopolitas. O ganho é validade estatística.

**No código:**
> Linhas 104–108: filtro `if len(langs) != 1: continue` seguido de `if lg not in LANGS_ALVO: continue`. A combinação garante particionamento exato em três grupos disjuntos.
> Linha 65: `LANGS_ALVO = {"eng": "Ingles", "deu": "Alemao", "por": "Portugues"}` — códigos ISO 639-3 (note que ISO 639-1 seria `en/de/pt`; o AT Protocol usa 639-3).

---

## 3. Pipeline de Dados e Pré-processamento

### 3.1 Extração de Features Estruturais

**Descrição técnica:**
Para cada post, são calculadas duas features:
- `num_nodes = 1 + interacoes` — número total de nós no grafo de propagação (raiz + respostas + reposts + quotes).
- `grau_root = interacoes` — grau do nó raiz, equivalente à popularidade imediata.

Ambas são contagens inteiras não-negativas, portanto sem unidade textual ou semântica — o que torna o RF, em princípio, language-agnostic *por construção*. A pergunta de RQ3 é se, na prática, o GossipCop induziu correlações espúrias entre essas contagens e algum atributo lingüístico que o Bluesky reproduz.

**No código:**
> Linhas 84–90: função `extrair_features(post)`. Note que `quotes` só é somado se for `int` (proteção contra `None` ou string).

---

### 3.2 Cálculo de Scores via `predict_proba`

**Descrição técnica:**
Para cada idioma, monta-se a matriz $X \in \mathbb{R}^{n \times 2}$ com as duas features e aplica-se `rf.predict_proba(X)[:, idx_fake]`. O resultado é um vetor de probabilidades em $[0, 1]$ — a "fração de árvores que classificam o post como fake".

**Detalhe sutil — `idx_fake = classes.index(0) if 0 in classes else 0`:**
o RF foi treinado com classes `{0=fake, 1=real}`, então a coluna de interesse no `predict_proba` é a coluna 0. A guarda `if 0 in classes` é defensiva contra modelos treinados com label encoder diferente.

**No código:**
> Linhas 118–124: função `calcular_scores`. Retorna `np.array([])` se a lista estiver vazia (proteção para idiomas com zero posts monolingues).

---

## 4. Construção do Grafo (não aplicável)

Este script **não constrói grafo**. As features estruturais já vêm tabuladas no JSONL do Bluesky (`reply_count`, `repost_count`, `quotes`), porque o pipeline upstream (script `02_construir_grafos_bluesky.py` da Fase 0) já consolidou os agregados. A análise opera sobre features escalares por post — não há message passing nem modelo GNN no caminho.

---

## 5. Métricas de Avaliação

### 5.1 Kolmogorov-Smirnov 2-sample test (`stats.ks_2samp`)

**Descrição técnica:**
Testa H₀: "duas amostras vêm da mesma distribuição contínua". A estatística é a distância máxima entre as funções de distribuição empíricas (CDFs) das duas amostras.

**Fórmula:**

Sejam $F_n(x)$ e $G_m(x)$ as CDFs empíricas das amostras $A = \{a_1, \ldots, a_n\}$ e $B = \{b_1, \ldots, b_m\}$. A estatística KS é:

$$D_{n,m} = \sup_{x} \left| F_n(x) - G_m(x) \right|$$

Sob H₀, e para $n, m \to \infty$:

$$\sqrt{\frac{nm}{n+m}} \cdot D_{n,m} \xrightarrow{d} K$$

onde $K$ é a distribuição de Kolmogorov, com $P(K \leq t) = 1 - 2\sum_{j=1}^{\infty}(-1)^{j-1}e^{-2j^2 t^2}$. O p-valor é $P(K \geq \sqrt{nm/(n+m)} \cdot D_{n,m})$.

**Interpretação variável por variável:**
- $D_{n,m} \in [0, 1]$: 0 ⇒ CDFs idênticas; 1 ⇒ suportes disjuntos.
- $n, m$: tamanhos amostrais. Com $n, m$ grandes (caso Bluesky), até $D \approx 0.01$ pode dar $p < 0.001$.
- O *sup* é tomado sobre todos os $x$, então o teste é sensível a qualquer diferença local nas CDFs (caudas, mediana, formato).

**Embasamento acadêmico:**

> 📖 **Massey, F. J. (1951)** — "The Kolmogorov-Smirnov Test for Goodness of Fit"
> *Journal of the American Statistical Association (JASA)*, v. 46, n. 253, pp. 68–78, 1951
> DOI: `10.1080/01621459.1951.10500769`
> **Localização:** Seção "The Test" (definição da estatística $D$ e tabelas de valores críticos), Seção "Power" (potência relativa ao χ²).
> **Relevância:** Massey formaliza o KS como teste *distribution-free* — sob H₀ a distribuição de $D$ não depende da distribuição subjacente, o que torna o KS o teste correto quando não há prior sobre o formato da distribuição de scores. Para RQ3, isso é essencial: a distribuição de `predict_proba` de um RF não é gaussiana (é multimodal por construção, com massa concentrada nos vértices do simplexo de votação), então testes paramétricos seriam inadequados.

**Por que KS é complementar ao Mann-Whitney:**
KS detecta diferenças de **formato** (e.g., uma distribuição é bimodal, outra unimodal) mesmo quando as medianas coincidem. MW detecta deslocamento de tendência central. Usar os dois reduz o risco de falso negativo onde uma distribuição "sutilmente diferente" passaria por um único teste.

**No código:**
> Linha 182: `ks_D, ks_p = stats.ks_2samp(sa, sb)` — implementação SciPy.

---

### 5.2 Mann-Whitney U test (`stats.mannwhitneyu`)

**Descrição técnica:**
Teste não-paramétrico para H₀: "duas amostras vêm de distribuições com a mesma localização". Equivalente ao Wilcoxon rank-sum. Não exige normalidade.

**Fórmula:**

Combine as duas amostras em uma lista de tamanho $n + m$ e atribua ranks (1 = menor, $n+m$ = maior). Seja $R_A$ a soma dos ranks dos elementos de $A$. Então:

$$U_A = R_A - \frac{n(n+1)}{2}, \qquad U_B = nm - U_A$$

A estatística reportada é $U = \min(U_A, U_B)$. Sob H₀ e $n, m$ grandes:

$$z = \frac{U - \mu_U}{\sigma_U}, \quad \mu_U = \frac{nm}{2}, \quad \sigma_U = \sqrt{\frac{nm(n+m+1)}{12}}$$

com $z \sim \mathcal{N}(0, 1)$ aproximadamente.

**Interpretação variável por variável:**
- $U \in [0, nm]$. $U = nm/2$ ⇒ amostras intercaladas (H₀ verdadeira).
- $U \to 0$ ou $U \to nm$ ⇒ uma amostra domina a outra em ordenação ⇒ rejeita H₀.
- Robusto a *outliers* porque opera em ranks, não em valores.

**Embasamento acadêmico:**

> 📖 **Mann, H. B. & Whitney, D. R. (1947)** — "On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other"
> *Annals of Mathematical Statistics*, v. 18, n. 1, pp. 50–60, 1947
> DOI: `10.1214/aoms/1177730491`
> **Localização:** Seção 1 (definição de $U$ e equivalência ao rank-sum de Wilcoxon 1945), Seção 3 (distribuição assintótica e Teorema 1 — normalidade aproximada de $U$).
> **Relevância:** Justificativa primária para usar MW no RQ3. O paper original prova que, sob H₀, $U$ tem variância e média conhecidas e converge para normal — permitindo p-valores exatos para amostras pequenas e aproximados para amostras grandes (o caso Bluesky). Como a saída do `predict_proba` do RF tem massa concentrada em alguns valores discretos (e.g., `0/100, 1/100, 2/100, ...` para 100 árvores), o teste paramétrico (t-test) seria mal-condicionado; MW lida com isso porque opera em ranks.

**Atenção a empates (`ties`):**
`predict_proba` com $B = 100$ árvores tem apenas 101 valores possíveis $\{0/100, 1/100, \ldots, 100/100\}$, gerando empates massivos. SciPy aplica correção de empates via `tie correction term` no denominador de $\sigma_U$. Isso preserva a validade do teste.

**No código:**
> Linhas 183–186: `mw_U, mw_p = stats.mannwhitneyu(sa, sb, alternative="two-sided")`. O `alternative="two-sided"` é correto: H₁ é "as distribuições diferem em qualquer direção".

---

### 5.3 Cohen's d (`cohens_d`, função custom)

**Descrição técnica:**
Métrica de **tamanho de efeito** padronizado — quantifica magnitude da diferença em unidades de desvio padrão pooled. Independente do tamanho amostral, ao contrário de p-valores.

**Fórmula:**

Para amostras $A, B$ com $n_A, n_B$ observações:

$$d = \frac{\bar{A} - \bar{B}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_A - 1) s_A^2 + (n_B - 1) s_B^2}{n_A + n_B - 2}}$$

onde $s_A^2, s_B^2$ são variâncias amostrais (`ddof=1`).

**Interpretação variável por variável (Cohen 1988):**
- $|d| < 0.2$ ⇒ efeito **desprezível** (idiomas praticamente equivalentes).
- $0.2 \leq |d| < 0.5$ ⇒ pequeno.
- $0.5 \leq |d| < 0.8$ ⇒ médio.
- $|d| \geq 0.8$ ⇒ grande.

**Embasamento acadêmico:**

> 📖 **Cohen, J. (1988)** — "Statistical Power Analysis for the Behavioral Sciences" (2ª ed.)
> *Lawrence Erlbaum Associates*, Hillsdale NJ, 1988. ISBN 0-8058-0283-5.
> **Localização:** Capítulo 2 ("The Significance of a Product-Moment $r_s$"), Seção 2.2 (definição de $d$); Capítulo 8, Seção 8.2 (limiares 0.2/0.5/0.8 explicitamente recomendados como convenções para "small/medium/large effect").
> **Relevância:** Cohen argumenta que p-valores em amostras grandes saturam (qualquer efeito microscópico vira "significativo"), tornando essencial reportar tamanho de efeito. Para RQ3, isso é o pivô do argumento: posts Bluesky em PT/DE/EN somam dezenas de milhares por idioma, o que praticamente garante $p < 0.001$ para qualquer KS/MW. **Apenas Cohen's d distingue "diferença real" de "diferença detectável por força bruta amostral"**. Se $|d| < 0.2$ entre PT e EN, o classificador é **praticamente** language-agnostic mesmo que estatisticamente o p-valor seja $< 10^{-6}$.

**No código:**
> Linhas 68–75: implementação manual com `ddof=1` (variância amostral, divisor $n-1$), guarda contra $n < 2$ e contra `pooled < 1e-12` (degenerescência numérica).

**Observação importante sobre o relatório (linhas 198–202):**
o script imprime os limiares de Cohen no `relatorio.txt`, garantindo que o leitor (banca do TCC) tenha a régua interpretativa imediatamente acessível — boa prática de comunicação científica.

---

### 5.4 Q-Q plot (visual diagnostic)

**Descrição técnica:**
Plota quantis empíricos da amostra $A$ contra quantis correspondentes da amostra $B$. Sob H₀ ("mesma distribuição"), os pontos caem sobre $y = x$. Desvios sistemáticos (curvatura, deslocamento, viés em uma cauda) revelam **onde** as distribuições diferem — algo que p-valores agregados ocultam.

**No código:**
> Linhas 250–252: `qs = np.linspace(0.01, 0.99, 50)` evita os extremos absolutos (que são instáveis para densidades empíricas), e `np.quantile` calcula 50 pares de quantis.
> Linha 253: scatter de `(qb, qa)` com a reta `y=x` pontilhada como referência visual.

**Por que importante para RQ3:**
um Q-Q plot pode revelar, e.g., que PT e EN coincidem na média (Cohen's d ≈ 0) mas divergem na cauda direita (poucos posts portugueses com `num_nodes` extremo recebem score muito alto). Esse tipo de "viés condicional" é diagnóstico-chave para a Seção 6.4 do TCC.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes desta abordagem

**(a) Tripla validação estatística complementar:**
KS detecta diferenças de formato; MW detecta deslocamento de mediana; Cohen's d quantifica magnitude. Cada teste cobre uma falha cega dos outros. Uma rejeição "dura" da H₀ exigiria coerência entre os três (KS-p baixo + MW-p baixo + |d| > 0.2). Uma equivalência "dura" exigiria |d| < 0.2 *independentemente* dos p-valores.

**(b) Uso do campo `langs` nativo do AT Protocol:**
em vez de classificar idioma com bibliotecas como `langid` ou `fastText` (que introduzem ruído de classificação ~3–5%), o script usa o metadado declarado pelo autor. Isso elimina uma fonte de ruído mas pressupõe declaração honesta — viável no Bluesky onde o campo é usado por clientes oficiais para filtragem de timeline.

**(c) Filtro monolingue estrito:**
descartar `len(langs) != 1` evita atribuição ambígua. Custo amostral compensado pelo volume bruto do Bluesky (~166k posts no dataset HF).

**(d) Saída completa e auditável:**
CSV por post permite re-análise externa (e.g., aplicar Wilcoxon signed-rank pareado, KS bootstrap, ou correção FDR de Benjamini-Hochberg para múltiplas comparações) sem re-executar o pipeline.

---

### 6.2 Limitações identificadas

**(L1) Apenas 2 features estruturais — *language-agnostic by design*, em parte:**
features `(num_nodes, grau_root)` são contagens inteiras puras, sem componente lingüística direta. A invariância esperada é quase tautológica — se o RF for invariante, isso pode refletir mais a pobreza das features do que uma propriedade profunda da estrutura. Um teste mais forte usaria features que **poderiam** ser sensíveis ao idioma (e.g., padrões de cascata, tempo entre eventos), mas o experimento atual já é informativo: estabelece um limite inferior para a invariância.

**(L2) Não há controle por feed:**
posts em alemão concentram-se em feeds europeus; em português, em feeds lusófonos. Comunidades diferentes têm padrões de propagação distintos por motivos sociológicos, não lingüísticos. O efeito "idioma" pode ser confundido com "comunidade". Mitigação parcial: rodar o experimento estratificado por feed (não implementado neste script — seria uma extensão natural).

**(L3) Múltiplas comparações sem correção:**
três pares (PT×EN, DE×EN, PT×DE) com α=0.05 dão $\alpha_{\text{FWER}} = 1 - 0.95^3 \approx 0.143$. Bonferroni daria $\alpha_{\text{corrigido}} = 0.0167$ por teste. Como o script reporta p-valores numéricos brutos, a correção pode ser aplicada externamente — mas o relatório textual não menciona isso.

**(L4) `langs` é declaração do autor, não classificação verificada:**
posts marcados `eng` podem incluir conteúdo em outros idiomas (gírias, citações, hashtags multilíngues). Ruído estimado: pequeno (<5%) em posts curtos.

**(L5) Score interpretado como probabilidade pode ser mal-calibrado:**
RFs são bem conhecidos por produzirem `predict_proba` com viés em direção a 0.5 (por causa da média entre árvores). Para o teste de **distribuição**, isso é irrelevante — o que importa é que o viés é o mesmo em todas as três amostras. Mas o relatório fala em "score fake-like" que pode ser super-interpretado como "probabilidade calibrada".

---

### 6.3 Comparação com o estado da arte

A literatura de detecção cross-lingual de fake news ataca o problema com modelos textuais multilíngues, não com features estruturais. Isso torna o experimento RQ3 deste TCC parcialmente **ortogonal** à literatura existente:

| Abordagem | Idiomas | Domínio das features | Resultado | Fonte |
|-----------|---------|----------------------|-----------|-------|
| Este script (RF estrutural) | PT, DE, EN | Topologia (`num_nodes`, `grau_root`) | A reportar (KS-D, MW-p, d por par) | — |
| mBERT zero-shot transfer | 104 idiomas | Texto (sub-word tokens) | Transferência cross-lingual surpreendentemente boa, mas degrada com distância tipológica | Pires et al. (2019) |
| XLM-R fake news multilíngue | EN→DE, EN→PT (entre outros) | Texto (sentencepiece) | F1 cross-lingual ~5–10pp abaixo de in-language | Conneau et al. (2020); Schwarz et al. (2020) |

> 📖 **Pires, T.; Schlinger, E.; Garrette, D. (2019)** — "How Multilingual is Multilingual BERT?"
> *Proceedings of ACL 2019*, pp. 4996–5001
> arXiv: `1906.01502`
> **Localização:** Seção 3 (cross-lingual zero-shot transfer com NER), Seção 4 (correlação entre similaridade tipológica e qualidade da transferência), Seção 5 (estrutura subjacente compartilhada entre línguas no espaço de embedding).
> **Relevância:** Estabelece que mBERT tem representações **parcialmente** language-agnostic — funciona melhor entre línguas tipologicamente próximas. Para o TCC, ancora a discussão sobre o que "language-agnostic" significa no contexto NLP, e contrasta com a invariância **estrutural** que RQ3 testa.

> 📖 **Conneau, A. et al. (2020)** — "Unsupervised Cross-lingual Representation Learning at Scale"
> *Proceedings of ACL 2020*, pp. 8440–8451
> arXiv: `1911.02116`
> **Localização:** Seção 3 (treino do XLM-R em CommonCrawl 2.5TB), Seção 5.1 (avaliação cross-lingual em XNLI), Seção 5.4 (curva *capacity dilution* — adicionar idiomas degrada qualidade per-idioma sem aumentar capacidade).
> **Relevância:** XLM-R é o estado-da-arte de representação cross-lingual; mostra que mesmo com escala massiva há *dilution* — modelos textuais não conseguem ser "perfeitamente" language-agnostic. O experimento RQ3 do TCC contrasta com isso: classificadores **estruturais** podem ser invariantes a idioma *por construção* (features são contagens), oferecendo uma rota alternativa para detecção multilíngue.

> 📖 **Schwarz, S.; Theóphilo, A.; Rocha, A. (2020)** — "EMET: Embeddings from Multilingual-Encoder Transformer for Fake News Detection"
> *ICASSP 2020*, pp. 2777–2781
> DOI: `10.1109/ICASSP40776.2020.9054673`
> **Localização:** Seção 3 (uso de mBERT como encoder universal para detecção de fake news), Seção 4 (avaliação em datasets EN/PT).
> **Relevância:** Trabalho representativo da abordagem **textual** para fake news multilíngue. Justifica por que features puramente estruturais (como neste script) seriam preferíveis em settings de baixíssimo recurso lingüístico — não dependem de tokenizadores nem de pré-treino multilíngue.

---

### 6.4 Resposta parcial à questão do TCC (Seção 6.4)

> **RQ3: "O classificador estrutural é language-agnostic?"**

A resposta tem três níveis dependendo do resultado empírico:

**Cenário A — H₀ não rejeitada (KS-p alto, MW-p alto, |d| < 0.2):**
o RF estrutural é, de fato, invariante a idioma. Isso **valida** o claim "o modelo decide pela ESTRUTURA" e fortalece a tese de vulnerabilidade topológica do TCC: o sinal aprendido em GossipCop (corpus EN puro) transfere-se para PT e DE *sem nenhum mecanismo de adaptação cross-lingual*. Conclusão metodológica: detecção estrutural é uma alternativa viável quando texto multilíngue é caro (mBERT/XLM-R requerem ~2GB+ de memória e GPU para inferência em escala).

**Cenário B — H₀ rejeitada com |d| < 0.2 (p baixo, efeito desprezível):**
estatisticamente significativo mas praticamente irrelevante — efeito atribuível ao tamanho amostral. Conclusão: language-agnostic *na prática*, mesmo que não *exatamente*. É o cenário esperado em datasets grandes e justifica a inclusão de Cohen's d na análise (Cohen 1988, Cap. 8).

**Cenário C — H₀ rejeitada com |d| ≥ 0.2:**
**refuta** o claim de invariância. O classificador absorveu sinal lingüístico via correlação espúria entre comunidade/idioma e padrão de propagação no GossipCop (e.g., posts em inglês têm cascatas mais virais por causa da composição amostral do dataset, não por algo intrínseco ao idioma). Esta seria uma **descoberta importante** para o TCC: contradiz a narrativa de "atalho topológico puro" e exige refinamento — talvez o RF estrutural não generaliza tão bem cross-platform/cross-lingual quanto o GossipCop sugere.

Em qualquer cenário, o experimento RQ3 **fecha o último flanco** da argumentação do TCC:

1. **Fase 5 (`14_topologia_sem_texto`)** mostrou que a topologia *sozinha* basta para classificar dentro do dataset.
2. **Fase 5 (`16_gnn_explainer_upfd`)** mostrou que o modelo *de fato olha* arestas estruturais, não texto.
3. **Fase 7 (`26_rq3_multilingual`)** testa se essa decisão estrutural transfere-se entre idiomas — completando a triangulação "estrutural + interpretável + cross-lingual".

Se Cenário A ou B, a contribuição central do TCC é: **detecção de fake news baseada exclusivamente em topologia é language-agnostic, oferecendo um caminho computacionalmente barato e independente de pré-treino multilíngue, em contraste com a abordagem mainstream baseada em mBERT/XLM-R (Pires 2019, Conneau 2020, Schwarz 2020)**.

Se Cenário C, o TCC reposiciona-se: **mesmo o classificador estrutural absorve viés cross-lingual via correlação espúria com comunidade/dataset, evidenciando que o problema de "atalho topológico" é mais profundo do que mero overfitting textual** — limitação relevante para deployment real e para a literatura de robustez.

---

## 7. Análise de Código

### 7.1 Erros / pontos de atenção identificados

**E1 — Quotes pode ser dict, não int (linha 88):**
```python
# ⚠️ Linha 88
if isinstance(post.get("quotes"), int):
    interacoes += post["quotes"]
```
Se `quotes` vier como `dict` (e.g., `{"count": 5, "items": [...]}` em alguma versão da API), o contador é silenciosamente ignorado. Defensivo, mas pode subestimar `grau_root` em datasets futuros.

```python
# ✅ Correção sugerida — extrair numericamente em ambos os casos
q = post.get("quotes")
if isinstance(q, int):
    interacoes += q
elif isinstance(q, dict) and isinstance(q.get("count"), int):
    interacoes += q["count"]
```

**E2 — Falta correção de múltiplas comparações:**
```python
# ⚠️ Linhas 159, 176 — três pares testados sem ajuste de α
pares = [("por", "eng"), ("deu", "eng"), ("por", "deu")]
```
Com α=0.05 e 3 testes por estatística (KS, MW), são 6 testes simultâneos. FWER ≈ 0.265.

```python
# ✅ Correção sugerida — Bonferroni ou Benjamini-Hochberg
from statsmodels.stats.multitest import multipletests
ks_ps = [r["KS_p"] for r in comp_rows]
_, ks_ps_corr, _, _ = multipletests(ks_ps, alpha=0.05, method="bonferroni")
# adicionar coluna ks_p_corr ao CSV
```

**E3 — Comparação por feed não implementada:**
o experimento confunde "idioma" com "comunidade" porque feeds têm composição lingüística distinta. Estratificar por feed isolaria o efeito puro de idioma. Não é um *bug* — é uma extensão valiosa para robustecer o argumento da Seção 6.4 do TCC.

**E4 — `predict_proba` sem aviso sobre calibração:**
o RF não está calibrado (não passou por Platt scaling nem isotonic regression). O "score" não é uma probabilidade calibrada — é uma fração de votos. Para o teste de **distribuição**, é irrelevante (qualquer transformação monotônica preserva os ranks de MW e os quantis de KS). Mas o relatório `relatorio.txt` deveria mencionar isso explicitamente para evitar interpretação equivocada por leitores externos.

**E5 — `idx_fake = 0` se 0 não está nas classes (linha 123):**
```python
idx_fake = classes.index(0) if 0 in classes else 0
```
Fallback silencioso para coluna 0 quando o RF foi treinado com classes diferentes — pode esconder bugs upstream. Melhor `raise ValueError("RF não tem classe 0 (fake)")`.

---

### 7.2 Ineficiências

**(I1) Loop Python sobre todos os posts em `carregar_posts_por_lang`:**
~166k posts × 3 campos JSON = ~500k acessos de dict. Não é gargalo na prática (segundos), mas usar `pandas.read_json(lines=True)` + filtro vetorizado seria 5–10× mais rápido se o dataset crescer.

**(I2) `predict_proba` chamado por idioma:**
três chamadas separadas com `n_eng + n_deu + n_por` posts no total. Concatenar tudo em uma única chamada e fatiar depois seria marginalmente mais rápido (RF paraleliza por amostra).

**(I3) Q-Q plot só compara dois pares (PT×EN, DE×EN), não PT×DE:**
o terceiro par está nas tabelas mas não nos gráficos — assimetria de visualização. Adicionar PT×DE completaria o quadro.

---

### 7.3 Boas práticas observadas

- **Filtro monolingue estrito** (`len(langs) == 1`) — purifica a partição.
- **Cohen's d implementado manualmente com `ddof=1` e guarda contra `pooled < 1e-12`** — robustez numérica correta.
- **Tripla validação estatística (KS + MW + d)** — cobertura ampla de modos de falha (formato, mediana, magnitude).
- **Q-Q plot ao lado de p-valores** — diagnóstico visual onde números agregados são ambíguos.
- **CSV por-post salvo** — viabiliza re-análise externa e correção de múltiplas comparações *post-hoc*.
- **Verificação de `n < 30` antes de testar** (linhas 178, 246) — respeita o limiar clássico para validade do teorema central do limite em MW assintótico.
- **Saída em pasta dedicada `figuras_tcc/rq3_multilingual`** — segue a convenção do pipeline (Fase 7), facilita consolidação no script `24_consolidar_tcc.py`.

---

## 8. Referências Bibliográficas

1. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`.

2. COHEN, J. **Statistical Power Analysis for the Behavioral Sciences**. 2ª ed. Hillsdale: Lawrence Erlbaum Associates, 1988. ISBN 0-8058-0283-5.

3. CONNEAU, A.; KHANDELWAL, K.; GOYAL, N.; CHAUDHARY, V.; WENZEK, G.; GUZMÁN, F.; GRAVE, E.; OTT, M.; ZETTLEMOYER, L.; STOYANOV, V. **Unsupervised Cross-lingual Representation Learning at Scale**. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 8440–8451, 2020. arXiv: `1911.02116`.

4. MANN, H. B.; WHITNEY, D. R. **On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other**. *The Annals of Mathematical Statistics*, v. 18, n. 1, pp. 50–60, 1947. DOI: `10.1214/aoms/1177730491`.

5. MASSEY, F. J. **The Kolmogorov-Smirnov Test for Goodness of Fit**. *Journal of the American Statistical Association*, v. 46, n. 253, pp. 68–78, 1951. DOI: `10.1080/01621459.1951.10500769`.

6. PIRES, T.; SCHLINGER, E.; GARRETTE, D. **How Multilingual is Multilingual BERT?**. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)*, pp. 4996–5001, 2019. arXiv: `1906.01502`.

7. SCHWARZ, S.; THEÓPHILO, A.; ROCHA, A. **EMET: Embeddings from Multilingual-Encoder Transformer for Fake News Detection**. *Proceedings of ICASSP 2020*, pp. 2777–2781, 2020. DOI: `10.1109/ICASSP40776.2020.9054673`.

8. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

9. DEMŠAR, J. **Statistical Comparisons of Classifiers over Multiple Data Sets**. *Journal of Machine Learning Research*, v. 7, pp. 1–30, 2006. (Referência para correção de múltiplas comparações; Bonferroni/Holm.) Disponível em: `https://jmlr.org/papers/v7/demsar06a.html`.

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Kolmogorov-Smirnov 2-sample (D) | Teste não-paramétrico para igualdade de distribuições; estatística é o sup da diferença entre CDFs empíricas | Massey (1951) |
| Mann-Whitney U | Teste não-paramétrico para deslocamento estocástico entre duas amostras; opera em ranks | Mann & Whitney (1947) |
| Cohen's d | Tamanho de efeito padronizado em unidades de desvio padrão pooled; |d|<0.2 = desprezível | Cohen (1988), Cap. 8 |
| Random Forest | Ensemble de árvores de decisão treinadas com bootstrap + amostragem de features | Breiman (2001) |
| `predict_proba` | Fração de árvores que classificam a amostra em cada classe (não calibrada por padrão) | Breiman (2001), Seção 6 |
| ISO 639-3 | Código de 3 letras para identificação de idiomas (`eng`, `deu`, `por`) — usado pelo AT Protocol | ISO/SIL Ethnologue |
| AT Protocol `langs` | Campo opcional do post Bluesky declarando idioma(s) — lista de códigos ISO 639-3 | docs.bsky.app |
| Language-agnostic | Propriedade de um classificador cuja saída é invariante a idioma da entrada | Pires et al. (2019) |
| Q-Q plot | Gráfico de quantis empíricos de duas amostras; pontos em y=x ⇒ distribuições iguais | Wilk & Gnanadesikan (1968) |
| H₀ (RQ3) | "Distribuição de score do RF é igual entre PT, DE, EN" | — |
| FWER | Family-Wise Error Rate — probabilidade de pelo menos um falso positivo em múltiplos testes | Demšar (2006) |
| Bonferroni | Correção conservadora: $\alpha' = \alpha / m$ com $m$ = nº de testes | Demšar (2006), Seção 4.1 |
| `ddof=1` | Delta degrees of freedom = 1 → variância amostral (divisor $n-1$, não $n$) | NumPy docs; Bessel correction |
| Pooled std | $s_p = \sqrt{((n_A-1)s_A^2 + (n_B-1)s_B^2)/(n_A+n_B-2)}$ — desvio padrão combinado para Cohen's d | Cohen (1988), Cap. 2 |
| Ties (empates) | Valores idênticos em ranks; corrigidos por SciPy via redução do denominador da variância | Mann & Whitney (1947), Seção 5 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 26_rq3_multilingual.py
Arquivo gerado: theory_andre/26_rq3_multilingual_doc.md
Fontes academicas utilizadas: 9
   1. Breiman (2001) — Random Forests, Machine Learning
   2. Cohen (1988) — Statistical Power Analysis (livro)
   3. Conneau et al. (2020) — XLM-R, ACL
   4. Mann & Whitney (1947) — Annals of Mathematical Statistics
   5. Massey (1951) — KS test, JASA
   6. Pires et al. (2019) — How Multilingual is mBERT, ACL
   7. Schwarz et al. (2020) — EMET, ICASSP
   8. Shu et al. (2020) — FakeNewsNet, Big Data
   9. Demsar (2006) — Statistical Comparisons, JMLR
Conceitos cobertos:
   - RandomForest probabilistico (predict_proba como fracao de votos)
   - Filtro monolingue via campo langs do AT Protocol (ISO 639-3)
   - Features estruturais (num_nodes, grau_root) language-agnostic by design
   - KS 2-sample (formato de distribuicao)
   - Mann-Whitney U (deslocamento de mediana, robusto a empates)
   - Cohen's d (tamanho de efeito independente de n)
   - Q-Q plot (diagnostico visual da invariancia)
   - 3 cenarios de resposta a RQ3 (H0 nao rejeitada / d<0.2 / d>=0.2)
   - Conexao com mBERT/XLM-R (alternativa estrutural a NLP multilingue)
Limitacoes:
   - Resultados numericos finais (KS-D, MW-p, d) nao reportados — depende de execucao do script
   - Confound idioma x feed nao isolado (extensao sugerida na Secao 6.2 L2)
   - Multiplas comparacoes sem correcao FWER (E2 na Secao 7.1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISAO — nao prosseguir para o proximo script.
```
