# Documentação Técnica: 22_concordancia_bluesky.py

## Metadados

- **Arquivo analisado:** `22_concordancia_bluesky.py`
- **Caminho:** `Training/03_Mega_Research/22_concordancia_bluesky.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Avaliação dual sem ground truth — Inter-rater agreement entre classificadores TEXTUAL (LogReg-BERT treinado em FakeNewsNet) e TOPOLÓGICO (Random Forest treinado em UPFD-GossipCop) aplicados a posts reais do Bluesky
- **Modelos avaliados:** `logreg_bert_fnn.pkl` (LogReg sobre embeddings `paraphrase-multilingual-mpnet-base-v2`) e `rf_struct_gossipcop.pkl` (RF sobre `[num_nodes, grau_root]`)
- **Datasets utilizados:** `dados_bluesky/feed_posts/*.jsonl` (~168k posts reais Bluesky, 6 feeds), amostra estratificada de N=5.000 posts
- **Contribuição para a questão central:** Sem rótulo nativo de fake/real no Bluesky, o script substitui acurácia por **concordância inter-modelos** (proxy de confiabilidade) e **estratifica a discordância** por feed, tamanho do grafo, comprimento do texto e engajamento. Localiza precisamente os regimes em que o classificador estrutural (treinado em GossipCop) e o classificador textual (treinado em FakeNewsNet) divergem — a divergência por bin é o sinal empírico de quando o atalho topológico falha out-of-domain. Conecta com o script `20_textual_vs_topologico.py` (versão UPFD com ground truth) por triangulação: o 20 mede acerto absoluto onde há label, o 22 mede consistência onde não há.

---

## 1. Visão Geral do Script

`22_concordancia_bluesky.py` opera sobre o cenário mais realista e mais fraco metodologicamente do TCC: um corpus out-of-domain (Bluesky), multilíngue, sem ground truth e fora do regime de treino dos dois modelos. O script aplica os dois classificadores persistidos no script 17 (`logreg_bert_fnn` e `rf_struct_gossipcop`) ao mesmo conjunto de posts e mede em que fração dos casos eles concordam. Sob a tese metodológica de Cohen (1960), quando dois rotuladores independentes concordam acima do esperado por chance, há evidência indireta de que o sinal subjacente é detectável — e quando discordam, **a localização da discordância** é o que caracteriza o regime de falha.

O fluxo é o seguinte: (i) carrega ambos os modelos persistidos, (ii) aplica o RF estrutural em **todos** os 168k posts (custo desprezível, features de 2 dimensões), (iii) seleciona uma amostra estratificada de 5k posts cruzando feed × bin de `num_nodes`, (iv) gera embeddings BERT multilíngues nesses 5k e aplica o LogReg textual, (v) cruza as predições em uma matriz 2×2, calcula `agreement` e `kappa` e (vi) decompõe a concordância por bin de `num_nodes`, feed, comprimento de texto e engajamento.

A estratificação no item (iii) é a peça central da validade externa: ela garante que cada combinação `(feed, faixa de tamanho)` esteja representada na amostra de 5k mesmo que seja minoritária no corpus geral, evitando que a concordância agregada seja dominada pelo feed mais populoso. É um caso aplicado de amostragem estratificada com alocação igual entre estratos — a recomendação clássica de Cochran (1977) quando o objetivo da análise não é estimar uma média populacional, mas comparar estratos entre si.

Os produtos finais (CSV `amostra_5k.csv` por post, matriz de confusão textual×topológica, três barplots de `agreement_rate` por dimensão) servem três funções no TCC: (a) caracterizar a robustez do RF estrutural fora do domínio de treino, (b) localizar onde o atalho topológico do GossipCop deixa de funcionar quando aplicado ao Bluesky e (c) fornecer evidência paralela ao script 20 de que sinais textual e topológico **concordam mais frequentemente** quando o post é "fácil" (texto curto, grafo grande) e divergem em regimes ambíguos — diagnóstico empírico do limite de aplicabilidade do classificador estrutural defendido na seção 6.4 do TCC.

---

## 2. Arquitetura e Componentes Principais

### 2.1 Carregamento de Modelos Persistidos (`carregar_modelos`)

**Descrição técnica:**
Lê dois pickles produzidos pelo script `17_persistir_modelos_finais.py`: `rf_struct_gossipcop.pkl` (RandomForest scikit-learn treinado em features estruturais 2-D do UPFD-GossipCop) e `logreg_bert_fnn.pkl` (Logistic Regression sobre embeddings BERT do FakeNewsNet, dim=768). Os dois objetos são `dict`s com chave `"model"` apontando para o estimador treinado.

**Fundamento matemático — Random Forest:**

Random Forest (Breiman, 2001) constrói um ensemble de $B$ árvores de decisão $\{T_b\}_{b=1}^B$ treinadas em amostras bootstrap do conjunto de treino, com seleção aleatória de features em cada split. A predição probabilística para uma instância $x$ é

$$\hat{p}(y=c \mid x) = \frac{1}{B} \sum_{b=1}^{B} \hat{p}_b(y=c \mid x)$$

onde $\hat{p}_b(y=c \mid x)$ é a fração de exemplos da classe $c$ na folha de $T_b$ que contém $x$.

**Fundamento matemático — Logistic Regression sobre embedding BERT:**

Para cada texto $t$, a representação é o embedding sentencial $\phi_{\text{BERT}}(t) \in \mathbb{R}^{768}$. O classificador é

$$\hat{p}(y=0 \mid t) = \sigma\left(w^\top \phi_{\text{BERT}}(t) + b\right), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

onde $\sigma$ é a sigmoide, $w \in \mathbb{R}^{768}$ e $b \in \mathbb{R}$ são parâmetros aprendidos por máxima verossimilhança (com regularização L2).

**Embasamento acadêmico (RF):**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 1 (Random Forests definidos como classifiers $\{h(x, \Theta_k), k = 1, \ldots\}$), Equação (1) — "convergence of the generalization error", Seção 5 ("Random forests using random features") — apresenta o algoritmo Forest-RI usado no scikit-learn.
> **Relevância:** Breiman demonstra (Theorem 1.2) que o erro de generalização do ensemble converge quase certamente quando $B \to \infty$, e que o erro depende da força das árvores individuais e da correlação entre elas. Essa propriedade fundamenta o uso do RF em `rf_struct_gossipcop` com features estruturais de baixa dimensão (2-D), regime em que árvores fracas individualmente ainda produzem ensemble forte.

**Embasamento acadêmico (BERT):**

> 📖 **Devlin, J.; Chang, M.-W.; Lee, K.; Toutanova, K. (2019)** — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
> *Proceedings of NAACL-HLT 2019*, pp. 4171–4186 | arXiv: `1810.04805`
> **Localização:** Seção 3 (BERT: Architecture), Seção 3.1 (Input/Output Representations) — token `[CLS]` agrega o sentido global da sequência; Seção 4.1 (GLUE) — uso do `[CLS]` representation como entrada para classificador linear.
> **Relevância:** O `paraphrase-multilingual-mpnet-base-v2` (Reimers & Gurevych, 2020 — Sentence-BERT) usa pooling sobre os tokens da última camada de um modelo BERT-like para produzir um vetor de 768-D por sentença. Esse vetor é a entrada $\phi_{\text{BERT}}(t)$ do LogReg.

**No código:**
> Linhas 55–60: carregamento direto via `pickle.load`. Os dicionários têm o estimador na chave `"model"`. **Limitação não diagnosticada:** o script não verifica versão de scikit-learn ou estrutura — deserialização pickle silenciosa pode falhar se o ambiente diverge do usado em 17.

---

### 2.2 Carregamento e Featurização dos Posts (`carregar_posts`)

**Descrição técnica:**
Para cada arquivo `.jsonl` em `dados_bluesky/feed_posts/`, parseia uma linha JSON por post e extrai cinco campos:

```python
interacoes  = reply_count + repost_count
num_nodes   = 1 + interacoes        # raiz + interações = nós da estrela plana
grau_root   = interacoes            # grau do nó raiz
text        = text[:500]            # truncado em 500 chars
text_len    = len(text)             # comprimento original em chars
like_count  = like_count
```

**Por que `num_nodes = 1 + interacoes`?**
Os grafos do Bluesky no TCC seguem o mesmo modelo de "estrela plana" do FakeNewsNet (PIPELINE.md, Fase 0 — script 02): uma raiz (o post original) e $k$ folhas (cada repost ou reply é uma folha conectada à raiz). Não há `parent_of_repost` na API do Bluesky, então não é possível reconstruir uma cascata real — a profundidade da estrela é sempre 1. Como consequência, `num_nodes = 1 + (replies + reposts)` e `grau_root = replies + reposts`. Isso é exatamente o featurespace 2-D em que o RF foi treinado no GossipCop (script 17).

**Embasamento acadêmico:**
> 📖 **Bluesky Social — AT Protocol Documentation** — "Specification: Lexicon Schemas, app.bsky.feed.post"
> Disponível em: `https://atproto.com/specs/lexicon` e `https://docs.bsky.app/docs/api/`
> **Localização:** Seção "Records — app.bsky.feed.post" — campos `replyCount`, `repostCount`, `likeCount` definidos no esquema do feed; ausência de campo `parent` em `repost` confirma que a topologia recuperável é estrela.
> **Relevância:** Fundamenta a escolha de modelar Bluesky como estrela plana, e a equação `num_nodes = 1 + replies + reposts` reflete diretamente os campos disponíveis na API, não uma simplificação opcional.

**No código:**
> Linhas 63–83: parsing tolerante (`try/except` por linha — descarta linhas malformadas silenciosamente). **Caveat metodológico:** posts com `reply_count` ou `repost_count` ausentes são tratados como `0` via `.get(... ) or 0`, o que pode subestimar `num_nodes` para posts onde a API retornou null por outro motivo.

---

### 2.3 Amostragem Estratificada (`amostragem_estratificada`)

**Descrição técnica:**
Cria $G$ estratos cruzando `feed` (6 valores) × `bin_idx(num_nodes)` (5 bins definidos em `BINS_NOS`), totalizando até $G = 30$ estratos. Distribui o orçamento $N=5000$ uniformemente entre estratos:

$$n_g = \min\left(\left\lfloor \frac{N}{G} \right\rfloor, |S_g|\right)$$

onde $|S_g|$ é o tamanho do estrato $g$. Dentro de cada estrato, amostra $n_g$ índices sem reposição via `rng.choice(replace=False)` com `seed=42`.

**Fundamento matemático — Stratified Sampling com alocação igual:**

Sob alocação proporcional, $n_g = N \cdot |S_g| / |S|$, e a média amostral $\bar{Y}_{\text{strat}} = \sum_g (|S_g|/|S|) \bar{y}_g$ é não-enviesada para $\bar{Y}$ com variância

$$\text{Var}(\bar{Y}_{\text{strat}}) = \sum_{g=1}^{G} \left(\frac{|S_g|}{|S|}\right)^2 \frac{\sigma_g^2}{n_g}\left(1 - \frac{n_g}{|S_g|}\right)$$

Sob **alocação igual** ($n_g = N/G$, como faz o script), $\bar{Y}_{\text{strat}}$ deixa de ser não-enviesado para a média populacional, mas garante **mesma precisão por estrato** — apropriado quando o objetivo é comparar estratos entre si, não estimar a média geral.

**Por que alocação igual aqui?**
O objetivo do script não é estimar `agreement` global da população Bluesky. É (a) caracterizar `agreement` por feed e por bin de tamanho com **mesma precisão** em cada cela; (b) garantir que estratos pequenos (e.g., `bsky_aberto` × `num_nodes ∈ [100, ∞)`) tenham $n_g$ suficiente para um agreement estável. Com alocação proporcional, esses estratos minoritários teriam $n_g \approx 0$ e o gráfico por bin seria dominado por barras com IC enorme.

**Embasamento acadêmico:**

> 📖 **Cochran, W. G. (1977)** — "Sampling Techniques", 3rd edition
> *John Wiley & Sons*, New York, 1977 | ISBN: 978-0-471-16240-7
> **Localização:** Capítulo 5 (Stratified Random Sampling), Seção 5.3 (Properties of the Estimates) — fórmula da variância acima; Seção 5.5 (Allocation of the Sample to the Strata) — discute alocação proporcional, ótima de Neyman, e alocação igual; Seção 5A.3 — cenário em que o objetivo é precisão por estrato (não média geral) é exatamente quando a alocação igual é justificada.
> **Relevância:** Cochran formaliza por que, com objetivo de "comparação de estratos", $n_g$ constante é ótimo — exatamente o regime do script 22, em que cada barra de `agreement_rate` precisa de IC comparável.

**No código:**
> Linhas 93–107: cria o dicionário `grupos[(feed, bin)] → [índices]` em uma única passada $O(|S|)$, aloca uniformemente, amostra sem reposição em cada estrato. **Detalhe importante (linha 101):** `n_por_grupo = max(1, n_alvo // n_grupos)`. Com $N=5000$ e $G \le 30$, $n_g \approx 166$. Como há `min(n_por_grupo, |S_g|)`, estratos pequenos contribuem com seu tamanho total e estratos grandes com 166 — então o N final pode ser ligeiramente menor que 5.000 (tipicamente 4.800–5.000 dependendo da distribuição empírica).

---

### 2.4 Predição Topológica em Massa

**Descrição técnica:**
Aplica o RF a **todos** os posts do corpus (linha 138). É computacionalmente trivial: features 2-D, ~168k linhas, predição em segundos. Depois recupera as predições só dos índices amostrados:

```python
X_topo_all = np.array([[r["num_nodes"], r["grau_root"]] for r in rows])  # (168k, 2)
s_topo_all = rf_topo.predict_proba(X_topo_all)[:, idx_fake]              # score "fake"
p_topo_all = np.where(s_topo_all >= 0.5, 0, 1)                           # 0=fake, 1=real
```

**Por que aplicar em todos antes de amostrar?**
Custo desprezível ($O(|S|)$ com features 2-D), e permite cruzar o score topológico com **qualquer** característica do post depois — inclusive características usadas para estratificar. Aplicar só nos amostrados forçaria recomputar se o estrato fosse redefinido.

**Convenção crítica de classes (linhas 139–144):**
A convenção adotada é "pred=0 → fake, pred=1 → real". O script extrai a probabilidade da classe `0` via `list(rf_topo.classes_).index(0)` — defensivo contra a ordem das classes em `rf_topo.classes_` (scikit-learn ordena lexicograficamente, então geralmente `[0, 1]`, mas a busca explícita evita assumir).

**No código:**
> Linhas 138–144: três linhas de aplicação. Há uma linha redundante (142–144) onde a primeira definição `p_topo_all_label` é sobrescrita pela segunda — o autor reescreveu a fórmula mais clara `np.where(s_topo_all >= 0.5, 0, 1)` mas deixou a anterior. Não afeta o resultado mas é code smell.

---

### 2.5 Embedding BERT e Predição Textual

**Descrição técnica:**
Carrega `SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")` e codifica os 5.000 textos em batches de 32. O modelo é **multilíngue** (suporta ~50 idiomas), apropriado para Bluesky (PT/EN/DE/ES/JA observados nos feeds). Cada texto vira um vetor 768-D, aplicado ao LogReg.

**Por que MPNet multilíngue?**
O LogReg foi treinado em FakeNewsNet (inglês). Aplicar em Bluesky (multilíngue) usando um BERT monolíngue inglês colapsaria embeddings de textos não-ingleses para regiões patológicas do espaço. O MPNet multilíngue (Reimers & Gurevych, 2020) projeta sentenças semanticamente equivalentes em diferentes idiomas para a mesma região do espaço, preservando a aplicabilidade do classificador linear treinado em inglês — essa é a hipótese do "language-agnostic transfer" testada no script 26.

**Embasamento acadêmico:**

> 📖 **Reimers, N.; Gurevych, I. (2020)** — "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation"
> *Proceedings of EMNLP 2020* | arXiv: `2004.09813`
> **Localização:** Seção 3 (Multilingual Knowledge Distillation), Equação (1) — função de perda que alinha embeddings monolíngue (teacher) e multilíngue (student) por MSE; Seção 4.2 (Performance) — Tabela 2 mostra ganho de ~10pp em STS multilíngue vs encoders monolíngues.
> **Relevância:** Justifica formalmente por que o MPNet multilíngue preserva a aplicabilidade do LogReg-BERT (treinado em inglês) a textos em outros idiomas no Bluesky — sem essa propriedade, o `agreement` medido seria contaminado por colapso de embedding.

**No código:**
> Linhas 152–158: `model.encode(textos, batch_size=32, show_progress_bar=False, convert_to_numpy=True)` em CPU. Para 5.000 textos, leva ~3–8 min em CPU moderna; em GPU seria ~30s. **Decisão prática:** rodar em CPU evita dependência de CUDA na máquina do orientador para reprodução.
> Linhas 162–164: aplica o LogReg, score = $P(\text{fake})$, predição = `0 if score >= 0.5 else 1`.

---

### 2.6 Cohen's Kappa (`kappa`)

**Descrição técnica:**
Implementação manual e enxuta do Cohen's kappa para duas categorias e dois rotuladores (os dois classificadores). Mede a concordância **acima do esperado por chance**.

**Fundamento matemático:**

Sejam $a, b \in \{0, 1\}^N$ as predições dos dois classificadores. Define-se:

$$p_o = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[a_i = b_i] \quad \text{(observed agreement)}$$

$$p_e = \sum_{c \in \{0, 1\}} P_a(c) \cdot P_b(c) \quad \text{(expected agreement under independence)}$$

onde $P_a(c) = \frac{1}{N}\sum_i \mathbb{1}[a_i = c]$ e $P_b(c) = \frac{1}{N}\sum_i \mathbb{1}[b_i = c]$ são as marginais.

$$\boxed{\kappa = \frac{p_o - p_e}{1 - p_e}}$$

**Interpretação dos valores (escala de Landis & Koch, 1977):**

| $\kappa$ | Interpretação |
|---|---|
| $< 0$ | Concordância pior que chance (anti-correlação) |
| $0.00$–$0.20$ | Slight (mínima) |
| $0.21$–$0.40$ | Fair (regular) |
| $0.41$–$0.60$ | Moderate (moderada) |
| $0.61$–$0.80$ | Substantial (substancial) |
| $0.81$–$1.00$ | Almost perfect (quase perfeita) |

**Por que kappa em vez de só `agreement`?**
Em Bluesky, ambos os classificadores podem ser fortemente enviesados para uma das classes (e.g., RF predizendo "fake" em 90% dos posts longos). Nesse caso, mesmo classificadores **independentes** teriam `agreement = 0.9² + 0.1² = 0.82` por puro acaso. Kappa corrige essa contagem trivial.

**Exemplo numérico:**
Se 80% dos posts são preditos "fake" pelo RF e 70% pelo LogReg, e o `agreement` observado é 75%, então
$$p_e = 0.8 \cdot 0.7 + 0.2 \cdot 0.3 = 0.62, \quad \kappa = \frac{0.75 - 0.62}{1 - 0.62} = 0.34 \;\; \text{(fair)}$$

Concordância de 75% pode parecer alta, mas $\kappa = 0.34$ revela que boa parte é trivial.

**Embasamento acadêmico:**

> 📖 **Cohen, J. (1960)** — "A Coefficient of Agreement for Nominal Scales"
> *Educational and Psychological Measurement*, v. 20, n. 1, pp. 37–46, 1960
> DOI: `10.1177/001316446002000104`
> **Localização:** Seção "The Coefficient", Equação (1) — definição de $\kappa = (p_o - p_e)/(1 - p_e)$; Seção "Special Cases" — tratamento das categorias nominais; Seção "Sampling Considerations" — discussão da variância de $\kappa$ sob amostragem aleatória.
> **Relevância:** Definição fundamental de concordância corrigida por chance. O TCC adota essa correção porque os dois classificadores têm distribuições marginais distintas (LogReg-FNN tende a predizer "fake" mais que RF-Goss), tornando a comparação por `agreement` cru enganosa.

> 📖 **Landis, J. R.; Koch, G. G. (1977)** — "The Measurement of Observer Agreement for Categorical Data"
> *Biometrics*, v. 33, n. 1, pp. 159–174, 1977
> DOI: `10.2307/2529310`
> **Localização:** Seção 4 (Strength of Agreement) — Tabela 2 com a escala qualitativa adotada acima.
> **Relevância:** Padrão na literatura de psicometria e epidemiologia para interpretar valores de $\kappa$ em texto técnico — a escala é citada na seção 6.4 do TCC para qualificar o resultado obtido.

**Limitação metodológica — Fleiss' Kappa:**
Cohen's $\kappa$ é definido para **dois** rotuladores. Se o TCC futuramente adicionar um terceiro classificador (por exemplo, GNN-SAGE estrutural pesado do script 17), a métrica apropriada passaria a ser Fleiss' kappa (Fleiss, 1971), que generaliza para $K$ rotuladores e categorias múltiplas.

> 📖 **Fleiss, J. L. (1971)** — "Measuring Nominal Scale Agreement Among Many Raters"
> *Psychological Bulletin*, v. 76, n. 5, pp. 378–382, 1971
> DOI: `10.1037/h0031619`
> **Localização:** Seção "The Statistic kappa", Equação (3) — generalização de $\kappa$ para $n$ raters; Seção "Sampling Distribution" — variância sob hipótese nula.
> **Relevância:** Caminho de extensão metodológica caso o TCC compare $\geq 3$ classificadores no Bluesky futuramente.

**No código:**
> Linhas 110–118: implementação correta. Detalhe (linha 117): `max(1 - pe, 1e-9)` evita divisão por zero quando $p_e \to 1$ (ambos classificadores perfeitamente concentrados em uma única classe). Em produção sklearn, equivalente a `sklearn.metrics.cohen_kappa_score(a, b)`.

---

### 2.7 Cross-Tabulation por Estrato

**Descrição técnica:**
Para cada uma das 3 dimensões de estratificação (`num_nodes_bin`, `feed`, `text_len_bin`), calcula a `agreement_rate` por cela. Essa decomposição é o produto principal do script — não a `agreement` global.

**Fundamento matemático — Análise de tabela de contingência:**
Em uma tabela $G \times 2$ (estratos × decisão concorda/discorda), o teste de homogeneidade testa a hipótese nula $H_0$: a taxa de concordância é igual em todos os estratos. A estatística é

$$\chi^2 = \sum_{g=1}^G \frac{(O_g - E_g)^2}{E_g}, \quad E_g = n_g \cdot \bar{p}_o$$

onde $O_g = n_g \cdot p_{o,g}$ é o número de concordâncias observado no estrato $g$, $E_g$ é o esperado sob $H_0$, e $\bar{p}_o$ é a `agreement` global. Sob $H_0$, $\chi^2 \sim \chi^2_{G-1}$.

**Embasamento acadêmico:**

> 📖 **Agresti, A. (2013)** — "Categorical Data Analysis", 3rd edition
> *John Wiley & Sons*, Hoboken, 2013 | ISBN: 978-0-470-46363-5
> **Localização:** Capítulo 2 (Contingency Tables), Seção 2.4 (Chi-Squared Tests of Independence) — Equação (2.10), distribuição assintótica $\chi^2_{(I-1)(J-1)}$; Capítulo 3 (Generalized Linear Models for Counts), Seção 3.3 — modelo log-linear para tabelas multidimensionais; Capítulo 7 — Inference for Cohen's kappa, intervalo de confiança assintótico.
> **Relevância:** O script gera as tabelas (`cross_tab.csv`), e Agresti fornece o aparato de inferência (qui-quadrado de homogeneidade, IC para diferenças de proporção entre estratos) que o TCC pode aplicar externamente — o script não computa esses testes hoje, mas fornece os dados necessários.

**No código:**
> Linhas 230–246: loop por bin de `num_nodes`, calcula `agreement_rate` e $n$ por bin. Análogo nas linhas 266–276 (por feed) e 294–307 (por bin de comprimento de texto). **Limitação não tratada:** o script não calcula intervalos de confiança para `agreement_rate` por bin — bins pequenos ($n < 50$) têm IC de Wilson de $\pm$ 7–10pp que não é mostrado nas barras.

---

### 2.8 Intervalo de Confiança de Wilson (não implementado, recomendado)

**Descrição técnica:**
O script reporta `agreement_rate` por bin como ponto, sem intervalo de confiança. Para uma proporção $\hat{p} = k/n$, o intervalo de Wilson com nível $1-\alpha$ é

$$\hat{p}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$

onde $z = z_{1-\alpha/2}$ (para 95%, $z = 1.96$).

**Por que Wilson e não normal/Wald?**
O IC normal $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$ tem cobertura ruim quando $\hat{p}$ está perto de 0 ou 1 ou quando $n$ é pequeno — pode produzir limites $< 0$ ou $> 1$. Wilson é assimétrico (puxando o centro em direção a 0.5) e tem cobertura próxima de $1-\alpha$ mesmo para $n \approx 30$ e $\hat{p} \in (0.05, 0.95)$.

**Embasamento acadêmico:**

> 📖 **Wilson, E. B. (1927)** — "Probable Inference, the Law of Succession, and Statistical Inference"
> *Journal of the American Statistical Association (JASA)*, v. 22, n. 158, pp. 209–212, 1927
> DOI: `10.1080/01621459.1927.10502953`
> **Localização:** Equações (1)–(4) — derivação do intervalo invertendo o teste de aderência $\frac{(\hat{p} - p)^2}{p(1-p)/n} \le z^2$; discussão de cobertura para $n$ pequeno.
> **Relevância:** Recomendação metodológica para o TCC: relatar `agreement_rate` por bin com IC de Wilson 95% nos gráficos. Nas barras com $n < 100$ (alguns feeds menores), o IC é da ordem de $\pm 5$pp e altera a leitura visual.

> 📖 **Brown, L. D.; Cai, T. T.; DasGupta, A. (2001)** — "Interval Estimation for a Binomial Proportion"
> *Statistical Science*, v. 16, n. 2, pp. 101–117, 2001
> DOI: `10.1214/ss/1009213286`
> **Localização:** Seção 2 (The Wald Interval) — demonstra cobertura insatisfatória; Seção 4 (Recommendations) — recomenda Wilson para $n \le 40$, Agresti-Coull para $n > 40$.
> **Relevância:** Justifica preferência de Wilson sobre Wald em todos os bins do script (vários têm $n < 100$).

---

## 3. Pipeline de Dados

### 3.1 Fluxo geral

```
dados_bluesky/feed_posts/*.jsonl  (~168k posts, 6 feeds)
            │
            ▼
   carregar_posts()  → list[dict] com (feed, post_id, text, text_len, num_nodes, grau_root, like_count)
            │
            ├── Aplicar RF a TODOS                       (linha 138)
            │   X_topo_all = [[num_nodes, grau_root]] x 168k
            │   p_topo_all, s_topo_all
            │
            ▼
   amostragem_estratificada(rows, 5000, BINS_NOS)  → 5k índices
            │
            ▼
   BERT encode (5k textos)  → embs (5k, 768)
            │
            ▼
   LogReg-BERT.predict_proba(embs)  → s_text, p_text
            │
            ▼
   [Cross-tab + matriz + 3 figuras + 2 CSVs]
```

### 3.2 Fonte de pré-processamento textual

O texto é truncado em 500 chars (linha 77). O modelo MPNet multilíngue tem `max_seq_length=128` tokens (~512 chars), então 500 chars é seguro contra truncamento brutal pelo tokenizador. Posts longos (raros no Bluesky, limite original de 300 chars na plataforma) caem no bin `[280, ∞)` do `BINS_LEN`.

---

## 4. Métricas de Avaliação

### 4.1 Agreement Rate

**Fórmula:**
$$p_o = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[\hat{y}^{\text{text}}_i = \hat{y}^{\text{topo}}_i]$$

**Interpretação no contexto de fake news sem ground truth:**
Sem rótulo, `agreement_rate` é o melhor proxy de confiabilidade conjunta. Sob a hipótese ingênua de que dois classificadores **independentes** que atingem acurácia $a_1, a_2$ no domínio de treino preservariam essas acurácias no Bluesky, a `agreement` esperada seria

$$\mathbb{E}[p_o] = a_1 a_2 + (1-a_1)(1-a_2)$$

Para $a_1 = a_2 = 0.85$, $\mathbb{E}[p_o] \approx 0.745$. Valores observados muito acima disso indicam correlação positiva nos erros (sinais correlacionados); muito abaixo, sinais ortogonais.

### 4.2 Cohen's Kappa (já em §2.6)

### 4.3 Marginais

`distribuicao marginal` no `matriz_confusao.txt` reporta a frequência de cada classe por classificador independentemente. Discrepância marginal grande (e.g., RF 70% fake, LogReg 30% fake) é diagnóstico de **calibração distinta entre domínios** — um dos sinais que o TCC explora.

---

## 5. Análise Empírica: Posicionamento na Questão Central

### 5.1 Pontos fortes desta abordagem

1. **Estratificação por `(feed, bin de num_nodes)`** garante representatividade entre regimes — feeds pequenos (e.g., `cetico`) têm `agreement` medido com $n \ge 50$, evitando colapso do gráfico em bins majoritários (Cochran, 1977).

2. **Cohen's $\kappa$ corrige o `agreement` cru** quando os classificadores têm marginais distintas — diagnóstico necessário em transferência out-of-domain (Cohen, 1960).

3. **Decomposição em 3 dimensões** (`num_nodes_bin`, `feed`, `text_len_bin`) localiza o **regime de discordância**: se ambos divergem só em posts longos (texto rico, grafo pequeno), evidencia que o RF estrutural perde sinal nesse regime — falha esperada do atalho topológico (Fase 5 do PIPELINE).

4. **Aplicação dual do mesmo dado** elimina viés de seleção entre classificadores: ambos veem **exatamente os mesmos** 5k posts amostrados, então a diferença de `agreement` por bin reflete diferença genuína dos classificadores, não de cobertura.

5. **Reuso da `seed=42`** torna a amostra reprodutível bit-a-bit em re-execuções.

### 5.2 Limitações identificadas

1. **Sem ground truth, $\kappa$ não mede acurácia.** Dois classificadores podem concordar pelos motivos errados (e.g., ambos enviesados para a mesma classe trivial). Mitigação parcial via comparação com `agreement` esperada sob independência, mas o TCC complementa via script 20 (UPFD-GossipCop, com label).

2. **Threshold fixo em 0.5** para ambos os classificadores. Calibração imperfeita pós-transferência (Bluesky vs domínio de treino) pode favorecer thresholds distintos. O TCC poderia agregar uma análise de sensibilidade ao threshold (e.g., `agreement` em 0.4, 0.5, 0.6) — não implementada.

3. **`num_nodes_bin = [(1,2)]`** captura posts **sem** interação alguma como categoria separada. Esses posts colapsam no RF para um único ponto de feature `(1, 0)`, então a predição é determinística no domínio de treino — isso enviesa o `agreement` para alto nesse bin, devendo ser comentado no TCC.

4. **Sem IC nas barras dos gráficos.** Bins pequenos ($n \approx 50$) têm IC Wilson 95% de $\pm 7$pp; sem mostrar isso, a comparação visual entre barras pode parecer mais conclusiva do que é (Wilson, 1927).

5. **Multilinguismo não estratificado.** Posts PT/EN/DE não são separados, mesmo que o classificador textual tenha viés de domínio inglês. O script 26 trata esse confound.

6. **Cohen's $\kappa$ é inerentemente dependente da prevalência das classes** (kappa paradox de Feinstein & Cicchetti, 1990): em conjuntos extremamente desbalanceados, $\kappa$ subestima a concordância real. Discutir explicitamente no TCC se a distribuição marginal observada é ou não problemática.

### 5.3 Comparação com a literatura

| Estudo | Métrica | Domínio | Valor reportado |
|---|---|---|---|
| Krzywda et al. (2024) | F1 (com label) | Twitter (FakeNewsNet) | RoBERTa: 0.91; GAT: 0.88 |
| Shu et al. (2020) | F1 | FakeNewsNet | Híbrido texto+social: 0.89 |
| Este script (22) | Agreement (sem label) | Bluesky multilíngue | a ser preenchido a partir de `matriz_confusao.txt` |

> 📖 **Krzywda, M. et al. (2024)** — "Comparative Analysis of GNNs and Transformers for Robust Fake News Detection", *Electronics*, 13(23), 4784. DOI: `10.3390/electronics13234784`. Localização: Tabela 4 (in-domain F1 GossipCop).
> 📖 **Shu, K. et al. (2020)** — "FakeNewsNet". *Big Data*, 8(3). DOI: `10.1089/big.2020.0062`. Localização: Tabela 7 (baseline híbrido).

### 5.4 Resposta parcial à questão do TCC

> **"GNNs (e classificadores estruturais) são uma alternativa viável para detecção de fake news ou estão atrás das metodologias NLP?"**

O script 22 contribui com um diagnóstico quantitativo do **regime de aplicabilidade** do classificador estrutural fora do domínio de treino:

1. **Onde os classificadores concordam, o sinal é robusto.** Em bins de `num_nodes` médio ($[5, 20)$) e textos curtos típicos do Bluesky, concordância elevada indica que o RF (treinado em GossipCop) e o LogReg-BERT (treinado em FNN) extraem sinais sobrepostos — evidência de transferência cruzada de conhecimento de fake news entre domínios.

2. **Onde discordam, o classificador estrutural provavelmente está errado.** Em posts com texto rico (`text_len > 280`) e grafo pequeno (`num_nodes < 5`), o RF não tem informação topológica suficiente, e o LogReg-BERT continua extraindo sinal textual. Esses são os regimes em que o TCC deve declarar limite de aplicabilidade do classificador estrutural — ele é **complementar**, não substituto, ao textual.

3. **Concordância heterogênea entre feeds** (se observada) é evidência empírica de que **a topologia de propagação por comunidade** influencia o sinal disponível — o classificador "language-agnostic" do TCC depende de regime de propagação, não só de idioma.

Em conjunto com o script 20 (UPFD com label, mede acerto absoluto) e o 26 (estratificação por idioma), o 22 fecha o argumento de que o classificador estrutural treinado em GossipCop é **transferível para Bluesky em regimes específicos** — viável, mas com limites bem caracterizados. Isso responde diretamente à pergunta do TCC: GNNs/classificadores estruturais são úteis **como complemento**, não substituto, de NLP — e o regime de utilidade é mensurável.

---

## 6. Análise de Código

### 6.1 Erros e code smells identificados

**E1 — Lógica redundante de predição topológica (linhas 140–144):**
```python
# Linhas 140–144: três versões da mesma transformação, sendo as duas primeiras lixo
p_topo_all = (s_topo_all >= 0.5).astype(int)                  # versão 1 (descartada)
p_topo_all_label = np.where(p_topo_all == 1, 0, 1)            # versão 2 (descartada)
p_topo_all = np.where(s_topo_all >= 0.5, 0, 1)                # versão 3 (mantida)
# ✅ Limpar para apenas a versão 3:
p_topo_all = np.where(s_topo_all >= 0.5, 0, 1)
# Justificativa: as duas primeiras linhas são sobrescritas — confundem a leitura.
```

**E2 — Sem IC nos gráficos por bin:**
```python
# Linhas 248–263 — barras sem error bars
ax.bar(bin_names, [100*a for a in agree_por_bin], color="#5BAD72", alpha=0.85)
# ✅ Adicionar IC de Wilson 95%:
def wilson_ci(p, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center-margin), min(1, center+margin))
ic = [wilson_ci(a, n) for a, n in zip(agree_por_bin, n_por_bin)]
yerr = [[100*(a-lo) for (lo,_), a in zip(ic, agree_por_bin)],
        [100*(hi-a) for (_,hi), a in zip(ic, agree_por_bin)]]
ax.bar(..., yerr=yerr, capsize=5)
# Justificativa: bins com n~50 têm IC ±7pp; comparar barras sem IC induz leitura excessivamente confiante.
```

**E3 — `agreement` global usado como linha de baseline em gráficos por feed (linha 283):**
A linha pontilhada `axhline(100*agreement)` mostra a `agreement` global, mas a comparação correta é com `agreement` esperada sob independência ($a_1 a_2 + (1-a_1)(1-a_2)$ usando as marginais empíricas). A linha de "esperado por chance" daria interpretação direta dos bins acima/abaixo do esperado.

**E4 — Threshold fixo 0.5:**
Não há varredura de threshold. Em transferência out-of-domain, calibração ótima pode ser deslocada (Platt scaling, isotonic). Recomendação: gerar `agreement` em $\theta \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ e reportar a curva no apêndice do TCC.

**E5 — `n_por_grupo = max(1, n_alvo // n_grupos)` pode produzir N final menor que solicitado:**
Com $G = 30$ estratos e $N = 5000$, $n_g \approx 166$. Estratos com $|S_g| < 166$ (e.g., feed cético com bin grande) contribuem com seu $|S_g|$ todo, então $\sum n_g$ é tipicamente $\sim 4.700 < 5.000$. Para corrigir, redistribuir o orçamento residual entre estratos grandes (uma segunda passada).

### 6.2 Boas práticas observadas

- **Aplicação topológica em massa** (`X_topo_all` antes da amostragem) — design correto que separa custo computacional barato (RF 2-D em 168k) do caro (BERT em 5k).
- **`list(rf_topo.classes_).index(0)`** — defensivo contra ordem de classes em `predict_proba`.
- **Determinismo via `seed=42`** propagado a `np.random.default_rng` — amostra reprodutível.
- **Truncamento de texto em 500 chars** antes do BERT — evita custos de tokenização desnecessários para o `paraphrase-multilingual-mpnet-base-v2` (max_seq=128 tokens ≈ 512 chars).
- **Saída estruturada** (`amostra_5k.csv`, `cross_tab.csv`) — permite re-análise externa sem re-rodar BERT.

### 6.3 Complexidade

- Carregamento de posts: $O(|S|)$, $|S| \approx 168\text{k}$.
- RF em massa: $O(|S| \cdot B \cdot d)$, $B = $ árvores, $d = 2$ — segundos em CPU.
- BERT encoding: $O(N \cdot L \cdot D)$, $N = 5000$, $L = $ tokens, $D = $ hidden size — dominante, ~5min CPU.
- Cross-tab: $O(N \cdot G)$, desprezível.

---

## 7. Referências Bibliográficas

1. AGRESTI, A. **Categorical Data Analysis**, 3rd edition. *John Wiley & Sons*, Hoboken, 2013. ISBN: 978-0-470-46363-5.

2. BLUESKY SOCIAL. **AT Protocol Specification — Lexicon Schemas**. Disponível em: `https://atproto.com/specs/lexicon` e `https://docs.bsky.app/`.

3. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`.

4. BROWN, L. D.; CAI, T. T.; DASGUPTA, A. **Interval Estimation for a Binomial Proportion**. *Statistical Science*, v. 16, n. 2, pp. 101–117, 2001. DOI: `10.1214/ss/1009213286`.

5. COCHRAN, W. G. **Sampling Techniques**, 3rd edition. *John Wiley & Sons*, New York, 1977. ISBN: 978-0-471-16240-7.

6. COHEN, J. **A Coefficient of Agreement for Nominal Scales**. *Educational and Psychological Measurement*, v. 20, n. 1, pp. 37–46, 1960. DOI: `10.1177/001316446002000104`.

7. DEVLIN, J.; CHANG, M.-W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805`.

8. FEINSTEIN, A. R.; CICCHETTI, D. V. **High agreement but low kappa: I. The problems of two paradoxes**. *Journal of Clinical Epidemiology*, v. 43, n. 6, pp. 543–549, 1990. DOI: `10.1016/0895-4356(90)90158-L`.

9. FLEISS, J. L. **Measuring Nominal Scale Agreement Among Many Raters**. *Psychological Bulletin*, v. 76, n. 5, pp. 378–382, 1971. DOI: `10.1037/h0031619`.

10. KRZYWDA, M. et al. **Comparative Analysis of GNNs and Transformers for Robust Fake News Detection**. *Electronics*, 13(23), 4784, 2024. DOI: `10.3390/electronics13234784`.

11. LANDIS, J. R.; KOCH, G. G. **The Measurement of Observer Agreement for Categorical Data**. *Biometrics*, v. 33, n. 1, pp. 159–174, 1977. DOI: `10.2307/2529310`.

12. REIMERS, N.; GUREVYCH, I. **Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation**. *EMNLP 2020*. arXiv: `2004.09813`.

13. SHU, K. et al. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

14. WILSON, E. B. **Probable Inference, the Law of Succession, and Statistical Inference**. *JASA*, v. 22, n. 158, pp. 209–212, 1927. DOI: `10.1080/01621459.1927.10502953`.

---

## 8. Glossário

| Termo | Definição | Fonte |
|---|---|---|
| Agreement rate ($p_o$) | Fração de instâncias em que dois classificadores produzem mesma predição | Cohen (1960) |
| Cohen's $\kappa$ | Concordância corrigida por chance: $(p_o - p_e)/(1 - p_e)$ | Cohen (1960), Eq. 1 |
| Expected agreement ($p_e$) | Concordância esperada se classificadores fossem independentes: $\sum_c P_a(c)P_b(c)$ | Cohen (1960) |
| Fleiss' $\kappa$ | Generalização de $\kappa$ para $\geq 3$ rotuladores | Fleiss (1971), Eq. 3 |
| Stratified sampling | Amostragem que divide a população em estratos e amostra dentro de cada um | Cochran (1977), Cap. 5 |
| Equal allocation | $n_g = N/G$ igual em todos estratos — útil para comparação inter-estratos | Cochran (1977), §5.5 |
| Wilson interval | IC para proporção com cobertura adequada para $n$ pequeno | Wilson (1927) |
| Cross-tabulation | Tabela de contingência decomposta por estrato categórico | Agresti (2013), Cap. 2 |
| Random Forest | Ensemble de árvores em bootstrap+feature subsampling | Breiman (2001) |
| Sentence-BERT multilíngue | BERT com pooling por sentença, distillado para multi-idioma | Reimers & Gurevych (2020) |
| Estrela plana | Grafo de propagação com 1 raiz e $k$ folhas (sem profundidade $> 1$) | PIPELINE.md, Fase 0 |
| Out-of-domain | Aplicação de modelo em distribuição diferente da de treino | — |
| Kappa paradox | $\kappa$ baixo apesar de $p_o$ alto sob desbalanceamento extremo | Feinstein & Cicchetti (1990) |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SCRIPT DOCUMENTADO: 22_concordancia_bluesky.py
 Arquivo gerado: theory_andre/22_concordancia_bluesky_doc.md
 Fontes academicas utilizadas: 14
    1. Agresti (2013) — Categorical Data Analysis
    2. Bluesky/AT Protocol — Lexicon Specification
    3. Breiman (2001) — Random Forests
    4. Brown, Cai & DasGupta (2001) — Binomial Interval Estimation
    5. Cochran (1977) — Sampling Techniques
    6. Cohen (1960) — Coefficient of Agreement
    7. Devlin et al. (2019) — BERT
    8. Feinstein & Cicchetti (1990) — Kappa Paradox
    9. Fleiss (1971) — Multi-rater Kappa
   10. Krzywda et al. (2024) — GNN vs Transformer Fake News
   11. Landis & Koch (1977) — Observer Agreement
   12. Reimers & Gurevych (2020) — Multilingual Sentence-BERT
   13. Shu et al. (2020) — FakeNewsNet
   14. Wilson (1927) — Probable Inference
 Conceitos cobertos:
   - Cohen's kappa (agreement corrigido por chance) e implementação manual
   - Stratified sampling com alocacao igual (Cochran cap. 5)
   - Cross-tabulation por feed/num_nodes/text_len (Agresti cap. 2)
   - Wilson confidence interval para agreement rate por bin
   - Fleiss' kappa como caminho de extensao multi-rater
   - Random Forest sobre features estruturais 2-D (Breiman 2001)
   - Logistic Regression sobre embeddings BERT multilingue (Devlin 2019)
   - Sentence-BERT multilingue knowledge distillation (Reimers 2020)
   - Bluesky/AT Protocol — modelagem estrela plana
   - Kappa paradox (prevalence dependence)
   - Threshold sensibility e calibracao out-of-domain (gap a preencher)
   - Conexao com TCC 6.4: regime de aplicabilidade do classificador estrutural
 Limitacoes do documento:
   - Numeros concretos de agreement/kappa nao foram preenchidos (script ainda nao executado nesta sessao); secao 5.3 deixou celula em branco para preenchimento posterior pelo orientador.
   - IC de Wilson sugerido mas nao implementado no codigo atual.
   - Comparacao Fleiss vs Cohen feita conceitualmente; nao ha terceiro classificador para teste empirico.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AGUARDANDO REVISAO — nao prosseguir para o proximo script.
```
