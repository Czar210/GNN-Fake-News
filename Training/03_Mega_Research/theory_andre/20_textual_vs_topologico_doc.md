# Documentação Técnica: 20_textual_vs_topologico.py

## Metadados

- **Arquivo analisado:** `20_textual_vs_topologico.py`
- **Caminho:** `03_Mega_Research/20_textual_vs_topologico.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Híbrido — análise de concordância entre dois classificadores heterogêneos (NLP textual + Random Forest topológico)
- **Modelos avaliados:** `LogisticRegression(BERT-768)` (textual) · `RandomForestClassifier([num_nodes, grau_root])` (topológico)
- **Datasets utilizados:** UPFD-GossipCop (com labels, com `feature="content"`) e Bluesky `feed_posts/*.jsonl` (sem ground truth)
- **Contribuição para a questão central:** Este script transforma a discussão "GNN vs NLP" em **um experimento de concordância** entre os dois paradigmas. Em vez de comparar F1 médios, mede *onde* texto e topologia veem a mesma realidade e *onde* divergem. As regiões de discordância — onde um classificador acerta e o outro erra — são justamente o terreno em que o TCC pode argumentar sobre o **domínio de competência** de cada paradigma e fundamentar a "regra dual" (Fase 8): combinar ambos em produção. É a peça que conecta o resultado negativo do baseline textual (script 10) com a vulnerabilidade topológica (script 14) sob uma única lente — a do meta-classificador implícito.

---

## 1. Visão Geral do Script

`20_textual_vs_topologico.py` executa um protocolo em quatro etapas sobre dois ambientes empíricos complementares:

1. **Etapa supervisionada (UPFD-GossipCop):** treina dois classificadores heterogêneos no `train+val`: `LogisticRegression` sobre `g.x[0]` (BERT-768) e `RandomForestClassifier` sobre `[num_nodes, grau_root]`. Avalia ambos no teste, computa F1-macro, matriz 2×2 de concordância de predições e Cohen $\kappa$.

2. **Análise de discordância:** para `pred_text != pred_topo`, mede F1 de cada classificador *condicional ao desacordo* — qual paradigma tem razão quando divergem. Os 15 casos com maior $|s_{\text{text}} - s_{\text{topo}}|$ são listados em `casos_discordancia.txt`.

3. **Visualização:** scatter $(s_{\text{text}}, s_{\text{topo}}) \in [0,1]^2$ colorido pelo rótulo real, diagonal $y=x$ como referência de concordância perfeita; os cortes em $0{,}5$ codificam as quatro células da matriz.

4. **Etapa não-supervisionada (Bluesky):** aplica o RF treinado no GossipCop sobre 168k posts Bluesky usando $[1+\text{interações}, \text{interações}]$ como surrogate de $[\text{num\_nodes}, \text{grau\_root}]$. Não treina LogReg textual no Bluesky (custo de gerar BERT em 168k posts); reporta só a distribuição de scores topológicos por feed, deferindo a comparação plena ao script 22.

A escolha UPFD-Goss + Bluesky é deliberada: UPFD-Goss tem rótulos (mede **acerto absoluto**), Bluesky não (mede só **consistência**). Sob Madras et al. 2018, duas funções de decisão independentes que concordam em $x$ formam evidência mais forte que cada uma isolada.

---

## 2. Inter-Rater Agreement: Cohen's $\kappa$

### 2.1 Por que não basta o `agreement_rate`

O cálculo trivial `concorda / n` (linha 87) sobrestima a concordância quando as classes são desbalanceadas. Se ambos os classificadores predizem a classe majoritária 90% das vezes, eles concordarão $\approx 0{,}82$ apenas por acaso, sem qualquer informação compartilhada genuína. Cohen (1960) propôs corrigir essa concordância pelo *chance agreement*:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

onde:
- $p_o = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\hat{y}^{\text{text}}_i = \hat{y}^{\text{topo}}_i]$ é a concordância observada (`agreement_rate` do código).
- $p_e = \sum_{k \in \{0,1\}} \pi^{\text{text}}_k \cdot \pi^{\text{topo}}_k$ é a concordância esperada por acaso, com $\pi^{(\cdot)}_k$ a frequência marginal da classe $k$ por classificador.

Interpretação canônica de Landis & Koch (1977): $\kappa < 0{,}20$ pobre, $0{,}21$–$0{,}40$ razoável, $0{,}41$–$0{,}60$ moderado, $0{,}61$–$0{,}80$ substancial, $\kappa > 0{,}80$ quase perfeito. Para o problema deste TCC, espera-se $\kappa$ moderado: os dois classificadores enxergam fenômenos distintos (semântica vs propagação) e por isso devem concordar acima do acaso, mas não devem ser equivalentes — caso contrário o paradigma topológico seria redundante.

**Embasamento acadêmico:**

> 📖 **Cohen, J. (1960)** — "A Coefficient of Agreement for Nominal Scales"
> *Educational and Psychological Measurement*, v. 20, n. 1, pp. 37–46
> DOI: `10.1177/001316446002000104`
> **Localização:** Seção "The Coefficient of Agreement" (pp. 39–41) — derivação de $\kappa$ como correção do *chance agreement*; Equação 1 (definição) e discussão sobre o limite superior $\kappa = 1$ (concordância perfeita) e $\kappa = 0$ (concordância igual ao acaso).
> **Relevância:** Referência canônica que justifica o uso de `cohen_kappa_score` (linha 89) em vez do simples `agreement_rate`. Para o TCC, $\kappa$ é a métrica que separa "os dois modelos veem a mesma coisa" (alta concordância informacional) de "os dois modelos chutam parecido" (alta concordância marginal).

> 📖 **Landis, J. R.; Koch, G. G. (1977)** — "The Measurement of Observer Agreement for Categorical Data"
> *Biometrics*, v. 33, n. 1, pp. 159–174
> DOI: `10.2307/2529310`
> **Localização:** Tabela 2 (p. 165) — escala interpretativa de $\kappa$ (poor / fair / moderate / substantial / almost perfect).
> **Relevância:** Fornece a régua de leitura empírica do $\kappa$ reportado em `gossipcop_metricas.txt`.

**No código:**
> Linha 89: `"kappa": round(float(cohen_kappa_score(p_text, p_topo)), 4)` — `cohen_kappa_score` do scikit-learn implementa a Equação 1 de Cohen (1960) com pesos uniformes (caso "nominal") apropriados para classificação binária com classes não ordinais.

---

### 2.2 Limitação do $\kappa$ binário

Em problemas binários muito desbalanceados, o $\kappa$ sofre do "paradoxo de Cohen" (Feinstein & Cicchetti, 1990): mesmo com $p_o$ alto, $\kappa$ pode despencar para próximo de zero quando uma das classes domina. Como o GossipCop tem distribuição moderadamente equilibrada nos splits oficiais, esse paradoxo não compromete a leitura aqui — mas é uma ressalva relevante quando este protocolo for transposto para o Bluesky (script 22), onde a heurística de rotulagem produz classes muito desbalanceadas.

---

## 3. Confusion / Concordance Matrix 2×2

### 3.1 Estrutura da matriz

A matriz `info["matriz_2x2"]` (linhas 80–82, 88) é uma **matriz de concordância** — uma confusion matrix entre as predições de dois classificadores, e não entre predição e ground truth. Sua estrutura é:

$$M_{ij} = \sum_{k=1}^{n} \mathbb{1}[\hat{y}^{\text{text}}_k = i] \cdot \mathbb{1}[\hat{y}^{\text{topo}}_k = j], \quad i,j \in \{0,1\}$$

A diagonal $M_{00} + M_{11}$ é a concordância (bruta), os off-diagonais $M_{01} + M_{10}$ são as discordâncias. As marginais $\sum_j M_{ij}$ e $\sum_i M_{ij}$ recuperam as taxas de predição "fake" de cada classificador isoladamente.

A análise das **células**, e não apenas dos totais, é o que diferencia esta análise de uma comparação trivial de F1. Cada célula tem uma interpretação específica:

| Célula | text | topo | Interpretação |
|---|---|---|---|
| $M_{00}$ | fake | fake | ambos votam fake — caso de **alta confiança** |
| $M_{11}$ | real | real | ambos votam real — caso de **alta confiança** |
| $M_{01}$ | fake | real | textual suspeita do conteúdo, topologia parece benigna — **zona cinzenta tipo A** |
| $M_{10}$ | real | fake | textual parece benigno, topologia suspeita da propagação — **zona cinzenta tipo B** |

**Embasamento acadêmico:**

> 📖 **Powers, D. M. W. (2011)** — "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation"
> *Journal of Machine Learning Technologies*, v. 2, n. 1, pp. 37–63
> arXiv: `2010.16061` (versão revisada 2020)
> **Localização:** Seção 2 (pp. 38–42) — definição da confusion matrix 2×2 e suas marginais; Seção 4 (Informedness, Markedness) — como matrizes de confusão expostas em quadrantes corrigem assimetrias que F1 médio mascara.
> **Relevância:** Justifica analisar as quatro células da matriz separadamente em vez de colapsar em uma única métrica. Powers argumenta que F1 é "a média harmônica de duas métricas que já agregam informação" e perde estrutura — exatamente o que a função `matriz_concordancia` (linha 72) preserva ao retornar a matriz crua.

**No código:**
> Linhas 80–82: laço explícito que constrói a matriz célula-a-célula. A escolha de não usar `confusion_matrix(p_text, p_topo)` do scikit-learn é didática (deixa a fórmula explícita) e idêntica em resultado.

---

### 3.2 F1 condicional à concordância

A novidade metodológica (linhas 91–99) é decompor o F1 em três valores condicionais:

$$F_1^{\text{concord}} = F_1\!\left(y\big|_{\hat{y}^{\text{text}}=\hat{y}^{\text{topo}}},\ \hat{y}^{\text{text}}\right),\quad F_1^{\text{disc},*} = F_1\!\left(y\big|_{\hat{y}^{\text{text}}\neq\hat{y}^{\text{topo}}},\ \hat{y}^{*}\right),\ * \in \{\text{text, topo}\}$$

Espera-se $F_1^{\text{concord}} \gg F_1^{\text{disc},*}$: quando os dois paradigmas convergem, formam um "ensemble fraco" por consenso (Dietterich 2000) que excede cada componente quando os erros são parcialmente independentes; quando divergem, pelo menos um erra por construção. A comparação $F_1^{\text{disc, text}}$ vs $F_1^{\text{disc, topo}}$ é a **tese central**: identifica o paradigma com voto de Minerva nos casos cinza — input direto para a regra dual da Fase 8.

---

## 4. Ensembling e Análise de Discordância

### 4.1 Concordância como evidência de robustez

Dietterich (2000) formaliza três razões para usar ensembles: **estatística** (reduz risco de escolher hipótese individualmente azarada), **computacional** (otimização local não atinge ótimo global) e **representacional** (a hipótese verdadeira pode estar fora de qualquer $\mathcal{H}$ individual; combinar expande). LogReg(BERT) e RF(estrutural) são radicalmente heterogêneos: $\mathbb{R}^{768}$ semântico vs $\mathbb{R}^2$ topológico. Sob o argumento (3), sua intersecção representacional é mínima — concordância vira evidência **independente** sobre a classe verdadeira.

**Embasamento acadêmico:**

> 📖 **Dietterich, T. G. (2000)** — "Ensemble Methods in Machine Learning"
> In: *Multiple Classifier Systems* (MCS 2000), Lecture Notes in Computer Science, v. 1857, Springer, pp. 1–15
> DOI: `10.1007/3-540-45014-9_1`
> **Localização:** Seção 1.2 (Three Fundamental Reasons), pp. 2–3 — argumentação estatística/computacional/representacional para ensembling; Seção 4 (Methods for Constructing Ensembles), pp. 6–10 — diversidade como pré-condição para que o ensemble exceda o componente individual.
> **Relevância:** Fundamento teórico para tratar a concordância LogReg×RF como um ensemble por consenso (`AND` lógico das predições). Quando os dois discordam, o ensemble de consenso *abstém-se* — exatamente o regime que Madras et al. (2018) modelam formalmente abaixo.

> 📖 **Wolpert, D. H. (1992)** — "Stacked Generalization"
> *Neural Networks*, v. 5, n. 2, pp. 241–259
> DOI: `10.1016/S0893-6080(05)80023-1`
> **Localização:** Seção 2 (The Basic Idea), pp. 242–245 — introdução da arquitetura de *stacking* onde um meta-classificador é treinado sobre as saídas (ou probabilidades) dos classificadores de base.
> **Relevância:** O par `(s_text, s_topo)` salvo em `gossipcop_concordancia.csv` (linhas 152–168) é exatamente o formato de meta-features que um stacker de Wolpert consome. Isto implica que o script 20 não apenas diagnostica concordância — ele também emite o input necessário para a "regra dual" do TCC (Fase 8), que pode ser implementada tanto como regra simbólica (e.g. `pred_final = pred_text if abs(s_text - s_topo) > τ else "abster"`) quanto como meta-modelo aprendido.

---

### 4.2 Disagreement-aware classification: Madras et al. (2018)

Madras, Pitassi & Zemel (2018) tratam formalmente quando um sistema deve **abster-se** ou delegar. Definem $h: \mathcal{X} \to \mathcal{Y} \cup \{\bot\}$ com perda

$$\mathcal{L}(h) = \mathbb{E}_{(x,y)}\!\left[\ell_0\!\left(h(x), y\right)\mathbb{1}[h(x)\neq\bot] + \ell_{\text{defer}}\mathbb{1}[h(x)=\bot]\right]$$

A regra natural deste script é: **abster sempre que $\hat{y}^{\text{text}} \neq \hat{y}^{\text{topo}}$** — regra de consenso com fundamento em Madras et al.: quando dois classificadores heterogêneos divergem, abster-se domina o estimador que insiste em decidir, sob qualquer custo razoável de delegação.

**Embasamento acadêmico:**

> 📖 **Madras, D.; Pitassi, T.; Zemel, R. (2018)** — "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer"
> *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, pp. 6147–6157
> arXiv: `1711.06664`
> **Localização:** Seção 3 (Learning to Defer Framework), pp. 3–5 — formulação do problema de abstenção como minimização de risco com custo explícito de delegação; Eq. (1) define a perda usada acima.
> **Relevância:** Justifica a "zona cinzenta" como categoria operacional, não como falha. Os 15 casos top de `casos_discordancia.txt` (linhas 180–192) materializam essa zona — são exatamente os $x$ onde o sistema de Madras et al. recomendaria $h(x) = \bot$.

---

### 4.3 Disagreement como sinal qualitativo

Os casos de discordância também são objeto de **inspeção qualitativa**. $M_{10}$ (textual diz "real", topologia diz "fake") tende a capturar artigos bem escritos com cascata anômala (e.g., bots) — fake news que escapa ao detector textual. $M_{01}$ (textual diz "fake", topologia diz "real") captura conteúdo controverso mas não viralizado. As colunas `text_acerta_solo`/`topo_acerta_solo` do CSV (linhas 153–167) isolam exatamente os casos em que **um** acerta e **o outro** erra — base para diagnosticar o domínio de competência de cada paradigma.

---

## 5. Componentes do Pipeline

### 5.1 Features textuais: BERT[CLS] da raiz

`g.x[0]` (dim. 768) é o embedding BERT do nó raiz — mesma feature do script 10, paridade deliberada. O `[CLS]` é definido como

$$\mathbf{h}_{[\text{CLS}]} = \text{Transformer}_L(\text{Embed}([\text{CLS}], w_1, \ldots, w_T))_{[0]} \in \mathbb{R}^{768}$$

com $L=12$ no BERT-base; treinado por NSP no pré-treino para condensar a sequência inteira.

**Embasamento acadêmico:**

> 📖 **Devlin et al. (2019)** — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805`.
> **Localização:** Seção 3.1 (Pre-training BERT) — token `[CLS]` e tarefa NSP; Seção 4.1 (GLUE) — uso do `[CLS]` para fine-tuning de classificação de sentença.
> **Relevância:** Justifica `g.x[0]` como feature textual; LogReg sobre 768 dim. já é competitivo porque o pré-treino condensa o sinal.

---

### 5.2 Features topológicas: $[\text{num\_nodes}, \text{grau\_root}]$

Apenas duas features escalares: $\text{num\_nodes}$ (tamanho da cascata, captura viralidade) e $\text{grau\_root}$ (arestas saindo do nó 0). Em UPFD estrela plana, $\text{grau\_root} \approx n-1$, mas é mantido como feature independente para Bluesky multinível.

**Justificativa:** uso *deliberado de features pobres* — o script 11 mostra que essas duas variáveis já alcançam F1 substancial em UPFD-Goss. O "RF topológico" é praticamente o **classificador de confound** do TCC. Se mesmo features triviais discordam do LogReg textual, há informação estrutural não-trivial a explorar (ou o confound é tão forte que dispensa modelagem fina).

**Embasamento acadêmico:**

> 📖 **Dou et al. (2021)** — "User Preference-aware Fake News Detection". *SIGIR 2021*, pp. 2051–2055. DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`.
> **Localização:** Seção 3 (UPFD Framework), pp. 2052–2053 — estrutura em árvore (raiz=notícia, folhas=engajamentos), feature `content` como concatenação BERT.
> **Relevância:** Fonte do dataset (linha 112); documenta o nó-raiz canônico que justifica `g.x[0]` como notícia e $\text{grau\_root}$ como engajamento direto.

> 📖 **Breiman (2001)** — "Random Forests". *Machine Learning*, v. 45, n. 1, pp. 5–32. DOI: `10.1023/A:1010933404324`.
> **Localização:** Seção 1 — definição do algoritmo; Seção 2 — robustez com poucas features e muitas árvores ($n_{\text{estimators}}=200$, linha 131).
> **Relevância:** Com 2 features, o RF aproxima uma regra threshold suavizada por bagging.

---

### 5.3 Score de classe e mapeamento de rótulos

```python
s_text = lr.predict_proba(Xt_te)[:, list(lr.classes_).index(0)]  # prob fake
s_topo = rf.predict_proba(Xk_te)[:, list(rf.classes_).index(0)]
```

A indexação por `lr.classes_.index(0)` é uma defesa contra inversão de label: garante que `s_text` e `s_topo` sejam **probabilidades da classe `0`** (fake na convenção UPFD), independentemente da ordem que scikit-learn use internamente. Esse cuidado é importante porque a interpretação visual do scatter plot e o ranqueamento dos casos de discordância (linhas 180–181) dependem de ambos os scores apontarem para a mesma classe.

---

### 5.4 Aplicação no Bluesky: surrogate de features

O par `(1 + interações, interações)` é o **surrogate** estrutural Bluesky: $\text{num\_nodes} \approx 1 + \text{replies} + \text{reposts}$, $\text{grau\_root} \approx \text{replies} + \text{reposts}$ (assume estrela — exato para reposts, aproximado para replies). É coerente com a paridade FNN/UPFD/Bluesky do PIPELINE.md (Fase 0). A comparação textual×topológica plena no Bluesky exigiria gerar BERT em 168k posts; é deferida ao script 22 sobre amostra.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 O que o script mede que outros scripts não medem

Os scripts 10/14/11 medem cada paradigma isoladamente. Este é o único que **mede a sobreposição informacional** entre textual e topológico. Responde diretamente: "se já tenho classificador textual, vale a pena adicionar topológico?". A resposta vem do $\kappa$ (sobreposição) e do F1 condicional ao desacordo (voto de Minerva nos casos cinza).

### 6.2 Articulação com a vulnerabilidade topológica

Script 14 mostra que sinal estrutural sozinho bate F1 alto em UPFD (viés topológico). Script 20 estende: se LogReg(BERT) e RF(topo) **concordam muito** ($\kappa$ alto), o BERT já capturou indiretamente o confound (colinearidade text-topo: artigos virais $\to$ longos/citados). Se **discordam muito**, cada paradigma capta um aspecto diferente — cenário ideal para regra dual.

### 6.3 Conexão direta com a Seção 6.4 do TCC

> **UPFD-Goss tem labels (mede acerto absoluto), Bluesky não (mede só consistência). Combinar os dois diagnostica em qual regime cada classificador é confiável. Quando concordam: alta confiança. Quando discordam: zona cinzenta. Casos de discordância são qualitativamente analisados — onde o textual acerta e o topológico erra (ou vice-versa) revela o domínio de competência de cada um. Base para "regra dual" (Fase 8): combinar ambos em produção.**

A documentação implementa essa narrativa exatamente:
- **"Acerto absoluto" no GossipCop:** F1 macro reportado em `gossipcop_metricas.txt` (linhas 174–175).
- **"Consistência" no Bluesky:** scores topológicos por feed em `bluesky_concordancia.csv` (linhas 244–249) — base para o script 22 cruzar com LogReg-textual em amostra menor.
- **"Quando concordam — alta confiança":** `f1_quando_concordam` (linha 96) — espera-se valor próximo de 1.
- **"Quando discordam — zona cinzenta":** `f1_text_quando_discordam` e `f1_topo_quando_discordam` (linhas 98–99) — revelam qual paradigma tende a estar correto no desacordo.
- **"Análise qualitativa" dos casos de discordância:** `casos_discordancia.txt` (linhas 182–192) — material direto para a defesa do TCC.
- **"Base para regra dual":** as colunas `text_acerta_solo` e `topo_acerta_solo` do CSV (linhas 161–167) são o ground truth para treinar/avaliar o stacker de Wolpert (1992) que a Fase 8 vai materializar.

### 6.4 Resposta parcial à questão central do TCC

> *"GNNs são uma alternativa viável para detecção de fake news?"*

A leitura que este script entrega é mais sutil que "sim/não":
1. Em UPFD-Goss, ambos atingem F1 alto individualmente; se $\kappa$ for substancial e $F_1^{\text{concord}}$ alto, o paradigma estrutural é viável **em concordância** com o textual, não em substituição.
2. Casos de discordância localizam onde GNNs poderiam adicionar valor (conteúdo plausível com propagação anômala, ou vice-versa).
3. No Bluesky, sem ground truth, o paradigma estrutural tem **valor de triagem em produção** mesmo quando comparação face-a-face com NLP é cara.

Resposta integrada: GNNs (e mesmo RF estrutural mínimo) são viáveis como **componente complementar** num sistema dual, não como substituto do textual. Mais defensável que afirmar superioridade absoluta — e `casos_discordancia.txt` permite ilustrar com exemplos concretos para a banca.

---

## 7. Análise de Código

### 7.1 Pontos fortes

- **Defesa contra inversão de classe** (linhas 129, 134, 241): `list(model.classes_).index(0)` blinda contra reordenamento; crítico para que scores não sejam silenciosamente invertidos.
- **Paridade com script 10:** mesma feature `g.x[0]`, mesmo `max_iter=1000`, mesmo `random_state=42`.
- **CSV granular** (linhas 152–168): preserva $(s_{\text{text}}, s_{\text{topo}}, \hat{y}, y, \text{flags})$ — suficiente para alimentar um stacker (Wolpert 1992) sem retreino.
- **Visualização honesta:** scatter $(s_{\text{text}}, s_{\text{topo}})$ codifica exatamente a matriz 2×2 com diagonal e cortes em $0{,}5$.

### 7.2 Limitações identificadas

- **L1 — Falta `confusion_matrix(y, pred)` por classificador:** apenas matriz de concordância é salva; FP/FN individuais ajudariam a parametrizar pesos da regra dual.
- **L2 — F1 condicional pode ser ruidoso:** sobre $\sim$400 discordâncias em UPFD-Goss, F1-macro condicional oscila muito; ideal reportar bootstrap CI ou $n$ por partição.
- **L3 — $\kappa$ sem ponderação:** correto para 2 classes; revisar se estender a multiclasse.
- **L4 — Bluesky surrogate unilateral:** só aplica RF, não LogReg textual; nome `bluesky_concordancia.csv` é enganoso (seria mais preciso `bluesky_score_topologico.csv`). Concordância plena fica para script 22.
- **L5 — Modelos não persistidos:** retreina LogReg/RF in-memory, redundante com script 17. Idealmente carregaria `logreg_bert_fnn.pkl` e `rf_struct_gossipcop.pkl` para evitar drift entre scripts.
- **L6 — Threshold fixo $0{,}5$:** aceitável em UPFD-Goss balanceado; no Bluesky desbalanceado, ótimo de Youden seria mais defensável.

### 7.3 Boas práticas observadas

- Uso de `random_state=RANDOM_SEED` em ambos os classificadores: reprodutibilidade garantida.
- `n_jobs=-1` no RF: paralelização correta para o tamanho do dataset.
- Tratamento defensivo de JSON malformado (linhas 234–236): `try/except` em `json.loads`.
- Saída em CSV com `encoding="utf-8"` e `newline=""`: portabilidade Windows/Linux preservada.

---

## 8. Referências Bibliográficas

1. COHEN, J. **A Coefficient of Agreement for Nominal Scales**. *Educational and Psychological Measurement*, v. 20, n. 1, pp. 37–46, 1960. DOI: `10.1177/001316446002000104`

2. LANDIS, J. R.; KOCH, G. G. **The Measurement of Observer Agreement for Categorical Data**. *Biometrics*, v. 33, n. 1, pp. 159–174, 1977. DOI: `10.2307/2529310`

3. POWERS, D. M. W. **Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation**. *Journal of Machine Learning Technologies*, v. 2, n. 1, pp. 37–63, 2011. arXiv: `2010.16061`

4. DIETTERICH, T. G. **Ensemble Methods in Machine Learning**. In: *Multiple Classifier Systems (MCS 2000)*, Lecture Notes in Computer Science, v. 1857, Springer, pp. 1–15, 2000. DOI: `10.1007/3-540-45014-9_1`

5. WOLPERT, D. H. **Stacked Generalization**. *Neural Networks*, v. 5, n. 2, pp. 241–259, 1992. DOI: `10.1016/S0893-6080(05)80023-1`

6. MADRAS, D.; PITASSI, T.; ZEMEL, R. **Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer**. *NeurIPS 2018*, pp. 6147–6157. arXiv: `1711.06664`

7. DEVLIN, J.; CHANG, M.-W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805`

8. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`

9. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *SIGIR 2021*, pp. 2051–2055. DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`

10. FEINSTEIN, A. R.; CICCHETTI, D. V. **High Agreement but Low Kappa: I. The Problems of Two Paradoxes**. *Journal of Clinical Epidemiology*, v. 43, n. 6, pp. 543–549, 1990. DOI: `10.1016/0895-4356(90)90158-L`

11. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`

---

## 9. Glossário

| Termo | Definição | Fonte |
|---|---|---|
| Cohen's $\kappa$ | Coeficiente de concordância entre dois classificadores corrigido por concordância esperada por acaso; varia em $(-1, 1]$ com $1$ = concordância perfeita | Cohen (1960), Eq. 1 |
| Matriz de concordância | Matriz 2×2 que cruza predições de dois classificadores (sem usar ground truth); diferente de confusion matrix tradicional | Powers (2011), §2 |
| $F_1$ condicional | F1 calculado sobre o subconjunto de exemplos onde os dois classificadores concordam (ou discordam); diagnóstico por região | Conceito deste TCC, fundamentado em Madras et al. (2018) |
| Learning to defer | Framework onde o classificador pode emitir $\bot$ (abstenção) em casos ambíguos | Madras et al. (2018), §3 |
| Stacking / Stacked Generalization | Meta-classificador treinado sobre as saídas de classificadores de base | Wolpert (1992), §2 |
| Ensemble heterogêneo | Ensemble onde os componentes têm representação radicalmente diferente (e.g., LogReg-BERT vs RF-topo); favorece independência de erro | Dietterich (2000), §1.2 |
| Zona cinzenta | Subconjunto de exemplos onde $\hat{y}^{\text{text}} \neq \hat{y}^{\text{topo}}$; candidato natural a abstenção/triagem manual | Conceito deste TCC |
| Regra dual | Política de combinação textual+topológico a ser implementada na Fase 8: consenso quando concordam, política calibrada quando discordam | Fase 8 do PIPELINE.md |
| `g.x[0]` | Embedding BERT-768 do nó raiz (notícia/post original) — única feature do classificador textual | Devlin et al. (2019), §3.1 |
| Surrogate Bluesky | Aproximação $[\text{num\_nodes}, \text{grau\_root}] \approx [1+\text{interações}, \text{interações}]$ aplicada quando cascata real não está disponível | Linhas 237–240 do script |
| `feature="content"` | Modo do dataset UPFD onde features dos nós são embeddings BERT da notícia + perfis de usuários | Dou et al. (2021), §3 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 20_textual_vs_topologico.py
Arquivo gerado: theory_andre/20_textual_vs_topologico_doc.md
Fontes academicas utilizadas: 11
   1. Cohen (1960) — Cohen's kappa
   2. Landis & Koch (1977) — escala interpretativa de kappa
   3. Powers (2011) — confusion matrix analysis
   4. Dietterich (2000) — Ensemble Methods in ML
   5. Wolpert (1992) — Stacked Generalization
   6. Madras et al. (2018) — Predict Responsibly / learning to defer
   7. Devlin et al. (2019) — BERT
   8. Breiman (2001) — Random Forests
   9. Dou et al. (2021) — UPFD (SIGIR)
  10. Feinstein & Cicchetti (1990) — paradoxos do kappa
  11. Shu et al. (2020) — FakeNewsNet
Conceitos cobertos:
  - Cohen's kappa e correcao de chance agreement
  - Matriz 2x2 de concordancia entre classificadores
  - F1 condicional a concordancia/discordancia
  - Ensembling heterogeneo (Dietterich)
  - Stacking / meta-classificador (Wolpert)
  - Learning to defer / abstencao (Madras)
  - BERT [CLS] como feature canonica
  - Random Forest sobre features topologicas minimas
  - Surrogate de features estruturais Bluesky
  - Conexao com Fase 8 do PIPELINE (regra dual)
  - Articulacao com Secao 6.4 do TCC
Limitacoes:
  - F1 condicional sem CI bootstrap (L2)
  - Bluesky surrogate apenas no lado topologico (L4)
  - Modelos nao persistidos (L5) — redundante com script 17
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISAO — nao prosseguir para o proximo script.
```
