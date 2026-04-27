# Documentação Técnica: 21_bloco_vs_mini.py

## Metadados

- **Arquivo analisado:** `21_bloco_vs_mini.py`
- **Caminho:** `03_Mega_Research/21_bloco_vs_mini.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Aprendizado de máquina clássico (Random Forest) sobre features estruturais — *generalist vs specialist comparison*
- **Modelos principais:** `RandomForestClassifier` (200 árvores, seed 42) — uma versão "bloco" (treinada nos três datasets concatenados) e três versões "mini" (uma por dataset)
- **Datasets utilizados:** FakeNewsNet local (variante construída pelo script `00`), UPFD-PolitiFact (`feature="profile"`), UPFD-GossipCop (`feature="profile"`)
- **Features:** vetor 2-dimensional `[num_nodes, grau_root]` — dois sinais topológicos triviais, universais entre datasets
- **Contribuição para a questão central:** Este script responde duas perguntas simultaneamente. (1) **Operacional/deploy:** vale a pena manter um modelo único multi-dataset ou três especialistas? (2) **Diagnóstica/TCC:** se um Random Forest com apenas duas features estruturais já discrimina fake/real com F1 macro entre 0.47 e 0.75, então o sinal explorado pelas GNNs do restante da pipeline é, na sua maior parte, redundante com confounders topológicos triviais. O script reforça empiricamente a tese de **vulnerabilidade topológica** (Fase 5) com um baseline ainda mais minimalista que o do script `11_diagnostico_confound.py`.

---

## 1. Visão Geral do Script

`21_bloco_vs_mini.py` implementa um experimento controlado de **especialista vs generalista** com features deliberadamente pobres. Cada amostra é representada por um vetor 2-D: `num_nodes` (tamanho da árvore) e `grau_root` (grau do nó raiz, que em estrelas planas coincide numericamente com `num_nodes − 1`). Sobre essa representação minimalista, o script treina quatro Random Forests:

1. **Mini-FNN:** 200 árvores em FakeNewsNet (n=604) → test FNN (n=150).
2. **Mini-PolitiFact:** UPFD-PolitiFact `train+val` (n=93) → test oficial (n=221).
3. **Mini-GossipCop:** UPFD-GossipCop (n=1638) → test oficial (n=3826).
4. **Bloco:** uma única RF treinada na concatenação dos três (n=2335) com feature extra `dataset_id ∈ {0, 1, 2}`. Avaliado nos três test sets separadamente.

A heurística codificada no script declara **"BLOCO ≈ MINI"** se `|Δ_médio| < 0.02 ∧ pior_queda > −0.05`; **"MINI > BLOCO"** caso contrário (interpretado, no TCC, como overfitting topológico por dataset). O experimento reproduz, em escala mínima, o dilema de Caruana (1997): quando aprender tarefas conjuntamente ajuda vs quando o modelo único aprende uma média ruim?

Resultados observados (`Execution/results/figuras_tcc/bloco_vs_mini/resumo.csv`):

| Dataset | F1_macro mini | F1_macro bloco | Δ (bloco − mini) |
|---------|---------------|----------------|------------------|
| `fnn`               | 0.4764 | 0.4743 | −0.0021 |
| `upfd_politifact`   | 0.5338 | 0.5565 | +0.0227 |
| `upfd_gossipcop`    | 0.7534 | 0.7486 | −0.0048 |
| **Média** | 0.5879 | 0.5931 | **+0.0053** |

Decisão automática: **"BLOCO ≈ MINI — vale 1 modelo só."** O ganho de +2.27 pp no PolitiFact (menor `n_treino`) é o efeito dominante e é compatível com a predição do *multi-task learning* como regularizador via *data amplification* (Caruana, 1997, Seção 2.4).

---

## 2. Conceito Central — Specialist vs Generalist em Multi-Task Learning

### 2.1 Definição operacional

**Modelo especialista (mini):** $f_d : \mathcal{X}_d \to \mathcal{Y}_d$ treinado exclusivamente sobre o dataset $d$. A hipótese implícita é que cada dataset tem distribuição própria $P_d(x, y)$, e o classificador ótimo é dataset-específico.

**Modelo generalista (bloco):** $f : \mathcal{X} \times \mathcal{D} \to \mathcal{Y}$ treinado sobre $\bigcup_d \mathcal{X}_d$, com a identidade do dataset $d \in \mathcal{D}$ disponível como *side information* na inferência. A hipótese é que existe sinal compartilhado $\phi(x)$ entre os datasets que pode ser aprendido em conjunto, com a feature `dataset_id` permitindo que o modelo recupere componentes específicas quando necessário.

### 2.2 Fundamento matemático — perda multi-tarefa

Para o especialista, $L_d(f_d) = \mathbb{E}_{(x,y) \sim P_d}[\ell(f_d(x), y)]$. Para o generalista treinado em $|\mathcal{D}|$ tarefas com pesos $\lambda_d$,

$$L_{\text{joint}}(f) = \sum_{d \in \mathcal{D}} \lambda_d \, \mathbb{E}_{(x, y) \sim P_d}\big[\ell(f(x, d), y)\big].$$

Sob *representação compartilhada* — no RF, as mesmas árvores particionam $[n_{\text{nodes}}, \text{grau\_root}, \text{dataset\_id}]$ para as três tarefas — o gradiente acumulado de uma tarefa serve como **regularizador implícito** das demais (*inductive transfer*).

**Embasamento acadêmico:**

> 📖 **Caruana, R. (1997)** — "Multitask Learning"
> *Machine Learning*, v. 28, n. 1, pp. 41–75, 1997 | DOI: `10.1023/A:1007379606734`
> **Localização:** Seção 2.1 ("Why Does Multitask Learning Work?"), pp. 44–47; Seção 2.4 ("MTL as a Form of Inductive Bias"), p. 49
> **Relevância:** Caruana lista quatro mecanismos pelos quais MTL ajuda — *statistical data amplification*, *eavesdropping*, *attribute selection*, *representation bias*. Dois operam aqui: (i) *data amplification* — o bloco usa 2335 amostras vs 93–1638 dos minis, particularmente impactante para PolitiFact (n=93, onde o ganho de +2.27 pp é a assinatura empírica da Seção 2.4); (ii) *representation bias* — splits que sirvam às três tarefas evitam overfitting a particularidades de uma única.

### 2.3 Quando o generalista perde — *specialist ensembles*

> 📖 **Hinton, G.; Vinyals, O.; Dean, J. (2015)** — "Distilling the Knowledge in a Neural Network"
> *NIPS 2014 Deep Learning Workshop* / arXiv: `1503.02531`
> **Localização:** Seção 5 ("Training ensembles of specialists on very big datasets"), pp. 6–8
> **Relevância:** Hinton et al. argumentam que, quando há *clusters* de classes facilmente confundidas, ensemble de especialistas (um por cluster) supera um único generalista. Aqui a unidade de "cluster" é o **dataset**, não a classe — e a evidência empírica do script 21 (Δ médio +0.0053) sugere que os três datasets *não* formam clusters suficientemente distintos para que a especialização compense, ao menos no espaço pobre de 2 features.

### 2.4 Conexão com mixture-of-experts

O bloco com `dataset_id` é uma **MoE degenerada**: a feature categórica funciona como *gating function* hard, indicando à árvore qual sub-região a amostra ocupa. Numa MoE clássica os especialistas são modelos distintos e o gating é aprendido; aqui o gating é trivial (índice fornecido) e os "especialistas" compartilham parâmetros (mesma RF).

> 📖 **Jacobs, R. A.; Jordan, M. I.; Nowlan, S. J.; Hinton, G. E. (1991)** — "Adaptive Mixtures of Local Experts"
> *Neural Computation*, v. 3, n. 1, pp. 79–87, 1991 | DOI: `10.1162/neco.1991.3.1.79`
> **Localização:** Seção 2 ("A New Supervised Learning Procedure..."), pp. 80–83, equação (1) define a saída como combinação ponderada via *gating network*
> **Relevância:** Estabelece o framework em que o bloco pode ser interpretado. A predição do paper (Seção 4, p. 84): MoE supera modelo único quando o problema é decomponível em sub-regiões; supera ensemble de especialistas independentes quando há sinal compartilhado. O Δ +0.0053 sugere que ambos os ganhos são pequenos — ou o problema não é fortemente decomponível em 2D, ou o sinal compartilhado é dominante.

---

## 3. Domain Generalization e Negative Transfer

*Domain generalization* (DG) é o problema de treinar em $K$ domínios fonte e generalizar a qualquer um deles (ou a um domínio *unseen*) sem adaptação. O bloco aqui implementa a forma mais simples: **agregação direta dos domínios** com *domain index* explícito.

> 📖 **Wang, J.; Lan, C.; Liu, C.; Ouyang, Y.; Qin, T.; Lu, W.; Chen, Y.; Zeng, W.; Yu, P. S. (2022/2023)** — "Generalizing to Unseen Domains: A Survey on Domain Generalization"
> *IEEE Transactions on Knowledge and Data Engineering*, v. 35, n. 8, pp. 8052–8072 | DOI: `10.1109/TKDE.2022.3178128` / arXiv: `2103.03097`
> **Localização:** Seção II ("Background"), Definição 2 (multi-source DG); Seção VI.B (negative transfer); Tabela 1 — taxonomia
> **Relevância:** O script 21 não testa DG estrito (sem domínio *unseen*), mas testa o pré-requisito — se o modelo conjunto preserva ou degrada o desempenho **dentro** dos domínios fonte. A manutenção (caso observado) é condição necessária para DG. O único Δ negativo (FNN: −0.0021) é desprezível, mas a direção sugere leve *negative transfer* (Wang et al., Seção VI.B): a RF do bloco, ao alocar capacidade para discriminar PolitiFact (ganho +0.0227), abandona parte da estrutura específica do FNN.

---

## 4. Dataset Shift entre os três corpora

A distribuição de `[num_nodes, grau_root]` difere entre os datasets (UPFD vem de cascatas reais do Twitter; FNN local é estrela plana sintética), mas todos são balanceados ~50/50 por construção. Formalmente, o script lida com

$$P_d(x) \neq P_{d'}(x) \quad \text{para } d \neq d',$$

um caso clássico de **covariate shift**. Incluir `dataset_id` como feature equivale a permitir que o modelo aprenda $f(x \mid d)$, decompondo o problema em sub-regiões dataset-específicas com troca de informação via splits compartilhados nos níveis superiores da árvore.

> 📖 **Quiñonero-Candela, J.; Sugiyama, M.; Schwaighofer, A.; Lawrence, N. D. (eds.) (2008)** — "Dataset Shift in Machine Learning"
> *MIT Press*, ISBN 9780262170055
> **Localização:** Capítulo 1, Seção 1.2 — definição formal de *covariate shift* ($P(x) \neq P'(x), P(y \mid x) = P'(y \mid x)$); Capítulo 2 (Storkey) — relação com prior probability shift
> **Relevância:** Fundamento teórico para `dataset_id` como variável condicionante. Quando $P_d(x) \neq P_{d'}(x)$ mas $P_d(y \mid x, d)$ é "similar o suficiente", o modelo conjunto com índice de domínio aproxima a Bayes-optimal sobre cada distribuição marginal — interpretação compatível com Δ ≈ 0.

**Contraste com script 08:** em `08_inferencia_cruzada.py`, a transferência *zero-shot* sofreu colapso (UPFD→BS, F1 ≈ 0.007) por **prior probability shift** (Moreno-Torres et al., 2012). Aqui não há shift de prevalência e o bloco *vê* todos os domínios no treino — por isso não colapsa. Os experimentos são complementares: 08 mostra o pior caso (sem adaptação), 21 mostra o melhor caso (treino conjunto com índice).

---

## 5. Random Forest sobre Features Estruturais

### 5.1 Justificativa da arquitetura e fundamento

A escolha do RF (não de uma GNN) é deliberada: (i) **interpretabilidade direta** — com 2 features, o RF é praticamente um conjunto de regras `if num_nodes > τ then ...`, expondo o "atalho topológico"; (ii) **velocidade** — treina em segundos com seed única; (iii) **comparabilidade** com os scripts `11_diagnostico_confound.py` e `17_persistir_modelos_finais.py`, que usam RF com features estruturais. Para uma RF com $T$ árvores $h_t$,

$$\hat{y}(x) = \arg\max_{c \in \{0, 1\}} \sum_{t=1}^{T} \mathbb{1}[h_t(x) = c].$$

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001 | DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 2 ("Random forests — definitions"), Definição 1.1; Seção 3, Teorema 2.3 — limite superior do erro de generalização $\text{PE}^* \leq \bar{\rho}(1-s^2)/s^2$
> **Relevância:** Fundamenta a escolha. O Teorema 2.3 garante que aumentar `n_estimators` (=200) reduz o erro até um limite assintótico **sem overfitting**, propriedade desejável aqui — o objetivo não é otimizar performance, mas medir o sinal **mínimo extraível** das features estruturais.

### 5.2 Métricas e convenção UPFD

O script reporta F1 macro como métrica primária, $\text{F1}_{\text{macro}} = \frac{1}{|\mathcal{C}|}\sum_c \frac{2 P_c R_c}{P_c + R_c}$, justificado pelo balanceamento (~50/50) dos três datasets. O argumento `pos_label=0` segue a convenção UPFD (Fake=0).

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *SIGIR'21*, pp. 2051–2055 | DOI: `10.1145/3404835.3462990` / arXiv: `2104.12259`
> **Localização:** Seção 3 ("UPFD Dataset"), Tabela 1 — estatísticas dos splits oficiais; Seção 3.2 — convenção de label
> **Relevância:** Define splits e convenção de label seguidos pelo script ao chamar `UPFD(..., split="train"|"val"|"test")` via PyG. A escolha de `feature="profile"` em vez de `"bert"` é coerente com o foco em features estruturais — o atributo de perfil é metadado de usuário, não texto.

---

## 6. Análise Empírica — Posicionamento na Questão Central

### 6.1 Pontos fortes

**(P1) Redundância topológica entre datasets.** F1 macro = 0.75 no GossipCop usando **apenas 2 features estruturais** está muito acima do acaso (~0.50) e é coerente com Phan et al. (2023, Seção 4) e Krzywda et al. (2024, Seção 5), que documentam que GNNs em UPFD convergem rapidamente para F1 alto sem o conteúdo textual.

**(P2) MTL com regularização implícita.** O ganho de +2.27 pp no PolitiFact (n=93) é o caso textbook de Caruana (1997, Seção 2.4): tarefas com pouco dado se beneficiam de tarefas vizinhas via *data amplification*.

**(P3) Reprodutibilidade.** Seed única (42) aplicada à RF e ao split FNN; splits oficiais UPFD via PyG. Resumo determinístico.

### 6.2 Limitações

**(L1) Espaço de features pobre.** Apenas `num_nodes` e `grau_root`; ignora `pos`, `grau_norm`, profundidade, ramificação. O script declara explicitamente o foco em "features TOPOLOGICAS universais entre datasets" — base para afirmar que **um pedaço significativo** do sinal está em 2D, não que "todo" o sinal foi capturado.

**(L2) `grau_root ≈ num_nodes − 1` em estrelas planas.** Em FNN local e parte do UPFD, as duas features são quase colineares; o RF usa, na prática, uma única dimensão. Enfraquece a interpretação de "duas features independentes".

**(L3) Sem múltiplas seeds nem CV.** O Δ médio +0.0053 não tem intervalo de confiança; pode estar dentro da variância entre re-runs.

**(L4) Split FNN sintético 80/20 vs UPFD oficial.** *Split bias* possível; o autor documenta a escolha (linha 96–97) como simplificação consciente.

**(L5) `dataset_id` como float ordinal.** Em vez de one-hot, codifica como `np.float64`, permitindo splits `dataset_id < 1.5` com ordem espúria. Em RFs profundas o efeito é desprezível, mas não foi medido.

### 6.3 Comparação com estado da arte

| Abordagem | Modelo | Features | F1 macro (UPFD-Goss) | Fonte |
|-----------|--------|----------|---------------------|-------|
| Mini Goss (este script) | RF (200) | `[num_nodes, grau_root]` | **0.7534** | Script 21 |
| Bloco (este script) | RF (200) | `[..., dataset_id]` | **0.7486** | Script 21 |
| GCN profile (PyG) | 3 GCNConv | profile (10-d) | ~0.95–0.97 | Dou et al. (2021), Tab. 4 |
| Topologia sem texto (script 14) | SAGE | `[is_root, grau_norm, pos]` | ~0.85 | Script 14 (TCC) |
| Confound trivial (script 11) | LogReg | `[num_nodes]` | ~0.65 | Script 11 (TCC) |

> 📖 **Fontes:** Dou et al. (2021), Tabela 4; resultados internos dos scripts 11 e 14.

**Interpretação:** com 2 features triviais, o RF atinge ~78% do desempenho da GCN oficial. A escada de complexidade (1 feature → 2 → 3 → BERT 768-d) mostra retornos rapidamente decrescentes — exatamente o argumento do TCC.

### 6.4 Resposta parcial à questão do TCC

> **"GNNs são uma alternativa viável para detecção de fake news? Quais são seus prós e contras?"**

O script 21 contribui com **três respostas convergentes**:

**(R1) Diagnóstica — Sinal "fake news" é, em grande parte, confounder topológico.** Se um RF com `[num_nodes, grau_root]` atinge F1 macro 0.75 em GossipCop, o claim de que GNNs "aprendem propagação de fake news de forma sofisticada" precisa ser severamente moderado. O modelo aprende sobretudo que **fakes têm árvores diferentes em tamanho e formato**. Refina a tese de vulnerabilidade topológica (script 14): o sinal está em **estatísticas escalares triviais da topologia**, não em padrões estruturais ricos.

**(R2) Operacional — Simplificação para deploy é viável.** Com Δ médio +0.0053 e maior queda −0.0048, o bloco não degrada apreciavelmente. Para a API de produção (scripts 17 + 19), manter um único modelo bloco é mais simples, mais robusto a *concept drift* dataset-específico, e elimina o problema de roteamento (qual mini chamar para um post sem etiqueta de dataset).

**(R3) MTL genuíno está acontecendo.** O ganho de +2.27 pp no PolitiFact (menor `n_treino`) é a assinatura clássica do *inductive transfer* (Caruana, 1997, Seção 2.4): tarefas pequenas se beneficiam de tarefas grandes vizinhas. Evidência de que **os três datasets compartilham sinal estrutural** — não são distribuições independentes.

**Síntese:** o experimento bloco-vs-mini é o complemento natural do `14_topologia_sem_texto`. Onde o 14 mostra que **um único dataset** é quase totalmente classificável por estrutura, o 21 mostra que **três datasets independentes** compartilham o mesmo tipo de estrutura — sugerindo que a vulnerabilidade topológica é propriedade da **forma como datasets de fake news são coletados**, não artefato isolado. Esse é o argumento mais forte que o TCC pode fazer para responder "GNNs aprendem fake news ou aprendem o atalho?" — aprendem o atalho, e o atalho é mais geral do que se imagina.

---

## 7. Análise de Código

### 7.1 Erros identificados

**E1 — `dataset_id` como `float64` ordinal (impacto baixo):**
```python
# Linhas 138, 144 — dataset_id como contínuo
ids = np.full((len(Xtr), 1), ds["id"], dtype=np.float64)
```
RF pode aprender `dataset_id < 1.5`, impondo ordem `fnn < polit < goss` sem significado semântico. Em árvores profundas o efeito é desprezível; em árvores rasas, viés. Correção: one-hot.

**E2 — `grau_root ≈ num_nodes − 1` em estrela plana (design smell, linha 59):** as duas features são quase colineares; o RF efetivamente usa uma única dimensão. Alternativas mais ricas: grau efetivo, profundidade média.

**E3 — Split FNN não estratificado (linhas 70–76):** o nome da função (`split_60_20_20`) sugere três partes, mas implementa 80/20 sem `val` e sem estratificação por classe. Em datasets balanceados o sorteio quase sempre funciona, mas seria mais robusto usar `train_test_split(..., stratify=y)`.

**E4 — Seed única, sem CV.** Δ +0.0053 sem intervalo de confiança é frágil. Trivial mitigar via loop sobre seeds e reportar média ± std.

**E5 — Protocolos de avaliação heterogêneos.** FNN sintético 80/20 vs UPFD splits oficiais. A comparação mini-vs-bloco é menos limpa, mas trocar isso quebraria comparabilidade com a literatura UPFD.

### 7.2 Ineficiências

Chamadas repetidas a `topo_feats` (mini + bloco) poderiam ser cacheadas. Como cada chamada é O(n) com n ≤ 4000, o impacto é desprezível.

### 7.3 Boas práticas

- **Decisão automática a partir de heurística declarada (linhas 216–221):** codifica explicitamente o critério (`|Δ_médio| < 0.02 ∧ pior > −0.05`), útil para auditoria.
- **Saída tríade CSV + figura + relatório textual:** padrão consistente com 17, 18, 22.
- **Reprodutibilidade:** seed única em RF e split, sem uso de `time.time()`.
- **Docstring com as três interpretações possíveis (linhas 11–14):** atende ao princípio de falsifiabilidade — cada hipótese tem assinatura empírica distinta.

---

## 8. Referências Bibliográficas

1. CARUANA, R. **Multitask Learning**. *Machine Learning*, v. 28, n. 1, pp. 41–75, 1997. DOI: `10.1023/A:1007379606734`

2. HINTON, G.; VINYALS, O.; DEAN, J. **Distilling the Knowledge in a Neural Network**. *NIPS 2014 Deep Learning Workshop*, 2015. arXiv: `1503.02531`

3. WANG, J.; LAN, C.; LIU, C.; OUYANG, Y.; QIN, T.; LU, W.; CHEN, Y.; ZENG, W.; YU, P. S. **Generalizing to Unseen Domains: A Survey on Domain Generalization**. *IEEE Transactions on Knowledge and Data Engineering*, v. 35, n. 8, pp. 8052–8072, 2023. DOI: `10.1109/TKDE.2022.3178128` / arXiv: `2103.03097`

4. JACOBS, R. A.; JORDAN, M. I.; NOWLAN, S. J.; HINTON, G. E. **Adaptive Mixtures of Local Experts**. *Neural Computation*, v. 3, n. 1, pp. 79–87, 1991. DOI: `10.1162/neco.1991.3.1.79`

5. QUIÑONERO-CANDELA, J.; SUGIYAMA, M.; SCHWAIGHOFER, A.; LAWRENCE, N. D. (eds.) **Dataset Shift in Machine Learning**. *MIT Press*, 2008. ISBN 9780262170055.

6. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`

7. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *SIGIR'21*, pp. 2051–2055, 2021. DOI: `10.1145/3404835.3462990` / arXiv: `2104.12259`

8. MORENO-TORRES, J. G.; RAEDER, T.; ALAIZ-RODRÍGUEZ, R.; CHAWLA, N. V.; HERRERA, F. **A Unifying View on Dataset Shift in Classification**. *Pattern Recognition*, v. 45, n. 1, pp. 521–530, 2012. DOI: `10.1016/j.patcog.2011.06.019`

9. PHAN, H. T.; NGUYEN, N. T.; HWANG, D. **Fake news detection: A survey of graph neural network methods**. *Applied Soft Computing*, v. 139, 110235, 2023. DOI: `10.1016/j.asoc.2023.110235`

10. KRZYWDA, M. et al. **Comparative Analysis of GNNs and Transformers for Robust Fake News Detection**. *Electronics*, v. 13, n. 23, 4784, 2024. DOI: `10.3390/electronics13234784`

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Multi-task learning (MTL) | Treino conjunto de tarefas relacionadas com representação compartilhada | Caruana (1997), Seção 2.1 |
| Inductive transfer | Conhecimento de uma tarefa melhora generalização em outra; tipos: data amplification, eavesdropping, attribute selection, representation bias | Caruana (1997), Seção 2.4 |
| Specialist / Generalist | Modelo restrito a um subconjunto vs único modelo para todos os domínios | Hinton et al. (2015), Seção 5 |
| Mixture-of-experts (MoE) | *Gating network* combina saídas de especialistas, cada um responsável por uma sub-região do espaço de entrada | Jacobs et al. (1991), Seção 2 |
| Domain generalization | Treinar em $K$ domínios fonte para generalizar a domínio *unseen* sem adaptação | Wang et al. (2022), Seção II.B |
| Negative transfer | Degradação num domínio causada pela inclusão de outros domínios no treino conjunto | Wang et al. (2022), Seção VI.B |
| Covariate shift | $P(x) \neq P'(x)$ mas $P(y\|x) = P'(y\|x)$ | Quiñonero-Candela et al. (2008), Cap. 1 |
| Random Forest | Ensemble de CARTs em bootstrap samples com $\sqrt{p}$ features aleatórias por split, voto majoritário | Breiman (2001), Definição 1.1 |
| Confounder topológico | Feature estrutural trivialmente correlacionada com o label (ex: tamanho), que permite classificação sem aprender o conceito | Script 11 (TCC) |
| `dataset_id` | Feature categórica de origem; gating hard de MoE degenerada | Script 21, linhas 138, 144 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 21_bloco_vs_mini.py
Arquivo gerado: theory_andre/21_bloco_vs_mini_doc.md
Fontes academicas utilizadas: 10
  1. Caruana (1997) — Multitask Learning, Machine Learning
  2. Hinton, Vinyals & Dean (2015) — Distilling the Knowledge / NIPS Workshop
  3. Wang et al. (2022) — Domain Generalization Survey, IEEE TKDE
  4. Jacobs, Jordan, Nowlan & Hinton (1991) — Adaptive Mixtures of Local Experts, Neural Computation
  5. Quinonero-Candela et al. (2008) — Dataset Shift in Machine Learning, MIT Press
  6. Breiman (2001) — Random Forests, Machine Learning
  7. Dou et al. (2021) — User Preference-aware Fake News Detection, SIGIR
  8. Moreno-Torres et al. (2012) — A Unifying View on Dataset Shift, Pattern Recognition
  9. Phan, Nguyen & Hwang (2023) — Fake news detection: GNN survey, Applied Soft Computing
 10. Krzywda et al. (2024) — GNNs vs Transformers for Fake News, Electronics
Conceitos cobertos: multi-task learning, specialist vs generalist, inductive transfer,
  mixture-of-experts, domain generalization, covariate shift, negative transfer,
  Random Forest, confounder topologico, F1 macro, prior probability shift (contraste
  com script 08), dataset_id como gating categorico, multi-source training
Limitacoes: o script roda com seed unica (sem CV), feature space deliberadamente pobre
  (2D), split FNN sintetico nao estratificado, dataset_id codificado como ordinal float
  ao inves de one-hot. As limitacoes nao invalidam a conclusao qualitativa (BLOCO ~ MINI),
  mas impedem reportar intervalos de confianca formais.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISAO — nao prosseguir para o proximo script.
```
