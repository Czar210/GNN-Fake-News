# Documentação Técnica: 10_baseline_textual.py

## Metadados

- **Arquivo analisado:** `10_baseline_textual.py`
- **Caminho:** `03_Mega_Research/10_baseline_textual.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** NLP — baseline textual clássico (Logistic Regression + Random Forest sobre embeddings BERT)
- **Modelos avaliados:** LogisticRegression · RandomForestClassifier (scikit-learn, sem GNN)
- **Dataset:** FakeNewsNet local, mesmos K=10 folds de `folds_fnn.pt`
- **Contribuição para a questão central:** Este script estabelece o **"teto textual"** — o desempenho alcançável usando apenas o embedding BERT do nó raiz, sem qualquer informação de propagação. É o baseline direto contra o qual todos os modelos GNN do TCC são comparados. O resultado — F1≈0.86 tanto para LogReg quanto para GNN — é a evidência central do resultado negativo.

---

## 1. Visão Geral do Script

`10_baseline_textual.py` é intencionalmente minimalista: extrai o embedding BERT do **nó raiz** de cada grafo (`grafo.x[0]`, dimensão 768), ignora todos os outros nós e arestas, e treina dois classificadores clássicos de scikit-learn nos mesmos K=10 folds compartilhados com os scripts GNN.

O design é deliberado: ao usar `x[0]` (apenas o artigo/post raiz), o baseline mede o quanto pode ser aprendido *sem* a estrutura da rede social. Se os GNNs não superarem este baseline de forma estatisticamente significativa, a hipótese central do TCC — de que a topologia de propagação adiciona sinal discriminativo — é refutada.

Os dois classificadores são complementares:
- **LogisticRegression**: modelo linear, interpretável, verifica se o espaço BERT-768 é **linearmente separável** para a tarefa
- **RandomForestClassifier**: modelo não-linear com ensemble de árvores de decisão, captura interações entre dimensões do embedding que a regressão logística ignora

---

## 2. Extração de Features — `grafo.x[0]`

**Descrição técnica:**
```python
X = torch.stack([g.x[0] for g in grafos]).numpy()   # shape: [N_grafos, 768]
y = np.array([g.y.item() for g in grafos])           # shape: [N_grafos]
```

`g.x` é a matriz de features dos nós do grafo $g$, com shape `[n_nos, 768]`. `g.x[0]` é o embedding BERT do **nó raiz** — o post/artigo original — de 768 dimensões, gerado pelo script 02 (Bluesky) ou script 00 (FakeNewsNet) usando um modelo BERT multilingual.

**Por que apenas `x[0]` e não a média de todos os nós?**
Usar apenas o nó raiz é o baseline mais conservador e mais informativo:
1. Representa o que um classificador *puramente textual* consegue com a informação do artigo original
2. Evita vantagem injusta — se usarmos todos os nós, estaríamos indiretamente usando informação de propagação (os replies/retweets) que o classificador linear não aprendeu a interpretar estruturalmente

Um baseline com `mean(g.x)` estaria mais próximo da informação disponível para uma GNN com global_mean_pool — mas misturaria features de usuários com features do artigo de forma ingênua.

**Embasamento acadêmico:**

> 📖 **Devlin, J.; Chang, M. W.; Lee, K.; Toutanova, K. (2019)** — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
> *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*, pp. 4171–4186
> arXiv: `1810.04805` | ACL Anthology: `N19-1423`
> **Localização:** Seção 3.1 (BERT: Pre-training) — arquitetura transformer bidirecional; Seção 4.1 (GLUE) — uso de `[CLS]` token embedding para classificação de sentenças
> **Relevância:** BERT é o modelo que gera o embedding de 768 dimensões usado como feature. O token `[CLS]` do BERT foi projetado exatamente para capturar a representação semântica global de um texto para tarefas de classificação — justificando seu uso como feature para LogReg e RandomForest.

---

## 3. Logistic Regression sobre BERT-768

**Descrição técnica:**
Regressão Logística é um classificador linear que aprende um hiperplano de separação no espaço de features. Para classificação binária (Fake/Real):

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

onde $\mathbf{w} \in \mathbb{R}^{768}$ é o vetor de pesos aprendido, $b$ é o bias, e $\sigma$ é a função sigmoide. O treinamento minimiza a cross-entropy com regularização L2 (padrão do scikit-learn: `C=1.0`).

**Interpretação do resultado F1≈0.86:**
Um F1≈0.86 obtido por um classificador **linear** sobre o embedding BERT significa que o espaço de 768 dimensões do BERT já é **linearmente separável** para este problema. Não há necessidade de transformações não-lineares — o BERT já aprendeu, no pré-treino, representações que separam notícias reais de falsas de forma quase linear.

Esta observação tem implicação direta para os GNNs: se o espaço é linearmente separável, uma GNN que agrega vizinhos e passa por camadas lineares (GCNConv, SAGEConv) tem um limite superior natural próximo a este valor — a menos que a topologia do grafo forneça sinal *adicional* além do texto do nó raiz.

**No código:**
> Linha 91: `LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)` — `max_iter=1000` é necessário pois o espaço BERT-768 é de alta dimensão e o solver leva mais iterações para convergir.

**Embasamento acadêmico:**

> 📖 **Cox, D. R. (1958)** — "The Regression Analysis of Binary Sequences"
> *Journal of the Royal Statistical Society: Series B (Methodological)*, v. 20, n. 2, pp. 215–242
> DOI: `10.1111/j.2517-6161.1958.tb00292.x`
> **Localização:** Seção 1 (Introduction) — formulação do modelo logístico para variáveis binárias; a função sigmoide como modelo canônico para probabilidade binária
> **Relevância:** Paper original da regressão logística — fundamento teórico para o `LogisticRegression` do scikit-learn.

---

## 4. Random Forest sobre BERT-768

**Descrição técnica:**
Random Forest é um método de ensemble que treina $T$ árvores de decisão em amostras com reposição (*bootstrap*) do conjunto de treino, selecionando um subconjunto aleatório de features em cada nó de divisão. A predição final é a moda das predições das $T$ árvores:

$$\hat{y} = \text{majority\_vote}\!\left(\{h_t(\mathbf{x})\}_{t=1}^T\right)$$

**Vantagens sobre LogReg no contexto BERT-768:**
- Captura interações não-lineares entre dimensões do embedding
- Robusto a features irrelevantes (BERT-768 tem muitas dimensões com sinal baixo)
- Não requer normalização dos features

**Desvantagem:** Com 200 árvores e 768 features, o modelo tem complexidade O(T · N · log N · √d) no treino — mais lento que LogReg, porém ainda rápido para N≈314 grafos.

> Linha 92–93: `RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)` — 200 árvores, paralelizado em todos os cores disponíveis.

**Embasamento acadêmico:**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 1 (Random Forests) — definição do algoritmo; Seção 2 (Characterizing the Accuracy of Random Forests) — análise de variância e bias; Seção 5 (Random Input Selection) — justificativa para seleção aleatória de features em cada split
> **Relevância:** Paper original do Random Forest — fundamento teórico para o `RandomForestClassifier` do scikit-learn.

---

## 5. Resultados

### 5.1 Baseline Textual (K=10 folds)

| Modelo | F1-macro | F1-fake | Accuracy |
|--------|----------|---------|----------|
| LogisticRegression | **0.8592 ± 0.0431** | 0.8587 ± 0.0419 | 0.8595 ± 0.0430 |
| RandomForest       | 0.8549 ± 0.0372 | **0.8604 ± 0.0324** | 0.8555 ± 0.0369 |

### 5.2 Comparação Direta com GNNs (mesmos folds)

| Modelo | F1-macro (posfull) | Δ vs LogReg |
|--------|-------------------|-------------|
| LogisticRegression (baseline) | 0.8592 ± 0.0431 | — |
| RandomForest (baseline) | 0.8549 ± 0.0372 | −0.0043 |
| GCN | 0.8608 ± 0.0366 | +0.0016 |
| GAT | 0.8632 ± 0.0366 | +0.0040 |
| SAGE | 0.8612 ± 0.0370 | +0.0020 |

**Interpretação:** As diferenças entre GNNs e LogReg são de ordem 0.002–0.004 em F1, muito abaixo do desvio padrão dos folds (≈0.037–0.043). Não há evidência de que qualquer GNN supera o baseline textual de forma praticamente significativa.

### 5.3 Separabilidade linear do espaço BERT

O fato de que `LogReg` (modelo **linear**) e `RandomForest` (modelo não-linear) obtêm F1 praticamente idêntico (0.8592 vs 0.8549) indica que:
- Não há interações não-lineares relevantes entre dimensões do BERT que o RandomForest possa explorar além do que o LogReg já capta
- O espaço BERT-768 é bem condicionado para esta tarefa — o modelo linear já está próximo do ótimo representacional para as features disponíveis

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 O resultado negativo em números

O "teto textual" é F1≈0.86 com apenas `grafo.x[0]` (embedding do artigo raiz). As GNNs, que também têm acesso a `grafo.x[1:]` (embeddings dos nós vizinhos) e `grafo.edge_index` (estrutura do grafo), obtêm F1≈0.86 — o mesmo valor.

**Conclusão:** A informação adicional disponível para as GNNs (topologia + embeddings de outros nós) **não se traduz em ganho de F1 mensurável**. A hipótese central do TCC — de que a estrutura de propagação adiciona sinal discriminativo — não é sustentada pelos dados do FakeNewsNet PolitiFact.

### 6.2 Por que o resultado negativo é válido cientificamente

Resultados negativos têm valor científico quando:
1. O experimento é bem controlado (mesmos folds, mesmas features base, protocolo transparente) ✓
2. A hipótese era plausível a priori (a literatura sugere que estrutura de propagação pode ajudar) ✓
3. O poder estatístico é adequado (K=10 folds, com desvio ~0.037, detecta diferenças > ~0.04) ✓
4. Os dados têm qualidade suficiente para testar a hipótese (grafos reais do Twitter, FakeNewsNet é benchmark estabelecido) ✓

> 📖 **Krzywda et al. (2024)** reportam o mesmo padrão: GNNs atingem F1≈0.71–0.86 no FakeNewsNet enquanto Transformers (BERT, RoBERTa) atingem F1≈0.86–0.99. No dataset FakeNewsNet PolitiFact especificamente, o patamar de ~0.86 é consistente entre métodos — indicando saturação do sinal disponível neste dataset.

### 6.3 Resposta à questão central

> **"GNNs são uma alternativa viável para detecção de fake news?"**

O script 10 fornece a comparação mais limpa e direta do TCC: o modelo mais simples possível (LogReg linear sobre um único vetor BERT) obtém **o mesmo F1** que as GNNs mais sofisticadas (GCN, GAT, SAGE com 3 camadas, global_mean_pool, Dropout, Adam). A viabilidade das GNNs como alternativa à NLP pura **não é sustentada** neste domínio específico (PolitiFact/Twitter, grafos de retweet, features BERT).

A ressalva importante: isso não significa que GNNs *nunca* ajudam em fake news — significa que **neste dataset, com estas features, o sinal de propagação está ausente ou é redundante com o sinal textual**.

---

## 7. Análise de Código

### 7.1 Design correto

Este é o script mais limpo do pipeline. Pontos positivos:
- **Sem efeitos colaterais**: não salva pesos, não gera visualizações pesadas — apenas CSV e relatório texto
- **Mesmos folds**: `iterar_folds()` garante paridade com os scripts GNN para t-test pareado posterior
- **Dois classificadores**: LogReg e RandomForest cobrem o espectro linear/não-linear, tornando o baseline mais robusto
- **Métricas duplas (F1_macro e F1_fake)**: reporta tanto a métrica geral quanto a de interesse específico (detecção de Fake)

### 7.2 Limitações do design

**L1 — Ausência de SVM como terceiro baseline:**
SVM com kernel RBF seria o complemento natural para o trio Linear/Ensemble/Kernel. Especialmente relevante porque SVM é historicamente forte em espaços de alta dimensão como BERT-768.

**L2 — Sem normalização de features:**
`grafo.x[0]` não é normalizado antes de passar para LogReg/RF. LogReg scikit-learn aplica L2 internamente via `penalty='l2'` (padrão), mas RF não normaliza. Com BERT embeddings de escala variável, `StandardScaler` ou `normalize=True` poderia melhorar marginalmente o LogReg.

**L3 — Apenas nó raiz (x[0]):**
O baseline não testa `mean(g.x)` como feature alternativa. Se a média dos embeddings de todos os nós superasse `x[0]`, isso indicaria que os nós vizinhos *individualmente* carregam sinal — distinto de carregarem sinal *via estrutura de grafo*. Essa comparação adicional fortaleceria o argumento do resultado negativo.

**L4 — Sem teste de significância GNN vs baseline:**
O script 09 testa GCN vs GAT vs SAGE, mas não testa GNN vs LogReg com t-test pareado. Para a monografia, este seria o teste mais importante: `ttest_rel(GCN_f1_por_fold, LogReg_f1_por_fold)`. Como os F1s são quase idênticos (diferença ~0.002), o resultado seria ns — confirmando que GNNs não superam o baseline.

---

## 8. Referências Bibliográficas

1. DEVLIN, J.; CHANG, M. W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805` / ACL: `N19-1423`

2. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`

3. COX, D. R. **The Regression Analysis of Binary Sequences**. *Journal of the Royal Statistical Society: Series B*, v. 20, n. 2, pp. 215–242, 1958. DOI: `10.1111/j.2517-6161.1958.tb00292.x`

4. SHU, K. et al. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`

5. KOHAVI, R. **A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection**. *IJCAI'95*, pp. 1137–1143, 1995.

6. KRZYWDA, M. et al. **Comparative Analysis of GNNs and Transformers for Robust Fake News Detection**. *Electronics 2024*, 13(23), 4784. DOI: `10.3390/electronics13234784`

7. GONG, S. et al. **Fake News Detection Through Graph-based Neural Networks: A Survey**. arXiv: `2307.12639`, 2023.

8. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *NeurIPS 2017*. arXiv: `1706.02216`

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Teto textual | Desempenho máximo alcançável com apenas features textuais do artigo raiz — limite superior natural para comparação com GNNs | Conceito deste TCC |
| `grafo.x[0]` | Embedding BERT-768 do nó raiz do grafo (artigo/post original) | Script 00/02 |
| Separabilidade linear | Propriedade de um espaço de features onde as classes podem ser separadas por um hiperplano — verificada pelo F1≈0.86 do LogReg | — |
| Regularização L2 | Penalização $\lambda\|\mathbf{w}\|_2^2$ que evita overfitting em modelos lineares; padrão no `LogisticRegression` do scikit-learn | — |
| Bootstrap | Amostragem com reposição usada pelo RandomForest para gerar diversidade entre as T árvores | Breiman (2001), Seção 1 |
| Feature importance | Métrica do RandomForest que indica quais dimensões do BERT-768 mais contribuem para a classificação — não computada neste script mas disponível via `rf.feature_importances_` | Breiman (2001) |
| Resultado negativo | Resultado científico onde a hipótese principal não é confirmada; válido quando o experimento é bem controlado e o poder estatístico é adequado | — |
