# Documentação Técnica: 19_aplicar_modelo_bluesky.py

## Metadados

- **Arquivo analisado:** `19_aplicar_modelo_bluesky.py`
- **Caminho:** `Training/03_Mega_Research/19_aplicar_modelo_bluesky.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Aplicação out-of-domain (cross-platform) de classificador estrutural — Random Forest treinado em GossipCop (UPFD) aplicado a posts do Bluesky **sem ground truth**
- **Modelos principais:** `RandomForestClassifier` estrutural (features `num_nodes`, `grau_root`) persistido em `Execution/weights/rf_struct_gossipcop.pkl` pelo script 17
- **Datasets utilizados:** Posts do Bluesky em `dados_bluesky/feed_posts/*.jsonl` (entrada); modelo RF previamente treinado em UPFD-GossipCop (não há treino aqui)
- **Contribuição para a questão central:** Demonstração **qualitativa** de transferência cross-platform. Sem rótulos no Bluesky, o script não mede F1/Accuracy. Em vez disso, produz (i) distribuições empíricas de score "fake-like" por feed, (ii) ranking de top-suspeitos como exemplos concretos para defesa do TCC, e (iii) seis visualizações HTML PyVis. Materializa o argumento da Fase 5 ("vulnerabilidade topológica") num cenário operacional realista — é a contraparte qualitativa do experimento quantitativo do script 08.

---

## 1. Visão Geral do Script

`19_aplicar_modelo_bluesky.py` executa **inferência apenas** (zero treino) de um Random Forest estrutural treinado em GossipCop sobre o universo total de posts Bluesky. Para cada post, reconstrói o vetor `[num_nodes, grau_root]` a partir dos campos `reply_count`, `repost_count` e (quando presente) `quote_count` do JSONL, aplica `predict_proba`, e armazena o score da classe Fake (label 0 na convenção UPFD).

A saída tem quatro componentes: (1) `scores_por_post.csv` — tabela longa com score de cada post; (2) `resumo_por_feed.csv` — agregado por feed (média, mediana, p95, % predito Fake); (3) `fig_distribuicao_scores.png` — histogramas sobrepostos por feed; (4) `top_suspeitos.txt` — top-5 posts mais "fake-like" por feed com texto truncado. Adicionalmente, são exportadas seis visualizações HTML PyVis de threads reais (3 alta-score + 3 score-mediano).

A natureza do experimento é **descritiva e ilustrativa**. Como Bluesky não possui rótulo nativo (vide PIPELINE.md script 01), nenhuma métrica supervisionada pode ser computada. O valor científico está em mostrar que o modelo, ao ser deslocado para uma plataforma e topologia distintas, produz uma distribuição **estruturada** (não uniforme, não colapsada), interpretável à luz das convenções de cada feed (cf. script 18). Trata-se de uma demonstração de **transfer learning indutivo** sem fine-tuning. O script encerra a sequência 17→18→19 da Fase 6: o 17 persiste o RF, o 18 caracteriza a topologia agregada por feed, e o 19 cruza ambos. Os scripts 22 e 23 consomem suas saídas para fechar a triangulação.

---

## 2. Conceito Central — Transfer Learning Indutivo Out-of-Domain

### 2.1 Formalização

Sejam $\mathcal{D}_S = \{(\mathbf{x}_i^S, y_i^S)\}_{i=1}^{n_S}$ o domínio fonte (UPFD-GossipCop, com rótulos) e $\mathcal{D}_T = \{\mathbf{x}_j^T\}_{j=1}^{n_T}$ o alvo (Bluesky, **sem rótulos**). O classificador $f_S: \mathbb{R}^2 \to [0,1]$ treinado em $\mathcal{D}_S$ é aplicado a $\mathcal{D}_T$ sem qualquer adaptação. Esta é a configuração canônica de **transfer learning indutivo cross-domain** sob ausência de rótulos no alvo.

**Embasamento acadêmico:**

> 📖 **Pan, S. J.; Yang, Q. (2010)** — "A Survey on Transfer Learning"
> *IEEE Transactions on Knowledge and Data Engineering*, v. 22, n. 10, pp. 1345–1359
> DOI: `10.1109/TKDE.2009.191`
> **Localização:** Seção 2.1 ("A Categorization of Transfer Learning Techniques"), Tabela 2 — relação entre tipos de transfer e domínios; Definição 2 (Transfer Learning); Seção 3 (Inductive Transfer Learning) — caso "no labeled data in target".
> **Relevância:** Define formalmente o cenário deste script (transferência sem rótulos no alvo) e justifica por que a única validação possível é qualitativa — não há perda nem métrica supervisionada.

### 2.2 Distinção vs. domain adaptation

O script **não** é domain adaptation: nenhuma técnica de alinhamento (importance weighting, MMD, adversarial DA) é aplicada. O modelo é deslocado *as-is* — o objetivo é mensurar o estado bruto da transferibilidade, não maximizá-la. Quiñonero-Candela et al. (2008) chamam essa configuração de *covariate shift* sob a hipótese conservadora: $P_T(\mathbf{x}) \neq P_S(\mathbf{x})$ mas $P(y|\mathbf{x})$ assumido invariante. No caso Bluesky, a invariância de $P(y|\mathbf{x})$ é a **hipótese a ser testada**, não verificada.

> 📖 **Quiñonero-Candela, J.; Sugiyama, M.; Schwaighofer, A.; Lawrence, N. D. (eds.) (2008)** — "Dataset Shift in Machine Learning"
> *MIT Press*, 2008. ISBN 978-0262170055
> **Localização:** Capítulo 1 (Introduction), Seção 1.3 — taxonomia de tipos de shift (covariate, prior, concept); Capítulo 2, Seção 2.1 — formalização $P_S(x,y) \neq P_T(x,y)$.
> **Relevância:** Vocabulário formal para descrever o shift que afeta a aplicação. A hipótese realista combina *covariate shift* (estrutura de threads diferente entre plataformas) **e** *prior probability shift* (proporção de classes diferente, vide script 08). O fato de o script 19 não corrigir nenhum dos dois é o que o torna demonstração e não validação.

---

## 3. Random Forest Estrutural

### 3.1 Modelo e formulação

`carregar_rf()` (linhas 48–51) recupera do pickle: o `RandomForestClassifier`, lista de features (`["num_nodes", "grau_root"]`) e mapeamento de rótulos (`{0: "fake", 1: "real"}`). Cada árvore $t \in \{1,\dots,T\}$ é treinada sobre bootstrap sample $\mathcal{D}_S^{(t)}$, com split em subconjunto aleatório $\mathcal{F}_t$ de features ($|\mathcal{F}_t| \approx \sqrt{d}$ default em sklearn). Predição agregada:

$$\hat{p}(y=c \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} \hat{p}_t(y=c \mid \mathbf{x})$$

**Embasamento acadêmico:**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 1 — definição $\{h(\mathbf{x}, \Theta_t)\}$; Equação (1) — limite superior $PE^* \leq \bar{\rho}(1 - s^2)/s^2$; Seção 4 ("Forest-RI") — algoritmo implementado em `sklearn.ensemble.RandomForestClassifier`; Teorema 2.3 — convergência quase certa do erro com $T \to \infty$.
> **Relevância:** Fundamenta tanto o uso de RF (robustez, baixa variância via averaging, sem necessidade de calibração explícita em ambientes com poucas features) quanto a interpretação de `predict_proba` como média sobre árvores i.i.d.

### 3.2 Por que RF e não SAGE?

Embora o script 17 também persista um SAGE estrutural, o 19 usa o RF por três motivos (PIPELINE.md, racional do 17): (i) **custo** — Bluesky tem ~166k posts, RF batch é segundos; SAGE exigiria construir `Data` por post; (ii) **interpretabilidade** — feature importance é trivial; (iii) **robustez OOD** — RF degrada graciosamente com splits axis-aligned, enquanto SAGE depende da topologia vista no treino (Hamilton, 2020, Cap. 5.4 sobre limitações de generalização cross-graph).

> 📖 **Hamilton, W. L. (2020)** — "Graph Representation Learning"
> Morgan & Claypool, v. 14, n. 3. ISBN 978-1681739632
> **Localização:** Capítulo 5 ("The Graph Neural Network Model"), Seção 5.4 — limitações de generalização cross-graph quando estatísticas estruturais divergem entre treino e teste.
> **Relevância:** Sustenta a escolha por RF: não tem dependência funcional do tamanho/estrutura do grafo além das features marginalizadas, enquanto SAGE depende explicitamente da vizinhança.

### 3.3 No código

> Linhas 105–112: `X = np.array(...)` constrói $X \in \mathbb{R}^{n \times 2}$; `probs = rf.predict_proba(X)` retorna $\hat{P} \in \mathbb{R}^{n \times 2}$ alinhado a `rf.classes_`. A linha 110 — `idx_fake = classes.index(0) if 0 in classes else 0` — isola a coluna Fake (label 0 em UPFD). O lookup explícito é defensivo: se o RF for re-treinado e a ordem de classes mudar, o significado do score não troca silenciosamente.

---

## 4. Engenharia de Features — Marginalização da Topologia

Linhas 54–63 implementam:

$$\text{num\_nodes}(p) = 1 + \text{reply\_count}(p) + \text{repost\_count}(p) + \mathbb{1}[\text{quotes} \in \mathbb{Z}] \cdot \text{quotes}(p)$$
$$\text{grau\_root}(p) = \text{num\_nodes}(p) - 1$$

A semântica é a "estrela plana" (PIPELINE.md, Fase 0): cada post é um grafo onde a raiz é o post e cada interação é um nó folha conectado por aresta única à raiz. Toda hierarquia conversacional (reply de reply, cadeia de repostes) é colapsada num só nível.

Sob essa redução, `num_nodes` e `grau_root` são **funções determinísticas uma da outra**: $\text{grau\_root} = \text{num\_nodes} - 1$. O RF opera efetivamente em **uma feature univariada** disfarçada de bivariada — qualquer split em `grau_root` é equivalente a um split em `num_nodes - 1`. Essa redundância é diagnóstica: o classificador estrutural reduz-se essencialmente a um threshold sobre o tamanho da cascata, conectando-se ao argumento do script 11 (confound diagnostic, F1>0.65 só com `num_nodes`).

O padrão `int(post.get("reply_count") or 0)` (linhas 56–57) trata simultaneamente: chave ausente, valor `null`, valor numérico/string. O `or 0` converte `None` em 0 antes do cast, evitando `TypeError`.

---

## 5. Score "Fake-like" — Definição e Caveat de Calibração

### 5.1 Definição operacional

`score_fake = ` $\hat{P}_{RF}(y=0 \mid \mathbf{x}) \in [0,1]$ é a probabilidade preditiva da classe Fake estimada pelo RF (linhas 111, 114–116). A predição categórica `pred` (linha 112) usa threshold 0.5 via `rf.predict(X)`, equivalente a $\arg\max_c \hat{P}(y=c|\mathbf{x})$.

### 5.2 Caveat metodológico — Calibração

Random Forests **não produzem probabilidades calibradas em geral** sobre o domínio de treino, e **não há razão a priori** para esperar calibração sobre um domínio deslocado. Niculescu-Mizil & Caruana (2005) demonstram que RFs empurram probabilidades para o centro do intervalo (oposto do boosting), exigindo calibração isotônica/Platt para uso decisório. Aqui, nem mesmo a calibração in-domain foi feita (script 17), e o modelo é aplicado out-of-domain.

Portanto, **`score_fake` no Bluesky não deve ser interpretado como probabilidade**. É um *score* ordinal cuja única propriedade utilizada é a relação de ordem: posts com score=0.85 são "mais suspeitos" que posts com score=0.32, no sentido de que caem em folhas onde a maioria dos votos (no treino GossipCop) era Fake. Essa é a interpretação que justifica o uso primário do script: o ranking dos top-5 e a inspeção da distribuição (não do valor absoluto).

**Embasamento acadêmico:**

> 📖 **Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. (2017)** — "On Calibration of Modern Neural Networks"
> *ICML 2017*, PMLR 70:1321–1330. arXiv: `1706.04599`
> **Localização:** Seção 2.1 — *perfectly calibrated*: $P(\hat{Y} = Y \mid \hat{P} = p) = p$ para todo $p \in [0,1]$; Seção 2.2 — *Expected Calibration Error* (ECE), Equação (3); Seção 4 — modelos de alta acurácia podem estar mal-calibrados.
> **Relevância:** A definição formal e a métrica ECE são as referências modernas para discutir por que `score_fake = 0.7` não significa "70% de chance de fake" no Bluesky. A literatura sobre RF (Niculescu-Mizil & Caruana, 2005) chega à mesma conclusão: scores brutos não satisfazem a definição de calibração.

> 📖 **Niculescu-Mizil, A.; Caruana, R. (2005)** — "Predicting Good Probabilities With Supervised Learning"
> *ICML 2005*, pp. 625–632. DOI: `10.1145/1102351.1102430`
> **Localização:** Seção 3 (Reliability Diagrams), Figura 1 — RF sem calibração: empilhamento longe dos extremos; Seção 4 — calibração isotônica recomendada para RFs.
> **Relevância:** Explica o padrão (concentração de scores no meio do intervalo) que provavelmente aparece em `fig_distribuicao_scores.png`, e justifica não interpretar `pct_pred_fake` como prevalência verdadeira de fake nos feeds.

---

## 6. Análise Distribucional por Feed

`resumo_por_feed.csv` (linhas 130–144) reporta por feed: `n_posts`, `score_medio`, `score_mediana`, `score_p95`, `pct_pred_fake`. Justificativa estatística: **média** captura o nível central mas é sensível a cauda direita pesada (típica de score viral); **mediana** descreve o "post típico" robustamente; **p95** caracteriza a cauda direita — onde estão os "casos extremos" para fact-checking manual; **pct_pred_fake** é threshold-binarizado a 0.5 (caveat de calibração §5.2 aplica).

A `fig_distribuicao_scores.png` (linhas 171–188) sobrepõe histogramas normalizados (`density=True`) por feed. O filtro `if len(sub) < 30: continue` (linha 176) descarta feeds com poucos posts onde KDE/hist têm alta variância. A figura é o instrumento principal para responder: **a distribuição de scores difere entre feeds?**

Reimers & Gurevych (2017) defendem o uso de distribuições (em vez de pontos) como ferramenta diagnóstica em NLP/IR — quando ground truth não está disponível, comparar distribuições preditivas entre subgrupos detecta heterogeneidade de comportamento do modelo, sinal indireto de captura de estrutura real.

> 📖 **Reimers, N.; Gurevych, I. (2017)** — "Reporting Score Distributions Makes a Difference"
> *EMNLP 2017*, pp. 338–348. DOI: `10.18653/v1/D17-1035` / arXiv: `1707.09861`
> **Localização:** Seção 3 ("Reporting Test Performance"), Figura 1 — exemplos de distribuições entre runs; Seção 5 ("Recommendations") — distribuições como objeto de primeira classe na análise.
> **Relevância:** Fundamenta reportar histogramas por feed em vez de só média/dp. Generaliza para "distribuição como objeto diagnóstico": se feeds têm distribuições visivelmente distintas (e.g., feed político centrado em score baixo, feed aberto em score moderado), é evidência de que o RF captura estrutura dependente do feed — coerente com a tese do TCC.

---

## 7. Top-Suspeitos e Visualização — Função Retórica

Linhas 158–168 ordenam os posts por score decrescente dentro de cada feed e gravam os 5 mais altos com `score`, `num_nodes`, `like_count` e os primeiros 200 caracteres do texto. O objetivo é fornecer **exemplos concretos** para a banca: ao apresentar a tese da vulnerabilidade topológica, ter posts reais ranqueados pelo modelo permite mostrar que (i) o modelo seleciona consistentemente posts com cascatas grandes, (ii) o conteúdo desses posts é frequentemente plausível como fake (ou claramente não-fake, evidenciando o problema do classificador estrutural).

Linhas 191–218 selecionam 6 amostras (3 alta-score + 3 mediano), todas com `num_nodes >= 5` (linha 198), gerando HTML interativo via `pyvis.network.Network`. A topologia desenhada é a estrela plana — RAIZ central laranja conectada a `num_nodes - 1` vizinhos verdes (linhas 208–211). Função puramente didática/retórica: tornar tangível o que "score alto" significa em termos de propagação real, sem exigir reconstrução mental do grafo.

---

## 8. Contexto Bluesky / AT Protocol

O Bluesky é uma rede social descentralizada baseada no AT Protocol, lançada publicamente em fevereiro de 2024. Cada post é assinado criptograficamente e armazenado em um *Personal Data Server* (PDS) sob controle do usuário; o protocolo define tipos de registros (`app.bsky.feed.post`, `repost`, `like`) e relações (reply via `reply.parent.uri`, repost via referência ao original).

Três aspectos relevantes para este script:
1. **Múltiplos feeds custom**: diferente do X/Twitter, há vários feeds curados por operadores. `feed_posts/*.jsonl` corresponde a esses feeds — estatísticas por feed (script 18) capturam diferenças algorítmicas de curadoria.
2. **Granularidade de interação**: `reply_count`, `repost_count`, `quote_count` são contagens diretas mantidas pelo PDS; o script consome sem reconstruir a árvore — fonte da redução para "estrela plana".
3. **Ausência de rótulos fake/real nativos**: ao contrário de FakeNewsNet (PolitiFact/GossipCop com checagem manual), Bluesky não rotula. A rotulagem heurística por keywords (script 01) é proxy explicitamente fraco — daí a impossibilidade de validação quantitativa.

A documentação oficial do AT Protocol (atproto.com/specs/atp) é a fonte normativa; não há ainda paper Tier-1 sobre o protocolo, por isso o tratamento aqui é descritivo.

---

## 9. Análise Empírica — Posicionamento na Questão Central (Seção 6.4 do TCC)

### 9.1 O que este script demonstra

1. **Operacionalidade da transferência cross-platform**: o RF roda sobre 166k posts em segundos, produzindo scores ordinais interpretáveis. Não colapsar (não retornar a mesma probabilidade para todos) já é evidência de que `num_nodes/grau_root` discriminam dentro do Bluesky — argumento para a tese de "atalho topológico universal".
2. **Heterogeneidade entre feeds**: variação de média/p95 entre feeds (e.g., políticos vs. científicos) corroboraria que o classificador captura propriedades estruturais correlacionadas com tipo de conteúdo, mesmo sem ler o texto.
3. **Material qualitativo concreto para defesa**: top-suspeitos + 6 visualizações HTML são artefatos diretamente apresentáveis na banca.

### 9.2 O que este script NÃO demonstra (limitações inegociáveis)

1. **Não mede acurácia**: sem ground truth, qualquer afirmação sobre "X% dos posts no Bluesky são fake" é inválida. `pct_pred_fake` reflete o threshold do modelo, não prevalência real.
2. **Não corrige prior probability shift**: como demonstrado no script 08, aplicação de modelos UPFD ao Bluesky sofre colapso de Precision por diferença drástica de prevalência (treino ~50% Fake vs. heurística ~0.58%). O score do RF herda esse desbalanceamento.
3. **Não valida invariância de $P(y|\mathbf{x})$**: pressuposto fundamental do covariate shift puro é que a relação features→label é a mesma em ambos os domínios. Sem rótulos no Bluesky, não testável — apenas assumido.
4. **Não distingue overfitting topológico de captura genuína**: se o RF aprendeu apenas que "GossipCop tem fakes virais", classificaria como fake qualquer post viral no Bluesky, independente do conteúdo. Sozinho, o script 19 não distingue esses cenários — é o script 22 (concordância com BERT) que oferece o teste cruzado.

### 9.3 Conexão com Seção 6.4 do TCC

Este script é **demonstração qualitativa, não validação quantitativa**. Sua função tripla na arquitetura do TCC:

- **Operacionalizar a Fase 5**: dar substrato visual e tabular ao argumento de "vulnerabilidade topológica universal" — o RF treinado em GossipCop, aplicado a uma plataforma que nunca viu (Bluesky pós-2024, AT Protocol), continua produzindo scores estruturados. Exatamente o predito pela tese: se o atalho topológico está embutido na coleta de datasets de fake news, ele se manifesta universalmente.
- **Servir de input para 22 e 23**: `scores_por_post.csv` é a base para a análise de concordância (22) com o classificador textual e para visualizações finais (23).
- **Conectar a tese ao plano operacional**: ao mostrar que o modelo "funciona" (produz saída consistente) numa plataforma totalmente diferente sem ajuste, reforça que o problema diagnosticado nos Caps. 4–5 não é artefato de um dataset específico — é propriedade da abordagem estrutural.

A resposta parcial à pergunta do TCC ("GNNs vs. NLP — viáveis?") é: **a aplicabilidade out-of-domain de classificadores estruturais leves é alta operacionalmente, mas duvidosa epistemicamente**. O modelo produz scores; mas sem ground truth e sem calibração, esses scores carregam pouca informação acionável. Compare com classificadores textuais BERT, cujos scores têm interpretação semântica localmente verificável post-a-post. A vantagem operacional do RF estrutural (velocidade, simplicidade) é contrabalançada pela opacidade epistêmica fora do domínio de treino.

---

## 10. Análise de Código

### 10.1 Erros e fragilidades

**E1 — Hipótese implícita sobre `rf.classes_`** (linhas 109–110): a cláusula `else 0` é fallback que silenciosamente assume "primeira coluna = fake" se classe 0 não existir. Defensivo demais — pode mascarar bug se o pickle foi gerado com convenção invertida. Correção: lançar `ValueError` se `0 not in classes`.

**E2 — `feature_names` carregado mas não usado**: linha 75 desempacota `feat_names` mas só imprime (linha 76). A construção de X (linha 106) usa hard-coded `["num_nodes", "grau_root"]` — se o RF for re-treinado com features adicionais, este script silenciosamente usa apenas as duas hard-coded. Correção: iterar sobre `feat_names`.

**E3 — `pred` baseado em threshold 0.5 sem aviso** (linha 112): em classificadores não-calibrados aplicados out-of-domain, esse threshold é arbitrário. `pct_pred_fake` (linha 138) herda essa arbitrariedade.

**E4 — Tratamento silencioso de PyVis ausente** (linhas 192–195): aceitável em produção, mas em pipeline de TCC o esperado é falha explícita — ausência de visualização passa despercebida.

**E5 — Falta de seed/hash do pickle**: o RF é determinístico após carregamento, mas o script não imprime `rf.random_state` nem hash SHA256 do pickle. Para auditoria, é útil saber qual versão foi usada.

### 10.2 Boas práticas observadas

- **Estrutura clara em 4 fases (`[1/4]`…`[4/4]`)**: facilita debugging e audita progresso.
- **CSVs com `extrasaction="ignore"`** (linha 123): permite adicionar campos ao dict sem quebrar o writer.
- **`density=True` no histograma** (linha 178): comparação entre feeds com tamanhos muito diferentes só é informativa quando normalizada.
- **Filtro `if len(sub) < 30`** (linha 176): regra de bolso para evitar densidade espúria.
- **Truncagem de texto em 200 chars** (linha 97): evita CSVs ilegíveis e protege contra posts maliciosamente longos.
- **Separação `top_suspeitos.txt` vs. `scores_por_post.csv`**: inspeção humana imediata vs. análises downstream.

### 10.3 Complexidade

Carregamento JSONL: $O(N)$ com $N \approx 1.66 \times 10^5$. `rf.predict_proba(X)`: $O(T \cdot \log(\bar{n}_{\text{leaf}}) \cdot N)$. Total dominado por I/O — segundos a poucos minutos.

---

## 11. Referências Bibliográficas

1. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`.

2. PAN, S. J.; YANG, Q. **A Survey on Transfer Learning**. *IEEE Transactions on Knowledge and Data Engineering*, v. 22, n. 10, pp. 1345–1359, 2010. DOI: `10.1109/TKDE.2009.191`.

3. QUIÑONERO-CANDELA, J.; SUGIYAMA, M.; SCHWAIGHOFER, A.; LAWRENCE, N. D. (eds.). **Dataset Shift in Machine Learning**. Cambridge, MA: MIT Press, 2008. ISBN 978-0262170055.

4. GUO, C.; PLEISS, G.; SUN, Y.; WEINBERGER, K. Q. **On Calibration of Modern Neural Networks**. *ICML 2017*, PMLR 70:1321–1330. arXiv: `1706.04599`.

5. NICULESCU-MIZIL, A.; CARUANA, R. **Predicting Good Probabilities With Supervised Learning**. *ICML 2005*, pp. 625–632. DOI: `10.1145/1102351.1102430`.

6. REIMERS, N.; GUREVYCH, I. **Reporting Score Distributions Makes a Difference**. *EMNLP 2017*, pp. 338–348. DOI: `10.18653/v1/D17-1035` / arXiv: `1707.09861`.

7. MORENO-TORRES, J. G.; RAEDER, T.; ALAIZ-RODRÍGUEZ, R.; CHAWLA, N. V.; HERRERA, F. **A Unifying View on Dataset Shift in Classification**. *Pattern Recognition*, v. 45, n. 1, pp. 521–530, 2012. DOI: `10.1016/j.patcog.2011.06.019`.

8. HAMILTON, W. L. **Graph Representation Learning**. Morgan & Claypool, v. 14, n. 3, 2020. ISBN 978-1681739632.

9. SHU, K. et al. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

10. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *SIGIR'21*, 2021. DOI: `10.1145/3404835.3462990` / arXiv: `2104.12259`. *(dataset UPFD-GossipCop usado para treinar o RF)*

---

## 12. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Transfer learning indutivo (sem labels no alvo) | $f_S$ aprendido em $\mathcal{D}_S$ com rótulos, aplicado em $\mathcal{D}_T$ sem rótulos/ajuste/perda supervisionada. | Pan & Yang (2010), Seção 3 |
| Out-of-domain (OOD) inference | Aplicação a dados cuja distribuição $P_T(\mathbf{x},y)$ difere da de treino $P_S$. | Quiñonero-Candela et al. (2008), Cap. 1 |
| Covariate shift | $P(\mathbf{x})$ muda mas $P(y\|\mathbf{x})$ é invariante. | Moreno-Torres et al. (2012), Seção 2 |
| Prior probability shift | $P(y)$ muda mas $P(\mathbf{x}\|y)$ é invariante — diagnosticado no script 08 para UPFD→Bluesky. | Moreno-Torres et al. (2012), Seção 3 |
| Random Forest | Ensemble de árvores em bootstrap samples, com seleção aleatória de features por split; agregação por média. | Breiman (2001), Seção 4 |
| Calibração | $P(\hat{Y}=Y \mid \hat{P}=p) = p$ para todo $p$. RFs em geral não são calibrados sem pós-processamento. | Guo et al. (2017), Seção 2.1; Niculescu-Mizil & Caruana (2005) |
| Score "fake-like" | $\hat{P}_{RF}(y=0\|\mathbf{x}) \in [0,1]$ usado para ranking; **não é probabilidade calibrada** cross-platform. | Definição operacional + Guo et al. (2017) |
| Estrela plana | Topologia onde a raiz conecta a todos os demais nós por arestas únicas, sem hierarquia interna. | PIPELINE.md, scripts 00 e 02 |
| Bluesky / AT Protocol | Rede social descentralizada com registros assinados criptograficamente; sem rótulos fake/real nativos. | atproto.com/specs/atp |
| ECE | *Expected Calibration Error*: erro esperado entre confiança e acurácia, sobre bins de probabilidade. | Guo et al. (2017), Equação 3 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 19_aplicar_modelo_bluesky.py
📄  Arquivo gerado: theory_andre/19_aplicar_modelo_bluesky_doc.md
📚  Fontes acadêmicas utilizadas: 10
    1. Breiman (2001) — Random Forests
    2. Pan & Yang (2010) — A Survey on Transfer Learning
    3. Quiñonero-Candela et al. (2008) — Dataset Shift in ML (MIT Press)
    4. Guo et al. (2017) — On Calibration of Modern Neural Networks
    5. Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities
    6. Reimers & Gurevych (2017) — Reporting Score Distributions
    7. Moreno-Torres et al. (2012) — A Unifying View on Dataset Shift
    8. Hamilton (2020) — Graph Representation Learning
    9. Shu et al. (2020) — FakeNewsNet
    10. Dou et al. (2021) — UPFD
🔍  Conceitos cobertos:
    - Transfer learning indutivo cross-platform sem labels no alvo
    - Random Forest (Breiman, predict_proba como média sobre árvores i.i.d.)
    - Covariate / prior probability shift
    - Calibração e por que score_fake não é probabilidade
    - Score distributions como diagnóstico em ausência de ground truth
    - Engenharia de features estrutural e marginalização da topologia
    - Contexto Bluesky / AT Protocol e implicações para validação
    - Conexão Fase 6.4 do TCC: demonstração qualitativa da vulnerabilidade topológica universal
⚠️  Limitações:
    - AT Protocol/Bluesky sem paper Tier-1 acadêmico; usado só spec oficial.
    - Sem rótulos no Bluesky, qualquer afirmação sobre acurácia é proibida — limitação do experimento.
    - Calibração específica do RF (script 17) não auditada com reliability diagram (fora do escopo).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```
