# Erros e Incoerências Identificados — GNN Fake News TCC

**Data da compilação:** 20 de abril de 2026
**Fontes:** auditoria metodológica + análise cruzada código vs TCC (`GNN_TCC_atualizado.zip`)
**Severidade:** 🔴 Crítico · 🟡 Significativo · 🟠 Menor

---

## Parte I — Erros Metodológicos no Código

### Erro 1 — Features Nodais Idênticos 🔴

**Onde:** [00_construir_grafos_fakenewsnet.py:116](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py#L116), e em outros 4 arquivos ([treinar_mega_dataset.py:62](Training/03_Mega_Research/treinar_mega_dataset.py#L62), [matrizes_ablation.py:102](Training/03_Mega_Research/matrizes_ablation.py#L102), [matriz_confusao.py:79](Training/03_Mega_Research/matriz_confusao.py#L79), [analise_topologia.py:102](Training/03_Mega_Research/analise_topologia.py#L102), [Interface/frontend/api/main.py:252](Interface/frontend/api/main.py#L252))

```python
x = torch.stack([x_raiz] * num_nos)
```

O embedding BERT do artigo é copiado para todos os N+1 nós do grafo. Topologia é estrela pura (`src=0`, `dst=1..N`).

**Consequência matemática:**
- GCN com features idênticos colapsa em mapeamento linear `W · x_bert` (com escalonamento por grau)
- GAT: atenção sobre features idênticos produz softmax uniforme → equivale a GCN
- SAGE: `W_l·x + W_r·mean(x) = (W_l+W_r)·x` quando todos iguais

**Resultado:** GCN ≈ GAT ≈ SAGE ≈ classificador linear sobre BERT em capacidade expressiva efetiva.

**Nuance:** após 1 camada GCN, raiz (grau N) e folhas (grau 1) diferem por escalonamento da normalização `D^{-1/2} Ã D^{-1/2}`. Há alguma heterogeneidade, mas é função trivial da topologia, não semântica.

---

### Erro 2 — GNN não supera baseline textual 🔴

**Evidência empírica:**

| Modelo | F1 (Fake) | Accuracy |
|---|---|---|
| Regressão Logística — BERT puro | 0.8514 | 0.8553 |
| GraphSAGE (10 runs) | 0.8557 ± 0.0184 | 0.8592 |
| GAT (10 runs) | 0.8649 ± 0.0162 | 0.8665 |
| GCN (10 runs) | 0.8691 ± 0.0126 | 0.8704 |

P-values confirmados em [Execution/results/teste_significancia/relatorio.txt](Execution/results/teste_significancia/relatorio.txt):
- GCN vs GAT: p=0.628 (ns)
- GCN vs SAGE: p=0.170 (ns)
- GAT vs SAGE: p=0.158 (ns)

**Consequência:** a hipótese central do TCC ("topologia carrega informação discriminativa além do texto") não pode ser afirmada positivamente — os grafos construídos não permitem testá-la.

---

### Erro 3 — Número de tweets como confound espúrio 🟡

Distribuição de nós por classe no FakeNewsNet:
```
FAKE: mean=93.6, median=112, std=58.6
REAL: mean=83.3, median=78,  std=63.6
t-test: t=2.30, p=0.022 (significativo)
```

Artigos fake têm ~12% mais tweets que reais. Classificador treinado só no número de nós atinge F1=0.564.

**Consequência:** o modelo pode estar usando "popularidade" como proxy para "fake". Esta correlação é específica do PolitiFact; em outros domínios verdadeiros populares também teriam muitos tweets.

**Correção técnica à auditoria original:** o vazamento de N não é pelo `global_mean_pool` (que sobre features idênticos retorna o mesmo vetor independente de N). Ocorre **após a 1ª camada GCN**, porque a normalização de grau gera escalas diferentes entre raiz (grau N) e folhas (grau 1).

---

### Erro 4 — Metodologia do teste de significância 🟡

**Onde:** [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py)

Usa `scipy.stats.ttest_rel` com:
- Conjunto de teste fixo (152 grafos)
- 10 runs com seeds distintas

**Problema:** testa apenas variância de **inicialização de pesos**, não variância de **generalização**. A "paridade" do ttest_rel é sobre o mesmo test set, não sobre diferentes sub-populações dos dados.

**Metodologia acadêmica adequada:** k-fold cross-validation estratificada (k=5 ou 10), com ttest_rel sobre as k métricas por par de modelos.

**O resultado "ns" ainda é válido?** Sim, conservadoramente. Mas o teste tem baixa potência para detectar diferenças pequenas (std≈0.015, n=10). Com k-fold poderia detectar Δ=0.02 se existisse.

---

## Parte II — Incoerências entre TCC e Implementação

### Incoerência 5 — Conclusão central contradiz os próprios dados 🔴

**Onde:** [conclusao.tex:9](Material/GNN_TCC_atualizado/capitulos/conclusao.tex#L9)

> *"a geometria da rede é, de fato, uma assinatura de veracidade superior ao texto"*

**Contradição:** Regressão Logística sobre BERT puro atinge F1=0.8514; GNNs ficam em 0.855–0.869. Como features nodais são idênticos (Erro 1), as GNNs leem texto, não topologia. A frase inverte a conclusão suportada pelos dados.

---

### Incoerência 6 — "0% de erro em >50 interações" com interpretação causal errada 🔴

**Onde:** [conclusao.tex:9](Material/GNN_TCC_atualizado/capitulos/conclusao.tex#L9) e [resultados.tex:80-83](Material/GNN_TCC_atualizado/capitulos/resultados.tex#L80-L83)

Atribui o 0% de erro em grafos grandes a "GNN explorando a geometria". Como grafos grandes têm todos os nós com o mesmo vetor, o que está sendo explorado provavelmente é o **tamanho do grafo** — confound espúrio documentado no Erro 3.

---

### Incoerência 7 — Experimento FakeNewsNet existe no código mas sumiu do TCC 🔴

O repo tem [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py) e [relatorio.txt](Execution/results/teste_significancia/relatorio.txt) com comparativo GCN/GAT/SAGE em FakeNewsNet (10 runs, p-values). O TCC **não menciona** esse experimento — apesar de ser o único com metodologia estatística real.

---

### Incoerência 8 — GAT listado como trabalho futuro mas já está implementado 🟡

**Onde:** [conclusao.tex:20](Material/GNN_TCC_atualizado/capitulos/conclusao.tex#L20)

Propõe GAT como evolução futura. Mas os arquivos [03_treinar_gat.py](Training/03_Mega_Research/03_treinar_gat.py), [04_comparar_gcn_gat.py](Training/03_Mega_Research/04_comparar_gcn_gat.py), [06_comparar_gcn_gat_sage.py](Training/03_Mega_Research/06_comparar_gcn_gat_sage.py) e [gat_model.py](Training/03_Mega_Research/gat_model.py) já existem. GAT já foi treinado e comparado.

---

### Incoerência 9 — Convenção de labels inconsistente 🟡

- [metodologia.tex:47](Material/GNN_TCC_atualizado/capitulos/metodologia.tex#L47): *"Real (0) ou Fake (1)"*
- [00_construir_grafos_fakenewsnet.py:72-73](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py#L72-L73): `df_fake["label"]=0`, `df_real["label"]=1`
- [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py): usa `pos_label=0` (Fake)

Texto e código divergem em qual classe é 0 e qual é 1.

---

### Incoerência 10 — Ensemble não é diversidade real 🟡

**Onde:** [metodologia.tex:53-58](Material/GNN_TCC_atualizado/capitulos/metodologia.tex#L53-L58)

Os 4 modelos (UPFD, Ctrl, Cético, Exa-Cético) são **todos GCN** sobre **os mesmos features idênticos** (Erro 1). Diferem apenas em class weight (penalidade de FP). Isso é calibração, não ensemble arquitetural.

A justificativa "Exa-Cético tem menor FN" é consequência mecânica direta do peso 6x na loss, não achado empírico.

---

### Incoerência 11 — Inferência UPFD→Bluesky atribuída à causa errada 🟡

**Onde:** [resultados.tex:16](Material/GNN_TCC_atualizado/capitulos/resultados.tex#L16)

Atribui o colapso de F1 à "diferença de domínio Twitter vs Bluesky". Causa mais imediata:

- FakeNewsNet (treino): ~50% fake
- Bluesky (teste): 0.58% fake (194/33.382)

Modelo calibrado para 50% de prevalência aplicado a 0.58% → precision colapsa por excesso de FP. Problema de **prevalência de classes**, não de domínio.

---

### Incoerência 12 — Objetivo específico prometido e não cumprido 🟡

**Onde:** [introducao.tex:23](Material/GNN_TCC_atualizado/capitulos/introducao.tex#L23)

> *"Comparar o desempenho preditivo de modelos focados puramente em texto com modelos que integram texto e topologia"*

O baseline de texto puro (BERT sem grafo) não aparece no capítulo de resultados. O dado existe (F1=0.8514 da Regressão Logística), mas não foi reportado no TCC.

---

### Incoerência 13 — "Dataset massivo 21GB" vs experimento reportável 🟠

**Onde:** [metodologia.tex:17,25](Material/GNN_TCC_atualizado/capitulos/metodologia.tex#L17)

Narrativa de escala do Bluesky (21GB, 168k posts, 152M interações) não bate com o experimento que gerou resultados estatísticos (FakeNewsNet, CSV pequeno baixado do GitHub).

---

### Incoerência 14 — Fundamentação descreve GCN sem mencionar degeneração 🟠

**Onde:** [fundamentacao.tex](Material/GNN_TCC_atualizado/capitulos/fundamentacao.tex)

Regra `D^{-1/2} Ã D^{-1/2} H W` descrita corretamente, mas sem mencionar que sobre features idênticos degenera em transformação linear — que é exatamente o que ocorre nos experimentos.

---

### Incoerência 15 — Relatório afirma vitória do GAT sem evidência 🟡

**Onde:** `Execution/results/upfd_benchmark_fakenewsnet/relatorio.txt` (relatório auto-gerado por `07_upfd_benchmark_triplo.py`)

> *"GAT supera GCN e GraphSAGE... atenção diferencia super-spreaders"*

Três problemas:
1. Δ GAT vs GCN em F1 é −0.004 (GCN na verdade ganha), p=0.628
2. Com features idênticos, atenção GAT colapsa para uniforme; não há super-spreaders distinguíveis
3. Regressão Logística atinge desempenho equivalente, eliminando explicação via grafo

Lógica de auto-interpretação (`if vencedor == "GAT": ...`) gera narrativa causal sem base.

---

### Incoerência 16 — Early stopping com critério errado 🟠

**Onde:** [07_upfd_benchmark_triplo.py:199](Training/03_Mega_Research/07_upfd_benchmark_triplo.py#L199) (usa `val_acc`)

Métrica final de avaliação é F1, mas seleção de checkpoint é accuracy. Para FakeNewsNet balanceado, impacto é mínimo; em datasets desbalanceados selecionaria modelo que prediz sempre majoritária.

*Nota:* [09_teste_significancia.py:124](Training/03_Mega_Research/09_teste_significancia.py#L124) já usa F1 — inconsistência é entre scripts, não universal.

---

### Incoerência 17 — GCNClassifier duplicado 🟠

Reimplementado inline em `06_*.py`, `07_*.py`, `08_*.py` com seed hardcoded `12345`. Em `09_*.py` usa seed parametrizado. Comportamento arquitetural idêntico, mas inicialização diferente entre scripts. Deveria estar centralizado em `gcn_model.py` por simetria com `gat_model.py` e `sage_model.py`.

---

## Itens Corretos (preservar)

🟢 Split em nível de grafo (sem leakage)
🟢 `pos_label` explícito em métricas
🟢 Best-checkpoint restore no early stopping
🟢 `zero_division=0` em precision/recall/F1
🟢 Separação rigorosa de test set
🟢 Seed parametrizado em gat_model/sage_model/gcn_model (09)
🟢 Verificação de existência de arquivos antes de carregar
