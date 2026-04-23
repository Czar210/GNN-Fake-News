# Possibilidades de Correção — GNN Fake News TCC

Resumo das opções de correção para cada erro/incoerência de `erros_correcao.md`. Quando há trade-off, listo 2–3 alternativas ordenadas do menor ao maior esforço.

---

## Erros de Código

### Erro 1 — Features nodais idênticos

**A) Features estruturais por nó (sem nova coleta)** ⭐ recomendado
- Raiz: embedding BERT do título
- Filhos: vetor composto por [profundidade na árvore, índice temporal normalizado, grau do nó, one-hot da posição]
- Dá à GNN algo não-trivial para aprender; quebra a equivalência GCN=GAT=SAGE
- Esforço: ~1 dia para reescrever `00_construir_grafos_fakenewsnet.py`

**B) Coletar texto real dos tweets**
- Feature por nó = BERT do tweet individual
- Requer X API (paga) ou fontes alternativas (Twitter dumps antigos, Archive.org)
- Esforço: alto + custo financeiro

**C) Usar UPFD original com features de perfil**
- UPFD oficial inclui `profile`, `bert`, `spacy`, `content` por nó
- Links GDrive expiraram, mas dataset está no Zenodo
- Esforço: baixo se o download funcionar; substitui o experimento atual

---

### Erro 2 — GNN não supera baseline textual

**A) Assumir e reportar** ⭐ recomendado
- Rodar Regressão Logística e incluir como linha de base no capítulo de resultados
- Reframe: "GNNs neste setup não adicionam poder discriminativo sobre BERT" é resultado científico válido
- Esforço: 1h para script + 30min para revisar texto

**B) Corrigir Erro 1 antes**
- Só faz sentido reportar GNN vs BERT se os features forem heterogêneos
- Se corrigir Erro 1, reavaliar se o ganho volta

---

### Erro 3 — Confound de nº de tweets

**A) Reportar como limitação conhecida** ⭐ recomendado
- Adicionar seção "Limitações" mencionando o t-test entre distribuições de nós
- Citar o classificador baseline só com nº de nós (F1=0.564)

**B) Normalizar/controlar**
- Subamostrar para que fake e real tenham mesma distribuição de tamanhos
- Ou adicionar nº de nós como feature explícita em todos os modelos (inclusive o baseline logístico) — elimina o proxy

**C) Ablation**
- Rodar GNN em grafos artificialmente balanceados por tamanho e ver se o F1 se mantém

---

### Erro 4 — Metodologia de significância

**A) K-fold estratificado (5 ou 10 folds)** ⭐ recomendado
- Substitui 10 runs no mesmo split por 10 folds do dataset inteiro (754 grafos)
- Mesma chamada `ttest_rel` mas sobre métricas de folds diferentes
- Esforço: ~2h para adaptar `09_teste_significancia.py`

**B) Manter como está + reconhecer limitação**
- Adicionar parágrafo explicando que o teste mede variância de inicialização
- Aceitável se "ns" continuar sendo a conclusão
- Baixo esforço, mas banca pode cobrar

**C) Bootstrap sobre o test set**
- Para cada run, reamostrar o test set 1000x e calcular IC 95%
- Complementa (A); bom pra mostrar robustez

---

## Incoerências TCC ↔ Código

### Incoerência 5 — Conclusão central falsa

**A) Reescrever a conclusão honestamente** ⭐ recomendado
- "GNNs não superaram estatisticamente um classificador textual no setup avaliado, porque os grafos construídos carregam o conteúdo textual replicado nos nós. A hipótese permanece em aberto; este trabalho identifica a limitação e propõe caminhos para testá-la."
- Baixo esforço; alinha texto aos dados

**B) Corrigir Erros 1+2 e refazer experimentos**
- Se GNN passar a ganhar, a conclusão original pode ficar (com nuances)
- Alto esforço

---

### Incoerência 6 — "0% de erro em >50 interações"

**A) Atribuir ao tamanho do grafo, não à topologia** ⭐ recomendado
- Reinterpretar a figura como: "grafos com mais nós têm melhor performance, o que reflete o confund de popularidade identificado"
- Mais honesto, menos impressionante

**B) Remover a seção inteira**
- Se o confound não for controlável, a figura vira evidência contra, não a favor

**C) Controlar e refazer**
- Rodar a análise só em grafos fake com N comparável a grafos real
- Se o 0% persistir, o efeito é real

---

### Incoerência 7 — FakeNewsNet ausente do TCC

**A) Adicionar capítulo/seção dedicada** ⭐ recomendado
- É o único experimento com metodologia estatística; não incluir é desperdiçar o melhor resultado do repo
- Usar tabela do `relatorio.txt` diretamente
- Esforço: ~3h para escrever

**B) Substituir Bluesky por FakeNewsNet como experimento principal**
- Se Bluesky tem bug de pipeline (grafos vazios), FakeNewsNet deveria ser o foco
- Implica reescrita parcial da metodologia

---

### Incoerência 8 — GAT como trabalho futuro

**A) Mover GAT para capítulo de resultados** ⭐ recomendado
- Apresentar o comparativo GCN/GAT/SAGE como experimento principal ou secundário
- Trabalho futuro passa a ser algo realmente não-implementado (ex: temporal GNN)
- Baixo esforço

**B) Deletar a menção e manter silêncio**
- Não recomendado — omite trabalho feito

---

### Incoerência 9 — Labels inconsistentes

**A) Padronizar no texto para 0=Fake** ⭐ recomendado
- Alinha com UPFD, com FakeNewsNet, com scripts, com `pos_label`
- Esforço: find/replace na metodologia
- Justificativa natural: "seguimos convenção UPFD onde 0=Fake"

**B) Mudar no código**
- Muito mais risco; quebra scripts e resultados já gerados

---

### Incoerência 10 — Ensemble sem diversidade real

**A) Chamar de "calibração" em vez de "ensemble"** ⭐ recomendado
- Renomear para "Framework de Calibração por Custo Assimétrico"
- Explica honestamente: mesma arquitetura, pesos diferentes na loss
- Baixo esforço; tecnicamente mais correto

**B) Ensemble real heterogêneo**
- Substituir os 4 GCN por {GCN, GAT, SAGE, BERT-linear}
- Alto esforço; exige retreino

**C) Remover o ensemble do TCC**
- Se a tabela de pesos (10/20/30/40%) não tem justificativa empírica, simplesmente retirar

---

### Incoerência 11 — Inferência cruzada: prevalência, não domínio

**A) Reinterpretar a seção** ⭐ recomendado
- "A queda reflete primariamente o desbalanceamento de prevalência (50% → 0.58%), não diferença de plataforma"
- Adicionar números: Precision colapsa por FP excessivos
- Baixo esforço

**B) Reportar AUPRC em vez de F1**
- Métrica menos sensível a prevalência
- Precisa recalcular sobre as mesmas predições salvas

**C) Subamostrar o teste Bluesky**
- Avaliar só num subset com ~50% fake
- Resultado mais comparável; mas sample pequena (194 fakes)

---

### Incoerência 12 — Baseline textual não reportado

**A) Adicionar tabela de baselines no capítulo de resultados** ⭐ recomendado
- Linhas: Regressão Logística (BERT), Random Forest (BERT), GCN, GAT, SAGE
- Cumpre objetivo específico da introdução
- Esforço: 1–2h

**B) Remover o objetivo específico da introdução**
- Alinha texto com o que foi feito, mas descarta contribuição possível

---

### Incoerência 13 — "Dataset massivo 21GB"

**A) Separar narrativas por experimento** ⭐ recomendado
- Experimento 1: Bluesky (21GB) — resultado exploratório
- Experimento 2: FakeNewsNet — resultado estatístico
- Cada um com seus números próprios; narrativa de escala só vale para (1)

**B) Remover menções à escala**
- Dataset size não é diferencial do trabalho; tirar foco dele

---

### Incoerência 14 — Fundamentação omite degeneração

**A) Adicionar parágrafo na fundamentação** ⭐ recomendado
- "Um caso-limite importante: quando os features nodais são idênticos, a regra de propagação GCN degenera em transformação linear escalonada pela normalização de grau. Este fato será relevante na análise dos resultados..."
- Prepara o leitor; vira honestidade, não erro
- Esforço: ~15min

---

### Incoerência 15 — Relatório GAT auto-gerado

**A) Remover interpretações causais automáticas** ⭐ recomendado
- Editar `07_upfd_benchmark_triplo.py` para reportar só números, sem narrativa
- Se quiser narrativa, que seja manual, revisada

**B) Adicionar guardas estatísticas**
- Só afirmar "vencedor" se p<0.05 no ttest

---

### Incoerência 16 — Early stopping por accuracy

**A) Trocar para F1** ⭐ recomendado
- Alinha com métrica de avaliação final
- Esforço: 2 linhas de código em `07_upfd_benchmark_triplo.py`

---

### Incoerência 17 — GCNClassifier duplicado

**A) Extrair para gcn_model.py** ⭐ recomendado
- Simetria com `gat_model.py` e `sage_model.py`
- Aceitar `seed` como parâmetro em todos os chamadores
- Esforço: ~1h refactor

---

## Priorização Sugerida

Ordenado por impacto ÷ esforço:

**Fazer primeiro (ganho alto, custo baixo):**
1. Incoerência 5 — reescrever conclusão central (texto)
2. Incoerência 9 — padronizar labels no texto
3. Incoerência 8 — mover GAT para resultados
4. Incoerência 14 — parágrafo sobre degeneração GCN
5. Incoerência 16 — early stopping por F1
6. Incoerência 12 + Erro 2 — adicionar baseline Regressão Logística

**Fazer se tiver tempo:**
7. Incoerência 7 — incorporar experimento FakeNewsNet
8. Incoerência 10 — renomear "ensemble" para "calibração"
9. Incoerência 11 — reinterpretar inferência cruzada
10. Erro 4 — migrar para k-fold

**Só se banca cobrar (alto esforço):**
11. Erro 1 — corrigir features nodais (A ou C)
12. Incoerência 15 — desfazer relatório auto-narrativo
13. Incoerência 17 — refatorar GCNClassifier
