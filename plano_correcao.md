# Plano de Correção e Execução — GNN Fake News TCC

**Data:** Atualizado (Abril 2026)
**Estratégia:** Provar ou refutar, com rigor estatístico pareado, que a topologia do FakeNewsNet adiciona sinal discriminativo além da regressão base textual (BERT).

**Mapeamento de Incoerências e Erros:**
- Fase 1: Incoerências 15, 16, 17.
- Fase 2: Erros 2, 3, 4 e Incoerência 12 (experimento).
- Fase 3: Erro 1.
- Fase 4: Incoerência 11 (parcial).
- Fase 5: Incoerências 5, 6, 7, 8, 9, 10, 11 (final), 12 (relato), 13, 14 + Erro 1 (parte teórica/fundamentação).

**Cronograma e Priorização:**
- **Semana 1:** Fases 1, 2A, 2B e 5A (Infraestrutura, Baselines e Redação teórica).
- **Semana 2:** Fases 2C e 3 (Avaliação de Confound e Positional Encodings).
- **Semana 3:** Fase 4 (Testes de Significância Finais).
- **Semana 4:** Fase 5B (Redação e Conclusões sobre Resultados).

---

## Fase 1 — Refatoração Base e Infraestrutura
> **Objetivo:** Preparar a base de código removendo redundâncias lógicas.

- [ ] **1.1. Consolidar Imports (Incoerência 17)**
  - **Ação:** Substituir as reimplementações inline da classe `GCNClassifier` pelo import absoluto `from gcn_model import GCNClassifier` nos scripts (06, 07, 08, 09).
  - **Entregável:** Código revisado e arquivo script salvo.
- [ ] **1.2. Mudar Early Stopping para F1 (Incoerência 16)**
  - **Ação:** Em todos os loops de treinamento de `07` e afins, confirmar formalmente o critério de validação `val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)`. Manter a lógica de restauração do melhor checkpoint, agora baseada em `val_f1` em vez de `val_acc`.
  - **Entregável:** Ajuste do scheduler para utilizar f1_macro e best-checkpoint restore referenciando a mesma métrica.
- [ ] **1.3. Remover Narrativas Causais (Incoerência 15)**
  - **Ação:** Remover logs textuais interpretativos automáticos (ex: "GAT venceu") de `07_upfd_benchmark_triplo.py`.
  - **Entregável:** Log modificado, puramente numérico e tabular.

---

## Fase 2 — Baselines, Pareamentos e Violações de Confound

### Fase 2A: Teto Textual Rigoroso (Erro 2 / Inc. 12)
- [ ] **2A.1. Baseline Oficial**
  - **Ação:** Criar `10_baseline_textual.py`. Aplicar `LogisticRegression` e `RandomForest` de modo isolado no texto do nó matriz (`X[0]`).
  - **Entregável:** Arquivo `10_baseline_textual.py` criado e log de console documentando o limite textual estrito do BERT.

### Fase 2B: Pareamento por K-Fold (Erro 4-A)
- [ ] **2B.1. Compartilhamento de Folds**
  - **Ação:** O mecanismo `StratifiedKFold(n_splits=10, random_state=42)` será instanciado UMA vez e salvo. Baseline de texto, GCN original e GNN Estrutural consumirão os mesmos índices garantindo variação emparelhada (`ttest_rel`).
  - **Entregável:** Arquivo `09_teste_significancia.py` modificado iterando sobre uma variação de divisões indexada unificada.

### Fase 2C: Diagnóstico de Confound (Erro 3-B/3-C)
- [ ] **2C.1. Diagnóstico do confound de tamanho**
  - **Ação Rápida:** Treinar `RandomForest` definindo a feature extra explicitamente como `X = concat(bert_raiz, [num_nodes])`.
  - **Gate de Decisão:** Executa o Subsampling Pareado local se o modelo baseline-por-N atingir F1 superior a 0.65.
  - **Entregável:** Relatório inferindo quantitativamente a força base de um sistema preditor observando unicamente o grau do nó raiz.

---

## Fase 3 — Positional Encodings Estrela (Erro 1 - Opção A)
> **Objetivo:** Injetar features posicionais que não colapsam sob a agregação GCN.
> **Gate Prévio:** Avançar para a alteração estrutural somente se baseline textual F1 for inferior a 0.90 (se for superior a 0.90, GNN não tem espaço para contribuir e a Fase 3 vira um limitante analítico provado).

- [ ] **3.0. Arquivamento do Baseline Bugado**
  - **Ação:** Criar cópia de segurança renomeando e isolando as pastas .pt originais contendo os vetores replicados. Servirá para o grupo de amostra da variação Original de Controle.
  - **Entregável:** Diretórios copiados num local como `fakenewsnet_bugado` garantindo arquivo.
- [ ] **3.1. Design das features posicionais (Erro 1-A)**
  - **Ação:** Reescrever rotina do construtor de grafos na geração 00 original.
  - **Design da Raiz:** Incorporará o encoding is_root explícito mantendo normalizações externas exclusivas: `[BERT(title) || is_root=1, grau_out_norm=N/N_max_global, pos=0.0]`. Dimensão: 771.
  - **Design do Filho (i listado na ordem temporal):** `[BERT(title) || is_root=0, grau_out_norm=0, pos=i/N]`. Dimensão: 771.
  - **Entregável:** Script `00_construir_grafos_fakenewsnet.py` finalizado contendo `assert x.unique(dim=0).shape[0] >= min(num_nos, 3)` prevenindo features nodais idênticos.
- [ ] **3.1.bis. Diagnóstico Experimental Intra-Encoding**
  - **Ação:** Opor isoladamente arquiteturas experimentais de ablação estrutural de grafos. Versão `Pos-min` isola posição sem grau. Versão `Pos-grau` tenta aferir impacto só do volume.
  - **Gate de Posição:** Se F1 da matriz `Pos-min` superar o `Baseline` com `p < 0.05`, provamos ganho atribuível ao encoding posicional puramente direcional.
  - **Gate de Confound:** Se `F1(Pos-grau) > F1(Pos-min) + 0.01`, sinal detectado é majoritariamente volumétrico; reconhecer como confirmação do Erro 3 na Fase 5B.2.
  - **Entregável:** Tabela comparativa pareada das variações estruturais testadas de forma individual.
- [ ] **3.2. Regenerar arquivos .pt**
  - **Ação:** Rodar script 00_construir_grafos_fakenewsnet.py.
  - **Entregável:** Arquivos .pt recriados com os features posicionais.

---

## Fase 4 — Nova Reavaliação Preditiva e Multi-Domínio
> **Objetivo:** Executar os benchmarks finais nos mesmos folds.

- [ ] **4.1. Benchmark Contrastivo Geral**
  - **Ação:** Modificar scripts testando simultaneamente: GNN Original (do arquivo bugado salvo) versus GNN Estrutural (Pos-Full construído na Fase 3) versus Baseline Textual.
  - **Entregável:** Uma matriz de confusão para cada modelo separadamente e um arquivo de log numérico global exportado.
- [ ] **4.2. Execução da Prova Limpa Pelo K-Fold Pareado**
  - **Ação:** Acionar `09_teste_significancia.py`.
  - **Critério de não-rejeição de H0:** Se o teste acusar *p-valor* maior de 0.05 validando ausência de superioridade estatística entre Modelo Baseline Textual VS GNN Estruturada (Pos-Full), Fases práticas finalizam. Finalizar testes de código e focar na escrita do TCC.
  - **Entregável:** Tabela com média ± desvio-padrão de F1 por modelo, e p-valor por par de modelos exportado em LaTeX.
- [ ] **4.3. Prevalência Estrita em Inferência (Bluesky)**
  - **Ação:** Calcular AUPRC a partir dos scores de predição (não das labels finais) nas inferências isoladas em novo corpus.
  - **Gate Secundário:** Caso `AUPRC` permaneça inferior a 0.10, admite-se categoricamente que "o modelo não transfere".
  - **Entregável:** CSV com AUPRC por modelo em `results/bluesky_auprc.csv`.

---

## Fase 5 — Redação Científica (TCC)

### Fase 5A — Redação Teórica Isolada
- [ ] **5A.1. Mover GAT para Resultados (Incoerência 8):** Retirar declarações prospectivas de Graph Attention. 
  - **Entregável:** Reposicionar GAT de "trabalho futuro" para "experimento executado" nas seções de Metodologia e Resultados.
- [ ] **5A.2. Correção de Rótulos (Incoerência 9):** Identificar textualmente Fake com o valor nulo referencial.
  - **Entregável:** Modificação simples no TeX afirmando `0=Fake` padronizado.
- [ ] **5A.3. Calibração Metodológica (Incoerência 10):** Limpar referências imprecisas de Ensembles da metodologia.
  - **Entregável:** Nova subseção: "Framework de Calibração com Pesos Assimétricos".
- [ ] **5A.4. Degeneração GCN na Fundamentação (Incoerência 14):** Descrever matematicamente que sub-redes contendo `x_i = x_j ∀ i,j` estagnam o passo geométrico GCN.
  - **Entregável:** Parágrafo indicando que a regra de propagação degenera em transformação linear escalonada pela normalização de grau, incapaz de produzir representações discriminativas entre classes.
- [ ] **5A.5. Limitações Físicas do Dataset:** Reconhecer explicitamente a limitação dependente de cascatas estrela sem cascatas densas adicionais.
  - **Entregável:** Subseção abordando a ausência de timestamps precisos no dataset FakeNewsNet em contraste com APIs atuais.

### Fase 5B — Redação Pós-Experimentos Conclusivos
- [ ] **5B.1. Tese Final das GNN (Incoerência 5):** 
  - *Se Condição Positiva:* "A codificação aplicada detectou incrementos residuais diretos provando viabilidade modesta da topologia residual do FakeNewsNet."
  - *Se Condição Não Significativa (ns):* Formular um resultado negativo cientificamente válido na pesquisa refutando presunções baseadas no dataset original.
  - **Entregável:** Conclusão final/Abstracts reescritos.
- [ ] **5B.2. Constatação do confound (Incoerência 6):** Ajustar citações que afirmavam 0% de erro em grafos extensos.
  - **Entregável:** Subseção de discussão constatando que o efeito se explica pelo confound de tamanho do grafo (Erro 3), não pela topologia.
- [ ] **5B.3. A Prova Estatística FNN Base (Incoerência 7):** Documentar os resultados do cross-validation.
  - **Entregável:** Adicionar K-fold como protocolo de validação estatística do experimento exposto extensivamente com tabelas.
- [ ] **5B.4. Diferença Isolada do Bluesky (Incoerência 11):** Relatar o AUPRC aferido limitando a extrapolação do corpus primário frente às diferenças de distribuição de dados entre as plataformas.
  - **Entregável:** Refinamento metodológico usando AUPRC como métrica adequada sob prevalência desbalanceada no domínio alvo (Bluesky).
- [ ] **5B.5. Ajustes de narrativa quantitativa (Incoerência 13):** Concentrar F1 absolutos no rigor local de FNN; remover citações de gigabytes no material estritamente descritivo de avaliação externa.
  - **Entregável:** Edições no capítulo de metodologia eliminando superlativos do experimento exploratório no Bluesky.
- [ ] **5B.6. Reportar Baseline Textual na Tabela de Resultados (Incoerência 12):** Incluir Regressão Logística e Random Forest sobre BERT como linhas da tabela principal de resultados, ao lado de GCN/GAT/SAGE e GNN-Estrutural (Pos-Full). O baseline textual deve aparecer antes da conclusão como referência de comparação.
  - **Entregável:** Tabela em `resultados.tex` com todos os modelos (textuais + grafos) apresentando F1 médio ± desvio-padrão e p-valor do `ttest_rel` vs baseline textual.
