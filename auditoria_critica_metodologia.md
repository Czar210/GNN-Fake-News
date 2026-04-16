# Auditoria Crítica de Metodologia — GNN Fake News TCC

**Autor da auditoria:** Revisão interna (assistida por Claude Sonnet 4.6)  
**Data:** 16 de abril de 2026  
**Escopo:** Todos os arquivos não commitados no branch `master` do repositório  
**Severidade:** 🔴 Crítico · 🟡 Significativo · 🟠 Menor · 🟢 Correto

---

## Sumário Executivo

Esta auditoria identificou **um problema metodológico central que invalida a premissa comparativa do trabalho**, acompanhado de cinco problemas significativos e três menores. O problema central é: os grafos construídos para o FakeNewsNet têm **todos os nós com features idênticos** (o embedding BERT do artigo é clonado para cada nó filho), tornando as GNNs matematicamente equivalentes a um classificador linear sobre o texto do artigo. Uma Regressão Logística simples atinge F1=0.8514 no mesmo conjunto de teste onde as GNNs atingem F1=0.855–0.869 — diferença de 0.4% a 1.8%, dentro do ruído estatístico.

**Isso não encerra o TCC. Reorienta-o.** A conclusão correta é mais honesta e igualmente publicável: *"a estrutura de propagação disponível neste dataset não adiciona poder discriminativo além do conteúdo textual."*

---

## Índice

1. [Problema Central — Features Nodais Idênticos](#1-problema-central)
2. [Consequência Direta — GNN ≈ Classificador Linear de BERT](#2-consequência-direta)
3. [Confund — Número de Tweets como Proxy Espúrio](#3-confund)
4. [Metodologia dos Testes de Significância](#4-testes-de-significância)
5. [Early Stopping com Critério Errado](#5-early-stopping)
6. [Conclusão do Relatório Factualmente Incorreta](#6-conclusão-incorreta)
7. [Inferência Cruzada — Causa Mal Atribuída](#7-inferência-cruzada)
8. [GCNClassifier Duplicado com Comportamento Diferente](#8-gcn-duplicado)
9. [Itens Corretos](#9-itens-corretos)
10. [Como a Percepção do Trabalho Mudou](#10-percepção-revisada)
11. [Caminhos à Frente](#11-caminhos-à-frente)

---

## 1. Problema Central

**Severidade: 🔴 Crítico**

### O que está errado

Em `00_construir_grafos_fakenewsnet.py`, linha 116:

```python
x = torch.stack([x_raiz] * num_nos)
```

O embedding BERT do **artigo** (nó raiz) é copiado literalmente para **todos os nós filhos** (usuários que tweetearam). Verificação empírica:

```
Max diff entre quaisquer dois nós de um grafo: 0.000000
São todos iguais? True
Topologia: todos src=0 (estrela pura)
```

Todo grafo do FakeNewsNet construído por este pipeline é uma **estrela uniforme**: um nó raiz conectado a N filhos, onde todos os N+1 nós carregam exatamente o mesmo vetor de 768 dimensões.

### Por que isso é fatal para a comparação de arquiteturas

**GCNConv** com features idênticos realiza:

```
h_v = W · x_bert · normalization_factor(grau_v)
```

A aggregação de vizinhos retorna o mesmo vetor, escalado pela normalização espectral. O `global_mean_pool` sobre todos os nós produz um vetor proporcional a `W · x_bert`. A GCN é um mapeamento linear do embedding do artigo para a classificação.

**GATConv** com features idênticos colapsa ainda mais:

```
score(h_i, h_j) = LeakyReLU(a^T [W·h_i || W·h_j])
```

Se `h_i = h_j` para todos os nós, todos os scores são idênticos, o softmax produz atenção uniforme — exatamente o mesmo resultado que GCN. O mecanismo de atenção é completamente ineficaz.

**SAGEConv** com features idênticos:

```
h_v = W_l · x_bert + W_r · mean(x_bert) = (W_l + W_r) · x_bert
```

Reduz a uma única transformação linear com mais parâmetros.

**Conclusão matemática:** Para grafos com features nodais homogêneos em topologia de estrela, GCN = GAT = SAGE = Classificador Linear (com diferentes matrizes de peso, mas mesma capacidade expressiva efetiva).

---

## 2. Consequência Direta — GNN ≈ Classificador Linear de BERT

**Severidade: 🔴 Crítico**

### Experimento de controle realizado

Treinamos uma Regressão Logística sobre os embeddings BERT brutos (sem grafo, sem GNN) no mesmo split de treino/teste:

| Modelo | F1 (Fake, pos_label=0) | Accuracy |
|---|---|---|
| **Regressão Logística — BERT puro** | **0.8514** | **0.8553** |
| GraphSAGE (média 10 runs) | 0.8557 ± 0.0184 | 0.8592 |
| GAT (média 10 runs) | 0.8649 ± 0.0162 | 0.8665 |
| GCN (média 10 runs) | 0.8691 ± 0.0126 | 0.8704 |

O ganho das GNNs sobre a regressão logística é de **+0.004 a +0.018**, dentro do desvio padrão de cada modelo. Os testes t pareados (10 runs) entre os próprios modelos GNN já mostraram p > 0.15 em todos os pares — ou seja, os GNNs não se distinguem entre si, e tampouco se distinguem de um classificador sem grafo.

### O que isso implica

A hipótese central do TCC — que "a estrutura de propagação carrega informação discriminativa além do conteúdo textual" — **não pode ser afirmada positivamente com estes dados**. Não porque seja falsa, mas porque os grafos construídos não permitem testá-la.

O sinal que os modelos estão usando é o conteúdo textual do título do artigo (via BERT), não a rede de difusão social. Isso vale tanto para Bluesky (grafos vazios por bug de pipeline) quanto para FakeNewsNet (grafos uniformes por design de features).

---

## 3. Confund — Número de Tweets como Proxy Espúrio

**Severidade: 🟡 Significativo**

### O que foi descoberto

Artigos fake e reais no FakeNewsNet têm distribuições significativamente diferentes de número de tweets:

```
Nós em grafos FAKE: mean=93.6, median=112, std=58.6
Nós em grafos REAL: mean=83.3, median=78,  std=63.6

t-test entre grupos: t=2.30, p=0.022 → diferença significativa (p < 0.05)
```

Artigos fake têm ~12% mais tweets que artigos reais neste dataset. Isso significa que a GNN tem um segundo sinal disponível além do BERT: **o tamanho do grafo**, que é função do número de tweets.

### Por que isso é um problema

Mesmo que os features nodais sejam idênticos, o número de nós influencia sutilmente a representação após `global_mean_pool` via normalização espectral. A GNN pode estar usando "popularidade do artigo" como feature proxy — e essa correlação com fake/real é específica do PolitiFact. Em outros domínios, artigos verdadeiros populares também teriam muitos tweets, invalidando o modelo.

Verificação: um classificador treinado apenas no número de nós atinge F1=0.564, confirmando que este sinal existe no dataset. O modelo não deveria ter acesso a ele se o objetivo é aprender padrões de conteúdo e propagação.

---

## 4. Testes de Significância

**Severidade: 🟡 Significativo**

### Justificativa do t-test pareado está parcialmente errada

O script `09_teste_significancia.py` usa `ttest_rel` (pareado) com a justificativa de que cada run usa a mesma partição treino/teste.

**O que está correto:** o conjunto de teste é fixo (152 grafos do `fakenewsnet_test.pt`). Runs com o mesmo índice k usam o mesmo seed.

**O que está incorreto:** a "paridade" testada é apenas **variância de inicialização de pesos**, não **variância de generalização**. O conjunto de teste nunca muda entre runs. Isso não testa se a diferença se mantém para diferentes sub-populações do dataset.

### Metodologia correta

Para uma comparação arquitetural academicamente sólida, o método adequado é **k-fold cross-validation estratificada**:

1. Dividir os 754 grafos em 5 ou 10 folds estratificados por classe
2. Para cada fold: treinar no restante, avaliar no fold
3. Comparar as k métricas por par de modelos com `ttest_rel`

Isso testa se a superioridade de uma arquitetura é consistente para **diferentes sub-populações dos dados**, não apenas para diferentes inicializações de pesos com os mesmos dados.

Com apenas 152 amostras de teste e desvio padrão de ±2% em F1, um resultado único (single split) tem erro de estimativa considerável.

### O resultado "ns" ainda é válido?

Sim. O resultado "não significativo" é correto e conservador — os modelos realmente não se distinguem. O problema é que a metodologia não é forte o suficiente para detectar diferenças pequenas se elas existissem. Com n_runs=10 e std≈0.015, a potência do teste para detectar diferença de 0.02 em F1 é baixa.

---

## 5. Early Stopping com Critério Errado

**Severidade: 🟠 Menor**

Em `07_upfd_benchmark_triplo.py:199` e `09_teste_significancia.py:148`, o checkpoint é selecionado por:

```python
if val_acc > best_val:  # critério: accuracy
```

Mas a métrica de avaliação final é **F1**. Para o FakeNewsNet balanceado (~50/50), accuracy ≈ F1, portanto o impacto é mínimo neste caso. Porém, a inconsistência é metodologicamente incorreta: o critério de seleção de modelo deveria ser a mesma métrica usada na avaliação final. Em datasets desbalanceados, isso poderia selecionar um checkpoint que maximiza accuracy (predizendo sempre a classe majoritária) com F1 próximo de zero.

**Correção simples:**

```python
# Calcular F1 na validação ao invés de accuracy
val_f1 = f1_score(y_val_true, y_val_pred, pos_label=0, zero_division=0)
if val_f1 > best_val:
    ...
```

---

## 6. Conclusão do Relatório Factualmente Incorreta

**Severidade: 🟡 Significativo**

O arquivo `Execution/results/upfd_benchmark_fakenewsnet/relatorio.txt` contém:

> "GAT supera GCN e GraphSAGE em grafos com estrutura real. O mecanismo de atenção diferencia efetivamente nós influentes na árvore de propagação (super-spreaders)."

Esta afirmação é **inválida** por três razões:

1. **Estatisticamente:** A diferença GAT vs GCN em F1 é de +0.014, com p=0.628 nos testes de significância. Não há evidência estatística de superioridade.

2. **Mecanisticamente:** Como demonstrado na Seção 1, com features nodais idênticos, a atenção GAT colapsa para uniforme. Não há "super-spreaders" distinguíveis — todos os nós têm o mesmo embedding.

3. **Empiricamente:** Uma Regressão Logística sem grafo atinge desempenho estatisticamente equivalente, eliminando a possibilidade de que a estrutura do grafo explique o desempenho.

O relatório gerado automaticamente pelo `script 07` assumiu que o vencedor do benchmark estava usando o grafo de forma efetiva. Essa lógica de auto-interpretação (`if vencedor == "GAT": linhas.append("O mecanismo de atenção...")`) é perigosa porque gera narrativa causal sem base.

---

## 7. Inferência Cruzada — Causa Mal Atribuída

**Severidade: 🟡 Significativo**

O relatório `inferencia_cruzada/relatorio.txt` atribui o colapso FNN→BS (F1≈0.007) a:

> "modelo UPFD aprendeu propagação em árvores ricas. No Bluesky (1 nó, 0 arestas), reduz a classificador linear BERT. A queda reflete diferença de domínio (Twitter vs Bluesky)"

A análise está incompleta. A causa **mais provável e mais imediata** é o desbalanceamento de classes:

- FakeNewsNet (treino): ~50% fake
- Bluesky (teste): 0.58% fake (194/33,382)

Um modelo calibrado para 50% de prevalência de fake, aplicado a um dataset com 0.58% de prevalência, terá **precision colapsando** por excesso de falsos positivos. O F1≈0.007 é consistente com Precision≈0.003 e Recall≈0.26 — o modelo encontra ~50 dos 194 fakes, mas produz ~15.000 falsos positivos.

A diferença de domínio (Twitter vs Bluesky) pode ser um fator adicional, mas não o fator primário. A inferência cruzada seria inválida mesmo que os dados fossem da mesma plataforma, dado o mismatch de prevalência de classes.

**Implicação metodológica:** Para testar transferência de domínio de forma válida, o conjunto de avaliação deve ter prevalência de classes similar ao treino, ou a métrica deve ser corrigida para prevalência (ex: AUPRC em vez de F1).

---

## 8. GCNClassifier Duplicado com Comportamento Diferente

**Severidade: 🟠 Menor**

`GCNClassifier` está reimplementado inline em:

- `06_comparar_gcn_gat_sage.py` — seed hardcoded `12345`
- `07_upfd_benchmark_triplo.py` — seed hardcoded `12345`
- `08_inferencia_cruzada.py` — seed hardcoded `12345`
- `09_teste_significancia.py` — seed como parâmetro (comportamento diferente)

O GCN no script de benchmark único (07) sempre inicializa com seed=12345. O GCN no script de significância (09) usa seeds variáveis por run. São instâncias arquiteturalmente idênticas mas com inicializações diferentes — correto para o propósito de cada script, mas não documentado explicitamente. Qualquer comparação direta de pesos entre os dois scripts seria inválida.

A solução correta é centralizar `GCNClassifier` num arquivo `gcn_model.py` por simetria com `gat_model.py` e `sage_model.py`.

---

## 9. Itens Corretos

🟢 **Split em nível de grafo** (não artigo): leakage correto evitado.

🟢 **`pos_label` explícito** em todas as chamadas de métricas: a convenção de labels invertida entre Bluesky e FakeNewsNet é tratada corretamente.

🟢 **Best-checkpoint restore** no early stopping: o modelo avaliado é sempre o melhor visto na validação, não o do último epoch.

🟢 **`zero_division=0`** em precision/recall/F1: evita crash silencioso em runs onde o modelo não prediz nenhuma fake.

🟢 **Verificação de existência de arquivos** antes de carregar: sys.exit com mensagem informativa ao invés de exception sem contexto.

🟢 **Separação de dados de teste**: o conjunto de teste nunca é visto durante o desenvolvimento do modelo ou seleção de hiperparâmetros.

🟢 **Seed no construtor dos modelos**: a refatoração para aceitar `seed` como parâmetro (em vez de hardcoded `12345`) foi necessária e correta para os testes de significância.

---

## 10. Como a Percepção do Trabalho Mudou

### Antes desta auditoria

A narrativa era: *"GAT falhou no Bluesky porque grafos de estrela são muito simples para atenção. Trouxemos o FakeNewsNet com grafos 'ricos e reais'. GAT venceu. A estrutura de propagação importa."*

### Depois desta auditoria

Essa narrativa está incorreta em dois níveis:

**Nível 1 — O Bluesky:** Grafos de 1 nó ocorreram por bug de pipeline (ID mismatch entre posts e reposts CSVs). Grafos de estrela rasos por si só não invalidam GAT — a atenção funciona em estrelas se os nós tiverem features distintos.

**Nível 2 — O FakeNewsNet:** Grafos de 88 nós com features idênticos são igualmente incapazes de revelar diferenças arquiteturais. A riqueza estrutural que construímos (número de nós, topologia) não é acompanhada por riqueza informacional (features heterogêneos por nó).

**O que realmente aconteceu em ambos os datasets:** as GNNs aprenderam a classificar o **texto do artigo** via BERT, não a **rede de propagação**. A hipótese central do trabalho ficou sem teste em ambos os experimentos.

### O que isso significa para o TCC

Esta descoberta **não invalida o trabalho** — reorienta sua contribuição:

1. O mapeamento de GNN sobre propagação de fake news é uma área ativa.
2. A descoberta de que features nodais homogêneos tornam GNNs equivalentes a classificadores de texto é um resultado experimental concreto.
3. A evidência quantitativa (baseline BERT F1=0.851 vs GNN F1=0.855–0.869, p>0.15) é um dado legítimo.
4. O diagnóstico do bug de pipeline Bluesky e a análise do confund de popularidade (p=0.022) são contribuições metodológicas.

A conclusão revisada — *"grafos de estrela com features nodais homogêneos não provêm vantagem para arquiteturas GNN sobre classificação de texto pura"* — é mais honesta, mais defensável numa banca e mais útil para trabalhos futuros do que declarar um vencedor entre GNN sem evidência estatística.

---

## 11. Caminhos à Frente

### Opção A — Corrigir a construção dos grafos (impacto máximo, esforço moderado)

Substituir o clone do embedding do artigo por **features distintos por nó**:

- **Nó raiz (artigo):** embedding BERT do título (como está)
- **Nós filhos (usuários):** embedding BERT do texto do tweet individual

O FakeNewsNet não fornece os textos dos tweets, apenas os IDs. Seria necessário coletar via Twitter API (agora X API — paga) ou usar outra fonte.

**Alternativa viável sem nova coleta:** features estruturais por nó:
- Profundidade na árvore (0 para raiz, 1 para filhos — neste caso todos = 1 por ser estrela)
- Índice temporal normalizado do tweet (ordem de chegada)
- One-hot ou embedding da posição no grafo

Isso daria à GNN algo estrutural para aprender, mesmo sem texto de usuário.

### Opção B — Reframing da narrativa (menor esforço, academicamente válido)

Manter os experimentos como estão e reformular a contribuição:

> "Investigamos se a estrutura de propagação social adiciona poder discriminativo para detecção de fake news além do conteúdo textual. Nos dois datasets avaliados — Bluesky (grafos degenerados por limitação de coleta) e FakeNewsNet (grafos de estrela com features homogêneos) — as arquiteturas GCN, GAT e GraphSAGE não superaram estatisticamente um classificador linear sobre embeddings BERT. O resultado alerta para a necessidade de grafos com features heterogêneos por nó para que as capacidades expressivas das GNNs sejam efetivamente exploradas."

Adicionalmente: incluir o baseline de Regressão Logística nos relatórios como ponto de referência explícito.

### Opção C — Usar o UPFD real com dataset alternativo

O UPFD original (links do GDrive expirados para PolitiFact/GossipCop) inclui features de perfil de usuário por nó (`bert`, `spacy`, `content`, `profile`). Com esses dados, cada nó teria features distintos e os benchmarks seriam válidos. O dataset pode ser encontrado no Zenodo ou solicitado diretamente aos autores.

---

## Evidências de Suporte Coletadas

```
# Features idênticos confirmados
Max diff entre nós de qualquer grafo: 0.0

# Baseline BERT sem grafo
Regressão Logística F1=0.8514 vs GCN F1=0.8691 (Δ=+0.018, dentro do std)

# Número de tweets correlacionado com classe
t-test fake vs real nós: t=2.30, p=0.022 → confund estrutural

# Testes de significância entre GNNs
GCN vs GAT: p=0.628 (ns)
GCN vs SAGE: p=0.170 (ns)
GAT vs SAGE: p=0.158 (ns)

# Topologia confirmada
Todos grafos: topologia estrela pura (único src=nó 0)
Cosine similarity entre embeddings de grafos de 10 e 100 nós: 1.000000
```

---

*Relatório gerado em 16/04/2026. Arquivos auditados: 7 scripts Python novos + 1 modificado + 4 relatórios de resultados.*
