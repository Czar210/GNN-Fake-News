# Documentação Técnica: 08_inferencia_cruzada.py

## Metadados

- **Arquivo analisado:** `08_inferencia_cruzada.py`
- **Caminho:** `03_Mega_Research/08_inferencia_cruzada.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** GNN — experimento de transferência entre domínios (cross-dataset inference)
- **Modelos principais:** GCNClassifier · GATClassifier · SAGEClassifier (pesos pré-treinados carregados, sem re-treino)
- **Datasets utilizados:** Bluesky (test set) · FakeNewsNet local (`data/fakenewsnet_test.pt`) — apenas inferência, sem treino
- **Contribuição para a questão central:** Este script realiza o experimento de validade externa mais crítico do TCC: testa se os modelos GNN generalizam *entre* redes sociais distintas. O resultado documenta empiricamente dois fenômenos que fundamentam a conclusão negativa — (1) modelos Bluesky transferem sinal textual (BERT) mas não de propagação; (2) modelos UPFD colapsam por *prior probability shift* quando testados no Bluesky, revelando que o F1 como métrica principal é inadequado para comparações cross-domain.

---

## 1. Visão Geral do Script

`08_inferencia_cruzada.py` **não treina nenhum modelo**. Carrega os pesos salvos pelos scripts anteriores e avalia quatro cenários em combinação:

| Cenário | Pesos de treino | Dataset de teste | pos_label (Fake) |
|---------|----------------|-----------------|-----------------|
| BS → BS   (in-domain)  | Bluesky        | Bluesky         | 1 |
| UPFD → UPFD (in-domain)| FakeNewsNet    | FakeNewsNet     | 0 |
| BS → UPFD (cross)      | Bluesky        | FakeNewsNet     | 0 |
| UPFD → BS (cross)      | FakeNewsNet    | Bluesky         | 1 |

Os dois cenários *cross* são o coração do experimento — medem a degradação ao sair do domínio de treino, tanto na direção esperada (BS→UPFD: modelo sem grafos testa em grafos reais) quanto na direção inversa (UPFD→BS: modelo de grafos ricos enfrenta prevalência de classes radicalmente diferente).

Duas visualizações são geradas: uma grade 3×2 de matrizes de confusão (linhas = arquiteturas, colunas = direção) e um gráfico de barras com quatro cenários por arquitetura, com linha divisória visual entre in-domain e cross-domain.

---

## 2. Conceito Central — Inferência Cruzada (Cross-Dataset Inference)

**Descrição técnica:**
Inferência cruzada (também chamada *zero-shot cross-domain transfer* ou *out-of-distribution generalization*) consiste em avaliar um modelo em dados de distribuição diferente da usada no treino, **sem qualquer adaptação**. É o teste mais severo de generalização: o modelo deve ter aprendido representações suficientemente invariantes ao domínio para transferir.

Para modelos GNN de fake news, o desafio é duplo:
1. **Mudança de distribuição textual** (Twitter vs. Bluesky): vocabulário, estilo, comprimento dos posts diferem entre plataformas
2. **Mudança de estrutura do grafo** (árvore de retweet com dezenas de nós vs. grafo com 1 nó e 0 arestas): as camadas GNN veem topologias completamente distintas das vistas durante treino

**Embasamento acadêmico:**

> 📖 **Mosallanezhad, A.; Karami, M.; Shu, K.; Mancenido, M. V.; Liu, H. (2022)** — "Domain Adaptive Fake News Detection via Reinforcement Learning"
> *Proceedings of The ACM Web Conference 2022 (WWW'22)*
> DOI: `10.1145/3485447.3512258` | arXiv: `2202.08159`
> **Localização:** Seção 1 (Introduction) — formulação do problema de cross-domain; Seção 3 (REAL-FND) — framework de adaptação por RL
> **Relevância:** Paper de referência que quantifica a degradação cross-domain para modelos de fake news e motiva a necessidade de adaptação de domínio. A degradação observada neste script (F1 cai de 0.86 para 0.65 no sentido BS→UPFD) é o fenômeno que o REAL-FND busca mitigar.

> 📖 **Gong, S. et al. (2023)** — "Fake News Detection Through Graph-based Neural Networks: A Survey"
> arXiv: `2307.12639`
> **Localização:** Seção sobre *open problems* — cross-domain generalization como desafio em aberto para GNNs de fake news
> **Relevância:** Confirma que a generalização entre domínios é um dos principais desafios não resolvidos em GNNs para fake news — o script 08 documenta empiricamente este desafio.

---

## 3. Dataset Shift — Fundamento do Colapso UPFD→BS

### 3.1 Prior Probability Shift (Mudança de Prevalência)

**Descrição técnica:**
O colapso quase total no cenário UPFD→BS (F1 ≈ 0.007) não é causado primariamente por diferença de arquitetura ou de domínio textual, mas sim por **prior probability shift**: a proporção da classe Fake nos dois datasets é radicalmente diferente.

- **FakeNewsNet (treino):** ~50% Fake / ~50% Real
- **Bluesky (teste):** ~0.58% Fake / ~99.42% Real

Um classificador treinado com 50% de exemplos Fake aprende a predizer Fake com frequência proporcional a essa prevalência. Ao ser aplicado num dataset com 0.58% de Fake, gera uma quantidade de **falsos positivos** desproporcional ao número de verdadeiros positivos, colapsando a **Precision** (e consequentemente o F1).

**Formalização matemática:**

Seja $\pi_{\text{train}}$ a prevalência de Fake no treino e $\pi_{\text{test}}$ a prevalência no teste. O F1 de um classificador com Precision $p$ e Recall $r$ **calibrados para** $\pi_{\text{train}}$ quando aplicado a $\pi_{\text{test}}$ sofre degradação proporcional à razão $\pi_{\text{test}} / \pi_{\text{train}}$.

Formalmente (Prior Probability Shift — Moreno-Torres et al., 2012):

$$P(y) \neq P'(y), \quad P(x \mid y) = P'(x \mid y)$$

As distribuições condicionais da feature dado o label se mantêm iguais entre domínios, mas a distribuição marginal das classes muda. Isso quebra a calibração do modelo sem necessariamente mudar seu poder discriminativo.

A métrica **AUPRC** (Area Under Precision-Recall Curve) é mais robusta a prior probability shift do que F1, pois não depende de um threshold fixo de decisão.

**Embasamento acadêmico:**

> 📖 **Moreno-Torres, J. G.; Raeder, T.; Alaiz-Rodríguez, R.; Chawla, N. V.; Herrera, F. (2012)** — "A Unifying View on Dataset Shift in Classification"
> *Pattern Recognition*, v. 45, n. 1, pp. 521–530, 2012
> DOI: `10.1016/j.patcog.2011.06.019`
> **Localização:** Seção 3 (Prior Probability Shift), Definição 3 — $P(y) \neq P'(y), P(x|y) = P'(x|y)$; Seção 5 — impacto no desempenho de classificadores binários
> **Relevância:** Fundamento teórico rigoroso para a causa primária do colapso UPFD→BS. A nota diagnóstica no `relatorio.txt` (linhas 360–366) descreve exatamente este fenômeno, mas sem citar a literatura formal — este paper preenche a lacuna.

**No código:**
> Linha 363: comentário no `salvar_relatorio()` descreve exatamente o prior probability shift: "Um modelo calibrado para 50% de prevalência produz excesso de falsos positivos num dataset com 0.58% de prevalência, colapsando a Precision e portanto o F1".
> Linha 367: "NOTA: para testar transferência válida, os datasets precisam ter prevalências de classe comparáveis ou usar AUPRC em vez de F1."

---

### 3.2 Impacto nos Resultados

| Cenário | GCN F1 | GAT F1 | SAGE F1 | Interpretação |
|---------|--------|--------|---------|---------------|
| BS → BS (in-domain) | 0.4128 | 0.0000 | 0.2489 | Grafos Bluesky degenerados (0 arestas) → modelo textual puro |
| UPFD → UPFD (in-domain) | 0.8456 | 0.8591 | 0.8553 | Grafos ricos → GNNs funcionam; GAT se recupera |
| **BS → UPFD (cross)** | **0.6518** | **0.6549** | **0.6578** | Queda de ~0.19 vs in-domain UPFD |
| **UPFD → BS (cross)** | **0.0067** | **0.0065** | **0.0066** | Colapso quasi-total por prior probability shift |

**Análise do cenário BS→UPFD (queda de ~0.19):**

Os modelos treinados no Bluesky nunca viram grafos com mais de 1 nó durante o treino — as camadas GNN convergiram para realizar apenas a transformação linear `W·x_raiz` (sem agregação de vizinhos). Ao encontrar grafos UPFD com dezenas de nós, a agregação de vizinhos produz representações imprevisíveis. Contudo, a primeira camada ainda aplica a transformação BERT útil — daí o F1 residual de ~0.65, atribuível à transferência de **sinal textual BERT** entre domínios de língua inglesa (PolitiFact e Bluesky cobrem tópicos similares: política, saúde).

**Análise do cenário UPFD→BS (F1≈0.007):**

A queda de 0.84 → 0.007 é inteiramente dominada pelo prior probability shift (0.58% vs 50% de prevalência Fake), não pela incapacidade discriminativa da GNN. O experimento é invalidade como teste de transferência de propagação — para testar transferência válida, seria necessário: (a) usar datasets com prevalências comparáveis, ou (b) recalibrar o threshold de decisão, ou (c) usar AUPRC.

---

## 4. Carregamento de Pesos e Gerenciamento de Modelos

**Descrição técnica:**
`carregar_modelo()` (linhas 135–169) implementa um *factory pattern* para instanciar e carregar os pesos salvos. A seleção do arquivo de pesos depende da origem:

```python
if origem == "bluesky":
    nome_arquivo = {"GCN": "pesos_gcn.pth", "GAT": "pesos_gat.pth", "SAGE": "pesos_sage.pth"}[arquitetura]
else:  # upfd
    nome_arquivo = f"pesos_{arquitetura.lower()}_upfd_{dataset_name}.pth"
```

**Tratamento de falha:** Se o arquivo `.pth` não existir, o modelo é retornado com pesos aleatórios (inicialização Glorot) e um aviso é impresso — comportamento silenciosamente degradado. Não lança exceção. Isso é intencional para permitir experimentos parciais, mas pode mascarar resultados inválidos se o usuário não perceber o aviso.

**Nota sobre inconsistência de valores in-domain Bluesky:**
Os valores in-domain Bluesky neste script (GCN=0.4128, GAT=0.0000, SAGE=0.2489) diferem dos valores do script 06 (GCN=0.4295, GAT=0.0000, SAGE=0.4592). Isso ocorre porque o script 06 **não salva pesos** — os pesos `pesos_gcn.pth`, `pesos_gat.pth`, `pesos_sage.pth` foram gerados em uma run anterior separada (provavelmente script 04 ou run standalone do script 06). A variância entre runs com seed fixo pode ocorrer se o seed não for aplicado ao DataLoader shuffle.

---

## 5. Convenção de Labels e pos_label Dinâmico

**Descrição técnica:**
O experimento enfrenta o problema de **label mismatch** entre os dois datasets:
- Bluesky: `label 1 = Fake`, `label 0 = Real`
- UPFD/FakeNewsNet: `label 0 = Fake`, `label 1 = Real`

A função `avaliar()` recebe `pos_label_fake` como parâmetro explícito, garantindo que as métricas de Precision, Recall e F1 sempre meçam a **classe Fake correta**, independente da convenção numérica do dataset:

```python
"f1": round(f1_score(y_true, y_pred, pos_label=pos_label_fake, zero_division=0), 4)
```

Nos cenários cross, o `pos_label` do **dataset de teste** é usado (não do treino), pois os labels numéricos são determinados pelos dados de teste:
- BS→UPFD: modelo treinado com Bluesky (Fake=1), testado em FakeNewsNet (Fake=0) → `pos_label=0`
- UPFD→BS: modelo treinado com FakeNewsNet (Fake=0), testado em Bluesky (Fake=1) → `pos_label=1`

**Risco de confusão:** Se o modelo UPFD aprende a predizer `0` para Fake, e no Bluesky Fake=1, as predições estarão invertidas (todas as Fakes preditas como "Real"). O script **não** realiza remapeamento de labels — depende da convenção de treinamento estar consistente com a convenção de teste.

---

## 6. Visualizações

### 6.1 Grade de Matrizes de Confusão (3×2)

`plot_matrizes_cruzadas()` (linhas 177–230) gera uma grade com:
- **Linhas:** arquitetura (GCN, GAT, SAGE)
- **Colunas:** direção da transferência (BS→UPFD, UPFD→BS)

Cada célula mostra a matrix de confusão com F1 e Accuracy no título. Os tick labels variam por coluna para refletir a convenção de label do dataset de teste — detalhe que evita confusão na interpretação visual.

### 6.2 Comparativo de 4 Cenários (F1)

`plot_comparativo_f1()` (linhas 233–289) é a visualização mais informativa: quatro grupos de barras (in-domain BS, in-domain UPFD, cross BS→UPFD, cross UPFD→BS) com uma linha tracejada vertical separando in-domain de cross-domain. Permite ver instantaneamente:
- Quanto cada arquitetura perde ao sair do domínio de treino
- Se a direção da transferência importa (sim: BS→UPFD cai ~22%; UPFD→BS cai ~99%)
- Se as arquiteturas se comportam de forma diferente no cross-domain (não: as três têm F1≈0.65 e F1≈0.007 nos dois cenários cross)

---

## 7. Análise Empírica: Posicionamento na Questão Central

### 7.1 O que o experimento realmente mede

Este script mede três coisas distintas, frequentemente confundidas:

**1. Transferibilidade do sinal textual BERT (BS→UPFD F1≈0.65):**
O F1 residual de ~0.65 não é evidência de que a GNN aprendeu representações de propagação transferíveis. É evidência de que o **classificador BERT subjacente** (aprendido pelas camadas de projeção linear da GNN sobre os embeddings de 768 dimensões) transfere entre domínios de língua inglesa. A GNN, neste caso, é essencialmente um classificador de texto que passa pelos nós de forma serial.

**2. Falha de calibração cross-domain por prior probability shift (UPFD→BS F1≈0.007):**
Não é evidência de que GNNs são ruins para propagação. É evidência de que F1 com threshold fixo colapsa quando a prevalência de teste difere em ~86× do treino. Um teste mais justo usaria AUPRC ou recalibraria o threshold.

**3. Dependência de grafos não-degenerados (comparação implícita):**
O fato de que todos os modelos BS→UPFD têm F1≈0.65, independente da arquitetura (GCN, GAT, SAGE), confirma que o treinamento no Bluesky não distinguiu as três arquiteturas de forma significativa — todas aprenderam essencialmente o mesmo classificador textual.

### 7.2 Resposta à questão central

> **"GNNs são uma alternativa viável para detecção de fake news?"**

O script 08 adiciona evidência de que:
1. **GNNs treinadas sem grafos (Bluesky degenerado)** aprendem classificadores textuais que transferem moderadamente entre domínios em inglês (F1=0.65) — mas não têm vantagem sobre modelos NLP puros.
2. **GNNs treinadas com grafos reais (UPFD)** não generalizam para plataformas com prevalência muito diferente sem recalibração — limitação operacional significativa para implantação em produção.
3. **As três arquiteturas se comportam de forma praticamente idêntica no cross-domain** (F1 varia em < 0.003 entre GCN, GAT e SAGE nos cenários cross) — confirmando que a escolha da arquitetura GNN é secundária frente às propriedades do dataset.

### 7.3 Comparação com estado da arte

| Abordagem | In-domain F1 | Cross-domain F1 | Fonte |
|-----------|-------------|-----------------|-------|
| GCN (este TCC, BS→UPFD) | 0.4128 (BS) / 0.8456 (UPFD) | 0.6518 | Script 08 |
| REAL-FND (Mosallanezhad et al., 2022) | ~0.75–0.85 | melhoria significativa vs baseline | arXiv:2202.08159 |
| Domain-specific NLP | >0.85 (in-domain) | degradação moderada | Krzywda et al. (2024) |

A abordagem REAL-FND usa RL para adaptação explícita ao domínio alvo — o que o script 08 confirma ser necessário, mas não implementa.

---

## 8. Análise de Código

### 8.1 Erros identificados

**E1 — Fallback silencioso para pesos não encontrados (risco alto):**
```python
# ❌ Linhas 164–165 — retorna modelo com pesos aleatórios sem interromper execução
if not path.exists():
    print(f"  [WARN] {path.name} nao encontrado -- modelo nao carregado (zeros).")
    return model   # pesos aleatórios (Glorot initialization)
```
Se o arquivo `.pth` não existe, o script continua e reporta métricas de um modelo aleatório como se fossem resultados válidos. Correto seria lançar `FileNotFoundError` ou ao menos marcar os resultados como inválidos.

```python
# ✅ Alternativa mais segura:
if not path.exists():
    raise FileNotFoundError(
        f"Arquivo de pesos '{path}' não encontrado. "
        f"Execute o script de treino correspondente primeiro."
    )
```

**E2 — Label mismatch não verificado nos cenários cross:**
O script assume que os modelos UPFD predizem `0` para Fake e os modelos Bluesky predizem `1` para Fake. Mas se um modelo colapsa (como o GAT no Bluesky), ele pode predizer uma única classe que não é a esperada. Uma verificação de consistência (ex: verificar se o modelo prediz ambas as classes no conjunto de validação in-domain antes de inferência cross) seria mais robusta.

**E3 — `confusion_matrix` sem `labels=[0,1]` explícito:**
```python
# ⚠️ Linha 129 — sem labels explícitos
"confusion": confusion_matrix(y_true, y_pred).tolist()
# Se uma classe não aparece em y_pred (ex: modelo colapsa), a matriz fica 1×1
# Corrigido (e documentado) nos scripts anteriores:
"confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
```

**E4 — Nomes de arquivos de pesos hard-coded:**
```python
nome_arquivo = {"GCN": "pesos_gcn.pth", ...}[arquitetura]
```
Estes arquivos não são gerados pelo script 06 (que não salva pesos). Dependem de execuções separadas cujo histórico não é rastreado automaticamente — fragilidade de reprodutibilidade.

### 8.2 Boas práticas observadas

- **pos_label dinâmico**: passar `pos_label_fake` como parâmetro explícito é correto e evita bugs silenciosos na comparação entre datasets com convenções invertidas.
- **Diagnóstico de prevalência no relatório** (linhas 360–367): a análise de causa do colapso UPFD→BS embutida no `salvar_relatorio()` é metodologicamente honesta e educativa.
- **Visualização de 4 cenários com linha divisória in-domain/cross** (`ax.axvline`): design claro para apresentação na monografia.
- **Inspeção de prevalência no carregamento** (linhas 73–76, 88–93): reporta % de Fake em cada split, permitindo diagnóstico rápido do desbalanceamento.

---

## 9. Referências Bibliográficas

1. MOSALLANEZHAD, A.; KARAMI, M.; SHU, K.; MANCENIDO, M. V.; LIU, H. **Domain Adaptive Fake News Detection via Reinforcement Learning**. *Proceedings of The ACM Web Conference 2022 (WWW'22)*, 2022. DOI: `10.1145/3485447.3512258` / arXiv: `2202.08159`

2. MORENO-TORRES, J. G.; RAEDER, T.; ALAIZ-RODRÍGUEZ, R.; CHAWLA, N. V.; HERRERA, F. **A Unifying View on Dataset Shift in Classification**. *Pattern Recognition*, v. 45, n. 1, pp. 521–530, 2012. DOI: `10.1016/j.patcog.2011.06.019`

3. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *SIGIR'21*, 2021. DOI: `10.1145/3404835.3462990` / arXiv: `2104.12259`

4. SHU, K. et al. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`

5. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *NeurIPS 2017*. arXiv: `1706.02216`

6. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. *ICLR 2017*. arXiv: `1609.02907`

7. VELIČKOVIĆ, P. et al. **Graph Attention Networks**. *ICLR 2018*. arXiv: `1710.10903`

8. GONG, S. et al. **Fake News Detection Through Graph-based Neural Networks: A Survey**. arXiv: `2307.12639`, 2023.

9. KRZYWDA, M. et al. **Comparative Analysis of GNNs and Transformers for Robust Fake News Detection**. *Electronics 2024*, 13(23), 4784. DOI: `10.3390/electronics13234784`

---

## 10. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Cross-dataset inference | Avaliação de modelo em dados de distribuição diferente da usada no treino, sem adaptação | Mosallanezhad et al. (2022) |
| Prior probability shift | Dataset shift onde P(y) muda entre treino e teste enquanto P(x\|y) se mantém | Moreno-Torres et al. (2012), Seção 3 |
| Covariate shift | Dataset shift onde P(x) muda mas P(y\|x) se mantém — distinto do prior shift | Moreno-Torres et al. (2012), Seção 2 |
| AUPRC | Area Under Precision-Recall Curve — métrica robusta a prior probability shift, preferível ao F1 em cenários desbalanceados cross-domain | Davis & Goadrich (2006) |
| In-domain | Avaliação onde treino e teste vêm da mesma distribuição | — |
| Out-of-domain / Cross-domain | Avaliação onde treino e teste vêm de distribuições distintas | — |
| Factory pattern | Padrão de projeto onde uma função cria e retorna instâncias de classe com base em parâmetros — usado em `carregar_modelo()` | Design Patterns, Gamma et al. (1994) |
| Label mismatch | Inconsistência na convenção numérica de labels entre dois datasets (ex: Fake=0 vs Fake=1) | — |
| Threshold de decisão | Valor de probabilidade acima do qual o classificador prediz a classe positiva; fixed at 0.5 (argmax) neste script | — |
