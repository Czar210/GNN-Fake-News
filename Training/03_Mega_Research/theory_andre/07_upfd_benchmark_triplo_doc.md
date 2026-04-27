# Documentação Técnica: 07_upfd_benchmark_triplo.py

## Metadados

- **Arquivo analisado:** `07_upfd_benchmark_triplo.py`
- **Caminho:** `03_Mega_Research/07_upfd_benchmark_triplo.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** GNN — benchmark triplo controlado em grafos de propagação reais (Twitter/FakeNewsNet)
- **Modelos principais:** GCNClassifier · GATClassifier · SAGEClassifier (mesmos do script 06)
- **Datasets utilizados:** UPFD PolitiFact, UPFD GossipCop (via PyG) · FakeNewsNet local (`data/fakenewsnet_*.pt`)
- **Contribuição para a questão central:** Este script é o teste crítico de validade externa do TCC: replica o benchmark do script 06 em grafos de propagação reais do Twitter (dezenas a centenas de nós, múltiplos hops) em vez dos grafos estrela degenerados do Bluesky. A comparação Bluesky×UPFD revela em que medida a profundidade e heterogeneidade do grafo afetam o desempenho relativo das arquiteturas GNN.

---

## 1. Visão Geral do Script

`07_upfd_benchmark_triplo.py` estende o protocolo do script 06 de duas formas: (1) substitui o dataset Bluesky pelo **UPFD** — grafos de propagação reais de Twitter coletados sobre notícias do PolitiFact e GossipCop — e (2) adiciona suporte a múltiplos modos de entrada via argumento `--dataset` (politifact, gossipcop via UPFD PyG; fakenewsnet via arquivos `.pt` locais) e `--feature` (bert, spacy, content, profile).

O loop de treino é idêntico ao script 06, com uma adição: **early stopping explícito** via `patience_count` (parada se 10 épocas consecutivas sem melhoria no F1-macro de validação). Os pesos treinados são salvos em `Execution/weights/pesos_{modelo}_upfd_{dataset}.pth` para reutilização com `--usar-pesos-salvos`.

Uma quarta visualização é adicionada: `curvas_treino.png` — curvas de loss de treino e F1-macro de validação ao longo das épocas para os três modelos, permitindo diagnóstico de overfitting e convergência.

A avaliação final usa `pos_label=0` para Precision, Recall e F1 — refletindo a convenção do UPFD onde **Fake=0** e **Real=1** (oposto do Bluesky), garantindo que as métricas meçam a detecção da classe de interesse (notícia falsa).

---

## 2. Dataset UPFD e FakeNewsNet

### 2.1 UPFD — User Preference-aware Fake News Detection

**Descrição técnica:**
O UPFD é um benchmark de grafos de propagação do Twitter para detecção de fake news, publicado por Dou et al. (SIGIR 2021). Cada grafo representa a **árvore de retweet** de uma notícia:

- **Nó raiz:** a notícia original (tweet-fonte)
- **Nós folha:** usuários do Twitter que retweetaram a notícia, direta ou indiretamente
- **Arestas:** usuário A → usuário B se A retweetou a notícia a partir do tweet de B; ou usuário → raiz se retweetou diretamente

Esta topologia é **hierárquica e profunda** — ao contrário do grafo estrela do Bluesky (1 hop), as árvores de retweet do UPFD têm múltiplos hops, gerando vizinhanças com heterogeneidade semântica que beneficia mecanismos de agregação mais expressivos.

**Estatísticas dos subconjuntos:**

| Subconjunto | Notícias | Nós totais | Nós médios/grafo |
|-------------|----------|-----------|-----------------|
| PolitiFact  | 314      | ~41,000   | ~130            |
| GossipCop   | 5,464    | ~314,000  | ~57             |

**Convenção de labels (invertida em relação ao Bluesky):**
- `label = 0` → **Fake** (notícia falsa)
- `label = 1` → **Real** (notícia verdadeira)

**Variantes de features por nó:**

| Feature | Dimensão | Fonte |
|---------|----------|-------|
| `bert`    | 768 | Tweets históricos do usuário codificados via BERT (bert-as-service) |
| `spacy`   | 300 | Tweets históricos do usuário codificados via spaCy word2vec |
| `content` | 310 | `spacy` (300) + `profile` (10) — concatenados |
| `profile` | 10  | 10 atributos do perfil Twitter do usuário |

O script usa `bert` por padrão — mesma dimensão (768) dos embeddings BERT do Bluesky, facilitando a comparação.

**Embasamento acadêmico:**

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'21)*
> DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`
> **Localização:** Seção 3.1 (Dataset Construction), Tabela 1 (estatísticas do dataset), Seção 3.2 (Node Features), Apêndice (split train/val/test)
> **Relevância:** Paper original que define o UPFD, a topologia de retweet-tree, a convenção label 0=Fake, e as quatro variantes de features nodais. Também contém tabela de resultados GCN/GAT/SAGE como baseline para comparação com este TCC.

**No código:**
> Linhas 110–117: `UPFD(root=..., name=dataset_name, feature=feature, split=...)` — download automático via PyG.
> Linha 219: `pos_label=0` em todas as métricas — alinhado com a convenção Fake=0 do paper.

---

### 2.2 FakeNewsNet — Conjunto Local

**Descrição técnica:**
FakeNewsNet é o repositório de dados no qual o UPFD é baseado. Contém notícias verificadas por fact-checkers do PolitiFact (político) e GossipCop (entretenimento), com contexto social coletado do Twitter.

**Embasamento acadêmico:**

> 📖 **Shu, K.; Mahudeswaran, D.; Wang, S.; Lee, D.; Liu, H. (2020)** — "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media"
> *Big Data*, v. 8, n. 3, pp. 171–188, 2020
> DOI: `10.1089/big.2020.0062` | arXiv: `1809.01286`
> **Localização:** Seção 3 (Dataset Description), Tabela 1 (estatísticas PolitiFact e GossipCop), Seção 2.2 (Social Context: retweet propagation graphs)
> **Relevância:** Paper original que define a coleção FakeNewsNet; fornece a base dos grafos de propagação do Twitter usados pelo UPFD. As estatísticas de grafos (número de nós, profundidade das árvores) fundamentam a comparação com os grafos Bluesky.

**No código:**
> Linhas 79–98: `carregar_fakenewsnet()` — alternativa local ao UPFD PyG quando os links do Google Drive expiraram.
> Argumento `--data-suffix`: carrega variantes `posfull/posmin/posgrau/bugado_original` de `data/fakenewsnet_<suffix>/`.

---

## 3. Arquitetura e Componentes Principais

Os três modelos (GCNClassifier, GATClassifier, SAGEClassifier) são idênticos aos documentados em `04_comparar_gcn_gat_doc.md` e `06_comparar_gcn_gat_sage_doc.md`. A diferença relevante aqui é a **dimensão de entrada variável**: enquanto o Bluesky sempre usa BERT-768, o script 07 recebe `num_features` dinamicamente do dataset (`train_ds.num_features`), permitindo usar qualquer variante de feature do UPFD.

> Linha 480–482: `GCNClassifier(num_features, 2)` — aceita qualquer dimensão de entrada.

**Contagem de parâmetros com num_features=768 (bert):**
- GCN: 57,858 | GAT: 219,330 | SAGE: 115,394 (ligeiramente maiores que script 06 por diferença no bias)

---

## 4. Loop de Treinamento — Early Stopping Explícito

### 4.1 Diferença em relação ao script 06

O script 07 adiciona **early stopping explícito** com `patience_count` (linhas 142–194):

```python
if val_f1 > best_val:
    best_val       = val_f1
    best_state     = {k: v.clone() ...}
    patience_count = 0
else:
    patience_count += 1

if patience_count >= 10:
    print("[Early stopping] 10 epocas sem melhoria.")
    break
```

Isso complementa o `ReduceLROnPlateau` (que reduz o LR mas não para o treino): após 10 épocas sem melhoria no F1-macro de validação, o treino termina antecipadamente. No script 06, o treino sempre roda até `epochs` completos.

**Motivação:** O FakeNewsNet/UPFD é um dataset menor (314 grafos no PolitiFact) — overfitting ocorre mais cedo, e parar precocemente é mais importante.

**Embasamento acadêmico:**

> 📖 **Prechelt, L. (1998)** — "Early Stopping — But When?"
> *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science, v. 1524, pp. 55–69
> Springer, Berlin, Heidelberg
> **Localização:** Seção 2 (Stop criteria), definição de GL (Generalization Loss) e critério de patience
> **Relevância:** Fundamento clássico para o uso de early stopping como regularização implícita — justifica a escolha de patience=10 épocas como compromisso entre convergência e overfitting.

---

### 4.2 Critério de Seleção de Modelo

O critério de seleção é **F1-macro de validação** (não accuracy), alinhado com a métrica principal de avaliação:
> Linha 174: `val_f1 = f1_score(..., average="macro", zero_division=0)`

Isso é crítico para datasets desbalanceados: usar accuracy como critério poderia selecionar modelos que maximizam a classe majoritária sem detectar fakes.

---

## 5. Avaliação — Convenção pos_label=0

**Descrição técnica:**
No UPFD, a classe de interesse (Fake) tem label 0. A função `avaliar_completo()` especifica `pos_label=0` em todas as métricas binárias:

```python
"precision": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
"recall":    round(recall_score(y_true, y_pred,    pos_label=0, zero_division=0), 4),
"f1":        round(f1_score(y_true, y_pred,        pos_label=0, zero_division=0), 4),
```

Isso garante que o F1 reportado é da **classe Fake** (label 0), não da classe Real — interpretação correta para o problema de detecção de desinformação, onde falsos negativos (fake classificada como real) têm custo maior.

**Nota de implementação:** Diferente do script 06 que usa `f1_score(..., zero_division=0)` sem pos_label (F1 binário default com pos_label=1), o script 07 corrige para pos_label=0. Essa diferença significa que os F1 dos dois scripts **não são diretamente comparáveis** sem essa correção.

---

## 6. Visualizações — Curvas de Treinamento

`plot_curvas()` (linhas 325–347) gera a figura adicional `curvas_treino.png` com dois painéis:
- **Loss de treino** ao longo das épocas para GCN, GAT e SAGE
- **F1-macro de validação** ao longo das épocas

Esta visualização permite identificar:
- Ponto de convergência de cada modelo
- Overfitting (loss de treino cai, F1 de validação estagna ou piora)
- Instabilidade de treinamento (oscilações do GAT em grafos rasos)
- Velocidade de convergência relativa entre arquiteturas

---

## 7. Salvamento de Pesos

O script 07 é o primeiro a persistir os pesos treinados:
> Linha 533: `torch.save(m.state_dict(), pesos[nome])`
> Nomes: `pesos_gcn_upfd_{dataset}.pth`, `pesos_gat_upfd_{dataset}.pth`, `pesos_sage_upfd_{dataset}.pth`

Com `--usar-pesos-salvos`, os pesos são carregados em vez de re-treinar — útil para reproduzir os resultados sem custo computacional. Se um arquivo não existe, o modelo é treinado do zero e o arquivo criado automaticamente.

---

## 8. Métricas de Avaliação — Resultados Obtidos

### 8.1 FakeNewsNet — variante `bugado_original` (features homogêneas)

| Modelo | Accuracy | Precision | Recall | F1 | Params | Tempo/ép. |
|--------|----------|-----------|--------|----|--------|-----------|
| GCN    | 0.8618   | 0.8354    | 0.8919 | **0.8627** | 57,666 | 0.38s |
| GAT    | 0.8618   | 0.8630    | 0.8514 | 0.8571 | 218,562 | 1.18s |
| SAGE   | 0.8553   | 0.8514    | 0.8514 | 0.8514 | 115,010 | 0.90s |
| Δ SAGE−GCN | −0.0065 | +0.0160 | −0.0405 | **−0.0113** | — | — |

**Ranking F1: GCN > GAT > SAGE | Amplitude: 0.0113**

### 8.2 FakeNewsNet — variante `posfull` (features posicionais)

| Modelo | Accuracy | Precision | Recall | F1 | Params | Tempo/ép. |
|--------|----------|-----------|--------|----|--------|-----------|
| GCN    | 0.8618   | 0.8442    | 0.8784 | 0.8609 | 57,858 | 0.16s |
| GAT    | 0.8882   | 0.8800    | 0.8919 | **0.8859** | 219,330 | 0.66s |
| SAGE   | 0.8553   | 0.8611    | 0.8378 | 0.8493 | 115,394 | 0.39s |
| Δ SAGE−GCN | −0.0065 | +0.0169 | −0.0406 | **−0.0116** | — | — |

**Ranking F1: GAT > GCN > SAGE | Amplitude: 0.0366**

### 8.3 GossipCop — benchmark UPFD oficial (content features)

| Modelo | F1-macro | F1-fake | Accuracy | Δ vs LogReg |
|--------|----------|---------|----------|------------|
| Baseline LogReg(x[0]) | 0.9540 | — | — | — |
| GCN    | 0.9411±0.0059 | 0.9403±0.0062 | 0.9411±0.0059 | **−0.0129 (PIORA)** |
| GAT    | 0.9390±0.0050 | 0.9385±0.0048 | 0.9390±0.0050 | **−0.0150 (PIORA)** |
| SAGE   | 0.9448±0.0041 | 0.9441±0.0039 | 0.9449±0.0041 | −0.0092 (EQUIVALE) |

**Resultado mais crítico do TCC: TODOS os GNNs ficam abaixo do baseline LogReg no GossipCop.**

---

## 9. Análise Empírica: Posicionamento na Questão Central

### 9.1 Resultado mais importante: GAT se recupera no UPFD

**Bluesky (grafos estrela, 1–50 nós):** GAT F1=0.0000 — colapso completo de atenção.
**UPFD PolitiFact (árvores de retweet, ~130 nós, múltiplos hops):** GAT F1=0.8571–0.8859 — desempenho competitivo.

Isso confirma empiricamente a hipótese de Brody et al. (2022): o GAT falha especificamente em grafos **homogêneos e rasos**, onde a atenção estática não consegue discriminar vizinhos. Em grafos mais profundos e heterogêneos (árvores de retweet Twitter), os nós têm features distintas (tweets de usuários diferentes) e a atenção encontra sinal discriminativo.

**Implicação para o TCC:** O colapso do GAT no script 06 foi um artefato do dataset, não uma limitação fundamental da arquitetura. O script 07 fornece a evidência contrafactual necessária.

### 9.2 Diferenças inter-arquiteturas são pequenas e potencialmente não-significativas

A amplitude máxima entre os três modelos no FakeNewsNet é 0.0366 (variante posfull). O relatório do próprio script adverte: *"Diferenças pequenas (< ~0.02) podem ser variância de inicialização."* Isso é consistente com a literatura:

> 📖 **Krzywda et al. (2024)** reportam diferenças igualmente pequenas entre GCN/GAT/SAGE em FakeNewsNet, concluindo que a escolha da arquitetura GNN tem impacto menor do que a qualidade das features nodais.

O script 09 (teste de significância) existe para resolver essa ambiguidade — ver `teste_significancia/relatorio.txt`.

### 9.3 GNNs abaixo do baseline no GossipCop — resultado negativo confirma-se

O resultado mais importante para o TCC: no GossipCop, **uma regressão logística sobre o primeiro nó** (LogReg em x[0]) supera todos os GNNs. Isso significa que:
1. A **estrutura do grafo** (topologia de propagação) não adiciona sinal discriminativo além do conteúdo do nó raiz
2. Os GNNs estão, na prática, "diluindo" o sinal do nó raiz ao agregar vizinhos menos informativos
3. O fenômeno generaliza do PolitiFact (script 06) para o GossipCop — não é específico de um dataset

**Resposta parcial à questão central:** Em grafos de propagação de fake news com features BERT em todos os nós, a topologia da rede social **não adiciona sinal discriminativo** consistente além do texto. GNNs equivalem ou perdem para modelos puramente textuais — evidência forte para o resultado negativo do TCC.

### 9.4 Comparação Bluesky × UPFD

| Modelo | Bluesky F1 (script 06) | UPFD FakeNewsNet F1 | Δ |
|--------|----------------------|--------------------|----|
| GCN    | 0.4295               | 0.8609–0.8627      | +0.43 |
| GAT    | 0.0000               | 0.8571–0.8859      | +0.86 |
| SAGE   | 0.4592               | 0.8493–0.8514      | +0.39 |

A melhoria dramática de Bluesky para UPFD se deve principalmente à qualidade dos **grafos** (UPFD tem topologia real, Bluesky tinha grafos degenerados com 1 nó e 0 arestas — conforme indicado em `analise_benchmarks_gnn.md`), não à superioridade das arquiteturas GNN. O UPFD fornece o ambiente experimental correto para comparar as arquiteturas.

---

## 10. Pontos Fortes e Limitações

### 10.1 Pontos fortes

- **Fallback robusto:** `carregar_upfd()` captura a exceção de links Google Drive expirados e sugere alternativa local (linha 113–117) — resiliência operacional importante num ambiente de pesquisa.
- **Early stopping explícito** evita overfitting em datasets pequenos (PolitiFact: 314 grafos).
- **Flexibilidade de feature:** suporte a 4 variantes permite testar hipóteses sobre o tipo de feature mais informativo.
- **Persistência de pesos** facilita reprodutibilidade e análise incremental.
- **Aviso de significância** embutido no relatório (`relatorio.txt`, linhas 408–413) — auto-documentação do limite de interpretação de uma única execução.

### 10.2 Limitações identificadas

**L1 — Mesma ausência de class weights:**
```python
criterion = torch.nn.CrossEntropyLoss()  # sem ponderação — igual ao script 06
```
No PolitiFact (balanceado 50/50 Fake/Real) isso não é crítico, mas no GossipCop, o desbalanceamento pode distorcer o treinamento.

**L2 — avg_loss incorreto (propagado do script 06):**
```python
avg_loss = total_loss / sum(d.num_graphs for d in train_loader)  # itera loader 2x
```
Itera o train_loader uma segunda vez por época apenas para somar `num_graphs`. Deve acumular durante o loop principal.

**L3 — UPFD links expirados:** O dataset UPFD via PyG depende de links Google Drive que expiraram. O fallback para `--dataset fakenewsnet` (arquivos locais) é a solução, mas requer que o script 00 tenha sido executado previamente.

**L4 — Curvas de validação mal rotuladas:**
> Linha 339: `axes[1].set(title="Acuracia de Validacao", ...)` — O eixo plota F1-macro (`hist["val"]`), não accuracy. Título incorreto; deveria ser "F1-macro de Validação".

**L5 — `patience_count` não é resetado pelo scheduler:** `ReduceLROnPlateau` e `patience_count` são mecanismos independentes com patiences diferentes (5 e 10, respectivamente). Quando o scheduler reduz o LR após 5 épocas estagnadas, o treino pode melhorar com LR menor — mas `patience_count` continua acumulando. O early stopping pode parar o treino antes do benefício do LR reduzido ser realizado.

---

## 11. Análise de Código

### 11.1 Erros identificados

**E1 — Título incorreto no gráfico de curvas (cosmético, mas relevante para monografia):**
```python
# ❌ Linha 339 — plota F1-macro mas rotula como accuracy
axes[1].set(title="Acuracia de Validacao", ylabel="Accuracy")

# ✅ Correção:
axes[1].set(title="F1-macro de Validação", ylabel="F1-macro")
```

**E2 — avg_loss com dupla iteração (propagado do script 06):**
```python
# ❌ Linha 175 — itera train_loader inteiro de novo
avg_loss = total_loss / sum(d.num_graphs for d in train_loader)

# ✅ Correção — acumular n_total durante o loop:
# n_total += data.num_graphs  (dentro do for data in train_loader)
avg_loss = total_loss / n_total
```

**E3 — Conflito entre ReduceLROnPlateau e early stopping:**
O `ReduceLROnPlateau` usa `patience=5` e o early stopping usa `patience_count >= 10`. Quando a LR é reduzida após 5 épocas e o modelo melhora na 6ª época, o `patience_count` já está em 5 — o early stopping pode disparar apenas 5 épocas depois. Considerando que o modelo acabou de melhorar, isso é prematuro.

Solução: resetar `patience_count = 0` também quando o scheduler reduz a LR (detectável via `optimizer.param_groups[0]['lr']`).

### 11.2 Boas práticas observadas

- **Inspeção automática do dataset** (`_inspecionar_dataset()`): reporta tamanho médio dos grafos, número de arestas e proporção de fakes — diagnóstico rápido e documentado de cada split.
- **Convenção de labels documentada no relatório** (`Nota: label 0 = Fake`): evita confusão na interpretação dos resultados.
- **Aviso de limitação embutido**: o script avisa que uma única execução não é suficiente para conclusões definitivas e remete ao script 09.
- **Fallback para links expirados**: captura `Exception` e orienta o usuário para alternativa local.

---

## 12. Referências Bibliográficas

1. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'21)*, 2021. DOI: `10.1145/3404835.3462990` / arXiv: `2104.12259`

2. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062` / arXiv: `1809.01286`

3. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *NeurIPS 2017*. arXiv: `1706.02216`

4. KIPF, T. N.; WELLING, M. **Semi-Supervised Classification with Graph Convolutional Networks**. *ICLR 2017*. arXiv: `1609.02907`

5. VELIČKOVIĆ, P. et al. **Graph Attention Networks**. *ICLR 2018*. arXiv: `1710.10903`

6. BRODY, S.; ALON, U.; YAHAV, E. **How Attentive are Graph Attention Networks?** *ICLR 2022*. arXiv: `2105.14491`

7. PRECHELT, L. **Early Stopping — But When?** *Neural Networks: Tricks of the Trade*, LNCS v. 1524, pp. 55–69. Springer, 1998.

8. KRZYWDA, M. et al. **Comparative Analysis of Graph Neural Networks and Transformers for Robust Fake News Detection**. *Electronics 2024*, 13(23), 4784. DOI: `10.3390/electronics13234784`

9. KINGMA, D. P.; BA, J. **Adam: A Method for Stochastic Optimization**. *ICLR 2015*. arXiv: `1412.6980`

10. GILMER, J. et al. **Neural Message Passing for Quantum Chemistry**. *ICML 2017*. arXiv: `1704.01212`

---

## 13. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| UPFD | User Preference-aware Fake News Detection — benchmark PyG com grafos de retweet do Twitter para PolitiFact e GossipCop | Dou et al. (2021) |
| FakeNewsNet | Repositório com notícias verificadas do PolitiFact e GossipCop com contexto social do Twitter | Shu et al. (2020) |
| Árvore de retweet | Grafo hierárquico onde raiz=notícia, folhas=usuários que retweetaram; aresta A→B se A retweetou de B | Dou et al. (2021), Seção 3.1 |
| Early stopping | Interrupção do treino quando a métrica de validação não melhora por N épocas consecutivas (patience) | Prechelt (1998) |
| pos_label | Parâmetro das métricas sklearn que define qual label é a "classe positiva" — aqui pos_label=0 (Fake) | scikit-learn docs |
| Resultado negativo | Cenário científico válido onde a hipótese experimental não se confirma — GNNs não superam baseline no GossipCop | — |
| bert-as-service | Serviço que extrai embeddings BERT de sentenças; usado para gerar os features de 768 dimensões do UPFD | Xiao (2018), GitHub |
