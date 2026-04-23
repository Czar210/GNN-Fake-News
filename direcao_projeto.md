# Direção do Projeto — GNN Fake News TCC

**Data:** Abril 2026
**Propósito:** Documento explicativo do projeto para contextualizar a nova direção depois da auditoria metodológica. Explica em linguagem clara o que o código faz, onde estava errado, o que está sendo corrigido e por quê.

---

## 1. O que é o projeto

O TCC tenta responder uma pergunta:

> **A topologia de uma cascata de compartilhamento (quem retweetou, em que ordem) ajuda a detectar fake news além do que o texto do artigo já diz?**

A intuição é: notícia falsa se espalha de jeito diferente da verdadeira — mais explosiva, mais viral em pouco tempo, com perfil de engajamento atípico. Se isso for verdade, uma rede neural que olha para o **grafo de propagação** (não só para o texto) deveria acertar mais.

Para testar isso, o projeto usa **Graph Neural Networks (GNNs)** aplicadas a grafos construídos a partir do dataset FakeNewsNet (PolitiFact).

---

## 2. O que é uma GNN (em 3 parágrafos)

Um grafo é um conjunto de **nós** ligados por **arestas**. No nosso caso, cada grafo representa uma notícia:
- O nó 0 é a notícia em si (raiz).
- Os nós 1, 2, ..., N são os tweets que comentaram/compartilharam a notícia (filhos).
- Cada aresta liga a raiz a um filho (formato "estrela").

Cada nó carrega um **vetor de features** — uma lista de números que representa o conteúdo do nó (por exemplo, um embedding BERT do texto).

Uma GNN é uma rede neural que, a cada camada, **mistura os features de cada nó com os features dos seus vizinhos**. Essa mistura é chamada de *agregação*. Depois de algumas camadas, cada nó "conhece" informação da sua vizinhança. No final, há um **pooling global** que junta todos os nós em um vetor único por grafo, que passa por um classificador para decidir: *fake* ou *real*.

O que muda entre as arquiteturas é **como a agregação funciona**:
- **GCN** (Graph Convolutional Network): soma normalizada pelo grau.
- **GAT** (Graph Attention Network): média ponderada com pesos aprendidos ("atenção").
- **GraphSAGE**: média simples, com uma matriz separada para o próprio nó e outra para os vizinhos.

---

## 3. Os três datasets e o papel de cada um

O projeto trabalha com **três fontes de dados diferentes**. Entender qual serve para quê evita confusão entre os scripts.

### 3.1. UPFD — a referência bibliográfica (dataset "ideal")

- **Fonte:** disponível direto no PyTorch Geometric via `torch_geometric.datasets.UPFD` (e no Zenodo).
- **O que tem:** para cada nó do grafo, quatro variantes de features — `profile` (metadados do usuário), `bert` (embedding do tweet), `spacy` (embedding alternativo), `content` (texto bruto).
- **O que é especial:** cada nó tem feature **real e diferente**, porque os tweets individuais estão disponíveis. É a forma correta de montar o grafo — e é exatamente o que *não* temos nos outros dois.
- **Scripts que tocam:** [baseline_upfd.py](Training/03_Mega_Research/baseline_upfd.py), [treinar_upfd_robusto.py](Training/03_Mega_Research/treinar_upfd_robusto.py), [restaurar_upfd.py](Training/03_Mega_Research/restaurar_upfd.py).
- **Status no projeto:** os links originais do GDrive expiraram, o código de restauração foi uma tentativa de recuperar, mas o experimento principal migrou para FakeNewsNet reconstruído. **UPFD permanece como referência conceitual** — inspira o formato do grafo ("estrela" com raiz = artigo + nós = usuários), mas não é mais usado em experimento ativo.

### 3.2. FakeNewsNet PolitiFact — o experimento estatístico principal

- **Fonte:** dois CSVs baixados de [github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) — `politifact_fake.csv` e `politifact_real.csv`.
- **O que tem:** para cada notícia — título, URL, e a coluna `tweet_ids` (lista de IDs dos tweets que propagaram aquela notícia). **Só os IDs, não o texto dos tweets.**
- **O que é limitado:** como não temos o texto individual de cada tweet, o código atual replica o BERT do título para todos os nós filhos. **Essa é a origem do Problema 1.**
- **Tamanho:** ~754 grafos após filtro `min_tweets ≥ 2`, balanceados ~50% fake / 50% real.
- **Scripts que tocam:**
  - **Construção:** [00_construir_grafos_fakenewsnet.py](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py) — produz `fakenewsnet_train.pt`, `fakenewsnet_val.pt`, `fakenewsnet_test.pt`.
  - **Treino individual:** [03_treinar_gat.py](Training/03_Mega_Research/03_treinar_gat.py), [05_treinar_sage.py](Training/03_Mega_Research/05_treinar_sage.py), [treinar_mega_dataset.py](Training/03_Mega_Research/treinar_mega_dataset.py).
  - **Comparação par a par:** [04_comparar_gcn_gat.py](Training/03_Mega_Research/04_comparar_gcn_gat.py), [06_comparar_gcn_gat_sage.py](Training/03_Mega_Research/06_comparar_gcn_gat_sage.py).
  - **Benchmark triplo (final):** [07_upfd_benchmark_triplo.py](Training/03_Mega_Research/07_upfd_benchmark_triplo.py).
  - **Significância estatística:** [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py) — o único com p-valores reportados.
- **Status no projeto:** é o **dataset que gera os números estatísticos** reportáveis no TCC. Toda a Fase 2, Fase 3 e Fase 4 (exceto 4.3) do [plano_correcao.md](plano_correcao.md) atuam aqui.

### 3.3. Bluesky — o experimento exploratório / inferência cruzada

- **Fonte:** dataset `Zaras210/bluesky-fake-news-dataset` do HuggingFace — ~21 GB, 168k posts, 152M interações.
- **O que tem:** posts com texto completo, reposts, replies, metadados de usuário. Rotulagem de fake é **heurística** (palavras-chave como "conspiracy", "fake", "hoax" + labels da API do Bluesky como "misleading").
- **Prevalência:** apenas **0.58% fake** (194 em 33.382 posts depois da filtragem). Extremamente desbalanceado.
- **Limitação descoberta:** ao construir grafos, a maioria dos posts Bluesky não tinha reposts associados no dataset baixado — os grafos **degeneraram para 1 nó, 0 arestas**. Isso está documentado em [08_inferencia_cruzada.py:13-22](Training/03_Mega_Research/08_inferencia_cruzada.py#L13).
- **Scripts que tocam:**
  - **Download:** [01_baixar_dataset_hf.py](Training/03_Mega_Research/01_baixar_dataset_hf.py) — seleciona `graphs.tar.gz` e `feed_posts/**` do HuggingFace e gera `posts_coletados.csv` + `reposts_coletados.csv` em [Training/01_BlueSky_Pipe/data/raw/](Training/01_BlueSky_Pipe/data/raw/).
  - **Construção:** [02_construir_grafos_bluesky.py](Training/03_Mega_Research/02_construir_grafos_bluesky.py) — lê os CSVs e monta grafos hierárquicos (ou estrela plana em fallback). Produz `grafos_bluesky_train.pt`, `grafos_bluesky_val.pt`, `grafos_bluesky_test.pt`.
  - **Inferência cruzada:** [08_inferencia_cruzada.py](Training/03_Mega_Research/08_inferencia_cruzada.py) — testa `FNN → BS` e `BS → FNN`.
- **Status no projeto:** não é o experimento estatístico principal. Vira **estudo de transferência de domínio** — mede quanto o modelo treinado em FakeNewsNet consegue classificar posts do Bluesky. A queda de F1 observada aqui foi originalmente atribuída à "diferença de plataforma" (Incoerência 11), mas a causa mais provável é o desbalanceamento de prevalência (50% → 0.58%). A Fase 4.3 do plano troca F1 por **AUPRC**, métrica adequada para prevalência desbalanceada.

### 3.4. Matriz de qual script toca qual dataset

| Script | UPFD | FakeNewsNet | Bluesky |
|---|:---:|:---:|:---:|
| `baseline_upfd.py` | ✓ | | |
| `treinar_upfd_robusto.py` | ✓ | | |
| `restaurar_upfd.py` | ✓ | | |
| `00_construir_grafos_fakenewsnet.py` | | ✓ | |
| `03_treinar_gat.py` | | ✓ | |
| `04_comparar_gcn_gat.py` | | ✓ | |
| `05_treinar_sage.py` | | ✓ | |
| `06_comparar_gcn_gat_sage.py` | | ✓ | |
| `07_upfd_benchmark_triplo.py` | | ✓ | |
| `09_teste_significancia.py` | | ✓ | |
| `treinar_mega_dataset.py` | | ✓ | |
| `matrizes_ablation.py` | | ✓ | |
| `01_baixar_dataset_hf.py` | | | ✓ |
| `02_construir_grafos_bluesky.py` | | | ✓ |
| `08_inferencia_cruzada.py` | | ✓ | ✓ |

### 3.5. Linha do tempo dos datasets no projeto

- **Passado (já feito):** tentativa de usar UPFD oficial → links quebraram → reconstruímos grafos estilo UPFD usando FakeNewsNet CSV + Bluesky. Benchmark GCN/GAT/SAGE rodou sobre FakeNewsNet. Inferência cruzada FNN↔BS rodou e mostrou queda de performance.
- **Presente (auditoria):** descoberta dos 4 erros e 13 incoerências, reformulação da hipótese central (ver seção 6).
- **Futuro (plano em execução):**
  - **UPFD:** permanece apenas como referência na escrita do TCC.
  - **FakeNewsNet:** ganha features posicionais por nó (Fase 3), baseline textual rigoroso (Fase 2A), k-fold pareado (Fase 2B), benchmark final (Fase 4.1–4.2).
  - **Bluesky:** reavaliado com AUPRC (Fase 4.3), reinterpretado como estudo de prevalência — não de transferência de domínio.

---

## 4. Como o código constrói os grafos hoje

Script: [Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py)

Passo a passo:
1. Baixa dois CSVs do FakeNewsNet: `politifact_fake.csv` e `politifact_real.csv`. Cada linha é uma notícia com título, URL e a coluna `tweet_ids` (lista de IDs de tweets que propagaram aquela notícia).
2. Gera o **embedding BERT do título** da notícia usando `paraphrase-multilingual-mpnet-base-v2`. O resultado é um vetor de 768 números por notícia.
3. Para cada notícia, constrói um grafo em estrela:
   - Nó 0 (raiz) = o artigo. Features = BERT(título).
   - Nós 1..N = os tweets. **Features = a MESMA cópia do BERT(título).** ← aqui mora o problema (Erro 1).
   - Arestas = raiz → cada filho.
4. Filtra grafos com menos de 2 tweets.
5. Divide em treino (60%), validação (20%) e teste (20%).
6. Salva como arquivos `.pt` em `Training/03_Mega_Research/data/`.

O trecho exato da duplicação do embedding está em [00_construir_grafos_fakenewsnet.py:116](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py#L116):

```python
x = torch.stack([x_raiz] * num_nos)
```

Essa linha duplica o vetor da raiz para todos os `num_nos` nós. Isso foi feito porque o dataset não tem o texto individual de cada tweet — só os IDs. Mas essa escolha quebra a matemática da GNN, como veremos.

---

## 5. Como os três modelos funcionam (simplificado)

Arquitetura comum a todos: **3 camadas de convolução** + **global mean pool** + **classificador linear de 64 → 2**.

### GCN — [gcn_model.py](Training/03_Mega_Research/gcn_model.py)
- Cada camada aplica: `H' = D⁻¹ᐟ² Ã D⁻¹ᐟ² H W`
- Traduzindo: cada nó recebe uma **soma dos vizinhos normalizada pelo grau** e multiplica por uma matriz de pesos `W`.
- Sem atenção, sem heads, sem distinção entre "self" e "vizinho".

### GAT — [gat_model.py](Training/03_Mega_Research/gat_model.py)
- Cada camada aprende **pesos de atenção** entre cada par (nó, vizinho).
- A atenção é um softmax sobre uma função dos features do par.
- Usa **4 cabeças de atenção** nas duas primeiras camadas (rodam em paralelo, com pesos independentes), e 1 cabeça na terceira.
- Ativação ELU e dropout interno.

### GraphSAGE — [sage_model.py](Training/03_Mega_Research/sage_model.py)
- Cada camada aplica: `H' = W_l · H + W_r · mean(H_vizinhos)`
- Duas matrizes separadas: uma para o próprio nó (`W_l`), outra para a média dos vizinhos (`W_r`).
- Indutivo: pode generalizar para nós nunca vistos no treino.

Os três foram treinados no mesmo dataset, com o mesmo split, e comparados em [07_upfd_benchmark_triplo.py](Training/03_Mega_Research/07_upfd_benchmark_triplo.py) e [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py).

---

## 6. O que deu errado — problemas críticos em linguagem simples

### Problema 1 — Todo nó tem o mesmo vetor 🔴

Porque o código copia BERT(título) para todos os nós, **os 3 modelos acabam processando grafos onde todos os nós são idênticos**. A GNN não tem informação diferente para usar entre os vizinhos.

**O que isso faz matematicamente:**
- GCN com features iguais degenera em `W · x_bert` escalonado pelo grau. Virou regressão linear.
- GAT com features iguais produz softmax uniforme. A atenção não aprende nada — vira GCN.
- SAGE: `W_l · x + W_r · mean(x) = (W_l + W_r) · x` quando todos iguais. Também colapsa.

**Tradução:** a GNN não está lendo topologia. Está lendo texto (o BERT do título), com um escalonamento trivial imposto pelo grau. Virou, na prática, um classificador textual.

---

### Problema 2 — A GNN não supera o texto puro 🔴

Consequência direta do Problema 1. Rodando Regressão Logística só em cima do BERT do título:

| Modelo | F1 (Fake) |
|---|---|
| Regressão Logística (BERT puro) | 0.851 |
| GraphSAGE | 0.856 |
| GAT | 0.865 |
| GCN | 0.869 |

Os ganhos são minúsculos e **estatisticamente indistinguíveis** entre GCN/GAT/SAGE (p > 0.15 em todos os pares). A hipótese central do TCC ("topologia carrega sinal além do texto") **não está sendo testada** — porque os grafos construídos não permitem testá-la.

---

### Problema 3 — Confound de tamanho 🟡

Notícias fake no FakeNewsNet têm, em média, 12% mais tweets que notícias reais (t-test: p = 0.022). Um classificador que olha **só para o número de nós do grafo** atinge F1 = 0.564. Ou seja, parte do que a GNN aprende pode não ser "topologia" — é só "notícia fake vira viral mais rápido neste dataset específico".

Isso é sinal real, mas é **específico do PolitiFact**. Não se transfere para outros domínios.

---

### Problema 4 — Teste de significância fraco 🟡

O script [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py) roda 10 treinos, cada um com uma seed diferente, **no mesmo conjunto de teste fixo**. Depois aplica `ttest_rel`.

**O que isso mede:** variância de inicialização de pesos.
**O que deveria medir:** variância de generalização (se o modelo é estável em diferentes sub-populações dos dados).

**Correção correta:** k-fold cross-validation — dividir o dataset em 10 partes, rodar cada modelo em 10 combinações diferentes de treino/teste, e comparar.

---

### Incoerências entre o TCC e o código

Estas são menos críticas mas precisam ser consertadas no texto:

| # | Problema |
|---|---|
| 5 | A conclusão diz "geometria é superior ao texto" — mas os dados não suportam isso. |
| 6 | O TCC atribui "0% de erro em grafos grandes" à topologia — mas provavelmente é só o confound de tamanho (Problema 3). |
| 7 | O experimento estatístico do FakeNewsNet existe no código mas não aparece no TCC. |
| 8 | GAT é listado como "trabalho futuro" — mas já foi implementado e comparado. |
| 9 | Labels inconsistentes: texto diz "0 = Real" em um lugar, código usa "0 = Fake". |
| 10 | Os 4 modelos do "ensemble" são todos GCN com pesos de loss diferentes. Isso é calibração, não ensemble. |
| 11 | Queda de performance em Bluesky foi atribuída à "mudança de domínio" — provavelmente é desbalanceamento de classes (FakeNewsNet tem 50% fake, Bluesky tem 0.58%). |
| 12 | Objetivo "comparar texto puro vs texto + topologia" — mas o baseline de texto puro nunca foi reportado. |
| 13 | Narrativa grandiosa sobre "21GB do Bluesky" não bate com o experimento estatístico real (FakeNewsNet, CSV pequeno). |
| 14 | Fundamentação descreve a regra do GCN mas não menciona a degeneração com features iguais (Problema 1). |
| 15 | Relatório auto-gerado afirma "GAT vence" sem ter significância estatística. |
| 16 | Early stopping usa accuracy, mas a métrica final é F1. |
| 17 | `GCNClassifier` está reimplementado inline em 4 scripts em vez de importado de `gcn_model.py`. |

---

## 7. A nova direção — o que estamos fazendo e por quê

A estratégia foi reformulada em uma frase:

> **Provar ou refutar, com rigor estatístico pareado, que a topologia do FakeNewsNet adiciona sinal discriminativo além do texto (BERT).**

Ou seja: **o TCC agora aceita que o resultado pode ser negativo** — e um resultado negativo, bem documentado, também é uma contribuição científica válida.

### O plano tem 5 fases (ver [plano_correcao.md](plano_correcao.md)):

**Fase 1 — Limpeza.** Consolidar imports duplicados, trocar early stopping para F1, remover narrativas causais automáticas. Arrumação de casa.

**Fase 2 — Estabelecer o teto textual.**
- **2A:** rodar Regressão Logística e Random Forest só em cima do BERT do título. Esse é o "teto" que a GNN precisa superar para ter valor.
- **2B:** instanciar K-fold estratificado uma única vez e compartilhar os mesmos folds entre todos os modelos. Agora o `ttest_rel` mede diferenças reais entre modelos, não ruído de inicialização.
- **2C:** treinar um Random Forest que usa só `num_nodes` como feature. Se ele atingir F1 > 0.65, confirmamos que o tamanho do grafo é um confound sério e disparamos subamostragem pareada.

**Fase 3 — Consertar o Problema 1 com features posicionais.**
- Arquivar os `.pt` originais (bugados) como grupo de controle — vamos precisar dele depois.
- Reescrever o construtor de grafos para dar a cada nó **features posicionais diferentes**:
  - Raiz: `[BERT(título) || is_root=1, grau_normalizado, pos=0]`
  - Filho i: `[BERT(título) || is_root=0, grau=0, pos=i/N]`
- Com isso, os 3 modelos passam a ter informação heterogênea para trabalhar. A atenção do GAT passa a fazer sentido. A soma normalizada do GCN deixa de ser trivial.
- **Diagnóstico intra-encoding:** rodar 3 variantes (só posição, só grau, tudo junto) para isolar de qual sinal vem o ganho (se houver).

**Fase 4 — Benchmark final.**
- Comparar GNN original bugada vs GNN com features posicionais vs baseline textual, todos nos mesmos folds.
- Rodar o `ttest_rel` pareado.
- Medir AUPRC (não F1) no Bluesky, porque AUPRC é robusta a desbalanceamento.

**Fase 5 — Escrever o TCC.**
- **5A (pode escrever agora):** correções textuais que independem dos resultados — renomear "ensemble" para "calibração", mover GAT para seção executada, padronizar `0=Fake`, adicionar parágrafo sobre a degeneração da GCN na fundamentação, declarar limitações do dataset.
- **5B (depende dos resultados da Fase 4):** conclusão condicional.
  - Se GNN com features posicionais ganhar: reportar o ganho.
  - Se empatar: reportar como **resultado negativo** — a topologia estrela pura do FakeNewsNet não carrega sinal além do texto. Isso é honesto e tem valor científico.

---

## 8. O que está correto e não precisa mexer

Nem tudo estava errado. Pontos que permanecem:

🟢 **Split em nível de grafo** — não há vazamento entre treino e teste.
🟢 **`pos_label` explícito** nas métricas.
🟢 **Best-checkpoint restore** no early stopping.
🟢 **`zero_division=0`** em precision/recall/F1.
🟢 **Seed parametrizado** em `gat_model`, `sage_model`, e no `09_teste_significancia.py`.
🟢 **Verificação de existência de arquivos** antes de carregar.

---

## 9. Resumo — o que mudou na direção

| Antes | Depois |
|---|---|
| Tese: "topologia é superior ao texto" | Tese: "provar ou refutar que topologia adiciona sinal além do texto" |
| GNN vs GNN (GCN/GAT/SAGE comparadas entre si) | GNN vs baseline textual (a verdadeira pergunta) |
| Features nodais copiadas da raiz | Features posicionais por nó (raiz marcada, filhos com `pos=i/N`) |
| Teste de significância com seeds diferentes no mesmo test set | K-fold estratificado com folds pareados entre modelos |
| Narrativa grandiosa sobre Bluesky (21GB) | FakeNewsNet como experimento estatístico principal; Bluesky como estudo exploratório |
| Conclusão afirmativa | Conclusão condicional (positiva ou negativa), ambas honestas |

---

## 10. Por que essa direção deve dar certo (ou um "dar certo" honesto)

A métrica de sucesso não é mais "GNN supera BERT". É:

1. Ter um **baseline textual rigoroso** reportado com intervalo de confiança.
2. Ter **features nodais heterogêneas** que permitam a GNN aprender algo topológico de verdade.
3. Ter um **teste estatístico pareado** que meça generalização, não ruído de inicialização.
4. Ter uma **conclusão honesta** que reflete o que os números dizem — seja ganho ou empate.

Com esses quatro itens, o TCC é defensável mesmo que a GNN empate com o texto. A contribuição deixa de ser "construímos uma GNN campeã" e passa a ser "medimos com rigor se topologia estrela pura do FakeNewsNet carrega sinal além do texto — e a resposta é X".

Esse é um TCC maduro.
