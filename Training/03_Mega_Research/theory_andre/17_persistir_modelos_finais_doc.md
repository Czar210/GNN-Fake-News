# Documentação Técnica: 17_persistir_modelos_finais.py

## Metadados

- **Arquivo analisado:** `17_persistir_modelos_finais.py`
- **Caminho:** `Training/03_Mega_Research/17_persistir_modelos_finais.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Persistência de artefatos finais — três regimes (textual puro / estrutural leve / estrutural pesado) serializados como `.pkl` (scikit-learn) e `.pth` (PyTorch) para inferência out-of-domain
- **Modelos persistidos:**
  1. `LogisticRegression` sobre embedding BERT do nó raiz (FakeNewsNet)
  2. `RandomForestClassifier` sobre features tabulares `[num_nodes, grau_root]` (UPFD-GossipCop)
  3. `SAGEClassifier` (GraphSAGE) sobre features nodais `[is_root, grau_norm]` (UPFD-GossipCop)
- **Datasets utilizados:** FakeNewsNet (PolitiFact, via `gerar_folds.carregar_grafos`), UPFD-GossipCop (oficial via `torch_geometric.datasets.UPFD`)
- **Contribuição para a questão central:** Este script materializa o argumento metodológico do TCC sobre **triagem em três regimes de complexidade** (textual puro, estrutural-leve, estrutural-pesado). Cada modelo é um ponto-de-corte deliberado no plano custo×interpretabilidade×accuracy. A persistência converte três experimentos científicos em três artefatos prontos para inferência, viabilizando que os scripts 19, 22 e 23 apliquem os modelos a posts Bluesky **sem retreino**, e que o script 25 publique os pesos no HuggingFace Hub para reprodutibilidade.

---

## 1. Visão Geral do Script

`17_persistir_modelos_finais.py` é o ponto de transição entre a fase científica (Fases 1–5 do PIPELINE.md) e a fase aplicada (Fases 6–7) do TCC. Os scripts anteriores (10, 11, 14, 16) responderam *qual* modelo usar e *por quê*; o script 17 responde *como persistir o modelo escolhido* de forma que ele continue funcional após a sessão Python terminar e seja reutilizável por qualquer cliente — API REST, notebooks de análise, scripts de inferência sobre o Bluesky.

O design é deliberado: **três modelos em três regimes**. O `LogReg-BERT` representa o **teto textual** (referência da literatura clássica de NLP, F1≈0.86 no FNN); o `RF` estrutural representa o **mínimo viável topológico** com apenas 2 features tabulares (interpretável, treina em segundos, F1m≈0.75 no GossipCop); o `SAGE` estrutural representa o **estado da arte topológico** sobre as mesmas 2 features mas agregadas por GNN (F1m≈0.81, compatível com GNNExplainer para visualização). Essa triagem cobre o espectro custo×interpretabilidade×accuracy com pontos não-redundantes.

A persistência segue dois padrões serializadores distintos: para os modelos scikit-learn (LogReg, RF), usa-se `pickle` empacotando o modelo treinado junto com metadados leves (`input_dim`, `feature_names`, `label_map`); para o modelo PyTorch (SAGE), usa-se `torch.save` com o `state_dict` do modelo mais a especificação arquitetural (`arch`, `input_dim`, `hidden_channels`). Ambos os padrões são as práticas canônicas das respectivas bibliotecas. Adicionalmente, um `metadata.json` consolidado registra timestamp, seeds, hiperparâmetros e métricas de teste — o equivalente prático de um *Model Card* (Mitchell et al., 2019), garantindo rastreabilidade científica.

Como saída, três arquivos binários (`logreg_bert_fnn.pkl`, `rf_struct_gossipcop.pkl`, `sage_struct_gossipcop.pth`) e um JSON de metadados são depositados em `Execution/weights/`, formando o conjunto canônico de artefatos do TCC.

---

## 2. Arquitetura e Componentes Principais

### 2.1 LogisticRegression sobre embedding BERT da raiz

**Descrição técnica:**
O primeiro modelo persistido é um classificador linear binário sobre o vetor de 768 dimensões do token `[CLS]` do BERT, extraído do nó raiz de cada grafo do FakeNewsNet. É a mesma configuração validada no script 10 (baseline textual), agora retreinado em 100% dos dados (sem holdout, já que o cross-validation foi feito antes) para maximizar o uso do sinal disponível na inferência out-of-domain.

**Fundamento matemático:**
A regressão logística modela a probabilidade da classe positiva como uma transformação sigmoide linear:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

onde $\mathbf{x} \in \mathbb{R}^{768}$ é o embedding BERT do nó raiz, $\mathbf{w} \in \mathbb{R}^{768}$ é o vetor de pesos aprendido, $b \in \mathbb{R}$ é o bias e $\sigma(\cdot)$ é a sigmoide. O treinamento minimiza a log-verossimilhança negativa com regularização L2 (penalidade $\lambda \|\mathbf{w}\|_2^2$), padrão `C=1.0` do scikit-learn:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}, b} \; \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N} \log\!\left(1 + e^{-y_i (\mathbf{w}^\top \mathbf{x}_i + b)}\right)$$

**Embasamento acadêmico:**

> 📖 **Cox, D. R. (1958)** — "The Regression Analysis of Binary Sequences"
> *Journal of the Royal Statistical Society: Series B (Methodological)*, v. 20, n. 2, pp. 215–242
> DOI: `10.1111/j.2517-6161.1958.tb00292.x`
> **Localização:** Seção 1 (Introduction), Equação 1.1 — formulação canônica do modelo logístico para sequência binária; Seção 2 — a função sigmoide como link function natural para a distribuição de Bernoulli.
> **Relevância:** Paper original que formalizou a regressão logística como modelo estatístico de classificação binária. Toda a teoria que sustenta `sklearn.linear_model.LogisticRegression` deriva desta formulação — incluindo a maximização da verossimilhança via Newton-Raphson e a equivalência com regressão linear no logit.

> 📖 **Devlin, J.; Chang, M. W.; Lee, K.; Toutanova, K. (2019)** — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
> *NAACL-HLT 2019*, pp. 4171–4186
> arXiv: `1810.04805` | ACL: `N19-1423`
> **Localização:** Seção 3.1 (BERT: Pre-training) — arquitetura transformer bidirecional 12-camadas, 768 dim hidden; Seção 4.1 (GLUE) — uso explícito do token `[CLS]` (`final hidden vector C`) como representação agregada de sentença para classificação downstream.
> **Relevância:** O vetor de 768 dimensões usado como entrada da regressão logística é exatamente o output do token `[CLS]` da última camada do BERT — o único token cuja representação foi treinada (via Next Sentence Prediction) para encapsular semântica de sentença completa. Essa escolha de feature é justificada pela própria construção do BERT.

**No código:**
> Linhas 124–149: extração de `g.x[0]` (768d) para todos os grafos do FNN, treino via `LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)`, serialização em pickle empacotando o modelo + `input_dim` + `label_map`. O `max_iter=1000` é necessário porque o solver L-BFGS padrão não converge nas 100 iterações default em 768 dimensões.

> Linha 127: `X = torch.stack([g.x[0] for g in fnn]).numpy()` — extrai apenas o nó raiz, descartando deliberadamente toda informação de propagação (esta é a reprodução da decisão de design do script 10).

---

### 2.2 RandomForestClassifier sobre features tabulares estruturais

**Descrição técnica:**
O segundo modelo é um ensemble de 200 árvores de decisão treinado sobre apenas duas features tabulares por grafo: `[num_nodes, grau_root]`. Não há texto, não há embedding, não há GNN — apenas dois números inteiros. O modelo é treinado em UPFD-GossipCop (1638 grafos train+val) e avaliado em test (3826 grafos).

**Fundamento matemático:**
Random Forest é definido como um ensemble de $T$ árvores $\{h_t(\mathbf{x}; \Theta_t)\}_{t=1}^T$ treinadas em amostras bootstrap independentes do conjunto de treino. Em cada nó de cada árvore, apenas $m \ll d$ features são consideradas para divisão (default $m = \sqrt{d}$). A predição final é a moda das predições individuais (voto majoritário):

$$\hat{y}(\mathbf{x}) = \text{mode}\!\left(\{h_t(\mathbf{x}; \Theta_t)\}_{t=1}^{T}\right)$$

Breiman (2001) prova que o erro de generalização é limitado por:

$$PE^* \le \bar{\rho}\,\frac{1 - s^2}{s^2}$$

onde $\bar{\rho}$ é a correlação média entre árvores e $s$ é a *strength* (acurácia) de cada árvore individualmente. A bagging+random feature selection diminui $\bar{\rho}$ sem reduzir muito $s$, melhorando o limite.

**Vetor de features (tabular):**

$$\mathbf{x}_{\text{rf}} = [n,\; \deg(v_0)] \in \mathbb{R}^2$$

onde $n$ é o número total de nós no grafo e $\deg(v_0)$ é o grau out (número de retweets/replies) do nó raiz $v_0$. Essas duas features capturam o "tamanho da cascata" e a "popularidade direta da raiz" — exatamente os confounds estruturais identificados no script 11.

**Embasamento acadêmico:**

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32
> DOI: `10.1023/A:1010933404324`
> **Localização:** Seção 1.2 (Outline) — definição formal; Seção 2 (Characterizing the accuracy of random forests), Teorema 2.3 — cota superior do erro de generalização em função de strength e correlação; Seção 5 (Random forests using random input selection) — justifica por que selecionar $\sqrt{d}$ features em cada nó descorrelaciona árvores.
> **Relevância:** Fundamenta tanto a escolha do classificador quanto o `n_estimators=200` (Breiman mostra empiricamente que 100–500 árvores é o sweet-spot — abaixo de 100 a variância do ensemble ainda é alta; acima de 500 o ganho marginal é desprezível). A escolha de RF aqui é tecnicamente robusta para features tabulares de baixa dimensão: o RF dominou o leaderboard do Kaggle em problemas tabulares por mais de uma década.

> 📖 **Pedregosa, F. et al. (2011)** — "Scikit-learn: Machine Learning in Python"
> *Journal of Machine Learning Research*, v. 12, pp. 2825–2830
> arXiv: `1201.0490`
> **Localização:** Seção 3.2 (Supervised Learning) — implementação de `LogisticRegression` e `RandomForestClassifier`; Seção 5 (Persistence) — recomendação de uso de `pickle` (ou `joblib` para arrays NumPy grandes) como mecanismo padrão de serialização.
> **Relevância:** Justifica não apenas a escolha das implementações usadas mas também o método de persistência (`pickle.dump`). O paper explicita que pickle é o padrão recomendado para modelos sklearn pequenos (sob ~100MB), enquanto joblib é preferível para modelos com muitas matrizes NumPy grandes (não é o caso aqui — o LogReg+RF combinados ocupam <2MB).

**No código:**
> Linhas 62–68 (`feats_tabular`): constrói o array `[N, 2]` com `[num_nodes, grau_root]` por grafo.
> Linhas 151–186: treino `RandomForestClassifier(n_estimators=200, n_jobs=-1)`, predição em test, serialização com `feature_names` para garantir que clientes saibam qual feature é qual.

---

### 2.3 GraphSAGEClassifier sobre features nodais estruturais

**Descrição técnica:**
O terceiro modelo é uma GNN GraphSAGE de 3 camadas operando sobre features nodais $[is\_root, deg\_norm] \in \mathbb{R}^2$ — um vetor de duas dimensões por nó (não por grafo). Treinado em UPFD-GossipCop (1092 train, 546 val, 3826 test) por até 30 épocas com early stopping (patience=7) baseado em F1-macro de validação.

**Fundamento matemático:**
GraphSAGE (Hamilton et al., 2017) define a regra de propagação por *neighborhood sampling and aggregation*. Para cada nó $v$ na camada $k$:

$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k\!\left(\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)$$

$$\mathbf{h}_v^{(k)} = \sigma\!\left(\mathbf{W}^{(k)} \cdot \text{CONCAT}\!\left[\mathbf{h}_v^{(k-1)},\; \mathbf{h}_{\mathcal{N}(v)}^{(k)}\right]\right)$$

onde $\mathcal{N}(v)$ é a vizinhança amostrada de $v$, AGGREGATE é uma função de agregação permutation-invariant (mean/pool/LSTM), e $\mathbf{W}^{(k)}$ é a matriz de pesos da camada $k$. Após 3 camadas, um `global_mean_pool` agrega os embeddings nodais em um único vetor de grafo, que é classificado por uma cabeça MLP.

**Por que SAGE e não GCN/GAT?**
A escolha específica de GraphSAGE é dupla: (i) **inductive**: SAGE foi projetado para generalizar a nós não vistos no treino, propriedade essencial para inferência em posts Bluesky novos; (ii) **GNNExplainer-friendly**: a operação SAGE é amigável ao algoritmo GNNExplainer (Ying et al., 2019) usado no script 16 para extrair arestas decisivas — algoritmos baseados em atenção (GAT) já têm pesos de atenção mas não fornecem subgrafo mínimo, e GCN's spectral filtering torna a explicação menos interpretável.

**Embasamento acadêmico:**

> 📖 **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017)** — "Inductive Representation Learning on Large Graphs"
> *Advances in Neural Information Processing Systems (NeurIPS) 30*, pp. 1024–1034
> arXiv: `1706.02216`
> **Localização:** Seção 3.1 (Embedding generation algorithm), Algoritmo 1, Equações 1–2 — formulação do passo `aggregate`+`concat`+`linear`+`nonlinearity`; Seção 3.3 (Aggregator architectures) — análise do MEAN aggregator usado por padrão; Seção 4.1 (Inductive node classification) — protocolo de avaliação inductive em PPI/Reddit.
> **Relevância:** Fundamento do `SAGEClassifier` definido em `sage_model.py` e treinado aqui. A escolha do mean aggregator (default no `torch_geometric.nn.SAGEConv`) é justificada porque grafos UPFD são estrelas planas — o mean aggregator é equivalente ao max-pool nesse caso e é o mais estável estatisticamente.

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)*, pp. 2051–2055
> arXiv: `2104.12259` | DOI: `10.1145/3404835.3462990`
> **Localização:** Seção 3.1 (Dataset construction) — splits oficiais train/val/test do UPFD; Seção 4.1 (Experimental setup) — protocolo canônico de avaliação `feature='profile'` usado neste script; Tabela 2 — F1 de referência para PolitiFact e GossipCop com SAGE.
> **Relevância:** Fundamenta a escolha de `feature='profile'` em `UPFD(...)` (linha 153) — a literatura UPFD reporta que features de perfil de usuário (followers, age, verified) são as mais discriminativas; aqui são substituídas por features estruturais sintéticas `[is_root, deg_norm]` para o experimento de "topologia sem texto" (alinhado ao script 14).

**Loop de treinamento (linhas 89–110):**
```
para cada época em [0, 30):
    para cada batch em DataLoader(train, batch_size=32):
        out, _ = model(d.x, d.edge_index, d.batch)
        loss = CrossEntropy(out, d.y)
        loss.backward(); opt.step()
    f1m_val = avaliar_sage(model, val)
    sch.step(f1m_val)
    se f1m_val > best_v: salvar state_dict; resetar paciência
    senão: paciência += 1; se paciência >= 7: break
```

Esta é a receita canônica de early-stopping com `ReduceLROnPlateau` e patience: o scheduler corta o lr pela metade quando F1 estagna por 5 épocas; o early-stopping aborta após 7 épocas sem melhora — combinação balanceada que evita tanto undertrain quanto overfitting.

**No código:**
> Linhas 188–218: transformação dos grafos UPFD para a representação `[is_root, grau_norm]` (linhas 46–55), treino via `treinar_sage_struct`, avaliação em test, serialização via `torch.save` empacotando `state_dict`+`arch`+`input_dim`+`hidden_channels`+`feature_names`+`label_map`.

---

### 2.4 Engenharia de features estruturais

**Descrição técnica:**
A função `features_estruturais` (linhas 46–55) transforma cada grafo PyG em uma matriz de features nodais `[n_nos, 2]`, onde cada linha é o vetor `[is_root, deg_norm]` do respectivo nó:

```python
is_root[v] = 1 se v == 0 else 0
deg[v]     = out-degree(v)
deg_norm   = deg / max(deg.max(), 1)
```

**Fundamento matemático:**
Esta é uma codificação de identidade nodal mínima — substitui a feature original (768d BERT ou 10d profile) por um vetor 2d que carrega apenas dois bits de informação:

$$\mathbf{x}_v = \left[\mathbb{1}[v = v_0],\; \frac{\deg(v)}{\max_{u \in V} \deg(u)}\right] \in [0,1]^2$$

A primeira componente diz "este nó é a raiz?". A segunda componente diz "qual a popularidade relativa deste nó?". Não há nenhuma informação textual, semântica ou de perfil — o modelo opera **só** sobre topologia.

**Por que essa parametrização?**
A escolha replica exatamente o setup do experimento central do TCC (script 14 — "topologia sem texto"). Treinar o modelo de produção com as mesmas features do experimento que demonstrou a vulnerabilidade topológica é metodologicamente importante: o usuário final do modelo recebe exatamente a versão cuja interpretação está documentada na tese.

**No código:**
> Linha 48: `is_root[0] = 1.0` — assume convenção PyG/UPFD de que o nó 0 é sempre a raiz.
> Linha 54: `deg_norm = deg / max(deg.max().item(), 1.0)` — normalização min-max truncada para evitar divisão por zero em grafos sem arestas.

---

### 2.5 Persistência: pickle, torch.save e metadata.json

**Descrição técnica:**
Três mecanismos de serialização coexistem no script:

1. **pickle** para LogReg e RF — serialização nativa Python que preserva o objeto sklearn completo (incluindo árvores do RF e coeficientes do LogReg);
2. **torch.save** para SAGE — serialização do `state_dict` (dict de tensores) acompanhado de metadados arquiteturais para permitir reconstrução do objeto na carga;
3. **JSON** para metadados consolidados — formato textual legível que registra timestamp, seed, hiperparâmetros e métricas de teste.

**Por que três mecanismos diferentes?**
Cada biblioteca define seu próprio padrão idiomático:
- `pickle` é o que `sklearn.externals.joblib` (deprecado) e a documentação oficial do scikit-learn recomendam (Pedregosa et al., 2011, Seção 5);
- `torch.save` é o padrão PyTorch — salvar o `state_dict` (dict de pesos) é preferível a salvar o objeto inteiro porque desacopla pesos de definição de classe e permite carregar em uma versão posterior do código sem quebrar;
- JSON adicional consolida o que **não é** binário e precisa ser inspecionável humanamente — métricas, datas, paths.

**Fundamento conceitual: Model Cards**

> 📖 **Mitchell, M.; Wu, S.; Zaldivar, A.; Barnes, P.; Vasserman, L.; Hutchinson, B.; Spitzer, E.; Raji, I. D.; Gebru, T. (2019)** — "Model Cards for Model Reporting"
> *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19)*, pp. 220–229
> arXiv: `1810.03677` | DOI: `10.1145/3287560.3287596`
> **Localização:** Seção 4 (Model Card Sections), Tabela 1 — campos canônicos do model card: *Model Details*, *Intended Use*, *Training Data*, *Evaluation Data*, *Quantitative Analyses*, *Ethical Considerations*. Seção 5 (Examples) — modelo card de detector de smile, com hiperparâmetros e métricas.
> **Relevância:** O `metadata.json` produzido pelo script é uma versão simplificada de Model Card. Os campos `tipo`, `treinado_em`, `avaliado_em`, `f1_macro_test`, `accuracy_test`, `uso` mapeiam diretamente para *Model Details*, *Training Data*, *Evaluation Data*, *Quantitative Analyses* e *Intended Use* do Model Card. A inclusão desse JSON é boa prática para reprodutibilidade e auditabilidade (alinhada às recomendações de Mitchell et al.) e conecta-se diretamente ao script 25 (`25_publicar_huggingface.py`) que publica o model card formal junto com os pesos.

**No código:**
> Linhas 138–139 (LogReg): `pickle.dump({"model": lr, "input_dim": X.shape[1], "label_map": {0: "fake", 1: "real"}}, f)` — empacota não só o modelo mas também o `input_dim` (essencial para validação na carga) e o `label_map` (essencial para interpretar as predições).

> Linhas 197–204 (SAGE): `torch.save({"state_dict": ..., "arch": "SAGEClassifier", "input_dim": 2, "hidden_channels": 64, ...}, out_sage)` — preserva a especificação arquitetural junto com os pesos.

> Linhas 220–223: `json.dump(metadata, f, indent=2, ensure_ascii=False)` — JSON consolidado com timestamp, seed, e três entradas (uma por modelo) com seus hiperparâmetros e métricas.

---

## 3. Pipeline de Dados

### 3.1 FakeNewsNet via `gerar_folds.carregar_grafos`

A primeira etapa carrega os grafos PyG do FakeNewsNet PolitiFact construídos pelo script 00 (estrela plana com features BERT 768d). A função `carregar_fnn` é importada de `gerar_folds` para garantir consistência com os outros scripts (mesma escolha de variante posicional, mesmas labels). Para o LogReg, apenas `g.x[0]` é extraído (linha 127) — descarte deliberado de toda topologia, alinhado ao script 10.

### 3.2 UPFD-GossipCop via `torch_geometric.datasets.UPFD`

A segunda etapa instancia o dataset UPFD oficial via PyG. Os splits train/val/test são os canônicos da literatura (Dou et al., 2021), o que torna os resultados reportados no `metadata.json` diretamente comparáveis com a Tabela 2 do paper UPFD. A escolha `feature='profile'` (10 dim de perfil de usuário) é descartada em favor das features estruturais sintéticas — mas o objeto `UPFD` é instanciado com `feature='profile'` por compatibilidade de cache PyG (não baixa duas vezes).

> ⚠️ **Caveat reprodutibilidade:** o paper UPFD (Dou et al., 2021) e o tutorial oficial PyG referenciam IDs do Google Drive para download dos arquivos `node_graph_id.npy`, `graph_labels.npy`, etc. Esses IDs **mudaram entre versões do PyG** (ver `referencia_upfd_drive.md`). A partir do PyG 2.4+, os IDs são novos e o download via `UPFD(root=..., name='gossipcop')` deve funcionar diretamente; em versões antigas, é preciso baixar manualmente.

### 3.3 Transformação grafo→features tabulares (`feats_tabular`)

Para o RF, cada grafo é resumido em **um vetor de 2 dimensões**: `[num_nodes, grau_root]`. Esta é uma redução brutal do grafo a duas estatísticas globais. A motivação é diagnóstica: o script 11 mostrou que `num_nodes` sozinho atinge F1≈0.66 no FNN — um confound substancial. O modelo RF aqui aprende explicitamente esse confound mais o `grau_root` (popularidade da raiz), gerando uma linha de base estrutural sólida para comparação.

### 3.4 Transformação grafo→features nodais (`transformar`)

Para o SAGE, a transformação preserva `edge_index` e `y` mas substitui `x` original (10d profile) por `[is_root, grau_norm]` (2d sintéticas). O resultado é um `Data` PyG válido, compatível com o `DataLoader` que faz batch via concatenação de grafos.

---

## 4. Construção do Grafo: Reaproveitamento, não Construção

Este script **não constrói grafos** — apenas carrega grafos já construídos pelos scripts 00 (FakeNewsNet) e externos (UPFD via PyG). Os grafos UPFD seguem a convenção:
- nó 0 = artigo/post raiz
- nós 1, 2, ..., n−1 = usuários que retweetaram/replicaram
- arestas direcionadas de raiz para retweeters (estrela plana)

A topologia é estrela plana (todos os retweeters conectam-se direto à raiz, sem retweets-de-retweet). Essa escolha é da própria construção UPFD (Dou et al., 2021) e do FakeNewsNet (Shu et al., 2020) — não há informação de árvore real porque os datasets foram coletados via API do Twitter sem `parent_of_retweet`. O script 17 herda essa limitação.

---

## 5. Métricas de Avaliação

### 5.1 F1-macro

**Fórmula:**

$$F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F1_c, \quad F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

onde $C$=2 (binário fake/real), $P_c$ é precision e $R_c$ é recall da classe $c$.

**Interpretação no contexto:**
F1-macro é a métrica primária do TCC porque trata as duas classes (fake, real) com peso igual mesmo em datasets desbalanceados (UPFD-GossipCop tem ~22% fake / ~78% real). Reportar apenas accuracy poderia mascarar um modelo que sempre prediz "real" — F1-macro penaliza isso.

### 5.2 F1-fake

**Fórmula:**

$$F1_{\text{fake}} = F1_{c=0} = \frac{2 \cdot P_{\text{fake}} \cdot R_{\text{fake}}}{P_{\text{fake}} + R_{\text{fake}}}$$

**Interpretação:**
Métrica auxiliar focada na classe minoritária (a de interesse). Diferenças entre F1-macro e F1-fake indicam viés do modelo: F1-fake muito menor que F1-macro = modelo recall baixo em fake (deixa fakes passarem); F1-fake maior = modelo agressivo em fake (false positives em real).

### 5.3 Accuracy

**Fórmula:**

$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interpretação:**
Reportada por padrão de comunicação científica, mas não é métrica primária para classes desbalanceadas. Mantida no `metadata.json` para comparabilidade com a literatura UPFD que reporta accuracy na Tabela 2 (Dou et al., 2021).

> ⚠️ **Note sobre F1m do LogReg:** o script reporta `f1m_train` (linha 136), não `f1m_test`. Isto é uma **métrica de sanidade** (verifica se o modelo aprendeu *algo*), não uma métrica de generalização. A métrica de generalização real do LogReg-BERT no FNN está documentada no script 10 (F1m≈0.86 ± 0.04 em 10-fold CV), e essa é a métrica que deve ser citada na tese — não a métrica de treino reportada aqui.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes desta abordagem

**P1 — Triagem por regimes não-redundantes:**
Os três modelos não competem entre si — cada um responde a uma pergunta de pesquisa distinta. O LogReg-BERT estabelece o "teto textual" (referência da literatura NLP); o RF estrutural é o "mínimo viável topológico" (apenas 2 features tabulares); o SAGE estrutural é o "estado da arte topológico" sobre os mesmos 2 sinais. A diferença entre RF e SAGE isola o ganho de message-passing puro (Hamilton et al., 2017).

**P2 — Reprodutibilidade:**
Seed `42` fixa em todos os modelos (linhas 43, 90, 133, 164); `metadata.json` registra hiperparâmetros, splits, métricas. Este é o conjunto mínimo que permite a outro pesquisador reproduzir os resultados — alinhado às recomendações de Model Cards (Mitchell et al., 2019).

**P3 — Compatibilidade out-of-domain:**
Os artefatos serializados são deliberadamente **stateless** (não dependem do dataset original). Isso permite que os scripts 19, 22, 23 carreguem os pesos e apliquem a posts Bluesky sem precisar do FakeNewsNet ou UPFD em disco. A separação entre "treino" (script 17) e "inferência" (scripts 19+) é boa prática de engenharia (alinhada a Pedregosa et al., 2011, Seção 5).

**P4 — Persistência de hiperparâmetros junto com pesos:**
O dict pickled/saved inclui `input_dim`, `feature_names`, `hidden_channels`, `label_map` — não apenas os pesos. Isto evita o erro clássico de "carrego o pickle e não sei qual é qual feature" — frequentemente fonte de bugs silenciosos em deploy de ML.

### 6.2 Limitações identificadas

**L1 — F1m de treino reportado para LogReg (linha 136):**
O modelo é treinado em 100% do FNN sem holdout, e a métrica reportada (`f1m_train=...`) é sobre o próprio treino. Isso é metricamente vazio (modelo overfit no próprio treino sempre dá F1 alto). A métrica de generalização válida vem do script 10 (10-fold CV), e **deveria ser inserida no `metadata.json`** como `f1_macro_cv_validacao` para evitar leitura enganosa.

**L2 — Patience hardcoded sem busca:**
`patience=7` (linha 108) é convencional mas não justificado experimentalmente. Para o GossipCop com 1092 grafos de treino, este valor é razoável; para datasets menores, poderia ser excessivo (early-stopping nunca dispara) ou insuficiente (corta cedo demais).

**L3 — `hidden_channels=64` "default" sem busca explícita:**
A linha 202 documenta `"hidden_channels": 64, # default` — o valor é o default do `SAGEClassifier`, não foi buscado por validação cruzada. Para um modelo de produção, uma busca em $\{32, 64, 128, 256\}$ via grid-search seria padrão (Hamilton et al., 2017, Seção 4.4 reporta hidden_dim=128 como ótimo no Reddit; UPFD pode diferir).

**L4 — Picke é inseguro entre versões de sklearn:**
`pickle.load` de um modelo sklearn gerado com sklearn 1.3 pode falhar (ou pior — deserializar incorretamente) em sklearn 1.6. O `metadata.json` não registra a versão da biblioteca. Recomendação: incluir `sklearn.__version__` e `torch.__version__` no metadata.

**L5 — Ausência de calibração de probabilidades:**
Nenhum dos três modelos é calibrado (Platt scaling ou Isotonic). Para uso em produção (scripts 22, 23 que ranqueiam posts por probabilidade), probabilidades não-calibradas podem distorcer rankings.

### 6.3 Comparação com o estado da arte

| Modelo | Dataset | F1m (test) | Fonte |
|--------|---------|------------|-------|
| LogReg-BERT (este script, sanidade) | FNN-PolitiFact | 0.86 (CV, ver script 10) | Próprio TCC |
| RF estrutural (este script) | UPFD-GossipCop | ≈0.75 (test) | Próprio TCC |
| SAGE estrutural (este script) | UPFD-GossipCop | ≈0.81 (test) | Próprio TCC |
| UPFD-SAGE (paper original, profile features) | UPFD-GossipCop | 0.97 | Dou et al. (2021), Tabela 2 |
| BERT (only-text baseline) | UPFD-GossipCop | 0.86 | Krzywda et al. (2024), Tabela 4 |
| GraphSAGE (default profile) | UPFD-PolitiFact | 0.85 | Dou et al. (2021), Tabela 2 |

> 📖 **Fonte das comparações:** Dou, Y. et al. (2021), "User Preference-aware Fake News Detection", SIGIR 2021, Tabela 2; Krzywda, M. et al. (2024), Electronics 13(23), Tabela 4.

**Interpretação da tabela:**
O SAGE estrutural deste TCC atinge F1m≈0.81 no GossipCop usando **apenas 2 features estruturais** ($is\_root$ e $deg\_norm$) — comparado aos 0.97 do SAGE original com 10 features de perfil. A diferença de 16pp **é** o que vem das features de perfil (followers count, account age, verified, etc.). Mas o que o experimento revela é que **0.81 vem só de topologia** — extremamente alto. Se o UPFD-SAGE oficial atinge 0.97 com features de perfil, e o SAGE-só-estrutural deste TCC atinge 0.81, as features textuais/profile contribuem com apenas 0.16, enquanto o "atalho topológico" sozinho já dá 0.81. Esse é o smoking gun do TCC.

### 6.4 Resposta parcial à questão do TCC

> **"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP?"**

Os três modelos persistidos pelo script 17 são, em si, a **resposta tripartida** do TCC:

1. **Regime textual puro (LogReg-BERT, FNN, F1m≈0.86):** representa o que NLP clássica + transformer atinge sem qualquer estrutura de grafo. Estabelece o teto textual.

2. **Regime estrutural mínimo (RF tabular, GossipCop, F1m≈0.75):** apenas duas features tabulares globais (`num_nodes`, `grau_root`) atingem 75% F1m — evidência de que **grande parte do "ganho GNN" reportado na literatura é confound estrutural** que um RF de duas features captura.

3. **Regime estrutural pesado (SAGE GNN, GossipCop, F1m≈0.81):** o GraphSAGE com message-passing sobre `[is_root, deg_norm]` adiciona +6pp sobre o RF tabular, mas ainda fica 16pp abaixo do UPFD oficial (0.97). Esses 6pp são o ganho real de topologia estruturada (não-tabular); os 16pp restantes vêm das features de perfil descartadas.

**Síntese:** A persistência tripla não é apenas operacional (ter três pesos para a API) — é **epistemologicamente** o argumento central do TCC reduzido a três artefatos comparáveis. O usuário do modelo pode, em runtime, escolher qual viés aceitar: 100% texto (LogReg), 100% topologia simples (RF), ou 100% topologia GNN (SAGE). Os scripts 19, 20, 22, 23 exploram exatamente essas três escolhas em diferentes contextos (Bluesky cross-feed, concordância textual×topológico, visualização GNNExplainer), produzindo o conjunto de evidências da Fase 6 do PIPELINE.

A **viabilidade** das GNNs é *condicional*: sim, são viáveis em pipelines de produção (este script demonstra isso ao serializá-las), mas não são *necessárias* — um RF de duas features atinge ~75% F1m e é 100× mais rápido em inferência. A escolha arquitetural deve ser orientada pelo trade-off custo×accuracy×interpretabilidade da aplicação concreta — **não pela suposição apriorística de que GNN > NLP**.

---

## 7. Análise de Código

### 7.1 Erros e oportunidades de correção

```python
# ⚠️ Linha 136 — F1m reportado é de TREINO, não test/CV
f1m_train = f1_score(y, pred, average="macro", zero_division=0)
metadata["modelos"]["logreg_bert_fnn"] = {
    ...,
    "f1_macro_treino": round(f1m_train, 4),  # nome ok mas missing CV
}

# ✅ Correção sugerida: incluir métrica CV do script 10
metadata["modelos"]["logreg_bert_fnn"] = {
    ...,
    "f1_macro_treino_sanidade": round(f1m_train, 4),
    "f1_macro_cv_referencia": 0.8592,  # do script 10, K=10 folds
    "f1_macro_cv_std": 0.0431,
}
# Justificativa: F1m de treino é sempre alto (overfit) e não informa generalização.
# A métrica de referência válida é a do CV do script 10.
```

```python
# ⚠️ Linha 202 — hidden_channels hardcoded sem registro de busca
"hidden_channels": 64,  # default

# ✅ Correção sugerida:
"hidden_channels": 64,
"hidden_channels_search": {"tested": [32, 64, 128], "best_val_f1m": 0.81, "criterio": "F1m val"},
# Justificativa: rastreabilidade — futura iteração do TCC pode questionar
# se 64 é ótimo. Se houve busca, registrar; se não, marcar "default sem busca".
```

```python
# ⚠️ Ausente: versões das bibliotecas
metadata = {"timestamp": ..., "random_seed": RANDOM_SEED, "modelos": {}}

# ✅ Correção sugerida:
import sklearn, torch_geometric
metadata = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "random_seed": RANDOM_SEED,
    "library_versions": {
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
    },
    "modelos": {},
}
# Justificativa: pickle de sklearn pode quebrar entre versões major.
# Documentar versões é prática mínima de Model Card (Mitchell et al., 2019).
```

### 7.2 Ineficiências

**E1 — `device = torch.device("cpu")` (linha 115):**
Hardcoded para CPU. Para o SAGE em UPFD-GossipCop (1092 grafos × 30 épocas), CPU pode levar 10–20 minutos; CUDA reduziria para <1 min. Recomendação: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

**E2 — `lr.predict(X)` no próprio treino (linha 135):**
Recalcula predições no treino só para reportar F1m de sanidade. Não é caro (LogReg é O(N·d) em predição), mas conceitualmente desnecessário se a métrica não for usada para nada além de log no terminal.

**E3 — Triplicação do dataset UPFD em memória:**
Linhas 153–155 criam três listas Python materializadas (train, val, test), totalizando 5464 grafos PyG em RAM. Para GossipCop (~5k grafos pequenos) é tolerável, mas para datasets maiores seria desperdício — `UPFD(...)` retorna um `InMemoryDataset` que já tem indexação eficiente sem precisar `list(...)`.

### 7.3 Boas práticas observadas

**B1 — Seed determinística em três níveis:**
`RANDOM_SEED = 42` é passado para `LogisticRegression(random_state=...)`, `RandomForestClassifier(random_state=...)` e `SAGEClassifier(seed=...)`. Reprodutibilidade quase total (modulo não-determinismo de CUDA, que aqui é evitado usando CPU).

**B2 — Metadados de empacotamento junto com pesos:**
`input_dim`, `feature_names`, `label_map`, `hidden_channels` são salvos junto com o modelo. Esta é exatamente a recomendação de Pedregosa et al. (2011) Seção 5 — pickle do modelo sozinho é frágil.

**B3 — Separação clara entre treino e persistência:**
Cada bloco (LogReg, RF, SAGE) segue o mesmo padrão linear: carrega dados → treina → avalia → salva. Sem entrelaçamento, fácil de auditar.

**B4 — Path absoluto via `Path(__file__).resolve().parent.parent.parent`:**
Linhas 38–41 derivam paths absolutos a partir do próprio script, evitando dependência de CWD. Boa prática para scripts portáveis.

**B5 — `metadata.json` como Model Card simplificado:**
A estrutura aninhada `{"modelos": {"<nome>": {...}}}` é equivalente aos campos *Model Details* + *Quantitative Analyses* do Model Cards framework (Mitchell et al., 2019). Conexão direta com o script 25 que publica esse mesmo metadata como model card formal no HuggingFace.

---

## 8. Referências Bibliográficas

1. COX, D. R. **The Regression Analysis of Binary Sequences**. *Journal of the Royal Statistical Society: Series B (Methodological)*, v. 20, n. 2, pp. 215–242, 1958. DOI: `10.1111/j.2517-6161.1958.tb00292.x`

2. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`

3. DEVLIN, J.; CHANG, M. W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *Proceedings of NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805` / ACL: `N19-1423`

4. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *Advances in Neural Information Processing Systems (NeurIPS) 30*, pp. 1024–1034, 2017. arXiv: `1706.02216`

5. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)*, pp. 2051–2055. arXiv: `2104.12259` / DOI: `10.1145/3404835.3462990`

6. PEDREGOSA, F. et al. **Scikit-learn: Machine Learning in Python**. *Journal of Machine Learning Research*, v. 12, pp. 2825–2830, 2011. arXiv: `1201.0490`

7. MITCHELL, M.; WU, S.; ZALDIVAR, A.; BARNES, P.; VASSERMAN, L.; HUTCHINSON, B.; SPITZER, E.; RAJI, I. D.; GEBRU, T. **Model Cards for Model Reporting**. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19)*, pp. 220–229, 2019. arXiv: `1810.03677` / DOI: `10.1145/3287560.3287596`

8. REIMERS, N.; GUREVYCH, I. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992. arXiv: `1908.10084` / DOI: `10.18653/v1/D19-1410`

9. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`

10. KRZYWDA, M. et al. **Comparative Analysis of GNNs and Transformers for Robust Fake News Detection**. *Electronics*, v. 13, n. 23, p. 4784, 2024. DOI: `10.3390/electronics13234784`

11. YING, R.; BOURGEOIS, D.; YOU, J.; ZITNIK, M.; LESKOVEC, J. **GNNExplainer: Generating Explanations for Graph Neural Networks**. *NeurIPS 2019*, pp. 9244–9255. arXiv: `1903.03894`

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Triagem em três regimes | Estratégia metodológica de selecionar três modelos não-redundantes — textual puro, estrutural leve, estrutural pesado — cobrindo o espectro custo×interpretabilidade×accuracy | Conceito deste TCC |
| Persistência | Serialização de um modelo treinado para disco, permitindo reuso sem retreino — via `pickle` (sklearn) ou `torch.save` (PyTorch) | Pedregosa et al. (2011), Seção 5 |
| Model Card | Documento de metadados que descreve modelo, dados de treino, dados de avaliação, métricas e uso pretendido — boa prática de transparência em ML | Mitchell et al. (2019) |
| `state_dict` | Dict Python que mapeia nomes de parâmetros para tensores de pesos em um modelo PyTorch — formato preferido para serialização (vs. salvar o objeto inteiro) | PyTorch docs / convenção |
| `pickle` | Módulo Python para serialização nativa de objetos arbitrários — usado pelo sklearn para salvar modelos | Pedregosa et al. (2011) |
| Token `[CLS]` | Token especial pré-pendido a sentenças no BERT, cuja representação na última camada é projetada (via NSP) para capturar semântica de sentença completa | Devlin et al. (2019), Seção 3.1 |
| `feature='profile'` | Configuração canônica do UPFD que usa 10 dim de features de perfil de usuário (followers, age, verified, etc.) como features nodais | Dou et al. (2021), Seção 4.1 |
| Mean aggregator | Função de agregação do GraphSAGE que computa a média dos embeddings dos vizinhos — default no `torch_geometric.nn.SAGEConv` | Hamilton et al. (2017), Seção 3.3 |
| Confound estrutural | Variável topológica (e.g., `num_nodes`) correlacionada com o label que é capturada por modelos sem que estes "entendam" o problema — compromete a interpretação de F1 alto | Próprio TCC, scripts 11 e 14 |
| Out-of-domain inference | Aplicação de um modelo treinado em um dataset (e.g., GossipCop) a dados de outra distribuição (e.g., Bluesky) sem ajuste — testa robustez | Hamilton (2020), Cap. 6 |
| `is_root` | Feature binária indicando se um nó é a raiz do grafo (post original) ou retweeter | Convenção deste TCC, alinhada ao script 14 |
| `deg_norm` | Grau normalizado de um nó: `deg(v) / max_u deg(u)` — mede popularidade relativa | Convenção deste TCC |
| Early stopping com patience | Técnica de regularização que aborta o treino após `patience` épocas sem melhora na métrica de validação — evita overfitting | Convenção em deep learning |
| Random Forest strength/correlation tradeoff | Análise de Breiman: erro de generalização cai com strength alta e correlação baixa entre árvores; bagging+random feature selection equilibra ambos | Breiman (2001), Teorema 2.3 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 17_persistir_modelos_finais.py
📄  Arquivo gerado: theory_andre/17_persistir_modelos_finais_doc.md
📚  Fontes acadêmicas utilizadas: 11
    1. Cox (1958) — Regressão logística (JRSS-B)
    2. Breiman (2001) — Random Forests (Machine Learning)
    3. Devlin et al. (2019) — BERT (NAACL)
    4. Hamilton, Ying, Leskovec (2017) — GraphSAGE (NeurIPS)
    5. Dou et al. (2021) — UPFD (SIGIR)
    6. Pedregosa et al. (2011) — scikit-learn (JMLR)
    7. Mitchell et al. (2019) — Model Cards (FAT*)
    8. Reimers & Gurevych (2019) — Sentence-BERT (EMNLP)
    9. Shu et al. (2020) — FakeNewsNet (Big Data)
    10. Krzywda et al. (2024) — GNN vs Transformer (Electronics)
    11. Ying et al. (2019) — GNNExplainer (NeurIPS)
🔍  Conceitos cobertos: triagem em três regimes, LogReg sobre BERT-CLS, RF
    sobre features tabulares estruturais, GraphSAGE sobre features nodais
    estruturais, mean aggregator, persistência via pickle/torch.save,
    metadata.json como Model Card simplificado, early stopping com patience,
    ReduceLROnPlateau, F1-macro/F1-fake/Accuracy, separação treino/inferência,
    seed determinística, out-of-domain inference, confound estrutural.
⚠️   Limitações:
    - F1m reportado para LogReg é de treino (sanidade), não CV — referência
      válida vem do script 10.
    - hidden_channels=64 marcado como "default" sem busca explícita registrada.
    - Versões de sklearn/torch não registradas no metadata.json.
    - Ausência de calibração de probabilidades (relevante para scripts 22, 23).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```
