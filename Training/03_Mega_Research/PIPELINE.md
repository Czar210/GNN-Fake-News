# Pipeline do Mega Research — Objetivo, Matéria-prima, Produto e Porquê

Documento de referência para entender o papel de cada script em
`Training/03_Mega_Research/`. Organizado por **fase experimental**, não por
ordem numérica.

---

## Fase 0 — Construção dos dados

### [00_construir_grafos_fakenewsnet.py](00_construir_grafos_fakenewsnet.py)
- **Objetivo:** construir grafos de propagação a partir do CSV bruto do FakeNewsNet (PolitiFact), em estrela plana com features posicionais.
- **Matéria-prima:** CSVs `politifact_fake.csv` e `politifact_real.csv` baixados do GitHub do KaiDMML; modelo BERT multilíngue.
- **Produto:** `.pt` em `Training/03_Mega_Research/data/fakenewsnet_<suffix>/`; variantes `posfull`, `posmin`, `posgrau`.
- **Porquê:** o CSV bruto não tem `parent_of_retweet` nem timestamp, então hidratar uma cascata real exigiria a API paga do X. A estrela plana é a única forma viável **e** cria o par de comparação contra UPFD ("mesmo dataset, topologia mais pobre") que isola o ganho que vem da cascata.
- **Caveat:** ordem dos `tweet_ids` é a do CSV, não cronológica.

### [01_baixar_dataset_hf.py](01_baixar_dataset_hf.py)
- **Objetivo:** baixar o dataset Bluesky do HuggingFace e converter em CSVs compatíveis com a pipeline.
- **Matéria-prima:** `Zaras210/bluesky-fake-news-dataset` no HF Hub.
- **Produto:** `posts_coletados.csv` e `reposts_coletados.csv` em `Training/01_BlueSky_Pipe/data/raw/`.
- **Porquê:** coletar 166k posts direto da API do Bluesky levaria semanas com rate limits; o dump do HF já está pronto. Como o Bluesky não tem rótulo nativo de fake/real, a rotulagem é heurística por keywords — explicitamente um proxy fraco, mas suficiente porque o experimento Bluesky é qualitativo (cross-feed e concordância), não dependente de ground truth.

### [02_construir_grafos_bluesky.py](02_construir_grafos_bluesky.py)
- **Objetivo:** construir grafos PyG do Bluesky em estrela plana.
- **Matéria-prima:** CSVs do Bluesky (saída do 01); BERT multilíngue.
- **Produto:** `.pt` em `data/bluesky_<suffix>/`; variantes `full`, `pos-min`, `pos-grau`, `notext`.
- **Porquê:** mesma limitação do 00 — sem `parent_of_repost`, só dá para construir estrela. A vantagem é que isso vira **paridade topológica perfeita** com FakeNewsNet, permitindo comparação direta sem confound estrutural.

---

## Definições reutilizadas (utilitários)

### [gcn_model.py](gcn_model.py)
Define `GCNClassifier` (3 GCNConv + ReLU). **Porquê:** baseline canônico (Kipf & Welling), reutilizado em 4+ scripts — centralizar evita divergência silenciosa de hiperparâmetros.

### [gat_model.py](gat_model.py)
Define `GATClassifier` (3 camadas com atenção). **Porquê:** representa a família "atenção sobre vizinhos" (Veličković et al.); diferenças GCN×GAT testam se ponderar vizinhos importa neste problema.

### [sage_model.py](sage_model.py)
Define `SAGEClassifier`. **Porquê:** completa o trio canônico (espectral, atenção, sampling); GraphSAGE escala melhor para grafos heterogêneos e é o mais usado em produção.

### [gerar_folds.py](gerar_folds.py)
Gera `StratifiedKFold(n_splits=10)` em `data/folds_fnn.pt`. **Porquê:** o t-test pareado do script 09 só é válido se todos os modelos forem avaliados nos **mesmos** folds; gerar uma vez e reutilizar elimina variância espúria.

---

## Fase 1 — Treino e comparação de arquiteturas (Bluesky)

### [03_treinar_gat.py](03_treinar_gat.py)
- **Objetivo:** treinar 4 variantes do GAT no Bluesky (baseline, controlado, cético, extra-cético).
- **Matéria-prima:** grafos `data/bluesky_*.pt`; `gat_model.py`.
- **Produto:** pesos em `Execution/weights/pesos_gat*.pth`.
- **Porquê:** Bluesky é altamente desbalanceado (a heurística rotula poucos posts como "fake"). Variar peso de classe é a forma mais simples de medir como o modelo se comporta em diferentes regimes de balanceamento — útil para escolher um "ponto de operação" que não colapse para a classe majoritária.

### [04_comparar_gcn_gat.py](04_comparar_gcn_gat.py)
- **Objetivo:** benchmark direto GCN vs GAT.
- **Matéria-prima:** grafos Bluesky; modelos GCN e GAT.
- **Produto:** matrizes, gráfico de métricas e relatório em `Execution/results/comparativo_gcn_gat/`.
- **Porquê:** GCN trata vizinhos com peso uniforme; GAT aprende pesos. Em estrelas planas (Bluesky/FNN), os vizinhos são quase indistinguíveis — então atenção pode ser desnecessária. Esse experimento testa empiricamente se o overhead do GAT compensa.

### [05_treinar_sage.py](05_treinar_sage.py)
- **Objetivo:** treinar 4 variantes do GraphSAGE.
- **Matéria-prima:** grafos Bluesky; `sage_model.py`.
- **Produto:** pesos em `Execution/weights/pesos_sage*.pth`.
- **Porquê:** mesma motivação do 03, mas para SAGE. Replicar a tríade completa de modelos garante que não estamos privilegiando uma família arquitetural por acidente.

### [06_comparar_gcn_gat_sage.py](06_comparar_gcn_gat_sage.py)
- **Objetivo:** benchmark triplo (GCN/GAT/SAGE) — extensão do 04.
- **Matéria-prima:** grafos Bluesky; três arquiteturas.
- **Produto:** ranking em `Execution/results/comparativo_gcn_gat_sage/`.
- **Porquê:** com 2 modelos qualquer diferença pode ser ruído; com 3 famílias diferentes, se todas convergem para o mesmo F1, o sinal está na **estrutura dos dados**, não na arquitetura — e isso é parte do argumento do TCC.

---

## Fase 2 — Benchmarks no UPFD oficial (referência da literatura)

### [07_upfd_benchmark_triplo.py](07_upfd_benchmark_triplo.py)
- **Objetivo:** benchmark GCN/GAT/SAGE no UPFD PolitiFact e GossipCop.
- **Matéria-prima:** UPFD via PyG; opcionalmente grafos do 00.
- **Produto:** matrizes, curvas e relatório em `Execution/results/upfd_benchmark_<dataset>/`.
- **Porquê:** o TCC precisa ancorar resultados num **benchmark da literatura**. Bluesky e FakeNewsNet construído por nós são experimentais; UPFD oficial é o que outros papers usam, então só comparações em cima dele são publicáveis.

### [13_benchmark_upfd_oficial.py](13_benchmark_upfd_oficial.py)
- **Objetivo:** comparar baseline textual vs GCN/GAT/SAGE no UPFD oficial, com 5 seeds.
- **Matéria-prima:** UPFD via PyG.
- **Produto:** `resultados.csv` em `Execution/results/fase4_benchmarks/benchmark_upfd_oficial/`.
- **Porquê:** o 07 compara arquiteturas; este compara **paradigmas** (texto puro vs grafo). É a pergunta direta do orientador: "quanto o GNN ganha sobre um classificador textual ingênuo?". Múltiplas seeds porque GNNs no UPFD têm variância alta entre rodadas.

---

## Fase 3 — Generalização entre datasets

### [08_inferencia_cruzada.py](08_inferencia_cruzada.py)
- **Objetivo:** modelo treinado num dataset aplicado em outro (BS↔FNN).
- **Matéria-prima:** modelos do 03/05/07; testes do BS e FNN.
- **Produto:** matrizes cruzadas em `Execution/results/fase4_benchmarks/inferencia_cruzada/`.
- **Porquê:** se o modelo aprendeu "fake news" como conceito, transfere; se aprendeu o atalho topológico daquele dataset específico, falha. Essa queda na transferência é evidência direta de overfitting topológico.

---

## Fase 4 — Baselines e diagnóstico de confound

### [09_teste_significancia.py](09_teste_significancia.py)
- **Objetivo:** t-test pareado entre GCN/GAT/SAGE em 10 folds.
- **Matéria-prima:** grafos FakeNewsNet; `data/folds_fnn.pt`.
- **Produto:** boxplot, violino, tabela em `Execution/results/fase4_benchmarks/teste_significancia/`.
- **Porquê:** "GAT bate GCN em 0.5pp" não diz nada sem teste estatístico. T-test pareado por fold é mais poderoso que não-pareado porque cancela variância do dado e isola só a diferença entre modelos.

### [10_baseline_textual.py](10_baseline_textual.py)
- **Objetivo:** F1 só com BERT da raiz (LogReg + RF), sem topologia.
- **Matéria-prima:** grafos FakeNewsNet; folds; embedding da raiz.
- **Produto:** `resultados.csv` em `Execution/results/fase2_baselines/baseline_textual/`.
- **Porquê:** estabelece o **teto textual**. Qualquer F1 que o GNN bater acima desse número vem necessariamente da topologia — separa as contribuições de forma matematicamente clean.

### [11_diagnostico_confound.py](11_diagnostico_confound.py)
- **Objetivo:** quanto F1 sobra usando só `num_nodes` como feature?
- **Matéria-prima:** grafos FakeNewsNet; folds.
- **Produto:** `resultados.csv` em `Execution/results/fase2_baselines/confound_diagnostico/`.
- **Porquê:** tamanho do grafo é confound clássico em propagação (fakes virais → mais nós → label correlacionado). Antes de afirmar que "o GNN aprende estrutura sutil", precisa quantificar quanto vem só do confound trivial; senão o argumento é circular.
- **Caveat:** gate de F1>0.65 é heurístico.

### [12_ablation_intra_encoding.py](12_ablation_intra_encoding.py)
- **Objetivo:** ablação `pos` vs `grau_norm` (variantes posfull/posmin/posgrau).
- **Matéria-prima:** grafos FakeNewsNet variantes; folds; baseline textual.
- **Produto:** tabela em `Execution/results/fase3_ablation/ablation_intra_encoding/`.
- **Porquê:** as 3 dims posicionais foram criadas para resolver o Erro 1 (features nodais idênticas), mas qual delas carrega o sinal? Essa ablação responde "ordinal vs popularidade" e simplifica a interpretação na escrita.

---

## Fase 5 — Vulnerabilidade topológica (núcleo do TCC)

### [14_topologia_sem_texto.py](14_topologia_sem_texto.py)
- **Objetivo:** GCN/GAT/SAGE com **só** features estruturais, sem BERT, em FNN/UPFD-Polit/UPFD-Goss.
- **Matéria-prima:** grafos FNN; UPFD via PyG; features `[is_root, grau_norm, pos]`.
- **Produto:** `resultados.csv` em `Execution/results/fase4_benchmarks/topologia_sem_texto/`.
- **Porquê:** **este é o experimento central do TCC**. Se o F1 sem nenhuma feature textual ainda fica próximo do F1 com BERT, fica matematicamente provado que o modelo **nunca precisou ler a notícia** — é o "smoking gun" da vulnerabilidade topológica.
- **Conexão com Fase 9 (script 30):** o F1m=0.814 agregado do SAGE estrutural em UPFD-GossipCop **não vem uniformemente** da topologia. O script 30 mostra que ele cresce de 0.795 em depth=2 (estrela rasa) até 0.815 em depth=3 e atinge picos em depth=5+. Ou seja, a parte do F1 que **realmente** depende de estrutura multi-hop está concentrada nos 30% dos grafos com cascata profunda; nos outros 70% o sinal é equivalente ao do RF tabular (num_nodes). Isso refina o "smoking gun": é vulnerabilidade topológica **multi-hop**, não só "volume".

### [15_analise_estrutural.py](15_analise_estrutural.py)
- **Objetivo:** análise descritiva fake vs real em métricas topológicas (com Cohen's d).
- **Matéria-prima:** grafos FNN e UPFD.
- **Produto:** `estatisticas.csv`, figuras em `Execution/results/figuras_tcc/analise_estrutural/`.
- **Porquê:** o 14 mostra que o modelo **consegue** explorar o atalho. Esse mostra que o atalho **existe nos dados**, antes de qualquer modelo. Atribui a culpa ao dataset (coleta enviesada), não ao GNN — é uma defesa metodológica importante.

### [16_gnn_explainer_upfd.py](16_gnn_explainer_upfd.py)
- **Objetivo:** GNNExplainer nos modelos SAGE estruturais — quais arestas/nós o modelo olha?
- **Matéria-prima:** modelos SAGE; UPFD-Goss/Polit; 20 amostras estratificadas.
- **Produto:** PNGs/HTMLs e `hop_importance.csv` em `Execution/results/figuras_tcc/gnnexplainer/`.
- **Porquê:** a triangulação fica completa quando 14 + 15 + 16 concordam: o sinal está nos dados, o modelo consegue extraí-lo, **e** ao explicar a decisão ele aponta para arestas estruturais. Sem o 16, alguém poderia argumentar que o modelo está olhando "outra coisa correlacionada" — esse script fecha a porta.

---

## Fase 6 — Modelos finais e aplicação no Bluesky

### [17_persistir_modelos_finais.py](17_persistir_modelos_finais.py)
- **Objetivo:** treinar e salvar 3 modelos canônicos para inferência/API.
- **Matéria-prima:** FNN, UPFD-Goss; embeddings BERT.
- **Produto:** `logreg_bert_fnn.pkl`, `rf_struct_gossipcop.pkl`, `sage_struct_gossipcop.pth` em `Execution/weights/`.
- **Porquê:** três modelos cobrem três regimes — **textual puro** (referência teto), **estrutural leve** (RF interpretável, rápido para deploy), **estrutural pesado** (SAGE para GNNExplainer e estado da arte). Triagem por custo×interpretabilidade×accuracy.

### [18_analise_bluesky_crossfeed.py](18_analise_bluesky_crossfeed.py)
- **Objetivo:** análise estrutural por feed do Bluesky.
- **Matéria-prima:** posts e threads do Bluesky.
- **Produto:** estatísticas por feed em `Execution/results/figuras_tcc/bluesky_crossfeed/`.
- **Porquê:** Bluesky não tem ground truth de fake — comparar fake vs real é impossível. Mas dá para comparar **feeds** (Political Science vs Science vs aberto), que captura como cada comunidade propaga conteúdo. É a forma de extrair sinal estrutural sem labels.

### [19_aplicar_modelo_bluesky.py](19_aplicar_modelo_bluesky.py)
- **Objetivo:** aplicar RF estrutural a posts do Bluesky e ranquear "mais fake-like".
- **Matéria-prima:** posts Bluesky; modelo RF.
- **Produto:** ranking em `Execution/results/figuras_tcc/bluesky_inferencia/`.
- **Porquê:** demonstração qualitativa de transferência out-of-domain: modelo treinado em GossipCop classificando posts de outra plataforma, sem ajuste. Os top-suspeitos viram exemplos concretos para a defesa do TCC.

### [22_concordancia_bluesky.py](22_concordancia_bluesky.py)
- **Objetivo:** concordância LogReg-BERT × RF-estrutural em 5k posts.
- **Matéria-prima:** posts Bluesky; modelos do 17.
- **Produto:** matriz e cross-tabs em `Execution/results/figuras_tcc/concordancia_bluesky/`.
- **Porquê:** sem ground truth, a métrica viável é **consistência**. Se os dois classificadores concordam, é provável que ambos estão certos; quando discordam, é onde o problema está mal-condicionado. Estratificar por feed/tamanho/idioma localiza onde cada um falha.

### [23_visualizar_threads_bluesky.py](23_visualizar_threads_bluesky.py)
- **Objetivo:** 6 visualizações de threads reais (PNG + HTML).
- **Matéria-prima:** `threads.txt.gz`; modelo RF.
- **Produto:** PNGs/HTMLs em `Execution/results/figuras_tcc/threads_bluesky/`.
- **Porquê:** defesa de TCC precisa de exemplos concretos, não só números agregados. Threads visualizadas tornam tangível para a banca o que "estrutura de propagação" significa, e o HTML interativo permite o orientador explorar sem rodar código.

---

## Fase 7 — Comparações finais e perguntas de pesquisa

### [20_textual_vs_topologico.py](20_textual_vs_topologico.py)
- **Objetivo:** quando textual e topológico concordam? Quem acerta nas discordâncias?
- **Matéria-prima:** UPFD-Goss; posts Bluesky; modelos do 17.
- **Produto:** CSVs e `casos_discordancia.txt` em `Execution/results/figuras_tcc/textual_vs_topologico/`.
- **Porquê:** UPFD-Goss tem labels (mede acerto absoluto), Bluesky não (mede só consistência). Combinar os dois diagnostica em qual regime cada classificador é confiável — base para a "regra dual" da Fase 8.

### [21_bloco_vs_mini.py](21_bloco_vs_mini.py)
- **Objetivo:** 1 modelo treinado em 3 datasets concatenados vs 3 especialistas.
- **Matéria-prima:** FNN, UPFD-Polit, UPFD-Goss; RF.
- **Produto:** `resumo.csv` em `Execution/results/figuras_tcc/bloco_vs_mini/`.
- **Porquê:** decisão prática de deploy. Se o modelo "bloco" (1 só) generaliza igual ou melhor, simplifica a pipeline; se especialistas vencem, é evidência de que cada dataset tem distribuição própria — outro indício de overfitting topológico por dataset.

### [26_rq3_multilingual.py](26_rq3_multilingual.py)
- **Objetivo:** distribuição de score estrutural em PT/DE/EN.
- **Matéria-prima:** posts Bluesky monolingues; RF.
- **Produto:** comparações em `Execution/results/figuras_tcc/rq3_multilingual/`.
- **Porquê:** se o classificador é **realmente** estrutural (não textual disfarçado), ele tem que ser invariante a idioma. Distribuições iguais entre PT/DE/EN validam o claim de "language-agnostic"; distribuições diferentes refutam. É a pergunta de pesquisa RQ3.

---

## Fase 8 — Consolidação para o TCC

### [24_consolidar_tcc.py](24_consolidar_tcc.py)
- **Objetivo:** consolidar todos os CSVs em tabelas LaTeX prontas + figuras + índice.
- **Matéria-prima:** todos os CSVs das fases 2–4.
- **Produto:** `*.tex` em `final/tabelas/`, figuras em `final/figuras/`, `INDICE.md`.
- **Porquê:** colar tabela manual do CSV no LaTeX é fonte garantida de drift entre números reportados e números reais (já aconteceu em iterações anteriores). Consolidação automatizada garante que o TCC sempre reflete o último experimento rodado.

### [27_figuras_comparativas.py](27_figuras_comparativas.py)
- **Objetivo:** gerar F16–F24 a partir dos CSVs já existentes (sem experimento novo).
- **Matéria-prima:** CSVs das fases anteriores.
- **Produto:** figuras adicionais em `Execution/results/figuras_tcc/comparativas/`.
- **Porquê:** foi pedido posterior do orientador. Manter separado de 24 evita misturar a pipeline original com adições incrementais. A próxima onda de revisões virou a Fase 9 (scripts 28–30) — análises novas, não só figuras, então justificou nova fase em vez de mais arquivos consolidadores.

### [25_publicar_huggingface.py](25_publicar_huggingface.py)
- **Objetivo:** publicar os 3 modelos no HF Hub com model card.
- **Matéria-prima:** pesos do 17; token HF.
- **Produto:** repo público no HF Hub.
- **Porquê:** reprodutibilidade científica + visibilidade. O TCC fica mais forte se a banca puder baixar os pesos e reproduzir; HF é o padrão de facto para isso e o link no model card vira referência citável.

---

## Fase 9 — Análise estratificada do erro (rodada complementar)

> Origem: sugestão revisora — quantificar onde, dentro da distribuição estrutural,
> o modelo erra. Hoje o TCC reporta F1 agregado + Cohen's d agregado, mas **não
> cruza as duas coisas**: não diz "o modelo erra mais nos grafos do tercil
> superior de tamanho". Esta fase fecha esse gap e prepara terreno para análises
> mais finas (outliers, cascatas profundas) sem retrabalho.
>
> **Achados principais (UPFD-GossipCop test, N=3826):**
> - Inversão distributiva confirmada: SAGE acerta **0 de 31** fakes contidos e **26 de 60** reais virais (abaixo de chance).
> - Gap SAGE−RF cresce com profundidade: +0.060 agregado → +0.134 em depth≥3 → +0.213 em depth≥5 (todos significantes via bootstrap, CI não cruza zero).
> - Distribuições estruturais são heavy-tail **mais agressivas que lognormal** (KS rejeita o fit), justificando uso de percentil empírico no script 29.
> - Cascata "estrela plana" não é universal: 30% dos grafos UPFD-GossipCop têm depth≥3, contradizendo parcialmente a §6.3.a do TCC (que se aplica só ao FNN construído por nós, não ao UPFD oficial).

### [28_distribuicao_e_estratificacao.py](28_distribuicao_e_estratificacao.py)
- **Objetivo:** caracterizar a distribuição dos grafos do test (fit lognormal) e medir F1/acerto do RF e do SAGE estratificado por tercil de cada métrica estrutural; sumarizar profundidade para preparar análise de cascatas.
- **Matéria-prima:** UPFD-GossipCop test (3826 grafos); modelos persistidos `rf_struct_gossipcop.pkl` e `sage_struct_gossipcop.pth` do script 17.
- **Produto:** `Execution/results/figuras_tcc/estratificacao/` com `por_grafo.csv` (insumo dos próximos scripts), `bins_f1.csv`, `distribuicao_fit.csv`, 4 figuras e `relatorio.txt` com decisões para 29 e 30.
- **Porquê:** três funções:
  1. **LOWESS** (smoother por janela móvel) de P(acerto) vs `num_nodes`/`depth`/`branching` mostra continuamente onde o modelo degrada — versão contínua dos tercis, sem cherry-picking de cortes.
  2. **Fit lognormal** com teste KS justifica usar percentil empírico (não z-score) na definição de outliers do script 29 — cascatas em redes sociais são heavy-tailed, não normais.
  3. **`por_grafo.csv`** com predição+acerto por grafo evita re-rodar modelos nos scripts 29 e 30.
- **Caveat:** rodado **só em UPFD-GossipCop test** porque é onde temos modelo persistido. PolitiFact não tem (script 17 não treinou) e replicar exigiria re-treino — reportar como limitação.

### [29_outliers.py](29_outliers.py)
- **Objetivo:** quantificar a degradação do modelo em 6 subgrupos extremos derivados *post-hoc* das distribuições reveladas pelo 28.
- **Matéria-prima:** `por_grafo.csv` produzido pelo 28 (não recarrega modelos).
- **Produto:** `Execution/results/figuras_tcc/outliers/` com `subgrupos.csv`, `exemplos.csv` (5 grafos representativos por subgrupo), 2 figuras e `relatorio.txt` com diagnóstico automático de inversão distributiva.
- **Porquê:** a §4.6 do TCC descreve narrativamente uma "inversão distributiva" (modelo erra sistematicamente em real-viral e fake-contido). Este script confirma numericamente: SAGE acc = **0.000 em fake_contido** (N=31) e **0.433 em real_viral** (N=60, abaixo de chance). Também mostra que SAGE bate RF em **+28pp em cauda_alta_n** e **+23pp em deep_extreme** — sustenta empiricamente a regra dual 0.8/0.2 da §3.9 e o trecho de limitações da §6.3.

### [30_cascatas_profundas.py](30_cascatas_profundas.py)
- **Objetivo:** medir o ganho do SAGE sobre o RF tabular em subsets de profundidade crescente; testar se o gap é desproporcional em cascatas multi-hop.
- **Matéria-prima:** `por_grafo.csv` do 28; UPFD-GossipCop test (recarregado só para visualização pyvis).
- **Produto:** `Execution/results/figuras_tcc/cascatas_profundas/` com `por_profundidade.csv`, `gap_subsets.csv`, `bootstrap_diff.csv` (CI 95% via 2000 reamostragens), 2 figuras e 3 visualizações HTML de cascatas profundas (depth ≥ 8).
- **Porquê:** se o SAGE-RF gap fosse uniforme entre profundidades, o ganho do GNN seria só "número de nós" (que o RF já captura). Como o gap **3.5×** em depth≥5 (+0.213) vs agregado (+0.060), com CI bootstrap que não cruza zero, fica demonstrado que o GNN agrega valor **onde há multi-hop pra propagar**, não em estrelas rasas. Achado central que conecta §4.4 (topologia sem texto) à §4.7 (GNNExplainer assimétrico).
- **Observação metodológica:** o achado contradiz parcialmente a §6.3.a do TCC ("FNN com cascatas estrela"). UPFD-GossipCop oficial **não** é estrela plana — só 70% dos grafos têm `depth_max=2`; os outros 30% têm cascatas multi-hop reais. A crítica de "estrela plana" se aplica ao FakeNewsNet construído pelo script 00, não ao UPFD oficial. Considerar refinar a redação.

---

## Diagrama do fluxo

```
Dados brutos                  Construção                 Modelos                  Análises                  Saídas TCC
───────────                   ──────────                 ───────                  ────────                  ──────────
FakeNewsNet CSV    ─────►     00 ─────►  fakenewsnet_*/.pt
                                                    │
HF Bluesky        ─►  01 ─►   02 ─────►  bluesky_*/.pt   ─┐
                                                          │
UPFD raw (PyG)    ──────────────────────► UPFD oficial    │
                                                          │
                                                          ▼
                              gerar_folds ─► folds_fnn.pt
                                                          │
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                                 03/05 (Bluesky)             07/13 (UPFD oficial)
                                 04/06 (compara)             10/11/12 (baselines + confound)
                                                                         │
                                                                         ▼
                                                              09 (significância)
                                                              14 (topologia sem texto)  ◄── núcleo do TCC
                                                              15/18 (análise estrutural)
                                                              16 (GNNExplainer)
                                                              08 (cross-dataset)
                                                                         │
                                                                         ▼
                                                              17 ─► weights/*.pkl/.pth
                                                                         │
                                          ┌──────────────────────────────┼───────────────┐
                                          ▼                              ▼               ▼
                                 19/22/23 (Bluesky)             20/21 (concord.)   26 (multilíngue)
                                                                         │
                                                                         ▼
                                                              24 + 27 ─► tabelas + figuras TCC
                                                              25 ─► HuggingFace público

Fase 9 (estratificação) — consome pesos do 17, independente da consolidação:

                              17 (weights) ───────────► 28 ─► por_grafo.csv
                                                              │
                                                   ┌──────────┴──────────┐
                                                   ▼                     ▼
                                            29 (outliers)        30 (cascatas profundas)
```
