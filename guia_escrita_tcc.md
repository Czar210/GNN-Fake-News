# Guia operacional de escrita — TCC GNN Fake News

**Status:** complemento do `estrutura_tcc_v2.md` (outline conceitual).
**Para quem:** vc escrevendo sozinho. Não tem prosa pronta — tem **scaffolds**: por seção, quais figuras/tabelas usar, onde colocar no fluxo, números canônicos, citações `.bib` exatas, e pegadinhas a evitar.

**Como usar este guia:**
1. Abra `estrutura_tcc_v2.md` na lateral pra ver a *justificativa* de cada seção.
2. Use este guia pra escrever a *prosa*: cada bloco abaixo dá os ingredientes (figs, números, citações, marcos argumentativos).
3. **Nunca duplique uma figura/tabela** — cada artefato aparece uma vez no corpo, no capítulo indicado.
4. Quando ver `[FIG:...]` ou `[TAB:...]`, é onde inserir `\ref{fig:xxx}` no LaTeX e a figura/tabela correspondente.

**Localização dos artefatos:**
- Tabelas LaTeX prontas: `Execution/results/figuras_tcc/final/tabelas/T*.tex`
- Figuras: `Execution/results/figuras_tcc/final/figuras/F*.png` (e pastas `F12_*/`, `F13_*/`)
- Índice mestre: `Execution/results/figuras_tcc/final/INDICE.md`

**Convenção de citações:**
- `\cite{kipf2017gcn}` → key do `Material/GNN_TCC_atualizado/Referencias.bib`
- Sempre que citar número específico (F1, d, etc.), referencie a fonte do dado entre parênteses no parágrafo: "(script 14, `topologia_sem_texto/resultados.csv`)".

---

# Capítulo 1 — Introdução

**Tamanho-alvo:** 4-6 páginas.
**Ordem de escrita:** ÚLTIMO (escrever depois que todos os outros estiverem prontos — vc só sabe contar a história depois de saber qual história contar).

## §1.1 Contextualização (1 página)
- **Argumento central:** desinformação é problema epistêmico real; NLP isolada perde robustez.
- **Citações:** `\cite{ieee_fakenewslandscape, pmc_overviewfakenews, scielo_posttruth}` — uma cada.
- **NÃO faça:** discurso longo sobre "era da pós-verdade". Banca técnica desliga. Máximo 2 parágrafos sobre contexto.
- **Pegadinha:** evite dizer "LLMs causaram a crise" — a desinformação precede LLMs. Diga "LLMs aceleram produção de texto convincente, comprometendo abordagens puramente textuais" — citar `\cite{arxiv_llmfakenews}`.

## §1.2 Motivação (0.5 página)
- **Argumento:** topologia de propagação como sinal *complementar*, não substituto.
- **Citação obrigatória:** `\cite{monti2019fakenews}` (Geometric Deep Learning para fake news, fundador da linha).
- **Não use** "robusto-a-idioma" → escreva: "o classificador estrutural não processa o texto, portanto independe do idioma no qual o post foi escrito".

## §1.3 Perguntas de pesquisa (0.5 página)
- Liste RQ1, RQ2, RQ3 **em formato de pergunta direta**, numerada, em parágrafos curtos.
- Texto exato sugerido pra RQ3: "É viável detectar fake news *apenas* pela topologia de propagação, em cenários onde o texto não é exploitável (multilíngue sem encoder específico, texto deletado/restrito, ou linguagem informal)?"

## §1.4 Contribuições (1 página)
Lista numerada de 4 contribuições. Para cada, 1-2 frases. **Use as exatas do outline §1.4** (já passaram por revisão adversarial). Adicione no final:
> "Os pesos persistidos estão publicados no Hugging Face Hub sob licença CC-BY-4.0 (ver Apêndice C)."

## §1.5 Estrutura do trabalho (0.5 página)
Mapa: "O Capítulo 2 apresenta... O Capítulo 3 detalha... etc." Um parágrafo por capítulo, 2 frases cada. **Sem figuras/tabelas neste capítulo.**

---

# Capítulo 2 — Fundamentação Teórica

**Tamanho-alvo:** 12-18 páginas.
**Ordem de escrita:** PENÚLTIMO (depois de Resultados — você sabe melhor o que precisa fundamentar).

## §2.1 Detecção de fake news: panorama (1.5 página)
- Sub-tópicos: NLP clássico, contexto social, propagação como sinal complementar.
- **Citações:** `\cite{pmc_overviewfakenews, ieee_fakenewslandscape, mdpi_cnn_llm_nlp}`.
- Tese desta seção: "abordagens isoladas (NLP-only ou estrutura-only) têm limites; trabalhos recentes movem para hibridização".

## §2.2 Grafos e redes sociais (1 página)
- Definições: grafo dirigido, árvore de propagação, raiz, profundidade, branching factor.
- **Sem citação obrigatória** — use textbook (cite Newman 2003 SIAM Review se quiser: `\cite{newman2003structure}`).
- **Tópico-âncora:** propagação numa rede social como cascata enraizada (post original = raiz; reposts/replies = filhos diretos ou indiretos).

## §2.3 Embeddings textuais (1 página)
- Sub-tópicos: word embeddings → BERT → Sentence-BERT.
- **Citações obrigatórias:** `\cite{devlin2019bert, reimers2019sbert}`.
- **Modelo específico usado:** `paraphrase-multilingual-mpnet-base-v2` (768d). Citar como nota de rodapé: "modelo da família \cite{reimers2019sbert} treinado em corpus paraphrase multilíngue".
- **Pegadinha:** Sentence-BERT ≠ BERT vanilla. Não diga "usamos BERT" — diga "usamos Sentence-BERT".

## §2.4 Redes Neurais de Grafos (3-4 páginas)
- Subseções: §2.4.1 GCN, §2.4.2 GAT, §2.4.3 GraphSAGE.
- **Citações obrigatórias:** `\cite{kipf2017gcn, velickovic2018gat, hamilton2017graphsage}`.
- **Para cada arquitetura:** equação central, intuição, complexidade, vantagem operacional.
- GCN: regra de propagação $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})$.
- GAT: introduz atenção por aresta — escreva $\alpha_{ij} = \text{softmax}(\text{LeakyReLU}(\mathbf{a}^T[Wh_i \| Wh_j]))$.
- SAGE: agregação inductiva — sample + aggregate.
- **Cite framework:** "Implementamos com PyTorch Geometric \cite{fey2019pyg}".
- **Pooling final:** mencionar `global_mean_pool` como decisão (alternativas: `add_pool`, `max_pool`).

## §2.5 Degeneração GCN com features homogêneas (2 páginas)
**ESTA SEÇÃO É CRÍTICA — promessa explícita do outline. Conteúdo obrigatório:**

1. Estabelecer notação: matriz de features $X \in \mathbb{R}^{N \times d}$, adjacência normalizada $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$.
2. **Cenário degenerado:** suponha $X_i = X_j \, \forall i, j$ (todas as linhas iguais). Então $X = \mathbf{1}_N x^T$ onde $x \in \mathbb{R}^d$.
3. **Demonstração:** $H^{(1)} = \sigma(\tilde{A} \mathbf{1}_N x^T W) = \sigma((\tilde{A}\mathbf{1}_N) (xW)^T)$. O termo $\tilde{A}\mathbf{1}_N$ é vetor de "graus normalizados" — mesmo escalar pra cada nó (modulado por grau).
4. **Conclusão:** representação por nó $h_i^{(1)}$ é múltiplo escalar do mesmo $xW$, com fator dependente apenas do grau de $i$. Sem informação para discriminar entre nós da mesma vizinhança.
5. **Implicação prática:** justifica adicionar features posicionais (`is_root`, `grau_norm`, `pos`) que quebram a homogeneidade — desenvolvido em §2.6.

**Sem figura nesta seção.** Equações em ambiente `align*` ou `equation`.

## §2.6 Encodings posicionais (1 página)
- Definir as 3 features adicionadas: `is_root` (binária, raiz=1), `grau_norm` = $\deg(v) / \deg_{\max\_global}$, `pos` = ordem BFS / num_filhos.
- **Justificar:** quebram a degenerência de §2.5. Sem citação canônica nesta área (é construção nossa); pode citar `\cite{arxiv_topologicalfeatures}` como referência relacionada de "node-level topological features".

## §2.7 Explicabilidade em GNNs (1 página)
- GNNExplainer: máscara de arestas, máscara de features, problema de otimização (mutual information).
- **Citação obrigatória:** `\cite{ying2019gnnexplainer}`.
- Mencionar que vc usa `torch_geometric.explain.Explainer` com `algorithm=GNNExplainer(epochs=200)`.

## §2.8 Validação estatística (1 página)
- k-fold estratificado, teste $t$ pareado vs independente (justificar uso do pareado), Cohen's $d$.
- **Citação obrigatória:** `\cite{cohen1988statistical}`.
- **Tabela de interpretação de Cohen's $d$** (faça inline, não como `[TAB:]`):

| $|d|$ | Interpretação |
|---|---|
| < 0.2 | desprezível |
| 0.2 – 0.5 | pequeno |
| 0.5 – 0.8 | médio |
| ≥ 0.8 | grande |

## §2.9 Trabalhos relacionados (2-3 páginas)
- Subseções: §2.9.1 Surveys (`\cite{pmc_gnnsurveyfakenews, pmc_overviewfakenews}`), §2.9.2 Métodos GNN canônicos para fake news (`\cite{monti2019fakenews, dou2021upfd, bian2020bigcn, mdpi_bidirectionalgcn}`), §2.9.3 Cross-language (`\cite{acl_hemt, crosslanguagefakenews, pmc_multilangs_covid}`), §2.9.4 Trabalhos brasileiros (`\cite{ufc_fakewhatsapp, pucsp_gnnfakenews, usp_fakenews}`), §2.9.5 Homofilia/assortatividade em redes (`\cite{newman2003mixing, pei2020geomgcn}`).
- **Final do capítulo (1 parágrafo):** posicionar este trabalho. Texto sugerido:
  > "O presente trabalho não propõe uma nova arquitetura GNN. Sua contribuição é uma **análise sistemática + diagnóstico** sobre quando o paradigma topológico funciona, ancorando achados positivos e negativos em uma tese estrutural unificada. **Limitação assumida:** não rodamos BiGCN \cite{bian2020bigcn} ou GCNFN diretamente — comparação numérica com esses métodos é feita apenas via valores publicados (ver §6.3)."

**Sem figuras neste capítulo.** Toda visualização vai pra Resultados.

---

# Capítulo 3 — Metodologia

**Tamanho-alvo:** 8-12 páginas.
**Ordem de escrita:** PRIMEIRO (junto com Resultados). Mais denso em fatos verificáveis = menos espaço pra hand-waving.

## §3.1 Visão geral do pipeline (1 página)
- **Figura obrigatória:** diagrama em blocos: dados brutos → grafos → features → modelo → avaliação. **Vc precisa criar essa figura** (não temos pronta). Sugestão: use `draw.io` ou `tikz` direto no LaTeX. Salve como `Imagens/F0_pipeline.png`.
- Texto: parágrafo descrevendo cada bloco em 1-2 frases.

## §3.2 Datasets (2 páginas)
- **`[TAB:T0]` (criar):** tabela com #grafos, #classes, fonte, licença, tamanho. Datasets: FakeNewsNet PolitiFact, UPFD-PolitiFact, UPFD-GossipCop, Bluesky.
- **Citações:** `\cite{shu2020fakenewsnet, dou2021upfd, kleppmann2024atproto, quelle2024bluesky}`.
- Para cada dataset, 1 parágrafo: o que é, quem coletou, quantos grafos, qual feature dominante.
- **Para Bluesky:** declarar explicitamente: "snapshot ~6GB de 2024-2025; sem ground-truth fake/real; usado em modo de demonstração no Capítulo 5".

## §3.3 Construção de grafos FakeNewsNet (1.5 página)
- Pipeline do `00_construir_grafos_fakenewsnet.py`: download CSVs → BERT embedding dos títulos → construção do grafo (raiz=notícia, filhos=tweets propagadores) → adição de features posicionais → cache em `data/fakenewsnet_posfull/`.
- **3 variantes** geradas (com flag `--feature-variant`): `posfull` (768+3=771d), `posmin` (1d só is_root), `posgrau` (2d).

## §3.4 Folds compartilhados (0.5 página)
- `gerar_folds.py` cria 10 folds estratificados, `seed=42`, salvos em `folds_fnn.pt` e reusados em todos os experimentos pra permitir teste pareado.
- **Justifique:** "Folds compartilhados são pré-requisito para que `scipy.stats.ttest_rel` seja válido — modelos comparados precisam ter sido avaliados nos *mesmos* exemplos."

## §3.5 Arquiteturas GNN (1.5 página)
- 3 camadas + `global_mean_pool` + linear final, 2 classes.
- **`[TAB:hyperparams]` (criar manualmente):** tabela com hyperparâmetros por arq.
- Defaults: `hidden_channels=64`, `dropout=0.5`, `lr=0.001`, `weight_decay=5e-4`, `batch_size=32`, otimizador Adam, scheduler `ReduceLROnPlateau` (patience=5), early stopping com patience=7, max_epochs=30.

## §3.6 Baselines não-GNN (1 página)
- LogReg-BERT-root: feature = `x[0]` do grafo (BERT do título da raiz).
- RF-tabular: features = `[num_nodes, grau_root]`.
- **Justificar:** servem como teto inferior + diagnóstico de confound (se RF-tabular sozinho atinge F1 alto, GNN não está capturando nada além de tamanho).

## §3.7 Protocolo de avaliação (1 página)
- F1-macro como métrica primária (justificar: classes balanceadas no FNN/UPFD; evita inflação por classe majoritária).
- k-fold pareado + `scipy.stats.ttest_rel` para comparação de modelos.
- Cohen's $d$ para análise estrutural (§4.5).
- **Reprodutibilidade caveat:** "Resultados foram obtidos em CPU (Python 3.12.10, torch 2.10.0+cpu); seeds são fixadas mas determinismo não é bit-exact entre máquinas — variação típica na 3ª casa decimal."

## §3.8 Análise de explicabilidade (0.5 página)
- GNNExplainer com `epochs=200`, `edge_mask_type="object"`, `node_mask_type=None`, modo `multiclass_classification` em nível de grafo.
- 20 amostras estratificadas em UPFD-GossipCop; análise de hop-distance (1-hop, 2-hop, 3+).

## §3.9 Classificador dual e regra de combinação **post-hoc** (0.5 página)
- **CRÍTICO:** introduzir como heurística post-hoc, não componente metodológico a priori.
- Texto sugerido: "Para a aplicação prática (Capítulo 5), combinamos o classificador textual (LogReg-BERT) e o topológico (RF estrutural) por uma regra de combinação. **Essa regra não é uma decisão metodológica a priori; é uma heurística derivada após observar os resultados de §4.6.** Sua justificativa empírica e a discussão das alternativas estão naquela seção."
- **Não detalhar a regra aqui.** Só anunciar. Detalhamento e justificativa em §4.6.

## §3.10 Reprodutibilidade (0.5 página)
- requirements.txt pinado, seed=42 universal, README.md como fonte autoritativa.
- Listar caveat de bit-exact (CUDA/cuDNN não-determinismo). Já está no README — só citar.

**Sem figuras neste capítulo, exceto F0 (pipeline) que vc cria.**

---

# Capítulo 4 — Resultados Experimentais

**Tamanho-alvo:** 15-22 páginas.
**Ordem de escrita:** SEGUNDO (depois de Metodologia). Mais denso do TCC.

**Lead obrigatório (1 parágrafo):** "Cada subseção responde uma RQ. Resultados negativos (ex.: PolitiFact em §4.4) são reportados com a mesma extensão dos positivos, em conformidade com a tese central do trabalho."

## §4.1 Baselines textuais (RQ1, contexto) [1.5 página]
- **Use:** `[TAB:T1]` `T1_baseline_textual_fnn.tex` no início.
- **Achado canônico:** LogReg sobre BERT da raiz no FNN, F1=0.859±0.043 (10 folds, k-fold).
- **Implicação a escrever:** "O texto da raiz, isoladamente, já carrega quase todo o sinal classificatório no FakeNewsNet. Modelos GNN só justificam-se se agregarem F1 *acima* desse baseline."
- **Fonte:** script 10, `Execution/results/fase2_baselines/baseline_textual/resultados.csv`.

## §4.2 Diagnóstico de confound topológico (RQ1) [1 página]
- **Sem figura/tabela específica** (o achado é simples: F1(num_nodes só)=0.523 — gate=0.65 não dispara).
- Texto: "Random Forest treinado apenas com `num_nodes` atinge F1=0.523 — abaixo do gate de 0.65 que indicaria confound dominante por tamanho. Concluímos que o tamanho do grafo *contribui*, mas não *explica* o sinal capturado pelos modelos subsequentes."
- **Fonte:** script 11, `fase2_baselines/confound_diagnostico/resultados.csv`.

## §4.3 Cross-dataset UPFD: nosso vs literatura (RQ1) [1.5 página]
- **Use:** `[TAB:T3]` `T3_cross_dataset_upfd.tex` (nossos números) seguida de `[TAB:T10]` `T10_upfd_vs_publicado.tex` (vs Dou et al.).
- **Achado:** PolitiFact ~0.78–0.82 (vs 0.846 publicado); GossipCop ~0.94–0.95 (vs 0.97).
- **Parágrafo obrigatório de qualificação (3 itens):**
  > "Esta comparação é **aproximada, não replicação**. Três diferenças deliberadas: (i) usamos feature `content` (310d) ao invés de `bert` (768d) — restrição de RAM em CPU/Windows ao densificar a matriz esparsa do GossipCop; (ii) hiperparâmetros próprios (lr=0.001, hidden=64, 3 camadas), pois a configuração exata de \cite{dou2021upfd} não é totalmente especificada no paper; (iii) reportamos F1-macro, eles reportam accuracy. O propósito da comparação é verificar que nosso pipeline está *no envelope* da literatura, não fazer ranking competitivo."
- **Fonte:** script 13.

## §4.4 Topologia sem texto: o achado central (RQ3, parte a) [3 páginas — seção mais importante]
- **Use:** `[TAB:T4]` `T4_topologia_sem_texto.tex` no início + **`[FIG:F18]` `F18_heatmap_topo_sem_texto.png`** logo abaixo da tabela.
- **Achado canônico:** SAGE-GossipCop com `[is_root, grau_norm]` apenas: **F1m=0.810 (std=0.0024, 10 seeds)**. PolitiFact: F1≈0.55. FNN: F1≈0.55-0.60.
- **Parágrafo obrigatório sobre o std=0.002** (texto pode adaptar do outline §4.4):
  > "A baixa variância (std=0.0024 sobre 10 seeds) merece justificativa. Auditoria do `script 14` confirma que (a) `torch.manual_seed(seed)` e `np.random.seed(seed)` são chamados antes de cada run; (b) `DataLoader` com `shuffle=True` consome do gerador semeado; (c) o construtor da `SAGEClassifier` recebe `seed=seed`. A baixa variância é consequência da combinação de (i) features de entrada de apenas 2 dimensões — pouco a aprender que dependa de inicialização — (ii) sinal estrutural muito forte no GossipCop (Cohen's d=1.53, ver §4.5) que conduz o otimizador ao mesmo ótimo independentemente da seed, (iii) split train/val/test fixo (UPFD oficial). Como controle negativo: a *mesma* arquitetura SAGE no UPFD-PolitiFact (Cohen's d ≈ 0) produz std de ~0.05, ordens de grandeza maior — confirmando que std baixo é função do dataset, não de bug de seed."
- **Fonte:** script 14, `fase4_benchmarks/topologia_sem_texto/resultados.csv`.

## §4.4-bis Consistência de score topológico entre idiomas (RQ3, parte b) [2 páginas]
- **Lead obrigatório (1 parágrafo):**
  > "Este experimento **não mede detecção em PT/DE** — Bluesky não tem ground-truth. Mede **consistência da distribuição de score** que o classificador estrutural produz para posts em diferentes idiomas. Consistência é condição *necessária mas não suficiente* para detecção multilíngue."
- **Use:** `[TAB:T11]` `T11_rq3_multilingual.tex` + **`[FIG:F14]` `F14_rq3_multilingual_dist.png`** (histograma 3 idiomas) + opcionalmente `[FIG:F15]` `F15_rq3_multilingual_qq.png` (Q-Q).
- **Achados canônicos:**
  - PT vs EN: d=−0.057 (desprezível). KS e MW rejeitam H0 (p<10⁻⁵) por causa de n=132k+2.7k, **mas o effect size está bem abaixo do limiar de Cohen — diferença estatisticamente detectável mas praticamente irrelevante**.
  - DE vs EN: d=−0.468 (efeito médio). Posts em DE têm score sistematicamente menor (média 0.26 vs 0.40 do EN).
  - PT vs DE: d=+0.739 (médio-grande).
- **Parágrafo de hipóteses alternativas (obrigatório):**
  > "Atribuímos a divergência DE vs EN a padrões de engajamento da comunidade germanófona no Bluesky (uso mais discreto da plataforma, menos repost/reply). **Hipóteses alternativas igualmente válidas, não descartáveis sem experimento adicional:** (1) shift de domínio — classificador foi treinado em GossipCop (celebridade/entretenimento), posts DE no Bluesky podem ser mais políticos/profissionais; (2) viés amostral — usuários DE podem constituir subgrupo demográfico específico; (3) desbalanço de cobertura — n_PT (2.7k) vs n_DE (9.7k) vs n_EN (132k), Cohen's d é robusto, mas cobertura de regimes raros não."
- **Resposta refinada à RQ3 (b) — frase final:**
  > "O classificador estrutural não processa texto, portanto seu mecanismo é arquiteturalmente independente do idioma. A consistência empírica dessa independência foi confirmada para PT vs EN; foi parcialmente refutada para DE vs EN, com causa não isolada entre idioma, comunidade, domínio temático e amostragem."
- **Fonte:** script 26, `figuras_tcc/rq3_multilingual/`.

## §4.5 Por que GossipCop funciona e PolitiFact não? (RQ2) [2 páginas]
- **Use:** `[TAB:T5]` `T5_cohens_d_estrutural.tex` no início + **`[FIG:F17]` `F17_forest_plot_cohens_d.png`** (forest plot) como figura principal + opcionalmente `[FIG:F1]` `F1_separabilidade.png` ou `[FIG:F2]` `F2_distribuicoes_estruturais.png`.
- **Achado:** GossipCop branching factor d=+1.53 (efeito grande); PolitiFact |d|<0.2 em todas (desprezível).
- **Parágrafo de fundamentação teórica (obrigatório):**
  > "O resultado é coerente com a literatura de homofilia e assortatividade em redes. Newman \cite{newman2003mixing} estabelece que redes onde nós da mesma classe se conectam preferencialmente exibem assortatividade alta; Pei et al. \cite{pei2020geomgcn} demonstram que GNNs convolucionais dependem de assortatividade por classe para serem efetivas. Em UPFD-GossipCop, fake e real são *estruturalmente assortativos* — cada classe forma cascatas com perfis de branching distintos. Em UPFD-PolitiFact, não. Isso explica *por que* a mesma arquitetura SAGE rende F1=0.81 em um dataset e F1≈0.55 no outro, sem precisar postular falha de modelo."
- **Tese central a reforçar:** "Modelos topológicos requerem diferença estrutural entre classes da ordem de magnitude de um efeito médio de Cohen ($|d| \geq 0.5$). **Esta tese é falsificável:** um dataset com $0.2 < d < 0.5$ e GNN performante a refutaria."
- **Fonte:** script 15.

## §4.6 Concordância textual × topológico (RQ1) [2 páginas]
- **Use:** `[TAB:T6]` `T6_concordancia_gossipcop.tex` + **`[FIG:F3]` `F3_textual_vs_topo_scatter.png`** (scatter LogReg vs SAGE em UPFD-GossipCop).
- **Achados:**
  - Quando concordam: F1=0.97.
  - Quando discordam: textual F1=0.91, topológico F1=0.08.
- **Parágrafo obrigatório sobre F1=0.08 (inversão distributiva):**
  > "F1=0.08 é substancialmente abaixo de chance (~0.50 em problema balanceado) e mesmo de predição puramente majoritária (~0.33). No subconjunto de discordância, o classificador topológico está **sistematicamente invertido**, não apenas errado. A explicação é uma **inversão distributiva**: a relação estrutura→classe está aproximadamente espelhada nesse subconjunto em relação ao train set. O RF aprendeu 'fake = grau_root alto' no GossipCop (Cohen's d=+1.53 confirma isso no agregado); mas as predições onde discorda do textual são justamente (i) *notícias reais que viralizaram fortemente* (grau_root alto → RF prevê fake, errando) e (ii) *fake que tiveram propagação contida* (grau_root baixo → RF prevê real, errando). Esse comportamento *justifica* a assimetria 0.8/0.2 da regra dual descrita a seguir, não a contradiz: o topológico é confiável apenas quando concorda com o textual."
- **Apresentar a regra dual (post-hoc):**
  - Quando $\text{pred}_T = \text{pred}_K$: $\text{score}_{\text{final}} = 0.5 \cdot s_T + 0.5 \cdot s_K$.
  - Quando $\text{pred}_T \neq \text{pred}_K$: $\text{score}_{\text{final}} = 0.8 \cdot s_T + 0.2 \cdot s_K$.
- **Inserir aqui** `[FIG:F24]` `F24_fluxograma_regra_dual.png` (diagrama da regra) — útil pra visualização.
- **Fonte:** script 20, `figuras_tcc/textual_vs_topologico/`.

## §4.7 GNNExplainer: o que o SAGE aprendeu (RQ1) [2 páginas]
- **Use:** **`[FIG:F19]` `F19_hop_importance.png`** (bar chart de hop1/2/3+ por estrato) — figura principal.
- Opcionalmente, escolher 2-3 PNGs de `F12_gnnexplainer_gossipcop/` pra ilustrar (1 FAKE-grande + 1 REAL-grande, lado a lado).
- **Achado central (assimetria por classe):**
  - FAKE-pequena/grande (N=5+5): hop1 ≈ 0.97-1.00 em todas.
  - REAL-grande (N=5): hop1 ≈ 0.50-0.62.
  - REAL-pequena (N=5): sem padrão consistente; reportar como observação preliminar (N insuficiente).
- **Quantificação +6.1pp:**
  > "RF estrutural treinado em UPFD-GossipCop train+val: F1=0.753 no test (3826 grafos, ver `metadata.json`). SAGE estrutural mesmo split: F1=0.814. Diferença de 6.1pp absolutos atribuível ao GNN — concentrada na classe REAL, conforme análise de hop importance acima."
- **Implicação para Cap 5:** "Concentração de massa em hop1 pode ser exposta na interface como proxy de confiança do modelo."
- **Fonte:** script 16, `figuras_tcc/gnnexplainer/{gossipcop,politifact}/`.

## §4.8 Painel mestre (figura única, fechamento do capítulo) [0.5 página]
- **Use:** **`[FIG:F16]` `F16_painel_f1_mestre.png`** — único conteúdo da subseção.
- Texto: "A Figura ... consolida visualmente os três tipos de modelo (textual / topológico puro / com texto) nos três datasets. O contraste GossipCop vs PolitiFact é manifesto: topologia pura aproxima-se da textual no primeiro; despenca para próximo de chance no segundo."

---

# Capítulo 5 — Aplicação: Bluesky e Ferramenta Web

**Tamanho-alvo:** 8-12 páginas.
**Ordem de escrita:** TERCEIRO (depois de 3 e 4).

**Lead obrigatório:** "Este capítulo é demonstração, não validação. O dataset Bluesky utilizado não possui rótulos fake/real; portanto não realizamos claims supervisionados. Toda a validação quantitativa do trabalho está no Capítulo 4."

## §5.1 Dataset Bluesky (1 página)
- 168.463 posts em 11 feeds temáticos (Blacksky, News, Science, Political Science, etc).
- Coletado via AT Protocol \cite{kleppmann2024atproto}; snapshot acadêmico de 2024-2025 \cite{quelle2024bluesky}.
- **Sem figura.**

## §5.2 Análise cross-feed (1 página)
- **Use:** `[FIG:F5]` `F5_bluesky_crossfeed_distrib.png` + `[FIG:F6]` `F6_bluesky_crossfeed_pares.png`.
- Achado: distribuições de likes/reposts/replies variam fortemente por feed.
- **Fonte:** script 18.

## §5.3 Aplicação dos modelos persistidos (1.5 página)
- **Use:** `[TAB:T8]` `T8_bluesky_inferencia_feeds.tex` + `[FIG:F7]` `F7_bluesky_scores_por_feed.png`.
- RF estrutural aplicado em 168k posts; distribuição de "score fake-like" por feed.
- **Limitação explícita (1 parágrafo):**
  > "O 'score' produzido NÃO é probabilidade calibrada. É a saída `predict_proba` do RF treinado em UPFD-GossipCop, aplicada a um domínio diferente (Bluesky). Sua interpretação correta é como *proxy ordinal* — útil para ranking relativo entre posts, não para decisão absoluta. Calibração apropriada (ex.: Platt scaling, isotônica) requer ground-truth no domínio alvo, ausente neste dataset."
- **Fonte:** script 19.

## §5.4 Concordância textual × topológico no Bluesky (2 páginas)
- **Use (esta é a seção mais densa de figuras):**
  - **`[FIG:F20]` `F20_bluesky_crosstab.png`** (cross-tab agregado) — figura principal
  - `[FIG:F8]` `F8_bluesky_matriz_confusao.png`
  - `[FIG:F9]` `F9_bluesky_agreement_size.png`
  - `[FIG:F10]` `F10_bluesky_agreement_feed.png`
  - `[FIG:F11]` `F11_bluesky_agreement_textlen.png`
  - `[FIG:F23]` `F23_bluesky_modelo_x_feed.png` (heatmap)
  - `[TAB:T9]` `T9_bluesky_concord_size.tex`
- **Achados a destacar:**
  - Political Science = 88.7% agreement (alto).
  - Posts médios (2-20 nós) = 42-44% (zona cinzenta).
  - Posts virais (20-100 nós) = 74%.
- **Implicação:** "Os classificadores convergem em casos extremos (posts triviais ou virais) e divergem em zona cinzenta. Esse padrão é informação útil para a interface — posts em zona cinzenta podem ser sinalizados como 'requer revisão humana'."
- **Fonte:** script 22.

## §5.5 Independência de idioma vs sensibilidade de comunidade (1 página)
- **Use:** `[TAB:T11]` `T11_rq3_multilingual.tex` + `[FIG:F14]` + `[FIG:F15]`.
- **NOTA**: este conteúdo TAMBÉM aparece na §4.4-bis. Aqui em §5.5 vc faz **uma versão reduzida** focada na implicação prática para Bluesky:
  > "A análise multilingüe (detalhada em §4.4-bis) confirma que o classificador estrutural pode ser aplicado a posts em qualquer idioma com interpretação consistente *do ponto de vista do mecanismo*; mas o threshold absoluto deve ser calibrado por comunidade. Para a ferramenta web, isso implica que a barra de decisão (≥0.5 = fake) deve ser ajustada por feed na próxima iteração — fora do escopo deste TCC."

## §5.6 Visualização de threads reais (0.5 página)
- **Use:** **`[FIG:F13]` `F13_threads_bluesky/`** — escolher 3-4 das 6 threads, lado a lado, em uma única figura composta. Pasta tem PNGs e HTMLs; usar PNGs.
- Texto: "Visualizamos 6 threads reais de tamanhos variados, com score topológico sobreposto."

## §5.7 Arquitetura da ferramenta web (1.5 página)
- FastAPI + Next.js + 3 modelos persistidos.
- Endpoint `/api/result` retorna 3 scores + flag concordância.
- **Snippet de código** (em ambiente `lstlisting`): mostrar o `combinar_scores()` da API (8-10 linhas máximo).
- **Sem figura nova.** Pode opcionalmente colocar screenshot da interface (gere com browser quando subir API).

### §5.7.1 Decisão de produto: bloco único vs especialistas (0.5 página)
- **Use:** `[TAB:T7]` `T7_bloco_vs_mini.tex` + `[FIG:F4]` `F4_bloco_vs_mini.png`.
- Achado: BLOCO ≈ MINI; PolitiFact tem leve vantagem (+0.023) com bloco.
- Decisão: 1 modelo grande é suficiente para deploy.
- **Marcar explicitamente:** "Esta subseção responde decisão de engenharia, não pergunta de pesquisa."

## §5.8 Pesos no Hugging Face Hub (0.5 página)
- Persistência reproduzível; model card; CC-BY-4.0.
- Comando para baixar: `from huggingface_hub import snapshot_download`.
- **Citação ao Apêndice C** para detalhes técnicos.

---

# Capítulo 6 — Discussão e Conclusão

**Tamanho-alvo:** 4-6 páginas.
**Ordem de escrita:** QUARTO (depois de Resultados e Aplicação).

## §6.1 Síntese dos achados (1.5 página)
- Estrutura por RQ. Para cada uma:
  - **RQ1** (sinal acima do textual?): "No FakeNewsNet, o texto da raiz já satura o sinal (LogReg=0.86); GNNs adicionam <2pp. No UPFD-GossipCop, topologia agrega mais (+6pp via SAGE). No UPFD-PolitiFact, GNN não agrega valor significativo."
  - **RQ2** (sob quais condições?): "Domínios com Cohen's d ≥ 0.5 em alguma métrica de cascata."
  - **RQ3** (só topologia é viável?): "Sim, condicional ao d acima. Independência de idioma confirmada para PT vs EN; parcialmente refutada para DE vs EN com causa não isolada."

## §6.2 Achado negativo é achado (1 página)
- **Defesa explícita do resultado de PolitiFact.**
- Texto sugerido:
  > "O resultado negativo em UPFD-PolitiFact não é falha de implementação nem indicação de modelo inadequado. É uma observação científica válida sobre a *aplicabilidade* do paradigma topológico. Cohen's d desprezível (Tabela ...) mostra que, para esse dataset, fake e real são estruturalmente indistinguíveis nas métricas avaliadas. Reportar isso com a mesma extensão dos achados positivos é necessário pela tese central do trabalho."

## §6.3 Limitações (1 página)
Lista numerada de 7 limitações:
- (a) FakeNewsNet com cascatas estrela (ausência de timestamps precisos).
- (b) Regra 0.8/0.2 derivada de UM experimento (UPFD-GossipCop).
- (c) Bluesky sem labels = só demonstração.
- (d) Reprodutibilidade não bit-exact (CUDA/cuDNN).
- (e) Sem comparação numérica direta com BiGCN/GCNFN.
- (f) Bluesky é snapshot temporal (2024-2025), não amostra probabilística — generalização a janelas futuras não garantida.
- (g) Ambiente CPU-only (Python 3.12.10, torch 2.10+cpu): hiperparâmetros podem ser subótimos em GPU.

## §6.4 Trabalhos futuros (1 página)
4 direções:
1. Pseudo-labels no Bluesky (LogReg-BERT como oráculo) → retreino especializado.
2. Calibração de probabilidades (Platt/isotônica) com ground-truth de algum subset.
3. Expansão para datasets PolitiFact-like recentes (Twitter pós-2023, Mastodon).
4. Replicação direta de BiGCN/GCNFN para benchmark numérico contra a literatura.

## §6.5 Considerações finais (0.5 página)
- 1 parágrafo curto, sem grandiloquência.
- Texto âncora: "Este trabalho não propôs uma nova arquitetura GNN. Propôs um diagnóstico — sob quais condições a topologia da propagação detecta fake news, e quando não detecta. A defesa de um resultado negativo (PolitiFact) lado a lado de um resultado positivo (GossipCop), ancorada em estatística de tamanho de efeito, é a contribuição metodológica que esperamos transcender o escopo do dataset específico."

**Sem figuras neste capítulo.**

---

# Apêndices

## Apêndice A — Hiperparâmetros completos
- Tabela com todos os hiperparâmetros por arquitetura. Use `Training/03_Mega_Research/{gcn_model.py, gat_model.py, sage_model.py}` como fonte.

## Apêndice B — Índice geral de figuras e tabelas
- Versão LaTeX-formatada do `Execution/results/figuras_tcc/final/INDICE.md`.

## Apêndice C — Estrutura do repositório
- `Training/03_Mega_Research/` (scripts 00–27, descrição em 1 linha cada).
- `Execution/{weights, results}` (artefatos persistidos).
- `Material/` (datasets brutos).
- `Interface/frontend/` (API + Next.js).
- README.md como fonte autoritativa.

## Apêndice D — Evolução metodológica (opcional)
- Tabela curta. Apenas se a banca pedir contexto.

---

# Pegadinhas globais (revisar antes de submeter)

1. **Não diga "isso prova que..."** — use "evidência sugere", "consistente com", "compatível com a hipótese de".
2. **Sempre que citar um número exato**, dê a fonte do CSV/script entre parênteses no parágrafo: `(script 14, fase4_benchmarks/topologia_sem_texto/resultados.csv)`.
3. **F1=0.81 com std=0.002 vs F1=0.814** (diferença de 4 milésimos): a primeira é média de 10 seeds (script 14, §4.4); a segunda é run único da seed=42 (script 17, §4.7). **Sempre cite a fonte exata** pra não parecer cherry-picking.
4. **Cuidado com ordem dos dígitos brasileiros vs americanos:** texto em PT use vírgula decimal (0,810); em EN use ponto (0.810). O resumo PT/EN do `main.tex` já está correto — verifique consistência interna.
5. **Toda figura precisa de `\caption{}` e `\label{fig:xxx}`.** Toda tabela idem (`\label{tab:xxx}`). Já implementado nas tabelas geradas pelo script 24.
6. **Não duplique tabelas/figuras entre capítulos.** Cada `\ref{}` aponta pra um único local. Se vc precisar referenciar T11 tanto em §4.4-bis quanto em §5.5, ele vai no §4.4-bis (primeiro lugar onde aparece) e em §5.5 vc usa `\ref{tab:rq3_multilingual}`.
7. **Ordem de figuras importa pra fluxo do leitor.** Sempre que possível: tabela ANTES da figura que ilustra a tabela.
8. **Apêndice D opcional comentado** no `main.tex` — descomente apenas se a banca pedir.
9. **Imagens**: ao usar no LaTeX, copie os PNGs de `Execution/results/figuras_tcc/final/figuras/` para `Material/GNN_TCC_atualizado/Imagens/` antes de upload pro Overleaf. Mantenha os mesmos nomes (F1_..., F16_...) pra rastreabilidade.

---

# Ordem recomendada de escrita

Conforme parecer da rodada 2: **3 → 4 → 5 → 2 → 6 → 1**.

Razão: capítulos densos em fato (Metodologia, Resultados, Aplicação) primeiro; capítulos destilatórios (Fundamentação, Conclusão, Introdução) depois — eles se beneficiam de já saber exatamente o que vc precisa fundamentar/concluir/introduzir.

Estimativa de páginas total: **51-76 páginas** (sem capa, sumário, refs e apêndices). Capítulos 4 e 5 dominam.
