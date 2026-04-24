# Materiais consolidados para o TCC

Tabelas em `tabelas/` (LaTeX) e figuras em `figuras/` (PNG/PDF).
Cada item indica o capitulo sugerido do TCC.

## Tabelas LaTeX

- **T1** `T1_baseline_textual_fnn.tex` -- Capitulo de Resultados (Fase 2). LogReg/RF sobre BERT(titulo).
- **T2** `T2_significancia_posfull.tex` -- Capitulo de Resultados. ttest_rel pareado por fold GCN vs GAT vs SAGE no FNN posfull (todos ns).
- **T3** `T3_cross_dataset_upfd.tex` -- Capitulo de Resultados. Compara archs nos 2 UPFDs oficiais.
- **T4** `T4_topologia_sem_texto.tex` -- Capitulo de Resultados. Pergunta: 'topologia sozinha detecta fake?'. Achado: SAGE-GossipCop=0.81.
- **T5** `T5_cohens_d_estrutural.tex` -- Capitulo de Discussao. Explica POR QUE GossipCop funciona (d=+1.53 em branching).
- **T6** `T6_concordancia_gossipcop.tex` -- Capitulo de Discussao. Quando ambos batem, F1=0.97. Quando discordam, textual ainda acerta (0.91), topo erra (0.08).
- **T7** `T7_bloco_vs_mini.tex` -- Capitulo de Discussao. Bloco unico ~ Mini, com leve vantagem em datasets pequenos (UPFD-Polit +0.023).
- **T8** `T8_bluesky_inferencia_feeds.tex` -- Capitulo de Aplicacao. Political Science=69% fake-like. News=7% (paradoxal -- viral mas nao 'fake-padrao').
- **T9** `T9_bluesky_concord_size.tex` -- Capitulo de Aplicacao. Concordancia maxima em posts 20-100 nos (74%); minima em posts medios (42-44%).
- **T10** `T10_upfd_vs_publicado.tex` -- Capitulo de Resultados/Benchmarks. Nossos GNNs ficam dentro do range publicado (PolitiFact ~0.78-0.82 vs 0.846 reportado; GossipCop ~0.94-0.95 vs 0.97 reportado). Diferenca explicada por feature ('content' 310d vs 'bert' 768d).

## Figuras

- **F1** `F1_separabilidade.png` -- Cohen's $d$ por metrica e dataset (3 datasets x 5 metricas)
- **F2** `F2_distribuicoes_estruturais.png` -- Distribuicoes fake vs real por dataset/metrica (histogramas sobrepostos)
- **F3** `F3_textual_vs_topo_scatter.png` -- Scatter score textual vs topologico no UPFD-GossipCop
- **F4** `F4_bloco_vs_mini.png` -- Bar chart MINI vs BLOCO grande por dataset
- **F5** `F5_bluesky_crossfeed_distrib.png` -- Distribuicao de likes/reposts/replies por feed Bluesky
- **F6** `F6_bluesky_crossfeed_pares.png` -- Heatmap Cohen's $d$ entre feeds Bluesky
- **F7** `F7_bluesky_scores_por_feed.png` -- Distribuicao de score 'fake-like' por feed Bluesky
- **F8** `F8_bluesky_matriz_confusao.png` -- Matriz de confusao 2x2 textual vs topologico no Bluesky
- **F9** `F9_bluesky_agreement_size.png` -- Agreement rate por bin de num\_nodes
- **F10** `F10_bluesky_agreement_feed.png` -- Agreement rate por feed Bluesky
- **F11** `F11_bluesky_agreement_textlen.png` -- Agreement rate por bin de comprimento de texto
- **F12** `F12_gnnexplainer_gossipcop/` -- GNNExplainer aplicado em SAGE/GossipCop (5 amostras)
- **F13** `F13_threads_bluesky/` -- Visualizacoes de 6 threads reais Bluesky de tamanhos variados, com score topologico sobreposto
