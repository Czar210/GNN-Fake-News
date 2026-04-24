# Scripts Legacy

Scripts movidos para cá em 2026-04-23 durante a reorganização da Fase 4 do
[plano_correcao.md](../../../plano_correcao.md).

Estes scripts **não fazem parte do pipeline corrigido** descrito em
[plano_execucao.md](../../../plano_execucao.md). Foram preservados por valor
histórico (lógica usada antes da auditoria) ou para referência cruzada.

**Atenção:** os caminhos relativos dentro destes scripts (e.g.
`os.path.dirname(os.path.dirname(...))`) **estão quebrados** após a movimentação
para esta subpasta. Isso é intencional — assume-se que ninguém vai mais
executá-los como estão. Se precisar ressuscitar algum, ajuste os paths
para subir um nível adicional.

## Categorias

- **Análises antigas:** `analise_topologia.py`, `matrizes_ablation.py`,
  `matriz_confusao.py`, `extrator_amostra.py`, `pipeline_mensuracao.py`
- **Treinos isolados antigos:** `treinar_mega_dataset.py`,
  `treinar_upfd_robusto.py`, `restaurar_upfd.py`, `treinamento_cruzado_escala.py`
- **Testes ad-hoc:** `teste_fresh_bluesky.py`, `teste_heatmap_mega_dataset.py`,
  `duelo_modelos_frescos.py`, `baseline_upfd.py`

## Pipeline atual (não-legacy)

Ver scripts em `Training/03_Mega_Research/`:
- `00_construir_grafos_fakenewsnet.py`
- `02_construir_grafos_bluesky.py`
- `gerar_folds.py`
- `10_baseline_textual.py` ... `13_benchmark_upfd_oficial.py`
- Modelos: `gcn_model.py`, `gat_model.py`, `sage_model.py`
