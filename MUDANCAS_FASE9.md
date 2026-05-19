# Mudanças da Fase 9 — análise estratificada do erro

> Atualização pra quem está acompanhando o TCC. Em commits: `3567d66` (código + docs) e a edição dos capítulos LaTeX que veio depois.

## TL;DR

- Implementei 3 scripts novos (`28`, `29`, `30`) que cruzam **predição do modelo** com **métricas estruturais do grafo** — pergunta: "onde, dentro da distribuição, o modelo erra?"
- Os números confirmaram a "inversão distributiva" que estava só descrita em prosa na §4.6 do TCC, e revelaram um achado novo: o ganho da GNN sobre o baseline tabular **cresce 3.5× em cascatas profundas**, com IC bootstrap que não cruza zero.
- Cap 3 e Cap 4 do TCC ganharam novas seções (§3.8 e §4.8) já scaffolded em LaTeX — figuras nos lugares, tabelas referenciadas, comentários `% ARGUMENTO:` e `% ESCREVER:` pro Cesar escrever a prosa.

---

## O que mudou no código

3 scripts novos em `Training/03_Mega_Research/`:

| Script | O que faz | Insumo | Output |
|---|---|---|---|
| `28_distribuicao_e_estratificacao.py` | LOWESS de P(acerto) vs métrica + fit lognormal + tercis. Salva `por_grafo.csv` com predição+acerto por grafo (insumo dos próximos). | UPFD-GossipCop test + modelos persistidos do script 17 | `figuras_tcc/estratificacao/` (4 figs + 3 CSVs + relatório) |
| `29_outliers.py` | 6 subgrupos extremos definidos via percentil empírico (não z-score, porque heavy-tail). Confirma inversão distributiva. | `por_grafo.csv` | `figuras_tcc/outliers/` (2 figs + 2 CSVs + relatório) |
| `30_cascatas_profundas.py` | F1 por valor de depth + bootstrap pareado pro gap SAGE−RF em subsets. | `por_grafo.csv` + dataset (só pra HTML) | `figuras_tcc/cascatas_profundas/` (2 figs + 3 CSVs + 3 HTMLs + relatório) |

[PIPELINE.md](Training/03_Mega_Research/PIPELINE.md) ganhou nova "Fase 9" documentando os três + box de achados principais no topo. O script `14_topologia_sem_texto.py` também ganhou uma referência cruzada (achado do 30 refina o "smoking gun" de "topologia" pra "topologia multi-hop").

---

## O que mudou no TCC (LaTeX)

**Cap 3 — Metodologia:** adicionei §3.8 "Análise estratificada do erro" entre Protocolo (§3.7) e Explicabilidade (§3.9, antes era 3.8). Descreve as três técnicas usadas: LOWESS, percentil empírico, bootstrap pareado. ~0.7 página quando escrita.

**Cap 4 — Resultados:** adicionei §4.8 "Análise estratificada do erro" com 4 subseções + síntese. Painel mestre virou §4.9. Estrutura:

- **§4.8.1** Distribuição estrutural heavy-tail (F25 + 1 parágrafo)
- **§4.8.2** LOWESS: onde o modelo degrada (F26, F27)
- **§4.8.3** Outliers e inversão distributiva (T12, F29, F30)
- **§4.8.4** Cascatas profundas (F28, T13, F31, F32)
- Síntese de fechamento (1 parágrafo)

Total: ~3 páginas de prosa quando escrita.

**Novos assets em `Material/GNN_TCC_atualizado/`:**
- 8 figuras: `Imagens/F25_*.png` até `F32_*.png`
- 2 tabelas: `Imagens/tabelas/T12_outliers.tex`, `T13_gap_profundidade.tex`

---

## Os achados principais (com números)

Tudo no UPFD-GossipCop test (N=3826), usando os modelos persistidos pelo script 17 (RF estrutural `[num_nodes, grau_root]` e SAGE estrutural `[is_root, grau_norm]`).

### 1. Inversão distributiva confirmada numericamente

A §4.6 do TCC já dizia que o classificador topológico "se inverte" no subconjunto de discordância (F1=0.08). Agora temos N e acurácia direta:

- **`fake_contido`** (y=fake ∩ grau_root ≤ p5, N=31): SAGE acerta **0 de 31**. Inversão total.
- **`real_viral`** (y=real ∩ grau_root > p95, N=60): SAGE acerta **26 de 60** (acc=0.433, abaixo de chance).

### 2. Gap da GNN cresce com profundidade da cascata

Bootstrap pareado, n_boot=2000, IC 95%:

| Subset | N | F1 RF | F1 SAGE | Gap | IC 95% |
|---|---|---|---|---|---|
| Agregado | 3826 | 0.753 | 0.814 | +0.060 | [+0.047, +0.074] |
| depth ≥ 3 | 1143 | 0.657 | 0.791 | **+0.134** | [+0.103, +0.165] |
| depth ≥ 5 | 256 | 0.528 | 0.741 | **+0.213** | [+0.139, +0.287] |

Todos significantes (IC não cruza zero). **Gap em depth≥5 é 3.5× o agregado** — a GNN não está só capturando "número de nós" (que o RF tabular vê), está explorando topologia multi-hop.

### 3. Distribuições heavy-tail mais agressivas que lognormal

Fit lognormal rejeitado por KS em todas as 3 métricas (`num_nodes` p=8e-33, `depth_max` p=0, `branching_avg` p=8e-12). Justifica metodologicamente o uso de percentil empírico (não z-score) na definição dos outliers.

### 4. Premissa do TCC parcialmente contraditada

A §6.3.a atual diz "FNN com cascatas estrela" como limitação. **UPFD-GossipCop oficial não é estrela plana** — 70% têm depth=2 mas 30% têm depth≥3 (com cauda até depth=9). A crítica de "estrela" se aplica ao FakeNewsNet construído pelo nosso script 00, **não** ao UPFD oficial. Tem que refinar a redação dessa limitação no Cap 6.

---

## Como olhar os resultados

Se quiser bater o olho rápido:

- **Os números em texto**: `Execution/results/figuras_tcc/estratificacao/relatorio.txt`, `outliers/relatorio.txt`, `cascatas_profundas/relatorio.txt`
- **As figuras "mais bonitas"** pra entender o achado:
  - [F31](Execution/results/figuras_tcc/cascatas_profundas/fig_f1_por_depth.png) — F1 por depth (bar chart)
  - [F32](Execution/results/figuras_tcc/cascatas_profundas/fig_gap_subsets.png) — gap com IC bootstrap
  - [F29](Execution/results/figuras_tcc/outliers/fig_outliers_acc.png) — acurácia por subgrupo extremo
- **CSV raw pra fuçar**: `Execution/results/figuras_tcc/estratificacao/por_grafo.csv` (3826 linhas, todas as predições + métricas)
- **HTMLs interativos** das 3 cascatas mais profundas: `cascatas_profundas/amostra_deep_*.html` (abrir no navegador)

---

## O que falta

**Pra essa rodada:** nada. Código rodado, docs atualizadas, capítulos scaffolded.

**Próximos passos do TCC:**
- Cesar escreve a prosa das §3.8 e §4.8 seguindo os comentários `% ARGUMENTO:`, `% RESUMO:` e `% ESCREVER (Cesar):` que deixei no LaTeX
- Refinar §6.3.a (Limitações) pra distinguir FNN nosso vs UPFD oficial em relação a "estrela plana"
- Decidir se entra figura/tabela equivalente pra UPFD-PolitiFact (atualmente Fase 9 só tem GossipCop, porque é onde temos modelo persistido — replicar exigiria re-treino)

Os outros 5 capítulos do TCC continuam como esqueleto com comentários-guia detalhados (já estavam antes da Fase 9). Ordem sugerida de escrita: 4 → 5 → 3 → 6 → 2 → 1.
