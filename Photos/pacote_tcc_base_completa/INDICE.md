# Índice — TCC GNN Fake News (base completa)

Mapa de cada capítulo, cada seção, cada figura/tabela. Use isto pra navegar
e priorizar o que escrever primeiro.

**Ordem de escrita recomendada:** 3 → 4 → 5 → 6 → 2 → 1
(Cap 1 e 2 são os mais fáceis quando você já conhece a história inteira.)

---

## Cap 1 — Introdução (4-6 pág, sem figuras)

| § | Seção | Tese central |
|---|---|---|
| 1.1 | Contextualização | Desinformação antecede LLMs; NLP isolada não basta |
| 1.2 | Motivação | Topologia é complementar e independente de idioma |
| 1.3 | Perguntas de pesquisa | 3 RQs operacionalizáveis |
| 1.4 | Contribuições | 4 contribuições distintas com artefatos rastreáveis |
| 1.5 | Estrutura do trabalho | Mapa dos 5 capítulos restantes |

---

## Cap 2 — Fundamentação Teórica (12-18 pág, sem figuras)

| § | Seção | Tese / Conteúdo |
|---|---|---|
| 2.1 | Detecção de fake news: panorama | NLP-only e estrutura-only têm limites; hibridização é a direção |
| 2.2 | Grafos e redes sociais | Propagação = cascata enraizada |
| 2.3 | Embeddings textuais | Sentence-BERT multilíngue, 768 dim |
| 2.4 | Redes Neurais de Grafos | 3 subseções: GCN, GAT, SAGE (com equações) |
| 2.5 | Degeneração do GCN com features homogêneas | **Crítica** — equação `eq:degeneracao` justifica encodings |
| 2.6 | Encodings posicionais | `is_root`, `grau_norm`, `pos` quebram a degeneração |
| 2.7 | Explicabilidade em GNNs | GNNExplainer (Ying et al. 2019) |
| 2.8 | Validação estatística | k-fold + ttest_rel + Cohen's d + bootstrap |
| 2.9 | Trabalhos relacionados | 6 subseções (surveys, métodos, multilíngue, BR, homofilia, posicionamento) |

**Tabelas inline:** `tab:cohen_d_interpretacao` (interpretação dos d).
**Equações inline:** `eq:gcn`, `eq:gat_alpha`, `eq:degeneracao`.

---

## Cap 3 — Metodologia (8-12 pág)

| § | Seção | Figura/Tabela | Tese |
|---|---|---|---|
| 3.1 | Visão geral do pipeline | `fig:pipeline` (TikZ inline) | 5 blocos idênticos para os 3 datasets |
| 3.2 | Datasets | `tab:datasets` (inline) | 3 datasets, papéis distintos |
| 3.3 | Construção FNN | — | Script 00 com 3 variantes (posfull, posmin, posgrau) |
| 3.4 | Folds compartilhados | — | 10 folds estratificados, seed=42 |
| 3.5 | Arquiteturas GNN | `tab:hyperparams` (inline) | 3 arch com mesmos hiperparâmetros |
| 3.6 | Baselines não-GNN | — | LogReg-BERT (textual) + RF tabular (estrutural) |
| 3.7 | Protocolo de avaliação | — | F1-macro + ttest_rel + Cohen's d |
| 3.8 | **Análise estratificada** (NOVO) | — | LOWESS + percentil empírico + bootstrap |
| 3.9 | Explicabilidade | — | GNNExplainer em 20 amostras estratificadas |
| 3.10 | Classificador dual | — | **Post-hoc**, anunciado aqui, justificado em §4.6 |
| 3.11 | Reprodutibilidade | — | requirements pinado + seed universal |

---

## Cap 4 — Resultados (15-22 pág, capítulo mais denso)

| § | Seção | Figuras/Tabelas | Tese central |
|---|---|---|---|
| 4.1 | Baselines textuais | T1 | LogReg-BERT F1=0.86 no FNN — teto textual |
| 4.2 | Diagnóstico de confound | — | RF(num_nodes) F1=0.52 — tamanho não explica |
| 4.3 | Cross-dataset UPFD vs literatura | T3, T10 | Pipeline no envelope dos publicados |
| 4.4 | **Topologia sem texto (RQ3a)** | T4, F18 | **ACHADO CENTRAL** — SAGE F1=0.81 sem texto |
| 4.4 (cont) | Multilíngue (RQ3b) | T11, F14 | PT vs EN: d=-0.06; DE vs EN: d=-0.47 |
| 4.5 | Por que GossipCop funciona (RQ2) | T5, F17 | Cohen's d ≥ 0.5 = condição necessária |
| 4.6 | Concordância textual × topológico (RQ1) | T6, F3, F24 + eq:regra_dual | Inversão distributiva — F1=0.08 na discordância |
| 4.7 | GNNExplainer | F19 | Hop1 ≈ 1.0 pra fake; ≈ 0.5-0.6 pra real-grande |
| 4.8.1 | (NOVO) Distribuição heavy-tail | F25 | KS rejeita lognormal → justifica percentil empírico |
| 4.8.2 | (NOVO) LOWESS | F26, F27 | Onde modelo degrada continuamente |
| 4.8.3 | (NOVO) Outliers | T12, F29, F30 | **fake_contido N=31 → SAGE acc=0.000** |
| 4.8.4 | (NOVO) Cascatas profundas | T13, F28, F31, F32 | **Gap 3.5× em depth≥5, bootstrap significativo** |
| 4.9 | Painel mestre | F16 | Síntese visual dos 3 modelos × 3 datasets |

---

## Cap 5 — Aplicação (8-12 pág)

| § | Seção | Figuras/Tabelas | Tese |
|---|---|---|---|
| 5.1 | Dataset Bluesky | — | 168k posts, 11 feeds, snapshot 2024-2025, **sem labels** |
| 5.2 | Análise cross-feed | F5, F6 | Comunidades têm regimes distintos |
| 5.3 | Aplicação dos modelos | T8, F7 | Score = proxy ordinal, NÃO probabilidade calibrada |
| 5.4 | Concordância no Bluesky | T9, F20, F9, F10, F23 | Political Science 88.7%; zona cinzenta 42-44% |
| 5.5 | Independência de idioma | — | Versão reduzida de §4.4-bis, foco prático |
| 5.6 | Threads reais | F13 (pasta) | 6 visualizações pra defesa |
| 5.7 | Arquitetura web | snippet de código | FastAPI + Next.js + 3 modelos + regra dual |
| 5.7 (cont) | Bloco vs especialistas | T7, F4 | BLOCO ≈ MINI — decisão de engenharia, não RQ |
| 5.8 | Hugging Face Hub | — | Pesos sob CC-BY-4.0 |

---

## Cap 6 — Conclusão (4-6 pág, sem figuras)

| § | Seção | Conteúdo |
|---|---|---|
| 6.1 | Síntese por RQ | RQ1, RQ2, RQ3 — resposta de cada |
| 6.2 | Achado negativo é achado | Defesa explícita do resultado PolitiFact |
| 6.3 | Limitações | Lista numerada de 7 limitações |
| 6.4 | Trabalhos futuros | 4 direções: pseudo-labels, calibração, datasets novos, BiGCN |
| 6.5 | Considerações finais | 1 parágrafo curto |

---

## Apêndices

| Apêndice | Conteúdo |
|---|---|
| A | Hiperparâmetros completos (já existe) |
| B | Índice de figuras e tabelas (já existe, gerado por script 24) |
| C | Estrutura do repositório (já existe) |
| D | Evolução metodológica (opcional, descomentar em main.tex se a banca pedir) |

---

## Inventário de figuras (32 figuras)

**Estruturais (Cap 4 §§4.4-4.7):**
- F1 Cohen's d separabilidade — **não usada** (substituída pela F17)
- F2 Distribuições estruturais — **não usada** (substituída pela F17 + tabelas)
- F3 Scatter textual vs topo (§4.6)
- F16 Painel f1 mestre (§4.9)
- F17 Forest plot Cohen's d (§4.5)
- F18 Heatmap topologia sem texto (§4.4)
- F19 Hop importance GNNExplainer (§4.7)
- F24 Fluxograma regra dual (§4.6)

**Análise estratificada (Cap 4 §4.8, NOVAS):**
- F25 Distribuição lognormal
- F26 LOWESS acerto
- F27 F1 por tercil
- F28 Distribuição de profundidade
- F29 Outliers acurácia
- F30 Outliers composição
- F31 F1 por depth
- F32 Gap subsets com IC bootstrap

**Multilíngue (Cap 4 §4.4):**
- F14 RQ3 multilingual dist
- F15 RQ3 multilingual qq — **não usada** (versão alternativa)

**Bluesky (Cap 5):**
- F4 Bloco vs mini (§5.7)
- F5 Bluesky crossfeed distribuição (§5.2)
- F6 Bluesky crossfeed pares heatmap (§5.2)
- F7 Bluesky scores por feed (§5.3)
- F8 Bluesky matriz confusão — **não usada** (substituída pela F20)
- F9 Bluesky agreement size (§5.4)
- F10 Bluesky agreement feed (§5.4)
- F11 Bluesky agreement textlen — **não usada** (redundante com F9, F10)
- F12 GNNExplainer GossipCop (pasta) — **não usada** (resumida em F19)
- F13 Threads Bluesky (pasta, §5.6)
- F20 Bluesky crosstab (§5.4)
- F23 Bluesky modelo × feed heatmap (§5.4)

**Figuras presentes no zip mas SEM uso atual** (5 figuras): F1, F2, F8, F11, F15. Pode deletar do zip ou deixar como reserva.

---

## Inventário de tabelas (13 tabelas)

| ID | Conteúdo | Seção |
|---|---|---|
| T1 | Baseline textual FNN | §4.1 |
| T2 | Significância posfull (paired t-test) | **não usada** no scaffolding atual |
| T3 | Cross-dataset UPFD | §4.3 |
| T4 | Topologia sem texto | §4.4 |
| T5 | Cohen's d estrutural | §4.5 |
| T6 | Concordância GossipCop | §4.6 |
| T7 | Bloco vs mini | §5.7 |
| T8 | Bluesky inferência por feed | §5.3 |
| T9 | Bluesky concordância × tamanho | §5.4 |
| T10 | UPFD nosso vs publicado | §4.3 |
| T11 | RQ3 multilíngue | §4.4 (bis) |
| T12 | **Outliers (NOVA)** | §4.8.3 |
| T13 | **Gap profundidade (NOVA)** | §4.8.4 |

T2 ficou sem casa porque o scaffolding atual não tem seção pra ela. Se quiser usar, ela cabe entre §4.4 e §4.5 ou no apêndice.

---

## Os 3 marcadores no LaTeX

Pra cada seção/subseção, deixei 3 tipos de comentário:

- `% TESE: ...` — afirmação central (1 frase). **Mantenha mentalmente** ao escrever.
- `% ANTES: ...` — bullets do que dizer antes da figura/tabela.
- `% DEPOIS: ...` — bullets do que dizer depois (interpretar números).
- `% TRANSICAO: ...` — ponte de 1 frase pra próxima seção.
- `% CONTEUDO: ...` — pra seções sem figura, é o equivalente.

**Conforme escreve a prosa, apaga o comentário correspondente.**
A tese pode ficar como verificação final ("o parágrafo que escrevi sustenta essa tese?").
