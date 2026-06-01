# FATOS_TCC.md — Fonte da verdade

Registro de cada fato verificável do TCC, por capítulo. **Ao escrever prosa,
cite o valor daqui — não da memória, não do scaffolding.** Os comentários do
LaTeX (`% TESE`, `% DEPOIS`) podem estar desatualizados; esta tabela não.

## Legenda de status

| Símbolo | Significado |
|---|---|
| ✅ | Verificado nesta sessão contra arquivo/script/paper |
| 📊 | Vem de tabela gerada (`Tx.tex`, produzida pelo script 24 a partir de CSV) |
| 📄 | Vem de `relatorio.txt` de um experimento |
| ⚠️ | **Discrepância** — fontes conflitam, resolver antes de finalizar |
| ❓ | Não verificado nesta sessão — pode estar velho, confirmar antes de usar |

## Regra de ouro

Se um número não está aqui, **não invente** — rode o script ou abra o CSV.
Se um número aqui está marcado ⚠️ ou ❓, **não use sem resolver primeiro**.

---

# Cap 1 — Introdução

Capítulo narrativo, poucos fatos duros. Os "fatos" são as definições do trabalho.

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| RQ1 | Topologia carrega sinal acima do BERT da raiz? | `estrutura_tcc_v2.md` | ✅ |
| RQ2 | Sob quais condições estruturais? (via Cohen's d) | idem | ✅ |
| RQ3 | É viável detectar só com topologia? (a: estrutural puro; b: invariância a idioma) | idem | ✅ |
| Tese central | Topologia funciona quando Cohen's d ≥ 0.5 em alguma métrica de cascata | idem | ✅ |
| Contribuições | 4: (a) ablation encodings; (b) GossipCop+ / PolitiFact−; (c) concordância + regra dual; (d) pesos HF + ferramenta web | idem | ✅ |
| Autores | André Messina Livingston, César Augusto Sibila, Enzo Kikuji Takida | `main.tex` | ✅ |

---

# Cap 2 — Fundamentação Teórica

Capítulo teórico. Os fatos são parâmetros/definições, não resultados.

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Modelo BERT | `paraphrase-multilingual-mpnet-base-v2`, 768 dim | `00_construir_grafos_fakenewsnet.py:102` | ✅ |
| Tipo de BERT | Sentence-BERT (não BERT vanilla) | `reimers2019sbert` | ✅ |
| Cohen's d — limiares | <0.2 desprezível; 0.2–0.5 pequeno; 0.5–0.8 médio; ≥0.8 grande | `cohen1988statistical` | ✅ |
| GCN | 3 camadas, agregação espectral, ~58k parâmetros | `gcn_model.py` | ✅ |
| GAT | 3 camadas, 4 heads de atenção, ~218k parâmetros | `gat_model.py` | ✅ |
| SAGE | 3 camadas SAGEConv, agregação mean, ~115k parâmetros | `sage_model.py` | ✅ |

---

# Cap 3 — Metodologia

## §3.2 Datasets

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| FNN PolitiFact — N grafos | **754** (452 train + 150 val + 152 test) | contagem direta dos `.pt` | ✅ |
| FNN — balanço de classes | 374 fake (49.6%) / 380 real | contagem direta | ✅ |
| FNN — origem | CSVs `politifact_fake.csv` + `politifact_real.csv` do GitHub KaiDMML | `00_*.py:60-62` | ✅ |
| FNN — citação | `shu2020fakenewsnet` | — | ✅ |
| FNN — avaliação | 10-fold CV sobre os 754 (não usa o split train/val/test dos `.pt`) | `gerar_folds.py` | ✅ |
| UPFD-PolitiFact — N grafos | **314** (62 train + 31 val + 221 test) | contagem direta | ✅ |
| UPFD-PolitiFact — balanço | 157 fake (50.0%) / 157 real | contagem direta | ✅ |
| UPFD-GossipCop — N grafos | **5.464** total (1092 train + 546 val + 3826 test) | contagem direta + metadata.json | ✅ |
| UPFD-GossipCop — balanço | 2732 fake (50.0%) / 2732 real | contagem direta | ✅ |
| UPFD — split | Fixo oficial (não k-fold); maioria vai pro teste (ex.: Polit 62/31/221) | contagem direta | ✅ |
| UPFD — citação | `dou2021upfd` | — | ✅ |
| Bluesky — N posts | **168.463** (soma dos 11 jsonl em `dados_bluesky/feed_posts/`) | contagem direta | ✅ |
| Bluesky — N feeds | **11** (Blacksky, News, Science, Political Science, AcademicSky, BookSky, Game Dev, GreenSky, #Disability, #UkrainianView, What's History) | paper Failla + arquivos locais | ✅ |
| Bluesky — origem | Dataset público de Failla & Rossetti, Zenodo DOI 10.5281/zenodo.14258401 | paper + Zenodo | ✅ |
| Bluesky — citação | `failla2024bluesky` (NÃO `quelle2024bluesky` — ver discrepância D5) | — | ✅ |
| Bluesky — subconjunto de trabalho | ~6 GB (pasta `dados_bluesky/`) | `ls dados_bluesky/` | ✅ |
| Bluesky — labels | Heurística keyword + moderation tags; **não usados como supervisão** | `01_*.py:45-59` | ✅ |

## §3.3 Construção de grafos FNN

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Script | `00_construir_grafos_fakenewsnet.py` | — | ✅ |
| Estrutura do grafo | Estrela plana: raiz=notícia, filhos=tweets | `00_*.py` docstring | ✅ |
| Por que estrela | CSV bruto não tem `parent_of_retweet` nem timestamp | `00_*.py:16` | ✅ |
| Dim das features (FNN/Bluesky) | **771** = 768 BERT + 3 posicionais | `00_*.py` | ✅ |
| BERT replicado em todos os nós | Sim — filhos recebem o mesmo BERT da raiz (não têm texto próprio) | `00_*.py:120-121` | ✅ |
| 3 variantes | posfull (768+3), posmin (768+1, grau zerado), posgrau (768+2, pos zerada) | `00_*.py` FEATURE_VARIANTS | ✅ |
| Features posicionais | `is_root` (1 raiz/0 filhos); `grau_norm` (N/N_max só na raiz); `pos` (i/N nos filhos, 0 na raiz) | `00_*.py` | ✅ |
| Caveat de `pos` | Ordem do CSV, não cronológica real | `00_*.py:16,134` | ✅ |
| Min tweets por grafo | 2 (`MIN_TWEETS`) — descarta cascatas degeneradas | `00_*.py:66` | ✅ |
| Max nós por grafo | 100 (`MAX_NOS_DEFAULT`) — trunca filhos em 99 | `00_*.py:65` | ✅ |

### Verificação matemática dos filtros do FNN (cap de 100 nós)

Distribuição de tamanho de cascata (tweets por notícia) nos CSVs brutos do
KaiDMML, **antes** de qualquer filtro — verificada baixando os CSVs e contando
`tweet_ids` por linha.

| Fato | Valor | Status |
|---|---|---|
| Notícias no CSV bruto (fake + real) | 1.056 | ✅ |
| Tweets por notícia — mediana | 30 | ✅ |
| Tweets por notícia — média | **552,6** (≈18× a mediana) | ✅ |
| Tweets por notícia — máximo | **29.060** (uma única notícia) | ✅ |
| Percentis | p75=246, p90=1.045, p95=2.043, p99=10.165 | ✅ |
| Distribuição | heavy-tail extremo (power-law-like) | ✅ |
| `MIN_TWEETS=2` descarta | 302 de 1.056 (28,6%) — notícias com <2 tweets | ✅ |
| Sobreviventes ao `MIN_TWEETS` | 754 grafos | ✅ |
| `MAX_NOS=100` trunca | **377 de 754 grafos (50,0%)** | ✅ |
| Massa de nós-filho removida pelo cap | **91,7%** (583.469 → 48.446 filhos) | ✅ |

**⚠️ Implicação metodológica (ver D6):** o cap de 100 **não** é trimming de
poucos outliers — trunca metade do dataset e remove 92% da massa de nós. Como
consequência, os 377 grafos truncados ficam todos com `num_nodes=100` e o mesmo
`grau_norm` na raiz — o modelo **não distingue** uma cascata de 100 tweets de
uma de 29.060. A justificativa do cap (§3.3) tem de reconhecer essa magnitude.

## §3.4 Folds

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Script | `gerar_folds.py` | — | ✅ |
| Config | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` | `gerar_folds.py` | ✅ |
| Saída | `data/folds_fnn.pt` | — | ✅ |

## §3.5 Arquiteturas e hiperparâmetros

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| lr / weight_decay | 0.001 / 5e-4 | `17_*.py:91` | ✅ |
| batch_size / epochs | 32 / 30 | `17_*.py` | ✅ |
| hidden_channels | 64 | `sage_model.py` | ✅ |
| optimizer / scheduler | Adam / ReduceLROnPlateau(factor=0.5, patience=5) | `17_*.py:91-93` | ✅ |
| early stopping | patience=7 sobre F1-macro de validação | `17_*.py:108` | ✅ |
| seed padrão | 42 (12345 no construtor dos modelos) | `17_*.py:43`, `sage_model.py:69` | ✅ |

## §3.6 Baselines não-GNN

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| LogReg-BERT | `LogisticRegression(max_iter=1000)` sobre `x[0]` (BERT raiz, 768d) | `17_*.py:133` | ✅ |
| RF tabular | `RandomForestClassifier(n_estimators=200)` sobre `[num_nodes, grau_root]` | `17_*.py:164` | ✅ |

## §3.7 Protocolo

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Métrica primária | F1-macro | — | ✅ |
| Teste de significância | `scipy.stats.ttest_rel` pareado sobre 10 folds | `09_*.py` | ✅ |
| Effect size | Cohen's d entre fake/real em métricas estruturais | `15_*.py` | ✅ |

## §3.8 Análise estratificada (Fase 9)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| LOWESS | Rolling-mean, janela 10% da amostra (mín. 30) | `28_*.py` | ✅ |
| Outliers | Percentil empírico p5/p95 (não z-score, por heavy-tail) | `29_*.py` | ✅ |
| Bootstrap | Pareado, n_boot=2000, CI 95% | `30_*.py` | ✅ |

## §3.11 Reprodutibilidade + Detalhes de Computação

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Sistema operacional | Windows 11 | `platform.system()` | ✅ |
| Python | 3.12.10 | `platform.python_version()` | ✅ |
| PyTorch | 2.10.0+cpu (**CPU-only — não há GPU**) | `torch.__version__` | ✅ |
| CPU | Intel64 Family 6 Model 183 (Raptor Lake, 14ª geração) | `platform.processor()` | ✅ |
| Cores lógicos | 16 | `os.cpu_count()` | ✅ |
| RAM total | 15,7 GB | `psutil.virtual_memory()` | ✅ |
| Seed universal | 42 | múltiplos scripts | ✅ |

### Hiperparâmetros de treino dos modelos finais (script 17)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| LogReg-BERT | `max_iter=1000`, `random_state=42` | `17_*.py:133` | ✅ |
| RandomForest estrutural | `n_estimators=200`, `random_state=42`, `n_jobs=-1` | `17_*.py:164` | ✅ |
| SAGE estrutural | `lr=0.001`, `weight_decay=5e-4`, `batch_size=32`, `epochs=30 (early stop patience=7)`, scheduler ReduceLROnPlateau (factor=0.5, patience=5) | `17_*.py:89-108` | ✅ |
| Otimizador SAGE | Adam | `17_*.py:91` | ✅ |

### Tamanho dos modelos persistidos (em `Execution/weights/`)

| Modelo | Tamanho | Status |
|---|---|---|
| `logreg_bert_fnn.pkl` | **6,9 KB** | ✅ |
| `rf_struct_gossipcop.pkl` | **9,6 MB** (200 árvores) | ✅ |
| `sage_struct_gossipcop.pth` | **71,2 KB** | ✅ |
| `metadata.json` | 1,6 KB | ✅ |
| **Total** | **~9,7 MB** para os 3 modelos | ✅ |
| Cache BERT FNN (`_bert_titulos_cache.pt`) | 7,0 MB | ✅ |

### Tempo de carga em produção (medição direta)

| Operação | Tempo | Status |
|---|---|---|
| Carregar os 3 modelos persistidos (já com PyG importado) | **1,4 s** | ✅ |
| Carregar o BERT multilingual (primeira vez) | **19,2 s** | ✅ |
| RAM consumida pelos 3 modelos | **~92 MB** | ✅ |

### Tempo de inferência por post (média de 200 runs em CPU)

| Etapa | Tempo | Status |
|---|---|---|
| BERT embedding (1 texto) | **17,9 ms** | ✅ |
| LogReg-BERT (sobre embedding pronto) | **0,07 ms** | ✅ |
| RF estrutural (sobre `[num_nodes, grau_root]`) | **50,8 ms** | ✅ |
| SAGE estrutural (grafo de 5 nós) | **0,56 ms** | ✅ |
| **Pipeline textual completo** (BERT + LogReg) | **~18 ms** | ✅ |
| **Pipeline estrutural-RF completo** | **~51 ms** | ✅ |
| **Pipeline estrutural-SAGE completo** | **~1 ms** | ✅ |

### BERT multilingual (atende RQ3 e pedido do orientador)

| Fato | Valor | Status |
|---|---|---|
| Modelo | `paraphrase-multilingual-mpnet-base-v2` | ✅ |
| Arquitetura base | XLM-RoBERTa fine-tuned (Sentence-BERT) | ✅ |
| Dimensão | 768 | ✅ |
| **Idiomas suportados** | **50+ idiomas** | ✅ |
| Inclui (não exaustivo) | pt, en, es, de, fr, it, ja, zh-cn, zh-tw, ar, ru, hi, ko, tr, nl, pl, sv, uk, vi, th, id, ms, fa, he, cs, da, fi, hu, no, ro, bg, sr, sk, sl, lt, lv, et, hr, ca, eu, gl, ... | ✅ |
| Fonte | https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | ✅ |

---

# Cap 4 — Resultados

## §4.1 Baseline textual (T1)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| LogReg-BERT FNN | F1-macro = **0.8593 ± 0.0431** | 📊 T1 | ✅ |
| RandomForest-BERT FNN | F1-macro = 0.8549 ± 0.0372 | 📊 T1 | ✅ |
| Protocolo | k-fold estratificado, k=10 | 📊 T1 | ✅ |

## §4.2 Confound topológico

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| RF só com `num_nodes` | F1 ≈ 0.52 (abaixo do gate 0.65) | scaffolding §4.2 / `11_*.py` | ❓ confirmar contra `fase2_baselines/confound_diagnostico/` |

## §4.3 Cross-dataset UPFD (T3, T10)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| GossipCop GCN / GAT / SAGE (com texto) | 0.9411 / 0.9390 / 0.9448 | 📊 T3 | ✅ |
| GossipCop LogReg_root | 0.9540 | 📊 T3 | ✅ |
| PolitiFact GCN / GAT / SAGE (com texto) | 0.7816 / 0.6936 / 0.8166 | 📊 T3 | ✅ |
| PolitiFact LogReg_root | 0.8415 | 📊 T3 | ✅ |
| vs Dou et al. — PolitiFact | nosso SAGE 0.817 vs publicado acc 0.846 | 📊 T10 | ✅ |
| vs Dou et al. — GossipCop | nosso SAGE 0.945 vs publicado acc 0.971 | 📊 T10 | ✅ |
| Nota da comparação | nosso F1-macro / feature `content` 310d; deles accuracy / feature `bert` 768d | 📊 T10 caption | ✅ |

## §4.4 Topologia sem texto (T4, F18) — ACHADO CENTRAL

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| SAGE-GossipCop sem texto | F1-macro = **0.8097 ± 0.0023** (10 seeds) | 📊 T4 | ✅ |
| GCN / GAT GossipCop sem texto | 0.6445 / 0.4834 | 📊 T4 | ✅ |
| SAGE-PolitiFact sem texto | **0.3314 ± 0.0093** ⚠️ ver D1 | 📊 T4 | ⚠️ |
| GCN / GAT PolitiFact sem texto | 0.3621 / 0.3800 | 📊 T4 | ✅ |
| SAGE-FNN sem texto | 0.5362 ± 0.0747 | 📊 T4 | ✅ |
| Features usadas | `[is_root, grau_norm]` (2 dim) | 📊 T4 caption | ✅ |

## §4.5 Por que GossipCop funciona (T5, F17)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| GossipCop — Cohen's d branching_avg | **+1.534** (efeito grande) | 📊 T5 | ✅ |
| GossipCop — Cohen's d num_nodes | +0.905 (efeito grande) | 📊 T5 | ✅ |
| PolitiFact — Cohen's d num_nodes | **−0.245** (pequeno, NÃO desprezível) ⚠️ ver D2 | 📊 T5 | ⚠️ |
| PolitiFact — Cohen's d branching_avg | +0.059 (desprezível) | 📊 T5 | ✅ |
| FNN — Cohen's d num_nodes / branching | +0.223 / +0.223 (pequeno) ⚠️ ver D2 | 📊 T5 | ⚠️ |
| Nota | em estrela, branching_avg = num_nodes − 1 (só a raiz é não-folha) | dedução estrutural | ✅ |

## §4.6 Concordância textual × topológico (T6, F3)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| F1 textual (LogReg-content) GossipCop | 0.9540 | 📊 T6 | ✅ |
| F1 topológico (RF estrutural) GossipCop | 0.7534 | 📊 T6 | ✅ |
| Agreement rate | 0.7588 | 📊 T6 | ✅ |
| Cohen's kappa | 0.5179 | 📊 T6 | ✅ |
| F1 quando ambos concordam | **0.9662** | 📊 T6 | ✅ |
| F1 textual quando discordam | 0.9139 | 📊 T6 | ✅ |
| F1 topológico quando discordam | **0.0845** | 📊 T6 | ✅ |
| Regra dual | concordam: 0.5/0.5; discordam: 0.8 textual / 0.2 topológico | `Interface/.../main.py` | ✅ |

## §4.7 GNNExplainer (F19)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Hop1 importância — FAKE | ≈ 0.97–1.00 | scaffolding §4.7 / `figuras_tcc/gnnexplainer/` | ❓ confirmar `hop_importance.csv` |
| Hop1 importância — REAL-grande | ≈ 0.50–0.62 | idem | ❓ |
| RF estrutural vs SAGE estrutural | 0.753 vs 0.814 → diff 6.1pp | 📊 T6 (RF) + Fase 9 (SAGE) | ✅ |
| Amostras | 20 estratificadas (5 por classe×tamanho) | `16_*.py` | ✅ |

## §4.8 Análise estratificada (Fase 9 — T12, T13, F25–F32)

Todos os números abaixo: **UPFD-GossipCop test, N=3826.** Verificados rodando
os scripts 28/29/30 nesta sessão.

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| RF estrutural — F1m agregado | 0.7534 | 📄 `28` relatorio | ✅ |
| SAGE estrutural — F1m agregado | 0.8138 | 📄 `28` relatorio | ✅ |
| Fit lognormal (KS) | rejeitado nas 3 métricas (p < 1e-11) | 📄 `28` relatorio | ✅ |
| Profundidade — depth=2 | 2683 grafos (70.1%) | 📄 `28` relatorio | ✅ |
| Profundidade — depth≥3 | 1143 grafos (29.9%) | 📄 `28` relatorio | ✅ |
| Outlier `fake_contido` | N=31, SAGE acc = **0.000** | 📄 `29` relatorio / 📊 T12 | ✅ |
| Outlier `real_viral` | N=60, SAGE acc = 0.433 | 📄 `29` relatorio / 📊 T12 | ✅ |
| Outlier `cauda_alta_n` | N=192, SAGE 0.823 vs RF 0.547 | 📊 T12 | ✅ |
| Outlier `deep_extreme` (depth≥5) | N=256, SAGE 0.762 vs RF 0.531 | 📊 T12 | ✅ |
| Gap depth≥1 (todos) | +0.060, IC95 [+0.047, +0.074] | 📊 T13 | ✅ |
| Gap depth≥3 | +0.134, IC95 [+0.103, +0.165] | 📊 T13 | ✅ |
| Gap depth≥5 | +0.213, IC95 [+0.139, +0.287] | 📊 T13 | ✅ |
| Razão gap(depth≥5)/gap(agregado) | 3.5× | 📄 `30` relatorio | ✅ |

## §4.4-bis RQ3 multilíngue (T11, F14)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| PT vs EN — Cohen's d | −0.057 (desprezível) | 📊 T11 | ✅ |
| DE vs EN — Cohen's d | −0.468 (médio) | 📊 T11 | ✅ |
| PT vs DE — Cohen's d | +0.739 (médio-grande) | 📊 T11 | ✅ |
| N por idioma | EN 132.312 / DE 9.686 / PT 2.661 | 📊 T11 | ✅ |

## §4 — significância FNN posfull (T2 — atualmente sem seção)

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| GCN / GAT / SAGE no FNN posfull | 0.8608 / 0.8632 / 0.8612 — todos ns entre si | 📊 T2 | ✅ |

---

# Cap 5 — Aplicação

| Fato | Valor | Fonte | Status |
|---|---|---|---|
| Posts pontuados pelo RF estrutural | **168.463** (todos) | `scores_por_post.csv` | ✅ |
| Posts amostrados na concordância | **3.728** (não 5k apesar do nome do arquivo) | `amostra_5k.csv` / soma T9 | ✅ (D3 resolvido — Apêndice E usa 3.728) |
| Threads visualizadas | 6 | `threads_bluesky/resumo.txt` | ✅ |
| Feed mais "fake-like" | Political Science: score 0.6663, 69.2% pred fake, N=357 | 📊 T8 | ✅ |
| Feed menos "fake-like" | News: score 0.2704, 6.7% pred fake, N=42112 | 📊 T8 | ✅ |
| Maior feed | Blacksky: N=86490, score 0.4391 | 📊 T8 | ✅ |
| Concordância — posts virais [20,100) | 74.0% | 📊 T9 | ✅ |
| Concordância — posts médios [2,20) | 42–44% | 📊 T9 | ✅ |
| Bloco vs Mini — FNN | MINI 0.4764 vs BLOCO 0.4743 (Δ −0.0021) | 📊 T7 | ✅ |
| Bloco vs Mini — GossipCop | MINI 0.7534 vs BLOCO 0.7486 (Δ −0.0048) | 📊 T7 | ✅ |
| Bloco vs Mini — PolitiFact | MINI 0.5338 vs BLOCO 0.5565 (Δ +0.0227) | 📊 T7 | ✅ |

---

# Cap 6 — Conclusão

Capítulo de síntese, sem fatos novos. As 7 limitações:

| # | Limitação | Status |
|---|---|---|
| a | FNN com cascatas estrela (vale só pro FNN nosso, NÃO pro UPFD oficial — ver D4) | ⚠️ |
| b | Regra dual 0.8/0.2 derivada de 1 experimento | ✅ |
| c | Bluesky sem labels = só demonstração | ✅ |
| d | Reprodutibilidade não bit-exact | ✅ |
| e | Sem comparação numérica direta com BiGCN/GCNFN | ✅ |
| f | Bluesky é snapshot temporal, não amostra probabilística | ✅ |
| g | Ambiente CPU-only | ✅ |

---

# ⚠️ Discrepâncias conhecidas — resolver antes de finalizar

### D1 — SAGE-PolitiFact sem texto: 0.55 vs 0.33

- **Abstract** (`main.tex`) e scaffolding §4.4 dizem: PolitiFact "colapsa, F1 ≈ 0,55".
- **T4** (tabela gerada do CSV) diz: SAGE-PolitiFact sem texto = **0.3314**.
- O 0.55 provavelmente é confusão com o FNN (SAGE-FNN = 0.5362).
- **Verdade:** 0.33. O colapso é mais severo do que o texto afirma — o modelo cai **abaixo da chance** (0.50). Corrigir o abstract e qualquer menção a "0.55" para PolitiFact.

### D2 — "Cohen's d < 0.2 em todas as métricas" do PolitiFact é falso

- **Abstract** diz: PolitiFact tem "|d| < 0,2 em todas as métricas avaliadas".
- **T5** diz: PolitiFact `num_nodes` d = **−0.245** (efeito pequeno, |d| > 0.2). FNN também tem |d| = 0.223 nas duas métricas.
- **Verdade:** o correto é "|d| < 0,5" (nenhum efeito médio) ou "|d| ≤ 0,25". A tese central (precisa d ≥ 0.5) **continua válida** — PolitiFact não chega a 0.5 — mas a frase "< 0,2" está factualmente errada. Corrigir abstract + §4.5.

### D3 — Amostra da concordância Bluesky: "5k" mas é 3.728

- Arquivo `amostra_5k.csv` e caption de T9 dizem "5k amostrado".
- Soma real das linhas / dos N de T9 = **3.728**.
- **Verdade:** 3.728. **Status (2026-05-25):** Corrigido — o Cap 5 deixou de existir (virou Apêndice E `apendices/E_aplicacao_bluesky.tex`, §sec:ap_bsky_concordancia) e o texto do apêndice usa 3.728. T9 não é mais referenciada na compilação.

### D4 — "FNN com cascatas estrela" como limitação universal

- §6.3.a do scaffolding implica que cascata estrela é problema geral.
- Fase 9 (script 30) mostrou: UPFD-GossipCop oficial tem 30% dos grafos com depth ≥ 3.
- **Verdade:** a crítica de "estrela plana" vale só pro FNN construído pelo script 00. UPFD oficial NÃO é estrela. Refinar a redação de §6.3.a.

### D5 — Citação `quelle2024bluesky` (JÁ CORRIGIDA)

- Era `@misc{quelle2024bluesky}` com título e URL errados (apontava pra paper de driving data synthesis).
- Corrigido para `@article{failla2024bluesky}` (Failla & Rossetti, PLOS ONE 2024). Substituído em todos os `.tex`.
- **Pendência:** `sibila2025bluesky` ainda não existe no `.bib` — criar quando publicar os artefatos no HF Hub.

### D6 — O cap de 100 nós do FNN é truncamento agressivo, não trimming de outliers

- Intuição inicial: o cap removeria "alguns grafos gigantes outliers".
- Verificação matemática (CSVs brutos): a distribuição de tamanho de cascata
  é heavy-tail extremo (mediana 30, média 552, máximo 29.060 tweets). O cap de
  100 trunca **50% dos grafos** e remove **91,7% da massa de nós-filho**.
- **Consequência:** os 377 grafos truncados ficam indistinguíveis por tamanho
  (todos `num_nodes=100`, mesmo `grau_norm` na raiz).
- **Ação:** a justificativa do cap em §3.3 deve declarar a magnitude real
  (50% truncados) e o motivo verdadeiro. Não descrever como "remoção de
  outliers" — isso subdimensiona o efeito e a banca pega. Enquadrar como
  decisão de projeto: em grafo-estrela os filhos têm features idênticas
  (BERT replicado), então truncar leaves redundantes é defensável — mas
  precisa ser dito explicitamente, não escondido.

---

# Pendências de bibliografia

Citações usadas no scaffolding mas **ainda ausentes** do `Referencias.bib`:

| Chave | Para que | Onde |
|---|---|---|
| `cleveland1979lowess` | LOWESS | §3.8 |
| `efron1993bootstrap` | Bootstrap pareado | §3.8 |
| `sibila2025bluesky` | Artefatos publicados no HF Hub | §3.2, §5 |

Buscar e adicionar na varredura final de bibliografia.
