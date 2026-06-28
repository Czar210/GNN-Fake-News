# Fontes e Créditos da Apresentação

De onde veio cada coisa que aparece nos slides. Serve para responder a banca quando perguntarem "de quem é essa fórmula?" e "como vocês fizeram esse gráfico?".

> **Tudo aqui está ancorado na fonte oficial do TCC**, não na memória:
> - Hiperparâmetros + refs dos métodos → **Apêndice A** (p. 79–82)
> - Cada figura/tabela → **script gerador** → **Apêndice B** (p. 83–86)
> - Datasets e suas fontes → **Tabela 3.1** (p. 34)
> - Chaves de citação → `Material/GNN_TCC_atualizado/Referencias.bib`

---

## 1. O princípio (duas caixas)

Toda coisa no slide cai em **uma de duas** caixas — e cada uma se credita de um jeito:

| Caixa | O que é | Como creditar |
|---|---|---|
| **(A) De terceiros** | fórmula, arquitetura, método estatístico, dataset que **não** é nosso | citar **autor e ano**: "(KIPF; WELLING, 2017)" |
| **(B) Nosso** | gráfico, tabela, diagrama, score, dataset tratado | "**elaboração própria**" + **nome do script** que gerou |

Regra de ouro: **um gráfico é sempre da caixa (B)** — vocês geraram, ninguém copiou. O que pode ser da caixa (A) é o **método** por trás dele (ex.: o *gráfico* do Cohen's d é nosso; o *Cohen's d* é de Cohen, 1988).

---

## 2. Arquiteturas, fórmulas e métodos → quem creditar

Use o formato ABNT autor-data (igual ao corpo do TCC). Chave do `.bib` na última coluna.

| O que aparece no slide | Crédito (ABNT) | Chave .bib |
|---|---|---|
| **GCN** (convolução espectral) | (KIPF; WELLING, 2017) | `kipf2017gcn` |
| **GAT** (atenção em grafos) | (VELIČKOVIĆ et al., 2018) | `velickovic2018gat` |
| **GraphSAGE** (Sample & Aggregate) | (HAMILTON; YING; LESKOVEC, 2017) | `hamilton2017graphsage` |
| **GNNExplainer** | (YING et al., 2019) | `ying2019gnnexplainer` |
| **Sentence-BERT** (embedding do título) | (REIMERS; GUREVYCH, 2019) | `reimers2019sbert` |
| Modelo **multilingual** (`mpnet-base-v2`) | (REIMERS; GUREVYCH, 2020) | `reimers2020multilingual` |
| **BERT** (base) | (DEVLIN et al., 2019) | `devlin2019bert` |
| **Cohen's *d*** (tamanho de efeito) — *a tese!* | (COHEN, 1988; SAWILOWSKY, 2009) | `cohen1988statistical`, `sawilowsky2009effect` |
| **k-fold** estratificado (k=10) | (KOHAVI, 1995) | `kohavi1995cv` |
| **Bootstrap** pareado (n=2000) | (EFRON; TIBSHIRANI, 1993) | `efron1993bootstrap` |
| **LOWESS** (suavização) | (CLEVELAND, 1979) | `cleveland1979lowess` |
| Comparar classificadores / teste pareado | (DEMŠAR, 2006) | `demsar2006statistical` |
| **Adam** (otimizador) | (KINGMA; BA, 2015) | `kingma2015adam` |
| **Dropout** (0,5) | (SRIVASTAVA et al., 2014) | `srivastava2014dropout` |
| **Early stopping** (patience) | (PRECHELT, 1998) | `prechelt1998early` |
| Oversmoothing (justifica 3 camadas) | (LI; HAN; WU, 2018) | `li2018deeper` |
| Heavy-tail / power-law das cascatas | (CLAUSET; SHALIZI; NEWMAN, 2009) | `clauset2009powerlaw` |
| **PyTorch Geometric** (ferramenta GNN) | (FEY; LENSSEN, 2019) | `fey2019pyg` |
| Origem do paradigma GNN p/ fake news | (MONTI et al., 2019) | `monti2019fakenews` |
| BiGCN (citado nas limitações) | (BIAN et al., 2020) | `bian2020bigcn` |

---

## 3. Datasets → fonte (Tabela 3.1 do TCC, p. 34)

| Dataset no slide | Fonte / crédito | Chave .bib |
|---|---|---|
| **FakeNewsNet PolitiFact** | (SHU et al., 2020) — CSVs do repositório KaiDMML/FakeNewsNet | `shu2020fakenewsnet` |
| **UPFD** (PolitiFact + GossipCop) | (DOU et al., 2021), SIGIR — distribuído via Google Drive | `dou2021upfd` |
| **Bluesky** (snapshot bruto, 168k posts) | (FAILLA; ROSSETTI, 2024), PLOS ONE — Zenodo DOI 10.5281/zenodo.14258401 | `failla2024bluesky` |
| **Bluesky tratado pelos autores** (HF Hub) | (SIBILA; LIVINGSTON; TAKIDA, 2025) — *elaboração própria* | `sibila2025bluesky` |

> No slide dos dados (slide 7): o Bluesky **bruto** é de Failla & Rossetti; o **tratado/publicado no Hugging Face é de vocês**. Deixe os dois claros — é uma contribuição do TCC.

---

## 4. As figuras dos slides → script gerador (Apêndice B do TCC)

Toda figura é **elaboração própria**. O crédito de rodapé nomeia o **script** (pasta `Training/03_Mega_Research/`) e, quando há, o **método** de terceiros por trás.

| Slide | Figura (nº no TCC) | Script gerador | Rodapé sugerido |
|---|---|---|---|
| 10 — achado central | Fig 4.1 / F18 | `14_topo_sem_texto.py` | *Elaboração própria (script 14).* |
| 11 — Cohen's *d* | Fig 4.3 / F17 | `15_cohens_d.py` | *Elaboração própria (script 15). Método: Cohen's d (COHEN, 1988).* |
| 14 — multi-hop | Fig 4.14 / F32 | `30_gap_profundidade.py` | *Elaboração própria (script 30). IC por bootstrap (EFRON; TIBSHIRANI, 1993).* |
| 15 — concordância | Fig 4.4 / F3 | `20_concordancia.py` | *Elaboração própria (script 20). Métrica: Cohen's κ.* |
| 16 — RQ3 idioma | Fig 4.2 / F14 | `19_rq3_multilingual.py` | *Elaboração própria (script 19). Dados: snapshot Bluesky (FAILLA; ROSSETTI, 2024).* |
| 18 — painel mestre | Fig 4.15 / F16 | `24_consolidar_tcc.py` + `27b_repintar_figuras.py` | *Elaboração própria (scripts 24/27b).* |
| 6 — diagramas das arquiteturas | Figs 2.2–2.4 | TikZ (desenho) | *Diagramas: elaboração própria. Arquiteturas: KIPF; WELLING (2017); VELIČKOVIĆ et al. (2018); HAMILTON et al. (2017).* |
| 13/17 — fluxograma | Fig 1.1 / 4.5 (F24) | TikZ (desenho) | *Elaboração própria (TikZ).* |
| 7 — árvore de propagação | Fig 2.1 | visualização de grafos UPFD | *Visualização própria sobre dados UPFD (DOU et al., 2021).* |

> Outras figuras (GNNExplainer F19→script 16; LOWESS F26→script 28; outliers F29→29) seguem o mesmo padrão: ver **Apêndice B, p. 84–86** — ele lista as 35 figuras e as 14 tabelas com o script de cada uma. É a sua "fonte da verdade" de proveniência.

---

## 5. Ferramentas (o "como fizemos") — para o slide de método/rigor

Stack de software, com a configuração exata na **Tabela A.5 (p. 82)**:

- **PyTorch 2.10.0+cpu** + **PyTorch Geometric 2.6.1** (FEY; LENSSEN, 2019) → GCN, GAT, SAGE
- **sentence-transformers / Transformers HF 4.x** → embeddings de texto
- **scikit-learn 1.5.x** → LogReg, Random Forest, `StratifiedKFold`
- **scipy 1.13.x** → `ttest_rel` (teste pareado), bootstrap
- Gráficos: **Python (matplotlib/seaborn)**, gerados pelos scripts numerados — *elaboração própria*

---

## 6. Como pôr isso nos slides (sem poluir)

1. **Rodapé discreto** em cada slide que usa figura/fórmula de fora: fonte ~10–12 pt, cinza, alinhada embaixo. Ex.: *"Elaboração própria (script 15). Método: Cohen's d (COHEN, 1988)."*
2. **Slide de Referências no fim** (depois do "Obrigado", ou como backup): lista só o que foi **citado nos slides** — ~12 a 18 referências, em ABNT, fonte menor. Não precisa colar as 32 do TCC; só as que apareceram.
3. **Datasets e ferramentas**: um selo/linha no slide 7 (dados) e no slide 9 (rigor) já resolve — não precisa repetir em todo slide.
4. **Consistência**: se citar "(COHEN, 1988)" num slide, use o mesmo formato em todos. Bate com o corpo do TCC (ABNT autor-data).

---

## 7. Lista pronta para o slide de Referências final

(as que de fato aparecem nos slides — copie/cole e ajuste a formatação ABNT no editor)

- BIAN, T. et al. Rumor Detection on Social Media with Bi-Directional GCN. **AAAI**, 2020.
- COHEN, J. **Statistical Power Analysis for the Behavioral Sciences**. 2. ed. Lawrence Erlbaum, 1988.
- DOU, Y. et al. User Preference-aware Fake News Detection. **SIGIR**, 2021.
- EFRON, B.; TIBSHIRANI, R. J. **An Introduction to the Bootstrap**. Chapman & Hall/CRC, 1993.
- FAILLA, A.; ROSSETTI, G. "I'm in the Bluesky Tonight": Insights from a Year's Worth of Social Data. **PLOS ONE**, v. 19, n. 11, 2024.
- FEY, M.; LENSSEN, J. E. Fast Graph Representation Learning with PyTorch Geometric. **ICLR Workshop**, 2019.
- HAMILTON, W. L.; YING, R.; LESKOVEC, J. Inductive Representation Learning on Large Graphs. **NeurIPS**, 2017.
- KINGMA, D. P.; BA, J. Adam: A Method for Stochastic Optimization. **ICLR**, 2015.
- KIPF, T. N.; WELLING, M. Semi-Supervised Classification with Graph Convolutional Networks. **ICLR**, 2017.
- KOHAVI, R. A Study of Cross-Validation and Bootstrap. **IJCAI**, 1995.
- MONTI, F. et al. Fake News Detection on Social Media using Geometric Deep Learning. **arXiv:1902.06673**, 2019.
- REIMERS, N.; GUREVYCH, I. Sentence-BERT. **EMNLP**, 2019.
- REIMERS, N.; GUREVYCH, I. Making Monolingual Sentence Embeddings Multilingual. **EMNLP**, 2020.
- SAWILOWSKY, S. S. New Effect Size Rules of Thumb. **JMASM**, v. 8, n. 2, 2009.
- SHU, K. et al. FakeNewsNet. **Big Data**, v. 8, n. 3, 2020.
- SIBILA, C. A.; LIVINGSTON, A. M.; TAKIDA, E. K. **Bluesky Fake News Dataset**. Hugging Face Datasets, 2025.
- VELIČKOVIĆ, P. et al. Graph Attention Networks. **ICLR**, 2018.
- YING, Z. et al. GNNExplainer: Generating Explanations for GNNs. **NeurIPS**, 2019.
