# Detecção de Fake News com Redes Neurais de Grafos

Repositório do TCC **"Detecção de Fake News com Redes Neurais de Grafos: Ablation Sistemática e Diagnóstico de Aplicabilidade Topológica"** — César Augusto Sibila, André Messina Livingston, Enzo Kikuji Takida (Ciência de Dados, PUC-SP, 2026).

## 📖 Sobre o trabalho

Investigamos sob quais condições a **topologia da propagação** de uma notícia em rede social pode complementar ou substituir a análise textual na detecção de *fake news*. Três perguntas de pesquisa organizam o trabalho:

- **RQ1** — Topologia carrega sinal acima do texto da raiz da árvore de propagação (BERT)?
- **RQ2** — Sob quais condições estruturais?
- **RQ3** — É viável detectar *fake news* **só** com topologia, em cenários sem texto explorável (multilíngue, deletado, informal)?

A resposta é **bilateral** e essa é a contribuição central:

- **Positivo**: GraphSAGE com 2 *features* estruturais por nó (sem texto) atinge **F1 = 0,810** em UPFD-GossipCop (10 *seeds*, σ=0,002).
- **Negativo**: a mesma arquitetura colapsa em UPFD-PolitiFact (F1 ≈ 0,33, abaixo da chance).

A diferença é explicada pelo Cohen's $d$ entre fake e real: $+1{,}53$ em *branching factor* no GossipCop, $|d| < 0{,}5$ em todas as métricas do PolitiFact. Base da **tese central falsificável**: modelos topológicos exigem $|d| \geq 0{,}5$ em pelo menos uma métrica de cascata para serem efetivos.

**Estado em 2026-06-04**: pipeline experimental completo (Fases 1–9), TCC redigido por inteiro (97 páginas, 76 referências), apêndices A–C+E preenchidos, pré-textuais ABNT incluídos (folha de aprovação, lista de figuras/tabelas/siglas).

## 🏗️ Estrutura do repositório

```
GNN Fake News/
├── Training/
│   ├── 01_BlueSky_Pipe/         # coleta + grafos Bluesky (legado)
│   ├── 02_UPFD_Benchmark/       # experimentos preliminares (legado)
│   └── 03_Mega_Research/        # pipeline principal — 30+ scripts numerados
├── Execution/                   # artefatos gerados (gitignored, regeneráveis)
│   ├── results/                 # CSVs, relatórios, figuras por fase
│   ├── weights/                 # modelos persistidos (LogReg, RF, SAGE)
│   └── scripts/                 # utilitários ad-hoc
├── Material/
│   ├── GNN_TCC_atualizado/      # CANON do TCC (LaTeX + .bib + figuras)
│   ├── politifact/, gossipcop/  # UPFD raw + processed
│   ├── politifact_pyg/, gossipcop_pyg/
│   ├── LIAR/, upfd_raw/         # datasets auxiliares
├── Interface/                   # ferramenta web demonstrativa
│   ├── frontend/                # Next.js + FastAPI (regra dual)
│   └── supabase/                # backend opcional
├── dados_bluesky/               # ~6 GB do Bluesky (gitignored)
├── Photos/orientador_fase9/     # material que foi apresentado ao orientador
├── Tests/                       # testes do código
├── FATOS_TCC.md                 # fonte da verdade — todo número do TCC ancorado
├── PENDENCIAS_TCC.md            # decisões em aberto e mudanças planejadas
├── REVISAO_CARA_DE_IA.md        # auditoria adversarial dos .tex
├── PROMPT_IA_TCC.md             # prompt pra orientar sessões de IA assistente
└── GUIA_COAUTORES_TCC.md        # divisão de trabalho entre coautores
```

A documentação canônica dos scripts está em [Training/03_Mega_Research/PIPELINE.md](Training/03_Mega_Research/PIPELINE.md).

## 🧠 Modelos

Três classificadores persistidos em `Execution/weights/`, gerados pelo [script 17](Training/03_Mega_Research/17_persistir_modelos_finais.py):

| Modelo | Tipo | Input | F1-macro (UPFD-GossipCop test) |
|---|---|---|---|
| LogReg-BERT | Regressão logística | BERT título (768 dim) | 0,86 (FNN) / 0,95 (GossipCop) |
| RF estrutural | Random Forest | `[num_nodes, grau_root]` (2 dim) | 0,753 |
| **SAGE estrutural** | GraphSAGE (3 camadas) | `[is_root, grau_norm]` (2 dim) + grafo | **0,814** |

A combinação dos três em produção segue a **regra dual** derivada *post-hoc* em §4.6 do TCC: pesos $0{,}5/0{,}5$ quando textual e topológico concordam; $0{,}8/0{,}2$ (a favor do textual) quando discordam. O dataset Bluesky processado pelos autores está publicado em [Hugging Face](https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset).

**Stack:**
- `torch 2.10.0+cpu`, `torch_geometric 2.6.1` para as GNNs (GCN, GAT, SAGE)
- `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`, 768 dim) para *embeddings* textuais
- `scikit-learn` para os *baselines* não-GNN e o RF estrutural
- `scipy.stats` (`ttest_rel`) + bootstrap pareado para significância estatística

## 📊 Achados centrais

Validados em UPFD-GossipCop *test set*, $N = 3.826$ grafos:

| Achado | Onde no TCC | Valor |
|---|---|---|
| LogReg-BERT no FNN — "teto textual" | T1, §4.1 | F1 $= 0{,}86$ |
| RF(`num_nodes`) sozinho — *confound* | §4.2 | F1 $= 0{,}52$ (abaixo do *gate* 0,65) |
| SAGE sem texto — **achado central** | T4, F18, §4.4 | F1 $= 0{,}810 \pm 0{,}002$ (10 *seeds*) |
| Cohen's $d$ *branching* GossipCop vs PolitiFact | T5, F17, §4.5 | $+1{,}53$ vs máx. $0{,}25$ |
| Concordância textual×topológico → F1; discordância → inversão | T6, §4.6 | $0{,}97$ vs $0{,}085$ |
| **Inversão distributiva** confirmada com $N$ | T12, F29, §4.8.3 | SAGE acc $= 0{,}000$ em `fake_contido` ($N = 31$) |
| **Gap multi-hop** com IC bootstrap | T13, F32, §4.8.4 | $+0{,}06 \rightarrow +0{,}21$ em depth $\geq 5$ |
| RQ3 multilíngue (PT vs EN) | T11, F14, §4.5 | $d = -0{,}057$ (desprezível) |

## 🚀 Como executar

### 1. Ambiente

```bash
pip install -r requirements.txt
```

Recomendado em `venv` ou `conda`. PyTorch e PyG têm *wheels* específicos — ver seção abaixo.

### 1b. PyTorch + PyTorch Geometric

```bash
# CPU-only (reproduz o TCC)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.6.1
```

Para GPU, ver a [matriz oficial do PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2. Datasets

Nenhum dataset é versionado (~10 GB total). Cada bloco descreve origem e destino.

**FakeNewsNet PolitiFact** (~3 MB CSV + 150 MB `.pt`) — origem [github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet):

```bash
cd Training/03_Mega_Research
python 00_construir_grafos_fakenewsnet.py --feature-variant full --output-suffix posfull --cpu
python gerar_folds.py
```

**UPFD oficial** (PolitiFact ~200 MB + GossipCop ~1,4 GB) — origem Dou et al. (SIGIR 2021), distribuído via Google Drive. IDs confirmados em 2026:

| Dataset | Drive ID |
|---|---|
| `politifact` | `1toou2GO0agoY_OS54LaCWEECQfe93nuq` |
| `gossipcop`  | `1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g` |

Download via PyG:

```bash
python -c "
from torch_geometric.data import download_google_url, extract_zip
import os
ids = {
    'politifact': '1toou2GO0agoY_OS54LaCWEECQfe93nuq',
    'gossipcop':  '1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g',
}
for nome, gid in ids.items():
    raw_dir = f'Material/{nome}/raw'
    os.makedirs(raw_dir, exist_ok=True)
    p = download_google_url(gid, raw_dir, 'data.zip')
    extract_zip(p, raw_dir); os.remove(p)
"
```

> ⚠️ GossipCop com `feature='bert'` (768 dim) precisa de ~1,8 GB de RAM. Em CPU/Windows pode estourar OOM — usar `feature='content'` (310 dim), que é mais leve e equivalente para os experimentos topológicos.

**Bluesky** (~6 GB) — *snapshot* acadêmico de [Failla & Rossetti (PLOS ONE, 2024)](https://doi.org/10.1371/journal.pone.0310330), 168.463 posts em 11 *feeds*. **Sem labels** *fake/real* — usado apenas no Apêndice E (aplicação demonstrativa). O dataset processado pelos autores está publicado em [huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset](https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset).

### 3. Pipeline completo

Sequência mínima para regenerar todos os artefatos do TCC (Tempo total em CPU: ~3-4 horas):

```bash
cd Training/03_Mega_Research

# Fase 0 — Construção de dados
python 00_construir_grafos_fakenewsnet.py --feature-variant full --output-suffix posfull --cpu
python 01_baixar_dataset_hf.py
python 02_construir_grafos_bluesky.py
python gerar_folds.py

# Fase 1 — Arquiteturas (Bluesky)
python 03_treinar_gat.py
python 04_comparar_gcn_gat.py
python 05_treinar_sage.py
python 06_comparar_gcn_gat_sage.py

# Fase 2 — Benchmarks UPFD oficial
python 07_upfd_benchmark_triplo.py
python 13_benchmark_upfd_oficial.py --cpu --seeds 5

# Fase 3 — Generalização cross-dataset
python 08_inferencia_cruzada.py

# Fase 4 — Baselines, confound, ablation
python 09_teste_significancia.py --cpu --data-suffix posfull
python 10_baseline_textual.py
python 11_diagnostico_confound.py
python 12_ablation_intra_encoding.py --cpu

# Fase 5 — Núcleo do TCC
python 14_topologia_sem_texto.py --cpu --seeds 10   # ACHADO CENTRAL
python 15_analise_estrutural.py
python 16_gnn_explainer_upfd.py --cpu

# Fase 6 — Modelos finais + Bluesky
python 17_persistir_modelos_finais.py
python 18_analise_bluesky_crossfeed.py
python 19_aplicar_modelo_bluesky.py
python 22_concordancia_bluesky.py
python 23_visualizar_threads_bluesky.py

# Fase 7 — Comparações finais + RQ3
python 20_textual_vs_topologico.py
python 21_bloco_vs_mini.py
python 26_rq3_multilingual.py

# Fase 8 — Consolidação para o TCC
python 24_consolidar_tcc.py                        # gera tabelas LaTeX + INDICE
python 27_figuras_comparativas.py
python 27b_repintar_figuras_orientador.py          # paleta semântica
python 27c_matrizes_confusao_mono.py
python 25_publicar_huggingface.py                  # publica no HF Hub (opcional)

# Fase 9 — Análise estratificada do erro
python 28_distribuicao_e_estratificacao.py         # LOWESS + fit lognormal
python 29_outliers.py                              # 6 subgrupos extremos
python 30_cascatas_profundas.py                    # bootstrap CI por profundidade
python 31_anotar_figuras_cap4.py                   # opcional: overlay de destaque
```

## 📐 Reprodutibilidade — caveats honestos

Os experimentos usam `random_state=42` em todo lugar (`StratifiedKFold`, RF/LogReg do *scikit-learn*, `torch.manual_seed`/`np.random.seed`/`random.seed` nos loops dos GNNs). Os resultados reportados em cada `relatorio.txt` **são reproduzíveis no mesmo ambiente**.

**O que NÃO é garantido:**

- **Bit-exact entre máquinas.** Diferenças de CUDA, cuDNN ou arquitetura de CPU mudam a aritmética de ponto flutuante. Métricas variam tipicamente na 3ª casa decimal.
- **Determinismo dentro do PyTorch.** Operações de `scatter`/`gather` usadas pelo PyG não têm *kernel* determinístico em GPU (ver [PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html)). Não usamos `torch.use_deterministic_algorithms(True)` porque lançaria erro em várias operações do PyG.

**Por que isso é OK academicamente:** todos os achados centrais foram validados com **10 seeds independentes** ([script 14](Training/03_Mega_Research/14_topologia_sem_texto.py)) e **k-fold pareado com `scipy.stats.ttest_rel`** ([script 09](Training/03_Mega_Research/09_teste_significancia.py)). A Fase 9 ([script 30](Training/03_Mega_Research/30_cascatas_profundas.py)) adicionou **bootstrap pareado com 2.000 reamostragens** para *subsets* *post-hoc*. A variabilidade está reportada lado a lado com a média; conclusões qualitativas não mudam entre execuções.

## 📚 Compilando o TCC

A fonte canônica do TCC está em [`Material/GNN_TCC_atualizado/`](Material/GNN_TCC_atualizado/). A compilação requer LaTeX (MiKTeX, TeX Live ou Overleaf):

```bash
cd Material/GNN_TCC_atualizado
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

O `.bst` (`abntex2-alf.bst`) e um *stub* mínimo de `abntex2.cls` acompanham o repositório para compilação em ambientes sem o pacote oficial `abntex2` instalado. PDF resultante: ~97 páginas.

## 📑 Documentos de controle

| Arquivo | Propósito |
|---|---|
| [FATOS_TCC.md](FATOS_TCC.md) | Fonte da verdade — todo número do TCC ancorado com status (✅/⚠️/❓) |
| [PENDENCIAS_TCC.md](PENDENCIAS_TCC.md) | Decisões em aberto e mudanças planejadas mas não executadas |
| [REVISAO_CARA_DE_IA.md](REVISAO_CARA_DE_IA.md) | Auditoria adversarial dos `.tex` por *workflow* `achar → verificar` |
| [PROMPT_IA_TCC.md](PROMPT_IA_TCC.md) | Prompt para orientar nova sessão de IA assistente |
| [GUIA_COAUTORES_TCC.md](GUIA_COAUTORES_TCC.md) | Divisão de trabalho entre os 3 coautores |
| [Training/03_Mega_Research/PIPELINE.md](Training/03_Mega_Research/PIPELINE.md) | Documentação canônica dos 30+ scripts |

## 📝 Licença

Código sob [LICENSE](LICENSE). O dataset Bluesky processado segue a licença do *snapshot* original de Failla & Rossetti (2024); os pesos dos modelos serão publicados sob CC-BY-4.0 quando subidos ao Hugging Face Hub.
