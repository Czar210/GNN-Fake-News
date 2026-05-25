# Detecção de Fake News com Redes Neurais de Grafos

Repositório do Trabalho de Conclusão de Curso (TCC). Investiga sob quais
condições a **topologia da propagação** de uma notícia em redes sociais
pode complementar — ou substituir — a análise textual na detecção de
*fake news*.

## 📖 Sobre o Projeto

Detecção de *fake news* é tradicionalmente abordada por NLP. Essa estratégia
perde robustez quando (i) LLMs geram texto gramaticalmente impecável e
(ii) o post está num idioma sem encoder específico. Este trabalho responde
**três perguntas de pesquisa**:

- **RQ1** — Topologia carrega sinal acima do texto da raiz da árvore de propagação?
- **RQ2** — Sob quais condições estruturais?
- **RQ3** — É viável detectar *fake news* **só** com topologia?

A resposta é bilateral. **Positivo**: GraphSAGE com 2 *features* estruturais
por nó (sem texto) atinge **F1=0.81** em UPFD-GossipCop. **Negativo**: a
mesma arquitetura colapsa em UPFD-PolitiFact (F1≈0.55). A diferença é
explicada por Cohen's d entre fake e real: +1.53 em GossipCop, <0.2 em
PolitiFact — base da **tese central falsificável** de que modelos
topológicos exigem $|d| \geq 0.5$ pra funcionar.

A **Fase 9** (rodada complementar de 2026-05) refinou o achado: o ganho
do GNN sobre o baseline tabular **triplica em cascatas profundas** (gap
+0.06 agregado → +0.21 em depth≥5, com IC bootstrap que não cruza zero).
Detalhes em [`HISTORIA_FASE9.md`](HISTORIA_FASE9.md) e [`MUDANCAS_FASE9.md`](MUDANCAS_FASE9.md).

---

## 🏗️ Estrutura do Repositório

```
GNN Fake News/
├── Training/                    # Pipelines de pesquisa
│   ├── 01_BlueSky_Pipe/         # coleta + grafos Bluesky
│   ├── 02_UPFD_Benchmark/       # experimentos preliminares UPFD
│   └── 03_Mega_Research/        # pipeline principal (30 scripts numerados)
│       ├── PIPELINE.md          # documentação canônica de cada script
│       └── 00..30_*.py          # ver PIPELINE.md
├── Execution/                   # Saídas (gitignored, regeneráveis)
│   ├── results/                 # relatórios + CSVs + figuras
│   │   ├── fase2_baselines/     # baseline textual + confound
│   │   ├── fase3_ablation/      # ablation positional encodings
│   │   ├── fase4_benchmarks/    # benchmarks finais + significância
│   │   └── figuras_tcc/         # figuras consolidadas pro TCC + Fase 9
│   ├── weights/                 # modelos persistidos (RF, LogReg, SAGE)
│   └── scripts/                 # utilitários ad-hoc
├── Interface/                   # Ferramenta web
│   ├── frontend/                # Next.js + FastAPI (regra dual)
│   └── supabase/                # backend Supabase
├── Material/                    # Datasets + Overleaf
│   ├── politifact/, gossipcop/  # UPFD raw + processed
│   ├── LIAR/                    # dataset auxiliar
│   ├── upfd_raw/                # dados originais
│   └── GNN_TCC_atualizado/      # projeto LaTeX completo
├── dados_bluesky/               # 6 GB do Bluesky (gitignored)
├── Photos/                      # figuras, screenshots, pacotes
│   ├── orientador_fase9/        # 8 figs + LEIAME pra apresentar
│   ├── pacote_overleaf_fase9.zip
│   └── pacote_tcc_base_completa.zip
├── Tests/                       # testes do código
├── HISTORIA_FASE9.md            # origem → execução → resultados Fase 9
├── MUDANCAS_FASE9.md            # diff de alto nível pra colegas
└── docker-compose.yml + Dockerfiles
```

**Pipeline principal** (`Training/03_Mega_Research/`): 30 scripts numerados.
A documentação canônica de cada um — objetivo, matéria-prima, produto,
porquê — está em [`PIPELINE.md`](Training/03_Mega_Research/PIPELINE.md),
organizada por **fase experimental** (não por ordem numérica).

---

## 🧠 Arquitetura do Modelo

Três modelos persistidos em [`Execution/weights/`](Execution/weights/),
gerados pelo [`script 17`](Training/03_Mega_Research/17_persistir_modelos_finais.py):

| Modelo | Tipo | Input | F1m (UPFD-GossipCop test) |
|---|---|---|---|
| **LogReg-BERT** | Regressão logística | BERT do título (768 dim) | 0.86 (FNN) |
| **RF tabular** | Random Forest | `[num_nodes, grau_root]` (2 dim) | 0.753 |
| **SAGE estrutural** | GraphSAGE (3 camadas) | `[is_root, grau_norm]` (2 dim) + grafo | **0.814** |

Os três cobrem três regimes — textual, estrutural leve, estrutural pesado —
e são combinados via **regra dual *post-hoc*** na ferramenta web (Cap 5).

Stack:
- `torch`, `torch_geometric` para as GNNs (GCN, GAT, SAGE)
- `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`, 768 dim) pro BERT
- `sklearn` pros baselines não-GNN e o RF tabular
- `scipy.stats` (`ttest_rel`) + bootstrap pareado pra significância

---

## 📊 Estado Atual do TCC

**Pipeline experimental: completo** (Fases 1-9 rodadas, todos os artefatos
em [`Execution/results/`](Execution/results/)).

**Redação LaTeX: scaffolded, sem prosa final.** Os 6 capítulos em
[`Material/GNN_TCC_atualizado/capitulos/`](Material/GNN_TCC_atualizado/capitulos/)
existem com headers, figuras/tabelas no lugar certo, e comentários `% TESE:`,
`% ANTES:`, `% DEPOIS:` e `% TRANSICAO:` guiando o que escrever.

**Achados centrais (UPFD-GossipCop test, N=3826):**

| Achado | Onde | Valor |
|---|---|---|
| LogReg-BERT no FNN — "teto textual" | T1, §4.1 | F1=0.86 |
| RF(num_nodes) sozinho — confound | §4.2 | F1=0.52 (abaixo do gate 0.65) |
| SAGE sem texto — *smoking gun* | T4, F18, §4.4 | F1=0.81 |
| Cohen's d branching GossipCop vs PolitiFact | T5, F17, §4.5 | +1.53 vs <0.2 |
| Concordância → F1, discordância → inversão | T6, §4.6 | 0.97 / 0.08 |
| **(Fase 9)** Inversão confirmada com N | T12, F29, §4.8.3 | SAGE acc=**0.000** em fake_contido (N=31) |
| **(Fase 9)** Gap multi-hop com IC | T13, F32, §4.8.4 | +0.06 → +0.21 em depth≥5 |
| RQ3 multilíngue (PT vs EN) | T11, F14, §4.4-bis | d=−0.057 (desprezível) |

**Pacotes prontos pra escrita** (em [`Photos/`](Photos/)):
- [`pacote_overleaf_fase9.zip`](Photos/pacote_overleaf_fase9.zip) — só os arquivos novos/modificados pela Fase 9 (12 arquivos, 384 KB)
- [`pacote_tcc_base_completa.zip`](Photos/pacote_tcc_base_completa.zip) — TCC inteiro scaffolded (86 arquivos, 7.3 MB) — usar pra reescrever do zero

---

## 🚀 Como Executar

### 1. Ambiente

```bash
pip install -r requirements.txt
```

> Recomendado em `venv` ou `conda`. PyTorch e PyG têm wheels específicos —
> ver seção "Instalando PyTorch + PyTorch Geometric" abaixo.

### 1.b Instalando PyTorch + PyTorch Geometric

O `requirements.txt` lista `torch==2.10.0` e `torch-geometric==2.7.0`, mas a
combinação certa depende do seu OS e da presença de GPU/CUDA. **Não confie
no `pip install` cego pro PyG** — eles distribuem *wheels* pré-compilados
específicos.

- **CPU-only (recomendado para reproduzir o TCC):**
  ```bash
  pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
  pip install torch-geometric==2.7.0
  ```

- **CUDA (se vc tem GPU NVIDIA):** consulte a [matriz oficial de instalação do PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) — eles geram o comando exato baseado no seu CUDA toolkit.

Os experimentos do TCC rodaram em CPU; tempos em "Reproduzindo o experimento" assumem CPU.

### 2. Dados

Nenhum dataset é versionado (~10 GB total). Cada bloco abaixo descreve onde
baixar e onde colocar.

#### 2.a FakeNewsNet PolitiFact (~3 MB CSV + 150 MB .pt)

**Origem:** [github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).

```bash
cd Training/03_Mega_Research
python 00_construir_grafos_fakenewsnet.py --feature-variant full --output-suffix posfull --cpu
python gerar_folds.py
```

Baixa CSV, gera BERT (cache em `data/_bert_titulos_cache.pt`), constrói
grafos com *features* posicionais e salva em
`data/fakenewsnet_posfull/`. Pras variantes do *ablation*:

```bash
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-min  --output-suffix posmin  --cpu
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-grau --output-suffix posgrau --cpu
```

#### 2.b UPFD oficial (PolitiFact + GossipCop) — Google Drive

**Tamanho:** PolitiFact ~200 MB + GossipCop ~1.4 GB.

**Origem:** *"User Preference-aware Fake News Detection"* (Dou et al., SIGIR 2021).
IDs ficam *hardcoded* no fonte do PyG e mudam entre versões — sempre conferir
contra a doc atual.

**Fonte autoritativa:** [pytorch_geometric.datasets.UPFD source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/upfd.html).

**IDs válidos (confirmados em 2026):**

| Dataset | Drive ID |
|---|---|
| `politifact` | `1toou2GO0agoY_OS54LaCWEECQfe93nuq` |
| `gossipcop`  | `1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g` |

**Download automatizado via PyG:**

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

> ⚠️ **GossipCop com `feature='bert'` precisa de ~1.8 GB de RAM**. Em CPU/Windows
> pode estourar OOM. Use `feature='content'` (310d) que é mais leve e equivalente
> pros experimentos topológicos.

#### 2.c Bluesky — dataset social acadêmico (~6 GB)

**Conteúdo:** 168 mil posts em 11 *feeds* temáticos (Blacksky, News, Science,
Political Science, etc.). **Sem labels fake/real** — usado **apenas no Cap 5
(aplicação)**, nunca como validação supervisionada.

**Origem:** dataset acadêmico do Bluesky (paper de pesquisa social, buscar
"Bluesky social network dataset feeds 2024" no Hugging Face/Zenodo/OSF).

Estrutura esperada:
```
dados_bluesky/
├── feed_posts/, feed_posts_likes/, feed_bookmarks.csv
├── followers.csv.gz, interactions.csv.gz
├── graphs.tar.gz, graphs_extracted/graphs/
└── scripts/
```

#### 2.d Pipeline legacy (Bluesky CSVs antigos)

A pasta `Training/01_BlueSky_Pipe/data/raw/` continha CSVs coletados via
`atproto` que ficaram **desalinhados** (~35 IDs em comum entre posts e
reposts). Preservados apenas para referência histórica.

---

## 🧪 Reproduzindo o experimento completo

Sequência mínima para regenerar **todos** os resultados do TCC após baixar os
dados. Ordem segue as Fases do PIPELINE.md.

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

# Fase 5 — Núcleo do TCC: vulnerabilidade topológica
python 14_topologia_sem_texto.py --cpu --seeds 10   # ACHADO CENTRAL
python 15_analise_estrutural.py
python 16_gnn_explainer_upfd.py --cpu

# Fase 6 — Modelos finais + Bluesky
python 17_persistir_modelos_finais.py              # gera os 3 modelos persistidos
python 18_analise_bluesky_crossfeed.py
python 19_aplicar_modelo_bluesky.py
python 22_concordancia_bluesky.py
python 23_visualizar_threads_bluesky.py

# Fase 7 — Comparações finais + RQ3
python 20_textual_vs_topologico.py
python 21_bloco_vs_mini.py
python 26_rq3_multilingual.py

# Fase 8 — Consolidação pro TCC
python 24_consolidar_tcc.py                        # gera T1-T11 LaTeX + INDICE
python 27_figuras_comparativas.py                  # gera F16-F24
python 25_publicar_huggingface.py                  # publica no HF Hub (opcional)

# Fase 9 — Análise estratificada (rodada complementar)
python 28_distribuicao_e_estratificacao.py         # LOWESS + fit + por_grafo.csv
python 29_outliers.py                              # 6 subgrupos extremos
python 30_cascatas_profundas.py                    # bootstrap CI + visualização
```

Tempo total estimado em CPU: **~3-4 horas** (a maior parte é BERT
*embeddando* textos; cache em runs subsequentes acelera).

---

## 📐 Reprodutibilidade — caveats honestos

Os experimentos usam `random_state=42` em todo lugar (`StratifiedKFold`,
RF/LogReg do scikit-learn, `torch.manual_seed`/`np.random.seed`/`random.seed`
nos loops dos GNNs). Os resultados reportados em cada `relatorio.txt` **são
reprodutíveis no mesmo ambiente** (mesmo OS, mesma versão exata de
torch/PyG/scikit-learn — daí o `requirements.txt` pinado).

**O que NÃO é garantido:**

- **Bit-exact entre máquinas.** Diferenças de CUDA toolkit, cuDNN, ou apenas
  CPU vs GPU mudam a aritmética de ponto flutuante. Métricas variam
  tipicamente na 3ª casa decimal.
- **Determinismo dentro do PyTorch.** Operações de scatter/gather usadas pelo
  PyG não têm kernel determinístico em GPU (ver [PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html)).
  Não usamos `torch.use_deterministic_algorithms(True)` porque ele lançaria
  erro em várias operações do PyG.
- **Bluesky inferência.** Ordem de leitura dependente do sistema de arquivos;
  os scripts amostram com seed fixa após carregar tudo, então a amostra
  final é determinística desde que todos os arquivos estejam presentes.

**Por que isso é OK academicamente:** todos os achados centrais foram
validados com **10 seeds independentes** (`13_benchmark_upfd_oficial.py`,
`14_topologia_sem_texto.py`) e **k-fold pareado com `scipy.stats.ttest_rel`**
(`09_teste_significancia.py`). A Fase 9 (`30_cascatas_profundas.py`)
adicionou **bootstrap pareado com 2000 reamostragens** pra subsets *post-hoc*.
A variabilidade está reportada lado a lado com a média; conclusões
qualitativas não mudam entre runs.

---

## 📚 Documentos complementares

Mapa dos `.md` na raiz, pra navegação rápida:

| Arquivo | Propósito |
|---|---|
| [`PIPELINE.md`](Training/03_Mega_Research/PIPELINE.md) | Documentação canônica dos 30 scripts (objetivo, matéria-prima, produto, porquê) |
| [`HISTORIA_FASE9.md`](HISTORIA_FASE9.md) | Como a Fase 9 surgiu: origem da ideia → execução → resultados |
| [`MUDANCAS_FASE9.md`](MUDANCAS_FASE9.md) | Diff de alto nível pra mostrar pra colegas / professores |
| [`FATOS_TCC.md`](FATOS_TCC.md) | **Fonte da verdade** — todo fato verificável do TCC por capítulo, com status e discrepâncias conhecidas. Consultar ao escrever prosa |
| [`PROMPT_IA_TCC.md`](PROMPT_IA_TCC.md) | Prompt para orientar uma IA assistente na escrita — apontar no início de cada sessão |
| [`GUIA_COAUTORES_TCC.md`](GUIA_COAUTORES_TCC.md) | Guia para os co-autores: ordem de escrita, divisão de trabalho, o que não fazer |
| [`estrutura_tcc_v2.md`](estrutura_tcc_v2.md) | Outline aprovado da redação, ancorado em artefatos experimentais |
| [`erros_correcao.md`](erros_correcao.md) | Auditoria inicial: 4 erros metodológicos + 13 incoerências |
| [`plano_correcao.md`](plano_correcao.md) | Plano de 5 fases que produziu o estado atual |
| [`plano_execucao.md`](plano_execucao.md) | Spec operacional estilo speckit para cada tarefa das Fases 1-4 |
| [`auditoria_critica_metodologia.md`](auditoria_critica_metodologia.md) | Auditoria adicional da metodologia |
| [`guia_escrita_tcc.md`](guia_escrita_tcc.md) | Orientações de escrita |
| [`guia_desenvolvimento.md`](guia_desenvolvimento.md) | Orientações de desenvolvimento |

**Para apresentar:** [`Photos/orientador_fase9/`](Photos/orientador_fase9/)
tem 8 figuras numeradas em ordem de impacto + LEIAME explicando cada uma.

**Para escrever:** [`Photos/pacote_tcc_base_completa.zip`](Photos/pacote_tcc_base_completa.zip)
contém o TCC inteiro *scaffolded* (6 capítulos + 32 figuras + 13 tabelas +
INDICE + COMO_USAR), pronto pra subir no Overleaf.
