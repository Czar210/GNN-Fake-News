# Detecção de Fake News com Redes Neurais de Grafos (GNNs) no Bluesky

Este repositório contém o código-fonte desenvolvido para o nosso Trabalho de Conclusão de Curso (TCC). O objetivo principal do projeto é aplicar técnicas de *Deep Learning* em grafos para identificar a propagação de desinformação (*Fake News*) em redes sociais, baseando-se nas metodologias do curso CS224W da Universidade de Stanford, com foco especial na rede **Bluesky** via AT Protocol.

## 📖 Sobre o Projeto

A detecção de *fake news* é frequentemente abordada apenas pela análise de texto. Este projeto eleva a análise ao incorporar a **topologia da rede de propagação**. 
O modelo utiliza o *benchmark* **UPFD (User Preference-aware Fake News Detection)** com a base de dados PolitiFact para analisar árvores de propagação de notícias, classificando o grafo inteiro como "Real" ou "Falso".

Além disso, o projeto inova ao aplicar esses modelos em ambientes de "Cold Start" (como a rede Bluesky), criando pipelines de extração de dados e comparando o desempenho entre dados estruturados e engajados (como o Twitter) e o fluxo nativo atual do Bluesky.

## 🏗️ Estrutura do Repositório

O projeto está dividido em três pilares principais de engenharia do TCC:

1. **Engenharia de Dados e Coleta (`Blue Sky/`):**
   - Scripts de extração do *firehose* público do Bluesky via API `atproto`.
   - Processamento focado na LGPD, não salvando dados sensíveis diretamente.
   - Construção de grafos de interação (follow, repost, like) usando `networkx` e análise exploratória de *features*.

2. **Inteligência Artificial e Deep Learning (`UPFD-GCN/` e `mesclagem/`):**
   - **GCN (Graph Convolutional Network):** Arquitetura central construída com `PyTorch Geometric` para capturar relações de comunicação a até 3 saltos.
   - Scripts de *Ablation Study* e "Duelo de Modelos": testes unindo dados de Twitter com Bluesky, gerando *benchmark* (`mesclagem/treinar_mega_dataset.py`, `duelo_modelos_frescos.py`).
   - Avaliação minuciosa de resultados via matriz de confusão e métricas estatísticas.

3. **Aplicação Prática e Frontend (`frontend/`):**
   - **Backend / API:** Server desenvolvido em FastAPI (via `main.py` e SQLAlchemy) para servir análises da IA em tempo real.
   - **Frontend / Interface do Usuário:** Portal Web programado em React e Next.js para visualização clara e dinâmica do processo da detecção de fake news.

## 🧠 Arquitetura do Modelo

A rede é composta por:
- Camadas de convolução em grafos (ex: `GCNConv`) para capturar as relações estruturais do usuário e das notícias.
- Representação semântica via *Embeddings*, transformando os textos em vetores utilizando `sentence-transformers` (NLP).
- Camada de agregação (*Global Mean Pooling*) que transforma os *embeddings* dos nós em um único vetor latente.
- Classificador linear final para a predição binária focada em *True* ou *Fake*.

## 🛠️ Tecnologias e Dependências

O projeto é construído sobre ecossistemas voltados à *Machine Learning* em grafos (GNNs) e processamento em nuvem, garantindo a reprodutibilidade.

**Principais Bibliotecas:**
- **I.A. e Redes Profundas:** `torch`, `torch_geometric`
- **NLP (Processamento Lógico Posterior):** `sentence-transformers`
- **Análise das Estruturas e Redes:** `networkx`
- **Ecossistema Web:** `fastapi` e `react`

## 🚀 Como Executar

### 1. Instalação do Ambiente Base

Recomenda-se a utilização de um ambiente virtual (ex: `venv` ou `conda`). Para instalar as dependências originais relativas ao treinamento da IA, execute:

```bash
pip install -r requirements.txt
```

*(Nota: Para rodar tanto a API do backend quanto a interface visual, recomenda-se verificar e inicializar os runtimes na pasta `frontend/`).*

### Reprodutibilidade — caveats honestos

Os experimentos usam `random_state=42` em todo lugar (`StratifiedKFold`, RF/LogReg do scikit-learn, `torch.manual_seed`/`np.random.seed`/`random.seed` nos loops dos GNNs). Os resultados reportados em cada `relatorio.txt` **são reprodutíveis no mesmo ambiente** (mesmo OS, mesma versão exata de torch/PyG/scikit-learn — daí o `requirements.txt` pinado).

**O que NÃO é garantido:**

- **Bit-exact entre máquinas.** Diferenças de CUDA toolkit, cuDNN, ou apenas CPU vs GPU mudam a aritmética de ponto flutuante. As métricas variam tipicamente na 3ª casa decimal.
- **Determinismo dentro do PyTorch.** Algumas operações de scatter/gather usadas pelo PyG não têm kernel determinístico em GPU (ver [PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html)). Não usamos `torch.use_deterministic_algorithms(True)` porque ele lançaria erro em várias operações do PyG.
- **Bluesky inferência.** O dataset Bluesky (~6 GB) tem ordem de leitura dependente do sistema de arquivos; os scripts amostram com seed fixa após carregar tudo, então a amostra final é determinística desde que todos os arquivos estejam presentes.

**Por que isso é OK academicamente:** todos os achados centrais foram validados com **10 seeds independentes** (`13_benchmark_upfd_oficial.py`, `14_topologia_sem_texto.py`) e **k-fold pareado com `scipy.stats.ttest_rel`** (`09_teste_significancia.py`). A variabilidade de seed (std ≈ 0.002–0.05 dependendo do modelo) está reportada lado a lado com a média; conclusões qualitativas não mudam entre runs.

---

## 📦 Como obter os dados

Nenhum dataset é versionado no repositório (todos juntos somam ~10 GB). Os scripts esperam a estrutura abaixo. Cada bloco descreve onde baixar e onde colocar.

### 1. FakeNewsNet PolitiFact (CSV pequeno, baixado automaticamente)

**Tamanho:** ~3 MB (CSVs) + ~150 MB (após gerar `.pt` com BERT)

**Origem:** [github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) — `dataset/politifact_fake.csv` e `dataset/politifact_real.csv`.

**Como obter:** o script faz tudo. Apenas rode:

```bash
cd Training/03_Mega_Research
python 00_construir_grafos_fakenewsnet.py --feature-variant full --output-suffix posfull --cpu
python gerar_folds.py
```

Isso baixa os CSVs do GitHub, gera embeddings BERT dos títulos (cache em `data/_bert_titulos_cache.pt`), constrói os grafos com features posicionais (`is_root`, `grau_norm`, `pos`) e os salva em `Training/03_Mega_Research/data/fakenewsnet_posfull/`.

Para reproduzir as variantes do experimento de ablation (`pos-min`, `pos-grau`):

```bash
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-min  --output-suffix posmin  --cpu
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-grau --output-suffix posgrau --cpu
```

### 1.b Instalando PyTorch + PyTorch Geometric

O `requirements.txt` lista `torch==2.10.0` e `torch-geometric==2.7.0`, mas a combinação certa depende do seu OS e da presença de GPU/CUDA. **Não confie no `pip install` cego pro PyG** — eles distribuem wheels pré-compilados específicos.

- **CPU-only (recomendado para reproduzir o TCC):**
  ```bash
  pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
  pip install torch-geometric==2.7.0
  ```

- **CUDA (se vc tem GPU NVIDIA):** consulte a [matriz oficial de instalação do PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) — eles geram o comando exato baseado no seu CUDA toolkit.

Os experimentos do TCC rodaram em CPU; tempos reportados em "Reproduzindo o experimento completo" assumem CPU.

### 2. UPFD oficial (PolitiFact + GossipCop) — Google Drive

**Tamanho:** PolitiFact ~200 MB + GossipCop ~1.4 GB (extraídos)

**Origem:** dataset oficial do paper *"User Preference-aware Fake News Detection"* (Dou et al., SIGIR 2021), distribuído via Google Drive. Os IDs ficam hardcoded no fonte do PyG e mudam entre versões — sempre conferir contra a documentação atual.

**Fonte autoritativa:** [pytorch_geometric.datasets.UPFD source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/upfd.html) (busque `file_ids` no código).

**IDs válidos (confirmados em 2026):**

| Dataset | Drive ID | Link de visualização |
|---|---|---|
| `politifact` | `1toou2GO0agoY_OS54LaCWEECQfe93nuq` | [drive.google.com/file/d/1toou…](https://drive.google.com/file/d/1toou2GO0agoY_OS54LaCWEECQfe93nuq/view) |
| `gossipcop` | `1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g` | [drive.google.com/file/d/1DkMA…](https://drive.google.com/file/d/1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g/view) |

**Como obter (download automatizado via PyG):**

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

Cada zip contém: `node_graph_id.npy`, `graph_labels.npy`, `A.txt`, `train_idx.npy`, `val_idx.npy`, `test_idx.npy`, e os `.npz` para cada feature (`profile`, `spacy`, `bert`, `content`).

Estrutura final esperada:
```
Material/
├── politifact/raw/   (~200 MB, 7 arquivos)
└── gossipcop/raw/    (~1.4 GB, 7 arquivos)
```

Após primeira leitura via `torch_geometric.datasets.UPFD`, o PyG materializa em `Material/<dataset>/processed/<feature>/{train,val,test}.pt`.

> ⚠️ **GossipCop com `feature='bert'` precisa de ~1.8 GB de RAM** para densificar a matriz esparsa. Em CPU/Windows pode estourar OOM. Use `feature='content'` (310d) que é mais leve e dá resultados equivalentes para os experimentos topológicos.

### 3. Bluesky — dataset social acadêmico (~6 GB)

**Tamanho:** ~6 GB extraído (15 GB+ se contar arquivos brutos do dataset original de 31 GB).

**Conteúdo:** posts, reposts, replies, quotes, followers — coletados de 11 feeds temáticos do Bluesky (Blacksky, News, Science, Political Science, etc.). Total de ~168 mil posts, ~63 milhões de reposts, ~87 milhões de replies.

**⚠ Sem labels fake/real.** Esse dataset é usado **apenas no capítulo de aplicação/demonstração** do TCC, nunca para validação supervisionada.

**Origem:** dataset acadêmico do Bluesky (paper de pesquisa social — buscar por "Bluesky social network dataset feeds 2024" no Hugging Face/Zenodo/OSF). O zip original tem ~31 GB; nós usamos um subconjunto de ~6 GB.

Estrutura esperada:
```
dados_bluesky/
├── feed_posts/                  # 11 .jsonl (um por feed)
├── feed_posts_likes/            # likes por feed (csv.gz)
├── feed_bookmarks.csv
├── followers.csv.gz             # rede de seguidores (~491 MB)
├── interactions.csv.gz          # interações (~1 GB)
├── graphs.tar.gz                # 891 MB; contém reposts/replies/quotes/threads
├── graphs_extracted/graphs/     # após extração: ~3.7 GB
└── scripts/                     # scripts originais do paper (data_collection,
                                 #   cleaning&processing, experiments)
```

**Como usar nos scripts:** após colocar a pasta `dados_bluesky/` na raiz do repositório, os scripts `18_analise_bluesky_crossfeed.py`, `19_aplicar_modelo_bluesky.py` e `22_concordancia_bluesky.py` funcionam direto.

### 4. Pipeline legacy (Bluesky CSVs antigos — não recomendado)

A pasta `Training/01_BlueSky_Pipe/data/raw/` continha CSVs antigos (`posts_coletados.csv`, `reposts_coletados.csv`) coletados via `atproto`. Estes CSVs são **desalinhados** (apenas ~35 IDs em comum entre posts e reposts) e **inadequados para treino**. Foram preservados apenas para referência histórica do pipeline; o caminho atual usa o `dados_bluesky/` de pesquisa acadêmica descrito acima.

---

## 🧪 Reproduzindo o experimento completo

Sequência mínima para regenerar todos os resultados do TCC após baixar os dados:

```bash
cd Training/03_Mega_Research

# Fase 2 — baselines + folds compartilhados
python 00_construir_grafos_fakenewsnet.py --feature-variant full --output-suffix posfull --cpu
python gerar_folds.py
python 10_baseline_textual.py
python 11_diagnostico_confound.py

# Fase 3 — ablation positional encodings
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-min  --output-suffix posmin  --cpu
python 00_construir_grafos_fakenewsnet.py --feature-variant pos-grau --output-suffix posgrau --cpu
python 12_ablation_intra_encoding.py --cpu
python 15_analise_estrutural.py

# Fase 4 — benchmarks finais
python 09_teste_significancia.py --cpu --data-suffix posfull   # tabela LaTeX
python 13_benchmark_upfd_oficial.py --cpu --seeds 5             # cross-dataset UPFD
python 14_topologia_sem_texto.py --cpu --seeds 10               # sem texto
python 16_gnn_explainer_upfd.py --cpu                           # explanations

# Persistencia + comparacoes finais
python 17_persistir_modelos_finais.py
python 18_analise_bluesky_crossfeed.py
python 19_aplicar_modelo_bluesky.py
python 20_textual_vs_topologico.py
python 21_bloco_vs_mini.py
python 22_concordancia_bluesky.py
```

Tempo total estimado em CPU sem GPU: ~3-4 horas (a maior parte é BERT embeddando textos; reusa cache em runs subsequentes).