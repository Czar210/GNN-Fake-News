# Guia de Desenvolvimento — Truth GNN Analytics

Esse documento é pra quem pegou esse repositório e quer entender o que cada coisa faz, onde mexer, e como rodar. Sem enrolação.

---

## Visão geral rápida

O projeto detecta fake news no Bluesky usando GNN (Graph Convolutional Network). A ideia central é que não basta olhar o texto de um post — a forma como ele se propaga na rede (quem compartilha, quem responde) também denuncia se é falso ou não.

O sistema funciona assim:
1. Recebe uma URL de post do Bluesky
2. Extrai o texto e as interações via AT Protocol
3. Gera um embedding BERT do texto (768 dimensões)
4. Monta um grafo de propagação (post no centro, interações como nós filhos)
5. Passa por 4 modelos GCN com níveis diferentes de ceticismo
6. Combina os resultados num ensemble ponderado e dá o veredito

---

## Estrutura de pastas

### `Material/`
Tudo que é referência, dataset bruto, ou material da monografia. Artigos em PDF, o dataset LIAR, o PolitiFact processado em .pt, e os arquivos LaTeX da tese (`GNN_TCC_extraido/`). Você não precisa mexer aqui pra rodar o sistema — é material de consulta e reprodução acadêmica.

### `Training/`
Os três pipelines de treinamento, organizados por origem:

- **`01_BlueSky_Pipe/`** — Pipeline original de coleta e processamento do Bluesky. O `main.py` aqui faz login no Bluesky, coleta posts por termo de busca, gera embeddings BERT, e constrói grafos com NetworkX. Os módulos de coleta e embedding ficam em `src/`. Os dados brutos e processados ficam em `data/`.

- **`02_UPFD_Benchmark/`** — Pipeline padrão do UPFD (User Preference-aware Fake News Detection). Um script único (`collect_train_extract_plot.py`) que baixa o dataset PolitiFact via PyTorch Geometric, treina o GCN, e gera visualização t-SNE.

- **`03_Mega_Research/`** — Onde a pesquisa avançada aconteceu. Scripts de treinamento cruzado (Twitter + Bluesky), ablation study, duelo de modelos, e análise de topologia. É aqui que nasceram os 4 modelos do ensemble.

### `Execution/`
Artefatos de execução:

- **`weights/`** — Os 5 arquivos `.pth` (pesos dos modelos treinados). Os 4 do ensemble mais uma variante de escala maior.
- **`results/`** — Matrizes de confusão, análises de vulnerabilidade topológica, e comparações entre modelos. Tudo em PNG.
- **`scripts/`** — Utilitários avulsos: clusterização de grafos, geração de grafos UPFD, inspeção de arquivos .pt.

### `Interface/`
O sistema completo de produção:

- **`frontend/api/`** — Backend em FastAPI. O `main.py` carrega os 4 modelos GCN no startup, expõe endpoints de análise (`/api/analyze`), resultado (`/api/result/{task_id}`), estatísticas (`/api/stats`), e histórico (`/api/history`). Inferência roda em background thread pra não travar a UI. O `database.py` gerencia o SQLite via SQLAlchemy.

- **`frontend/web/`** — Frontend em Next.js 14 + React 18 + Tailwind. Interface estilo glassmorphism. Mostra o resultado da análise com gráfico SVG de propagação, scores de confiança por modelo, e tabela de histórico.

- **`supabase/`** — Migrações SQL para setup de pgvector e função de matching de documentos. Configuração para uso futuro com Supabase cloud.

### `Photos/`
Gráficos e visualizações HTML interativas dos grafos (gerados com Pyvis). Útil pra apresentações e pra monografia.

### `Tests/` e `Flow/`
Ainda vazios. Reservados pra scripts de teste e diagramas de fluxo/linhagem de dados.

---

## Como rodar

### Jeito rápido (Windows)
Dê dois cliques no `iniciar_tcc.bat`. Ele:
- Verifica se Python e Node estão instalados
- Cria o `.venv` e instala dependências se necessário
- Instala `node_modules` se necessário
- Abre dois terminais: backend na porta 8000, frontend na porta 3000

### Manualmente

**Backend:**
```bash
cd Interface/frontend/api
../../../.venv/Scripts/python.exe -m uvicorn main:app --reload
```

**Frontend (em outro terminal):**
```bash
cd Interface/frontend/web
npm install   # só na primeira vez
npm run dev
```

### Com Docker
```bash
docker-compose up --build
```
- API: http://localhost:8000
- Frontend: http://localhost:3001

O Docker não inclui os datasets grandes nem os scripts de treinamento — só o necessário pra rodar a API e o frontend.

---

## Os 4 modelos do ensemble

| Modelo | Arquivo | Peso no ensemble | O que faz |
|--------|---------|-------------------|-----------|
| UPFD Baseline | `pesos_gcn.pth` | 10% | Treinado no Twitter/PolitiFact. Referência. |
| Bluesky Controlado | `pesos_bs_ctrl.pth` | 20% | Treinado em dados controlados do Bluesky. |
| Bluesky Cético | `pesos_bs_cetico.pth` | 30% | Penaliza 6x falsos positivos. Prefere não acusar sem certeza. |
| Bluesky Extra Cético | `pesos_bs_ex_cetico.pth` | 40% | O mais conservador. Tem o maior peso justamente por isso. |

O score final é a média ponderada. Se passar de 0.5, o post é classificado como fake.

---

## Dependências importantes

- **PyTorch 2.9 + PyTorch Geometric 2.7** — O core do modelo. Pesado mas inevitável.
- **sentence-transformers** — Carrega o modelo BERT `paraphrase-multilingual-mpnet-base-v2` (~400MB). Fica em memória após o primeiro carregamento.
- **atproto** — Client da AT Protocol pra acessar o Bluesky.
- **FastAPI + SQLAlchemy** — Stack do backend.
- **Next.js 14 + Tailwind** — Stack do frontend.

Todas as deps Python estão em `requirements.txt` (ambiente completo) e `requirements.api.txt` (só o necessário pra API).

---

## Variáveis de ambiente

Crie um arquivo `.env` na raiz com:
```
BSKY_HANDLE=seu.handle.bsky.social
BSKY_PASSWORD=sua_senha_de_app
```

Sem isso o sistema roda em modo simulação (não extrai dados reais do Bluesky).

---

## Onde mexer pra cada tipo de tarefa

- **Quer treinar um modelo novo?** → `Training/03_Mega_Research/`
- **Quer mudar a lógica de inferência ou os pesos do ensemble?** → `Interface/frontend/api/main.py`
- **Quer alterar a interface?** → `Interface/frontend/web/src/app/page.tsx`
- **Quer adicionar um dataset?** → `Material/`
- **Quer mudar como os dados são coletados do Bluesky?** → `Training/01_BlueSky_Pipe/src/collection/collect.py`
- **Quer mudar como os embeddings são gerados?** → `Training/01_BlueSky_Pipe/src/features/text_embedder.py`

---

## Reproduzindo do zero (pra quem clonou e quer treinar tudo de novo)

**Pré-requisitos:** Python 3.11+, Node 20+, ~8GB de RAM livre, ~5GB de disco. GPU é opcional (funciona com CPU, só demora mais).

### 1. Ambiente
```bash
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.txt
```

### 2. Treinar o modelo base (UPFD/PolitiFact)
O script `02_UPFD_Benchmark/collect_train_extract_plot.py` baixa o dataset PolitiFact automaticamente via PyTorch Geometric e treina do zero:
```bash
cd Training/02_UPFD_Benchmark
python collect_train_extract_plot.py
```
Isso gera o `pesos_gcn.pth` (modelo baseline). Mova pra `Execution/weights/`.

Alternativamente, se já tiver os `.pt` do PolitiFact em `Material/politifact/processed/bert/`:
```bash
cd Training/03_Mega_Research
python treinar_upfd_robusto.py
```

### 3. Coletar dados do Bluesky
Crie o `.env` na raiz com suas credenciais Bluesky, depois:
```bash
cd Training/01_BlueSky_Pipe
python main.py
```
Isso coleta posts, gera embeddings BERT, e salva em `data/`.

### 4. Treinar os modelos Bluesky (ensemble)
Com os dados do passo anterior:
```bash
cd Training/03_Mega_Research
python treinar_mega_dataset.py          # gera pesos_gcn_bluesky_ESCALA_MAIOR.pth
python analise_topologia.py             # treina os modelos cético e extra-cético
```
Os pesos são salvos em `Execution/weights/`. Depois disso o sistema está pronto pra rodar.

### 5. Gerar matrizes de confusão e métricas
```bash
cd Training/03_Mega_Research
python matriz_confusao.py               # matrizes básicas
python matrizes_ablation.py             # ablation study
```
Resultados vão pra `Execution/results/`.

---

## Coisas pra ficar de olho

- O `database.py` usa caminho relativo pro SQLite. Funciona local e no Docker, mas se mudar a estrutura de pastas, confere se o path continua correto.
- O `fakes_testados.json` é append-only. Se rodar muito, ele cresce indefinidamente.
- O dataset massivo (21GB no Hugging Face) não está incluso no repo. Se precisar, baixe separadamente.
- Os `.pth` não são versionados no git (estão no .gitignore). Se clonar o repo do zero, precisa treinar os modelos ou copiar os pesos de outra fonte.
- O dataset massivo do Bluesky (~21GB extraído) não está incluso no repo. Ele está disponível no Hugging Face: https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset — baixe e extraia em `Material/dados_bluesky/` pra reproduzir os experimentos de treinamento cruzado. O `.gitignore` já ignora essa pasta.
