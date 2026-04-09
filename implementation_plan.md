# Planejamento de Refatoração: GNN Fake News

Refatorar o repositório `tcc-gnn-fake-news` para uma nova estrutura mais limpa na pasta `GNN Fake News`. Esta refatoração visa organizar o projeto por áreas funcionais, removendo metadados desnecessários e arquivos de "sujeira" (como `CLAUDE.md`).

## Estrutura Proposta para `GNN Fake News`

```text
GNN Fake News/
├── 📚 Material/           # Artigos, PDFs, Datasets (LIAR, Politifact, UPFD)
├── 🧠 Training/           # Módulos de treinamento organizados por origem
│   ├── 01_BlueSky_Pipe/   # Pipeline original e análise Blue Sky
│   ├── 02_UPFD_Benchmark/ # Código de treino padrão UPFD-GCN
│   └── 03_Mega_Research/  # Pesquisa Mega Dataset (antiga "mesclagem")
├── 📉 Execution/          # Pontos de entrada universais e modelos salvos
│   ├── weights/           # Arquivos .pth de pesos dos modelos
│   ├── results/           # Matrizes de confusão, métricas e logs preservados
│   └── scripts/           # Scripts de execução refatorados
├── 🖼️ Photos/             # Gráficos e HTMLs interativos
├── 🧪 Tests/              # Scripts de teste
├── 🌊 Flow/                # Fluxogramas e linhagem de dados
├── 🖥️ Interface/          # Frontend e lógica de API/Database
├── guia_desenvolvimento.md # Guia abrangente de folders, arquivos e desenvolvimento
├── iniciar_tcc.bat         # Launcher Universal Portátil [NEW]
├── implementation_plan.md  # Este documento
└── README.md
```

## Mudanças Propostas

### 🚀 Launcher Universal (`iniciar_tcc.bat`) [NEW]
Reconstruir o script para ser agnóstico ao ambiente:
- **Auto-detecção:** Identifica se Python e Node.js estão instalados.
- **Venv Inteligente:** Verifica se o `.venv` existe; se não, cria e instala `requirements.txt`.
- **Integridade Frontend:** Verifica `node_modules`; se não existir, roda `npm install`.
- **Caminhos Relativos:** Remove caminhos hardcoded para funcionar em qualquer diretório/máquina.

### 🧠 Organização de Treinamentos (`Training/`)
Mover os arquivos mantendo a rastreabilidade da origem:
- **01_BlueSky_Pipe:** `main.py` da raiz e `Blue Sky/src/`.
- **02_UPFD_Benchmark:** Tudo que estava em `UPFD-GCN/src/`.
- **03_Mega_Research:** Tudo que estava em `mesclagem/`.

### 📉 Gestão de Resultados e Logs (`Execution/`)
- **weights/:** Centralizar todos os arquivos `.pth`.
- **results/:** Mover o conteúdo de `mesclagem/resultados/`. 
- **Logs:** Analisei as pastas e os logs de execução (como métricas de treino) serão movidos para esta pasta para manter o histórico sem poluir a raiz.

### 🛠️ Guia de Desenvolvimento
Criar o arquivo `guia_desenvolvimento.md` contendo:
- Explicação detalhada de cada pasta e arquivo.
- Fluxo de desenvolvimento (baseado em `tcc.md` e referências da tese).
- **Guia Docker:** Instruções para usar `Dockerfile.api`, `Dockerfile.web` e `docker-compose.yml`.
- **Guia Database:** Configuração do `truth_analytics.db` (SQLite) e integração Supabase.
- **Guia Dataset:** Instruções para utilizar o dataset do [Hugging Face](https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset).

### 📚 Material
- Mover `Busca de Artigos para TCC.md` e `.pdf`.
- Mover diretórios de datasets: `LIAR/`, `politifact/`, `UPFD/`.
- Mover `dados_bluesky/`.
- Mover documentos de texto: `monografia.md`, `tcc.md`.
- **Excluir:** `CLAUDE.md`, `.git`, `.venv`, e zips temporários (`GNN_TCC.zip`).

### 🧠 Training
- Consolidar modelos GNN.
- Trazer a lógica principal de `UPFD-GCN/src/` e `Blue Sky/src/`.

### 📉 Execution
- `main.py` (ponto de entrada refatorado).
- `clusterizar_grafos.py`, `gerar_grafo_upfd.py`.
- Arquivos de configuração: `requirements.txt`, `Dockerfile`, `docker-compose.yml`.

### 🖼️ Photos
- `resultado_final_clusters.png`.
- Visualizações HTML interativas da pasta `UPFD/`.

### 🖥️ Interface
- Diretório `frontend/`.
- Diretório `supabase/`.

## Perguntas em Aberto

1. **Estrutura Blue Sky:** Devo manter a subpasta `Blue Sky` ou mesclar seu `src` diretamente em `Training`?
2. **Logs de Execução:** Existem arquivos de log específicos (.txt/.log) que você deseja preservar em `Execution/`?
3. **Scripts de Lote:** Deseja manter o script `iniciar_tcc.bat`?

## Plano de Verificação

- Verificar a estrutura de diretórios no explorador de arquivos.
- Garantir que `CLAUDE.md` e outros arquivos de "sujeira" NÃO estejam presentes.
- Validar que todos os scripts de treinamento e datasets foram movidos corretamente.
