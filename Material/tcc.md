# 🎓 Guia Estrutural e de Navegação do TCC
**Detecção de Fake News via GNN no Bluesky**

Este documento é a "bússola" do seu TCC. Ele conecta o plano estratégico (`TODO.md`) diretamente aos arquivos e pastas do código-fonte. Use este guia para saber **exatamente onde procurar, onde codificar e onde extrair dados** para a monografia.

---

## 🏗️ 1. Engenharia de Dados e Coleta (O Motor do Projeto)

Onde a mágica da aquisição de dados via AT Protocol (Bluesky) acontece.
**📍 Diretório Principal:** `Blue Sky/`

- **Coleta de Dados:** `Blue Sky/src/collection/` (ex: `collect.py`)
  - *Aqui você estuda a API do Bluesky.* Quando for escrever a Metodologia sobre o "Dataset Bluesky", é esse código que você vai documentar.
- **Geração de Features e Análise:** `Blue Sky/src/features/` e `Blue Sky/src/analysis/`
  - *No `TODO.md`:* É nesta etapa que entra a **otimização do Data Lake**. Se for criar a pipeline para Parquet/SQLite e resolver o problema do CSV enorme de 21GB, você modificará ou criará scripts nesta área.

## 🧠 2. Inteligência Artificial e Deep Learning (O Cérebro)

Onde as Redes Neurais em Grafos (GNN) são treinadas, avaliadas e comparadas.
**📍 Diretórios Principais:** `mesclagem/`, `UPFD-GCN/` e Arquivos na Raiz.

*A pasta `mesclagem/` é o coração acadêmico dos seus resultados empíricos.*

- **O Modelo Original (Benchmark):** `UPFD-GCN/src/collect_train_extract_plot.py`
  - Código baseado em dados do Twitter (`UPFD/`, `politifact/`). Representa a linha de base (baseline).
- **Treinamento e Mega Dataset (Ablation Study):** `mesclagem/treinar_mega_dataset.py` e `mesclagem/teste_heatmap_mega_dataset.py`
  - *No `TODO.md`:* Útil para o "Ablation Study" ou comparação com blocos massivos de dados.
- **Experimentação e Duelo de Modelos:** `mesclagem/duelo_modelos_frescos.py` e `mesclagem/teste_fresh_bluesky.py`
  - *No `TODO.md`:* Responde diretamente ao "Duelo de Modelos". Aqui você puxa os dados e descobre o desempenho contra o problema do "Cold Start" no Bluesky.
- **Rigor Científico / Métricas:** `mesclagem/matriz_confusao.py` e `mesclagem/pipeline_mensuracao.py`
  - *No `TODO.md`:* Essencial para o TCC! Daqui sairão as as matrizes de confusão detalhadas e, futuramente, os testes de significância (t-test).
- **Pesos Salvos:** `mesclagem/pesos_gcn.pth` (o modelo treinado em si). E imagens gráficas como `resultado_final_clusters.png` na raiz do projeto (ótimas para colocar nos slides da defesa!).

## 🌐 3. Aplicação Prática e Frontend (A Vitrine)

O portal web para demonstrar sua solução ao vivo.
**📍 Diretório Principal:** `frontend/`

- **Backend / API (Servindo o Modelo):** `frontend/api/main.py` e `frontend/api/database.py`
  - Usa FastAPI e SQLite (`truth_analytics.db`).
  - *No `TODO.md`:* Se for implementar as `Background Tasks` (Celery/FastAPI) para o processamento não travar, é no `main.py` da API que você deve mexer.
- **Frontend / Interface do Usuário:** `frontend/web/src/app/`
  - Interface desenvolvida em React/Next.js (`layout.tsx`, `globals.css`).
  - *No `TODO.md`:* É aqui que você precisará integrar o pacote **`react-force-graph`** para criar a visualização interativa das bolhas (Demo ao Vivo).

## 📝 4. Roteiro Prático para a Escrita da Monografia

Ao redigir seu texto no Word/LaTeX, saiba de onde tirar as informações:

1.  **Fundamentação Teórica (GNNs):** Olhe o script base de treinamento em `UPFD-GCN/` para descrever como os nós e arestas (seguidores/mensagens) se comportam na sua implementação. Se adicionar *GAT* ou *GraphSAGE* (como pede o `TODO.md`), descreva a troca das camadas (layers) na classe do modelo em PyTorch Geometric.
2.  **Metodologia (O seu diferencial):**
    -   *Coleta:* `Blue Sky/src/collection/collect.py`
    -   *Arquitetura do Fluxo de Dados:* Conecte a explicação de como o frontend (`frontend/api/main.py`) conversa com o script de backend em tempo real.
3.  **Resultados e Discussão:**
    -   Extraia as imagens de `resultado_final_clusters.png` e os prints de console do `mesclagem/pipeline_mensuracao.py` para provar que a rede realmente funciona.
4.  **LGPD (LGPD e Privacidade):**
    -   Ao redigir o capítulo de Ética, analise o payload que o `collect.py` puxa do Bluesky. Comprove no texto que nenhum dado privado ou senha é requisitado, apenas posts públicos do firehose/AT Protocol.

---
**💡 Conclusão de Foco:**
Seu projeto está bem modularizado! Toda vez que o orientador cobrar **Resultados**, corra para a pasta `mesclagem/`. Quando cobrar **Otimização de Software**, corra para `Blue Sky/` (pipeline de dados) e `frontend/` (background tasks limitando travamentos).
