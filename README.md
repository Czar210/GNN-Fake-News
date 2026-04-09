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