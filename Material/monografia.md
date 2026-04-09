# 📚 Guia de Redação da Monografia: Mapeamento de Conteúdo e Código

Este guia foi desenhado para te ajudar a escrever a sua monografia (TCC) dividida por capítulos. Para cada seção, indicamos **o que você deve escrever** e **onde no seu código você vai encontrar as provas, diagramas e dados** para embasar o texto.

---

## 📖 Capítulo 1: Introdução

A introdução deve convencer o leitor da relevância do seu trabalho.

*   **O que escrever:**
    *   **Contextualização:** Discuta a "era da pós-verdade" e como a detecção clássica de fake news (baseada apenas no texto/NLP) está saturada e é facilmente burlada pelas IAs gerativas.
    *   **O Problema do "Cold Start":** Mencione como redes novas (como o Bluesky) sofrem para classificar informações logo que nascem, diferente de redes maduras como o Twitter.
    *   **Objetivos:** Seu foco é detectar fake news analisando o comportamento da rede e as bolhas ideológicas (quem mapeia e quem compartilha), ao invés de olhar só para o texto.

*   **Onde buscar inspiração/dados no projeto:**
    *   Leia o seu próprio `README.md` atual para alinhar o tom do discurso proposto inicialmente na criação da arquitetura do projeto.

---

## 🧠 Capítulo 2: Fundamentação Teórica

Aqui você dá a aula de teoria para os avaliadores. Seja didático.

*   **O que escrever:**
    *   **NLP e Embeddings:** Explique como textos (posts) são transformados em números e vetores usando BERT ou Sentence-Transformers.
    *   **Teoria de Grafos em Redes Sociais:** Defina o que são **Nodos** (usuários/bots) e **Arestas** (relações de follow/like/repost) no contexto do Bluesky.
    *   **Deep Learning on Graphs (GNNs):** Faça uma explicação intuitiva de como a convolução em grafos funciona (GCN). Se for implementar GAT (Attention Networks) ou GraphSAGE, explique a diferença teórica de dar "pesos" às arestas.

*   **Onde buscar inspiração/dados no projeto:**
    *   Olhe o arquivo `UPFD-GCN/src/collect_train_extract_plot.py`. Ao explicar a teoria da PyTorch Geometric (PyG), use a classe do modelo (Ex: `class GCN(torch.nn.Module)`) como exemplo estrutural da teoria aplicada.
    *   Se for falar sobre Embeddings, lembre de como a pasta `Blue Sky/src/features/` vetoriza o texto coletado.

---

## 🛠️ Capítulo 3: Metodologia (Seu Diferencial)

Aqui você explica **como você construiu** tudo isso. Essa é a parte mais autoral do TCC.

*   **O que escrever:**
    *   **O Dataset Bluesky:** Detalhe a infraestrutura criada para extrair dados da nova rede. Explique o uso da API e do AT Protocol.
    *   **Engenharia de Dados (Opcional, mas brilhante):** Fale sobre como você lidou com o mega dataset de 21GB (CSV x Parquet/SQLite) e o Data Lake local.
    *   **Arquitetura do Sistema:** Descreva o fluxo do usuário na ponta até a IA. Construa um diagrama mostrando o Frontend da Web consultando a API, que por sua vez pede as previsões da GNN.

*   **Onde buscar inspiração/dados no projeto:**
    *   **Para o Dataset Bluesky:** A base do seu texto estará nos arquivos em `Blue Sky/src/collection/` (ex: `collect.py`). Você pode printar trechos desse código para mostrar como o AT Protocol retorna a rede de perfis.
    *   **Para a Arquitetura do Sistema:** Extraia dados de `frontend/api/main.py` (Backend FastAPI) e `frontend/web/src/app/` (Interface React/NextJS).
    *   **Fluxo de Treinamento Geral:** Explique os processos baseados em `mesclagem/treinar_mega_dataset.py`.

---

## 📊 Capítulo 4: Resultados, Experimentos e Discussão

Este é o ápice científico do trabalho. Aqui você mostra que sua ideia de usar grafos funciona.

*   **O que escrever:**
    *   **Duelo de Modelos:** Mostre a comparação cruel: pegar o modelo treinado com dados antigos do Twitter (dataset UPFD/LIAR) e testá-lo cruzado com os dados frescos nativos do Bluesky.
    *   **Ablation Study (Estudo de Ablação):** Mostre que avaliar a notícia *apenas usando texto vs avaliar usando Texto + Grafo* tem uma diferença brutal de precisão.
    *   **Rigor Estatístico:** Apresente os gráficos gerados, matrizes de confusão (Errou x Acertou) segregadas por domínio da notícia (Saúde, Política, Tech).
    *   **Limitações Atuais:** Seja humilde. Discuta a dificuldade da rotulagem manual da verdade no volume exponencial do Bluesky e potenciais vieses algorítmicos.

*   **Onde buscar inspiração/dados no projeto:**
    *   **Para as Tabelas e Gráficos:** Extraia tudo que sair do painel `mesclagem/pipeline_mensuracao.py` e `mesclagem/matriz_confusao.py`.
    *   **Para o "Duelo":** Use os logs e outputs dos testes `mesclagem/duelo_modelos_frescos.py` e `mesclagem/teste_fresh_bluesky.py`.
    *   **Evidências Visuais:** Pegue a imagem `resultado_final_clusters.png` na pasta raiz e coloque no núcleo do capítulo como prova de como as bolhas ideológicas se separam.

---

## ⚖️ Capítulo Opcional: Ética e LGPD

Trabalhar com dados de usuários é algo delicado em bancas avaliadoras de hoje em dia.

*   **O que escrever:** 
    *   Deixar claro que não foi raspado (scraped) nenhum dado privado, senha ou DM. Todo dado processado provém do "firehose" público do AT Protocol com licença de abertura do Bluesky.

*   **Onde buscar inspiração/dados no projeto:**
    *   Mostrar no código `Blue Sky/src/collection/` as chamadas assíncronas que dependem 100% de rotas públicas autorizadas da API, garantindo conformidade total de privacidade.
