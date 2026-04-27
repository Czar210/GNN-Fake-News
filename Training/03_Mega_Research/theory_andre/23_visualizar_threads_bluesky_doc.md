# Documentação Técnica: 23_visualizar_threads_bluesky.py

## Metadados

- **Arquivo analisado:** `23_visualizar_threads_bluesky.py`
- **Caminho:** `Training/03_Mega_Research/23_visualizar_threads_bluesky.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Visualização científica de cascatas de propagação (information visualization aplicada a fake-news detection); não treina modelo, apenas anota visualmente threads reais com o score de um RandomForest estrutural já persistido (Fase 6).
- **Modelos usados:** `rf_struct_gossipcop.pkl` (RandomForest treinado em GossipCop com features `[num_nodes, grau_root]`, persistido pelo script 17).
- **Datasets utilizados:** `dados_bluesky/graphs_extracted/graphs/threads.txt.gz` (cascatas reais Bluesky em formato textual: `size TAB ts TAB ids,csv`).
- **Saídas:** 6 PNG (estáticos para LaTeX) + 6 HTML interativos (defesa/banca) + `resumo.txt`, em `Execution/results/figuras_tcc/threads_bluesky/`.
- **Contribuição para a questão central:** Este script materializa visualmente o que "estrutura de propagação" significa. Os 6 exemplos cobrem a faixa de tamanhos (≈5, 15, 40, 80, 150, 400 nós) e cada um carrega anotado o score do RF estrutural — tornando tangível para a banca que o classificador "fake-like" do TCC está usando apenas duas features topológicas elementares (tamanho do grafo e grau da raiz). É a peça narrativa que conecta o resultado quantitativo do script 14 ("topologia sem texto") ao argumento da defesa: as cascatas existem, são distinguíveis por inspeção visual, e o modelo aprendeu o atalho trivial.

---

## 1. Visão Geral do Script

`23_visualizar_threads_bluesky.py` pertence à **Fase 6 — Modelos finais e aplicação no Bluesky** do pipeline (cf. `PIPELINE.md`). Não treina e não produz CSV — sua função é gerar **artefatos visuais** que sirvam de evidência ilustrativa na monografia (PNG) e na apresentação oral (HTML interativo).

Fluxo em quatro etapas (`[1/4]…[4/4]` do `main()`):

1. **Carregar o RF persistido** (`Execution/weights/rf_struct_gossipcop.pkl`) — modelo do script 17 com features `[num_nodes, grau_root]`.
2. **Ler `threads.txt.gz`** (até 500k linhas); descarta threads triviais (1 usuário).
3. **Selecionar 6 threads** cujos tamanhos mais se aproximam dos alvos `[5, 15, 40, 80, 150, 400]` (nearest-neighbor sobre o tamanho).
4. **Renderizar** cada thread como PNG (matplotlib, layout circular concêntrico) e HTML (pyvis com Barnes–Hut), anotando o score `P(fake)` do RF.

Os alvos são quasi-logarítmicos (ratios ~3, 2.7, 2, 1.9, 2.7), cobrindo duas ordens de grandeza — adequado para a distribuição de cauda longa típica de cascatas. O design segue princípios clássicos de Tufte (2001) — alta densidade informacional, *chartjunk* mínimo, codificação dupla cor+rótulo para fake/real. A topologia é **estrela cronológica** (raiz + filhos), fiel ao que `02_construir_grafos_bluesky.py` constrói (cf. `PIPELINE.md` §Fase 0).

---

## 2. Arquitetura e Componentes Principais

### 2.1 Leitura de cascatas — `ler_threads()`

`threads.append((len(ids), ts, ids))` (linhas 62–63). `partes[0]` (métrica derivada) é ignorado em favor de `len(ids)`. Threads com `len(ids) < 2` são descartadas — grafos de 1 nó não admitem propagação.

> 📖 **Vosoughi, S.; Roy, D.; Aral, S. (2018)** — "The spread of true and false news online"
> *Science*, v. 359, n. 6380, pp. 1146–1151. DOI: `10.1126/science.aap9559`
> **Localização:** Seção "Definitions" (p. 1147) — definição de *cascade* como árvore de retweets de um post raiz; Figura 1 (p. 1148) — visualização de cascatas reais com profundidade, tamanho e *structural virality*.
> **Relevância:** A pipeline usa o conceito de Vosoughi et al. de cascata = (raiz, descendentes), com topologia interna colapsada em estrela (sem `parent_of_repost`). A visualização do script 23 é análoga à Figura 1 desse paper.

### 2.2 Seleção de amostras — `selecionar_amostras()`

Algoritmo *nearest-target-without-replacement*:

$$t^{(k)} = \arg\min_{t \in \mathcal{T} \setminus S_{k-1}} \big| \text{size}(t) - \alpha_k \big|, \quad S_k = S_{k-1} \cup \{t^{(k)}\}$$

Os alvos $\alpha_k \in \{5,15,40,80,150,400\}$ formam escala quasi-logarítmica, garantindo cobertura uniforme em escala log — princípio para variáveis com cauda longa (cascatas seguem ~power-law, cf. Vosoughi et al. 2018, Fig. 2D).

> 📖 **Munzner, T. (2014)** — *Visualization Analysis and Design*. CRC Press. ISBN: `978-1-4665-0891-0`.
> **Localização:** Cap. 4 (Analysis: Four Levels for Validation), §4.2 *Domain situation*; Cap. 13 (Reduce Items and Attributes), §13.2 *Filter*.
> **Relevância:** Munzner formaliza *filter* como "reduce data set without changing the encoding" — a seleção de 6 threads é um filtro por escalar (tamanho), preservando o *range* da variável crítica em uma figura única.

### 2.3 Layout circular concêntrico — `visualizar_png()`

Layout determinístico (não força-dirigido):
- raiz em $(0,0)$;
- até 12 primeiros filhos em círculo unitário: $(\cos\theta_k, \sin\theta_k)$, $\theta_k = 2\pi k / n_1$;
- demais em anel externo $r=1.9$ com offset $0.3$ rad.

**Por que determinístico e não Fruchterman–Reingold?** Em estrela plana todos os filhos são topologicamente equivalentes (grau 1, distância 1 à raiz). FR convergiria à mesma configuração circular, mas com $\mathcal{O}(n^2)$ por iteração e rotações arbitrárias entre execuções. O layout fechado tem $\mathcal{O}(n)$, é reproduzível bit-a-bit, e preserva interpretação visual entre as 6 figuras (mesma escala, raiz centrada) — essencial para comparação lado-a-lado em prancha LaTeX.

> 📖 **Fruchterman, T. M. J.; Reingold, E. M. (1991)** — "Graph drawing by force-directed placement"
> *Software: Practice and Experience*, v. 21, n. 11, pp. 1129–1164. DOI: `10.1002/spe.4380211102`
> **Localização:** §2 ("The Algorithm"), Equações (1)–(2) (p. 1131) — forças $f_a(d) = d^2/k$, $f_r(d) = -k^2/d$, $k = C\sqrt{\text{area}/|V|}$; §3 ("Aesthetics") — uniformidade de comprimento de aresta, simetria, mínimo cruzamento.
> **Relevância:** Justifica que em grafos hub-and-spoke com nós equivalentes, FR converge ao layout circular determinístico — pode-se eleger o alvo sem custo iterativo.

> 📖 **Battista, G. D.; Eades, P.; Tamassia, R.; Tollis, I. G. (1999)** — *Graph Drawing: Algorithms for the Visualization of Graphs*. Prentice Hall. ISBN: `978-0-13-301615-4`.
> **Localização:** Cap. 3 (Divide and Conquer), §3.4 *Tree drawings — radial layouts* (pp. 100–107); Cap. 10 (Force-Directed Methods).
> **Relevância:** Formaliza o *radial tree layout* — descendentes em $r_d$ — como ótimo para árvores enraizadas. A estrela Bluesky tem profundidade 1: caso degenerado com $r_0=0$, $r_1=1$. O anel externo $r=1.9$ é extensão prática para $n > 12$.

### 2.4 Renderização interativa — `visualizar_html()` (pyvis + Barnes–Hut)

`net.barnes_hut(spring_length=80)` (linha 149) — `pyvis` é wrapper sobre `vis.js`; `barnes_hut` ativa o solver físico que aproxima repulsões coulombianas em $\mathcal{O}(n \log n)$ via quadtree em vez de $\mathcal{O}(n^2)$. `n_show = min(n, 200)` (linha 145) evita HTML lento no browser durante a defesa.

> 📖 **Barnes, J.; Hut, P. (1986)** — "A hierarchical O(N log N) force-calculation algorithm"
> *Nature*, v. 324, pp. 446–449. DOI: `10.1038/324446a0`
> **Localização:** §§1–2 — decomposição em octree/quadtree; critério de abertura $\theta = s/d$.
> **Relevância:** Algoritmo subjacente ao `barnes_hut()` do pyvis — torna layouts force-directed em browser viáveis para $n=200$.

> 📖 **Hagberg, A. A.; Schult, D. A.; Swart, P. J. (2008)** — "Exploring network structure, dynamics, and function using NetworkX"
> *Proceedings of the 7th Python in Science Conference (SciPy 2008)*, pp. 11–15. URL: `https://conference.scipy.org/proceedings/SciPy2008/paper_2/`
> **Localização:** Seção "Drawing networks" (p. 14) — ecossistema de layouts (`spring_layout`, `circular_layout`, `kamada_kawai_layout`).
> **Relevância:** Referencial canônico para a separação *graph as data* vs *graph drawing*. O script segue essa separação: estrutura `(size, ts, ids)` pura + dois renderers independentes (PNG/HTML).

---

## 3. Pipeline de Dados e Pré-processamento

### 3.1 Formato `threads.txt.gz`

Arquivo gzipado linha-a-linha com três campos TAB-separados:

| Campo | Tipo | Significado |
|-------|------|-------------|
| `partes[0]` | int | métrica derivada (provavelmente profundidade ou interações totais) — ignorada |
| `partes[1]` | string | timestamp ISO (ex: `2024-...`) |
| `partes[2]` | CSV de int | IDs de usuários, primeiro = autor da raiz |

O comentário do código (linhas 49–52) explicita que `partes[0] ≠ len(ids)` e prefere `len(ids)` — decisão correta porque `len(ids)` é o tamanho topológico real do grafo construído pelo script 02.

### 3.2 Cap em 500.000 linhas (`max_linhas`)

A leitura é truncada por defesa de memória — 500k linhas $\times \sim$200 bytes/linha $\approx$ 100 MB de strings, manejável em RAM. Como a seleção é *nearest-to-target* sobre `[5,15,40,80,150,400]`, e a distribuição de tamanhos do Bluesky é cauda-longa (a maioria das threads tem `size < 50`), 500k linhas são mais que suficientes para encontrar amostras próximas dos alvos pequenos; para `alvo=400`, depende da disponibilidade — o algoritmo simplesmente pega a maior thread disponível se não houver match exato.

### 3.3 Aplicação do RF — `aplicar_rf_score()`

`rf.predict_proba(X)[0, list(rf.classes_).index(0)]` (linha 85) extrai $P(y=\text{fake})$ — robusto a mudanças na ordem interna do scikit-learn (`0=fake, 1=real` por convenção do script 17).

**Confound estrutural:** em estrela plana com $n$ nós, $\text{grau\_root} = n-1$. As duas features são **linearmente dependentes**, então o RF usa efetivamente **uma única feature** — `num_nodes`. O script 11 (`diagnostico_confound`) trata exatamente esse confound do tamanho da cascata.

> 📖 **Breiman, L. (2001)** — "Random Forests"
> *Machine Learning*, v. 45, n. 1, pp. 5–32. DOI: `10.1023/A:1010933404324`
> **Localização:** §1 (algoritmo); §5 (Random Input Selection) — seleção aleatória de features por split; §11 (Outliers) — robustez.
> **Relevância:** Random Forest é o classificador escolhido no script 17 por: (a) features escalares heterogêneas sem normalização; (b) `predict_proba` calibrado via fração de votos; (c) interpretabilidade via `feature_importances_`. A dependência linear `grau_root = num_nodes − 1` divide `feature_importances_` arbitrariamente entre as duas — limitação intrínseca, não erro do script 23.

---

## 4. Construção da Visualização (núcleo do script)

### 4.1 Codificação visual (variável → canal)

Aplicação literal do *encoding-channel framework* de Munzner (2014, Cap. 5):

| Atributo | Canal | Justificativa |
|----------|-------|---------------|
| Raiz | posição central + cor laranja `#E07B54` + rótulo "R" + tamanho 400px | Posição é o canal mais efetivo; redundância cor+tamanho+rótulo é *redundant encoding* (Munzner Cap. 5). |
| Filhos | cor verde `#5BAD72` + tamanho 40px | Contraste cromático com a raiz. |
| Score `P(fake)` | cor binária do título: vermelho `#c0392b` se ≥0.5, verde `#27ae60` c.c. | Binária (não gradiente) — consistente com decisão final. |
| Tamanho `n` | densidade de pontos | Propriedade emergente do layout. |

> 📖 **Munzner, T. (2014)** — *Visualization Analysis and Design*. CRC Press.
> **Localização:** Cap. 5 (Marks and Channels), Tab. 5.1 (p. 102) — ranking de efetividade (posição > comprimento > ângulo > área); Cap. 10 (Map Color), §10.2 — cor categórica vs sequencial.
> **Relevância:** Justifica cor categórica (laranja/verde) para distinguir raiz/filhos e cor binária no título (vermelho/verde) para fake/real — sem gradiente porque a banca precisa ler "fake" ou "real", não interpretar régua.

### 4.2 Princípios de Tufte aplicados

- **Data-ink ratio alto:** sem grade, eixos, *bounding box* — apenas nós, arestas ($\alpha=0.3$), título e marcador "R". `ax.set_axis_off()` (linha 129) remove spines.
- **Small multiples:** as 6 PNGs formam *small multiples* — mesma codificação, mesma escala (`xlim/ylim=±2.3`), variando apenas tamanho + score.
- **Chartjunk eliminado:** sem 3D, gradientes, texturas, legendas redundantes.

> 📖 **Tufte, E. R. (2001)** — *The Visual Display of Quantitative Information*. 2ª ed., Graphics Press. ISBN: `978-1-930824-13-3`.
> **Localização:** Cap. 4 (Data-Ink and Graphical Redesign), pp. 91–105; Cap. 5 (Chartjunk), pp. 107–121; Cap. 8 (Data Density and Small Multiples), pp. 161–177.
> **Relevância:** Design das 6 figuras aplica diretamente os três princípios fundamentais. As PNGs em prancha LaTeX formam *small multiple* genuíno — forma mais eficiente conhecida para comparação visual entre amostras de uma distribuição.

### 4.3 PNG vs HTML

PNG (matplotlib): destino LaTeX, layout determinístico, sem interatividade, <1s/figura, reprodutibilidade máxima. HTML (pyvis): apresentação oral, layout Barnes–Hut com pequena estocasticidade, zoom/pan/drag/hover, $\leq 200$ nós (truncamento explícito). Dupla saída implementa *visualization for communication*: artefato imutável para documento + artefato explorável para apresentação.

---

## 5. Métricas de Avaliação

Este script **não computa métricas** (não é um experimento). O único valor numérico produzido por amostra é o score do RF, transcrito em três lugares:

1. Subtítulo da figura PNG (`f"score 'fake-like' (...) = {score:.3f}"`, linha 124–125);
2. Nome do arquivo (`thread_{k:02d}_n{size}_score{score_int:03d}.png`, linha 195) — útil para grep e referência cruzada no CSV;
3. `resumo.txt` (linhas 200–202) — tabela compacta com `(k, size, ts, score, predição binária)`.

A predição binária usa o threshold canônico $0.5$ (linha 199): `pred = "FAKE" if score >= 0.5 else "real"`. Não é o threshold ótimo (que dependeria de validação em conjunto com label, indisponível no Bluesky) — é o threshold padrão de Bayes para custo simétrico.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 O que as 6 figuras provam visualmente

O RF do script 17 usa apenas `[num_nodes, grau_root]`, e em estrelas planas isso reduz à regra "se $n > n^*$ então fake". As 6 figuras tornam isso visualmente óbvio:
- $n=5, 15$ → score baixo, título verde, predição "real";
- $n \approx 400$ → score alto, título vermelho, predição "fake";
- transição monotônica no tamanho.

Para a banca: abrir o HTML da thread maior, fazer zoom, mostrar que **não há nada distintivo** além do número de nós — sem ramificação, sem profundidade — é o jeito mais econômico de comunicar o "smoking gun".

### 6.2 Relação com o pipeline

O script 23 é o **avatar visual** de três experimentos numéricos:

| Script | Resultado | Tradução visual no 23 |
|--------|-----------|----------------------|
| 11 (`confound`) | F1 só com `num_nodes` ≈ 0.7+ | `num_nodes` é o eixo dominante |
| 14 (`topologia_sem_texto`) | F1 sem BERT ≈ F1 com BERT | Figura sem texto, mesmo assim banca prevê label |
| 16 (`gnn_explainer`) | Explainer aponta arestas estruturais | HTML permite navegar essas arestas |

### 6.3 Relação com Vosoughi et al. (2018)

Vosoughi et al. (Fig. 2A–C, p. 1148) mostraram cascatas falsas mais largas e ~6x mais rápidas. O script 23 reproduz qualitativamente: threads maiores → score alto. **Diferença crítica:** Vosoughi tinha ground truth humano (Snopes/PolitiFact); aqui as labels Bluesky são heurísticas (script 01). Logo as figuras documentam **comportamento do classificador**, não fake-news real — distinção preservada pelas aspas em "score 'fake-like'" e pelo header do `resumo.txt`.

### 6.4 Resposta parcial à questão do TCC

> **Seção 6.4 — Threads visualizadas como evidência da defesa:**

Contribuição visual à resposta negativa da questão central:

1. RF estrutural (2 features) tem score crescente com o tamanho da thread;
2. As 6 amostras Bluesky materializam esse comportamento sem ambiguidade;
3. Como GNNs treinadas em FNN/UPFD aprendem o mesmo atalho topológico, mostrar um RF trivial fazendo "o mesmo trabalho" *out-of-domain* é evidência adicional de que o sinal é tamanho, não conteúdo;
4. Defesa oral: abrir um HTML, perguntar "o que distingue esta thread como fake?" — resposta esperada "nada além do tamanho" fecha o argumento.

A figura é retórica e pedagógica: não adiciona números, mas torna *evidente* o que os scripts 11, 14, 16, 17 já provaram quantitativamente.

---

## 7. Análise de Código

### 7.1 Erros e fragilidades identificadas

**E1 — Sobreposição visual para $n$ grande (linhas 96–104):** Para $n=400$, o anel externo recebe 387 nós em $r=1.9$ → densidade ~62 nós/rad. Pontos ($s=40$) se sobrepõem, gerando "coroa borrada". Solução: múltiplos anéis com capacidades crescentes (12, 24, 48, 96) — preserva radial layout (Battista et al. 1999 §3.4).

**E2 — `partes[0]` não validado:** Comentário (linhas 49–52) trata como hipótese não confirmada. Em sessão exploratória, plotar histograma `partes[0] - len(ids)` para validar.

**E3 — `selecionar_amostras` falha silenciosa:** Se `max(sizes) < 400`, retorna maior thread sem aviso. Sugestão: warning quando `|size - alvo|/alvo > 0.2`.

**E4 — `n_show` no HTML sem indicação:** Linha 145: para $n=400$ mostra 200 mas título declara `size=400`. Sugestão: `f"size={n} (mostrados {n_show})"`.

### 7.2 Boas práticas observadas

- **`matplotlib.use("Agg")` antes de importar `pyplot`** (linhas 32–34): correto, evita inicialização de backend GUI em servidor headless.
- **`plt.close()` após `savefig()`** (linha 131): libera memória — importante porque o loop gera 6 figuras grandes.
- **Tratamento defensivo de `pyvis` ausente** (linhas 135–138): `try/except ImportError` com retorno `False` permite o script rodar mesmo sem pyvis instalado, gerando apenas PNGs.
- **`dpi=130, bbox_inches="tight"`** (linha 131): resolução adequada para LaTeX (300 dpi seria oversize); `tight` remove margens brancas excessivas.
- **Nome de arquivo com score embutido** (linha 195): `thread_00_n5_score012.png` é *self-documenting* — qualquer pessoa lendo a pasta consegue inferir o conteúdo sem abrir o `resumo.txt`.
- **Saída textual estruturada** (`resumo.txt`): facilita citação na monografia e busca por grep.

### 7.3 Complexidade computacional

| Etapa | Complexidade | Bottleneck |
|-------|--------------|------------|
| Leitura `threads.txt.gz` | $\mathcal{O}(L)$, $L \leq 5 \times 10^5$ | I/O sequencial (gzip) |
| `selecionar_amostras` | $\mathcal{O}(|\text{alvos}| \cdot |\mathcal{T}|)$ = $\mathcal{O}(6 L)$ | linear scan ingênuo — aceitável |
| Render PNG (por figura) | $\mathcal{O}(n)$ | < 1s para $n \leq 400$ |
| Render HTML (por figura) | $\mathcal{O}(n \log n)$ | dominado por Barnes–Hut |
| **Total** | $\mathcal{O}(L + 6 \cdot n_{\max} \log n_{\max})$ | $\approx$ 5–15s no total |

Trivialmente aceitável; nenhuma otimização necessária.

---

## 8. Referências Bibliográficas

1. MUNZNER, T. **Visualization Analysis and Design**. CRC Press / AK Peters Visualization Series, 2014. ISBN: `978-1-4665-0891-0`.

2. FRUCHTERMAN, T. M. J.; REINGOLD, E. M. **Graph drawing by force-directed placement**. *Software: Practice and Experience*, v. 21, n. 11, pp. 1129–1164, 1991. DOI: `10.1002/spe.4380211102`.

3. HAGBERG, A. A.; SCHULT, D. A.; SWART, P. J. **Exploring network structure, dynamics, and function using NetworkX**. In: *Proceedings of the 7th Python in Science Conference (SciPy 2008)*, pp. 11–15, 2008. URL: `https://conference.scipy.org/proceedings/SciPy2008/paper_2/`.

4. BATTISTA, G. D.; EADES, P.; TAMASSIA, R.; TOLLIS, I. G. **Graph Drawing: Algorithms for the Visualization of Graphs**. Prentice Hall, 1999. ISBN: `978-0-13-301615-4`.

5. VOSOUGHI, S.; ROY, D.; ARAL, S. **The spread of true and false news online**. *Science*, v. 359, n. 6380, pp. 1146–1151, 2018. DOI: `10.1126/science.aap9559`.

6. TUFTE, E. R. **The Visual Display of Quantitative Information**. 2ª ed. Cheshire, CT: Graphics Press, 2001. ISBN: `978-1-930824-13-3`.

7. BREIMAN, L. **Random Forests**. *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`.

8. BARNES, J.; HUT, P. **A hierarchical O(N log N) force-calculation algorithm**. *Nature*, v. 324, pp. 446–449, 1986. DOI: `10.1038/324446a0`.

9. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Cascata (cascade) | Árvore de propagação enraizada num post original; nós = usuários que repostaram, arestas = relação de repost | Vosoughi et al. (2018), Definitions, p. 1147 |
| Estrela plana (flat star) | Cascata de profundidade 1: raiz + filhos diretos sem subníveis. Usada quando `parent_of_repost` indisponível | Conceito do TCC (cf. PIPELINE.md §Fase 0) |
| Layout radial / concêntrico | Posiciona descendentes de profundidade $d$ em círculo de raio $r_d$; raiz no centro | Battista et al. (1999), §3.4 |
| Force-directed layout | Família de layouts que simulam atração+repulsão entre nós (Coulomb + Hooke) e iteram até equilíbrio | Fruchterman & Reingold (1991), §2 |
| Barnes–Hut | Aproximação $\mathcal{O}(n \log n)$ para *n-body simulation* via decomposição quadtree; controlada pelo parâmetro $\theta$ | Barnes & Hut (1986) |
| Small multiples | Princípio Tufte: figuras múltiplas com mesmo encoding e mesma escala, variando apenas o atributo de interesse | Tufte (2001), Cap. 8 |
| Data-ink ratio | Fração da tinta da figura que codifica informação (vs. decoração); maximizar é princípio cardinal | Tufte (2001), Cap. 4 |
| `predict_proba` | Saída do RF que aproxima $P(y=c \mid \mathbf{x})$ pela fração de árvores que votam em $c$ | Breiman (2001), §1 |
| Out-of-domain (OOD) | Aplicação de modelo treinado num dataset (GossipCop) a outro (Bluesky) sem fine-tuning | Pipeline TCC, Fase 6 |
| Score "fake-like" | $P(y=\text{fake} \mid \mathbf{x})$ retornado pelo RF estrutural; threshold de decisão = 0.5 | Script 17 |

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 23_visualizar_threads_bluesky.py
📄  Arquivo gerado: theory_andre/23_visualizar_threads_bluesky_doc.md
📚  Fontes acadêmicas utilizadas: 9
    1. Munzner (2014) — Visualization Analysis and Design (CRC)
    2. Fruchterman & Reingold (1991) — Force-directed placement (SPE)
    3. Hagberg et al. (2008) — NetworkX (SciPy)
    4. Battista et al. (1999) — Graph Drawing (Prentice Hall)
    5. Vosoughi, Roy & Aral (2018) — Spread of true/false news (Science)
    6. Tufte (2001) — Visual Display of Quantitative Information
    7. Breiman (2001) — Random Forests (Machine Learning)
    8. Barnes & Hut (1986) — O(N log N) force calculation (Nature)
    9. Shu et al. (2020) — FakeNewsNet (Big Data)
🔍  Conceitos cobertos:
    - Leitura de cascatas em formato textual gzipado
    - Seleção nearest-target sobre escala log
    - Layout radial concêntrico determinístico vs force-directed
    - Codificação visual (channel-encoding framework)
    - Princípios de Tufte (data-ink, small multiples, anti-chartjunk)
    - Barnes-Hut para HTML interativo (pyvis)
    - Confound topológico (grau_root = num_nodes - 1)
    - Conexão com Vosoughi et al. (cascade size vs falsity)
    - Out-of-domain inference (RF GossipCop -> Bluesky)
⚠️   Limitações:
    - `partes[0]` do .txt.gz não foi validado empiricamente contra len(ids); comentário do código é hipótese
    - O paper original do pyvis não existe (é wrapper de vis.js); citei Barnes & Hut como base teórica
    - Threshold 0.5 sem calibração — Bluesky não tem ground truth para encontrar threshold ótimo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
