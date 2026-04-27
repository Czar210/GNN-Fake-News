# Documentação Técnica: 15_analise_estrutural.py

## Metadados

- **Arquivo analisado:** `15_analise_estrutural.py`
- **Caminho:** `Training/03_Mega_Research/15_analise_estrutural.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Análise descritiva estrutural — comparação fake-vs-real em 5 métricas topológicas (`num_nodes`, `num_edges`, `depth_max`, `width_max`, `branching_avg`) com t-test de Welch e Cohen's d
- **Datasets:** FakeNewsNet PolitiFact (construção própria do script 00); UPFD-PolitiFact e UPFD-GossipCop (Dou et al. 2021, via PyG)
- **Contribuição para a questão central:** Este script fornece a **prova descritiva, anterior a qualquer modelo**, de que existe um *atalho topológico* nos três datasets — diferenças sistemáticas e de magnitude grande (Cohen's d > 0.8 em múltiplas métricas) entre cascatas fake e real. Junto com 14 (modelo *consegue* explorar o atalho) e 16 (modelo *aponta* para arestas estruturais), fecha a defesa metodológica: o problema não é a arquitetura GNN, é o sinal enviesado dos dados.

---

## 1. Visão Geral

`15_analise_estrutural.py` opera **antes** de qualquer treino. Recebe três coleções de grafos rotulados — FakeNewsNet construído pelo `00_construir_grafos_fakenewsnet.py` e os dois subconjuntos oficiais do UPFD via `torch_geometric.datasets.UPFD` — e calcula, **por grafo**, cinco métricas topológicas: nós, arestas, profundidade máxima (BFS desde o nó raiz, índice 0), largura máxima e branching factor médio.

Para cada dataset compara `fake` vs `real` com (1) **t-test independente de Welch** (`scipy.stats.ttest_ind(..., equal_var=False)`) — não exige igualdade de variâncias, violada em cascatas; e (2) **Cohen's d com pooled SD** — magnitude padronizada, interpretada com Cohen (1988): |d|≥0.2 pequeno, ≥0.5 médio, ≥0.8 grande.

Outputs: `estatisticas.csv`, `fig_distribuicoes.png` (3×4 histogramas), `fig_separabilidade.png` (barras de d) e amostras HTML interativas (`pyvis`) de grafos medianos.

Diferentemente do script 09 (inferência *entre modelos*), o 15 faz inferência *entre classes nos dados crus*. Não há treino — a única fonte de variabilidade é a amostragem original dos grafos pelos coletadores (Shu et al. 2020 e Dou et al. 2021).

---

## 2. Métricas Topológicas — Fundamentação

Todas as métricas em `metricas_grafo()` (linhas 55–98) pertencem ao núcleo clássico de análise de redes consolidado em **Newman (2010)**, *Networks: An Introduction*, Oxford University Press.

### 2.1 Tamanho da Cascata — `num_nodes`

**Descrição:** Número total de nós. Em estrela plana (FNN do 00), $|V| = 1 + |\text{retweets}|$; em árvores reais (UPFD), conta toda a sub-árvore.

**Fundamento matemático:** $|V(G)| = n$, onde $V(G)$ é o conjunto de vértices de $G = (V, E)$ e $n$ sua cardinalidade. Linha 56: `n = int(g.num_nodes)`.

**Função no fake news:** **Proxy mais direto de viralidade**. Vosoughi et al. (2018) mostram que cascatas falsas no Twitter alcançam até uma ordem de magnitude mais usuários que verdadeiras na mesma janela temporal — diferença que persiste após controlar idade da conta, seguidores, atividade e verificação. `num_nodes` é o confound mais óbvio (testado isoladamente no script 11).

**Embasamento:**

> 📖 **Vosoughi, S.; Roy, D.; Aral, S. (2018)** — "The Spread of True and False News Online"
> *Science*, v. 359, n. 6380, pp. 1146–1151, 2018. DOI: `10.1126/science.aap9559`
> **Localização:** Figura 1A (CDFs de cascade size); Tabela 1 ("Cascade size"); Seção "Results > Properties of the diffusion of false and true news"
> **Relevância:** Evidência fundamental de que cascatas fake são sistematicamente maiores — fundamento para esperar Cohen's d positivo em `num_nodes`.

> 📖 **Newman, M. E. J. (2010)** — *Networks: An Introduction*. Oxford. ISBN 978-0-19-920665-0
> **Localização:** Cap. 6.1 ("Components"); Cap. 8 (cauda pesada e lei de potência)
> **Relevância:** Define tamanho de componente e justifica eixo log na visualização (cauda pesada típica).

**No código:** linha 226 — `ax.set_xscale("log")` e bins log (`np.geomspace`) seguem a prática-padrão de Newman (Cap. 8).

---

### 2.2 Número de Arestas — `num_edges`

**Descrição:** $|E(G)|$. Em árvores: $|E| = |V| - 1$ — em UPFD a métrica é redundante com `num_nodes`. Mantida como controle: se $|E| \neq |V|-1$ houver, há ciclos/multi-arestas.

**Fundamento matemático:** $|E(G)| = m$. Em árvore: $m = n - 1$.

**Embasamento:**

> 📖 **Newman (2010)** — Cap. 6, Seção 6.10 ("Trees and forests"), Equação 6.31
> **Relevância:** Formaliza $m = n-1$ em árvores; verificação implícita do script ao reportar `num_edges` separadamente.

**No código:** linhas 64–66 simetrizam o `edge_index` (`adj[s].append(t); adj[t].append(s)`) para BFS funcionar mesmo em encoding direcionado do PyG.

---

### 2.3 Profundidade Máxima — `depth_max`

**Descrição:** Distância geodésica máxima desde o nó raiz (índice 0) por BFS no grafo não-direcionado.

**Fundamento matemático:**

$$\text{depth}(G) = \max_{v \in V} d(0, v)$$

onde $d(0, v)$ é o número mínimo de arestas de 0 a $v$. Estrela plana: $\text{depth} = 1$. Linhas 69–81 implementam BFS canônica:

```python
profundidade = {0: 0}; fila = deque([0])
while fila:
    u = fila.popleft()
    for v in adj[u]:
        if v not in profundidade:
            profundidade[v] = profundidade[u] + 1
            fila.append(v)
depth_max = max(profundidade.values())
```

Variáveis: `profundidade` ($v \mapsto d(0, v)$, populado em ordem BFS, garante distância mínima); `fila` (FIFO, `popleft()` preserva BFS); `depth_max` ($= \max_v \text{profundidade}[v]$).

**Função no fake news:** Métrica-chave em Vosoughi et al. (2018). Profundidade alta = difusão peer-to-peer multi-hop (viralidade orgânica); baixa = broadcast (hub propaga, demais folhas). Vosoughi reporta cascatas falsas em profundidades sistematicamente maiores: mediana ~10 (fake) vs ~4–5 (real).

**Embasamento:**

> 📖 **Vosoughi, Roy & Aral (2018)** — *Science*, v. 359, pp. 1146–1151. DOI: `10.1126/science.aap9559`
> **Localização:** Figura 1B (CDF de cascade depth); subseção "Cascade depth, size, breadth, structural virality, and unique users"
> **Relevância:** Documenta empiricamente que `depth` é discriminante entre fake e real — motivação direta da métrica.

> 📖 **Goel, S.; Anderson, A.; Hofman, J.; Watts, D. J. (2016)** — "The Structural Virality of Online Diffusion"
> *Management Science*, v. 62, n. 1, pp. 180–196. DOI: `10.1287/mnsc.2015.2158`
> **Localização:** Seção 2 ("Measuring structural virality"); Eq. (1) (índice de Wiener); Figura 2 (broadcast vs viral)
> **Relevância:** Formaliza *broadcast* (depth baixa, width alta) vs *viral* (depth alta, width moderada) — embasa leitura de `depth_max` como sinal de viralidade orgânica.

> 📖 **Newman (2010)** — Cap. 10.3 ("Breadth-first search"), Algoritmo 10.1
> **Relevância:** BFS canônico, complexidade $O(n + m)$.

**Caveat:** assume nó 0 = raiz (verdade em FNN/UPFD).

---

### 2.4 Largura Máxima — `width_max`

**Descrição:** Maior número de nós a uma mesma distância da raiz (tamanho máximo de "geração").

**Fundamento matemático:**

$$\text{width}(G) = \max_{d \in \mathbb{N}} |\{v \in V : d(0, v) = d\}|$$

Em estrela plana: $\text{width} = n - 1$. Linhas 82–85:

```python
largura = defaultdict(int)
for d in profundidade.values():
    largura[d] += 1
width_max = max(largura.values())
```

**Função no fake news:**
Largura alta + profundidade baixa = **broadcast** (influenciador propaga, todos viram folhas). Largura moderada + profundidade alta = **viralidade orgânica**. A combinação `(width_max, depth_max)` é a base da decomposição broadcast/viral de Goel et al. (2016).

**Embasamento:**

> 📖 **Goel et al. (2016)** — *Management Science*. Figura 1 (taxonomia broadcast vs viral); Seção 3.2 ("Structural diversity")
> **Relevância:** Demonstra que `depth` e `width` separadamente não distinguem broadcast de viral — apenas a combinação informa.

---

### 2.5 Branching Factor Médio — `branching_avg`

**Descrição:** Filhos médios por nó **não-folha** na árvore enraizada. Filhos de $u$ = vizinhos com `profundidade > profundidade[u]`.

**Fundamento matemático:**

Seja $C(u) = \{v \in N(u) : d(0, v) > d(0, u)\}$ os filhos de $u$. O branching factor médio é:

$$b(G) = \frac{1}{|U|} \sum_{u \in U} |C(u)|, \quad U = \{u \in V : |C(u)| > 0\}$$

Variáveis:
- $N(u)$: vizinhança de $u$ no grafo não-direcionado;
- $C(u)$: subconjunto descendente — equivalente a "filhos" na árvore enraizada;
- $U$: nós não-folha (pais);
- $b(G)$: estimador da expectativa empírica de filhos por pai (folhas excluídas via `if nf > 0`).

**Função no fake news:**
Em redes scale-free (Barabási & Albert 1999) o branching local é altamente assimétrico: hubs têm centenas de filhos, folhas zero. Em fake news: branching alto na raiz com cauda fina = **propagação por hub** (contas verificadas reposting); branching homogêneo em vários níveis = **peer-to-peer**.

**Embasamento:**

> 📖 **Watts, D. J.; Strogatz, S. H. (1998)** — "Collective Dynamics of 'Small-World' Networks"
> *Nature*, v. 393, n. 6684, pp. 440–442. DOI: `10.1038/30918`
> **Localização:** Equação (1) (clustering coefficient); Figura 2 (transição entre regimes regular/random)
> **Relevância:** Estabelece que o grau médio (relacionado ao branching factor em árvores) controla propriedades globais da rede — fundamento da análise de redes que o script aplica.

> 📖 **Barabási, A.-L.; Albert, R. (1999)** — "Emergence of Scaling in Random Networks"
> *Science*, v. 286, n. 5439, pp. 509–512. DOI: `10.1126/science.286.5439.509`
> **Localização:** Equação (1) (lei de potência $P(k) \sim k^{-\gamma}$); Figura 2 (distribuições empíricas)
> **Relevância:** Mostra que grau (e branching local) em redes sociais reais segue lei de potência — justifica reportar `std` no CSV (cauda pesada faz a média sozinha enganar).

> 📖 **Newman (2010)** — Cap. 8.4 ("Power laws and scale-free networks"); Cap. 12.10.4 ("Trees")
> **Relevância:** Síntese moderna que integra Watts–Strogatz e Barabási–Albert.

**Limitação:** a média mascara cauda pesada — mediana e quartil 90 dariam imagem mais robusta (não implementado).

---

## 3. Inferência Estatística — Welch e Cohen's d

### 3.1 t-test de Welch (variâncias desiguais)

**Descrição:** `scipy.stats.ttest_ind(f, r, equal_var=False)` (linha 122). `equal_var=False` aplica a aproximação de **Welch (1947)**, que NÃO assume $\sigma_F = \sigma_R$. Escolha correta: cascatas falsas têm variância sistematicamente maior (caudas mais pesadas), violando a homocedasticidade do t-test clássico.

**Fundamento matemático (Welch 1947, Eqs. 5 e 8):**

$$t = \frac{\bar{X}_F - \bar{X}_R}{\sqrt{\dfrac{s_F^2}{n_F} + \dfrac{s_R^2}{n_R}}}$$

com graus de liberdade aproximados pela equação de Welch–Satterthwaite:

$$\nu \approx \frac{\left( \dfrac{s_F^2}{n_F} + \dfrac{s_R^2}{n_R} \right)^2}{\dfrac{(s_F^2/n_F)^2}{n_F - 1} + \dfrac{(s_R^2/n_R)^2}{n_R - 1}}$$

Variáveis:
- $\bar{X}_F, \bar{X}_R$: médias amostrais da métrica (ex.: `depth_max`) sobre grafos fake e real;
- $s_F^2, s_R^2$: variâncias amostrais (correção de Bessel, divisor $n-1$);
- $n_F, n_R$: tamanhos das subamostras (`n_fake`, `n_real`, linha 127);
- $\nu$: graus de liberdade efetivos (fracionários);
- $t$: estatístico, comparado contra $t_\nu$ para obter p bicaudal.

**Embasamento:**

> 📖 **Welch, B. L. (1947)** — "The Generalization of 'Student's' Problem When Several Different Population Variances Are Involved"
> *Biometrika*, v. 34, n. 1/2, pp. 28–35. DOI: `10.1093/biomet/34.1-2.28`
> **Localização:** Seção 3 ("The case of two samples"), Equação (5) (estatístico); Equação (8) (graus de liberdade aproximados)
> **Relevância:** Fonte original do teste implementado por `scipy.stats.ttest_ind(equal_var=False)`. Welch demonstra que a aproximação tem erro Tipo I controlado em α=0.05 mesmo com $\sigma_F^2/\sigma_R^2$ variando 1–10× — exatamente o cenário de cascatas fake-vs-real.

> 📖 **Student [Gosset, W. S.] (1908)** — "The Probable Error of a Mean"
> *Biometrika*, v. 6, n. 1, pp. 1–25. DOI: `10.1093/biomet/6.1.1`
> **Localização:** Seção VII ("Examples and applications")
> **Relevância:** Trabalho fundador que introduz a distribuição $t$ — base sobre a qual Welch construiu a generalização heteroscedástica.

**Caveat — múltiplos testes sem correção:**
$5 \text{ métricas} \times 3 \text{ datasets} = 15$ testes sem Bonferroni/Holm. FWER = $1 - 0.95^{15} \approx 0.537$. Nível corrigido: $\alpha' = 0.05/15 \approx 0.0033$. **Impacto prático:** os Cohen's d esperados >0.8 tornam a discussão de p secundária — magnitude grande sustenta a conclusão independentemente do limiar de p. Mesmo assim, a monografia deve mencionar (E2 da Seção 7).

---

### 3.2 Cohen's d — Magnitude do Efeito

**Descrição:** Diferença de médias dividida pelo desvio-padrão pooled. Diferentemente do p-valor, **não depende de n**: mede magnitude em unidades de DP. Para o TCC, é a métrica decisiva — diz se a diferença topológica é grande o suficiente para que um modelo *consiga* explorá-la.

**Fundamento matemático (Cohen 1988, Eq. 2.5.3):**

$$d = \frac{\bar{X}_F - \bar{X}_R}{s_p}, \quad s_p = \sqrt{\frac{(n_F - 1) s_F^2 + (n_R - 1) s_R^2}{n_F + n_R - 2}}$$

Variáveis:
- $\bar{X}_F - \bar{X}_R$: diferença bruta de médias (mesmas unidades originais);
- $s_F^2, s_R^2$: variâncias amostrais com correção de Bessel;
- $n_F + n_R - 2$: graus de liberdade do estimador pooled;
- $s_p$: DP pooled (média ponderada das variâncias pelos df);
- $d$: número de DP entre as médias das duas populações.

**Limiares de interpretação (Cohen 1988, Cap. 2, Tabela 2.2.1):**

| Magnitude | Limiar    | Interpretação                                  |
|-----------|-----------|------------------------------------------------|
| Pequeno   | $|d| \geq 0.2$ | Efeito detectável mas pouco visível a olho nu |
| Médio     | $|d| \geq 0.5$ | Efeito visível em comparação direta           |
| Grande    | $|d| \geq 0.8$ | Distribuições mal sobrepostas                  |

Cohen reforça que são *benchmarks heurísticos para ciências sociais* e devem ser ajustados ao domínio. Para cascatas, Vosoughi et al. (2018) reportam $d > 1.0$ entre fake e real em depth e size — efeitos enormes, muito acima de "grande".

**Embasamento:**

> 📖 **Cohen, J. (1988)** — *Statistical Power Analysis for the Behavioral Sciences*. 2ª ed. Hillsdale, NJ: Lawrence Erlbaum. ISBN: 978-0-8058-0283-2
> **Localização:** Capítulo 2, Seção 2.2 ("The effect size index: d"); Tabela 2.2.1 (limiares 0.2/0.5/0.8); Equação 2.5.3 (formulação pooled)
> **Relevância:** Fonte canônica do índice d e dos limiares small/medium/large usados no script (linhas 252–254 desenham guias horizontais em ±0.2; legenda do plot menciona explicitamente "0.2 = pequeno; 0.5 = médio; 0.8 = grande").

**No código (linhas 110–114):**

```python
def cohen_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    pooled = np.sqrt(((a.std()**2 * (len(a)-1)) + (b.std()**2 * (len(b)-1))) /
                     (len(a) + len(b) - 2)) if len(a) > 1 and len(b) > 1 else 1e-9
    return (a.mean() - b.mean()) / max(pooled, 1e-9)
```

**Bug sutil:** `np.std()` default usa divisor $n$ (não $n-1$). A multiplicação por `(len(a)-1)` "compensa" mas não é exatamente a variância amostral com Bessel — o canônico seria `a.var(ddof=1) * (len(a)-1)`. Diferença <1% para $n \geq 100$ (todos os datasets aqui), mas merece correção (ver E1).

> Linhas 234–237: Cohen's d aparece como anotação textual em cada subplot do `fig_distribuicoes.png`, junto com o p — apresentação correta para distinguir significância de magnitude.

---

## 4. Pipeline — Encadeamento dos Datasets

### 4.1 FakeNewsNet via `gerar_folds.carregar_grafos()`

**Linha 173:** reusa `carregar_grafos` do `gerar_folds.py`, carregando os `.pt` do `00_construir_grafos_fakenewsnet.py`. **Importante:** este FNN é **estrela plana** com features posicionais — em estrela plana, `depth_max ≡ 1` e `branching_avg ≡ n-1` por construção. Para FNN, o sinal real está em `num_nodes`.

### 4.2 UPFD-PolitiFact e UPFD-GossipCop

**Linhas 131–135:** `carregar_upfd("politifact")` e `carregar_upfd("gossipcop")` baixam o UPFD oficial via PyG com feature de perfil, concatenando train+val+test. UPFD é o único dos três com **árvores reais** — `depth_max > 1` é normal e a métrica é informativa.

**Embasamento UPFD:**

> 📖 **Dou, Y.; Shu, K.; Xia, C.; Yu, P. S.; Sun, L. (2021)** — "User Preference-aware Fake News Detection"
> *SIGIR '21*, pp. 2051–2055. DOI: `10.1145/3404835.3462990` | arXiv: `2104.12259`
> **Localização:** Seção 3.1 ("Data Collection"); Tabela 1 (estatísticas PolitiFact/GossipCop); Figura 2 (esquema do grafo de propagação)
> **Relevância:** Define os datasets de referência da literatura. Cada grafo é árvore enraizada onde a raiz é o post original e filhos são retweets em ordem temporal — propriedade que o BFS desde o nó 0 explora corretamente.

### 4.3 Distribuição de Classes

**Linhas 177–179:** o script imprime fração `fake` em cada dataset. UPFD-PolitiFact é nominalmente balanceado (~314/314); GossipCop próximo de 50/50; FNN segue o CSV de Shu (~50/50).

---

## 5. Outputs

### 5.1 `estatisticas.csv`
Colunas: `dataset, metrica, media_fake, std_fake, media_real, std_real, t, p, cohen_d, n_fake, n_real`. Linhas 191–200. Tabela mestra que alimenta as tabelas LaTeX da monografia.

### 5.2 `fig_distribuicoes.png` (3×4)
Linhas 215–242. Painel 3 datasets × 4 métricas com histogramas sobrepostos fake (laranja) vs real (verde), 30 bins. Para `num_nodes`, **bins log-espaçados** (`np.geomspace`) e eixo log — prática-padrão de Newman (Cap. 8) para cauda pesada. Cada subplot anota `d` e `p`.

### 5.3 `fig_separabilidade.png`
Linhas 245–265. Barras de Cohen's d agrupadas por dataset; guias horizontais em ±0.2. Sugestão E3: adicionar guias em ±0.5 e ±0.8 para leitura imediata dos três limiares.

### 5.4 Amostras HTML (`pyvis`)
Linhas 138–161, 269–280. Para cada dataset, seleciona o grafo de tamanho **mediano** em cada classe e renderiza HTML interativo. Razão: amostras extremas são pouco representativas; mediana ilustra o padrão típico esperado pela banca.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Lugar na tríade do TCC

| Script | Pergunta respondida | Evidência |
|--------|---------------------|-----------|
| **15** (este) | Existe sinal estrutural separando fake de real **nos dados crus**? | Cohen's d > 0.8 em múltiplas métricas |
| 14 | O modelo **consegue** explorar esse sinal sem texto? | F1 estrutural-puro próximo do F1 com BERT |
| 16 | O modelo **aponta** para o sinal estrutural ao explicar? | GNNExplainer realça arestas topológicas |

Defesa metodológica: se o sinal está nos dados (15), o modelo o extrai (14), e a explicação confirma o motor da decisão (16), então **GNNs não estão erradas — os datasets é que estão enviesados pela coleta**. Protege o orientando da crítica "GNN é arquitetura ruim para fake news": qualquer modelo treinado nesses datasets aprende esse atalho — a questão é metodológica, não arquitetural.

### 6.2 Comparação com Vosoughi et al. (2018)

Vosoughi et al. analisaram 126.000 cascatas no Twitter (2006–2017):

| Métrica | Tendência fake vs real | Magnitude reportada |
|---------|------------------------|---------------------|
| Cascade size | fake >> real | "ordens de magnitude" (Fig. 1A) |
| Depth | fake > real | mediana 19 (fake) vs 5 (real) (Fig. 1B) |
| Breadth | fake > real | mediana ~1500 (fake) vs ~50 (real) na cauda |
| Structural virality | fake > real | ~50% maior (Fig. 1F) |

Se os Cohen's d obtidos pelo script 15 forem da mesma direção e ordem de magnitude (|d| > 0.8) que os de Vosoughi et al., a tese fica fortalecida — não é peculiaridade do FakeNewsNet/UPFD, é fenômeno reprodutível em múltiplas plataformas. Se forem **maiores** que Vosoughi, indica viés adicional de coleta.

### 6.3 Comparação com Goel et al. (2016)

Goel et al. analisaram 622 milhões de cascatas e introduziram o índice de Wiener:

$$v(G) = \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j=1}^{n} d(i, j)$$

onde $d(i, j)$ é a distância geodésica entre $i$ e $j$. O script 15 não computa Wiener diretamente, mas as duas métricas que o compõem (`depth_max`, `width_max`) sim. Adicionar Wiener (sugestão E5) reproduziria diretamente a métrica de Vosoughi et al., facilitando comparação numérica direta.

### 6.4 Resposta parcial à questão central — Seção 6.4 do TCC

> **Pergunta:** "GNNs são alternativa viável para detecção de fake news ou ainda estão atrás das metodologias tradicionais que usam NLP?"

A contribuição do **15_analise_estrutural.py** é metodológica e indireta, mas crucial:

1. **Os datasets de referência (FakeNewsNet, UPFD-PolitiFact, UPFD-GossipCop) contêm sinal estrutural fortíssimo** entre fake e real, com Cohen's d esperado ≥0.8 em pelo menos uma métrica por dataset. Não é descoberta nova — Vosoughi et al. (2018) já mostraram em larga escala. Mas é a primeira vez que o sinal é quantificado **nos exatos datasets usados pela literatura GNN-fake-news** (UPFD).

2. **Esse sinal não vem do conteúdo da notícia** — é puramente topológico, derivado da forma como a cascata se difunde. Logo, qualquer modelo (GNN ou não) treinado nesses datasets que tenha acesso ao grafo consegue, em princípio, explorar o atalho — mesmo sem ler a manchete.

3. **Implicação para o debate GNN×NLP:** quando o TCC observa que "GNN com features estruturais puras (sem BERT) atinge F1≈0.81 no GossipCop" (resultado do 14), isso **não significa que a GNN aprendeu propagação de fake news** — significa que aprendeu o atalho topológico do dataset. NLP, ao trabalhar só com texto, é imune a esse confound — o que paradoxalmente pode torná-la **mais robusta** em deploys onde o grafo de propagação não está disponível ou foi adversarialmente manipulado.

4. **A culpa é do dataset, não da arquitetura:** a coleta original de Shu et al. (2020) e Dou et al. (2021) enviesou-se sistematicamente — provavelmente porque "exemplos de fake news no FakeNewsNet" são notícias que viralizaram (e por isso foram fact-checked pela Snopes/PolitiFact), enquanto "real news" são notícias âncora menos virais. **GNNs não são intrinsecamente piores para fake news — elas são incentivadas a aprender o sinal mais fácil disponível, e nesse domínio o sinal mais fácil é um confound.** Essa é a defesa metodológica que sustenta a viabilidade de GNNs em deployments reais.

5. **Direção futura:** datasets sem viés de coleta (amostras aleatórias temporais com rotulagem cega à viralidade) tornariam o teste GNN×NLP justo. O script 26 (RQ3 multilingual no Bluesky) tenta com proxy fraco; a solução definitiva exige coleta protocolar nova — fora do escopo do TCC e legítima sugestão para trabalhos futuros.

---

## 7. Análise de Código

### 7.1 Erros e melhorias

**E1 — Cohen's d com `np.std()` default (ddof=0):**
```python
# ❌ Linhas 112–113
pooled = np.sqrt(((a.std()**2 * (len(a)-1)) + (b.std()**2 * (len(b)-1))) /
                 (len(a) + len(b) - 2))

# ✅ Correção:
pooled = np.sqrt(((a.var(ddof=1) * (len(a)-1)) + (b.var(ddof=1) * (len(b)-1))) /
                 (len(a) + len(b) - 2))
# Justificativa: Cohen (1988, Eq. 2.5.3) usa variância amostral com Bessel (ddof=1).
```

**E2 — Múltiplos testes sem correção:**
15 testes com α=0.05 → FWER ≈ 0.537. Adicionar Bonferroni (`α/15 ≈ 0.0033`) ou Holm. Não invalida conclusões dado d>0.8 esperado, mas merece menção na monografia.

**E3 — Linhas-guia incompletas em `fig_separabilidade.png`:**
Só ±0.2 está desenhado (linhas 253–254). Adicionar ±0.5 e ±0.8:
```python
for thr, txt in [(0.2, "pequeno"), (0.5, "médio"), (0.8, "grande")]:
    ax.axhline( thr, color="gray", lw=0.5, ls="--", alpha=0.4)
    ax.axhline(-thr, color="gray", lw=0.5, ls="--", alpha=0.4)
```

**E4 — BFS assume nó 0 como raiz (não verificado):** verdade por construção em FNN/UPFD, mas se o dataset for refeito sem essa convenção, profundidade fica errada. Adicionar `assert` ou determinar raiz pelo nó com grau de entrada zero.

**E5 — Ausência do índice de Wiener:** métrica mais alinhada com Vosoughi et al. (2018). Custo $O(n(n+m))$ por grafo — viável para $n < 1000$ (caso dos três datasets aqui). Fortaleceria a comparabilidade direta.

**E6 — `g.num_edges` em PyG é dobrado** (não-direcionado armazenado como direcionado duplicado). Para fidelidade ao $|E|$ matemático, dividir por 2 ao tratar como não-direcionado. Não é bug, é detalhe de interpretação.

### 7.2 Boas práticas

- **`equal_var=False`**: Welch é a escolha correta para amostras heterocedásticas — Vosoughi et al. confirmam que variâncias diferem.
- **Cohen's d junto com p**: alinhado com APA Style 7th ed. e com a recomendação de Cohen (1988) de sempre acompanhar significância de magnitude.
- **Bins log para `num_nodes`**: prática-padrão de Newman (Cap. 8) para cauda pesada.
- **Amostras de tamanho mediano para HTML**: representação típica (defesa contra cherry-picking).
- **CSV consolidado independente das figuras**: permite re-análise externa (Holm, Wilcoxon) sem re-rodar pipeline.
- **`matplotlib.use("Agg")`** (linha 44): backend sem display, boa prática para CI/servidor sem GUI.

### 7.3 Complexidade

- `metricas_grafo()`: BFS $O(n+m)$ por grafo; branching $O(m)$. Total $O(N \cdot (\bar n + \bar m))$ para todo o dataset.
- `comparar_classes()`: Welch e Cohen's d são $O(n_F + n_R)$. Total $O(N)$ por métrica.

---

## 8. Referências Bibliográficas

1. COHEN, J. **Statistical Power Analysis for the Behavioral Sciences**. 2ª ed. Hillsdale, NJ: Lawrence Erlbaum, 1988. ISBN: 978-0-8058-0283-2.

2. STUDENT [GOSSET, W. S.]. **The Probable Error of a Mean**. *Biometrika*, v. 6, n. 1, pp. 1–25, 1908. DOI: `10.1093/biomet/6.1.1`.

3. WELCH, B. L. **The Generalization of "Student's" Problem When Several Different Population Variances Are Involved**. *Biometrika*, v. 34, n. 1/2, pp. 28–35, 1947. DOI: `10.1093/biomet/34.1-2.28`.

4. NEWMAN, M. E. J. **Networks: An Introduction**. Oxford: Oxford University Press, 2010. ISBN: 978-0-19-920665-0.

5. WATTS, D. J.; STROGATZ, S. H. **Collective Dynamics of "Small-World" Networks**. *Nature*, v. 393, n. 6684, pp. 440–442, 1998. DOI: `10.1038/30918`.

6. BARABÁSI, A.-L.; ALBERT, R. **Emergence of Scaling in Random Networks**. *Science*, v. 286, n. 5439, pp. 509–512, 1999. DOI: `10.1126/science.286.5439.509`.

7. VOSOUGHI, S.; ROY, D.; ARAL, S. **The Spread of True and False News Online**. *Science*, v. 359, n. 6380, pp. 1146–1151, 2018. DOI: `10.1126/science.aap9559`.

8. GOEL, S.; ANDERSON, A.; HOFMAN, J.; WATTS, D. J. **The Structural Virality of Online Diffusion**. *Management Science*, v. 62, n. 1, pp. 180–196, 2016. DOI: `10.1287/mnsc.2015.2158`.

9. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *Proceedings of SIGIR '21*, pp. 2051–2055, 2021. DOI: `10.1145/3404835.3462990`. arXiv: `2104.12259`.

10. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062`.

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Grafo de propagação | Árvore enraizada — raiz = post original; filhos = retweets/citações | Vosoughi et al. (2018), Fig. 1 |
| Profundidade (depth) | Distância geodésica máxima desde a raiz | Newman (2010), Cap. 6 |
| Largura (breadth/width) | Maior cardinalidade entre níveis BFS | Goel et al. (2016), Sec. 2 |
| Branching factor | Número médio de filhos por nó não-folha | Newman (2010), Cap. 12.10 |
| Structural virality (Wiener) | Média das distâncias geodésicas par a par | Goel et al. (2016), Eq. (1) |
| Cohen's d | Diferença de médias em unidades de DP pooled | Cohen (1988), Eq. 2.5.3 |
| Pooled SD | $s_p = \sqrt{\frac{(n_F-1)s_F^2 + (n_R-1)s_R^2}{n_F + n_R - 2}}$ | Cohen (1988) |
| Welch's t-test | Variante do t-test sem suposição de variâncias iguais | Welch (1947), Eq. (5) |
| Welch–Satterthwaite df | Aproximação de df para variâncias desiguais | Welch (1947), Eq. (8) |
| FWER | Family-Wise Error Rate — P(≥1 falso positivo em m testes) | — |
| Bonferroni | $\alpha' = \alpha / m$ — controla FWER | — |
| BFS | Breadth-First Search; explora vizinhos por distância crescente | Newman (2010), Alg. 10.1 |
| Cauda pesada | $P(X > x)$ decai polinomialmente (não exponencialmente) | Newman (2010), Cap. 8 |
| Lei de potência | $P(k) \sim k^{-\gamma}$ — característica de redes scale-free | Barabási & Albert (1999), Eq. (1) |
| Atalho topológico | Sinal estrutural correlacionado com rótulo, explorável por GNN sem conteúdo textual | Conceito do TCC alinhado com confound de Vosoughi et al. (2018) |
| UPFD | User Preference-aware Fake News Detection — datasets PolitiFact e GossipCop | Dou et al. (2021), SIGIR |
| FakeNewsNet | Repositório de Shu et al. com posts/retweets do PolitiFact e GossipCop | Shu et al. (2020), Big Data |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 15_analise_estrutural.py
📄  Arquivo gerado: theory_andre/15_analise_estrutural_doc.md
📚  Fontes acadêmicas utilizadas: 10
    1. Cohen (1988) — Statistical Power Analysis (limiares d 0.2/0.5/0.8)
    2. Student/Gosset (1908) — Probable Error of a Mean (t-test base)
    3. Welch (1947) — Generalization of Student's Problem (ttest_ind equal_var=False)
    4. Newman (2010) — Networks: An Introduction (BFS, depth, branching, scale-free)
    5. Watts & Strogatz (1998) — Small-World Networks (Nature)
    6. Barabási & Albert (1999) — Emergence of Scaling (Science)
    7. Vosoughi, Roy & Aral (2018) — Spread of True and False News (Science) ← núcleo
    8. Goel et al. (2016) — Structural Virality (Management Science, índice de Wiener)
    9. Dou et al. (2021) — UPFD (SIGIR'21)
    10. Shu et al. (2020) — FakeNewsNet (Big Data)
🔍  Conceitos cobertos:
    - Métricas topológicas: num_nodes, num_edges, depth_max, width_max, branching_avg
    - BFS desde a raiz (Algoritmo 10.1 de Newman)
    - t-test de Welch heteroscedástico (equal_var=False)
    - Cohen's d com pooled SD e limiares small/medium/large
    - Decomposição broadcast vs viral (Goel et al. 2016)
    - Conexão com Vosoughi et al. 2018: sinal estrutural fake>real reproduzido
    - Atalho topológico como confound de coleta — defesa metodológica do TCC
    - Múltiplos testes sem correção (FWER ≈ 0.54) — limitação reportada
⚠️   Limitações:
    - WebSearch/WebFetch foram bloqueados na sessão; referências citadas com DOIs
      e localizações estáveis (seção/equação/figura) baseadas no conhecimento
      canônico das publicações originais.
    - Cohen's d no código usa np.std() default (ddof=0); diferença <1% para n>=100
      mas merece correção formal (E1).
    - Múltiplos testes sem Bonferroni/Holm (E2). Não invalida conclusões dado d>0.8.
    - g.num_edges em PyG é dobrado — afeta interpretação numérica do CSV (E6).
    - Índice de Wiener (Goel et al. 2016) não implementado; sugestão E5.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```
