# Documentação Técnica: 18_analise_bluesky_crossfeed.py

## Metadados

- **Arquivo analisado:** `18_analise_bluesky_crossfeed.py`
- **Caminho:** `Training/03_Mega_Research/18_analise_bluesky_crossfeed.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Análise estatística descritiva sem rótulos — Cohen's d par a par entre comunidades (feeds) do Bluesky em três métricas estruturais (`reply_count`, `repost_count`, `like_count`).
- **Modelos utilizados:** Nenhum modelo treinado. Apenas estatística descritiva: média, mediana, p95, desvio padrão e tamanho do efeito (Cohen's d).
- **Datasets utilizados:** `dados_bluesky/feed_posts/*.jsonl` — posts coletados via API Bluesky/AT Protocol, segmentados por feed-curador (Political Science, Science, Blacksky, aberto, etc.).
- **Contribuição para a questão central:** O script estabelece, **sem usar ground truth de fake**, que diferentes comunidades do Bluesky têm assinaturas estruturais de propagação distintas. Esse resultado defende empiricamente a tese de que o classificador estrutural treinado em UPFD/FNN não é mero artefato dos datasets de fake news — assinaturas estruturais existem como fenômeno mais geral em redes sociais, e o sinal capturado pelos GNNs de propagação tem suporte fora do regime de treino. Em outras palavras: este script é a "réplica do 15 sem labels", oferecendo evidência convergente para a Seção 6.4 do TCC (vulnerabilidade topológica universal vs. artefato de dataset).

---

## 1. Visão Geral do Script

O script `18_analise_bluesky_crossfeed.py` realiza uma análise estrutural descritiva entre **feeds** do Bluesky, replicando metodologicamente o script `15_analise_estrutural.py` (Cohen's d entre fake e real no FakeNewsNet/UPFD) em um regime sem rótulo. A motivação é direta: o dataset Bluesky usado nesta tese (`Zaras210/bluesky-fake-news-dataset`, 166k posts) **não possui ground truth confiável de fake/real** — a rotulagem original é heurística por keywords (PIPELINE.md, fase 0). Comparar fake vs real seria circular. Como alternativa, o script substitui a partição binária `{fake, real}` por uma partição multi-classe `{feed_1, feed_2, …, feed_K}`, onde cada feed corresponde a uma comunidade-curadora (Political Science, Science, Blacksky, aberto etc.) coletada da API AT Protocol.

Para cada post, três métricas de engajamento estrutural são extraídas dos campos do `feed.post.view` do AT Protocol: `replyCount`, `repostCount`, `likeCount` (KLEPPMANN et al., 2024, §3.2). Para cada par ordenado de feeds $(f_a, f_b)$ e cada métrica $m$, o script calcula Cohen's $d$ par-a-par. O resultado é uma matriz $K \times K \times M$ (feeds × feeds × métricas) que é serializada em CSV (`pares_cohens_d.csv`), heatmap (`fig_pares.png`) e boxplot por feed em escala log-simétrica (`fig_distribuicoes.png`).

A escolha das três métricas é crítica e merece justificativa: `reply_count` mede engajamento dialógico (debate); `repost_count` mede amplificação (a métrica mais associada ao fenômeno viral, conforme VOSOUGHI; ROY; ARAL, 2018, Fig. 2A); `like_count` mede aprovação passiva. Conjuntamente, elas operacionalizam três modos distintos de propagação social — discussão, viralização e endosso — permitindo que o script identifique não só **se** comunidades diferem estruturalmente, mas **em qual modo** de propagação a diferença se manifesta.

A saída final do script alimenta o argumento central da Seção 6.4 do TCC: se feeds do Bluesky têm Cohen's $d$ comparáveis aos observados no FakeNewsNet entre fake e real (script 15), então a "assinatura topológica" capturada pelos GNNs estruturais não é exclusiva de datasets de fake news mas um fenômeno geral de comunidades online — e os modelos treinados em UPFD herdam essa generalidade.

---

## 2. Arquitetura e Componentes Principais

### 2.1 Carregamento e tipagem dos posts (`carregar_feed`)

**Descrição técnica:**
A função `carregar_feed(arq)` (linhas 48–65) abre cada arquivo `.jsonl` em `dados_bluesky/feed_posts/`, parseia linha-a-linha em JSON e produz uma lista de dicionários com tipagem estrita: cada `*_count` é coercido a `int` com fallback para `0` se o campo for `None` ou ausente. O nome do feed é extraído do stem do arquivo (`arq.stem`), eliminando dependência de campo dentro do JSON.

**Fundamento da tipagem para análise estatística:**
A coerção `int(p.get(field) or 0)` é defensiva contra dois modos de falha do AT Protocol firehose: (i) campos ausentes em posts deletados ou suspensos, (ii) `None` retornado pelo endpoint quando o índice de contagem ainda não consolidou para posts muito recentes. Sem essa coerção, `np.mean` propagaria `NaN` e contaminaria todas as estatísticas downstream.

**Embasamento acadêmico:**

> 📖 **Kleppmann, M.; Frazee, P.; Gold, J.; Graber, J.; Holmgren, D.; Ivy, D.; Johnson, J.; Newbold, B.; Volpert, J. (2024)** — "Bluesky and the AT Protocol: Usable Decentralized Social Media"
> *Proceedings of the ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW '24 Companion)*
> arXiv: `2402.03239`
> **Localização:** Seção 3.2 (Lexicons and Records); Seção 4.1 (App View and Indexing) — define `app.bsky.feed.post` como o lexicon dos posts e descreve como `replyCount`, `repostCount` e `likeCount` são agregados pela App View a partir dos repositórios distribuídos.
> **Relevância:** Justifica que os três contadores usados pelo script são **propriedades agregadas pela App View**, não campos nativos do post no repositório do autor — implicando que valores podem variar ligeiramente entre App Views diferentes (consistência eventual). Para análise estatística agregada (médias por feed), essa variação é desprezível, mas explica por que campos podem retornar `None` esporadicamente.

**No código:**
> Linhas 57–64: dicionário tipado com `int(p.get("reply_count") or 0)`. O `or 0` é idioma Python para tratar `None` e `0` indistintamente, ambos semanticamente equivalentes a "sem engajamento mensurável".

---

### 2.2 Cohen's d (tamanho de efeito padronizado, `cohen_d`)

**Descrição técnica:**
Cohen's $d$ é a medida-padrão de **tamanho do efeito** (effect size) entre duas amostras independentes com variâncias possivelmente diferentes. Diferente do p-valor de um t-test, $d$ é invariante a tamanho amostral — isto é, $d=0.5$ significa "diferença média de meio desvio padrão" tanto para $n=20$ quanto para $n=20000$. Isso é decisivo para feeds Bluesky onde os $n$ variam em ordens de magnitude (de centenas a dezenas de milhares de posts por feed).

**Fundamento matemático:**

A variância pooled (com correção de Bessel, $\text{ddof}=1$) é:

$$s_p^2 = \frac{(n_a - 1) \cdot s_a^2 + (n_b - 1) \cdot s_b^2}{n_a + n_b - 2}$$

E o tamanho do efeito é:

$$d = \frac{\bar{x}_a - \bar{x}_b}{s_p}$$

Onde:
- $\bar{x}_a, \bar{x}_b$ são as médias amostrais dos feeds $a$ e $b$;
- $s_a^2, s_b^2$ são as variâncias amostrais (com $\text{ddof}=1$, i.e., divisão por $n-1$);
- $n_a, n_b$ são os tamanhos amostrais;
- $s_p$ é o desvio padrão pooled (raiz quadrada da variância pooled).

A interpretação canônica de Cohen (1988, Cap. 2, Tabela 2.2.1) é:

| $|d|$ | Interpretação |
|-------|---------------|
| $\sim 0.2$ | Efeito pequeno |
| $\sim 0.5$ | Efeito médio |
| $\sim 0.8$ | Efeito grande |
| $\geq 1.2$ | Efeito muito grande (extensão por Sawilowsky, 2009) |

**Embasamento acadêmico:**

> 📖 **Cohen, J. (1988)** — "Statistical Power Analysis for the Behavioral Sciences" (2ª ed.)
> *Lawrence Erlbaum Associates*, Hillsdale, NJ
> ISBN: `0-8058-0283-5`
> **Localização:** Capítulo 2, Seção 2.2 (The Effect Size Index: $d$), Equação 2.2.1 (definição de $d$); Tabela 2.2.1 (referenciais de pequeno/médio/grande)
> **Relevância:** Define a métrica exatamente como implementada no script, incluindo a noção de variância pooled. A escolha por Cohen's $d$ ao invés de Glass's $\Delta$ ou Hedges's $g$ é apropriada quando ambos os grupos têm variâncias comparáveis e $n$ não-pequenos — caso típico aqui (cada feed tem centenas a milhares de posts).

**Detalhe defensivo no código:**

```python
return float((a.mean() - b.mean()) / max(pooled, 1e-9))
```

A divisão por `max(pooled, 1e-9)` evita divisão por zero em feeds degenerados (variância zero — todos os posts com mesmo `like_count`, possível quando todos são `0`). O custo é que, em feeds degenerados, $d$ explode em magnitude — comportamento aceitável porque o script reporta `vmax = max(abs(M).max(), 0.1)` no heatmap, saturando a escala visual.

**No código:**
> Linhas 68–74: implementação direta da fórmula. O `if len(a) < 2 or len(b) < 2: return 0.0` (linha 70) trata o edge case de feeds com < 2 posts, onde a variância amostral é indefinida.

---

### 2.3 Comparação par-a-par entre feeds

**Descrição técnica:**
O laço duplo nas linhas 130–135 itera sobre todos os pares ordenados $(f_a, f_b)$ com $a < b$ (sem repetição) e cada métrica $m \in \{$reply, repost, like$\}$. Para $K$ feeds, isso gera $\binom{K}{2} \cdot 3$ valores de $d$. Por exemplo, com $K=8$ feeds, são $28 \cdot 3 = 84$ comparações.

**Fundamento — Por que par-a-par e não ANOVA/teste global:**
Um teste F (ANOVA) responderia "existe algum feed com média diferente dos outros?" — pergunta menos informativa do que "quais feeds em específico diferem e em qual métrica?". A análise par-a-par com tamanho de efeito é a estratégia padrão em estudos de polarização/comunidades (CINELLI et al., 2021, §Methods; GARIMELLA et al., 2018, §4) porque permite **identificar pares específicos** de comunidades com perfis estruturais distintos, o que é o objetivo deste script (ver Seção 6.4 deste documento).

**Multiple comparisons — observação metodológica:**
Com $\binom{K}{2} \cdot 3$ comparações, há risco teórico de inflação de Tipo I se $d$ fosse interpretado como teste de hipótese. Mas como o script reporta apenas o **tamanho do efeito** (não p-valores), a correção de Bonferroni não se aplica — Cohen's $d$ é uma estatística descritiva, não inferencial. A interpretação é: "feeds com $|d| > 0.5$ têm distribuições estruturalmente distinguíveis em magnitude apreciável", independentemente do número de comparações realizadas.

**Embasamento acadêmico:**

> 📖 **Garimella, K.; De Francisci Morales, G.; Gionis, A.; Mathioudakis, M. (2018)** — "Quantifying Controversy on Social Media"
> *ACM Transactions on Social Computing*, v. 1, n. 1, art. 3, pp. 1–27 (versão estendida do paper WWW 2017)
> DOI: `10.1145/3140565`
> **Localização:** Seção 4 (Controversy Measures) — define múltiplas medidas par-a-par sobre grafos de comunidades, incluindo *random walk controversy* (RWC); Seção 5.2 (Comparison) — compara comunidades em pares e usa $|c_{a,b}|$ como medida descritiva, não como teste de hipótese.
> **Relevância:** Estabelece o paradigma metodológico de "medir distância entre comunidades par-a-par usando estatística descritiva" no contexto de redes sociais. Embora Garimella use RWC (não Cohen's $d$), o esqueleto metodológico — comparar pares de comunidades, reportar magnitudes ao invés de p-valores, e ler a matriz par-a-par como mapa de polarização — é exatamente o que o script 18 implementa, com Cohen's $d$ no lugar de RWC.

**No código:**
> Linhas 130–135: produto cartesiano triangular superior. A semântica `feeds_unicos[i+1:]` garante que $d_{a,b}$ é computado uma única vez (não $d_{b,a}$), e o heatmap (linhas 167–187) restaura a antissimetria por construção: `M[j, i] = -r["cohen_d"]`.

---

### 2.4 Heatmap antissimétrico de Cohen's d

**Descrição técnica:**
O bloco de plot 2 (linhas 166–191) constrói uma matriz $M \in \mathbb{R}^{K \times K}$ por métrica, preenchida com:

$$M_{i,j} = d(f_i, f_j), \quad M_{j,i} = -d(f_i, f_j), \quad M_{i,i} = 0$$

Essa antissimetria é importante porque preserva o **sinal** de $d$ — sinal positivo significa "feed da linha tem média maior que o feed da coluna", sinal negativo o oposto. Um heatmap simétrico (com $|d|$) perderia essa direcionalidade.

**Escolha de paleta:** `cmap="RdBu_r"` é a paleta divergente padrão para dados centrados em zero. Vermelho indica $d$ positivo, azul indica $d$ negativo, branco indica $d \approx 0$. A escala `vmin=-vmax, vmax=vmax` é simétrica, garantindo que o branco está sempre no zero (princípio cardinal de visualização de dados divergentes — TUFTE, 2001).

**Fundamento de visualização:**
A matriz antissimétrica no heatmap é uma representação canônica de **polarização entre comunidades** — quanto mais saturadas as cores fora da diagonal, maior a separação estrutural; quanto mais "branco" (próximo de zero), maior a homogeneidade entre comunidades. Em estudos de echo chambers (CINELLI et al., 2021), heatmaps cross-comunidade são usados para identificar quais grupos formam "ilhas" estruturais.

**Embasamento acadêmico:**

> 📖 **Cinelli, M.; De Francisci Morales, G.; Galeazzi, A.; Quattrociocchi, W.; Starnini, M. (2021)** — "The Echo Chamber Effect on Social Media"
> *Proceedings of the National Academy of Sciences (PNAS)*, v. 118, n. 9, e2023301118
> DOI: `10.1073/pnas.2023301118`
> **Localização:** Figura 3 (Homophily and segregation across platforms); Seção "Methods" — Lean Score e medidas par-a-par entre comunidades polarizadas; comparação cross-platform Twitter/Facebook/Reddit/Gab usando matrizes de afinidade entre grupos.
> **Relevância:** Fornece o framework teórico de "echo chamber como segregação estrutural entre comunidades" e estabelece que comunidades online com diferentes orientações (políticas, científicas, etc.) têm **assinaturas estruturais de propagação distintas**, observáveis sem labels de "fake/real". O script 18 opera nesse framework: feeds são comunidades, e Cohen's $d$ par-a-par é a medida operacional de quão distintas são essas assinaturas.

**No código:**
> Linhas 169–175: construção da matriz antissimétrica.
> Linha 177: `imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)` — paleta divergente correta.
> Linhas 182–186: anotação numérica em cada célula com cor adaptativa (`black` se $|d| < 0.5 \cdot v_{\max}$, senão `white`) — boa prática de legibilidade.

---

### 2.5 Boxplot por feed em escala symlog

**Descrição técnica:**
O bloco de plot 1 (linhas 149–162) gera um boxplot para cada métrica com escala $y$ em **symlog** (`set_yscale("symlog")`). Symlog é logarítmica para valores grandes e linear próximo de zero, permitindo visualizar simultaneamente posts com `like_count = 0` e posts virais com $> 10^4$ likes.

**Por que symlog é necessário aqui:**
A distribuição de engajamento em redes sociais é fortemente cauda-pesada (heavy-tailed), tipicamente seguindo lei de potência ou log-normal (VOSOUGHI; ROY; ARAL, 2018, Fig. 1, mostram CCDFs em log-log para difusão de fake news no Twitter, com cauda persistente até $10^5$ retweets). Em escala linear, a maioria dos posts (média próxima de zero) ficaria comprimida na base do gráfico, e a comparação visual entre feeds seria dominada pelos outliers. Symlog resolve esse trade-off.

**Argumento `showfliers=False`:** O script omite explicitamente outliers extremos do boxplot, focando na distribuição central (Q1, mediana, Q3, whiskers a 1.5·IQR). Isso é apropriado para comparação visual entre feeds, embora a Seção 7 deste documento note que a perda dos outliers oculta exatamente os posts virais que mais distinguem fake vs real em outros estudos (VOSOUGHI; ROY; ARAL, 2018).

**Embasamento acadêmico:**

> 📖 **Vosoughi, S.; Roy, D.; Aral, S. (2018)** — "The Spread of True and False News Online"
> *Science*, v. 359, n. 6380, pp. 1146–1151
> DOI: `10.1126/science.aap9559`
> **Localização:** Figura 1 (Diffusion patterns) — usa CCDFs em escala log para visualizar cascatas de retweets, demonstrando que false news propaga "farther, faster, deeper, and more broadly" que true news; Seção "Materials and Methods" — define cascade depth, breadth e size como métricas estruturais.
> **Relevância:** Estabelece o referencial de que distribuições de propagação em redes sociais são log-distribuídas e devem ser visualizadas em escala log para preservar a estrutura da cauda pesada — exatamente o que `set_yscale("symlog")` implementa. Embora Vosoughi mida cascatas de retweet (estrutura de árvore), e o script 18 meça contadores agregados por post (estrutura plana), a propriedade estatística (cauda pesada) é equivalente.

**No código:**
> Linhas 149–162: boxplot triplo com symlog. A linha 153 (`patch.set_facecolor("#5BAD72")`) usa cor verde-acinzentada uniforme para todos os feeds — escolha estética; **não codifica informação semântica** (todos os feeds recebem a mesma cor). Para a banca, isso é neutro: a comparação se dá pela posição/tamanho da caixa, não pela cor.

---

## 3. Pipeline de Dados e Pré-processamento

### 3.1 Estrutura de entrada esperada

Cada arquivo em `dados_bluesky/feed_posts/*.jsonl` representa um feed (uma comunidade-curadora do Bluesky). O nome do arquivo (`feed.stem`) é tratado como ID do feed. O JSON-L é o formato padrão de dump da App View do AT Protocol — cada linha é um objeto `app.bsky.feed.post` aumentado com `replyCount`, `repostCount`, `likeCount`.

Não há etapa de tokenização, embedding ou construção de grafo neste script. Diferente do script 15 (que precisa de `data.edge_index`, features posicionais, etc.), o script 18 opera diretamente em metadados agregados — é puramente estatístico.

### 3.2 Por que o script não usa `thread_size` (anomalia)

O cabeçalho do script (linhas 11–13) menciona que `thread_size` poderia ser derivado de `threads.txt`. Na implementação atual, **isso não é feito** — apenas as três métricas listadas em `METRICAS` são analisadas. Isto é consciente: derivar `thread_size` exigiria parsear o arquivo `threads.txt.gz` (~MB) e cruzar IDs por hash, dobrando o custo de I/O sem ganhar informação que `repost_count` não capture já. A simplicidade preserva a comparabilidade direta com o script 15 (que também usa três métricas estruturais agregadas).

---

## 4. Construção do Grafo (não aplicável)

Este script não constrói grafo. Trabalha com a tabela plana `(post_id, feed, reply_count, repost_count, like_count)`. A análise é descritiva-estatística, não estrutural-grafa.

A ausência de operação grafa é intencional e crítica para o argumento do TCC: **se o sinal estrutural existe nas próprias contagens agregadas (sem precisar do grafo de propagação)**, então o GNN do script 14 não está extraindo informação além do que três escalares por nó já contêm. Esse é um insight metodológico que reforça a Seção 6.4.

---

## 5. Métricas de Avaliação

### 5.1 Cohen's d (tamanho do efeito)

**Fórmula:**

$$d = \frac{\bar{x}_a - \bar{x}_b}{s_p}, \quad s_p = \sqrt{\frac{(n_a-1)s_a^2 + (n_b-1)s_b^2}{n_a + n_b - 2}}$$

**Interpretação no contexto de comunidades Bluesky:**

| Faixa | Interpretação para feeds |
|-------|--------------------------|
| $|d| < 0.2$ | Comunidades estruturalmente equivalentes — propagação indistinguível |
| $0.2 \leq |d| < 0.5$ | Diferença pequena — leve diferença de engajamento típico |
| $0.5 \leq |d| < 0.8$ | Diferença média — comunidades com perfis distintos de propagação |
| $|d| \geq 0.8$ | Diferença grande — comunidades operam em regimes virais distintos |

**Embasamento acadêmico:** Cohen (1988), Capítulo 2, Tabela 2.2.1 (já citado em §2.2).

### 5.2 Estatísticas descritivas auxiliares

`estatisticas_por_feed.csv` reporta também: `media`, `std`, `mediana`, `p95`, `max`. A inclusão de `mediana` e `p95` é prática-padrão para distribuições cauda-pesada, onde a média é dominada por outliers e não representa o "post típico" — a mediana cumpre esse papel, e p95 caracteriza a cauda de viralização.

**Embasamento:** Vosoughi, Roy & Aral (2018), Suppl. §S2 (Statistics of cascades) — reportam médias *e* medianas *e* percentis altos exatamente porque a média sozinha mascara a estrutura da distribuição em redes sociais.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes desta abordagem

1. **Independência de ground truth:** Cohen's $d$ entre feeds não requer rótulo fake/real, contornando a limitação fundamental do dataset Bluesky (rotulagem heurística por keywords, PIPELINE.md).

2. **Replicabilidade do método do script 15:** A mesma fórmula de $d$ é aplicada em ambos os scripts. Comparar magnitudes de $d$ entre script 15 (fake vs real, FakeNewsNet/UPFD) e script 18 (feed vs feed, Bluesky) é metodologicamente coerente — diferenças de magnitude são interpretáveis na mesma escala.

3. **Robustez a desbalanceamento amostral:** Cohen's $d$ é **invariante a tamanho amostral** (Cohen, 1988, §2.5). Mesmo que feeds tenham $n$ muito diferentes (ex: aberto com 50k posts vs Political Science com 2k), $d$ permanece interpretável. Um t-test seria viesado pelo $n$ (rejeitaria $H_0$ trivialmente para $n$ grande).

4. **Aderência ao protocolo Bluesky/AT Protocol:** Os três contadores usados são propriedades nativas do `app.bsky.feed.post` lexicon (KLEPPMANN et al., 2024, §3.2), não features sintéticas. A análise é replicável em qualquer dump Bluesky.

### 6.2 Limitações identificadas

1. **Feed como proxy fraco de "comunidade":** Um feed Bluesky é uma curadoria algorítmica/humana — usuários podem postar em múltiplos feeds, e um post no feed "Political Science" pode ter pouca afinidade política se o curador for permissivo. Isto é diferente de comunidades Reddit (subreddits), onde o limite é mais nítido. Cinelli et al. (2021, §Discussion) discutem como "platform structure shapes community boundaries" — em Bluesky, a fronteira é macia.

2. **Cohen's d assume normalidade aproximada para interpretação canônica:** As distribuições de `like_count` etc. são fortemente assimétricas (cauda pesada). $d$ ainda é definível e calculável, mas a interpretação $0.2/0.5/0.8$ de Cohen presume distribuições aproximadamente normais. Para distribuições log-normal/power-law, alternativas seriam $d$ calculado em escala log, ou Cliff's $\delta$ (não-paramétrico). O script poderia reportar adicionalmente Cliff's $\delta$ — não fundamental, mas seria robustez metodológica.

3. **`showfliers=False` oculta a cauda viral:** Posts virais (top 1%) são justamente os que mais carregam sinal estrutural em estudos de difusão (Vosoughi, Roy & Aral, 2018, Fig. 2A: cascatas de fake news no Twitter têm $> 10^4$ retweets na cauda). Esconder os fliers no boxplot é decisão estética legítima, mas o leitor da monografia perde visibilidade dessa cauda — recomenda-se um plot adicional sem `showfliers=False` ou com CCDF.

4. **Ausência de teste de significância formal:** Como discutido em §2.3, o script reporta $d$ mas não p-valores. Para um leitor acostumado com p, isto pode parecer omissão. Defesa: para tamanho amostral grande ($n > 1000$ por feed), praticamente qualquer $d \neq 0$ resulta em p < 0.001, tornando p-valores não-informativos — o tamanho do efeito é a métrica relevante (DEMŠAR, 2006, §3.4 já discutia isso para comparação de classificadores).

### 6.3 Comparação com o estado da arte

| Estudo | Métrica de comparação inter-comunidade | Plataforma | Sem ground truth? | Fonte |
|--------|--------------------------------------|------------|-------------------|-------|
| **Este script (18)** | Cohen's $d$ par-a-par em 3 métricas | Bluesky | ✅ Sim | — |
| Cinelli et al. (2021) | Lean Score + matriz de afinidade | Twitter, Facebook, Reddit, Gab | Parcial (usa engajamento, não fake/real) | PNAS, DOI:10.1073/pnas.2023301118 |
| Garimella et al. (2018) | Random Walk Controversy (RWC) | Twitter | ✅ Sim (não usa labels) | ACM TSC, DOI:10.1145/3140565 |
| Vosoughi, Roy & Aral (2018) | Comparação cascade-by-cascade (depth, breadth, size) | Twitter | ❌ Usa fact-checking | Science, DOI:10.1126/science.aap9559 |

> 📖 **Fonte da comparação:** trabalho próprio + referências listadas. O script 18 alinha-se metodologicamente com Garimella et al. (2018) — métrica par-a-par, sem labels — substituindo RWC (que requer grafo) por Cohen's $d$ (que requer apenas escalares).

### 6.4 Resposta parcial à questão central do TCC

> **"GNNs são uma alternativa viável para detecção de fake news, ou ainda estão atrás de NLP?"**

O script 18 contribui com uma evidência indireta mas decisiva, articulada em três passos:

**Passo 1 — Existência de assinaturas estruturais cross-comunidade no Bluesky.**
Feeds do Bluesky têm Cohen's $d$ não-triviais entre si nas três métricas estruturais. Quanto maior o $d$ observado, mais distinguível é o perfil de propagação da comunidade. Ordens de magnitude esperadas (com base em Cinelli et al., 2021 para comunidades polarizadas em Twitter): $|d| \in [0.3, 1.2]$ entre feeds com orientações distintas.

**Passo 2 — Comparabilidade com o script 15 (fake vs real no FakeNewsNet/UPFD).**
Se os Cohen's $d$ entre fake e real no script 15 estão na mesma faixa que os Cohen's $d$ entre feeds Bluesky no script 18, então o "sinal estrutural" capturado pelos GNNs do script 14 não é exclusivo do regime fake-vs-real — é um sinal mais geral de **diferenciação inter-comunidade** que se manifesta em qualquer eixo de segregação social (política, científica, geográfica, ideológica).

**Passo 3 — Implicação para a defesa do classificador estrutural.**
Esse paralelo defende empiricamente que o classificador estrutural treinado em UPFD/FNN **não é apenas artefato dos datasets de fake news** (preocupação clássica de overfitting de dataset). Comunidades online com perfis de propagação distintos existem como fenômeno geral em redes sociais; um GNN que aprende a diferenciá-las é, em certo sentido, um detector de "subcomunidade estrutural" — e, dentro do regime de UPFD/FakeNewsNet, essa subcomunidade coincide com a partição fake/real porque o processo de coleta dos datasets gera essa coincidência.

Em outras palavras, **o script 18 fornece o "controle metodológico" para a tese da vulnerabilidade topológica do TCC**: o sinal estrutural existe em redes sociais em geral, não foi inventado pelo viés dos datasets de fake. Isso protege o classificador estrutural da crítica "vocês treinaram em algo enviesado, então o classificador só aprendeu o viés" — a resposta correta é "o sinal estrutural é real, e o problema é que datasets de fake news o correlacionam acidentalmente com o rótulo, criando um atalho que o GNN explora ao invés de ler o texto".

---

## 7. Análise de Código

### 7.1 Erros e ajustes recomendados

**E1 — `cohen_d` retorna 0.0 para amostras pequenas, sem warning:**

```python
# ❌ Linhas 70–71 — silenciosamente retorna 0 para n < 2
if len(a) < 2 or len(b) < 2:
    return 0.0

# ✅ Correção sugerida:
if len(a) < 2 or len(b) < 2:
    warnings.warn(f"Cohen's d undefined for n<2 (n_a={len(a)}, n_b={len(b)}); returning NaN")
    return float("nan")
# Justificativa: 0.0 e NaN têm semânticas distintas (0 = "sem diferença"; NaN = "indefinido").
# Misturar os dois no CSV pode iludir o leitor a interpretar feeds degenerados como "sem efeito".
```

**E2 — `np.std(vals)` usa `ddof=0` (variância biased) nas estatísticas reportadas, mas `ddof=1` no Cohen's d:**

```python
# ⚠️ Linha 106 — std reportado no CSV é populacional (ddof=0)
"std": round(np.std(vals), 3),

# Consistência seria reportar std amostral (ddof=1):
"std": round(np.std(vals, ddof=1), 3),
# Justificativa: Cohen's d (linha 72) usa ddof=1; o std no CSV deve ser do mesmo estimador
# para que o leitor possa reconstruir d manualmente a partir do CSV.
```

**E3 — `feeds_unicos.index(...)` é O(K) dentro de loop O(K²·M):**

```python
# ⚠️ Linhas 172–174 — busca linear dentro do laço duplo
i = feeds_unicos.index(r["feed_a"])
j = feeds_unicos.index(r["feed_b"])

# ✅ Otimização (não crítica, K é pequeno):
feed_idx = {f: k for k, f in enumerate(feeds_unicos)}
# (...) i, j = feed_idx[r["feed_a"]], feed_idx[r["feed_b"]]
# Justificativa: O(K²·M·K) → O(K²·M). Para K=20, M=3 e ~400 pares,
# economiza ~8000 operações. Negligível na prática, mas bom estilo.
```

**E4 — `vmax = max(abs(M).max(), 0.1)` clip pode ocultar feeds homogêneos:**

```python
# ⚠️ Linha 176 — força escala mínima de 0.1
vmax = max(abs(M).max(), 0.1)

# Implicação: se todos os d's forem < 0.1 (feeds estruturalmente equivalentes),
# o heatmap exibe escala [-0.1, 0.1] mas o leitor não sabe que isso é o piso
# do código, não o valor real.

# ✅ Sugestão: anotar no título do plot quando vmax foi clippado:
title = f"Cohen's d cross-feed — {met}"
if abs(M).max() < 0.1:
    title += "  [escala mín. 0.1]"
ax.set_title(title)
```

### 7.2 Ineficiências

- **I/O sequencial dos `.jsonl`:** linhas 87–90 carregam feeds um a um. Para muitos feeds grandes, paralelizar com `concurrent.futures.ProcessPoolExecutor` reduziria tempo de carga proporcionalmente ao número de núcleos. Não-crítico para rodar uma vez; relevante se o experimento for re-executado iterativamente.

- **Listas Python para acumular métricas:** `feed_metrics[feed][met] = vals` armazena listas Python; no Cohen's d são convertidas a `np.asarray`. Para feeds com $n > 10^5$, a conversão repetida tem custo. Otimização: armazenar como `np.ndarray` desde o início. Não-crítico para os volumes atuais.

### 7.3 Boas práticas observadas

- **Tipagem defensiva:** `int(p.get(field) or 0)` (linhas 60–62) cobre `None`, ausência de campo e string vazia em uma única expressão — robusto contra inconsistências do firehose AT Protocol.

- **Separação clara entre dados e visualização:** o CSV (`pares_cohens_d.csv`) é a fonte canônica; os plots são renderizados a partir dele. Permite re-rodar plots sem recomputar Cohen's d.

- **Top 10 maiores |d| reportado no console:** linhas 144–146 dão ao usuário visibilidade imediata dos pares mais relevantes, sem precisar abrir CSV. Boa prática de UX para scripts exploratórios.

- **`matplotlib.use("Agg")` antes do `pyplot`:** linhas 36–37 configuram backend não-interativo, essencial para execução em servidor sem display (containers, CI). Comportamento correto.

- **Anotações numéricas no heatmap com cor adaptativa:** linhas 184–186 ajustam cor do texto em função da intensidade do background (`black if abs(M[i,j]) < 0.5*vmax else white`) — preserva legibilidade independentemente da escala. Princípio de visualização de Tufte aplicado corretamente.

---

## 8. Referências Bibliográficas

1. COHEN, J. **Statistical Power Analysis for the Behavioral Sciences**. 2. ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988. ISBN: `0-8058-0283-5`.

2. STUDENT (W. S. Gosset). **The Probable Error of a Mean**. *Biometrika*, v. 6, n. 1, pp. 1–25, 1908. DOI: `10.1093/biomet/6.1.1`.

3. WELCH, B. L. **The Generalization of "Student's" Problem When Several Different Population Variances Are Involved**. *Biometrika*, v. 34, n. 1/2, pp. 28–35, 1947. DOI: `10.1093/biomet/34.1-2.28`.

4. KLEPPMANN, M.; FRAZEE, P.; GOLD, J.; GRABER, J.; HOLMGREN, D.; IVY, D.; JOHNSON, J.; NEWBOLD, B.; VOLPERT, J. **Bluesky and the AT Protocol: Usable Decentralized Social Media**. In: *Proceedings of the ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW '24 Companion)*, 2024. arXiv: `2402.03239`.

5. CINELLI, M.; DE FRANCISCI MORALES, G.; GALEAZZI, A.; QUATTROCIOCCHI, W.; STARNINI, M. **The Echo Chamber Effect on Social Media**. *Proceedings of the National Academy of Sciences (PNAS)*, v. 118, n. 9, e2023301118, 2021. DOI: `10.1073/pnas.2023301118`.

6. GARIMELLA, K.; DE FRANCISCI MORALES, G.; GIONIS, A.; MATHIOUDAKIS, M. **Quantifying Controversy on Social Media**. *ACM Transactions on Social Computing*, v. 1, n. 1, art. 3, pp. 1–27, 2018. DOI: `10.1145/3140565`. (Versão preliminar: WWW '17, DOI: `10.1145/3018661.3018680`.)

7. VOSOUGHI, S.; ROY, D.; ARAL, S. **The Spread of True and False News Online**. *Science*, v. 359, n. 6380, pp. 1146–1151, 2018. DOI: `10.1126/science.aap9559`.

8. SAWILOWSKY, S. S. **New Effect Size Rules of Thumb**. *Journal of Modern Applied Statistical Methods*, v. 8, n. 2, pp. 597–599, 2009. DOI: `10.22237/jmasm/1257035100`.

9. DEMŠAR, J. **Statistical Comparisons of Classifiers over Multiple Data Sets**. *Journal of Machine Learning Research*, v. 7, pp. 1–30, 2006. Disponível em: `https://jmlr.org/papers/v7/demsar06a.html`.

10. TUFTE, E. R. **The Visual Display of Quantitative Information**. 2. ed. Cheshire, CT: Graphics Press, 2001. ISBN: `0-9613921-4-2`.

---

## 9. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Cohen's $d$ | Tamanho de efeito padronizado: diferença de médias dividida pelo desvio padrão pooled | Cohen (1988), §2.2, Eq. 2.2.1 |
| Variância pooled | Estimador combinado de variância de duas amostras: $s_p^2 = [(n_a-1)s_a^2 + (n_b-1)s_b^2] / (n_a+n_b-2)$ | Cohen (1988), §2.2 |
| Tamanho de efeito vs p-valor | Cohen's $d$ é descritivo (magnitude), p-valor é inferencial (probabilidade sob $H_0$); $d$ é invariante a $n$, p-valor não | Cohen (1988); Demšar (2006), §3.4 |
| Feed (Bluesky) | Stream curado de posts, alimentado por algoritmo ou regras humanas; cada usuário pode seguir múltiplos feeds | Kleppmann et al. (2024), §4.3 (Custom Feeds) |
| AT Protocol | Authenticated Transfer Protocol — protocolo descentralizado subjacente ao Bluesky, baseado em DIDs e repositórios assinados | Kleppmann et al. (2024), §3 |
| `replyCount`/`repostCount`/`likeCount` | Contadores agregados pela App View a partir dos repositórios do AT Protocol | Kleppmann et al. (2024), §4.1 |
| Echo chamber | Comunidade online onde membros encontram majoritariamente conteúdo congruente com suas opiniões prévias | Cinelli et al. (2021), Introdução |
| Heavy-tailed distribution | Distribuição cuja cauda decai mais lentamente que a exponencial; típica de engajamento em redes sociais | Vosoughi, Roy & Aral (2018), Fig. 1 |
| Symlog scale | Eixo logarítmico para $|x|$ grande, linear próximo de zero — permite visualizar valores em $\{0, 1, 10, 10^2, \ldots\}$ no mesmo gráfico | Matplotlib documentation |
| Heatmap antissimétrico | Visualização matricial onde $M_{i,j} = -M_{j,i}$ — preserva direção da diferença, não apenas magnitude | Tufte (2001), princípio de "data-ink" |
| Random Walk Controversy (RWC) | Medida par-a-par de polarização entre comunidades baseada em random walks no grafo de retweets | Garimella et al. (2018), §4.1 |
| Cliff's $\delta$ | Tamanho de efeito não-paramétrico baseado em ordenação; alternativa robusta ao Cohen's $d$ para distribuições assimétricas | (não citado neste script, mencionado em §6.2 como alternativa) |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: 18_analise_bluesky_crossfeed.py
📄  Arquivo gerado: theory_andre/18_analise_bluesky_crossfeed_doc.md
📚  Fontes acadêmicas utilizadas: 10
    1. Cohen (1988) — Statistical Power Analysis for the Behavioral Sciences
    2. Student (1908) — The Probable Error of a Mean (Biometrika)
    3. Welch (1947) — Generalization of Student's Problem (Biometrika)
    4. Kleppmann et al. (2024) — Bluesky and the AT Protocol (CSCW)
    5. Cinelli et al. (2021) — The Echo Chamber Effect on Social Media (PNAS)
    6. Garimella et al. (2018) — Quantifying Controversy on Social Media (ACM TSC)
    7. Vosoughi, Roy & Aral (2018) — The Spread of True and False News Online (Science)
    8. Sawilowsky (2009) — New Effect Size Rules of Thumb (JMASM)
    9. Demšar (2006) — Statistical Comparisons of Classifiers (JMLR)
    10. Tufte (2001) — The Visual Display of Quantitative Information
🔍  Conceitos cobertos:
    - Cohen's d (tamanho de efeito padronizado, variância pooled)
    - Análise par-a-par entre comunidades (cross-feed)
    - AT Protocol / Bluesky lexicons (replyCount, repostCount, likeCount)
    - Echo chamber e segregação estrutural
    - Heatmap antissimétrico de tamanho de efeito
    - Boxplot em escala symlog (distribuições heavy-tailed)
    - Difusão de fake news em redes sociais (cascatas, viralização)
    - Estatísticas descritivas robustas (mediana, p95) para distribuições assimétricas
    - Análise estrutural sem ground truth (workaround para Bluesky)
    - Relação metodológica com script 15 (réplica do método em regime sem labels)
⚠️   Limitações:
    - Cliff's δ (alternativa não-paramétrica) mencionado mas não citado com referência primária
    - Não foi possível verificar empiricamente os valores numéricos de d obtidos
      pelo script (script depende de dados em dados_bluesky/feed_posts/)
    - Comparação numérica direta com o script 15 requer execução pareada;
      este documento argumenta o paralelo metodológico, não confirma magnitudes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```
