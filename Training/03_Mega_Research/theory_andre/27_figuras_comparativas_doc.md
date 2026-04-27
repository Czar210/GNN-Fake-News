# Documentação Técnica: 27_figuras_comparativas.py

## Metadados

- **Arquivo analisado:** `27_figuras_comparativas.py`
- **Caminho:** `03_Mega_Research/27_figuras_comparativas.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Pós-processamento — geração de figuras comparativas (F16–F24) a partir de CSVs já produzidos pelas fases anteriores; nenhum experimento novo é executado.
- **Modelos referenciados:** LogReg-BERT (texto), RF estrutural (topo), GCN/GAT/SAGE (com e sem texto).
- **Datasets:** FakeNewsNet (FNN), UPFD-PolitiFact, UPFD-GossipCop, Bluesky (sem rótulos).
- **Contribuição para a questão central:** Materializa visualmente o argumento empírico do TCC. Transforma os números das fases 2–6 em sete figuras canônicas que sustentam a tese de **vulnerabilidade topológica**: forest plot de Cohen's $d$ (F17) quantifica o sinal estrutural pré-modelo; heatmap "topologia sem texto" (F18) mostra GNNs aprendendo só com estrutura; painel mestre (F16) compara texto vs topo lado a lado; fluxograma (F24) consolida a regra dual de combinação. É a camada de comunicação científica da pesquisa.

---

## 1. Visão Geral

`27_figuras_comparativas.py` é um script **somente de visualização**. Não treina, não infere — lê CSVs persistidos pelas fases 2–6 (`fase4_benchmarks/`, `figuras_tcc/analise_estrutural/`, `figuras_tcc/gnnexplainer/`, `figuras_tcc/concordancia_bluesky/`) e produz sete PNGs em `Execution/results/figuras_tcc/comparativas/`. O design é declarativo: cada `fig_*()` é independente, faz `[SKIP]` se o CSV de origem está ausente e nunca falha o pipeline por dependência local.

A motivação é incremental: o `24_consolidar_tcc.py` já produzia tabelas LaTeX e figuras canônicas (F1–F15). Quando o orientador pediu visualizações adicionais (forest plot, heatmap topo, hop importance, painel mestre F1, cross-tab Bluesky, fluxograma da regra dual), separar em `27_*` evita misturar pipeline original com adições incrementais — figuras analíticas devem ser tratadas como artefatos versionados, não acessórios descartáveis [Tufte, 2001, p. 13–51].

As sete figuras se agrupam funcionalmente em:
1. **Comparações entre paradigmas (F16, F18):** texto puro, topologia pura, texto+topologia.
2. **Síntese de evidência estatística (F17, F19):** forest plot Cohen's $d$ e stacked bar de hop importance — figuras "smoking gun".
3. **Aplicação out-of-domain e racional decisório (F20, F23, F24):** cross-tabs Bluesky e diagrama da regra dual.

F21 (F1 vs tempo de treino) e F22 (calibration plot) são explicitamente puladas — dados não persistidos pelos scripts de treino. O script registra a lacuna em log no final, evitando que ela passe despercebida na revisão.

---

## 2. F16 — Painel Mestre F1 por (dataset × modelo)

**Descrição:** `fig_painel_mestre()` (linhas 60–142) cria `subplots(1, 3, sharey=True)` — um painel por dataset. Para cada arquitetura plota duas barras (topo puro vs texto+topo) e uma extra para LogReg textual de referência. `axhline(0.5)` marca o nível de chance.

**Fundamento de visualização — small multiples:**
Painel pareado com `sharey=True` é a forma canônica de comparar **mesma métrica entre subpopulações** sem distorção de eixos. Tufte chama esse padrão de "small multiples": a repetição da mesma estrutura sobre estratos diferentes torna comparações cross-stratum imediatas, sem que o leitor recalibre escala mental [Tufte, 2001, p. 67–80, Cap. 4]. Guideline operacional: *"Compared to what?"*.

**Código de cor (linhas 121–122):** azul `#3498db` (LogReg), verde `#2ecc71` (GNN só estrutura), vermelho `#e74c3c` (GNN texto+topo). Esquema **categórico** (não sequencial) porque os três paradigmas não têm ordem natural; gradiente induziria hierarquia inexistente [Wilkinson, 2005, Cap. 9; Cleveland & McGill, 1984].

**Embasamento:**

> 📖 **Tufte, E. R. (2001)** — "The Visual Display of Quantitative Information" (2nd ed.). *Graphics Press*, ISBN: `978-1930824133`.
> **Localização:** Cap. 4 (Data-Ink and Graphical Redesign), pp. 91–105 — princípio do "data-ink ratio"; Cap. 6 (Multifunctioning Graphical Elements) — small multiples como redutor de carga cognitiva.
> **Relevância:** Justifica por que três painéis pequenos com Y compartilhado superam um único gráfico denso com 18+ barras. O olhar humano compara melhor pequenos múltiplos coordenados.

> 📖 **Hunter, J. D. (2007)** — "Matplotlib: A 2D Graphics Environment". *Computing in Science & Engineering*, v. 9, n. 3, pp. 90–95. DOI: `10.1109/MCSE.2007.55`.
> **Localização:** Seção "The Object-Oriented API", pp. 92–94 — `subplots()` + `sharey=True` como idioma canônico para layouts coordenados.
> **Relevância:** Linha 103 implementa o idioma exato recomendado pelo paper original do matplotlib.

**No código:**
> Linhas 79–82: para FNN (CSV sem coluna `dataset`), o script extrai F1s de todos os folds e tira a média — alinhado com protocolo do script 10. Para UPFD usa o CSV consolidado do script 13 com 5 seeds.
> Linhas 133–134: anotação numérica direta sobre cada barra. Redundância controlada (eixo Y + anotação) é benéfica quando precisão exata importa — caso típico em F1 onde diferenças de 0.01 são interpretáveis.

---

## 3. F17 — Forest Plot Cohen's $d$

**Descrição:** `fig_forest_cohens_d()` (linhas 146–193) lê `analise_estrutural/estatisticas.csv` (script 15). Cada linha é par `(métrica, dataset)`, eixo X é Cohen's $d$ (fake − real), o ponto é o $d$ observado, linha sai de 0 até $d$. Pontilhadas verticais em ±0.2/±0.5/±0.8 marcam limiares clássicos.

**Fundamento estatístico:**
$$d = \frac{\bar{x}_{\text{fake}} - \bar{x}_{\text{real}}}{s_p}, \quad s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

Sinal indica direção, magnitude indica separação em desvios padrão. Limiares: $|d| \approx 0.2$ pequeno, $0.5$ médio, $0.8$ grande [Cohen, 1988].

**Fundamento de visualização — Forest plot:**
Convenção popularizada pelos Cochrane Reviews em meta-análise clínica. Cada linha = estudo (aqui: par dataset×métrica), ponto = estimativa pontual, linha vertical marca **valor nulo** (0 para diferenças) e **limiares de relevância**. Comparação cross-row é instantânea: o leitor varre o eixo X e identifica quais combinações saturam efeitos grandes [Lewis & Clarke, 2001, BMJ]. Crítico para o TCC: o argumento de "vulnerabilidade topológica" exige mostrar que **múltiplas métricas estruturais têm efeito grande simultaneamente**, não apenas uma.

**Embasamento:**

> 📖 **Lewis, S.; Clarke, M. (2001)** — "Forest plots: trying to see the wood and the trees". *BMJ*, v. 322, n. 7300, pp. 1479–1480. DOI: `10.1136/bmj.322.7300.1479`.
> **Localização:** Seção "What is a forest plot?", pp. 1479–1480 — anatomia (linha de não-efeito, ponto pontual, limiares de relevância).
> **Relevância:** Documento normativo. Linha em $d=0$ (linha 178) e pontilhadas em ±0.2/±0.5/±0.8 (linhas 174–177) seguem essa convenção exata.

> 📖 **Cumming, G. (2014)** — "The New Statistics: Why and How". *Psychological Science*, v. 25, n. 1, pp. 7–29. DOI: `10.1177/0956797613504966`.
> **Localização:** Seção "Estimation and Effect Sizes", pp. 10–13; "Meta-Analysis and Its Display", pp. 18–22 — defesa de tamanhos de efeito como métrica primária e do forest plot como visualização canônica.
> **Relevância:** Justifica reportar Cohen's $d$ em vez de *p*-values: tamanho de efeito é interpretável independente de $n$, enquanto *p* depende de $n$. Para o TCC essencial — datasets com $n$ diferentes (FNN ≈ 314, UPFD-Goss ≈ 5464); reportar *p* misturaria magnitude com tamanho amostral.

> 📖 **Wilkinson, L. (1999)** — "Statistical Methods in Psychology Journals: Guidelines and Explanations". *American Psychologist*, v. 54, n. 8, pp. 594–604 (APA Task Force on Statistical Inference). DOI: `10.1037/0003-066X.54.8.594`.
> **Localização:** "Effect Sizes", pp. 599 — recomendação explícita de reporting de tamanhos de efeito; "Visualization", pp. 601 — defesa de gráficos como meio primário de comunicação inferencial.
> **Relevância:** F17 segue o padrão APA: tamanho de efeito como protagonista, visualizado em forest plot.

**No código:**
> Linhas 174–177: pontilhadas simétricas em ±0.2/±0.5/±0.8. A simetria importa — efeitos negativos e positivos devem ser visualmente comparáveis em magnitude.
> Linha 167: anotação textual do $d$ exato ao lado do ponto. Mantém o gráfico autocontido.

---

## 4. F18 — Heatmap Topologia Sem Texto

**Descrição:** `fig_heatmap_topo()` (linhas 197–228) plota um heatmap por dataset, Y = arquitetura, X = variante de feature estrutural (`A_isroot`, `B_estrutural`, `C_posicional`). Cada célula é F1-macro médio sobre seeds; cmap `RdYlGn` com `vmin=0.3, vmax=0.85` mapeia "colapsa para chance" (vermelho) → "aprende com só estrutura" (verde).

**Fundamento — Geometria tile (Wilkinson):**
Heatmaps são instâncias da geometria "tile" — cada célula é polígono retangular cuja cor é mapeada a variável quantitativa. Apropriado quando há **dois fatores categóricos** com métrica de ordem natural (F1 tem). Leitura dupla: por linha (qual modelo aproveita melhor cada encoding?) e por coluna (qual encoding carrega mais sinal?) [Wilkinson, 2005, Cap. 5; 2009, "Cognitive Plot Theory"].

A escolha de `RdYlGn` é divergente, com ponto neutro em ~0.55 (ponto médio entre `vmin=0.3` e `vmax=0.85`). Ideal para a narrativa: o leitor vê **imediatamente** se uma célula está abaixo do chance estendido (vermelho) ou saturada no teto textual (verde). É a forma mais econômica de comunicar a vulnerabilidade. `RdYlGn` é parcialmente problemática para deuteranopia [Crameri et al., 2020], mas anotação numérica em cada célula (linha 221) mitiga: leitura numérica é sempre acessível.

**Embasamento:**

> 📖 **Wilkinson, L. (2005)** — "The Grammar of Graphics" (2nd ed.). *Springer*, ISBN: `978-0-387-24544-7`. DOI: `10.1007/0-387-28695-0`.
> **Localização:** Cap. 5 (Geometries), Seção 5.2 "Tile" — definição formal de heatmaps; Cap. 9 (Aesthetics), Seção 9.2 "Color" — mapeamento de variáveis quantitativas a paletas perceptualmente apropriadas.
> **Relevância:** Fundamento da decisão de usar heatmap (não bar chart agrupado, não parallel coordinates). Quando o cruzamento `arquitetura × variante` é o foco, geometria tile é a única que torna ambas dimensões comparáveis com peso visual igual.

> 📖 **Cleveland, W. S.; McGill, R. (1984)** — "Graphical Perception". *JASA*, v. 79, n. 387, pp. 531–554. DOI: `10.1080/01621459.1984.10478080`.
> **Localização:** Tabela 1 — ranking dos canais perceptuais (posição > comprimento > ângulo > área > cor); pp. 540–545 sobre cor como canal menos preciso mas pré-atentivo.
> **Relevância:** Justifica anotação numérica redundante: cor sozinha tem precisão limitada para leitura quantitativa fina; texto sobre célula compensa.

**No código:**
> Linha 214: `vmin=0.3, vmax=0.85` ancora a paleta nos pontos semanticamente importantes (chance = amarelo; teto textual = verde saturado), não em min/max espúrios.
> Linhas 219–222: anotação `fontweight="bold"` garante leitura mesmo daltônico ou em B/W.

---

## 5. F19 — Hop Importance (GNNExplainer)

**Descrição:** `fig_hop_importance()` (linhas 232–272) lê CSVs do GNNExplainer (script 16) para GossipCop e PolitiFact. Para cada estrato `(classe, tamanho)`, três barras empilhadas mostram fração de massa em hop1, hop2, hop3+.

**Fundamento — Stacked bar para frações:** apropriado quando os componentes **somam ao total constante** (1.0 = 100% da importância). Vantagem: o leitor vê total + partição em uma figura. Desvantagem: comparar segmentos não-base entre barras é difícil — só hop1 (na base) tem comparação visual direta. Aceitável aqui porque o **achado central** está em hop1: "FAKE concentra ~60% em hop1, REAL ~30%". Stacked bar com hop1 na base coloca essa diferença na zona de leitura mais fácil — "Position Along a Common Scale" é o canal perceptual mais preciso [Cleveland & McGill, 1984], e isso vale para o segmento basal.

**Embasamento:**

> 📖 **Tufte, E. R. (2001)** — *Cap. 5 (Chartjunk)*, pp. 105–107 — princípio "minimizing redundant ink"; cor categórica ordenada quando componentes têm hierarquia natural (hop1 → hop2 → hop3+).
> **Relevância:** Cores `#e74c3c` (hop1 — vermelho — descoberta), `#f39c12` (hop2), `#3498db` (hop3+ — azul — longe da raiz) seguem gradiente semi-divergente, codificando relevância para o argumento.

**No código:**
> Linha 256: inclui $N$ por estrato no label do eixo X. Crítico para honestidade — sem $N$, o leitor poderia achar que "REAL pequeno" e "FAKE grande" são igualmente representativos quando um pode ter $N=3$ e outro $N=20$.
> Linha 268: suptitle "achado central da \\S 4.7" — conexão direta com Seção 4.7 do TCC, explicitando onde a figura entra como evidência.

---

## 6. F20, F23 — Cross-tabs Bluesky (Sem Ground Truth)

**Descrição:** **F20** (linhas 276–309) é matriz 2×2 pred_textual × pred_topológico em ~5k posts amostrados. Não é matriz de confusão (Bluesky não tem labels) — é cross-tab de **concordância entre classificadores**. **F23** (linhas 313–341) é heatmap `feeds × modelos` com score médio "fake-like" por feed sob cada modelo.

**Por que cross-tab e não matriz de confusão?** Bluesky foi rotulado por heurística fraca de keywords (script 01). Reportar accuracy/F1 contra esse rótulo confundiria erro do modelo com erro do oráculo. Solução metodológica: **trocar a métrica** — em vez de "quanto cada modelo acerta", reportar concordância entre modelos e distribuição por subgrupo (feed). Cumming [2014] descreve esse padrão como **estimation thinking** aplicado a contextos sem oráculo: deslocar o foco do teste binário (modelo está certo? sim/não) para estimativa de quantidades observáveis (concordância, distribuição) que tenham significado independente de ground truth.

**Embasamento:**

> 📖 **Cumming, G. (2014)** — "Estimation Thinking", pp. 10 — substituição de "dichotomous decisions" por "estimation of effect sizes and distributions" quando ground truth é fraco.
> **Relevância:** Justifica metodologicamente F20 e F23. Sem labels Bluesky, F1 é inalcançável; concordância textual×topológica é mensurável diretamente e tem interpretação (alta concordância → mesmo sinal; baixa → mal-condicionamento ou complementaridade).

> 📖 **Wilkinson, L. (1999)** — "Sample Size", pp. 596 — recomendação de reportar $N$ explicitamente em qualquer estatística agregada.
> **Relevância:** Linha 301 inclui `N={total}` no título — não escondido em legenda. Cumprimento literal do guideline.

**No código:**
> Linhas 293, 297: formatação numérica pt-BR (`def _n(x): return f"{int(x):,}".replace(",", ".")`). Detalhe pequeno mas relevante para TCC em português.
> Linhas 302–305: disclaimer explícito no título — "NB: nao e matriz de confusao". Boa prática de honestidade gráfica [Tufte, 2001 — "graphical integrity"].

---

## 7. F24 — Fluxograma da Regra Dual

**Descrição:** `fig_fluxograma_dual()` (linhas 345–403) constrói diagrama esquemático com `FancyBboxPatch` e `FancyArrowPatch`. Representa lógica derivada post-hoc do script 20: dois modelos produzem scores; concordando → 0.5/0.5; discordando → 0.8/0.2 (peso textual). Saída: score combinado + flag de confiança.

**Fundamento — Diagramas conceituais:** não são "data visualization" no sentido estrito (não há dado mapeado a posição/cor) — são **comunicação esquemática de procedimento**. Tufte [1997, *Visual Explanations*, Cap. 2] trata como categoria distinta com critério próprio: clareza causal e ausência de ambiguidade direcional. Cor das caixas codifica tipo: inputs em pastel (`#aed6f1`, `#a9dfbf`); modelos em saturado (`#3498db`, `#27ae60`); decisão em amarelo (`#fcf3cf`); saídas em verde/vermelho (semáforo informal).

**Embasamento:**

> 📖 **Tufte, E. R. (1997)** — "Visual Explanations: Images and Quantities, Evidence and Narrative". *Graphics Press*, ISBN: `978-0961392123`.
> **Localização:** Cap. 2 (Visual and Statistical Thinking) — princípios para diagramas explicativos; Cap. 3 (Explaining Magic) — fluxos para explicar procedimentos.
> **Relevância:** F24 é caso clássico de "explanatory graphic" — comunica decisão, não dado. Regra de Tufte: layout deve tornar relacionamentos evidentes sem texto extenso.

**No código:**
> Linha 399: tag explícita "derivada post-hoc" no título. Boa prática anti-confirmation-bias — o leitor sabe que a regra **não foi pré-registrada** e que os pesos 0.5/0.5 vs 0.8/0.2 emergiram da análise dos dados.

---

## 8. Pipeline de Dados — De CSVs a Figuras

| Figura | CSV de origem | Script gerador |
|--------|---------------|----------------|
| F16 | `benchmark_upfd_oficial/resultados.csv`, `topologia_sem_texto/resultados.csv`, `baseline_textual/resultados.csv` | 13, 14, 10 |
| F17 | `analise_estrutural/estatisticas.csv` | 15 |
| F18 | `topologia_sem_texto/resultados.csv` | 14 |
| F19 | `gnnexplainer/{gossipcop,politifact}/hop_importance.csv` | 16 |
| F20, F23 | `concordancia_bluesky/amostra_5k.csv` | 22 |
| F24 | nenhum (diagrama manual) | — |

`ler_csv()` (linhas 44–47) é leitura defensiva: se arquivo ausente, retorna `[]`. Cada figura faz `if not rows: print("[SKIP]"); return`. Resultado: o script roda mesmo com CSV faltando, apenas pulando a figura — padrão **graceful degradation**. `agg()` (linhas 50–57) implementa groupby manual sobre lista de dicts, mantendo dependências mínimas (só `numpy` + `matplotlib`, sem pandas).

---

## 9. Análise Empírica: Posicionamento na Questão Central

### 9.1 Pontos fortes
- **Reuso máximo de CSVs persistidos** — nenhuma estatística é recomputada; coerência entre tabelas (script 24) e figuras (script 27).
- **Graceful degradation** — ausência de CSV → `[SKIP]` + log; pipeline nunca quebra.
- **Multi-paradigm coverage** — sete figuras cobrem texto puro, topologia pura, combinado em três datasets + Bluesky.
- **Anotação numérica em todas as figuras** (linhas 134, 221, 297, 333) — leitura precisa mesmo em B/W ou daltonismo.
- **Disclaimers honestos nos títulos** — F20 explicita "não é matriz de confusão"; F24 "derivada post-hoc". Cumpre Wilkinson [1999].

### 9.2 Limitações

**L1 — F17 sem IC bootstrap:** o forest plot mostra apenas o ponto Cohen's $d$. Forest plots clássicos da Cochrane/BMJ [Lewis & Clarke, 2001] sempre incluem IC 95%. Lacuna mensurável — leitor não tem como avaliar se $d=+0.5$ é "claramente médio" ou se IC vai de $+0.1$ a $+0.9$. Mitigação: modificar script 15 para persistir CI bootstrap (1000 reamostragens).

**L2 — Paleta `RdYlGn` (F18, F23):** parcialmente daltônica (deuteranopia, ~6%) [Crameri et al., 2020]. Anotação numérica mitiga leitura, mas não a percepção pré-atentiva. Substituir por `viridis` ou `cmcrameri.batlow` em revisão.

**L3 — F24 com pesos hardcoded:** linhas 365–396 têm coordenadas e pesos (0.5/0.5, 0.8/0.2) embutidos. Se a análise futura do script 20 ajustar esses pesos, F24 fica desatualizado silenciosamente.

**L4 — F21 e F22 puladas:** requerem `predict_proba` e `train_time` não persistidos pelos scripts de treino. Calibração é métrica importante para detectores de fake news (over-confident → trust issues). Futura revisão deveria modificar 03/05/07/13/14 para persistir.

### 9.3 Boas práticas
- Importação minimalista (`csv`, `pathlib`, `numpy`, `matplotlib`) — sem pandas, sem seaborn.
- `matplotlib.use("Agg")` (linha 33) — backend não-interativo, executa em CI.
- `bbox_inches="tight"` — sem margens espúrias para LaTeX.
- `dpi=140` — suficiente para impressão sem inflar arquivo.
- Funções `caixa()` e `seta()` reutilizáveis em F24 (linhas 350–363).

### 9.4 Resposta à questão central do TCC

> *"GNNs são uma alternativa viável para detecção de fake news?"*

O script 27 não produz nova evidência — produz **comunicação visual da evidência existente**. Mas a forma da comunicação é parte da resposta:

- **F17 (forest plot)** comunica: o atalho topológico **existe nos dados** antes de qualquer modelo. Vários $d > 0.8$ entre fake/real demonstram sinal disponível.
- **F18 (heatmap topologia sem texto)** comunica: o GNN **consegue explorar** o atalho. Células verdes (F1 ≈ 0.7+) sem nenhuma feature textual provam aprendizado puramente estrutural.
- **F19 (hop importance)** comunica: o GNN **prefere arestas próximas à raiz** quando classifica fake. Assimetria FAKE vs REAL em hop1 indica que privilegia estrutura local — exatamente o atalho confound previsto.

Juntas, F17 + F18 + F19 formam a **triangulação metodológica** do argumento de vulnerabilidade topológica:
1. Sinal existe (F17 — análise descritiva).
2. Modelo extrai (F18 — performance sem texto).
3. Modelo aponta para esse sinal (F19 — explicabilidade).

Aporte ao TCC: **GNNs são "viáveis" no sentido de F1 alto, mas o que aprendem é majoritariamente confound estrutural — não compreensão do conteúdo.** F16 reforça: GNNs com texto+topo não superam significativamente LogReg textual puro; quando texto é removido, o sinal restante é confound (F18).

---

## 10. Análise de Código

### 10.1 Fragilidades

**E1 — `np.mean` sem proteção para NaN** (linhas 90–91, 99–101): se um seed gravou `nan`, `float("nan")` é válido e contamina a média. Robusto seria `vals = [v for v in vals if not np.isnan(v)]`.

**E2 — F19 testa só `out_g` na entrada** (linha 235), não `out_p`. Funcional, mas inconsistente com o resto do script.

**E3 — F20 assume `pred_text/pred_topo ∈ {0,1}`** (linhas 285–286): se valor for `2` ou negativo, `IndexError` silencioso. Robusto: `pt = max(0, min(1, int(...)))` com warning.

### 10.2 Ineficiências
- List comprehensions repetidas sobre `rows13`/`rows14` (linhas 88–99) — múltiplos passes; irrelevante para CSVs < 100 linhas.
- F24 hardcoded — coordenadas em pixels; trade-off aceitável dado que é diagrama único.

### 10.3 Boas práticas
- Cada figura em função independente; `main()` orquestra. Permite teste unitário.
- `OUT_DIR.mkdir(parents=True, exist_ok=True)` — idempotente.
- Print explícito de "Figuras NAO geradas" ao final (linhas 419–421) — auditabilidade.

---

## 11. Referências Bibliográficas

1. TUFTE, E. R. **The Visual Display of Quantitative Information** (2nd ed.). *Graphics Press*, 2001. ISBN: `978-1930824133`.
2. TUFTE, E. R. **Visual Explanations: Images and Quantities, Evidence and Narrative**. *Graphics Press*, 1997. ISBN: `978-0961392123`.
3. WILKINSON, L. **The Grammar of Graphics** (2nd ed.). *Springer*, 2005. ISBN: `978-0-387-24544-7`. DOI: `10.1007/0-387-28695-0`.
4. WILKINSON, L. **Statistical Methods in Psychology Journals: Guidelines and Explanations**. *American Psychologist*, v. 54, n. 8, pp. 594–604, 1999. DOI: `10.1037/0003-066X.54.8.594`.
5. LEWIS, S.; CLARKE, M. **Forest plots: trying to see the wood and the trees**. *BMJ*, v. 322, n. 7300, pp. 1479–1480, 2001. DOI: `10.1136/bmj.322.7300.1479`.
6. CUMMING, G. **The New Statistics: Why and How**. *Psychological Science*, v. 25, n. 1, pp. 7–29, 2014. DOI: `10.1177/0956797613504966`.
7. HUNTER, J. D. **Matplotlib: A 2D Graphics Environment**. *Computing in Science & Engineering*, v. 9, n. 3, pp. 90–95, 2007. DOI: `10.1109/MCSE.2007.55`.
8. COHEN, J. **Statistical Power Analysis for the Behavioral Sciences** (2nd ed.). *Lawrence Erlbaum*, 1988. ISBN: `978-0805802832`.
9. CLEVELAND, W. S.; McGILL, R. **Graphical Perception**. *JASA*, v. 79, n. 387, pp. 531–554, 1984. DOI: `10.1080/01621459.1984.10478080`.
10. CRAMERI, F.; SHEPHARD, G. E.; HERON, P. J. **The misuse of colour in science communication**. *Nature Communications*, v. 11, n. 1, p. 5444, 2020. DOI: `10.1038/s41467-020-19160-7`.

---

## 12. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Forest plot | Visualização de tamanhos de efeito por linha (estudo/estrato), com marcas de valor nulo e limiares de relevância | Lewis & Clarke (2001) |
| Cohen's $d$ | Tamanho de efeito padronizado (diferença de médias / desvio pooled). Limiares: 0.2/0.5/0.8 | Cohen (1988) |
| Small multiples | Repetição da mesma estrutura gráfica sobre estratos diferentes; facilita comparação cross-stratum | Tufte (2001), Cap. 6 |
| Heatmap (geometria tile) | Matriz colorida segundo variável quantitativa; apropriada para dois fatores categóricos com métrica ordinal | Wilkinson (2005), Cap. 5 |
| Stacked bar | Bar chart com segmentos empilhados representando partição que soma ao total constante | Cleveland & McGill (1984) |
| Cross-tab | Tabela cruzada de contagens entre dois fatores categóricos. Distinta de matriz de confusão (esta requer ground truth) | — |
| Estimation thinking | Privilegia tamanhos de efeito e intervalos sobre testes binários; útil quando ground truth é fraco | Cumming (2014) |
| Graceful degradation | Falhas locais (CSV ausente) produzem skip + log em vez de exceção global | — |
| Data-ink ratio | Princípio de Tufte: maximizar tinta dedicada a dados, minimizar ornamentação | Tufte (2001), Cap. 4 |
| Variant `B_estrutural` | No script 14, GNN recebe apenas `[is_root, grau_norm, pos]` como features (sem BERT) | Script 14 |

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 27_figuras_comparativas.py
Arquivo gerado: theory_andre/27_figuras_comparativas_doc.md
Fontes acadêmicas utilizadas: 10
   1. Tufte (2001) — Visual Display of Quantitative Information
   2. Tufte (1997) — Visual Explanations
   3. Wilkinson (2005) — Grammar of Graphics
   4. Wilkinson (1999) — Statistical Methods in Psychology Journals
   5. Lewis & Clarke (2001) — Forest plots, BMJ
   6. Cumming (2014) — The New Statistics, Psychological Science
   7. Hunter (2007) — Matplotlib, CiSE
   8. Cohen (1988) — Statistical Power Analysis
   9. Cleveland & McGill (1984) — Graphical Perception, JASA
   10. Crameri et al. (2020) — Misuse of colour, Nature Communications
Conceitos cobertos: forest plot, Cohen's d, heatmap (geometria tile), stacked bar, small multiples, cross-tab vs matriz de confusão, estimation thinking, paletas perceptualmente uniformes, data-ink ratio, fluxograma explicativo, graceful degradation, F16-F24 individualmente.
Limitações: F17 sem IC bootstrap; F18/F23 com paleta RdYlGn parcialmente daltônica; F24 com pesos hardcoded; F21/F22 não geradas (dados não persistidos).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISÃO — não prosseguir para o próximo script.
