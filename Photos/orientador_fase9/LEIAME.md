# Figuras pro orientador — Fase 9 (análise estratificada)

Ordem dos arquivos = ordem sugerida pra apresentar. Vai do "wow" pro detalhado.

## 0. `00_o_que_e_heavy_tail.png` — Defesa metodológica (mostrar se ele perguntar)

Justifica por que usei **percentil empírico** (p5/p95) em vez de z-score
pra definir os outliers da figura 03. 4 painéis (num_nodes, grau_root,
branching_avg, e grau_root em log-log) com Gaussiana ajustada sobreposta.

A Gaussiana erra em todos os casos: extende-se a valores negativos sem
sentido físico, subestima a cauda direita. O p95 real é sempre maior que
μ+1.645σ (o p95 esperado se fosse Gaussiana). O painel inferior-direito
em log-log mostra a linha quase reta característica de lei de potência.

**Quando mostrar:** se o orientador perguntar "por que percentil e não
σ?". Não é figura principal — é defesa metodológica.

Caveat honesto: o heavy-tail no UPFD não é dramático (max ≈ 4-7× a
mediana, não 1000× como em distribuições power-law puras tipo Bluesky).
Provavelmente UPFD aplica cap interno de tamanho. Mas a metodologia
de percentil continua correta — é invariante à forma da distribuição.

## 1. `01_gap_subsets_BOOTSTRAP_CI.png` — A figura central

Mostra a diferença F1(SAGE) − F1(RF) em subsets de profundidade crescente, com
**IC 95% bootstrap** (2000 reamostragens). Barras verdes = IC inferior > 0
(SAGE significativamente melhor). O gap em `depth≥5` é **3.5× o agregado**.

Argumento: a GNN não está só capturando "número de nós" — está aprendendo
propagação multi-hop. Em estrelas rasas (depth=2, 70% dos grafos), o ganho
é marginal; em cascatas reais, vale a pena.

## 2. `02_f1_por_depth.png` — Mesmo achado em granularidade fina

F1 do RF e SAGE por valor exato de depth. Padrão monotônico crescente
até depth=7 (depth=8,9 têm N pequeno e não entram).

## 3. `03_outliers_acuracia.png` — Inversão distributiva confirmada

Acurácia em 6 subgrupos extremos. Os dois críticos:
- `fake_contido` (N=31): SAGE acerta **0 de 31** — inversão total
- `real_viral` (N=60): SAGE acerta 26 de 60 — abaixo de chance

Antes essa "inversão" era só descrita narrativamente na §4.6 do TCC
(F1=0.08 na discordância). Agora tem N e acurácia direta.

## 4. `04_outliers_composicao.png` — Heatmap do mesmo achado

Linhas: % fake verdadeira / % predito fake pelo RF / % predito fake pelo SAGE.
Quando as 3 linhas divergem em uma coluna, o modelo está mal-calibrado
naquele subgrupo. Visual do que a figura 03 mostra como números.

## 5. `05_lowess_acerto.png` — LOWESS: onde o modelo degrada

P(acerto) suavizada (rolling-mean) em função de num_nodes, depth, branching.
SAGE em azul, RF em laranja. Cruzamento das curvas = onde cada modelo ganha.

## 6. `06_f1_por_tercil.png` — Versão discreta da figura 05

F1 por tercil estrutural. Em num_nodes: SAGE sobe monotonicamente
(0.65 → 0.73 → 0.77); RF desce (0.62 → 0.68 → 0.58). Em depth, o tercil
intermediário fica vazio porque a variável é discreta com 70% no mesmo valor.

## 7. `07_distribuicao_lognormal.png` — Defesa metodológica

Histograma das 3 métricas com ajuste lognormal. KS rejeita o fit em todas
(p < 10⁻¹¹). Significa: cauda **mais agressiva que lognormal**. Justifica
metodologicamente o uso de percentil empírico (não z-score) nos outliers.

## 8. `08_distribuicao_profundidade.png` — Premissa do TCC contradita

Contagem por valor de depth. 70% dos grafos UPFD-GossipCop têm depth=2,
mas **30% têm depth ≥ 3, com cauda até depth=9**. Contradiz a §6.3.a do TCC
que afirma "FNN com cascatas estrela" como limitação universal — a crítica
só vale pro FNN construído por nós (script 00), não pro UPFD oficial.

---

## Bônus: `cascatas_interativas_html/`

3 cascatas reais com depth≥8 do UPFD-GossipCop. Abrir no navegador —
permite hover, zoom, arrastar nós. Boas pra defesa oral; não entram no PDF.

| Arquivo | depth | num_nodes | true | RF | SAGE |
|---|---|---|---|---|---|
| `amostra_deep_1_idx3725_depth9.html` | 9 | 164 | REAL | REAL ✓ | REAL ✓ |
| `amostra_deep_2_idx1132_depth8.html` | 8 | 171 | FAKE | REAL ✗ | REAL ✗ |
| `amostra_deep_3_idx3270_depth8.html` | 8 | 160 | FAKE | REAL ✗ | FAKE ✓ |

A terceira é interessante: é exatamente um caso onde o SAGE ganha do RF —
cascata profunda, fake, RF erra, SAGE acerta.
