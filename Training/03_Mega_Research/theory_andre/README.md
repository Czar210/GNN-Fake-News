# theory_andre — Documentação Acadêmica do Mega Research

Esta pasta contém a documentação técnica acadêmica de cada um dos 28 scripts da
pipeline `Training/03_Mega_Research/`, seguindo o protocolo do
[CLAUDE.md](CLAUDE.md) definido pelo orientador André.

## Estrutura

- **CLAUDE.md** — protocolo de análise (lido pelos agentes Claude para gerar os docs)
- **NN_<nome>_doc.md** — um documento por script, em ordem numérica
- **ANALISE_COMPARATIVA_FINAL.md** — síntese consolidada respondendo à questão central do TCC
- **README.md** — este arquivo

## Origem dos documentos

| Faixa | Autor | Observação |
|---|---|---|
| 00–10 | Claude do André (sessões prévias) | Originais em `C:\Users\cesar\Downloads\theory\theory\`, copiados aqui em 2026-04-25 |
| 11–27 | Claude do César (esta sessão, 2026-04-25) | Geração em paralelo via subagentes, seguindo o protocolo do CLAUDE.md |
| ANALISE_COMPARATIVA_FINAL | Claude do César (síntese) | Baseado nos achados de todos os scripts e nas conversas prévias sobre vulnerabilidade topológica |

## Caveat importante sobre referências (docs 11–27)

Os subagentes que produziram os docs 11–27 **não tiveram acesso ao WebSearch/WebFetch
nesta sessão** (bloqueio de ambiente). As referências Tier-1 (papers, DOIs, IDs arXiv,
seções, equações) foram produzidas a partir do conhecimento prévio dos modelos.

**O que isso significa:**

- ✅ Os papers citados existem e são canônicos (Kipf & Welling 2017, Devlin 2019,
  Veličković 2018, Ying 2019, Dou 2021 etc.) — IDs arXiv e DOIs estão corretos para
  os papers principais.
- ⚠️ Localizações específicas ("Seção 3.2, Equação 5") podem ter desvios menores
  em relação aos papers reais — recomenda-se **conferir manualmente** as citações
  antes de submeter o TCC à banca.
- ⚠️ Surveys mais recentes (2023-2024) podem ter sido citados aproximadamente —
  conferir DOI antes de citar.

Os docs 00–10 (André) foram produzidos em sessões com acesso a WebSearch e tendem
a ter referências mais precisas.

## Ordem de leitura recomendada

Para entender o argumento do TCC, ler nesta ordem:

1. **CLAUDE.md** — entender o protocolo
2. **00, 01, 02** — como os dados foram construídos
3. **10_baseline_textual** — teto textual (referência)
4. **13_benchmark_upfd_oficial** — números canônicos publicáveis
5. **14_topologia_sem_texto** — **NÚCLEO DO TCC**, F1 sem texto
6. **15_analise_estrutural** — atalho está nos dados (Cohen's d)
7. **16_gnn_explainer_upfd** — modelo de fato olha estrutura
8. **08_inferencia_cruzada** — falha de generalização
9. **ANALISE_COMPARATIVA_FINAL** — síntese geral

Os demais scripts complementam ou sustentam o argumento.

## Cobertura por fase do TCC

| Fase | Scripts | Status |
|---|---|---|
| 0 — Construção de dados | 00, 01, 02 | ✅ docs 00-02 |
| 1 — Treino arquiteturas (Bluesky) | 03, 04, 05, 06 | ✅ docs 03-06 |
| 2 — Benchmark UPFD | 07, 13 | ✅ docs 07, 13 |
| 3 — Generalização | 08 | ✅ doc 08 |
| 4 — Baselines + confounds | 09, 10, 11, 12 | ✅ docs 09-12 |
| 5 — Vulnerabilidade topológica | **14**, 15, 16 | ✅ docs 14-16 |
| 6 — Modelos finais e Bluesky | 17, 18, 19, 22, 23 | ✅ docs 17-19, 22, 23 |
| 7 — Comparações e RQs | 20, 21, 26 | ✅ docs 20, 21, 26 |
| 8 — Consolidação | 24, 25, 27 | ✅ docs 24, 25, 27 |
| Síntese | — | ✅ ANALISE_COMPARATIVA_FINAL.md |

## Pontos de divergência entre docs e conversas

- O nosso script `00_construir_grafos_fakenewsnet.py` constrói **estrela plana**.
  O UPFD oficial (Dou et al. 2021) é **cascata em árvore**, hidratada via API do
  Twitter. O `00` existe deliberadamente para criar a versão "topologia pobre"
  que isola o efeito da cascata.
- Bluesky **não tem ground truth** de fake/real — análises (scripts 18, 19, 22,
  23, 26) são qualitativas ou baseadas em concordância entre modelos.
- O número exato de F1 nos exemplos da ANALISE_COMPARATIVA_FINAL é referencial
  (baseado nos resultados reportados no TCC) — verificar com `Execution/results/`
  para os números mais recentes.
