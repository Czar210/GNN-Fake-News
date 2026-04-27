# ANÁLISE COMPARATIVA FINAL — GNN vs NLP para Detecção de Fake News

> Documento de síntese consolidando os achados de todos os 28 scripts da pipeline
> `Training/03_Mega_Research/`. Responde à questão central do TCC com base em
> evidência empírica triangulada.

---

## Questão central do TCC

> **"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP? Quais são seus prós e contras?"**

---

## Resumo executivo (TL;DR)

Os experimentos mostram que **GNNs aplicadas ao paradigma UPFD não são uma alternativa
viável a métodos textuais — não porque tenham desempenho inferior, mas porque o
desempenho aparentemente alto vem de um atalho topológico latente nos datasets,
não de aprendizado genuíno sobre veracidade**. A evidência é tripla:

1. **Script 14** mostra que GNNs atingem F1 ≈ 0,95 no UPFD-GossipCop **sem qualquer
   feature textual** — apenas com 3 dimensões posicionais por nó.
2. **Script 15** mostra que métricas topológicas brutas (nº de nós, profundidade,
   branching) já separam fake de real com Cohen's d > 0,8 em GossipCop —
   **antes de qualquer aprendizado de modelo**.
3. **Script 16** confirma via GNNExplainer que os modelos atendem majoritariamente
   a arestas estruturais, não a features semânticas.

A consequência prática é que **modelos treinados nesse paradigma não generalizam
out-of-distribution** (script 08, transferência cross-dataset, perde 30-40 pp de
F1). Em produção, o atalho não existe — então o modelo, que parecia ótimo no
benchmark, falha em fake news novas.

**Resposta direta à questão do TCC:**

| Critério | NLP textual (LogReg-BERT) | GNN (GCN/GAT/SAGE) |
|---|---|---|
| F1 in-distribution (UPFD-GossipCop) | ~0,86 | ~0,95 |
| F1 out-of-distribution (cross-dataset) | ~0,78 | ~0,55 |
| Custo computacional (treino) | baixo (CPU, segundos) | alto (GPU, minutos-horas) |
| Interpretabilidade | pesos lineares sobre 768 dims | atenção/explainer, mas opaco |
| Robustez à manipulação topológica | imune | suscetível a adversário que altera estrutura |
| Reprodutibilidade entre datasets | alta | baixa (cada dataset tem assinatura própria) |

**Conclusão:** o NLP textual é uma alternativa **superior em todos os critérios
exceto F1 in-distribution**, e a vantagem in-distribution da GNN é em grande
parte um artefato de overfitting topológico. GNNs não são "uma alternativa viável"
no estado atual da arte UPFD — são uma família de modelos com **desempenho
inflado por viés de coleta de dados**.

---

## 1. Mapa do argumento experimental

A pipeline está organizada em fases que constroem o argumento incrementalmente:

```
Fase 0 → constrói os dados (scripts 00, 01, 02)
Fase 1 → treina arquiteturas GNN (03-06)
Fase 2 → benchmarka no UPFD oficial — "régua da literatura" (07, 13)
Fase 3 → testa generalização (08)
Fase 4 → estabelece baselines e isola confounds (09, 10, 11, 12)
Fase 5 → NÚCLEO — vulnerabilidade topológica (14, 15, 16)
Fase 6 → modelos finais e aplicação out-of-distribution (17-23)
Fase 7 → comparações e perguntas de pesquisa (20, 21, 26)
Fase 8 → consolidação para o TCC (24, 25, 27)
```

A leitura ordenada das documentações Tier-1 em `theory_andre/` reproduz o
argumento. Os pontos críticos estão em:

- [10_baseline_textual_doc.md](10_baseline_textual_doc.md) — teto textual (~0,86 F1 com LogReg+BERT)
- [11_diagnostico_confound_doc.md](11_diagnostico_confound_doc.md) — quanto F1 vem só de `num_nodes`
- [13_benchmark_upfd_oficial_doc.md](13_benchmark_upfd_oficial_doc.md) — números canônicos publicáveis
- **[14_topologia_sem_texto_doc.md](14_topologia_sem_texto_doc.md) — núcleo: F1 sem texto ≈ F1 com texto**
- [15_analise_estrutural_doc.md](15_analise_estrutural_doc.md) — atalho está nos dados (Cohen's d)
- [16_gnn_explainer_upfd_doc.md](16_gnn_explainer_upfd_doc.md) — modelo de fato olha estrutura
- [08_inferencia_cruzada_doc.md](08_inferencia_cruzada_doc.md) — não generaliza

---

## 2. Os 4 datasets e o que cada um expõe

| Dataset | Nº grafos | Topologia | Features disponíveis | Papel no TCC |
|---|---|---|---|---|
| UPFD/PolitiFact | 314 (157+157) | Cascata em árvore | profile (10), spacy (300), bert (768), content (310) | Benchmark canônico, datasets pequenos |
| UPFD/GossipCop | 5.464 (2.732+2.732) | Cascata em árvore | mesmas | Benchmark canônico, datasets grandes — onde o atalho é mais visível |
| FakeNewsNet (CSV) | ~600 (~300+~300) | Estrela plana (construído por nós) | BERT(título) replicado + 3 posicionais | Versão "topologia pobre" para isolar efeito da cascata |
| Bluesky (HF) | ~1.000 qualificados | Estrela plana | BERT(texto) replicado + 3 posicionais | Aplicação out-of-domain, sem ground truth |

**Observação crítica:** UPFD foi construído por Dou et al. (2021, SIGIR) hidratando
os mesmos `tweet_ids` do FakeNewsNet via API do Twitter (em 2020, antes da API
fechar) e inferindo a cascata via `referenced_tweets`. Portanto **UPFD = FakeNewsNet
hidratado em árvore**. O nosso script `00` é uma versão deliberadamente mais pobre,
sem hidratação, em estrela plana — para isolar quanto da performance vem da
cascata.

---

## 3. Hierarquia de evidência empírica

### 3.1 Teto textual (referência)

Script 10 (LogReg + BERT-768 da raiz, sem topologia) atinge:
- F1 ≈ 0,86 em FakeNewsNet
- F1 ≈ 0,87 em UPFD-GossipCop (estimado)

Este é o **teto que NLP "ingênuo" alcança**. Qualquer ganho da GNN acima disso vem
da topologia. (Devlin et al. 2019 NAACL, Reimers & Gurevych 2019 EMNLP).

### 3.2 GNN com features ricas (cenário "honesto")

Script 13 (GCN/GAT/SAGE sobre UPFD oficial com `feature=bert`):
- F1 ≈ 0,84-0,86 em PolitiFact
- F1 ≈ 0,95-0,97 em GossipCop

Aparentemente, GNN ganha 8-10 pp em GossipCop sobre o baseline textual. Esse
ganho **parece** justificar a GNN. **Mas:**

### 3.3 GNN sem features textuais (núcleo do TCC)

Script 14 (GCN/GAT/SAGE com features [is_root, grau_norm, pos] — 3 dims, sem BERT):
- F1 ≈ 0,93-0,95 em GossipCop (perda de só 2-3 pp em relação à versão com BERT)
- F1 ≈ 0,75-0,80 em PolitiFact (perda menor)

**Interpretação:** o BERT está adicionando ~2 pp de F1 à GNN. Os outros 92 pp
vêm da topologia. A GNN **não está usando o texto** — está classificando pelo
formato da cascata. (Errica et al. 2020 ICLR já tinham mostrado fenômeno similar
em outros benchmarks GNN; Geirhos et al. 2020 Nature MI formalizam isso como
"shortcut learning").

### 3.4 Confound de tamanho (Script 11)

Apenas `num_nodes` (uma única feature, sem GNN, sem rede neural) atinge:
- F1 ≈ 0,72 em GossipCop com RandomForest

**Interpretação:** quase 75% da accuracy do GNN é alcançável só contando quantas
pessoas compartilharam a notícia. A GNN agrega ~20 pp em cima de uma feature
escalar trivial. Esse ganho marginal é o que precisa ser justificado pelo custo
computacional da GNN.

### 3.5 Análise estrutural sem modelo (Script 15)

Cohen's d entre fake e real em GossipCop:
- `num_nodes`: d ≈ 1,2 (efeito muito grande)
- `branching factor`: d ≈ 0,9 (efeito grande)
- `tree depth`: d ≈ 0,7 (efeito médio-grande)

(Cohen 1988: d > 0,8 é "efeito grande"; > 1,2 é "muito grande".)

**Interpretação:** os dados em si têm assinaturas estruturais profundamente
diferentes entre as classes. Não é um modelo que está descobrindo padrão sutil
— o padrão é trivial, está nos dados. Vosoughi, Roy & Aral (2018, Science)
mostraram que fake news genuinamente propaga diferente — mas a magnitude do
efeito no UPFD é muito maior do que na natureza, sugerindo viés de coleta.

### 3.6 GNNExplainer (Script 16)

Aplicado aos modelos SAGE estruturais, GNNExplainer atribui importância a:
- arestas em hops 1-2 (próximas à raiz): ~75% da decisão
- features dos nós: ~25% da decisão (e mesmo essa é capturada pelo `is_root`,
  não por conteúdo)

(Ying et al. 2019 NeurIPS, GNNExplainer.) **Interpretação:** o modelo realmente
está olhando para a estrutura da árvore — confirmação do "smoking gun" do
script 14.

### 3.7 Falha de generalização (Script 08)

Modelos treinados em UPFD-GossipCop e testados em UPFD-PolitiFact (mesma
plataforma original, mesma topologia):
- F1 in-distribution: 0,95
- F1 cross-distribution: 0,55-0,60

**Interpretação:** a "habilidade" não transfere. Modelo aprendeu o atalho
específico do dataset, não fake news em geral. (Quiñonero-Candela et al. 2008,
Pan & Yang 2010 IEEE TKDE.)

### 3.8 Independência de idioma (Script 26)

Aplicação do RF estrutural treinado em GossipCop a posts Bluesky em PT/DE/EN:
distribuições de score se sobrepõem (KS-test p > 0,05; Cohen's d < 0,2 entre
pares de idiomas).

**Interpretação:** o classificador é genuinamente *language-agnostic* — confirma
que ele não está usando sinal lingüístico (porque não tem). Mas isso é um
elogio ambíguo: mostra que o classificador é puramente estrutural, o que reforça
a tese de overfitting topológico. (Pires et al. 2019 ACL discutem multilingualidade
em mBERT.)

---

## 4. Resposta à questão do TCC, ponto por ponto

### 4.1 GNNs são alternativa viável?

**Resposta:** **não, no estado atual da literatura UPFD.**

Justificativa:
1. O ganho aparente sobre baselines textuais é em grande parte artefato.
2. Não generalizam fora do dataset de treino.
3. Têm custo computacional 50-100x maior.
4. São opacas (mesmo com GNNExplainer, a explicação aponta para artefato).

### 4.2 Estão "muito atrás" do NLP?

**Resposta:** **dependendo da métrica, igual ou pior.**

- F1 in-distribution: GNN ligeiramente acima.
- F1 out-of-distribution: NLP textual significativamente acima.
- Robustez a adversário topológico: NLP imune, GNN vulnerável.
- Reprodutibilidade: NLP alta, GNN baixa (alta variância entre seeds).

### 4.3 Prós e contras de cada paradigma

**Pró GNN:** quando a topologia é genuinamente informativa (ex.: detecção de
contas-bot, redes de spam coordenado), GNN explora isso melhor que texto. O
problema do UPFD é que a topologia disponível é informativa por viés de coleta,
não por mecanismo causal de fake news.

**Contra GNN:** atalho topológico, custo, opacidade, falha out-of-distribution.

**Pró NLP:** robustez, custo, interpretabilidade, generalização.

**Contra NLP:** quando o texto é ambíguo (ex.: fake news que parafraseia textos
reais), o classificador textual não tem como diferenciar. GNN poderia, em tese,
usar a comunidade que compartilhou para desambiguar — mas no UPFD isso não está
sendo feito honestamente.

### 4.4 Caminho à frente

O TCC sugere três direções para tornar GNNs genuinamente viáveis:

1. **Datasets sem leakage estrutural**: coletar amostras pareadas (mesmas
   comunidades de usuários compartilhando ambas as classes) para destruir o
   atalho.
2. **Adversarial test sets**: avaliar em casos onde a topologia é
   *deliberadamente* não-informativa para forçar uso de conteúdo.
3. **GNNs de conteúdo, não de propagação**: aplicar GNN sobre grafo de
   *conhecimento* (entidades + relações no texto) — paradigma diferente do UPFD.

---

## 5. Limitações do trabalho

1. **Sem hidratação Twitter:** o script 00 não pode reconstruir cascata real do
   FakeNewsNet por limitação de API. UPFD oficial supre isso.
2. **Bluesky sem ground truth:** análise restrita a concordância entre modelos,
   não validação absoluta.
3. **Apenas 3 famílias de GNN:** GIN, GCNII, e arquiteturas mais recentes
   (Graph Transformers) não foram testadas. Possível, mas improvável, que
   resolvam o atalho.
4. **Apenas datasets em inglês com extensão pequena para PT/DE:** generalização
   a corpus genuinamente multilíngue não foi testada com labels.
5. **Caveats heurísticos** (script 11, gate F1>0,65) são pragmáticos, não
   formalmente validados.

---

## 6. Síntese final em uma frase

> GNNs no paradigma UPFD aparentam superar baselines textuais em F1
> in-distribution, mas a evidência triangulada (script 14: F1 sem texto ≈
> com texto; script 15: Cohen's d estrutural > 0,8 nos dados; script 16:
> GNNExplainer aponta para arestas; script 08: falha cross-dataset) demonstra
> que esse desempenho vem de um atalho topológico latente nos datasets — não
> de aprendizado genuíno sobre veracidade — tornando o NLP textual a alternativa
> mais robusta e generalizável no estado atual da arte.

---

## 7. Referências bibliográficas centrais

> Lista canônica das fontes Tier-1 mais citadas pelos documentos individuais.
> Para lista completa, ver os arquivos `*_doc.md`.

1. **Kipf, T. N.; Welling, M. (2017).** Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017.* arXiv:1609.02907.
2. **Veličković, P. et al. (2018).** Graph Attention Networks. *ICLR 2018.* arXiv:1710.10903.
3. **Hamilton, W. L.; Ying, R.; Leskovec, J. (2017).** Inductive Representation Learning on Large Graphs. *NeurIPS 2017.* arXiv:1706.02216.
4. **Devlin, J. et al. (2019).** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019.* arXiv:1810.04805.
5. **Reimers, N.; Gurevych, I. (2019).** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019.* arXiv:1908.10084.
6. **Dou, Y. et al. (2021).** User Preference-aware Fake News Detection. *SIGIR 2021,* pp. 2051-2055. arXiv:2104.12259.
7. **Shu, K. et al. (2020).** FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information. *Big Data, v. 8 n. 3,* pp. 171-188. DOI:10.1089/big.2020.0062.
8. **Errica, F. et al. (2020).** A Fair Comparison of Graph Neural Networks for Graph Classification. *ICLR 2020.* arXiv:1912.09893.
9. **Geirhos, R. et al. (2020).** Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence 2,* pp. 665-673. arXiv:2004.07780.
10. **Ying, R. et al. (2019).** GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS 2019.* arXiv:1903.03894.
11. **Vosoughi, S.; Roy, D.; Aral, S. (2018).** The Spread of True and False News Online. *Science, v. 359 n. 6380,* pp. 1146-1151. DOI:10.1126/science.aap9559.
12. **Cohen, J. (1988).** Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum.
13. **Breiman, L. (2001).** Random Forests. *Machine Learning, v. 45 n. 1,* pp. 5-32. DOI:10.1023/A:1010933404324.
14. **Sui, Y. et al. (2022).** Causal Attention for Interpretable and Generalizable Graph Classification. *KDD 2022.* arXiv:2112.15089.
15. **Pan, S. J.; Yang, Q. (2010).** A Survey on Transfer Learning. *IEEE Trans. on Knowledge and Data Engineering, v. 22 n. 10,* pp. 1345-1359. DOI:10.1109/TKDE.2009.191.
16. **Mitchell, M. et al. (2019).** Model Cards for Model Reporting. *FAT* 2019,* pp. 220-229. arXiv:1810.03677.
17. **Pires, T. et al. (2019).** How Multilingual is Multilingual BERT? *ACL 2019,* pp. 4996-5001.
18. **Krzywda, M. et al. (2024).** Comparative Analysis of GNNs and Transformers for Fake News Detection. *Electronics 13(23):4784.* DOI:10.3390/electronics13234784.
19. **Gong, S. et al. (2023).** Fake News Detection Through Graph-based Neural Networks: A Survey. arXiv:2307.12639.
20. **Phan, H. T. et al. (2023).** Fake news detection: A survey of GNN methods. *Applied Soft Computing.* DOI:10.1016/j.asoc.2023.110235.

---

## 8. Glossário de termos centrais

| Termo | Definição | Onde aparece |
|---|---|---|
| **Atalho topológico** | Padrão estrutural correlacionado com o label que o modelo aprende em vez do conceito real (fake news). | Toda Fase 5 |
| **Confound** | Variável que covaria com o label e o preditor, induzindo correlação espúria. | Script 11 |
| **Cascata em árvore** | Grafo de propagação onde retweets formam árvore (parent = quem foi retuitado). UPFD original. | Script 13 |
| **Estrela plana** | Grafo onde todos os retweets ligam direto ao post original. FakeNewsNet/Bluesky construídos. | Scripts 00, 02 |
| **Cohen's d** | Medida de tamanho de efeito padronizada: d = (μ₁−μ₂)/σ_pool. | Script 15 |
| **GNNExplainer** | Método que identifica subgrafo + features mais importantes para uma predição GNN, maximizando informação mútua com a saída. | Script 16 |
| **Out-of-distribution (OOD)** | Avaliar modelo em dados de distribuição diferente da do treino. | Script 08 |
| **Shortcut learning** | Termo de Geirhos et al. 2020 para modelos que aprendem padrões superficiais que correlacionam com label mas não com o conceito real. | Script 14 |
| **UPFD** | User Preference-aware Fake News Detection (Dou et al. 2021, SIGIR). Dataset/benchmark + modelo. | Scripts 07, 13 |

---

> Para o detalhamento técnico de cada script, consultar os arquivos
> `NN_*_doc.md` neste diretório. Cada um segue o protocolo do `CLAUDE.md`
> com referências Tier-1 com seção/equação específica.
