# Estrutura proposta v2.2 — TCC GNN Fake News

**Status:** outline pós-revisão adversarial rodada 2 (Passo 1 da Fase 5, rev 2). Aprovado pela revisão anterior para Passo 2 com 6 ajustes não-bloqueantes — todos incorporados.
**Mudanças vs v2.1 (rodada 2):** §4.4-bis com lead "consistência ≠ detecção" + 3 hipóteses alternativas (shift domínio, viés amostral, desbalanço) + distinção significância estatística vs prática; §4.6 reescrita de "vota majoritária" → inversão distributiva (real virais + fake contidos); §4.7 N=5 explicitado e "bimodal" → "sem padrão consistente"; +6pp em §4.7 ancorado em metadata.json (RF=0.753 vs SAGE=0.814); §6.3 com limitações (f) snapshot temporal Bluesky e (g) ambiente CPU-only; §2.5 com promessa de derivação matemática completa na prosa.
**Mudanças vs v2 (rodada 1):** RQ3 reformulada; std=0.002 verificado; §4.7 GNNExplainer N=20; §4.8 movido pra Cap 5; "Mudanças críticas" → Apêndice D; refs Newman/Pei adicionadas; §4.3 qualificada; §3.9 regra dual post-hoc; §4.6 parágrafo sobre F1=0.08.
**Versão anterior:** descartada (ensemble de 4 GCNs + narrativa de 0% erro com 50+ interações foram refutadas pelos experimentos).
**Princípio:** cada subseção é ancorada em **um artefato experimental nominal** (script + CSV + tabela LaTeX gerada). Se uma subseção não tem âncora, é candidata a remoção.

---

## Tese central (uma frase, não-negociável)

> Topologia de propagação detecta fake news **quando o domínio apresenta diferença estrutural mensurável entre classes** (Cohen's d ≥ 0.5 em pelo menos uma métrica de cascata); fora dessa condição, modelos topológicos colapsam — independentemente da arquitetura GNN.

Os achados positivos (GossipCop) e negativos (PolitiFact) sustentam essa tese conjuntamente.

---

## Perguntas de pesquisa (RQs)

- **RQ1.** A topologia da propagação carrega sinal discriminante para detecção de fake news, *acima* do que a feature textual da raiz (BERT) já fornece?
- **RQ2.** Se sim, sob quais condições estruturais? (operacionalizado via Cohen's d entre fake/real em métricas de cascata)
- **RQ3.** É viável detectar fake news **apenas** pela topologia (sem texto), em cenários onde o texto não é exploitável (multilíngue sem encoder específico, texto deletado/restrito, ou linguagem informal de redes sociais)? *Operacionalização:* (a) experimento UPFD-GossipCop só com features estruturais; (b) experimento Bluesky filtrado por idioma (PT vs EN) confirmando que o classificador estrutural é independente do idioma do post.

Cada capítulo de Resultados/Aplicação responde diretamente uma ou mais RQs — declarado no início da seção.

---

## Capítulo 1 — Introdução

| § | Seção | Conteúdo | Artefato/Âncora |
|---|-------|----------|-----------------|
| 1.1 | Contextualização | Desinformação como problema epistêmico; limites de NLP isolada (LLMs geram texto gramaticalmente impecável). | Cita **introducao.tex atual §1** (revisar superlativos) |
| 1.2 | Motivação | Topologia como sinal complementar ao texto. *Não* é "substituto", é "complementar e robusto-a-idioma". | — |
| 1.3 | Perguntas de pesquisa | Listar RQ1–RQ3 acima, explicitamente. | — |
| 1.4 | Contribuições | (a) ablation sistemática de encodings posicionais corrigindo Erro 1; (b) demonstração de que topologia funciona em GossipCop e *não* em PolitiFact, com explicação estrutural via Cohen's d; (c) **análise da concordância textual×topológico**, com regra de combinação derivada *post-hoc* a partir do experimento de discordância; (d) pesos persistidos + ferramenta web. | T1–T10, F1–F13, HF Hub |
| 1.5 | Estrutura do trabalho | Mapa dos capítulos. | — |

**Crítica esperada do outro IA:** "RQs muito ambiciosas pra TCC?" Resposta antecipada: cada RQ tem 1-2 experimentos diretos.

---

## Capítulo 2 — Fundamentação Teórica

| § | Seção | Conteúdo | Status |
|---|-------|----------|--------|
| 2.1 | Detecção de fake news: panorama | NLP clássico, contexto social, propagação. | Reescrever |
| 2.2 | Grafos e redes sociais | Definições básicas, propagação como árvore. | Reescrever |
| 2.3 | Embeddings textuais | Sentence-BERT (Reimers & Gurevych 2019), `paraphrase-multilingual-mpnet-base-v2`, multilingualidade. | **Adicionar** |
| 2.4 | Redes Neurais de Grafos | GCN (Kipf & Welling 2017), GAT (Veličković 2018), GraphSAGE (Hamilton 2017). Foco em: regra de propagação, agregação, pooling global. | Expandir SAGE |
| 2.5 | Degeneração GCN com features homogêneas | **Erro 1 explicado matematicamente**: se `x_i = x_j ∀ i,j` (matriz X de rank 1 por coluna), a regra de propagação $\hat{D}^{-1/2}(\hat{A})\hat{D}^{-1/2}XW$ reduz a transformação linear escalonada por grau, incapaz de produzir representações discriminativas entre nós da mesma classe. **Promessa para a prosa final:** derivação completa com matriz de adjacência normalizada e demonstração de que a saída da camada é, em cada nó, um múltiplo escalar da única coluna de X. Justifica o uso de encodings posicionais. | **Adicionar** (Item 5A.4 do plano) |
| 2.6 | Encodings posicionais | `is_root`, `grau_norm`, `pos` — derivação e por quê funcionam quebrando a homogeneidade. | **Adicionar** |
| 2.7 | Explicabilidade em GNNs | GNNExplainer (Ying et al. 2019) — máscara de arestas, interpretação. | **Adicionar** |
| 2.8 | Validação estatística | k-fold estratificado, ttest_rel pareado (por que pareado importa), Cohen's d como effect size. | **Adicionar** |
| 2.9 | Trabalhos relacionados | UPFD (Dou et al. SIGIR 2021), FakeNewsNet (Shu et al. 2018), BiGCN, GCNFN. **Homofilia e assortatividade em redes** (Newman 2003; Pei et al. ICLR 2020 "Geom-GCN") — fundamenta teoricamente *por que* domínios com diferença estrutural por classe favorecem GNNs. Posicionar nosso trabalho: *não somos um método novo; somos uma análise sistemática + diagnóstico de quando o paradigma funciona*. **Limitação assumida:** não rodamos BiGCN/GCNFN diretamente — comparação numérica é via valores publicados (ver §6.3e). | **Adicionar/expandir** |

**Crítica esperada:** "Capítulo de fundamentação enorme demais." Resposta: vc pode mover 2.5–2.6 pra início da Metodologia se for pedido. Mantenho como fundamentação porque é teórico.

---

## Capítulo 3 — Metodologia

| § | Seção | Conteúdo | Âncora |
|---|-------|----------|--------|
| 3.1 | Visão geral do pipeline | Diagrama: dados brutos → grafos → features → modelo → avaliação. | Figura nova (gerar) |
| 3.2 | Datasets | (a) FakeNewsNet PolitiFact, (b) UPFD PolitiFact, (c) UPFD GossipCop, (d) Bluesky 6GB acadêmico. Tabela com #grafos, #classes, fonte, licença. | T0 nova (gerar via script 24) |
| 3.3 | Construção de grafos FakeNewsNet | `00_construir_grafos_fakenewsnet.py` com 3 variantes (`posfull`, `posmin`, `posgrau`). Detalhar features por nó. | Script 00 |
| 3.4 | Folds compartilhados | `gerar_folds.py` — 10 folds estratificados, seed=42, reusados em todos os experimentos pra permitir teste pareado. | `folds_fnn.pt` |
| 3.5 | Arquiteturas GNN | GCN, GAT, SAGE com 3 camadas + global_mean_pool + linear. Hiperparâmetros tabulados. | Tabela hyperparams (gerar) |
| 3.6 | Baselines não-GNN | LogReg-BERT, RF-tabular sobre `[num_nodes, grau_root]`. **Justificativa**: serve de teto inferior + diagnóstico de confound. | Scripts 10, 11 |
| 3.7 | Protocolo de avaliação | F1-macro como métrica primária; k-fold pareado + `scipy.stats.ttest_rel`; effect size Cohen's d para análise estrutural. | Script 09 |
| 3.8 | Análise de explicabilidade | GNNExplainer aplicado em SAGE-GossipCop e SAGE-PolitiFact, parametrização (epochs, edge_size). | Script 16 |
| 3.9 | Classificador dual e regra de combinação **(post-hoc)** | LogReg-BERT (texto) + RF-estrutural (topo). **Importante:** a regra 0.8/0.2 é uma heurística derivada *após* a observação dos resultados de §4.6, não um componente metodológico definido a priori. Apresentada aqui apenas para precedência expositiva; sua justificativa empírica está em §4.6. | Script 20, API |
| 3.10 | Reprodutibilidade | requirements.txt pinado, seed=42, Python 3.12.10 CPU; caveats de bit-exact (CUDA/cuDNN). | README §Reprodutibilidade |

---

## Capítulo 4 — Resultados Experimentais

**Lead da seção:** "Cada subseção responde uma RQ. Resultados negativos são reportados com a mesma extensão dos positivos."

### 4.1 Baselines textuais (RQ1, contexto)
- **Conteúdo:** LogReg + RF sobre `x[0]` (BERT raiz) em FakeNewsNet, k-fold.
- **Achado:** LogReg F1=0.859±0.043 — texto da raiz **já carrega quase todo o sinal** no FNN.
- **Implicação:** GNN só agrega valor se topologia adicionar sinal *acima* desse baseline.
- **Âncora:** T1 (`T1_baseline_textual_fnn.tex`), script 10.

### 4.2 Diagnóstico de confound topológico (RQ1)
- **Conteúdo:** RF([num_nodes]) vs RF([BERT, num_nodes]).
- **Achado:** F1(num_nodes só)=0.523 — tamanho **não é** confound dominante (gate=0.65 não dispara).
- **Implicação:** o sinal do GNN não é só "fake viraliza mais" no FNN.
- **Âncora:** Script 11. (Tabela própria? Considerar adicionar T11 no script 24.)

### 4.3 Cross-dataset UPFD: nosso vs literatura (RQ1)
- **Conteúdo:** GCN/GAT/SAGE em UPFD-PolitiFact e UPFD-GossipCop, 3-5 seeds; **comparação aproximada** contra Dou et al. 2021 Tabela 4.
- **Achado:** PolitiFact ~0.78–0.82 (vs 0.846 publicado); GossipCop ~0.94–0.95 (vs 0.97).
- **Importante (declarado no texto):** **NÃO é replicação exata.** Diferenças deliberadas: (a) feature `content` 310d vs `bert` 768d (custo de RAM em CPU/Windows); (b) hiperparâmetros próprios (lr=0.001, hidden=64, 3 camadas vs configuração não totalmente especificada no paper); (c) métrica F1-macro vs accuracy. A comparação serve para verificar que nosso pipeline está *no envelope* da literatura, não para ranking competitivo.
- **Implicação:** achados subsequentes não são artefato de implementação errada.
- **Âncora:** T3 (`T3_cross_dataset_upfd.tex`), T10 (`T10_upfd_vs_publicado.tex`), script 13.

### 4.4 Topologia sem texto: o achado central (RQ3, parte a)
- **Conteúdo:** 3 variantes (A=`is_root`, B=estrutural, C=posicional) × 3 archs × 10 seeds.
- **Achado:** SAGE-GossipCop com **apenas `[is_root, grau_norm]`** atinge **F1m=0.810 (std=0.0024)** — *sem texto*. PolitiFact UPFD não passa de ~0.55 (chance estrutural).
- **Sobre a baixa variância (std=0.002):** auditoria do `script 14` confirma que (a) `torch.manual_seed(seed)` + `np.random.seed(seed)` são chamados antes de cada run, (b) DataLoader com `shuffle=True` consome do gerador semeado, (c) o construtor da SAGE recebe `seed=seed`. A variância baixa é **consequência da combinação** (i) features de entrada de apenas 2 dimensões — pouco a aprender que dependa de inicialização — (ii) sinal estrutural muito forte no GossipCop (Cohen's d=1.53) que leva o otimizador ao mesmo ótimo independentemente da seed, (iii) split train/val/test fixo (UPFD oficial). Para verificação: a **mesma arquitetura SAGE no UPFD-PolitiFact (Cohen's d ≈ 0)** produz std de ~0.05, ordens de grandeza maior — confirmando que o std baixo do GossipCop é função do dataset, não de bug de seed.
- **Implicação:** detecção topológica pura é viável em domínios certos; **resposta direta à RQ3 (parte a)**: sim, *condicional* à existência de sinal estrutural mensurável.
- **Âncora:** T4 (`T4_topologia_sem_texto.tex`), script 14, CSV em `fase4_benchmarks/topologia_sem_texto/resultados.csv`.

### 4.4-bis Consistência de score topológico entre idiomas (RQ3, parte b)
- **Escopo declarado (lead):** este experimento **não mede detecção em PT/DE** (Bluesky não tem ground-truth). Ele mede **consistência da distribuição de score** que o classificador estrutural produz para posts em diferentes idiomas. Consistência é **condição necessária mas não suficiente** para detecção multilíngue funcionar.
- **Conteúdo:** Aplicar o RF estrutural (treinado em GossipCop, inglês) sobre 3 subconjuntos monolíngues disjuntos do Bluesky: EN (132.312 posts), DE (9.686), PT (2.661). Idiomas extraídos do campo `langs` nativo do AT Protocol — sem inferência por modelo.
- **Hipótese H0:** distribuição de score 'fake-like' é igual entre idiomas.
- **Métricas:** Cohen's d (effect size, robusto a N), Kolmogorov-Smirnov 2-sample, Mann-Whitney U.
- **Achado:**
  - **PT vs EN: d = −0.057.** KS e MW rejeitam H0 com p < 10⁻⁵, **mas o tamanho amostral (n_EN = 132k) torna qualquer diferença real estatisticamente detectável**. O Cohen's d está bem abaixo do limiar de efeito pequeno (Cohen 1988, |d|<0.2): a diferença é estatisticamente detectável mas **praticamente irrelevante**.
  - **DE vs EN: d = −0.468** → efeito médio. Posts em alemão têm score topológico sistematicamente menor (média 0.26 vs 0.40).
  - **PT vs DE: d = +0.739** → efeito médio-grande. PT está mais próximo de EN do que de DE.
- **Interpretação (uma de várias hipóteses plausíveis):** atribuímos a divergência DE vs EN a padrões de engajamento da comunidade germanófona no Bluesky (uso mais discreto da plataforma, menos repost/reply). **Hipóteses alternativas igualmente válidas, não descartáveis sem experimento adicional:**
  1. **Shift de domínio:** classificador foi treinado em GossipCop (celebridade/entretenimento); posts DE no Bluesky podem ser mais políticos/profissionais — a divergência seria de domínio, não de idioma/cultura.
  2. **Viés amostral:** usuários DE no Bluesky podem constituir um subgrupo demográfico/profissional específico (early adopters, comunidade tech) e não representar "a comunidade germanófona em geral".
  3. **Desbalanço de cobertura:** n_PT (2.7k) vs n_DE (9.7k) vs n_EN (132k) é robusto para Cohen's d, mas a cobertura de regimes raros de engajamento é desigual.
- **Resposta refinada à RQ3 (parte b):** o classificador estrutural **não processa texto**, portanto seu mecanismo de decisão é arquiteturalmente independente do idioma do post. A *consistência empírica* dessa independência foi confirmada para PT vs EN; foi parcialmente refutada para DE vs EN, com causa não isolada entre idioma, comunidade, domínio temático e amostragem.
- **Âncora:** Script `26_rq3_multilingual.py`. Tabela T11 (a gerar via script 24). Figuras F14 (histograma sobreposto 3 idiomas) e F15 (Q-Q plot). CSVs em `figuras_tcc/rq3_multilingual/`.

### 4.5 Por que GossipCop funciona e PolitiFact não? (RQ2)
- **Conteúdo:** análise de Cohen's d entre fake/real para `num_nodes`, `branching_factor`, `width`, `depth`, `density`.
- **Achado:** GossipCop tem **branching factor d=+1.53** (efeito grande, Cohen 1988); PolitiFact tem `|d| < 0.2` em todas as métricas (efeito desprezível).
- **Fundamentação teórica:** o resultado é coerente com a literatura de **homofilia/assortatividade** em redes (Newman 2003) e com observações de que GNNs convolucionais dependem de assortatividade por classe para serem efetivas (Pei et al. ICLR 2020, "Geom-GCN"). Em GossipCop, fake e real são *estruturalmente assortativos* (cada classe forma cascatas com perfis de branching distintos); em PolitiFact, não. Isso explica *por que* a mesma arquitetura SAGE rende F1=0.81 em um e F1≈0.55 no outro, sem precisar postular falha de modelo.
- **Implicação:** **resposta direta à RQ2** — topologia ajuda quando há diferença estrutural mensurável entre classes (Cohen's d ≥ 0.5 em ao menos uma métrica de cascata). Tese central justificada com base teórica, não só descritiva.
- **Âncora:** T5 (`T5_cohens_d_estrutural.tex`), F1, F2, script 15.

### 4.6 Concordância textual × topológico (RQ1)
- **Conteúdo:** No UPFD-GossipCop test, comparar predições do LogReg-BERT vs SAGE-estrutural; Cohen's kappa; F1 quando concordam vs discordam.
- **Achado:** Quando concordam, F1=0.97. Quando discordam, textual F1=0.91 vs topológico F1=0.08.
- **Sobre o F1=0.08 do topológico na discordância (parágrafo dedicado):** F1=0.08 é *substancialmente abaixo de chance* (~0.50 em problema balanceado) e também abaixo do que seria predição puramente majoritária (~0.33). Isso indica que, no subconjunto onde os classificadores divergem, o topológico está **sistematicamente invertido**, não apenas "errado". A explicação é uma **inversão distributiva**: no subconjunto de discordância, a relação estrutura→classe está aproximadamente espelhada em relação ao train set. O RF aprendeu a regra "fake = grau_root alto" no GossipCop (Cohen's d=+1.53 confirma isso no agregado); mas as predições onde discorda do textual são justamente (i) **notícias reais que viralizaram fortemente** (grau_root alto → RF prevê fake, errando) e (ii) **fake que tiveram propagação contida** (grau_root baixo → RF prevê real, errando). Em conjunto, essas amostras formam um subconjunto onde a regra estrutural se inverte — o que produz o F1=0.08 observado. Esse comportamento *justifica* a assimetria 0.8/0.2 da regra dual, não a contradiz: o topológico é confiável apenas quando concorda com o textual; quando discorda, está reportando o cenário onde sua heurística falha por construção.
- **Implicação:** justifica a regra **post-hoc** 0.8/0.2 da API e, mais importante, fornece um critério interpretável de confiança (concordância) que pode ser exposto ao usuário final.
- **Âncora:** T6 (`T6_concordancia_gossipcop.tex`), F3, script 20.

### 4.7 GNNExplainer: o que o SAGE aprendeu (RQ1, interpretabilidade)
- **Conteúdo:** Aplicar GNNExplainer em **20 amostras estratificadas** do SAGE-GossipCop test (5 fake-pequena, 5 fake-grande, 5 real-pequena, 5 real-grande, mediana de num_nodes = 49) e **10 amostras** do SAGE-PolitiFact (estratos REAL com 0 amostras corretas — modelo não classifica REAL = limitação reportada).
- **Achado central (assimetria por classe):** a fração de importância concentrada em arestas a **1-hop** (raiz→filhos diretos) varia drasticamente por classe:
  - **FAKE-pequena/grande (N=5+5):** hop1 ≈ **0.97-1.00** em todas as 10 amostras (modelo decide quase exclusivamente pela vizinhança imediata da raiz).
  - **REAL-grande (N=5):** hop1 ≈ **0.50-0.62** (massa distribuída entre hops mais distantes).
  - **REAL-pequena (N=5):** hop1 sem padrão consistente, com amostras concentradas nos extremos (0.05-0.07 e 0.93-0.95). **Reportado como observação preliminar — N=5 é insuficiente para caracterizar modo bimodal versus alta variância intra-grupo.**
- **Interpretação:** o SAGE aprendeu uma heurística *assimétrica*: usa intensamente o grau direto da raiz para confirmar FAKE (consistente com a observação de que fake news viralizam = grau_root alto); para classificar REAL-grande, o modelo *busca* informação em estrutura mais distante, com sucesso parcial. Essa assimetria explica por que o RF tabular (que vê apenas `[num_nodes, grau_root]`) captura a maior parte do sinal — para a classe FAKE, esse sinal é *suficiente*; para REAL, é onde o GNN agrega valor. **Quantitativamente:** RF estrutural F1=0.753 (`metadata.json`, treinado em UPFD-GossipCop train+val, avaliado em 3826 grafos do test); SAGE estrutural F1=0.814 no mesmo split — diferença de **6.1pp** absolutos atribuível ao GNN.
- **Implicação adicional para o capítulo de Aplicação:** a confiança do modelo deve ser ponderada — predições FAKE com hop1 alto ≈ "modelo viu o que esperava"; predições REAL ≈ "modelo trabalhou mais e tem mais incerteza". Isso pode informar o design da interface (mostrar concentração da explicação como proxy de confiança).
- **Âncora:** F12 (`F12_gnnexplainer_gossipcop/`), script 16 refatorado, `hop_importance.csv` em `figuras_tcc/gnnexplainer/{gossipcop,politifact}/`.

*[§4.8 anterior — "Bloco grande vs especialistas" — movido para §5.6.1, pois é decisão de engenharia/produto, não respondia a RQ1/2/3.]*

---

## Capítulo 5 — Aplicação: Bluesky e Ferramenta Web

**Lead:** "Este capítulo é demonstração, não validação. Bluesky não tem labels fake/real; portanto não fazemos claims supervisionados."

### 5.1 Dataset Bluesky (acadêmico, 6 GB)
- 168k posts, 11 feeds temáticos, AT Protocol. Sem ground-truth.
- Justificativa de uso: pipeline real-world, demonstrar generalização *qualitativa*.
- **Âncora:** README §3, script 18.

### 5.2 Análise cross-feed (Cohen's d entre feeds)
- **Achado:** distribuições de likes/reposts/replies variam fortemente por feed (Political Science vs Blacksky).
- **Âncora:** F5, F6, script 18.

### 5.3 Aplicação dos modelos persistidos
- RF estrutural aplicado em todos os 168k posts; distribuição de "score fake-like" por feed.
- **Limitação explícita:** "score" não é prob calibrada; é proxy treinado em GossipCop.
- **Âncora:** T8 (`T8_bluesky_inferencia_feeds.tex`), F7, script 19.

### 5.4 Concordância textual × topológico no Bluesky
- 5k posts amostrados; LogReg-BERT vs RF-estrutural; cross-tab por tamanho/feed/comprimento de texto.
- **Achado:** Political Science = 88.7% agreement (alto). Posts médios (2-20 nós) = 42-44% (zona cinzenta). Posts virais (20-100 nós) = 74%.
- **Implicação:** classificadores convergem em casos extremos, divergem em "zona cinzenta" — informação útil para a interface.
- **Âncora:** T9, F8, F9, F10, F11, script 22.

### 5.5 Visualização de threads reais
- 6 threads de tamanhos variados, score sobreposto.
- **Âncora:** F13, script 23.

### 5.6 Arquitetura da ferramenta web
- FastAPI backend + Next.js frontend; carrega 3 modelos persistidos; regra dual de combinação.
- Endpoint `/api/result` retorna 3 scores + flag de concordância.
- **Âncora:** `Interface/frontend/api/main.py`, README.

### 5.6.1 Decisão de produto: bloco único vs especialistas (movido de §4.8)
- **Conteúdo:** Treinar UM SAGE em UPFD-PolitiFact + UPFD-GossipCop concatenados (com `dataset_id` como feature) vs dois especialistas separados.
- **Achado:** BLOCO ≈ MINI; vantagem leve em datasets pequenos (PolitiFact +0.023).
- **Implicação:** decisão de produto — 1 modelo grande é suficiente; reduz custo operacional e simplifica deploy. **Esta subseção é justificada como engenharia, não responde RQ científica.**
- **Âncora:** T7 (`T7_bloco_vs_mini.tex`), F4, script 21.

### 5.7 Pesos no Hugging Face Hub
- Persistência reproduzível; model card com limitações; CC-BY-4.0.
- **Âncora:** Script 25, repo HF.

---

## Capítulo 6 — Discussão e Conclusão

| § | Seção | Conteúdo |
|---|-------|----------|
| 6.1 | Síntese dos achados | Resumir RQ1–RQ3 com resposta de cada. |
| 6.2 | Achado negativo é achado | Defender explicitamente o resultado de PolitiFact: não é "modelo ruim", é "domínio sem sinal estrutural". Cite Cohen's d. |
| 6.3 | Limitações | (a) FNN com cascatas estrela (Item 5A.5); (b) regra 0.8/0.2 derivada de UM experimento; (c) Bluesky sem labels = só demo; (d) reprodutibilidade não bit-exact; (e) sem comparação numérica direta com BiGCN/GCNFN; **(f) Bluesky é snapshot temporal de 2024-2025 (~6 GB), não amostra probabilística — generalização a janelas temporais futuras não é garantida; (g) ambiente CPU-only (Python 3.12.10, torch 2.10+cpu): hiperparâmetros podem ser subótimos em GPU, embora o pipeline rode em ambos.** |
| 6.4 | Trabalhos futuros | Pseudo-labels no Bluesky (já planejado p/ Item 4 da lista do user); calibração de probabilidades; expansão pra PolitiFact-like real (Twitter recente?). |
| 6.5 | Considerações finais | 1 parágrafo curto, sem grandiloquência. |

---

## Anexos sugeridos
- **A.** Tabela completa de hiperparâmetros por arquitetura.
- **B.** INDICE.md de figuras/tabelas (já gerado pelo script 24).
- **C.** Estrutura de diretórios do repositório.
- **D.** Evolução metodológica (opcional, se a banca pedir contexto sobre por que a versão final difere de propostas anteriores). Conteúdo: tabela curta com decisões de design revisitadas durante a execução do TCC. Não entra no corpo principal — fica como apêndice histórico para examinador interessado.

---

## Próximos passos após aprovação deste outline

1. Crítica adversarial (vc + outro IA) → ajustes na estrutura
2. **Passo 2:** revisão de `Referencias.bib` — lista anotada do que adicionar/remover
3. **Passo 3:** escrita capítulo a capítulo, validando cada um antes de prosa final
