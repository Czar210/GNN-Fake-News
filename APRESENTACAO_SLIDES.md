# Roteiro de Slides — Defesa do TCC

**Detecção de Fake News com Redes Neurais de Grafos: Ablation Sistemática e Diagnóstico de Aplicabilidade Topológica**
André Messina Livingston · César Augusto Sibila · Enzo Kikuji Takida
PUC-SP · Curso de Ciência de Dados e Inteligência Artificial · 2026

> **Como usar este arquivo.** Cada slide tem: **VISUAL** (o que aparece na tela — pouco texto!), **NA TELA** (os bullets/títulos que ficam projetados) e **FALA** (o que vocês dizem — roteiro, não para ler em voz alta palavra por palavra). Os números já estão conferidos contra o `FATOS_TCC.md`. A meta é **20 min**: ~17 slides de conteúdo, ritmo de ~1 a 1,5 min por slide, deixando ~5 min de folga para a banca.
>
> A regra de ouro da defesa: **uma ideia por slide, o slide é pano de fundo, vocês são a apresentação.** Não leiam o slide.

---

## Estrutura geral (o arco da história)

A apresentação conta uma história em 3 atos, espelhando o arco do próprio TCC:

1. **O problema e a virada de chave** (slides 1–4) — por que detectar fake news só pelo texto está ficando frágil, e a aposta: e se olhássemos *como* a notícia se espalha, não só *o que* ela diz?
2. **O método e o achado central** (slides 5–9) — a ficha técnica (dados, arquiteturas, rigor) e o resultado bilateral: funciona num dataset, colapsa no outro.
3. **O porquê, a aplicação e a reflexão** (slides 10–17) — a tese falsificável que explica os dois lados, o aprofundamento (multi-hop), a ferramenta, e o fechamento honesto.

### Divisão entre os três (montada para as forças de cada um)

A divisão **não** é em três blocos iguais — é por perfil, para proteger a nota:

| Apresentador | Slides | Tempo | Papel e por quê |
|---|---|---|---|
| **André** (curte tecnologia, evita matemática) | **1–6** | ~7 min | Abre e conta a história (problema → virada de chave → perguntas) + ficha técnica de **dados e arquiteturas**. É tecnologia pura, **sem estatística**. O *porquê-matemático* de a SAGE vencer ele passa pro César. |
| **César** (o motor) | **7–11 e 16–17** | ~8 min | Leva **todo o núcleo matemático** (rigor, achado central, tese do Cohen's *d*, multi-hop) e **fecha** (limitações + conclusão), **conduzindo a arguição**. É quem aguenta as perguntas duras. Bluesky também é dele no Q&A (ele curou o dataset). |
| **Enzo** (está se reaproximando do trabalho) | **12–15** | ~5 min | Bloco **mais seguro e 100% roteirizável**: aplicação (RQ3/Bluesky, ferramenta web), painel mestre e virtudes. São slides de "ler a figura / mostrar o que construímos", de baixo risco de pergunta brutal. Fica **no meio, blindado pelo César** dos dois lados. |

> **Por que César no meio E no fim:** o miolo matemático (slides 8–9) é o pico técnico e o fecho (16–17) leva direto pra arguição — os dois pedem a pessoa mais segura. André abre forte e relaxa; Enzo entrega um bloco concreto e protegido; César ancora o que pode dar pergunta.

> **O que cada um TEM que saber de cor (mínimo inegociável):**
> - **Todos os três:** a frase da tese central (slide 9) e os 3 números-âncora (F1=0,81 GossipCop / 0,33 PolitiFact / Cohen's *d* +1,53).
> - **César:** tudo. É a rede de segurança.
> - **André:** o que cada arquitetura faz (1 frase cada) e por que usar 3 (não depender de uma só).
> - **Enzo:** os 4 scripts dos slides 12–15 decorados + saber dizer, em uma frase, qual é a contribuição do trabalho ("um critério de *quando* a topologia funciona").

> **Transições (frase de gancho ao passar a bola):**
> - André→César (após slide 6): *"...a SAGE foi a melhor. Por que ela vence — e o quanto isso importa — quem mostra é o César."*
> - César→Enzo (após slide 11): *"...a partir daqui o Enzo mostra como isso vira aplicação no mundo real."*
> - Enzo→César (após slide 15): *"...pra fechar, o César assume as limitações e a conclusão."*

---

# ATO 1 — O problema e a virada de chave

## Slide 1 — Capa

**VISUAL:** Título, três autores, orientador, PUC-SP, Ciência de Dados, 2026. Fundo limpo. (Ver guia de design.)

**NA TELA:**
- *Eyebrow:* Defesa de TCC · Ciência de Dados e Inteligência Artificial
- **Detecção de Fake News com Redes Neurais de Grafos**
- *Ablation Sistemática e Diagnóstico de Aplicabilidade Topológica*
- André Messina Livingston · César Augusto Sibila · Enzo Kikuji Takida
- Orientação: Prof. Eric Bacconi Gonçalves · Prof. Rooney Ribeiro Albuquerque Coelho
- Pontifícia Universidade Católica de São Paulo (PUC-SP) · 2026

> **Capa — obrigatório constar:** PUC-SP + "Ciência de Dados e Inteligência Artificial" + os três autores + os dois orientadores + ano. Logo da PUC no canto (ver guia de design, template A).
> **Atenção ao título:** manter **"Ablation Sistemática"** (igual ao TCC depositado e à *keyword* do resumo) — não trocar por "Ablação", senão a capa diverge do PDF que a banca tem em mãos.

**FALA (André):** "Bom dia. Nosso trabalho investiga uma pergunta simples de enunciar e difícil de responder: dá pra detectar fake news **sem ler o texto** — só olhando como a notícia se espalha pela rede? A resposta, como a gente vai mostrar, é *depende* — e a contribuição do trabalho é justamente dizer **de quê** depende."

---

## Slide 2 — O problema é real e é de escala

**VISUAL:** Imagem/ícone de propagação em rede + 1 dado de impacto. Pouquíssimo texto.

**NA TELA:**
- Desinformação = problema epistêmico com dano físico real (COVID-19: "remédios milagrosos", uso de desinfetante)
- Escala de *Big Data*: criar uma mentira é instantâneo; desmenti-la é lento e caro
- → Verificação manual é inviável. Detecção automática é necessidade técnica.

**FALA (André):** "A desinformação não é só ruído de internet — tem dano documentado. Na pandemia, narrativas falsas levaram gente a tomar risco real. E o volume é de Big Data: produzir uma mentira é instantâneo, checar é lento. Não dá pra fazer na mão. Precisa de detecção automática."

---

## Slide 3 — Por que o texto sozinho não basta (a virada de chave)

**VISUAL:** Lado a lado — esquerda "TEXTO (PLN)" com um ⚠️; direita "TOPOLOGIA (como espalha)". Setinha da esquerda enfraquecendo, da direita robusta.

**NA TELA:**
- Abordagem tradicional = PLN sobre o texto. Dois pontos cegos:
  - **LLMs** já escrevem texto falso gramaticalmente impecável → fronteira texto-humano/sintético se desfaz
  - **Idiomas** não cobertos por modelos pré-treinados → texto vira inexplorável
- **A aposta:** a desinformação não se distingue só pelo *conteúdo*, mas pelo **padrão com que se espalha** — quem compartilha o quê, para quem, em que ordem.
- Sinal topológico: não depende do idioma, resiste a paráfrase e a texto de LLM.

**FALA (André):** "A forma clássica de detectar é analisar o texto com PLN. Mas isso tem dois furos crescentes. Primeiro: LLMs hoje escrevem mentira com gramática perfeita — a fronteira entre texto humano e sintético some. Segundo: se a notícia está num idioma sem bom modelo, o texto vira inútil. A virada de chave do trabalho é olhar pra outra coisa: **não o que a notícia diz, mas como ela se espalha**. Esse padrão de propagação é um grafo — e é independente do idioma."

---

## Slide 4 — As três perguntas de pesquisa

**VISUAL:** Três cartões empilhados, RQ1/RQ2/RQ3, cada um com uma frase curta.

**NA TELA:**
- **RQ1** — A topologia carrega sinal *acima* do que o texto da raiz (BERT) já fornece?
- **RQ2** — *Sob quais condições estruturais* isso vale? (operacionalizado por Cohen's *d*)
- **RQ3** — Dá pra detectar fake news **só** com topologia, quando não há texto explorável? (a: sem texto; b: multilíngue)

**FALA (André):** "O trabalho inteiro se organiza em torno de três perguntas. A RQ1: a estrutura ajuda além do texto? A RQ2 é a mais importante: *sob quais condições*? — porque a gente desconfiava que não fosse sempre. E a RQ3: dá pra ir ao extremo e usar **só** estrutura, sem texto nenhum? É isso que fomos medir — começando pelos dados e pelas arquiteturas que usamos."

---

# ATO 2 — O método e o achado central

## Slide 5 — Ficha técnica: os dados

**VISUAL:** Tabela enxuta de 4 linhas + ilustração da árvore de propagação (Fig 2.1 do TCC: raiz = notícia, filhos = reposts).

**NA TELA:**

| Dataset | Grafos | Papel |
|---|---|---|
| FakeNewsNet PolitiFact | 754 | construído por nós (estrela) |
| UPFD-PolitiFact | 314 | benchmark oficial (Dou et al. 2021) |
| UPFD-GossipCop | 5.464 | benchmark oficial — o "cavalo de batalha" |
| **Bluesky** | 168.463 posts / 11 feeds | snapshot real, multilíngue, **autocurado** (publicado no HF) |

- Cada notícia vira um **grafo de propagação**: raiz = a notícia; filhos = quem repostou/respondeu.

**FALA (André):** "Usamos três datasets de benchmark — dois do PolitiFact e o GossipCop, que é o maior, com 5.464 grafos. A ideia: cada notícia vira um grafo, onde a raiz é a notícia e os filhos são as interações. Além disso, montamos uma aplicação sobre o Bluesky — 168 mil posts reais, multilíngues — que **publicamos no Hugging Face**. Usamos datasets oficiais justamente pra poder comparar com a literatura e não inventar a régua."

---

## Slide 6 — Ficha técnica: as três arquiteturas

**VISUAL:** Três colunas (GCN / GAT / SAGE), cada uma com um mini-diagrama (Figs 2.2–2.4) e a "mecânica" em uma linha.

**NA TELA:**
- **GCN** — convolução espectral, média dos vizinhos (~58k params). *Baseline clássico.*
- **GAT** — atenção: aprende **pesos** por vizinho, 4 heads (~218k params).
- **GraphSAGE** — indutiva, separa "eu" do "meu vizinho" (Wₗ·h + Wᵣ·h_𝒩) (~115k params). **A que venceu.**
- Por que **três**? Para mostrar que o achado **não é artefato de uma arquitetura** — é uma propriedade dos dados.

**FALA (André):** "Comparamos três arquiteturas de propósito. GCN, a clássica, que faz média dos vizinhos. GAT, que aprende com *atenção* o peso de cada vizinho. E GraphSAGE, indutiva, que separa a representação do próprio nó da dos vizinhos. Testamos as três por um motivo metodológico: se o resultado dependesse de uma arquitetura específica, seria um truque. Quando as três concordam, o sinal está nos **dados**, não no modelo. E spoiler: a SAGE foi a melhor. **Por que ela vence — e o quanto isso importa — quem mostra é o César.**"

---

## Slide 7 — Ficha técnica: rigor (por que dá pra confiar)

**VISUAL:** Quatro selos/ícones: ablation · 10-fold · Cohen's *d* · bootstrap.

**NA TELA:**
- **Ablation de features**: desligamos sistematicamente o texto e cada *encoding* posicional → isola de onde vem o sinal
- **Protocolo estatístico**, não "rodou uma vez": k-fold pareado + `ttest_rel`, **10 seeds** no achado central, **bootstrap** (2.000 reamostragens) nos subgrupos
- **Cohen's *d*** para medir *quão separadas* fake e real estão estruturalmente — não só "se" mas "quanto"
- **CPU-only**, seed 42, reprodutível: não precisa de GPU pra repetir

**FALA (César):** "O que separa isto de um experimento de fim de semana é o rigor. Três coisas. Um: *ablation* — desligamos o texto e cada feature, uma a uma, pra saber de onde o sinal realmente vem. Dois: estatística de verdade — não rodamos uma vez e comemoramos; é k-fold pareado com teste-t, 10 seeds no resultado principal, e bootstrap nos subgrupos. Três: Cohen's *d*, que mede o *tamanho* da diferença entre fake e real. E tudo roda em CPU, com seed fixa — qualquer um repete."

> **Se a banca perguntar de `num_nodes` ser confound:** já testamos (§4.2). RF só com número de nós dá F1≈0,52, abaixo do nosso *gate* de 0,65. O sinal não é "grafo grande = fake".

---

## Slide 8 — O achado central: bilateral

**VISUAL:** **A figura que vende o trabalho.** Use a Fig 4.1 ou um par de barras: GossipCop em **verde** (sobe a 0,81) vs PolitiFact em **vermelho** (despenca a 0,33, abaixo da linha de chance 0,50).

**NA TELA:**
- GraphSAGE com **2 features estruturais por nó**, **sem texto nenhum**:
  - ✅ **UPFD-GossipCop: F1 = 0,810** (± 0,002, 10 seeds) — quase o teto textual
  - ❌ **UPFD-PolitiFact: F1 = 0,33** — *abaixo da chance* (0,50). Colapso.
- Mesma arquitetura. Mesmo pipeline. Resultado oposto.

**FALA (César):** "Aqui está o coração do trabalho. A mesma GraphSAGE, com **só duas features estruturais por nó e zero texto**, atinge F1 de 0,81 no GossipCop — perto do teto que o texto dá. Mas a **mesma** arquitetura, no PolitiFact, **colapsa pra 0,33** — abaixo da chance, que é 0,50. Ou seja: às vezes a topologia sozinha quase resolve o problema; às vezes destrói. A pergunta óbvia é: **por quê?** E essa é a parte de que a gente mais se orgulha."

---

## Slide 9 — O porquê: a tese falsificável (Cohen's *d*)

**VISUAL:** Forest plot do Cohen's *d* (Fig 4.3) com as linhas de limiar 0,2 / 0,5 / 0,8. GossipCop estoura a +1,53; PolitiFact e FNN ficam abaixo de 0,5.

**NA TELA:**
- A diferença estrutural entre fake e real, medida por Cohen's *d*:
  - GossipCop: ***d* = +1,53** no *branching factor* (efeito **grande**)
  - PolitiFact: **|*d*| < 0,5** em *toda* métrica (no máximo 0,25)
- **Tese central (deliberadamente falsificável):**
  > *"Modelos topológicos exigem |d| ≥ 0,5 em pelo menos uma métrica de cascata para serem efetivos; abaixo disso, colapsam — independentemente da arquitetura GNN."*

**FALA (César):** "A explicação é limpa. A gente mede, com Cohen's *d*, o quão diferentes são as estruturas de fake e real. No GossipCop, fake e real têm formas **muito** diferentes — *d* de 1,53, efeito grande. No PolitiFact, são quase idênticas — *d* abaixo de 0,5. Daí a nossa tese, e ela é **falsificável** de propósito: a topologia só funciona quando há diferença estrutural de pelo menos um efeito médio de Cohen. Abaixo disso, qualquer GNN colapsa. Isso é raro num trabalho de ML — a gente formulou uma regra que pode ser **refutada** por um contraexemplo. E ela não para no número agregado — olhem só onde esse ganho se concentra."

> **Esse é o slide-chave da defesa.** Se a banca lembrar de uma coisa, que seja esta. Vale projetar a frase da tese em destaque (slide tipo "citação").

---

# ATO 3 — Aprofundamento, aplicação e reflexão

## Slide 10 — Não é só tamanho: o ganho é multi-hop

**VISUAL:** Fig 4.13 ou 4.14 — gap RF→SAGE crescendo com a profundidade da cascata (barras verdes).

**NA TELA:**
- Será que a SAGE só está "contando nós"? **Não.**
- Ganho do SAGE sobre o RF tabular (só `num_nodes`, `grau_root`) **cresce com a profundidade**:
  - depth ≥ 1: **+0,060** → depth ≥ 5: **+0,213** (3,5×) — IC 95% bootstrap não cruza zero
- **GNNExplainer** confirma: em fakes, a SAGE concentra importância a 1-hop; em reais grandes, distribui para hops distantes → ela lê **propagação**, não só tamanho.

**FALA (César):** "Uma crítica natural: 'a SAGE só está medindo se o grafo é grande'. A gente testou isso. Comparamos a SAGE com um Random Forest que só vê tamanho e grau. O ganho da SAGE **cresce com a profundidade** da cascata — de 6 pontos no geral pra 21 pontos nas cascatas mais profundas, com intervalo de confiança que não toca o zero. E o GNNExplainer mostra *onde* ela olha: ela usa propagação multi-hop, não só o tamanho. Esse é o mecanismo por trás do número."

---

## Slide 11 — Texto × topologia: quando concordam e quando brigam

**VISUAL:** Matriz/scatter de concordância (Fig 4.4) + a regra dual em destaque.

**NA TELA:**
- Modelo textual e topológico no GossipCop (Cohen's κ = 0,52, concordância moderada):
  - Quando **concordam** → F1 = **0,97** (altíssima confiança)
  - Quando **discordam** → o topológico despenca pra F1 = **0,08**; o textual segura 0,91
- → **Regra dual** (derivada *post-hoc*): concordam → 0,5/0,5; discordam → **0,8 texto / 0,2 topologia**

**FALA (César):** "Os dois sinais — texto e estrutura — não são redundantes nem sempre concordam. Quando concordam, a confiança é altíssima: F1 de 0,97. Quando discordam, é aí que mora o perigo: o sinal topológico desaba pra 0,08, enquanto o textual aguenta. Disso derivamos uma regra prática de combinação: peso igual quando concordam, e favorece o texto quando brigam. **Importante: essa regra foi derivada depois de ver os dados** — a gente trata como conjectura, não como lei, e diz isso nas limitações. A partir daqui, o Enzo mostra como isso vira aplicação no mundo real."

---

## Slide 12 — RQ3: a estrutura é agnóstica a idioma

**VISUAL:** Distribuição do score topológico PT vs EN praticamente sobrepostas (Fig 4.2).

**NA TELA:**
- Aplicamos o classificador **estrutural** (não vê texto) sobre subconjuntos do Bluesky por idioma:
  - **PT vs EN: *d* = −0,057** → distribuições de score **estatisticamente equivalentes**
  - DE vs EN: *d* = −0,468 (pequeno, mas não trivial)
- Como o classificador **não processa texto**, ele é arquiteturalmente **independente do idioma** — exatamente o que a RQ3 buscava.

**FALA (Enzo):** "Aqui fechamos a RQ3, a promessa lá do começo. Pegamos o classificador puramente estrutural e aplicamos em posts do Bluesky em idiomas diferentes. Entre português e inglês, a diferença é de −0,057 — praticamente zero. Faz sentido: se o modelo não lê texto, o idioma não importa pra ele. Essa é a vantagem estrutural do paradigma topológico, demonstrada num dado real e multilíngue."

> **Honestidade (se perguntarem):** o Bluesky **não tem rótulos** fake/real. Aqui medimos *consistência distributiva*, não acerto supervisionado. É demonstração de aplicabilidade, não validação de acurácia — e está dito como limitação (c).

---

## Slide 13 — Da pesquisa ao artefato: a ferramenta

**VISUAL:** Print da ferramenta web + o fluxograma da regra dual (Fig 1.1).

**NA TELA:**
- Três modelos persistidos (LogReg-BERT, RF estrutural, SAGE estrutural) combinados pela regra dual → **ferramenta web** (Next.js + FastAPI)
- Carrega em ~1,4 s, infere em milissegundos, **roda em CPU**
- **Publicado:** dataset Bluesky processado + pesos no **Hugging Face Hub**

**FALA (Enzo):** "O trabalho não parou no PDF. Empacotamos os três modelos numa ferramenta web que recebe um post e devolve um score combinado pela regra dual, com uma *flag* dizendo se texto e estrutura concordam. Roda em CPU, infere em milissegundos. E publicamos o dataset Bluesky tratado e os pesos no Hugging Face — qualquer pessoa pode reproduzir e usar."

---

## Slide 14 — Painel mestre (a foto de tudo)

**VISUAL:** Fig 4.15 — o painel com três datasets × três tipos de modelo (azul textual / verde topológico puro / laranja completo).

**NA TELA:**
- Leitura **horizontal** (mesmo modelo, datasets diferentes): topologia pura quase alcança o texto no GossipCop; **despenca** perto da chance no PolitiFact.
- Leitura **vertical** (mesmo dataset): no PolitiFact, somar topologia ao texto **não acrescenta nada**.
- Tudo isso previsto pelo Cohen's *d*.

**FALA (Enzo):** "Esta figura resume o trabalho inteiro. Lendo na horizontal: o modelo topológico puro, em verde, encosta no textual no GossipCop e desmorona no PolitiFact. Lendo na vertical: no PolitiFact, a estrutura não adiciona nada ao texto. E os dois efeitos são previstos pela mesma régua — o Cohen's *d*. Uma figura, a tese inteira."

---

## Slide 15 — As virtudes do trabalho (por que ele importa)

**VISUAL:** 4–5 selos/ícones. Slide de "respiro" antes do fecho.

**NA TELA:**
- **Diagnóstico, não demonstração** — a literatura tende a reportar só o que funciona; nós reportamos o caso negativo *com o mesmo rigor*
- **Tese falsificável** — uma regra que pode ser refutada (Popper), rara em ML aplicado
- **Honestidade bilateral** — o resultado negativo do PolitiFact é tratado como **evidência**, não como fracasso
- **Rigor estatístico** — 10 seeds, teste pareado, bootstrap, Cohen's *d*
- **Reprodutível e acessível** — CPU-only, código + dados + pesos públicos

**FALA (Enzo):** "Por que esse trabalho importa? Três virtudes. Primeira: é um **diagnóstico**, não uma demonstração. A área costuma publicar só os casos em que a topologia funciona; a gente apresenta o caso em que ela **falha** com o mesmo rigor — e isso é mais informativo. Segunda: a tese é **falsificável** — a gente se expôs à refutação de propósito. Terceira: rigor estatístico e reprodutibilidade total, em CPU. O resultado negativo do PolitiFact não é um problema do trabalho; é parte da contribuição. Pra fechar, o César assume as limitações e a conclusão."

---

## Slide 16 — Limitações (assumidas de frente)

**VISUAL:** Lista curta e honesta. Mostrar que vocês *já sabem* das fraquezas desarma a banca.

**NA TELA:**
- **Estrela no FakeNewsNet nosso** — o CSV bruto não tem parentesco entre reposts → grafo vira estrela rasa; e o cap de 100 nós trunca 50% dos grafos. *(Vale só pro FNN que construímos — o UPFD oficial NÃO é estrela: 30% têm profundidade ≥ 3.)*
- **Regra dual 0,8/0,2** — derivada de **um** experimento; é conjectura, não lei
- **Bluesky sem rótulos** — aplicação demonstrativa, não validação supervisionada
- **Sem GPU / não bit-exact entre máquinas** — variação na 3ª casa decimal
- **Sem comparação numérica direta com BiGCN/GCNFN**

**FALA (César):** "E as limitações, que a gente assume de frente. O FakeNewsNet que construímos vira estrela rasa, porque o CSV bruto não tem o parentesco entre reposts — mas é importante: isso vale só pro **nosso** FNN; o UPFD oficial não é estrela, tem 30% de grafos profundos. A regra dual saiu de um experimento só, então é conjectura. O Bluesky não tem rótulos, então ali é demonstração. E rodamos em CPU, sem comparar número a número com BiGCN. Nada disso derruba o achado central — e preferimos dizer do que esconder."

---

## Slide 17 — Conclusão: as três respostas

**VISUAL:** As 3 RQs do slide 4, agora **respondidas**. Fechamento.

**NA TELA:**
- **RQ1** — Depende do dataset. No GossipCop, sim (ganho mensurável); no FNN, o texto já satura.
- **RQ2** — A condição é mensurável **a priori**: **|*d*| ≥ 0,5** em alguma métrica de cascata.
- **RQ3** — Sim, *condicionalmente*: F1 = 0,81 sem texto no GossipCop; e o sinal é **agnóstico a idioma** (PT≈EN).
- **Contribuição central:** não "mais uma GNN que detecta fake news", e sim um **critério de quando usá-la**.

**FALA (César):** "Fechando. RQ1: depende — e a gente sabe de quê depende. RQ2: a condição é **medível antes de treinar** — Cohen's *d* maior que 0,5. RQ3: sim, condicionalmente, e o sinal não depende do idioma. A contribuição não é 'mais uma rede que detecta fake news' — é um **critério de aplicabilidade**: saber, de antemão, quando vale a pena usar topologia e quando é melhor confiar no texto. Obrigado — estamos à disposição para perguntas."

---

## Slide 18 — Obrigado / Perguntas

**NA TELA:** Título, os três nomes, PUC-SP · Ciência de Dados e Inteligência Artificial, e os links (repositório, Hugging Face). Logo da PUC.

---

# Banco de perguntas esperadas (e como responder)

> Treinem isto. Metade da nota da defesa é a desenvoltura na arguição. Respostas curtas, seguras, e sempre ancorando num número do TCC.

**1. Por que o PolitiFact colapsa *abaixo* da chance (0,33 < 0,50)? Não deveria pelo menos chutar 0,50?**
Porque o modelo não fica indeciso — ele aprende uma regra estrutural que está *invertida* naquele domínio. Como fake e real são estruturalmente quase idênticos (|*d*| < 0,5), o pouco sinal que existe aponta pro lado errado de forma sistemática, e o F1-macro pune isso. Isso *confirma* a tese: sem separação estrutural, não há o que aprender.

**2. O cap de 100 nós trunca 50% dos grafos e 92% da massa de nós. Isso não enviesa tudo?**
É uma decisão de projeto que assumimos explicitamente (§3.3, limitação a). Em grafo-estrela, os filhos têm features **idênticas** (BERT replicado), então truncar folhas redundantes é defensável. Mas a magnitude é grande e a gente declara isso — não chamamos de "remoção de outliers", porque seria subdimensionar. Importante: esse cap afeta só o **nosso** FNN; os resultados centrais são no UPFD oficial, que não passa por esse filtro.

**3. Seus números estão abaixo do Dou et al. (0,817 vs 0,846). Por quê?**
Comparação indicativa, não bit-exact (Tabela 4.4). Três diferenças: usamos *feature* `content` (310d) e não `bert` (768d), por restrição de RAM em CPU; reportamos **F1-macro** e eles **accuracy**; e o ambiente é diferente. A *tendência* bate; o foco do nosso trabalho não é bater o SOTA, é o **diagnóstico** de quando a topologia funciona.

**4. A regra dual 0,8/0,2 — de onde vem? É generalizável?**
Foi derivada *post-hoc*, de um único experimento de concordância no GossipCop (§4.6). Tratamos como **conjectura**, não resultado validado — está explícito na limitação (b). Generalizá-la a outros domínios é trabalho futuro.

**5. O Bluesky não tem rótulos. Como vocês validam algo ali?**
Não validamos acurácia ali — e dizemos isso (limitação c). O que medimos é *consistência distributiva* do score entre idiomas (RQ3-b). É evidência de **aplicabilidade independente de idioma**, não de acerto. A validação de acurácia está nos datasets rotulados (UPFD).

**6. `num_nodes` não é um confound? "Grafo grande = fake"?**
Testamos diretamente (§4.2): RF só com `num_nodes` dá F1≈0,52, abaixo do nosso *gate* de 0,65. E mostramos (slide 10) que o ganho da SAGE cresce com a *profundidade*, não com o tamanho — sinal multi-hop, confirmado pelo GNNExplainer.

**7. Por que GraphSAGE venceu GCN e GAT?**
Porque a GCN clássica **degenera** com features nodais homogêneas — provamos isso matematicamente (§2.5): a média espectral colapsa nós idênticos. A SAGE separa a representação do nó da dos vizinhos (Wₗ e Wᵣ distintos) e é indutiva, então sobrevive a features homogêneas. O GAT, com mais parâmetros, não compensou nesse regime de poucas features.

**8. Por que três datasets e não um?**
Pra contrastar. Se só tivéssemos o GossipCop, diríamos "topologia funciona" — falso em geral. O PolitiFact é o que dá o **caso negativo** e permite formular a tese falsificável. O contraste *é* a contribuição.

**9. Isso funciona em português / no mundo real?**
O sinal estrutural é agnóstico a idioma por construção (slide 12, PT≈EN). Para o mundo real, a limitação é coletar a árvore de propagação completa — no nosso FNN o CSV não trazia parentesco. Com dados de propagação ricos (como o UPFD oficial), o paradigma se sustenta.

**10. Qual o trabalho futuro mais importante?**
Validar a regra dual em mais domínios; coletar propagação com parentesco e timestamps reais (sair da estrela); e testar a tese do |*d*| ≥ 0,5 em datasets novos — que é justamente onde ela pode ser refutada.

---

# Checklist de ensaio

- [ ] Cronometrar: 20 min é **pouco**. Cada um cronometra seu bloco isolado.
- [ ] Treinar as **transições** entre apresentadores (a frase de gancho).
- [ ] Decorar **3 números-âncora** cada um. Não precisa decorar tudo — precisa saber onde achar.
- [ ] Slide 9 (tese) e slide 8 (achado bilateral) têm que sair **perfeitos** — são o pico.
- [ ] Preparar 2–3 **slides de backup** (tabelas T4, T5, T13 completas) para puxar se a banca pedir detalhe.
- [ ] Quem responde o quê na arguição: combinem antes. **Estatística / Cohen's *d* / colapso do PolitiFact / cap de 100 nós / curadoria do Bluesky → César** (é a rede de segurança de todos). **Arquiteturas / ferramenta / stack → André.** **Enquadramento / motivação / "qual a contribuição" → Enzo** (com César pronto a complementar). Regra geral: quem domina o tema responde, *independentemente de quem apresentou o slide*.
- [ ] Levar o PDF do TCC aberto num segundo dispositivo para consultar número exato se travar.
