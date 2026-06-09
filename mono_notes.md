# Notas de Revisão — The Humanizer Acadêmico

> Sessões de revisão com pipeline completo. Flags anotados por seção para heatmap final.

---

## Sessão 1 — Resumo
**Arquivo:** `main.tex` (linhas 44–84)
**Nível de risco:** 🔴 Alto | **Modo:** Pipeline completo
**Registro detectado:** Impessoal ABNT (≈8 marcadores impessoais vs 1 plural)

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 2/10 | Texto técnico e específico; sem vocabulário AI óbvio |
| Rigor Acadêmico | 8/10 | Claims ancorados em métricas (F1, σ, d); RQs como perguntas reduzem ligeiramente |
| Objetividade | 9/10 | Sem framing emocional; achados bilaterais apresentados diretamente |
| Consistência de Registro | 7/10 | Um slip: "Realizamos" quebra o impessoal do restante |

### Flags — 2 total: 0 🔴 / 2 🟡 / 0 🟢

**[A — Registro] — 🟡 Médio**
> "Realizamos ablation sistemática em três *datasets*"

Diagnóstico: única ocorrência de primeira pessoa do plural em texto cujo restante usa exclusivamente impessoal ABNT ("é identificada", "são validados", "este trabalho investiga"). Não é inconsistência sistêmica — é um slip pontual de registro.
Correção sugerida: "Conduziu-se ablation sistemática em três *datasets*" ou "Foi realizada ablation sistemática em três *datasets*"

---

**[D2 — Pergunta retórica como gancho] — 🟡 Médio**
> "Três perguntas de pesquisa orientam o trabalho: (RQ1) topologia carrega sinal discriminante acima do texto da raiz da árvore de propagação? (RQ2) sob quais condições estruturais? (RQ3) é viável detectar fake news *apenas* pela topologia?"

Diagnóstico: as três RQs são formuladas como interrogativas no Resumo. Em prosa ABNT, as questões de pesquisa são declarativas — o Resumo não é o lugar para suspense retórico; é onde o leitor encontra o escopo do trabalho enunciado diretamente.
Correção sugerida: "Três questões de pesquisa orientam o trabalho: (RQ1) se a topologia carrega sinal discriminante acima do texto da raiz da árvore de propagação; (RQ2) sob quais condições estruturais isso ocorre; (RQ3) se é viável detectar fake news apenas pela topologia."

### Top 3 Mudanças de Maior Impacto
1. Converter RQs para forma declarativa (D2) — elimina o único padrão de pergunta retórica no Resumo
2. Substituir "Realizamos" por construção impessoal (Registro) — consistência total do registro

### Notas do Autor
*(espaço para observações após revisão)*

---

## Sessão 2 — Introdução
**Arquivo:** `capitulos/01_introducao.tex`
**Nível de risco:** 🔴 Alto | **Modo:** Pipeline completo
**Registro detectado:** Inconsistente — primeiras 300 palavras em Impessoal ABNT; seção Contribuições em 1ª pessoa do plural; slip em RQ2

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 3/10 | Texto técnico e específico; poucos marcadores AI; inconsistência de registro é o maior problema |
| Rigor Acadêmico | 8/10 | Claims ancorados em resultados específicos (F1, d) e citações relevantes |
| Objetividade | 7/10 | "não um acessório, mas uma necessidade" e "menos respondida do que parece" introduzem framing retórico pontual |
| Consistência de Registro | 5/10 | Alternância sistemática: 3 seções impessoal × 1 seção plural completa + slips pontuais |

### Flags — 9 total: 1 🔴 / 7 🟡 / 1 🟢

**[Registro — Inconsistência sistêmica] — 🔴 Alto**
O capítulo alterna entre Impessoal ABNT (Contextualização, Motivação, Estrutura do Trabalho) e Primeira Pessoa do Plural (Contribuições inteira + "Operacionalizamos" em RQ2). A alternância não segue padrão legítimo — "Apresentamos" como navegação seria aceitável, mas o padrão inclui "Demonstramos, com prova matemática", "Mostramos, em protocolo", "Caracterizamos numericamente", que são observações de autoria, não meramente navegação estrutural.
Decisão necessária do autor: escolher um registro único para todo o documento e aplicar uniformemente.

---

**[B — Negação-contraste decorativa] — 🟡 Médio**
> "Modelos computacionais de detecção tornam-se, portanto, não um acessório, mas uma necessidade."

Diagnóstico: construção retórica de negação-contraste que visa gerar impacto antes de uma afirmação que deveria ser declarativa direta.
Correção: "Modelos computacionais de detecção tornam-se, portanto, um requisito técnico para o problema."

---

**[A — Registro / plural isolado] — 🟡 Médio**
> "Operacionalizamos essa pergunta através do tamanho de efeito de Cohen"

Diagnóstico: única ocorrência de plural no corpo das RQs, cujo entorno é impessoal.
Correção: "Essa pergunta é operacionalizada através do tamanho de efeito de Cohen"

---

**[A — Registro / plural agrupado — Contribuições] — 🟡 Médio**
> "Apresentamos quatro contribuições" / "Demonstramos, com prova matemática" / "Mostramos, em protocolo de k-fold" / "Caracterizamos numericamente" / "derivamos, a partir dessa caracterização"

Diagnóstico: 5 instâncias de 1ª pessoa do plural na seção Contribuições (linhas 178–215), em documento predominantemente impessoal. Grupo coerente internamente, mas incompatível com o registro do restante do capítulo.
Correção (se mantido impessoal): "São apresentadas quatro contribuições" / "Demonstra-se, com prova matemática" / "O protocolo de k-fold mostra que" / etc.

---

**[D2 — Framing retórico] — 🟡 Médio**
> "Sob quais condições essa premissa se sustenta é uma pergunta menos respondida do que parece."

Diagnóstico: "menos respondida do que parece" é retórico — implica que o leitor subestimou a abertura. Abre com forma interrogativa transformada em declarativa, mantendo a cadência de gancho.
Correção: "A literatura topológica não fornece critérios mensuráveis para avaliar quando empregar arquiteturas topológicas, centrando-se predominantemente em resultados positivos em *datasets* específicos; os casos de falha do paradigma são reportados parcialmente."

---

**[A — Hedge performativo] — 🟡 Médio**
> "É pertinente registrar o que este trabalho não propõe."

Diagnóstico: segue o padrão de "vale ressaltar que" / "cabe destacar que" — introduz ênfase retórica onde a afirmação deveria ser direta.
Correção: "Este trabalho não propõe nova arquitetura GNN, nem *benchmark* novo. A contribuição primária é diagnóstica: [...]"

---

**[D2 — Perguntas retóricas nas RQs] — 🟡 Médio**
> RQ1 / RQ2 / RQ3 formuladas como interrogativas (linhas 155–172)

Diagnóstico: as três RQs são formuladas como interrogativas. Em prosa ABNT, questões de pesquisa são declarativas. Mesmo padrão identificado no Resumo — persiste na Introdução.
Correção: converter para declarativo com *se*: "RQ1: se a topologia carrega sinal discriminante acima do que a feature textual fornece; RQ2: sob quais condições estruturais o paradigma é efetivo; RQ3: se é viável detectar fake news apenas pela topologia."

---

**[C — Travessão / elaborativo] — 🟡 Médio**
> "uma regra de combinação com pesos assimétricos --- registrada explicitamente como heurística post-hoc, não como decisão metodológica a priori"

Diagnóstico: travessão escondendo relação elaborativa/restritiva; vírgula é suficiente.
Correção: "uma regra de combinação com pesos assimétricos, registrada explicitamente como heurística post-hoc e não como decisão metodológica a priori"

---

**[C — Travessão / aposto longo] — 🟢 Baixo**
> "sustenta a tese central do trabalho --- modelos topológicos exigem diferença estrutural entre classes da ordem de |d| ≥ 0,5 em pelo menos uma métrica de cascata para serem efetivos --- formulada de maneira deliberadamente falsificável."

Diagnóstico: uso de aposto com travessão é legítimo; a frase é longa demais, criando ambiguidade sobre o escopo de "formulada". Funciona, mas quebrar em duas melhora a legibilidade.
Sugestão: "sustenta a tese central do trabalho: modelos topológicos exigem diferença estrutural entre classes da ordem de |d| ≥ 0,5 em pelo menos uma métrica de cascata para serem efetivos. Essa tese é formulada de maneira deliberadamente falsificável."

### Top 3 Mudanças de Maior Impacto
1. Resolver inconsistência de registro (🔴): escolher impessoal ou plural para o documento inteiro — afeta toda a seção Contribuições + slips pontuais
2. Converter RQs para forma declarativa (🟡 D2) — corrige padrão recorrente identificado também no Resumo
3. Eliminar negação-contraste + hedge performativo (🟡) — dois pontos retóricos de remoção trivial

### Notas do Autor
*(espaço para observações após revisão)*

---

## Sessão 3 — Fundamentação Teórica
**Arquivo:** `capitulos/02_fundamentacao.tex`
**Nível de risco:** 🔴 Alto | **Modo:** Pipeline completo
**Registro detectado:** Inconsistente — texto predominantemente impessoal com 13 instâncias de plural dispersas; padrão idêntico à Introdução

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 2/10 | Capítulo altamente técnico; vocabulário AI praticamente ausente |
| Rigor Acadêmico | 9/10 | Excelente ancoragem em citações; prova matemática bem estruturada; uma imprecisão em "transcende o idioma" |
| Objetividade | 8/10 | Texto seco e preciso; sem hype em adjetivos técnicos |
| Consistência de Registro | 4/10 | 13 instâncias de plural ao longo do capítulo inteiro no mesmo padrão sistêmico da Introdução |

### Flags — 6 total: 1 🔴 / 3 🟡 / 2 🟢

**[Registro — Inconsistência sistêmica] — 🔴 Alto**
13 instâncias de 1ª pessoa do plural ao longo do capítulo:
- l. 23: "adotamos como foco a categoria de fabricação"
- l. 68: "que formalizamos por meio do critério mensurável"
- l. 238: "Como discutiremos em §X"
- l. 240: "no nosso domínio está em estatísticas globais"
- l. 309: "Neste trabalho utilizamos K = 4 cabeças"
- l. 441: "Adotamos três codificações posicionais nodais"
- l. 513: "Adotamos neste trabalho o GNNExplainer"
- l. 554: "Adotamos quatro instrumentos complementares"
- l. 617: "Adotamos n_boot = 2.000 em todos os experimentos"
- l. 675: "A nossa configuração foge da literatura"
- l. 677: "utilizamos a feature content"
- l. 679-680: "adotamos hiperparâmetros próprios" / "reportamos F1-macro"
- l. 787: "não reimplementamos o BiGCN"

Diagnóstico: todas são decisões de design ou navegação metodológica — padrão LEGÍTIMO em plural mode, mas flags em impessoal mode. Mesma decisão estrutural da Introdução; resolver uma vez para todo o documento.

---

**[C + D2 — Travessão + interrogativa embutida] — 🟡 Médio**
> "O diagnóstico de complementaridade abre uma questão operacional --- *quando* cada componente do híbrido carrega sinal informativo? --- que formalizamos por meio do critério mensurável"

Diagnóstico: travessão esconde relação explicativa; interrogativa embutida segue padrão D2.
Correção: "O diagnóstico de complementaridade levanta a questão de quando cada componente do híbrido carrega sinal informativo, questão formalizada por meio do critério mensurável apresentado no Capítulo X."

---

**[C + D2 — Travessão + interrogativas embutidas] — 🟡 Médio**
> "A pergunta de pesquisa central deste trabalho --- *sob quais condições a topologia da propagação detecta fake news?* --- remete a uma questão clássica em teoria de redes: quando é que classes de vértices em um grafo se diferenciam por estrutura?"

Diagnóstico: dois D2 em sequência — travessão com RQ como interrogativa + segunda interrogativa retórica. RQ reformulada como interrogativa pela 3ª vez (após Resumo e Introdução).
Correção: "A pergunta de pesquisa central deste trabalho — se a topologia da propagação detecta fake news e sob quais condições — remete a uma questão clássica em teoria de redes sobre quando classes de vértices em um grafo se diferenciam por estrutura."

---

**[Step 2 — Precisão Argumentativa] — 🟡 Médio**
> "É, antes, evidência de que o sinal estrutural *transcende* o idioma, propriedade desejável para métodos híbridos futuros."

Diagnóstico: "transcende" generaliza além do domínio testado. A evidência (d = −0,057 em snapshot Bluesky sem ground truth) suporta equivalência de distribuições, não transcendência geral.
Correção: "É, antes, evidência de que o sinal estrutural não é degradado por fronteiras linguísticas no contexto estudado, propriedade relevante para métodos híbridos que operem sobre dados multilíngues."

---

**[B — Fraseado de literatura crescente] — 🟢 Baixo**
> "A literatura tem produzido uma família crescente de métodos específicos para GNNs"

Diagnóstico: próximo do padrão AI "há um crescente corpo de literatura sobre"; citação imediata ancora a afirmação. Substituição opcional.
Sugestão: "Yuan et al. sintetizaram em revisão taxonômica a família de métodos de explicabilidade específicos para GNNs"

---

**[Exemplo positivo — Estrutura bem construída] — 🟢**
> "propõe como contribuição primária um *diagnóstico de aplicabilidade*: um critério mensurável, diferença estrutural entre classes da ordem de |d| ≥ 0,5 em pelo menos uma métrica de cascata, que decide, *antes do treinamento*, se há chance de o paradigma topológico funcionar em um domínio dado."

Diagnóstico: dois-pontos + vírgulas para aposto longo — uso correto. Modelo a seguir para outras instâncias.

### Top 3 Mudanças de Maior Impacto
1. Resolver inconsistência de registro (🔴): 13 instâncias neste capítulo — mesma decisão do documento inteiro
2. Dois travessões com interrogativas embutidas (🟡): substituir por dois-pontos + declarativa
3. "Sinal estrutural transcende o idioma" (🟡): recalibrar para escopo real da evidência

### Notas do Autor
*(espaço para observações após revisão)*

---

## Sessão 4 — Metodologia
**Arquivo:** `capitulos/03_metodologia.tex`
**Nível de risco:** 🟡 Médio | **Modo:** Scan parcial (vocabulário AI + travessão + adjetivos hype)
**Registro detectado:** Inconsistente — body impessoal com 9 instâncias de plural; padrão idêntico às sessões anteriores

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 2/10 | Capítulo muito técnico; vocabulário AI ausente |
| Rigor Acadêmico | 9/10 | Escolhas metodológicas bem justificadas com citações; caveat de reprodutibilidade exemplar |
| Objetividade | 9/10 | Texto seco e preciso; sem drama |
| Consistência de Registro | 5/10 | 9 instâncias de plural (ll. 169, 211, 303, 308, 352, 359, 422, 451, 453) |

### Flags — 3 total: 1 🔴 / 0 🟡 / 2 🟢

**[Registro — Inconsistência sistêmica] — 🔴 Alto**
9 instâncias de 1ª pessoa do plural em documento predominantemente impessoal:
- l. 169: "que usamos para comparar dois modelos"
- l. 211: "utilizamos a variante content"
- l. 303: "usamos F1-macro para alinhar com práticas mais recentes"
- l. 308: "utilizamos o teste t pareado"
- l. 352: "adotamos três técnicas complementares"
- l. 359: "Implementamo-la via rolling mean"
- l. 422: "combinamos o classificador textual"
- l. 451: "conduzimos varredura explícita sobre 10 seeds"
- l. 453: "reportamos média e desvio-padrão"

Diagnóstico: mesmo padrão das sessões anteriores — todas as instâncias são decisões de design ou navegação metodológica, legítimas em plural mode, flags em impessoal mode.

---

**[Vocabulário — "amplamente"] — 🟢 Baixo**
> "regime amplamente reportado na literatura \cite{barabasi1999emergence, clauset2009powerlaw}"

Diagnóstico: "amplamente" está na lista de AI vocab como vago pluralizador, mas aqui as citações imediatas (Barabási, Clauset) ancoram a afirmação. Aceito como legítimo; substituição opcional.
Sugestão se quiser eliminar: "regime documentado por Barabási e Albert e Clauset et al. como característico de redes sociais"

---

**[Vocabulário — verbo informal] — 🟢 Baixo**
> "Esse modelo encarna o confound trivial da detecção de fake news em redes"

Diagnóstico: "encarna" é levemente metafórico/informal para prosa técnica; não compromete o rigor, mas substituto mais preciso existe.
Sugestão: "Esse modelo representa o confound trivial da detecção de fake news em redes"

### Nota positiva
O capítulo usa corretamente **dois-pontos e vírgulas** para apostos em toda a extensão — nenhum travessão mal usado. Padrão de pontuação a seguir nos outros capítulos.

### Top 3 Mudanças de Maior Impacto
1. Resolver inconsistência de registro (🔴): 9 instâncias — mesma decisão do documento inteiro
2. "encarna" → "representa" (🟢) — substituição trivial
3. "amplamente reportado" → citar diretamente (🟢) — opcional

### Notas do Autor
*(espaço para observações após revisão)*

---

## Sessão 5 — Resultados Experimentais
**Arquivo:** `capitulos/04_resultados.tex`
**Nível de risco:** 🟡 Médio | **Modo:** Scan parcial (dramatic-reveal, adjetivos emocionais em métricas, stat-bomb)
**Registro detectado:** Inconsistente — body impessoal com 16 instâncias de plural; padrão idêntico às sessões anteriores

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 3/10 | Alguns hedges e um adjetivo emocional; predominantemente técnico |
| Rigor Acadêmico | 9/10 | Excelente: claims ancorados em métricas, ICs bootstrap, testes estatísticos; resultado negativo reportado com igual extensão |
| Objetividade | 7/10 | Dois D2 em openers de seção; "drástica" e "interessante" adicionam framing editorial |
| Consistência de Registro | 4/10 | 16 instâncias de plural — máximo entre todas as sessões até agora |

### Flags — 9 total: 1 🔴 / 4 🟡 / 4 🟢

**[Registro — Inconsistência sistêmica] — 🔴 Alto**
16 instâncias de 1ª pessoa do plural — máximo entre os capítulos revisados:
ll. 53, 69, 73, 114, 115, 119, 122, 125, 141, 266, 390, 473, 594, 755, 827, 860
Padrão: decisões de design/navegação metodológica (mesma classificação das sessões anteriores).

---

**[D2 — Dramatic-reveal com interrogativa] — 🟡 Médio**
> "A pergunta operacional desta seção, e o achado mais central do trabalho, é a seguinte: *é possível treinar uma GNN para detectar fake news sem fornecer texto a ela, apenas com sinal estrutural sobre o grafo de propagação?*"

Diagnóstico: o achado mais importante do trabalho é introduzido como interrogativa dramática. O resultado já é conhecido pelo leitor do Resumo; o framing como "é a seguinte:" + pergunta cria suspense artificial antes de um achado que deveria ser declarado diretamente.
Correção: "Esta seção reporta o achado central do trabalho: o GraphSAGE treinado apenas com duas features estruturais por nó (sem nenhum texto) atinge F1-macro de 0,810 em UPFD-GossipCop, respondendo afirmativamente à parte (a) da RQ3."

---

**[D2 — Interrogativa como fechamento de parágrafo introdutório] — 🟡 Médio**
> "Os números deste trabalho só sustentam os achados sobre topologia se estiverem *calibrados* contra a literatura: sob diferenças conhecidas e declaradas de configuração, eles estão no envelope dos publicados?"

Diagnóstico: parágrafo de contextualização termina com interrogativa retórica. Em resultados, a questão de calibração deve ser declarada, não perguntada — o leitor está aqui para ler a resposta, não a pergunta.
Correção: "Esta seção verifica que os resultados deste trabalho, sob diferenças conhecidas e declaradas de configuração, estão no envelope dos valores publicados por Dou et al."

---

**[A — Hedge performativo] — 🟡 Médio**
> "Um padrão transversal merece registro: nas três bases, a métrica depth\_max concentra os menores tamanhos de efeito"

Diagnóstico: "merece registro" segue o padrão de "vale ressaltar que" — o ponto deve ser feito diretamente.
Correção: "Nas três bases, a métrica depth\_max concentra os menores tamanhos de efeito: [...]"

---

**[A — Hedge performativo] — 🟡 Médio**
> "Vale uma distinção terminológica: a observação de que 'UPFD-GossipCop oficial possui 30% de grafos com depth ≥ 3' contradiz a leitura simplificada de que 'UPFD é tudo estrela'."

Diagnóstico: "vale uma distinção" = "vale ressaltar que" — o ponto deveria ser feito diretamente.
Correção: "A observação de que o UPFD-GossipCop oficial possui 30% de grafos com depth ≥ 3 contradiz a leitura simplificada de que 'UPFD é tudo estrela'."

---

**[B — Adjetivo emocional em resultado] — 🟢 Baixo**
> "A diferença entre domínios é drástica: o UPFD-GossipCop apresenta d = +1,53 [...] O UPFD-PolitiFact, em contraste, não alcança nem mesmo um efeito médio"

Diagnóstico: "drástica" é adjetivo emocional aplicado a resultado quantitativo. Imediatamente ancorado pelos números (d=+1,53 vs d=0,25), o que mitiga o problema. Substituição opcional.
Sugestão: "A diferença entre domínios é de grande magnitude: [...]"

---

**[A — Advérbio AI quantificado] — 🟢 Baixo**
> "Ele está *substancialmente abaixo* de duas referências: do nível de chance (≈0,50) e da predição puramente majoritária (≈0,33)."

Diagnóstico: "substancialmente" está na lista de AI vocab. Quantificado imediatamente após, o que atenua o problema. Aceito como marginal.
Sugestão: "Ele está abaixo de duas referências: do nível de chance (≈0,50) e da predição puramente majoritária (≈0,33)."

---

**[E — Verbo informal em seção de resultados] — 🟢 Baixo**
> "A pergunta *interessante* a respeito do classificador textual e do topológico tomados em conjunto não é qual dos dois é melhor [...] mas *onde* os dois discordam"

Diagnóstico: "interessante" é avaliação editorial do autor sobre qual pergunta vale a pena; prosa técnica prescinde de julgamentos de interesse. A estrutura de negação-contraste ("não é [...] mas") também é borderline B.
Sugestão: "A análise complementar do classificador textual e do topológico tomados em conjunto foca não no desempenho individual de cada um, mas nos casos em que discordam [...]"

---

**[Nota positiva — Objetividade bilateral] — 🟢**
> "Resultados negativos (em particular o desempenho dos modelos topológicos no UPFD-PolitiFact) são reportados com a mesma extensão dos positivos: a refutação de uma hipótese tem peso evidencial equivalente à sua confirmação"

Diagnóstico: declaração explícita de simetria evidencial no primeiro parágrafo do capítulo. Padrão exemplar — elimina a principal categoria de D3 antes de ela aparecer.

### Top 3 Mudanças de Maior Impacto
1. Resolver inconsistência de registro (🔴): 16 instâncias — mais do que qualquer outro capítulo
2. Dois D2 em openers de seção (🟡): substituir interrogativa + "é a seguinte:" por declarativa direta
3. Dois hedges performativos (🟡): "merece registro" e "vale uma distinção" → cortar o hedge, fazer o ponto

### Notas do Autor
*(espaço para observações após revisão)*

---

## Sessão 6 — Discussão e Conclusão
**Arquivo:** `capitulos/06_conclusao.tex`
**Nível de risco:** 🔴 Alto | **Modo:** Pipeline completo
**Registro detectado:** Inconsistente — body impessoal com 7 instâncias de plural (ll. 57, 132, 136, 146, 163, 201, 247)

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 3/10 | Alguns construtos retóricos; texto principalmente limpo. Abertura exemplar em tom |
| Rigor Acadêmico | 8/10 | Excelente ancoragem em referências cruzadas; uma citação fantasma sobre o viés da literatura GNN |
| Objetividade | 7/10 | Título de seção informal, negação-contraste no parágrafo central, RQs como interrogativas persistem |
| Consistência de Registro | 5/10 | 7 instâncias de plural; menos do que capítulos anteriores |

### Flags — 9 total: 1 🔴 / 5 🟡 / 3 🟢

**[Registro — Inconsistência sistêmica] — 🔴 Alto**
7 instâncias de 1ª pessoa do plural:
- l. 57: "que formulamos de modo deliberadamente falsificável"
- l. 132: "Reconhecemos sete limitações do trabalho"
- l. 136: "grafo construído por nós"
- l. 146: "FakeNewsNet que construímos"
- l. 163: "nossa publicação do dataset processado"
- l. 201: "identificamos quatro direções de continuidade"
- l. 247: "é a contribuição metodológica que pretendemos seja reaplicável"

---

**[D2 — RQs como interrogativas nos headers] — 🟡 Médio**
> `\paragraph{RQ1: a topologia carrega sinal acima do que o texto da raiz já fornece?}`
> `\paragraph{RQ3: é viável detectar fake news apenas pela topologia, em cenários sem texto exploitável?}`

Diagnóstico: as RQs são reformuladas como interrogativas pela 4ª vez no documento (após Resumo, Introdução e Fundamentação). Em ABNT, questões de pesquisa são declarativas.
Correção: converter para `\paragraph{RQ1: resultado quanto ao sinal topológico acima do textual}` ou declarativo com *se*.

---

**[D2 — Sub-questões interrogativas embutidas na RQ3] — 🟡 Médio**
> "A resposta se divide nas duas partes embutidas na pergunta. Quanto à parte (a), *viável sem texto algum?*, a Tabela X mostra que sim [...]. Quanto à parte (b), *indiferente ao idioma do post?*, a §X [...]"

Diagnóstico: as sub-questões "(a) viável sem texto algum?" e "(b) indiferente ao idioma do post?" são apostos interrogativos embutidos em texto declarativo. Em prosa ABNT, a parte (a) e parte (b) são declaradas como condições de teste, não como perguntas abertas.
Correção: "Quanto à parte (a) — viabilidade sem texto — a Tabela X confirma que sim, [...]" ou reestruturar como declarativo.

---

**[E/B — Título de seção informal] — 🟡 Médio**
> `\section{Achado negativo é achado}`

Diagnóstico: título de seção com construção slogã / repetição anafórica ("achado [...] é achado"). Em ABNT, títulos de seção são descritivos e neutros. O conteúdo da seção é excelente; o título não combina com o tom acadêmico.
Sugestão: `\section{O resultado negativo de UPFD-PolitiFact como evidência da tese central}` ou `\section{Interpretação do resultado negativo}`

---

**[B — Negação-contraste retórica] — 🟡 Médio**
> "Reportar este achado com a mesma extensão dos achados positivos não é formalismo: é exigência da tese central."

Diagnóstico: "não é X: é Y" é construção de negação-contraste que visa impacto retórico onde uma afirmação declarativa seria mais precisa.
Correção: "Reportar este achado com a mesma extensão dos achados positivos é exigência metodológica da tese central, que só é falsificável se os casos negativos forem apresentados com igual rigor."

---

**[Step 2 — Citação fantasma] — 🟡 Médio**
> "A literatura de detecção de fake news com GNNs tende a reportar predominantemente resultados positivos"

Diagnóstico: afirmação sobre o viés da literatura sem citação específica. Os surveys citados na Fundamentação (§1.2) encontraram esse padrão — mas aqui a afirmação é feita sem referência cruzada.
Correção: adicionar `\cite{pmc_gnnsurveyfakenews, pmc_overviewfakenews}` ou referência cruzada à §\ref{sec:surveys}.

---

**[B — Negação-contraste substantiva] — 🟢 Baixo**
> "Uma teoria que afirma 'topologia funciona sob a condição X' é fortalecida, não enfraquecida, pela demonstração de que ela não funciona quando X falha."

Diagnóstico: contraste "fortalecida/enfraquecida" é substantivo — articula a lógica falsificacionista de Popper. Não é retórica decorativa; é argumento filosófico. Aceito. Anotar como exemplo de negação-contraste legítima.

---

**[A — Filler de abertura de seção] — 🟢 Baixo**
> "A partir dos resultados obtidos, identificamos quatro direções de continuidade."

Diagnóstico: "A partir dos resultados obtidos" é adjacente ao padrão "diante do exposto", mas aponta especificamente para os resultados do capítulo anterior. Borderline aceitável.
Sugestão: "Os resultados apresentados sugerem quatro direções de continuidade."

---

**[A — Claim ligeiramente inflado] — 🟢 Baixo**
> "é a contribuição metodológica que pretendemos seja reaplicável para além deste dataset específico"

Diagnóstico: "para além de" é levemente inflado; "pretendemos" + "reaplicável" é modesto e calibrado, mas a frase poderia ser mais direta.
Sugestão: "é a contribuição metodológica principal do trabalho, aplicável a outros domínios onde o critério de Cohen's d possa ser mensurado previamente ao treinamento."

### Nota positiva — Trabalhos Futuros exemplares
As quatro direções de continuidade (ll. 203-231) são concretas, com método, condição e escopo específicos. Nenhuma instância de engagement bait ("abre vastas perspectivas", "imensas implicações para a área"). Modelo a seguir.

### Nota positiva — Considerações Finais sem D3
O fechamento do trabalho (ll. 237-252) é calibrado: a contribuição é descrita de forma específica ("diagnóstico de aplicabilidade [...] reaplicável para além deste dataset"), e o código/pesos publicados são apresentados como mecanismo de falsificação, não como impacto grandiose.

### Top 3 Mudanças de Maior Impacto
1. Título "Achado negativo é achado" (🟡): substituir por título descritivo em estilo ABNT
2. RQs como interrogativas (🟡 D2): converter headers de RQ para declarativo — corrige padrão recorrente pela última vez
3. Citação fantasma na literatura (🟡): adicionar referência cruzada ao survey citado na Fundamentação

### Notas do Autor
*(espaço para observações após revisão)*

---

## Sessão 7 — Apêndice E: Aplicação Demonstrativa (Bluesky)
**Arquivo:** `apendices/E_aplicacao_bluesky.tex`
**Nível de risco:** 🟢 Baixo | **Modo:** Scan mínimo (vocabulário AI explícito + travessão; registro plural aceito)
**Registro:** Plural ("Utilizamos", "Aplicamos", "comparamos") — aceito per skill para seção descritiva de ferramenta

### Pontuação
| Dimensão | Nota | Justificativa |
|---|---|---|
| Similaridade-AI | 2/10 | Um advérbio borderline; texto técnico descritivo |
| Rigor Acadêmico | 9/10 | Limitações explicitadas na abertura e no fechamento; scope bem delimitado |
| Objetividade | 9/10 | Tom direto e descritivo; sem hype |
| Consistência de Registro | 8/10 | Plural aceito nesta seção (registro ligeiramente relaxado per skill) |

### Flags — 2 total: 0 🔴 / 0 🟡 / 2 🟢

**[A — Advérbio sem medida] — 🟢 Baixo**
> "consideravelmente abaixo do pico anterior"

Diagnóstico: "consideravelmente" está na lista de AI vocab. O valor 55,9% é citado, e o "pico anterior" de 74% foi mencionado na frase anterior, mas a diferença (18 pontos percentuais) não é expressa explicitamente.
Sugestão: "cai para 55,9% — 18 pontos abaixo do pico da faixa anterior (74,0%)"

---

**[C — Travessão / sequencial] — 🟢 Baixo**
> "A ferramenta foi construída como prova de conceito de *deploy*, não como produto auditado para uso em moderação real --- decisões sobre conteúdo a partir desses *scores* devem respeitar as limitações documentadas"

Diagnóstico: o travessão separa dois enunciados independentes (disclaimer + instrução). Substituível por ponto final.
Correção: "A ferramenta foi construída como prova de conceito de *deploy*, não como produto auditado para uso em moderação real. Decisões sobre conteúdo a partir desses *scores* devem respeitar as limitações documentadas no Capítulo X [...]"

### Nota positiva
Apêndice delimita o escopo demonstrativo já na frase de abertura ("É matéria demonstrativa, não validatória") e repete as limitações no fechamento. Padrão de autocontenção exemplar.

### Notas do Autor
*(espaço para observações após revisão)*

---

## HEATMAP MACRO — Todas as Sessões

| Seção | Sim-AI | Rigor | Objetiv. | Consist. Reg. | 🔴 | 🟡 | 🟢 | Total |
|---|---|---|---|---|---|---|---|---|
| S1 Resumo | 2/10 | 8/10 | 9/10 | 7/10 | 0 | 2 | 0 | 2 |
| S2 Introdução | 3/10 | 8/10 | 7/10 | 5/10 | 1 | 7 | 1 | 9 |
| S3 Fundamentação | 2/10 | 9/10 | 8/10 | 4/10 | 1 | 3 | 2 | 6 |
| S4 Metodologia | 2/10 | 9/10 | 9/10 | 5/10 | 1 | 0 | 2 | 3 |
| S5 Resultados | 3/10 | 9/10 | 7/10 | 4/10 | 1 | 4 | 4 | 9 |
| S6 Conclusão | 3/10 | 8/10 | 7/10 | 5/10 | 1 | 5 | 3 | 9 |
| S7 Apêndice E | 2/10 | 9/10 | 9/10 | 8/10 | 0 | 0 | 2 | 2 |

### Ranking de Prioridade para Revisão (sessões por impacto)

**1. 🔴 Introdução (S2) — 9 flags, Consistência 5/10, Objetividade 7/10**
Maior contagem de flags do documento. Concentra: inconsistência de registro 🔴, 5 instâncias de plural agrupadas nas Contribuições, 2 D2, negação-contraste, hedge performativo. A decisão de registro (impessoal vs. plural) precisa ser feita aqui e propagada para todos os outros capítulos.

**2. 🔴 Resultados Experimentais (S5) — 9 flags, Consistência 4/10, maior volume de plural (16)**
Apesar de 0 🔴 além do registro, tem o maior volume de plural do documento (16 instâncias) e dois D2 em openers de seção de alto impacto (incluindo a seção do achado central). Dois hedges performativos de remoção trivial.

**3. 🔴 Discussão e Conclusão (S6) — 9 flags, Objetividade 7/10**
Alto risco de D3 — que não se concretizou, mas tem: título de seção slogã, RQs como interrogativas pela 4ª vez, negação-contraste, citação fantasma. Menor volume de plural (7), mas o título "Achado negativo é achado" é a mudança de maior visibilidade.

**4. 🔴 Fundamentação Teórica (S3) — 6 flags, Consistência 4/10**
Menos flags, mas Consistência de Registro 4/10 com 13 instâncias de plural. Dois travessões com interrogativas embutidas e um claim de precisão ("transcende o idioma").

**5. 🔴 Metodologia (S4) — 3 flags, nota mais alta do documento**
Capítulo mais limpo. Apenas registro (9 instâncias). Dois itens 🟢 triviais.

**6. Resumo (S1) — 2 flags, limpo**
Dois 🟡 simples: "Realizamos" e RQs interrogativas. Resolução imediata.

**7. Apêndice E (S7) — 2 flags, fora de escopo de registro**
Dois 🟢 triviais. Apêndice demonstrativo.

### Padrão Recorrente Dominante no Documento
**Inconsistência de registro** foi encontrada em 5 das 6 seções principais (ausente apenas no Apêndice E, onde plural é aceito). As RQs formuladas como interrogativas aparecem em 5 seções (Resumo, Introdução, Fundamentação, e Conclusão). Resolver os dois padrões uma única vez no documento eliminaria a maior parte dos 🔴 e vários 🟡.

### Decisão Estrutural Urgente (pré-revisão)
Antes de qualquer edição textual: **escolher o modo de registro do documento inteiro** — Impessoal ABNT ou Primeira Pessoa do Plural. Esta única decisão resolve ~51 instâncias de plural catalogadas nas 6 sessões.

---
