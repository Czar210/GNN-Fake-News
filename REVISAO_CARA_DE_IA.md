# Revisão "cara de IA" — itens pendentes

> Gerado por um workflow (achar → verificar adversarial) em 2026-05-28, varrendo
> os capítulos com prosa (3, 4, 6 + Apêndice E). De 25 achados confirmados,
> **11 já foram aplicados direto nos `.tex`** (ALTA/MÉDIA sem ressalva).
> Este arquivo guarda os **14 restantes** — os de severidade baixa e os que
> precisam do seu olho antes de colar.

> **Status (2026-06-03):** Cesar pediu o lote de 8 recomendados — **aplicados nos `.tex`** os itens **1, 3, 4, 7, 10, 12, 13, 14** com as ressalvas do verificador. Pulados/mantidos no original: **2, 6, 8/9, 11**. Já obsoletos antes do lote: **5** (a prosa atual de §4.8 já tem $3.826$ ancorado em FATOS) e parte do **14** sobre inserção de $88{,}7\%$ (idem). Detalhes por item abaixo.

**Como ler:** os trechos estão com acento normal para leitura. Ao colar no
`.tex`, use o padrão de acento do arquivo (`\~ao`, `\'e`, `\c{c}`). Comandos
LaTeX (`\ref`, `\emph`, `\S`, `$...$`) estão preservados.

**Legenda de status:**
- 🟢 **pendente** — reescrita segura, só não apliquei por ser severidade baixa.
- ⚠️ **revisar antes** — a sugestão tem ressalva (número inserido, gramática, ou troca de sentido). NÃO colar no automático.

---

## ✅ Já aplicados nos `.tex` (referência)

- **Cap 3** `sec:dual`: removido "É importante registrar, já aqui…"; tricolon e cross-ref órfão reescritos (harmonizados num trecho só).
- **Cap 3** `sec:arquiteturas`: "A Tabela X reúne os hiperparâmetros." → versão que faz a referência trabalhar.
- **Cap 3** `sec:reprodutibilidade`: "…é tratada em três níveis." → abertura que já nomeia os três eixos.
- **Cap 4** `sec:multilingual`: "É necessário delimitar… antes de discutir…" → direto.
- **Cap 4** `sec:multilingual`: tricolon parentético (adoção tardia, uso discreto, autoselecionada) → dois mecanismos.
- **Cap 4** `sec:estrat_lowess`: "Esta subseção prepara a leitura das…" → anúncio direto.
- **Cap 4** `sec:estrat_outliers`: "A presente subseção confirma esse fenômeno…" → "Aqui esse efeito aparece quantificado…".
- **Cap 4** `sec:painel_mestre`: "consolida visualmente o argumento que percorre todo este capítulo" → o que a figura mostra.
- **Cap 6** RQ3: "A resposta também é bilateral e composta de duas partes." (pleonasmo) → enxuto.

---

## 📘 Cap 3 — Metodologia (3 pendentes)

### 🟢 1. Frase-rótulo — `sec:folds`
- ❌ "A motivação é estatística."
- ✅ "A razão para isso é estatística: comparar dois modelos par a par exige que ambos sejam avaliados sobre os mesmos exemplos."
- ⚠️ **Ressalva minha:** a frase *seguinte* no seu texto ("O teste *t* pareado… exige que cada par… tenha sido obtido sobre os *mesmos exemplos*") já diz isso. Se colar a sugestão inteira, vira redundância. Alternativa mais limpa: **só apagar** "A motivação é estatística." e deixar o parágrafo começar direto em "O teste *t* pareado…".

### 🟢 2. "não se trata de… mas de…" — `sec:construcao_fnn`
- ❌ "não se trata de remoção de *outliers* raros, mas de decisão de projeto com efeito substancial"
- ✅ "esse corte não é uma limpeza de *outliers* raros, e sim uma decisão de projeto que afeta metade dos dados."
- Obs.: "afeta metade dos dados" reafirma os 50% já ditos na frase anterior — talvez cortar essa cauda.

### 🟢 3. Generalização panorâmica — `sec:protocolo`
- ❌ "situação comum em detecção de fake news em produção, onde a classe *fake* pode ser minoritária"
- ✅ "por exemplo com a classe *fake* em minoria"

---

## 📗 Cap 4 — Resultados (4 pendentes)

### ⚠️ 4. "não apenas erra: ele está invertido" — `sec:concordancia`
- ❌ "o classificador topológico não apenas *erra*: ele está *sistematicamente invertido*."
- ✅ "o classificador topológico erra abaixo do que erraria uma predição aleatória: seus rótulos estão anticorrelacionados com a classe verdadeira, ou seja, sistematicamente invertidos."
- ⚠️ **Revisar:** (a) confirme que "abaixo do acaso" descreve mesmo o F1 desse subconjunto (você cita $F1=0{,}085$ logo adiante — bate). (b) o agente escreveu "**suas** rótulos" — corrija para "**seus** rótulos".

### ⚠️ 5. Reanúncio que INSERE NÚMERO — `sec:estratificacao` (abertura)
- ❌ "…esconde três comportamentos distintos do modelo, que a presente seção isola com três lentes complementares"
- ✅ "…é um número único que esconde três comportamentos distintos do modelo. Para separá-los, aplicamos três lentes complementares sobre os $3.826$ [grafos]"
- ⚠️ **NÃO COLAR sem conferir:** a reescrita **inseriu `$3.826$`**, número que não estava no trecho original. Isso viola a regra nº 1 do `PROMPT_IA_TCC.md` (nunca inventar número). Cheque contra o `FATOS_TCC.md` se 3.826 é o nº de grafos do *test* GossipCop; se não for, use a versão sem o número: "Para separá-los, aplicamos três lentes complementares."

### 🟢 6. "tem fundamentação teórica na literatura" — `sec:porque`
- ❌ "Este resultado tem fundamentação teórica na literatura de *homofilia* e *assortatividade* em redes."
- ✅ "A explicação para esse contraste passa pela noção de *assortatividade* por classe, e antes dela pela de *homofilia*, em redes."

### 🟢 7. "Registramos, por fim…" — `sec:estrat_cascatas`
- ❌ "Registramos, por fim, uma precisão terminológica:"
- ✅ "Vale uma distinção terminológica: a observação"

---

## 📕 Cap 6 — Conclusão (6 pendentes)

### ⚠️ 8 e 9. Abertura do capítulo — escolher UMA das duas reescritas
O parágrafo de abertura é meta-textual ("Este capítulo é o lugar onde se assume responsabilidade… afirmações sem essa âncora foram deliberadamente excluídas"). O workflow gerou **duas alternativas** para o mesmo parágrafo — não aplique as duas.

- ❌ **Original:** "Este capítulo é o lugar onde se assume responsabilidade pelo que os experimentos efetivamente mostraram. Cada afirmação quantitativa aqui apresentada é ancorada na tabela, figura ou seção que a sustenta; afirmações sem essa âncora foram deliberadamente excluídas."
- ✅ **Alternativa A (reescreve tudo):** "Este capítulo reúne e interpreta os resultados dos capítulos anteriores, organizando-os em torno das perguntas de pesquisa que guiaram o trabalho. As conclusões que seguem apoiam-se nas tabelas, figuras e seções correspondentes, citadas ao longo do texto."
- ✅ **Alternativa B (mantém a 1ª frase):** "Este capítulo é o lugar onde se assume responsabilidade pelo que os experimentos efetivamente mostraram. As seções a seguir retomam cada pergunta de pesquisa, citam os resultados nas tabelas e figuras correspondentes e discutem o que eles permitem ou não permitem concluir."

### 🟢 10. Paralelismo "funciona quando… inverte exatamente onde…" — seção "Achado negativo"
- ❌ "ele funciona quando a regra ``*fake* = cascata maior'' vale, e inverte exatamente onde essa regra falha."
- ✅ "Esses subgrupos demarcam o *limite operacional* do mecanismo topológico: ele depende de que a regra ``*fake* = cascata maior'' valha, e erra justamente nos casos em que essa regra não se aplica."
- 💡 Bom argumento do verificador: "inverte **exatamente**" era impreciso (o subgrupo `real_viral` acerta ~43%, não é inversão perfeita).

### 🟢 11. "e não apenas mais uma demonstração" — seção "Achado negativo"
- ❌ "este trabalho oferece o diagnóstico de aplicabilidade, e não apenas mais uma demonstração"
- ✅ "ao trazer explicitamente o caso negativo, este trabalho propõe como contribuição primária um diagnóstico de aplicabilidade do mecanismo topológico."

### ⚠️ 12. "esperamos transcender o escopo" — Considerações finais
- ❌ "é a contribuição metodológica que esperamos transcender o escopo do *dataset* específico."
- ✅ (do workflow) "é a contribuição metodológica que pretendemos reaplicável para além deste *dataset* específico."
- ⚠️ **Revisar:** a sugestão do workflow ainda saiu meio torta ("que pretendemos reaplicável"). Eu colaria assim: **"é a contribuição metodológica que pretendemos seja reaplicável para além deste *dataset* específico."**

### 🟢 13. "Quatro direções naturais se desdobram" — Trabalhos futuros
- ❌ "Quatro direções naturais se desdobram dos achados deste trabalho."
- ✅ "A partir dos resultados obtidos, identificamos quatro direções de continuidade."

---

## 📙 Apêndice E — Bluesky (1 pendente)

### ⚠️ 14. Generaliza de 1 feed + insere número — seção "Concordância textual × topológico"
- ❌ "o que sugere que comunidades politicamente engajadas geram sinais textuais e estruturais mais alinhados."
- ✅ (do workflow) "O *feed* *Political Science* atinge $88{,}7\%$ de concordância agregada — o mais alto da amostra. Como é um único *feed* e a métrica é apenas concordância entre os dois classificadores (sem *ground-truth*), não é possível afirmar a causa; uma hipótese a investigar é que conteúdo político produza texto e padrão de propagação mais consistentes entre si."
- ⚠️ **Revisar:** (a) a reescrita é bem mais longa (vira mini-parágrafo) — veja se cabe nas ~3 páginas do apêndice. (b) **inseriu `$88,7\%$`** — confirme o valor antes de colar. O *ganho* aqui é real e alinhado com seu tom ("honestidade acima de marketing"): rebaixa "sugere que comunidades…" a hipótese explícita e lembra a ausência de *ground-truth*.
