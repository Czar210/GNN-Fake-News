# Pendências e decisões em aberto — TCC GNN Fake News

> Toda mudança planejada mas ainda não executada vive aqui. Leia este arquivo
> ao retomar uma sessão de escrita — antes de propor algo novo, conferir se
> o "novo" não é na verdade item pendente que já tem decisão tomada.

Última atualização: 2026-05-25.

---

## ✅ Mudança estrutural EXECUTADA em 2026-05-25 — Cap 5 → Apêndice E

**Decisão (do Cesar):** o orientador disse que o TCC está "muito grande". A escolha foi pela **Opção 1**: mover todo o Cap 5 (Aplicação Bluesky + ferramenta web) para um **Apêndice E "Aplicação demonstrativa"** condensado em 3-4 páginas, em vez de cortar o conteúdo por completo.

**Status:** ✅ Executada em 2026-05-25. Arquivos resultantes:
- `apendices/E_aplicacao_bluesky.tex` (criado, ~3 páginas)
- `capitulos/05_aplicacao.tex.deprecated` (arquivo original arquivado, mantido para consulta caso precise reaproveitar conteúdo)
- `main.tex`: include do Cap 5 removido, include do Apêndice E adicionado, resumo/abstract reduzidos
- Cap 3 §3.10 (sec:dual), Cap 4 §4.7, Cap 6 §6.3(c)/§6.4(1)/§6.4(2)/§6.5: referências a `cap:aplicacao` → `ap:aplicacao` e a `sec:bsky_*`/`sec:web_arquitetura`/`sec:hf_hub` redirecionadas
- Cap 1 §1.4 e §1.5 scaffolding: contribuição (d) e mapa de capítulos referenciam o apêndice

**Por que essa opção foi escolhida (não Opção 2 nem 3):**
- A análise multilíngue (PT vs EN, Cohen's d = −0,057) é rigorosa e responde metade da RQ3 → preserva como §4.4-bis no Cap 4.
- A ferramenta web é a contribuição (d) do §1.4 → vira material curto no apêndice, não some.
- Apêndice não conta como "corpo principal" pro tamanho percebido pelo orientador.
- 12 figuras + 3 tabelas + 168k posts processados não viram desperdício.

### Itens em cascata (todos executados em 2026-05-25)

| Lugar | Ação | Status |
|---|---|---|
| `main.tex` | Removido `\include{capitulos/05_aplicacao}` | ✅ |
| `main.tex` (bloco apêndices) | Adicionado `\include{apendices/E_aplicacao_bluesky}` | ✅ |
| `apendices/E_aplicacao_bluesky.tex` | Criado (~3 páginas): dataset Failla, T8, F7, concordância textual×topológico (3.728 posts, D3 corrigido), regra dual + HF Hub | ✅ |
| `capitulos/04_resultados.tex` §4.4-bis | Mantido no Cap 4; transição já não dependia do Cap 5 | ✅ (sem mudança necessária) |
| `capitulos/04_resultados.tex` §4.7 | Ref `cap:aplicacao` → `ap:aplicacao` | ✅ |
| `capitulos/06_conclusao.tex` §6.3(c) | Reformulado: aponta para §4.4-bis e Apêndice E | ✅ |
| `capitulos/06_conclusao.tex` §6.4(1) | Ref `cap:aplicacao`→`ap:aplicacao` e `sec:bsky_dataset`→`sec:ap_bsky_dataset` | ✅ |
| `capitulos/06_conclusao.tex` §6.4(2) | Ref `sec:bsky_inferencia`→`sec:ap_bsky_inferencia` | ✅ |
| `capitulos/06_conclusao.tex` §6.5 | Refs `sec:hf_hub`/`sec:web_arquitetura` → `Apêndice~\ref{ap:aplicacao}` | ✅ |
| `capitulos/03_metodologia.tex` §3.10 (sec:dual) | Refs `cap:aplicacao`/`sec:web_arquitetura` → `ap:aplicacao` | ✅ |
| `main.tex` resumo + abstract | Frase Bluesky reduzida; ferramenta web movida para Apêndice E | ✅ |
| `capitulos/01_introducao.tex` §1.4 (contribuição d) | Scaffolding atualizado: aponta para Apêndice E | ✅ |
| `capitulos/01_introducao.tex` §1.5 (mapa) | Scaffolding atualizado: 2→3→4→6 + Apêndice E | ✅ |
| `capitulos/05_aplicacao.tex` | Renomeado para `05_aplicacao.tex.deprecated` (arquivado) | ✅ |
| Tabela síntese T14 | Permanece como está | (sem mudança) |
| Figuras `F5, F6, F8, F9, F10, F11, F13, F14, F15, F20, F23` | Não referenciadas no apêndice; apenas F7 entra (distribuição de scores por feed) | (limpeza opcional posterior) |
| Tabelas `T7, T9` | Não referenciadas no apêndice; T8 entra | (limpeza opcional posterior) |

**Tempo real:** ~40 min (uma única sessão Claude).

**Pendência residual:** verificar visualmente após compilar o PDF se o conjunto F5/F6/F8–F15/F20/F23 e T7/T9 podem ser deletados das pastas `Imagens/` e `Imagens/tabelas/` (não estão mais referenciados, mas vale guardar até a banca).

---

## 📝 Pendências de redação (ordem da escrita)

Status dos capítulos (após a rodada de hoje, 2026-05-25):

| Capítulo | Prosa | Próxima ação |
|---|---|---|
| Cap 1 (Introdução) | ❌ Vazio (5 seções com `% ESCREVER`) | Escrever por último, sabendo a história completa |
| Cap 2 (Fundamentação) | ❌ Vazio (8 seções com `% ESCREVER`) | Penúltimo na ordem |
| Cap 3 (Metodologia) | ✅ §§3.3–3.11 escritas; §§3.1 e 3.2 já estão polidas pelo Cesar | Limpar `% ESCREVER` antigos restantes (2 blocos) |
| Cap 4 (Resultados) | ✅ §§4.1–4.7 escritas; §4.8 (Fase 9) escrita; §4.9–4.10 ok | Limpar `% ESCREVER` restantes (4 blocos) |
| ~~Cap 5 (Aplicação)~~ | ✅ Migrado para `apendices/E_aplicacao_bluesky.tex` em 2026-05-25 |
| Cap 6 (Conclusão) | ✅ Reescrito 2x com fatos ancorados; pronto | Revisar quando RQ3 mudar (depois de mover Cap 5) |

---

## ✂️ Padrões a varrer preventivamente em todos os capítulos

Feedback do orientador identificou padrões. Aplicar varredura quando o Cesar pedir:

- **Meta-comentário sobre o próprio texto** ("em conformidade com a tese central", "como veremos a seguir", "este capítulo se propõe a"). Substituir pela afirmação substantiva direta.
- **"Antes de X é necessário Y"** — fórmula procedural de manual técnico, não prosa acadêmica. Reescrever afirmando o fenômeno diretamente.
- **"A §X retoma esse ponto..."** — frase órfã de "fica ligado pro próximo episódio". Integrar o cross-ref no meio do raciocínio (parêntese) ou cortar.
- **Justificativa da existência de tabela/figura** ("a Tabela X serve de âncora visual..."). Tabela boa não precisa de justificativa. Cortar.

Items do feedback do professor já aplicados: **[1], [2], [5], [7]**.
Items ainda não recebidos do Cesar: **[3], [4], [6]** (presumivelmente em fila).

---

## 📚 Pendências de bibliografia

Citações usadas no LaTeX mas **ainda ausentes** do `Referencias.bib`. Caçar na varredura final de bibliografia:

| Chave | Para que | Onde | Status |
|---|---|---|---|
| `cleveland1979lowess` | LOWESS | §3.8 | ✅ Adicionado em 2026-06-03 |
| `efron1993bootstrap` | Bootstrap pareado | §3.8 | ✅ Adicionado em 2026-06-03 |
| `sibila2025bluesky` | Artefatos publicados no HF Hub (não publicados ainda) | §3.2, §3.11, Apêndice E, §6.4 | ⏳ Pendente até publicar no HF Hub |

### Refs metodológicas adicionadas em 2026-06-03 (Tier 1+2+3 do escopo aprovado)

| Chave | Justifica | Onde aplicada |
|---|---|---|
| `vosoughi2018spread` | Propagação assimétrica fake×real | §1.1 (Contextualização) |
| `gharpure2020cdc` | Relatório CDC desinfetantes (39%) | §1.1 |
| `bakshy2015exposure` | Algoritmos amplificam interação | §1.1 |
| `delvicario2016spreading` | Echo chambers e viés de confirmação | §1.1 |
| `reimers2020multilingual` | Sentence-BERT multilingual | §3.3 |
| `kohavi1995cv` | k=10 folds | §3.4 |
| `barabasi1999emergence` | Heavy-tail/power-law em redes sociais | §3.3, §3.8, §4.8.1 |
| `clauset2009powerlaw` | Ajuste de power-law em dados empíricos | §3.3, §3.8, §4.8.1 |
| `li2018deeper` | Oversmoothing → 3 camadas | §3.5 |
| `xu2019gin` | Pooling agregador invariante | §3.5 |
| `srivastava2014dropout` | Dropout = 0.5 | §3.5 |
| `kingma2015adam` | Adam lr=1e-3 | §3.5 |
| `masters2018batch` | Batch pequeno (32) | §3.5 |
| `prechelt1998early` | Early stopping com patience | §3.5 |
| `sawilowsky2009effect` | Refinamento da escala Cohen | §3.7 |

**Importante:** `sibila2025bluesky` precisa ser criado quando o Cesar efetivamente publicar os artefatos no Hugging Face. Antes disso, vai dar *undefined reference* na compilação — pode marcar como `% TODO_CITE: sibila2025bluesky` se quiser que a compilação não pare.

---

## 📄 Descrições de figuras do Takida (`imagens_tcc.docx`, 2026-06-04)

O Takida produziu legendas + interpretação para **18 figuras** num docx. Avaliação completa foi feita em 2026-06-04. Status por figura:

### ✅ Já aproveitadas no Cap 4 (insights novos enxertados na prosa)

| Figura | Onde | Insight enxertado |
|---|---|---|
| F17 | §4.5 (`sec:porque`) | **Eixo horizontal vs vertical**: `depth_max` tem |d| ≈ 0 nos 3 datasets. Diferença é largura/branching, não profundidade |
| F19 | §4.7 (`sec:explainer`) | **No PolitiFact a assimetria classe-a-classe DESAPARECE**: hop1 dominante para fake E real — confirmação mecanicista do colapso |

### ⚠️ 4 erros do docx (NÃO colar literal sem corrigir)

1. **F14 multilíngue** — Takida escreve "topologia característica de **desinformação**" para o pico em 0,9. Bluesky não tem ground truth — viola a abertura obrigatória de §4.4-bis. **Corrigir para**: "score topológico elevado" (sem rotular como fake).
2. **F16 painel mestre** — Takida diz "FNN sem comparativo topológico disponível". **Errado**: o painel TEM as 3 barras topo puro (0,53/0,54/0,54). O que falta é "texto+topo" (esse não fizemos no FNN).
3. **F23 (modelo×feed)** — Takida cita "**5 mil postagens**". É **3.728** (D3 do FATOS). Discrepância já corrigida no Apêndice E.
4. **Todo material Cap 5 do Takida** — foi escrito antes da decisão Cap 5 → Apêndice E. Maior parte fica órfã (ver abaixo).

### 🗂️ Material Cap 5 órfão do Takida (descrições não incorporadas)

8 figuras descritas por Takida que **não cabem mais** no Apêndice E condensado. Decisão: arquivar como referência para o caso de reabrir o escopo do apêndice. Material em `Downloads/imagens_tcc.docx`, parágrafos correspondentes:

| Figura | Tema | Por que ficou órfã |
|---|---|---|
| F5 | Bluesky crossfeed (boxplots engajamento) | Apêndice E não tem análise crossfeed |
| F6 | Bluesky crossfeed (heatmap Cohen's d) | Idem |
| F20 | Bluesky crosstab (textual×topo) | Substituído por descrição curta no Apêndice E §sec:ap_bsky_concordancia |
| F9 | Agreement por tamanho de thread | Apêndice E só menciona em texto corrido |
| F10 | Agreement por feed | Idem (Political Science 88,7% mencionado) |
| F23 | Score médio modelo×feed | Apêndice E não tem; descrição usa N errado (5k vs 3,7k) |
| F13a/b | Threads ilustrativas | Apêndice E não inclui visualização de threads |
| F4 | Bloco vs mini | Apêndice E não discute essa decisão de engenharia |

**Se o orientador reabrir o escopo do Apêndice E**, todas essas descrições podem ser reaproveitadas — está tudo no docx do Takida em `Downloads/imagens_tcc.docx`. Antes de colar literal, **F23 precisa do número corrigido (5k → 3.728)** e F20 idem.

### ✅ Figuras descritas e já cobertas pela prosa do Cap 4

F18, F25, F26, F27, F28, F29, F30, F31, F32, F16, F3, F24 — Takida descreveu, mas a prosa atual do Cap 4 já cobre os mesmos pontos sem necessidade de retrabalho.

---

## 🔬 Discrepâncias D1–D6 do `FATOS_TCC.md`

| ID | Status |
|---|---|
| D1 — SAGE-PolitiFact F1 = 0,33 (não 0,55) | ✅ Corrigido em resumo + abstract + Cap 6 |
| D2 — Cohen's d PolitiFact \|d\| < 0,5 (não < 0,2) | ✅ Corrigido em resumo + abstract + Cap 6 |
| D3 — Amostra concordância Bluesky 3.728 (não 5k) | ✅ Corrigido no Apêndice E (texto da §sec:ap_bsky_concordancia usa 3.728) |
| D4 — UPFD oficial não é tudo estrela | ✅ Refletido em §3.3 + §4.8.4 + §6.3(a) |
| D5 — Citação `quelle2024bluesky` → `failla2024bluesky` | ✅ Corrigido em todos os `.tex` e `.bib` |
| D6 — Cap de 100 nós trunca 50% (não "outliers raros") | ✅ Refletido em §3.3 (verificação matemática) + §6.3(a) |

---

## 🖼️ Figuras planejadas mas não criadas

| Figura | Status | Notas |
|---|---|---|
| "Passagem de uma fake news pelo modelo" (user-journey) | ❌ Combinado fazer depois | Pedido do professor. Diagrama: post entra → BERT → grafo → GNN → score |
| Screenshots da ferramenta web | ❌ Não viável pela IA | Só Cesar pode capturar — relevante se Apêndice E for criado |
| Exemplo simples de fake news pra abrir Cap 1 | ❌ Não criado | Usar uma das threads do F13 com legenda explicativa, quando escrever Cap 1 |

---

## ❓ Confirmar com o orientador (próxima conversa do Cesar com ele)

- ~~A "matriz de confusão página 25 RF" que o professor mencionou é qual figura exatamente?~~ **✅ Confirmado em 2026-06-04: era as DUAS (F16 painel mestre + F20 cross-tab Bluesky). Ambas já foram repintadas em paleta semântica via script `27b`.**
- Se Apêndice E é aceitável pelo orientador (alternativa a cortar tudo do Bluesky). **⏳ Pendente — Cesar vai ver com ele.**

### Itens novos executados em 2026-06-04

- `sibila2025bluesky` no `.bib`: ✅ Criado apontando para o dataset publicado em https://huggingface.co/datasets/Zaras210/bluesky-fake-news-dataset (Cap 3 §3.11 e Apêndice E §sec:ap_ferramenta agora citam o **dataset** publicado, não os pesos do modelo — pesos ficam no repositório Git via `Execution/weights/`).
- Figura "Passagem de fake news pelo modelo" (Figura 1.1): ✅ Criada em TikZ no Cap 1 §1.2 (Motivação), mostrando o fluxo `post → BERT/features estruturais → LogReg/RF/SAGE → regra dual → score`. Versão Cap 1 é ilustrativa; a figura F0 do Cap 3 §3.1 segue como diagrama abstrato do pipeline.
- Trocar "Chapter/Appendix" para PT: ✅ Verificado. O babel `brazil` já renderiza tudo como "Capítulo X" e "Apêndice X" automaticamente (confirmado via `pdftotext`: zero ocorrências de "Chapter" no PDF compilado). Única ocorrência de "Appendix" sobra no abstract em inglês obrigatório bilíngue (linha `main.tex:129`), que está dentro do bloco `\begin{otherlanguage*}{english}` — manter conforme norma ABNT.

---

## 🗂️ Onde os arquivos canônicos vivem

- **Fonte da verdade (CANON):** `C:\Users\cesar\Documents\GitHub\GNN Fake News\Material\GNN_TCC_atualizado\`
- **NÃO usar mais:** o cache do app (`C:\Users\cesar\AppData\Local\...\outputs\tcc\`) — está deprecated; CANON sincronizado mais recente.
- **Workflow:** edita em CANON, compila em CANON. Outro Claude (Cowork) precisa de pasta conectada explicitamente pra acessar.
