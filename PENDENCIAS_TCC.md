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

| Chave | Para que | Onde |
|---|---|---|
| `cleveland1979lowess` | LOWESS | §3.8 |
| `efron1993bootstrap` | Bootstrap pareado | §3.8 |
| `sibila2025bluesky` | Artefatos publicados no HF Hub (não publicados ainda) | §3.2, §3.11, §5/Apêndice E, §6.4 |

**Importante:** `sibila2025bluesky` precisa ser criado quando o Cesar efetivamente publicar os artefatos no Hugging Face. Antes disso, vai dar *undefined reference* na compilação — pode marcar como `% TODO_CITE: sibila2025bluesky` se quiser que a compilação não pare.

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

- A "matriz de confusão página 25 RF" que o professor mencionou é qual figura exatamente? A página 25 do PDF anterior tinha o **Painel Mestre** (F16), não uma matriz de confusão real. Candidatas: F20 (cross-tab Bluesky, pág 28) ou F16. Confirmar pra repintura final.
- Se Apêndice E é aceitável pelo orientador (alternativa a cortar tudo do Bluesky).

---

## 🗂️ Onde os arquivos canônicos vivem

- **Fonte da verdade (CANON):** `C:\Users\cesar\Documents\GitHub\GNN Fake News\Material\GNN_TCC_atualizado\`
- **NÃO usar mais:** o cache do app (`C:\Users\cesar\AppData\Local\...\outputs\tcc\`) — está deprecated; CANON sincronizado mais recente.
- **Workflow:** edita em CANON, compila em CANON. Outro Claude (Cowork) precisa de pasta conectada explicitamente pra acessar.
