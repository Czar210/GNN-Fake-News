# Pacote Overleaf — Fase 9 (análise estratificada)

Como subir o conteúdo deste zip pro Overleaf, arquivo por arquivo.

## Antes de começar

**Faça backup do seu projeto Overleaf atual** (Menu → Download Source).
Os dois arquivos `.tex` na pasta `capitulos/` deste zip **substituem** os
existentes no Overleaf. Se você fez edições locais em `03_metodologia.tex`
ou `04_resultados.tex` depois do último commit do GitHub, vai precisar
fazer merge manual.

Os outros 10 arquivos (8 PNGs + 2 `.tex` de tabela) são **adições novas**
— não substituem nada, só somam.

## Estrutura esperada do seu projeto no Overleaf

```
<seu projeto>/
├── main.tex
├── Referencias.bib
├── capitulos/
│   ├── 01_introducao.tex
│   ├── 02_fundamentacao.tex
│   ├── 03_metodologia.tex      ← VAI SER SUBSTITUÍDO
│   ├── 04_resultados.tex       ← VAI SER SUBSTITUÍDO
│   ├── 05_aplicacao.tex
│   └── 06_conclusao.tex
├── apendices/
│   └── ... (A, B, C, D)
└── Imagens/
    ├── F1_*.png ... F24_*.png  ← já existem
    ├── F25_*.png ... F32_*.png ← NOVOS (vão ser adicionados)
    └── tabelas/
        ├── T1_*.tex ... T11_*.tex ← já existem
        ├── T12_outliers.tex            ← NOVO
        └── T13_gap_profundidade.tex    ← NOVO
```

## Mapeamento dos 12 arquivos do zip

### Capítulos modificados (2 arquivos — vão substituir)

| Arquivo no zip | Vai pra | Mudança |
|---|---|---|
| `capitulos/03_metodologia.tex` | `capitulos/03_metodologia.tex` | **Adicionou §3.8** "Análise estratificada do erro" entre §3.7 (Protocolo) e §3.9 (Explicabilidade — antes era §3.8) |
| `capitulos/04_resultados.tex` | `capitulos/04_resultados.tex` | **Adicionou §4.8 inteira** com 4 subseções (4.8.1 distribuição, 4.8.2 LOWESS, 4.8.3 outliers, 4.8.4 cascatas profundas). Painel mestre virou §4.9 |

### Figuras novas (8 PNGs — vão somar)

Todas vão pra pasta `Imagens/` do seu projeto. Sobem na mesma pasta onde
estão F1 a F24.

| Arquivo no zip | Onde aparece no LaTeX |
|---|---|
| `Imagens/F25_distribuicao_log.png` | §4.8.1 — fig:distribuicao_log |
| `Imagens/F26_lowess_acerto.png` | §4.8.2 — fig:lowess_acerto |
| `Imagens/F27_f1_por_tercil.png` | §4.8.2 — fig:f1_por_tercil |
| `Imagens/F28_distribuicao_profundidade.png` | §4.8.4 — fig:distribuicao_profundidade |
| `Imagens/F29_outliers_acc.png` | §4.8.3 — fig:outliers_acc |
| `Imagens/F30_outliers_composicao.png` | §4.8.3 — fig:outliers_composicao |
| `Imagens/F31_f1_por_depth.png` | §4.8.4 — fig:f1_por_depth |
| `Imagens/F32_gap_subsets.png` | §4.8.4 — fig:gap_subsets |

### Tabelas novas (2 arquivos `.tex` — vão somar)

Vão pra pasta `Imagens/tabelas/` (mesma pasta onde estão T1 a T11).

| Arquivo no zip | Onde é chamada no LaTeX |
|---|---|
| `Imagens/tabelas/T12_outliers.tex` | §4.8.3 via `\input{Imagens/tabelas/T12_outliers}` |
| `Imagens/tabelas/T13_gap_profundidade.tex` | §4.8.4 via `\input{Imagens/tabelas/T13_gap_profundidade}` |

## Passo a passo no Overleaf

1. **Backup**: Menu (☰ canto superior esquerdo) → "Download Source" → salva um zip do estado atual.
2. **Suba as figuras**: na árvore de arquivos do Overleaf, clica na pasta `Imagens` → ícone de "upload" (seta pra cima) → arrasta as 8 PNGs do zip.
3. **Suba as tabelas**: entra em `Imagens/tabelas` no Overleaf → upload → arrasta `T12_outliers.tex` e `T13_gap_profundidade.tex`.
4. **Substitui os capítulos**:
   - Opção A (mais simples — substitui inteiro): apaga `capitulos/03_metodologia.tex` e `capitulos/04_resultados.tex` no Overleaf, depois faz upload dos do zip.
   - Opção B (merge manual): abre cada arquivo do zip num editor local lado a lado com a versão atual do Overleaf, e cola só os blocos novos. Use isso se vc fez edições locais nos `.tex` que valem a pena preservar.
5. **Recompila**: clica em "Recompile". Se faltar alguma imagem, o log do LaTeX vai dizer qual e onde.

## O que vc vai precisar escrever depois

Os capítulos têm comentários em três tipos pra te guiar:

- `% ARGUMENTO: <tese>` — a afirmação central da seção, em 1-2 linhas. **Não apaga** — é seu lembrete de pra onde a prosa precisa ir.
- `% RESUMO DO QUE VAI AQUI: <bullets>` — o que precisa ser dito, com números já apurados.
- `% ESCREVER (Cesar): <prompts>` — perguntas/sugestões específicas pra cada subseção. **Apaga conforme escreve**.

Em §4.8 especificamente, todos os números já estão nas figuras e nas
tabelas — você só precisa contar a história. Nenhuma decisão metodológica
nova é exigida da prosa; é estilo + conexões com seções vizinhas.

## Compilação esperada

Depois de subir tudo:
- O `\tableofcontents` vai automaticamente adicionar §3.8 (Cap 3) e §4.8 + §4.9 (Cap 4).
- As referências `\ref{...}` dos novos labels (`sec:estratificacao`, `sec:metod_estratificacao`, etc.) só funcionam **depois de recompilar duas vezes** — o LaTeX precisa de uma passada pra catalogar os labels novos e outra pra resolvê-los. Normal.
- Se aparecer warning "undefined reference" depois de 2 compilações, alguma `\ref` ficou apontando pra label que não existe — confere o que ficou em vermelho no PDF.

## Arquivos que **NÃO** estão neste zip mas existem no commit do GitHub

Pra contexto: o commit `3567d66` no GitHub também tem os scripts Python
(`28`, `29`, `30`), os CSVs de resultado, e o `PIPELINE.md` atualizado.
Esses **não vão pro Overleaf** — são código que gera os dados, ficam
fora do TCC. Mas se a banca quiser ver, está tudo no repo.
