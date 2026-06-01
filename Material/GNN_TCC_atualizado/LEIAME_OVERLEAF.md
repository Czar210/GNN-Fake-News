# Pacote Overleaf — TCC GNN Fake News (versão para escrita)

## Como usar

1. **Subir no Overleaf:**
   - New Project → Upload Project → arraste o `.zip` deste diretório.
   - Overleaf detecta `main.tex` como entry point automaticamente.

2. **Compilação:**
   - Sequência: `LaTeX → BibTeX → LaTeX → LaTeX` (botão "Recompile" do Overleaf faz tudo).
   - Classe usada: `abntex2` (já configurada).

3. **Onde escrever:**
   - Cada capítulo é um arquivo em `capitulos/0X_*.tex` com **comentários estruturados** (`% ESCREVER (...)`) indicando o que vai em cada seção, qual figura inserir, qual citação usar, e parágrafos âncora literais quando o conteúdo passou por revisão adversarial.
   - Substitua os comentários `%` pela prosa final.

## Estrutura

```
Material/GNN_TCC_atualizado/
├── main.tex                    # Documento principal (resumo PT/EN, includes, bib, apêndices)
├── Referencias.bib             # 32 entries (10 canônicas + 4 corrigidas + 18 mantidas)
├── INDICE.md                   # Mapa de figuras/tabelas → capítulo (referência rápida)
├── LEIAME_OVERLEAF.md          # este arquivo
├── capitulos/
│   ├── 01_introducao.tex       # esqueleto (ESCREVER por último)
│   ├── 02_fundamentacao.tex    # esqueleto (ESCREVER em penúltimo)
│   ├── 03_metodologia.tex      # esqueleto + F0 (TikZ pipeline) + T0 + T_hyperparams (inline)
│   ├── 04_resultados.tex       # esqueleto + 11 \input/\includegraphics prontos
│   ├── 05_aplicacao.tex        # esqueleto + 9 \input/\includegraphics prontos
│   └── 06_conclusao.tex        # esqueleto (sem figuras)
├── apendices/
│   ├── A_hiperparametros.tex
│   ├── B_indice_figuras_tabelas.tex
│   ├── C_estrutura_repositorio.tex
│   └── D_evolucao_metodologica.tex   # opcional, comentado em main.tex
└── Imagens/
    ├── F1_*.png ... F24_*.png         # 22 figuras (F12 e F13 como subpastas)
    ├── F12_gnnexplainer_gossipcop/    # 5 PNGs de amostras explicadas
    ├── F13_threads_bluesky/           # 6 PNGs de threads reais
    └── tabelas/T1_*.tex ... T11_*.tex # 11 tabelas LaTeX prontas (\input)
```

## Ordem de escrita recomendada

`3 → 4 → 5 → 2 → 6 → 1` (capítulos densos primeiro; destilatórios depois).

Razão: Metodologia e Resultados têm conteúdo concreto (números, citações, achados); Fundamentação e Introdução destilam melhor depois que vc sabe exatamente o que fundamentar/introduzir.

## Tamanho estimado

- Cap 1 (Introdução): 4-6 páginas
- Cap 2 (Fundamentação): 12-18 páginas
- Cap 3 (Metodologia): 8-12 páginas
- Cap 4 (Resultados): 15-22 páginas
- Cap 5 (Aplicação): 8-12 páginas
- Cap 6 (Conclusão): 4-6 páginas
- Apêndices: 4-8 páginas

**Total esperado: 55-84 páginas** (sem capa, sumário, referências).

## Pegadinhas globais

1. **Não diga "isso prova que..."** — use "evidência sugere", "consistente com", "compatível com a hipótese de".
2. **Sempre que citar número exato**, dê fonte do CSV/script entre parênteses no parágrafo.
3. **F1=0.81 com std=0.002 vs F1=0.814** (diferença de 4 milésimos): primeira é média de 10 seeds (script 14, §4.4); segunda é run único da seed=42 (script 17, §4.7). Cite a fonte exata pra não parecer cherry-picking.
4. **Decimal usa ponto** (consistente com PT-BR técnico): `0.810`, não `0,810`. (Texto técnico em PT também aceita; verifique convenção da sua instituição.)
5. **Apêndice D opcional** está comentado no `main.tex`. Descomente apenas se a banca pedir contexto sobre "por que a versão final difere de propostas anteriores".
6. **Toda figura precisa `\caption{}` e `\label{fig:xxx}`**. Tabelas idem (`\label{tab:xxx}`). Já implementado.

## Documentos complementares (no repositório, fora do ZIP)

- `estrutura_tcc_v2.md` — outline conceitual com justificativa de cada seção (revisado em 3 rodadas adversariais, conceito A no parecer final).
- `guia_escrita_tcc.md` — guia operacional por seção (parágrafos âncora literais, fontes dos CSVs, citações `.bib` exatas).

Mantenha estes 2 abertos enquanto escreve — `02_fundamentacao.tex`/`04_resultados.tex` apontam frequentemente pra eles via comentário.
