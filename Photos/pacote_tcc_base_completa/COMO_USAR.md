# Como usar este pacote — base completa do TCC

Pacote feito pra **reescrever o TCC do zero** com estrutura limpa, todas
as figuras e tabelas no lugar, e prompts minimalistas dizendo o que
escrever antes e depois de cada elemento.

## O que tem aqui

```
pacote_tcc_base_completa/
├── COMO_USAR.md            ← você está lendo
├── INDICE.md               ← mapa completo: capítulo → seção → figuras
├── main.tex                ← arquivo mestre (\include cada capítulo)
├── Referencias.bib         ← bibliografia
├── capitulos/
│   ├── 01_introducao.tex   ← scaffolding limpo, 6 capítulos
│   ├── 02_fundamentacao.tex
│   ├── 03_metodologia.tex
│   ├── 04_resultados.tex
│   ├── 05_aplicacao.tex
│   └── 06_conclusao.tex
├── apendices/
│   ├── A_hiperparametros.tex
│   ├── B_indice_figuras_tabelas.tex
│   ├── C_estrutura_repositorio.tex
│   └── D_evolucao_metodologica.tex
└── Imagens/
    ├── F1..F32 .png        ← 28 PNGs prontos
    ├── F12_gnnexplainer_gossipcop/  ← pasta de PNGs auxiliares
    ├── F13_threads_bluesky/         ← pasta de PNGs de threads
    └── tabelas/
        └── T1..T13 .tex    ← 13 tabelas LaTeX prontas
```

## A estrutura dos comentários no LaTeX

Cada seção tem este padrão:

```latex
\section{Título}
\label{sec:label}

% TESE: <1 frase, o argumento central da seção>

% ANTES: <o que escrever antes da figura/tabela>
% - bullet 1
% - bullet 2

\begin{figure}[H]\centering
\includegraphics{...}
\caption{...}
\label{fig:...}
\end{figure}

% DEPOIS: <o que escrever depois pra interpretar>
% - bullet 1
% - bullet 2

% TRANSICAO: <1 frase de ponte pra próxima seção>
```

**Fluxo de escrita:**
1. Lê a `% TESE:` — sabe pra onde a seção vai.
2. Escreve o parágrafo de `% ANTES:` usando os bullets como guia.
3. Olha a figura/tabela já incluída no lugar certo.
4. Escreve o parágrafo de `% DEPOIS:` interpretando os números.
5. Escreve a transição.
6. **Apaga os comentários conforme escreve.**

Pra seções sem figura/tabela (Cap 1, 2, 6 inteiros), o padrão é:
```latex
% TESE: ...
% CONTEUDO: <bullets do que dizer>
% TRANSICAO: ...
```

## Como subir no Overleaf

### Opção A — Projeto novo do zero (recomendado se quer reescrever inteiro)

1. No Overleaf: New Project → Upload Project → arrasta o zip inteiro.
2. Aguarda o Overleaf descompactar.
3. Confirma que o arquivo principal é `main.tex` (Menu → Settings → Main document).
4. Recompila. PDF deve sair em 30s.

### Opção B — Substituir projeto existente

1. Abre o projeto atual no Overleaf.
2. **Faz backup**: Menu → Download Source.
3. Apaga os arquivos antigos (`capitulos/*.tex`, `apendices/*.tex`).
4. Faz upload da pasta `capitulos/` deste pacote, depois `apendices/`.
5. Confere que `main.tex` e `Referencias.bib` da raiz estão atualizados.

### Opção C — Merge manual (cuidado)

Se você já escreveu prosa em algum capítulo e quer preservar, faz
diff manual entre cada `.tex` deste pacote e o do Overleaf. Mais
trabalhoso mas seguro contra perda.

## Ordem de escrita sugerida

Conforme a estrutura aprovada pela revisão adversarial (descrita em
`estrutura_tcc_v2.md` no repo):

1. **Cap 3 (Metodologia)** — fácil porque é descritivo + a maioria das
   tabelas já estão prontas. Dá pra escrever sem dados.
2. **Cap 4 (Resultados)** — escreva em paralelo com Cap 3. Aqui mora a
   tese. Use bastante números — todos os achados estão nas tabelas e
   comentários `% DEPOIS:`.
3. **Cap 5 (Aplicação)** — depois de 3 e 4 porque referencia ambos.
4. **Cap 6 (Conclusão)** — só faz sentido escrever depois de 4 e 5.
5. **Cap 2 (Fundamentação)** — penúltimo. Você vai saber melhor o que
   precisa fundamentar quando já viu os dados.
6. **Cap 1 (Introdução)** — último. Conta a história sabendo o final.

Tamanho-alvo total: ~50-70 páginas + apêndices. Cabe em 1-2 semanas de
escrita focada.

## O que NÃO está aqui (intencionalmente)

- **Código Python.** Os scripts que geraram os números estão no repo
  GitHub em `Training/03_Mega_Research/`. Não vão pro Overleaf.
- **CSVs de resultado.** Idem — ficam no repo.
- **HTMLs interativos.** As 6 threads do Bluesky e as 3 cascatas
  profundas viram demo oral, não PDF.
- **Apêndice D ativado.** `apendices/D_evolucao_metodologica.tex` existe
  no zip mas está comentado em `main.tex`. Descomenta só se a banca
  pedir contexto da evolução do trabalho.

## Erros comuns ao compilar

- **`File ... not found`** — algum nome de figura ou tabela está errado.
  Confere `INDICE.md` pro nome canônico.
- **`Undefined reference`** depois de 1 compilação — normal. **Recompila
  uma segunda vez** que o LaTeX resolve os `\ref{}`.
- **`Missing bibliography`** — recompila com BibTeX ativado no Overleaf
  (Menu → Settings → Compiler → ative "bibtex").
- **`\input` falha em alguma tabela** — verifica se o `.tex` da tabela
  existe em `Imagens/tabelas/`. T2 fica sem casa por padrão (ver INDICE).

## Onde olhar quando travar

- Achado específico de número: `INDICE.md` lista qual figura/tabela tem
  cada número.
- Argumento de uma seção: o comentário `% TESE:` da seção é o resumo de
  1 frase do que ela defende.
- Conexão entre seções: comentários `% TRANSICAO:` mostram a costura.
- Documento de história (origem das ideias da Fase 9): repo GitHub
  `HISTORIA_FASE9.md`.
- Dúvida sobre o pipeline de código: repo GitHub
  `Training/03_Mega_Research/PIPELINE.md`.
