% Minimal abntex2 compatibility shim for environments without abntex2
% Wraps the standard 'report' class and provides the most common abntex2 commands.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{abntex2}[2024/01/01 Minimal abntex2 shim]

\DeclareOption{12pt}{\PassOptionsToClass{12pt}{report}}
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{report}}
\DeclareOption{a4paper}{\PassOptionsToClass{a4paper}{report}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[12pt,oneside,a4paper]{report}

\RequirePackage[a4paper,top=3cm,bottom=2cm,left=3cm,right=2cm]{geometry}
\RequirePackage{setspace}
\onehalfspacing

% Metadata commands
\newcommand{\titulo}[1]{\def\@titulo{#1}\title{#1}}
\newcommand{\autor}[1]{\def\@autor{#1}\author{#1}}
\newcommand{\local}[1]{\def\@local{#1}}
\newcommand{\data}[1]{\def\@data{#1}\date{#1}}
\newcommand{\instituicao}[1]{\def\@instituicao{#1}}
\newcommand{\orientador}[1]{\def\@orientador{#1}}

% Cover page
\newcommand{\imprimircapa}{%
  \begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Large\bfseries\@titulo\par}
    \vspace{2cm}
    {\large\@autor\par}
    \vfill
    {\large\@local, \@data\par}
  \end{titlepage}
}

% Title page (folha de rosto)
\newcommand{\imprimirfolhaderosto}{%
  \begin{titlepage}
    \centering
    \vspace*{2cm}
    {\large\@autor\par}
    \vspace{3cm}
    {\Large\bfseries\@titulo\par}
    \vfill
    {\large\@local\\\@data\par}
  \end{titlepage}
}

% resumo environment (like abstract but labeled "Resumo")
\newenvironment{resumo}[1][Resumo]{%
  \chapter*{#1}
  \addcontentsline{toc}{chapter}{#1}
}{%
  \par
}

% apendicesenv environment
\newenvironment{apendicesenv}{%
  \appendix
}{%
}

\newcommand{\partapendices}{%
  \part*{Ap\^endices}
}

% citacao environment (direct quote)
\newenvironment{citacao}{%
  \begin{quote}\small
}{%
  \end{quote}
}

% --- Folha de aprovacao (placeholder pre-defesa) ---
\newcommand{\imprimirfolhadeaprovacao}{%
  \begin{titlepage}
    \centering
    \vspace*{1.5cm}
    {\large\@autor\par}
    \vspace{2cm}
    {\large\bfseries\@titulo\par}
    \vspace{2cm}
    \begin{minipage}{0.75\textwidth}
      Trabalho de Conclus\~ao de Curso apresentado ao curso de
      Ci\^encia de Dados da Pontif\'icia Universidade Cat\'olica de
      S\~ao Paulo como requisito parcial para obten\c{c}\~ao do t\'itulo
      de Bacharel em Ci\^encia de Dados.
    \end{minipage}
    \vspace{2cm}
    Aprovado em: \rule{5cm}{0.4pt}
    \vspace{2cm}
    \par
    {\bfseries BANCA EXAMINADORA\par}
    \vspace{1.5cm}
    \rule{10cm}{0.4pt}\par
    Prof. Orientador(a) -- PUC-SP\par
    \vspace{1.5cm}
    \rule{10cm}{0.4pt}\par
    Prof. Examinador(a) -- PUC-SP\par
    \vspace{1.5cm}
    \rule{10cm}{0.4pt}\par
    Prof. Examinador(a) -- PUC-SP\par
    \vfill
    {\@local, \@data\par}
  \end{titlepage}
}

% --- Lista de siglas/abreviaturas ---
% Uso: \begin{siglas} \item[GCN] Graph Convolutional Network ... \end{siglas}
\newenvironment{siglas}{%
  \chapter*{Lista de abreviaturas e siglas}%
  \addcontentsline{toc}{chapter}{Lista de abreviaturas e siglas}%
  \begin{description}\setlength{\itemsep}{2pt}
}{%
  \end{description}
}

% --- Lista de simbolos ---
\newenvironment{simbolos}{%
  \chapter*{Lista de s\'imbolos}%
  \addcontentsline{toc}{chapter}{Lista de s\'imbolos}%
  \begin{description}\setlength{\itemsep}{2pt}
}{%
  \end{description}
}

% --- Comandos auxiliares para insercao de listas padroes ABNT ---
\newcommand{\imprimirlistadefiguras}{%
  \listoffigures
  \addcontentsline{toc}{chapter}{Lista de figuras}%
}

\newcommand{\imprimirlistadetabelas}{%
  \listoftables
  \addcontentsline{toc}{chapter}{Lista de tabelas}%
}

% Make \chapter work nicely
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
