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

% Make \chapter work nicely
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
