%% abntex2-alf.bst, v-1.9.7 laurocesar
%% Copyright 2012-2018 by abnTeX2 group at http://www.abntex.net.br/ 
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%   http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%% 
%% The Current Maintainer of this work is the abnTeX2 team, led
%% by Lauro C?sar Araujo. Further information are available on 
%% http://www.abntex.net.br/
%%
%% Creator and original mantainer: Gerald Weber <gweber@codigolivre.org.br>
%% Copyright 2001-2002 by the abnTeX group at http://abntex.codigolivre.org.br
%%
%% This file is an *extensive* modification from a file generated with the docstrip utility.

ENTRY
  { address author
    booktitle booksubtitle
    chapter conference-number conference-year conference-location
    edition editor editortype
    dimensions furtherresp
    howpublished illustrated institution
    isbn issn journal key month note marginnote number organization
    org-short pages pagename
    publisher reprinted-from reprinted-text school
    series subtitle section title type url urlaccessdate
    volume year year-presented
    abnt-and-type
    abnt-doi abnt-dont-use-etal
    abnt-etal-cite abnt-etal-list abnt-etal-text abnt-emphasize abnt-experimental
    abnt-full-initials
    abnt-last-names abnt-ldots-type
    abnt-missing-year abnt-cite-style abnt-no-etal-label abnt-note
    abnt-repeated-author-omit abnt-repeated-title-omit
    abnt-show-options abnt-refinfo abnt-thesis-year abnt-url-package
    abnt-title-command
    abnt-verbatim-entry
    iso-abbreviation iso-author-punctuation iso-date-place
    }{}{ label extra.label year.label}

STRINGS { abnt.bst.revision }
STRINGS { r s t }
STRINGS {abnt.etal.text abnt.emphasize previous.author previous.title}
INTEGERS {abnt.alf  abnt.cite.style abnt.doi
          abnt.experimental abnt.etal.cite abnt.etal.list
          abnt.full.initials
          abnt.last.names abnt.ldots.type
          abnt.missing.year
          abnt.refinfo  abnt.repeated.author.omit abnt.repeated.title.omit
	  abnt.show.options abnt.and.type
          abnt.title.command
	  abnt.thesis.year
	  abnt.url.package abnt.verbatim.entry
	  iso.abbreviation iso.author.punctuation iso.date.place
	  repeated.author}

INTEGERS { title.lenght title.pos first.word no.bracket}
STRINGS { title.char}
INTEGERS {output.state before.all mid.sentence after.sentence after.block }
INTEGERS {url.type}

FUNCTION {abnt.doi.expand.to.url} {#0}
FUNCTION {abnt.doi.link}          {#1}
FUNCTION {abnt.doi.leave.as.is}   {#2}

FUNCTION {unknown.url} {#0}
FUNCTION {http.url}    {#1}
FUNCTION {file.url}    {#2}
FUNCTION {ftp.url}     {#3}
FUNCTION {doi.url}     {#4}
FUNCTION {mailto.url}  {#5}

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$ newline$ }
        { output.state before.all = 'write$
            { add.period$ " " * write$ }
          if$}
      if$
      mid.sentence 'output.state := }
  if$
  s
}
FUNCTION {output.nonnull.dash}
{ 's :=
  output.state mid.sentence =
    { " --- " * write$ }
    { output.state after.block =
        { add.period$ write$ newline$ write$}
        { output.state before.all = 'write$
            { add.period$ " " * write$ }
          if$}
      if$
      mid.sentence 'output.state := }
  if$
  s
}
FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}
FUNCTION {output.dash}
{ duplicate$ empty$
    'pop$
    'output.nonnull.dash
  if$
}
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "campo " t * " vazio em " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}
FUNCTION {add.blank} {  " " * before.all 'output.state :=}
FUNCTION {date.block} {new.block}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}
FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { abnt.emphasize "{"  *  swap$ * "}" * }
  if$
}

FUNCTION {tie.or.space.prefix}
{ duplicate$ text.length$ #5 <
    { "~" }
    { " " }
  if$
  swap$
}

FUNCTION {no.space} { { "" } swap$ }
FUNCTION {space.word} { " " swap$ * " " * }

FUNCTION {bbl.and}
{ abnt.and.type #0 = {" e "} 'skip$ if$
  abnt.and.type #1 = {" \& "} 'skip$ if$}
FUNCTION {bbl.chapter} { "cap." }
FUNCTION {bbl.colon} {": "}
FUNCTION {bbl.editors} { "Ed." }
FUNCTION {bbl.editor} { "Ed." }
FUNCTION {bbl.edition} { "ed." }
FUNCTION {bbl.empty} {""}
FUNCTION {bbl.etal}{abnt.etal.text "" = {""} {" " abnt.etal.text *} if$}
FUNCTION {bbl.doi.url} {"http://dx.doi.org/"}
FUNCTION {bbl.in}{ "In" iso.abbreviation 'skip$ { bbl.colon *} if$}
FUNCTION {bbl.ldots}
{abnt.ldots.type
  {abnt.ldots.type #1 = {"\ldots"}   'skip$ if$
   abnt.ldots.type #2 = {"$\ldots$"} 'skip$ if$
   abnt.ldots.type #3 = {"{...}"}    'skip$ if$}
  {""}
if$}
FUNCTION {bbl.master}  { "Mestrado" }
FUNCTION {bbl.missing.year}
{abnt.missing.year
  {"[s.d.]"}
  {""}
 if$}
FUNCTION {bbl.monography} { "Monografia" }
FUNCTION {bbl.mthesis} { "Disserta\c{c}\~{a}o" }
FUNCTION {bbl.no.address} {"[S.l.]"}
FUNCTION {bbl.no.address.no.publisher} {"[S.l.: s.n.]"}
FUNCTION {bbl.no.publisher} {"[s.n.]"}
FUNCTION {bbl.number} { iso.abbreviation {"no."} {"n."} if$ }
FUNCTION {bbl.nr} { "n." }
FUNCTION {bbl.of} { "de" }
FUNCTION {bbl.phd} { "Doutorado" }
FUNCTION {bbl.phdthesis} { "Tese" }
FUNCTION {bbl.pages}
  {pagename empty$
    { "p." }
    {pagename}
   if$
   }
FUNCTION {bbl.page} { bbl.pages }
FUNCTION {bbl.reprint} {"Separata de"}
FUNCTION {bbl.semicolon} { "; "}
FUNCTION {bbl.same.field} {"\underline{\ \ \ \ \ \ \ \ }"}
FUNCTION {bbl.techrep} { "Rel. Tec." }
FUNCTION {bbl.url.accessdate}
  {iso.abbreviation { "Cited "} {"Acesso em: "} if$}
FUNCTION {bbl.url.available}
{ iso.abbreviation {"Available from Internet: "} {"Dispon{\'\i}vel em: "} if$}
FUNCTION {bbl.void} {""}
FUNCTION {bbl.volume} { iso.abbreviation {"vol."} {"v."} if$ }

MACRO {jan} {"jan."}
MACRO {feb} {"fev."}
MACRO {mar} {"mar."}
MACRO {apr} {"abr."}
MACRO {may} {"maio"}
MACRO {jun} {"jun."}
MACRO {jul} {"jul."}
MACRO {aug} {"ago."}
MACRO {sep} {"set."}
MACRO {oct} {"out."}
MACRO {nov} {"nov."}
MACRO {dec} {"dez."}

FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}
FUNCTION {str.to.int}
{ duplicate$ is.num
    {chr.to.int$ #48 -}
    'skip$
  if$
}
FUNCTION {bibinfo.check}
{ swap$ duplicate$ missing$
    {pop$ pop$ ""}
    { duplicate$ empty$
        {swap$ pop$}
        {swap$ pop$}
      if$
    }
  if$
}
FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    {swap$ "faltando " swap$ * " em " * cite$ * warning$ pop$ ""}
    { duplicate$ empty$
        {swap$ "vazio " swap$ * " em " * cite$ * warning$}
        {swap$ pop$}
      if$
    }
  if$
}

STRINGS  { bibinfo}
INTEGERS { nameptr namesleft numnames }

FUNCTION {get.person} {"{vv }{ll}{, jj}{, ff}" format.name$}
FUNCTION {get.last.name} {#1 "{ll}" format.name$}
FUNCTION {get.first.name} {#1 "{ff}" format.name$}
FUNCTION {get.von} {#1 "{vv}" format.name$}
FUNCTION {get.jr} {#1 "{jj}" format.name$}

FUNCTION {format.single.name}
{
s nameptr get.person 'r :=
abnt.last.names
  {r #1 "{ll}{ jj}" format.name$ "u" change.case$
   abnt.full.initials
    {r #1 "{, ff}{ vv}" format.name$ *}
    {r #1 "{, f.}{ vv}" format.name$ *}
  if$}
  {r #1 "{ll}" format.name$
   #1 "{ll}" format.name$
   r #1 "{ jj}" format.name$ * "u" change.case$
   abnt.full.initials
    {r #1 "{, ff}{ vv}" format.name$ *
     r #1 "{ll}" format.name$
     #1 "{ ff}{ vv}" format.name$ *}
    {iso.author.punctuation
      {r #1 "{, f{}}{ vv}" format.name$ *}
      {r #1 "{, f.}{ vv}" format.name$ *} if$
     r #1 "{ll}" format.name$
     #1 "{ f.}{ vv}" format.name$ *}
  if$}
if$
}
FUNCTION {format.names}
{ 'bibinfo :=  duplicate$ empty$ 'skip$
  {'s :=
   "" 't :=
   #1 'nameptr :=  s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    { format.single.name
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {nameptr #1 #1 + = numnames abnt.etal.list > and
            {abnt.etal.list {"others" 't := #1 'namesleft :=} 'skip$ if$ }
            'skip$
          if$
          namesleft #1 >
            { "; " * t * }
            { numnames #2 >
                { "" * }
                'skip$
              if$
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                { bbl.etal *}
                { "; " * t *}
              if$ }
          if$ }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft := }
  while$
  } if$
}

FUNCTION {format.names.ed} {format.names}

FUNCTION {format.authors}
{
author "author" format.names
}

FUNCTION {get.bbl.editor} { editor num.names$ #1 > 'bbl.editors 'bbl.editor if$ }

FUNCTION {format.book.pages}
{ pages "pages" bibinfo.check  duplicate$ empty$
    'skip$
    {  "~" * bbl.pages * }
  if$
}

FUNCTION {format.isbn}
{ isbn "isbn" bibinfo.check duplicate$ empty$ 
    'skip$
    {new.block "ISBN " swap$ *}
  if$
}

FUNCTION {format.issn}
{ issn "issn" bibinfo.check
  duplicate$ empty$ 'skip$
    {new.block "ISSN " swap$ *}
  if$
}

FUNCTION {format.note}
{ note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$ }
      if$
      note #2 global.max$ substring$ * "note" bibinfo.check }
  if$
}

FUNCTION {format.howpublished}
{ howpublished empty$
    { "" }
    { howpublished #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$ }
      if$
      howpublished #2 global.max$ substring$ * "howpublished" bibinfo.check }
  if$
}

FUNCTION {format.title.as.author}
{ 's :=
  #1 'title.pos :=
  #1 'first.word :=
  #1 'no.bracket :=
  "" 't :=
  {first.word}
    {s title.pos #1 substring$ 'title.char :=
      title.char "{" = {#0 'no.bracket :=} 'skip$ if$
      no.bracket 'skip$ {title.char "}" = {#1 'no.bracket :=} 'skip$ if$}  if$
      type$ "journalpart" =
        {title.char "." = title.char ":" = or title.char "!" = or
	 title.char "," = or title.char ";" = or title.char "?" = or}
	{title.char " " =}
      if$ s t = or no.bracket and
       {#0 'first.word :=
       t title.char * "u" change.case$
       title.pos #1 + 'title.pos :=
       s title.pos global.max$ substring$ * 's :=}
       {title.pos #1 + 'title.pos := t title.char * 't :=}
     if$
     }
  while$
s
}

FUNCTION {compare.title}
{abnt.repeated.title.omit
  {duplicate$ "" = 'skip$
    {duplicate$ previous.title = repeated.author and
      {duplicate$ 'previous.title := pop$ bbl.same.field}
      {duplicate$ 'previous.title :=}
     if$}
   if$}
 'skip$
 if$
}

FUNCTION {apply.title.command}
{abnt.title.command
  {duplicate$ empty$ 'skip$
    {"\bibtextitlecommand{" type$ * "}{" * swap$ * "}" *}
    if$}
  'skip$ if$
}

FUNCTION {format.title}
{author empty$ editor empty$ and organization empty$ and
  {title format.title.as.author}
  {title  duplicate$ empty$ {pop$ ""} {"t" change.case$} if$}
 if$
 subtitle  empty$ 'skip$ {subtitle ": " swap$ * "t" change.case$ *} if$
 apply.title.command
 compare.title
}

FUNCTION {n.dashify}
{
  't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {word.in} { bbl.in " " * }

FUNCTION {format.date}
{ year empty$
    { month empty$
	{ bbl.missing.year }
	{ "existe o campo month mas nao o campo year em " cite$ * warning$ month}
      if$}
    { month empty$
	{ year n.dashify}
	{ month " " * year * }
      if$}
  if$
}
FUNCTION {format.btitle}
{author empty$ editor empty$ and organization empty$ and
    {title format.title.as.author "title" bibinfo.check duplicate$ empty$ 'skip$ {} if$
    subtitle  empty$ 'skip$ {subtitle ": " swap$ * "t" change.case$ *} if$}
    {title "title" bibinfo.check emphasize duplicate$ empty$ 'skip$ {} if$
    subtitle  empty$ 'skip$ {subtitle ": " swap$ * "t" change.case$ *} if$}
if$
apply.title.command
compare.title
}

FUNCTION {format.volume}
{volume duplicate$ empty$
   'skip$
   {volume #1 #1 substring$ is.num
      {bbl.volume swap$ tie.or.space.prefix * *}
      'skip$
    if$}
 if$
}

FUNCTION {format.number}
{ number duplicate$ empty$
    'skip$
    {bbl.number swap$ tie.or.space.prefix * *}
  if$
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { format.volume
      number empty$ 'skip$ {output format.number} if$
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$
          emphasize * }
      if$
    }
  if$
}

FUNCTION {format.bvolume.if.no.series}
{ series empty$
    { format.bvolume }
    { "" }
  if$
}

FUNCTION {format.number.or.volume}
{number empty$
  {format.volume}
  {number}
 if$
}

FUNCTION {format.volume.number.series}
{ number empty$ volume empty$ and
  { series empty$
    { "" }
    {" (" series "series" bibinfo.check * ")" *}
    if$}
  { series empty$
    {""}
    { " ("  series "series" bibinfo.check * ", " *
     format.number.or.volume * ")" *  }
     if$ }
  if$
}

FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    {output.state mid.sentence = { "l" } { "t" } if$ change.case$
      "edition" bibinfo.check
    edition #1 #1 substring$ is.num {". " * bbl.edition * } 'skip$ if$}
  if$
}
INTEGERS { multiresult }
FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$ }
  while$
  multiresult
}
FUNCTION {format.pages}
{ pages duplicate$ empty$ 'skip$
    { duplicate$ multi.page.check
        { n.dashify bbl.pages swap$}
        { bbl.page swap$ }
      if$
      tie.or.space.prefix "pages" bibinfo.check * * }
  if$
}
FUNCTION {format.journal.pages}
{ pages empty$ {""}
    { pages n.dashify bbl.pages
          swap$ tie.or.space.prefix "pages" bibinfo.check * *}
  if$
}

FUNCTION {get.url.type}
{s #1 #5 substring$ "http:" = {http.url}
 {s #1 #6 substring$ "https:" = {http.url}
  {s #1 #5 substring$ "file:" = {file.url}
   {s #1 #4 substring$ "ftp:"  = {ftp.url}
    {s #1 #4 substring$ "doi:"  = {doi.url}
     {s #1 #7 substring$ "mailto:" = {mailto.url} {unknown.url} if$}
      if$}
     if$}
    if$}
   if$}
  if$
 'url.type :=
}


INTEGERS {url.pos url.length}
FUNCTION {filter.url.tex}
{s text.length$ #1 + 'url.length :=
 #1 'url.pos := "" 't :=
  {url.pos url.length <}
    {s url.pos #1 substring$
    duplicate$ "_" = {pop$ "\_" } 'skip$ if$
    duplicate$ "%" = {pop$ "\%" } 'skip$ if$
    duplicate$ "&" = {pop$ "\&" } 'skip$ if$
    t swap$ * 't :=
    url.pos #1 + 'url.pos :=
    }
  while$
t
}
FUNCTION {filter.url.html}
{s text.length$ #1 + 'url.length :=
 #1 'url.pos := "" 't :=
  {url.pos url.length <}
    {s url.pos #1 substring$
    duplicate$ "%" = {pop$ "\%" } 'skip$ if$
    t swap$ * 't :=
    url.pos #1 + 'url.pos :=
    }
  while$
t
}
FUNCTION {expand.doi}
{s
url.type doi.url =
  {#5 global.max$ substring$ bbl.doi.url swap$ *}
  'skip$ if$
}

FUNCTION {compose.url}
{'s :=
 get.url.type
 bbl.url.available
 abnt.url.package #0 =
   { "$<$"   *
   abnt.doi abnt.doi.expand.to.url = 'expand.doi {s} if$
   * "$>$" *} 'skip$ if$
 abnt.url.package #1 =
   { "\url{" * s * "}" *} 'skip$ if$
}
FUNCTION {format.url}
{ url empty$ 'skip$
    {url compose.url output new.sentence }
  if$
 urlaccessdate duplicate$ empty$ 'skip$
    { bbl.url.accessdate swap$ *}
  if$
  "urlaccessdate" bibinfo.check output new.sentence
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type "l" change.case$
          "type" bibinfo.check }
      if$
      chapter tie.or.space.prefix "chapter" bibinfo.check * *
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$ }
  if$
}

FUNCTION {format.booktitle}
{author empty$ editor empty$ and organization empty$ and
    {booktitle format.title.as.author "booktitle" bibinfo.check duplicate$ empty$ 'skip$ {} if$
    booksubtitle  empty$ 'skip$ {booksubtitle ": " swap$ * "t" change.case$ *
    apply.title.command} if$}
    {booktitle "booktitle" bibinfo.check emphasize duplicate$ empty$ 'skip$ {} if$
    booksubtitle  empty$ 'skip$ {booksubtitle ": " swap$ * "t" change.case$ *
    apply.title.command} if$}
if$
}
FUNCTION {format.editors}
{ editor  empty$
   {"inbook" type$ = 'bbl.same.field 'skip$ if$}
   {author empty$
    {editor "editor" format.names " (" *
     editortype empty$ {bbl.editor} {editortype} if$
     ")" * * }
    {"inbook" type$ =
      {bbl.same.field }
      {editor "editor" format.names " (" *
       editortype empty$ {bbl.editor} {editortype} if$
       ")" * *}
    if$}
   if$}
  if$
}
FUNCTION {format.in.ed.booktitle}
{ editor empty$
    {organization empty$ not author empty$ not and
      {word.in organization "u" change.case$ * output
       booktitle empty$ {""} {new.sentence format.booktitle} if$}
      {word.in
       booktitle empty$ 'skip$ {format.booktitle *} if$}
     if$}
    {word.in format.editors * output
     booktitle empty$ {""} {new.sentence format.booktitle} if$}
  if$
  output
  new.sentence
}

FUNCTION {format.field.address}
{ 't :=
  address empty$ organization empty$ and url empty$ and t empty$ and
    {bbl.no.address.no.publisher}
    {address empty$
      {url empty$  {bbl.no.address ": "} {"" ""} if$}
      {address ": "}
    if$
    t empty$
      {organization empty$ {* bbl.no.publisher * } 'pop$ if$}
      {* t * }
     if$}
  if$
}

FUNCTION {format.publisher.address}
{ publisher "publisher" bibinfo.check format.field.address}

FUNCTION {format.publisher.address.relax}
{publisher empty$ address empty$ and {""}
 {publisher "publisher" bibinfo.check format.field.address}
if$}

FUNCTION {format.organization.address}
{ organization "organization" bibinfo.check format.field.address}

FUNCTION {compare.authorship}
{abnt.repeated.author.omit
  {duplicate$ "" = 'skip$
    {duplicate$ previous.author =
      {duplicate$ 'previous.author :=
       pop$ bbl.same.field
       #1 'repeated.author :=}
      {duplicate$ 'previous.author :=
       #0 'repeated.author :=}
     if$}
   if$}
 'skip$
 if$
}

FUNCTION {format.author.or.organization}
{  author empty$
    { organization empty$ 'skip$
      { organization "u" change.case$ "organization"  bibinfo.check output}
      if$}
    { format.authors output.nonnull }
  if$
  compare.authorship
}

FUNCTION {format.editor.or.organization}
{  editor empty$
    { organization empty$ 'skip$
      { organization "u" change.case$ "organization"  bibinfo.check output}
      if$}
    { format.editors output.nonnull }
  if$
  compare.authorship
}

FUNCTION {format.author.or.editor.or.organization}
{ author empty$
    { editor empty$
        { organization empty$
	    'skip$
            { organization "u" change.case$ "organization"  bibinfo.check output}
          if$}
        { format.editors output.nonnull }
      if$}
    { format.authors output.nonnull }
  if$
  compare.authorship
}

FUNCTION {format.journal}
{
  journal emphasize
}

FUNCTION {format.names.for.label1}
{ 's := "" 't :=
   #1 'nameptr :=  s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    {abnt.last.names
       {s nameptr "{ll}{ jj}" format.name$}
       {s nameptr "{ll}" format.name$ #1 "{ll}" format.name$
        s nameptr "{ jj}" format.name$ *}
     if$
     't :=
      nameptr #1 >
        {namesleft #1 >
          { numnames abnt.etal.cite > abnt.etal.cite #0 = not and
	    {#0 'namesleft := abnt.etal.cite {bbl.etal *} 'skip$ if$}
	    {", " * t * } if$ }
            { t "others" =
              {abnt.etal.cite {bbl.etal *} 'skip$ if$}
                {numnames nameptr =
                {bbl.and * t *}
		'skip$ if$}
             if$
	     }
          if$ }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft := }
  while$
}

FUNCTION {format.names.for.label2}
{ 's := "" 't :=
   #1 'nameptr :=  s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    {abnt.last.names
       {s nameptr "{ll}{ jj}" format.name$}
       {s nameptr "{ll}" format.name$ #1 "{ll}" format.name$
        s nameptr "{ jj}" format.name$ *}
     if$
     abnt.cite.style #0 = {"u" change.case$} 'skip$ if$ 't :=
      nameptr #1 >
        {namesleft #1 >
            { numnames abnt.etal.cite > 
	      {#0 'namesleft := abnt.etal.cite {bbl.etal *} 'skip$ if$} 
	      {"; " * t * } if$ }
            { t "OTHERS" = t "others" = or
	      {abnt.etal.cite {bbl.etal *} 'skip$ if$}
	      {numnames nameptr = { bbl.semicolon * t *} 'skip$ if$}
             if$
	     }
          if$ }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft := }
  while$
}

FUNCTION {format.title.for.label}
{  's :=
  #1 'title.pos :=   #1 'first.word :=
  "" 't :=
  {first.word}
    {s title.pos #1 substring$ 'title.char :=
        title.char "." = title.char ":" = or title.char ";" = or
	title.char "," = or title.char " " = or s t = or
       {#0 'first.word :=
       title.pos #1 + 'title.pos :=}
       {title.pos #1 + 'title.pos := t title.char * 't :=}
     if$
     }
  while$
s t =
  {t  "}{" * t
  abnt.cite.style #0 = {"u" change.case$ * } 'skip$ if$}
  {t bbl.ldots * "}{" * t
  abnt.cite.style #0 = {"u" change.case$} 'skip$ if$ * bbl.ldots *}
if$
}

FUNCTION {calc.extended.label}
{ "{"
  author empty$ not
    {author format.names.for.label1 * "}{" * author format.names.for.label2 *}
    {editor empty$ not
     {editor format.names.for.label1 * "}{" *  editor format.names.for.label2 *}
     {org-short empty$ not
      {org-short * "}{" * org-short "u" change.case$ *}
      {organization empty$ not
       {organization * "}{" * organization "u" change.case$ *}
        {title empty$ not
         {title format.title.for.label *}
         {"nada}{NADA" *}
       if$}
      if$}
     if$}
    if$}
   if$
 "}{" *
 year empty$ 'skip$ {year *} if$
 'label :=
}

FUNCTION {calc.simple.label}
{ author empty$ not
    {author format.names.for.label1}
    {editor empty$ not
     {editor format.names.for.label1}
     {organization empty$ not
      {organization}
      {title empty$ not
       {title}
       {""}
      if$}
     if$}
    if$}
   if$
 year empty$ 'skip$ {" " * year *} if$
}

FUNCTION {output.bibitem}
{ newline$
  abnt.alf
    {"\bibitem[" write$
    calc.simple.label
    write$ "]{"}
    {"\bibitem{"}
  if$
  write$ cite$ write$ "}" write$ newline$
  abnt.refinfo #1 = {"\abntrefinfo" write$ label write$ newline$} 'skip$ if$
  "{" write$
  "" before.all 'output.state :=
}

FUNCTION {write.entry.verbatim}
{newline$}

FUNCTION {fin.entry}
{ add.period$ write$
  "}" write$
  write.entry.verbatim
}

FUNCTION {write.field.verbatim}
{
duplicate$ missing$
  {pop$ pop$} 
  {swap$
  "={" * swap$ * "}" *
  s swap$ * 's :=
  }
if$
}


FUNCTION {format.year}
{year empty$
   { "campo year vazio ou faltando em " cite$ *
    abnt.missing.year {" (vou colocar [s.d.])" *} 'skip$ if$ warning$
    abnt.missing.year {", " } {""} if$ bbl.missing.year *}
    {year n.dashify "year" bibinfo.check}
if$
}

FUNCTION {format.year.presented}
{year-presented empty$
   {""}
   {year-presented n.dashify}
if$
}

FUNCTION {format.address}
{address empty$
  { bbl.no.address}
  { address}
 if$
}

FUNCTION {format.thesis.type}
{ type$ "phdthesis" =
    {bbl.phdthesis}
    {type$ "mastersthesis" =
       {bbl.mthesis}
       {type$ "monography" =
          {bbl.monography}
	  {bbl.void}
        if$}
     if$}
  if$
  type empty$
    {type missing$
       {type$ "phdthesis" =
          { " (" bbl.phd * ")" *}
          {type$ "mastersthesis" = { " (" bbl.master * ")" *} {pop$ ""} if$}
        if$}
       {""}
     if$}
     {type$ "thesis" =
       {type}
       {" (" type  * ")" * "type" bibinfo.check}
      if$}
  if$
 *
}

FUNCTION {thesis}
{ output.bibitem
  format.authors "author" output.check new.block
  format.btitle "title" output.check new.block
  format.thesis.type output.nonnull
  school "school" bibinfo.warn output.dash
  address "address" bibinfo.check output
  format.date "year" output.check new.sentence
  format.note output new.block
  format.url
  fin.entry
}

FUNCTION {phdthesis}     { thesis }
FUNCTION {mastersthesis} { thesis }
FUNCTION {monography}    { thesis }

FUNCTION {article}
{ output.bibitem
  format.author.or.organization  new.sentence
  format.title "title" output.check new.block
  format.journal output
  format.volume output
  format.number output
  format.pages output
  format.date output
  format.issn output new.block
  format.note output new.block
  format.url
  fin.entry
}
FUNCTION {book}
{ output.bibitem
  format.author.or.editor.or.organization new.sentence
  format.btitle "title" output.check new.sentence
  format.edition output new.sentence
  format.publisher.address output
  format.year output new.sentence
  format.book.pages output
  format.note output
  format.isbn output new.sentence
  format.url
  fin.entry
}
FUNCTION {booklet}
{ output.bibitem
  format.author.or.editor.or.organization new.sentence
  format.btitle "title" output.check new.sentence
  format.edition output new.sentence
  address "address" bibinfo.check output
  format.year output new.sentence
  format.book.pages output new.sentence
  format.note output
  format.isbn output new.sentence
  format.url
  fin.entry
}

FUNCTION {hidden}
{ output.bibitem
  fin.entry}

FUNCTION {inbook}
{ output.bibitem
  format.author.or.organization new.sentence
  format.title "title" output.check new.block
  format.in.ed.booktitle
  format.edition output new.sentence
  format.publisher.address output
  format.year output new.sentence
  format.chapter.pages output new.sentence
  format.isbn output new.block
  format.note output new.block
  format.url
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.author.or.organization new.sentence
  format.title "title" output.check new.sentence
  format.in.ed.booktitle
  format.edition output new.sentence
  format.publisher.address output
  format.year output
  format.chapter.pages output new.sentence
  format.isbn output new.block
  format.note output new.block
  format.url
  fin.entry
}
FUNCTION {inproceedings}
{ output.bibitem
  format.author.or.organization new.sentence
  format.title "title" output.check  new.sentence
  format.in.ed.booktitle
  format.publisher.address output
  format.year output new.sentence
  format.pages output
  format.isbn output
  format.issn output new.sentence
  format.note output new.sentence
  format.url
  fin.entry
}
FUNCTION {conference} { inproceedings }
FUNCTION {manual}
{ output.bibitem
  format.author.or.editor.or.organization new.sentence
  format.btitle "title" output.check new.block
  format.edition output new.sentence
  format.address output
  format.year output new.sentence
  format.book.pages output new.sentence
  format.note output new.sentence
  format.url
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.author.or.editor.or.organization new.sentence
  format.btitle output new.sentence
  format.publisher.address.relax output
  format.year output new.sentence
  format.howpublished output  new.block
  format.note output  new.block
  format.url
  fin.entry
}

FUNCTION {default.type} { misc }
READ
FUNCTION {sortify}
{ purify$
  "l" change.case$
}
INTEGERS { len }
FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}
FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { format.single.name 't :=
      nameptr #1 >
        {
          nameptr #3
          #1 + =
          numnames #3
          > and
            { "others" 't :=
              #1 'namesleft := }
            'skip$
          if$
          "   "  *
          namesleft #1 = t "others" = and
            { "zzzzz" * }
            { t sortify * }
          if$
        }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}
FUNCTION {any.sort}
{ author empty$
  { organization empty$
    { editor empty$
      { title empty$
        { key empty$
          { "para ordenar, precisa dos campos author, organization, editor, title ou key in " cite$ * warning$ ""}
          { key sortify }
        if$}
        { title sortify}
      if$}
      { editor sort.format.names }
    if$}
    { organization sortify }
  if$}
  { author sort.format.names }
if$
}

FUNCTION {presort}
{ 
 any.sort
  "    " * year field.or.null sortify * "    " * title field.or.null
  sort.format.title * #1 entry.max$ substring$
  abnt.alf 'skip$ {pop$ ""} if$ 'sort.key$ :=
  year empty$ {""} {year} if$ 'year.label :=
  calc.extended.label
}


INTEGERS {key.pos value.pos}
FUNCTION {extract.cvs.key}
{ 's :=
  #1 'key.pos :=
  #0 'value.pos :=
  "" 't :=
  {value.pos #0 = value.pos #1 = or}
    { s key.pos #1 substring$ 'title.char :=
      value.pos #0 =
      { title.char ":" = {#1 'value.pos := key.pos #1 + 'key.pos :=} 'skip$ if$}
      { title.char "$" =
        {#2 'value.pos :=}
        {t title.char * 't :=}
	if$}
     if$
     key.pos #1 + 'key.pos :=
     }
  while$
  t
}

FUNCTION {set.default.abnt.variables}
{
  "$RCSfile: abnt-alf.bst,v $ " extract.cvs.key
  "abnt-alf.bst,v " = 'abnt.alf :=
  "$Revision: v-1.9.7 $" extract.cvs.key 'abnt.bst.revision :=
  #0 'abnt.and.type     :=
  "\emph" 'abnt.emphasize :=
  #0 'abnt.cite.style :=
  #0 'abnt.experimental :=
  abnt.doi.expand.to.url 'abnt.doi :=
  #3 'abnt.etal.cite :=
  #3 'abnt.etal.list :=
  "et al." 'abnt.etal.text :=
  #0 'abnt.full.initials :=
  #0 'abnt.last.names :=
  #1 'abnt.ldots.type :=
  #0 'abnt.missing.year :=
  #0 'repeated.author :=
  #0 'abnt.repeated.author.omit :=
  #0 'abnt.repeated.title.omit :=
  #0 'abnt.refinfo :=
  #0 'abnt.show.options :=
  #0 'abnt.thesis.year :=
  #0 'abnt.title.command :=
  #0 'abnt.url.package :=
  #0 'abnt.verbatim.entry :=
}
FUNCTION {set.default.iso.variables}
{ #0 'iso.abbreviation :=
  #0 'iso.author.punctuation :=
  #0 'iso.date.place :=
}


EXECUTE {set.default.abnt.variables}
EXECUTE {set.default.iso.variables}
FUNCTION {process.early.abnt.options}
{
 abnt-and-type missing$
 abnt-experimental missing$ and
 abnt-etal-cite missing$ and
 abnt-etal-text missing$ and
 abnt-ldots-type missing$ and
 abnt-cite-style missing$ and
 abnt-no-etal-label missing$ and
 abnt-title-command missing$ and
 'skip$ {call.type$} if$
}
ITERATE {process.early.abnt.options}
FUNCTION {abntreprintinfo}
{"\providecommand{\abntreprintinfo}[1]{%" write$ newline$
 " \citeonline{#1}}" write$ newline$}
EXECUTE {abntreprintinfo}
ITERATE {presort}
SORT
STRINGS { last.label next.extra}
INTEGERS { last.extra.num }
FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
}

FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  label extra.label * 'label :=
  year.label extra.label * 'year.label :=
  extra.label 'next.extra :=
  label "}" * 'label :=
}
ITERATE {forward.pass}
REVERSE {reverse.pass}
FUNCTION {begin.bib}
{
  abnt.alf
    {"\setlength{\labelsep}{0pt}" write$}
    'skip$
  if$
  "\begin{thebibliography}{}" write$ newline$
  preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  abnt.refinfo #1 =
  {"\providecommand{\abntrefinfo}[3]{}" write$ newline$}
  'skip$ if$
  "\providecommand{\abntbstabout}[1]{}" write$ newline$
  abnt.title.command
    {"\providecommand{\bibtextitlecommand}[2]{#2}" write$ newline$}
    'skip$ if$
  "\abntbstabout{" abnt.bst.revision * "}" * write$ newline$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
FUNCTION {process.entry}{call.type$}
ITERATE {process.entry}
FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
