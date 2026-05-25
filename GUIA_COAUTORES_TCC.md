# Guia para co-autores — como escrever o TCC sem se perder

> Para André, César e Enzo. Leia antes de começar a escrever sua parte.

## Onde estamos

O pipeline experimental está **pronto** — 30 scripts, todos os resultados
gerados, 32 figuras e 13 tabelas consolidadas. O que falta é **escrever a
prosa** dos 6 capítulos. Os capítulos já existem como "esqueleto":
estrutura, figuras no lugar certo, e comentários dizendo o que escrever.

## A regra número 1 — o `FATOS_TCC.md`

Todo número que você escrever (F1, contagem de grafos, Cohen's d, %, etc.)
**tem de vir do arquivo `FATOS_TCC.md`** (na raiz do repositório).

- Não escreva número de memória nem do rascunho/TCC antigo — o rascunho
  antigo tem erros já mapeados.
- Se o número não está no FATOS, não use até verificar.
- O FATOS marca cada fato com um status: ✅ (confiável), ⚠️ (tem conflito,
  resolver antes), ❓ (não verificado).

## Ordem de escrita

Escreva **nesta ordem** — não comece pela introdução:

1. Capítulo 3 — Metodologia
2. Capítulo 4 — Resultados
3. Capítulo 5 — Aplicação
4. Capítulo 6 — Conclusão
5. Capítulo 2 — Fundamentação
6. Capítulo 1 — Introdução

Caps 1 e 2 são os últimos porque é mais fácil escrever a introdução quando
você já sabe como a história termina.

## Como dividir o trabalho entre os três

Para não haver colisão (duas pessoas editando o mesmo arquivo ao mesmo tempo):

- Divida **por capítulo**, não por seção. Cada um pega um capítulo inteiro.
- Avise no grupo qual capítulo está editando.
- Quem terminar primeiro pega o próximo da fila.
- Caps 3 e 4 podem ser escritos em paralelo (são independentes).
- Use o Overleaf (edição colaborativa) ou combine commits no Git para não
  sobrescrever o trabalho do outro.

## Como escrever cada seção

Abra o arquivo `.tex` do capítulo (em `capitulos/`). Cada seção tem
comentários que guiam:

- `% TESE:` — a ideia central da seção. Seu texto tem de sustentar isso.
- `% ANTES:` — o que dizer antes da figura/tabela.
- `% DEPOIS:` — o que dizer depois (explicar os números).
- `% TRANSICAO:` — como emendar na próxima seção.

Escreva a prosa e **apague o comentário** correspondente. Quando todos os
comentários da seção sumiram, a seção está pronta.

## O que NÃO fazer

- Não invente números (ver regra 1).
- Não mexa nos scripts nem nos resultados — estão congelados.
- Não descreva a "regra dual" (pesos 0.8/0.2) como decisão planejada — ela
  foi derivada **depois** de ver os resultados. O scaffolding explica isso.
- Não repita o mesmo conteúdo em duas seções.
- Não escreva `tweet_ids` solto no LaTeX — underscores quebram a
  compilação. Use `\texttt{tweet\_ids}`.

## Discrepâncias já conhecidas (não reintroduza)

O `FATOS_TCC.md` lista 6 discrepâncias (D1–D6) entre o rascunho antigo e os
dados reais. As que mais importam:

- **D1**: SAGE-PolitiFact sem texto é **0,33** — não 0,55.
- **D2**: Cohen's d do PolitiFact é **< 0,5** (não "< 0,2").
- **D3**: a amostra de concordância do Bluesky é **3.728** — não "5k".

Leia a seção de discrepâncias do FATOS antes de escrever os caps 4 e 5.

## Usando uma IA para ajudar

Pode usar uma IA (Claude) para revisar o que você escreveu. Para ela
entender o projeto, **comece a conversa apontando o arquivo
`PROMPT_IA_TCC.md`** (na raiz) — ele orienta a IA sobre onde olhar e quais
regras seguir.

Mas: **a IA também erra.** Todo número que ela sugerir, confira no
`FATOS_TCC.md`. A IA ajuda a estruturar, revisar e apontar problemas; a
verdade dos números está no FATOS.

## Material pronto para usar

| Arquivo | Para quê |
|---|---|
| `FATOS_TCC.md` | A fonte da verdade dos números, por capítulo |
| `estrutura_tcc_v2.md` | O plano detalhado de cada capítulo e seção |
| `Material/GNN_TCC_atualizado/` | O projeto LaTeX completo (Overleaf) |
| `Photos/pacote_tcc_base_completa.zip` | TCC inteiro com scaffolding, pronto pra subir no Overleaf do zero — tem `INDICE.md` e `COMO_USAR.md` dentro |
| `Training/03_Mega_Research/PIPELINE.md` | O que cada script de experimento faz |
| `PROMPT_IA_TCC.md` | Prompt para orientar uma IA assistente |

## Resumo em uma frase

Escreva na ordem 3→4→5→6→2→1, um capítulo por pessoa, sempre puxando os
números do `FATOS_TCC.md`, apagando os comentários-guia conforme avança.
