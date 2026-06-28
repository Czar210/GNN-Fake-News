# Guia de Design dos Slides — Defesa do TCC

Como os slides devem **parecer**: paleta, tipografia, grid, templates e o mapa de quais figuras do TCC entram em cada slide. O objetivo é um visual sóbrio, "de paper", que não compete com a fala — mas com um truque: **a cor conta a história sozinha**.

---

## 0. O princípio que organiza tudo: a cor é semântica

O próprio TCC já usa um código de cores consistente nas figuras (4.1, 4.15, fluxograma 1.1). **Adotem o mesmo código nos slides** — a banca vê a figura do PDF e o slide falando a mesma língua visual. É o detalhe que faz parecer um trabalho coeso, não um PowerPoint genérico.

| Cor | Significado no TCC | Uso nos slides |
|---|---|---|
| 🔵 **Azul** | sinal/modelo **textual** (BERT, LogReg) | entrada de dados, texto, "o caminho clássico" |
| 🟢 **Verde** | **topológico puro** / "o modelo **aprende**" | a aposta do trabalho, resultados positivos, ✅ |
| 🟠 **Laranja/Vermelho** | modelo **completo** / "**colapsa**" | resultado negativo, alerta, ❌, PolitiFact |
| 🟡 **Amarelo** | a **regra dual** (decisão) | a combinação texto×topologia, o "ponto de decisão" |

> Regra prática: **verde = funciona, vermelho/laranja = falha.** Não use vermelho pra "erro de design" — use pra "o caso que colapsa". Isso transforma o slide do achado bilateral (slide 8) numa imagem que se explica antes de você abrir a boca.

---

## 1. Paleta de cores

Paleta sóbria sobre fundo claro (recomendado para banca acadêmica — projeta melhor em qualquer sala e imprime bem).

```
FUNDO        #FFFFFF  (branco)  ou  #FAFAF7 (off-white quente, mais "papel")
TEXTO        #1A1A2E  (quase-preto azulado, menos duro que #000)
TÍTULOS      #14213D  (azul-noite)

— Paleta semântica (a que conta a história) —
AZUL textual     #2563EB   (forte)   /  #93C5FD (claro, fundos)
VERDE aprende    #16A34A   (forte)   /  #86EFAC (claro)
LARANJA colapsa  #EA580C   (forte)   /  #FDBA74 (claro)
VERMELHO alerta  #DC2626   (só para "abaixo da chance" / ❌)
AMARELO dual     #EAB308   (usar com parcimônia, só na regra dual)

— Neutros de apoio —
CINZA linha      #6B7280   (eixos, linhas de chance, texto secundário)
CINZA fundo box  #F3F4F6   (cartões, tabelas zebradas)
```

**Contraste:** texto escuro sobre fundo claro sempre. Se optarem por tema escuro (fundo `#0F172A`), inverta: texto `#E2E8F0`, e as cores semânticas ficam *mais vibrantes*. Tema escuro é mais arriscado em projetor de sala — só se testarem na sala antes.

**Regra dos 60-30-10:** ~60% neutro (fundo/texto), ~30% azul-noite (estrutura, títulos), ~10% cor semântica (só onde importa). Não pinte o slide inteiro.

---

## 2. Tipografia

A fonte do TCC é Computer Modern (serifada, cara de LaTeX). Nos slides, contraste **sans para a estrutura** + **serifa só para a tese** (callback ao documento) + **mono para código**.

| Papel | Fonte | Alternativas | Tamanho |
|---|---|---|---|
| **Títulos** | **Inter** (ou Source Sans 3) | Helvetica Neue, Archivo | 32–40 pt, semibold |
| **Corpo / bullets** | **Inter** regular | Source Sans, Roboto | 22–26 pt (nunca < 20) |
| **A tese (slide 9), citações** | **uma serifa** — Lora ou Source Serif | Georgia, PT Serif | 26–30 pt, itálico |
| **Código / nomes de feature** | **JetBrains Mono** | Fira Code, Consolas | 18–22 pt |

**Por que mono para código:** termos como `[is_root, grau_norm]`, `num_nodes`, `branching_avg`, `ttest_rel` **sempre** em monoespaçada. Sinaliza "isto é técnico/literal" e evita o LaTeX-trauma de underscore. Nunca escreva esses nomes em fonte de corpo.

**Por que serifa na tese:** o slide 9 projeta a frase da tese central. Pôr ela em serifa itálica (a "voz do documento") cria um momento solene e a destaca do resto. É o único lugar com serifa — por isso funciona.

**Regras:** no máximo **2 famílias** por slide (mono não conta). Não use mais de 2 pesos. Nada de Comic Sans, nada de fontes "tech" futuristas — isto é ciência, não startup.

---

## 3. Grid e layout

- **Proporção 16:9.** Margens generosas: ~8% de cada lado. Slide respira.
- **Hierarquia fixa:** título no topo (sempre na mesma posição), conteúdo no miolo, rodapé fino com nº do slide + "PUC-SP 2026" + título curto do TCC.
- **Máximo 5 bullets por slide.** Idealmente 3. Cada bullet ≤ 1 linha e meia. Se precisa de mais, são dois slides.
- **Uma ideia = um slide.** O slide 8 (achado bilateral) não disputa atenção com mais nada.
- **Figura grande, legenda pequena.** Quando o slide é uma figura do TCC, ela ocupa ~70% da área; o resto é título + 1 frase de leitura.
- **Alinhamento à esquerda** para texto (mais legível que centralizado em bullets). Títulos podem centralizar nos divisores de ato.

### Slides divisores de ato (opcional, ajudam o ritmo)
Três slides de "respiro" entre os atos, fundo azul-noite cheio, só o número e o nome do ato em branco grande:
`ATO 1 · O problema` / `ATO 2 · O método e o achado` / `ATO 3 · Porquê e aplicação`.

---

## 4. Templates de slide (5 tipos)

**A. Capa / encerramento** — título grande centralizado, autores, faixa de cor fina (azul→verde→laranja, o arco do trabalho), logo PUC no canto.

**B. Slide de conteúdo padrão** — título à esquerda no topo, 3–5 bullets, espaço à direita para um ícone ou mini-figura. 70% texto / 30% visual.

**C. Slide de figura** — título curto no topo, figura do TCC ocupando o miolo, **uma** frase de leitura embaixo ("verde sobe, vermelho despenca"). Sem bullets.

**D. Slide de citação/tese** (só o slide 9) — fundo off-white, a frase da tese em serifa itálica grande, centralizada, com uma barra de cor à esquerda. Atribuição pequena: "Tese central — §4.6".

**E. Slide comparativo "isto vs aquilo"** (slides 3 e 8) — split vertical: esquerda azul (texto/GossipCop), direita laranja-vermelha (alerta/PolitiFact). O contraste de cor *é* o conteúdo.

---

## 5. Mapa de figuras do TCC → slides

Vocês já têm 32 figuras prontas no repositório (`Imagens/`, identificadores F1–F35). Reusem — não recriem. As essenciais:

| Slide | Figura do TCC | Por que / como tratar |
|---|---|---|
| 5 (dados) | **Fig 2.1** (árvores de propagação fake/real) | mostra o objeto "grafo de propagação" de cara |
| 6 (arquiteturas) | **Figs 2.2 / 2.3 / 2.4** (GCN/GAT/SAGE) | lado a lado, simplificadas; pode redesenhar como 3 ícones limpos |
| 8 (achado central) | **Fig 4.1** ou par de barras | **recortar** só GossipCop (verde) vs PolitiFact (vermelho). É o slide-pico. |
| 9 (tese) | **Fig 4.3** (forest plot Cohen's *d*) | manter as linhas de limiar 0,2/0,5/0,8 visíveis |
| 10 (multi-hop) | **Fig 4.13** ou **4.14** | a 4.14 (barras verdes do gap) é mais limpa pra slide |
| 11 (concordância) | **Fig 4.4** (scatter) | ou simplificar pra uma matriz 2×2 com os 4 números |
| 12 (RQ3 idioma) | **Fig 4.2** (distribuições PT/EN/DE) | destacar a sobreposição PT≈EN |
| 13 (ferramenta) | **Fig 1.1** (fluxograma) + print da web | o fluxograma já usa o código de cores — perfeito |
| 14 (painel mestre) | **Fig 4.15** | usar inteira, é a síntese; talvez o slide mais importante depois do 9 |

> **Atenção de legibilidade:** figura de paper tem fonte pequena e muita informação. Para slide: **aumente a fonte dos eixos/legendas**, **remova o que não vai comentar**, e garanta que a cor lê de longe. Se a figura original tem 6 barras e você só fala de 2, mostre 2. O repositório tem o script `27b_repintar_figuras_orientador.py` que já aplica a paleta semântica — use as versões repintadas.

---

## 6. Ícones e elementos visuais

- **Ícones de linha** (estilo Lucide / Feather), monocromáticos na cor do contexto. Nada de clipart 3D.
- ✅ / ❌ só nas cores semânticas (verde/vermelho), com moderação.
- **Selos/chips** para os 4 pilares de rigor (slide 7) e as virtudes (slide 15): retângulos arredondados, fundo claro da cor, texto escuro.
- **Setas** para fluxo (texto → modelo → score). Finas, cinza ou na cor da etapa.
- Evitem: sombras pesadas, gradientes berrantes, transições animadas de slide (fade simples no máximo), emojis decorativos fora do par ✅/❌.

---

## 7. Ferramenta recomendada para montar

Três caminhos, do mais rápido ao mais "cara de TCC":

1. **PowerPoint / Google Slides** — mais rápido, controle total, fácil de dividir entre os três. Recomendado se o tempo é curto. Montem um *slide master* com a paleta e fontes deste guia para padronizar.
2. **Marp** (Markdown → slides) — o `APRESENTACAO_SLIDES.md` está estruturado de forma que dá pra converter quase direto. Vantagem: versionável no Git, consistência automática. Bom pra quem prefere texto a arrastar caixinha. (`marp` CLI ou extensão do VS Code.)
3. **Beamer (LaTeX)** — máxima coerência com o TCC (mesma fonte, mesmo clima), mas o mais lento de iterar. Só se sobrar tempo e alguém dominar Beamer.

> Sugestão pragmática: **Google Slides** com um master bem feito. Os três editam ao mesmo tempo, e dá pra ensaiar com o cronômetro embutido (modo apresentador).

---

## 8. Checklist visual antes de fechar

- [ ] Mesma fonte, mesmos tamanhos, mesma posição de título em **todos** os slides (use master/tema).
- [ ] Código sempre em mono (`num_nodes`, `[is_root, grau_norm]`).
- [ ] Verde = funciona, vermelho = colapsa — **consistente** do slide 1 ao 18.
- [ ] Nenhum slide com mais de 5 bullets nem fonte < 20 pt.
- [ ] Figuras recoloridas/recortadas, legíveis a 3 metros (teste: afaste-se da tela).
- [ ] Tese (slide 9) em serifa, em destaque — o pico visual.
- [ ] Rodapé discreto com nº de slide (ajuda a banca a referenciar na arguição).
- [ ] Testar o projetor **na sala** antes — cor e contraste mudam muito de tela pra projetor.
- [ ] Exportar um **PDF de backup** dos slides (caso o software falhe no dia).
