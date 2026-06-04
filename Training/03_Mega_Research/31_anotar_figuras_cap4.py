"""
31_anotar_figuras_cap4.py

Sobrepõe anotações (círculo + texto em negrito + seta) nas figuras-chave
do Capítulo 4 do TCC, destacando o ponto mais importante de cada uma.

As anotações ficam em uma camada por cima da figura original; o PNG
original NÃO é modificado. As versões anotadas são salvas em
Material/GNN_TCC_atualizado/Imagens/anotadas/<nome>.png.

Para usar no .tex, basta trocar o caminho:
    Imagens/F31_f1_por_depth.png
    -> Imagens/anotadas/F31_f1_por_depth.png

Uso:
    python 31_anotar_figuras_cap4.py
    python 31_anotar_figuras_cap4.py --solo F31   # gera apenas uma
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Diretórios
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "Material" / "GNN_TCC_atualizado" / "Imagens"
DST_DIR = SRC_DIR / "anotadas"
DST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
COLOR_CIRCLE = (220, 30, 30, 255)          # vermelho vivo
COLOR_TEXT_BG = (255, 255, 224, 235)       # amarelo-pálido translúcido
COLOR_TEXT_BORDER = (180, 130, 0, 255)     # mostarda
COLOR_TEXT_FG = (20, 20, 20, 255)          # quase preto
ARROW_W = 5
CIRCLE_W = 5

# Tenta uma fonte sistema; cai para default se não achar
def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",      # Arial Bold (Win)
        "C:/Windows/Fonts/calibrib.ttf",     # Calibri Bold
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",   # mac
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # linux
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Núcleo de desenho
# ---------------------------------------------------------------------------
def annotate(
    src: Path,
    *,
    circle_cx_frac: float,
    circle_cy_frac: float,
    circle_r_frac: float,
    text: str,
    text_box_corner: str = "TL",      # TL, TR, BL, BR
    text_box_w_frac: float = 0.32,
    font_size: int | None = None,
    arrow: bool = True,
) -> Path:
    """
    Desenha círculo+texto sobre a imagem em src e salva em DST_DIR.

    Frações são relativas às dimensões da imagem (0..1).
    text_box_corner indica em qual canto vai a legenda.
    """
    img = Image.open(src).convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ---- círculo ----
    cx = circle_cx_frac * W
    cy = circle_cy_frac * H
    r = circle_r_frac * min(W, H)
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        outline=COLOR_CIRCLE,
        width=CIRCLE_W,
    )

    # ---- caixa de texto ----
    if font_size is None:
        font_size = max(18, int(min(W, H) * 0.025))
    font = _load_font(font_size)

    box_w = int(W * text_box_w_frac)
    pad = max(8, font_size // 3)

    # quebra de linha simples por largura
    def _wrap(words: list[str]) -> list[str]:
        lines: list[str] = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=font) <= box_w - 2 * pad:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    lines = _wrap(text.split())
    line_h = font_size + 4
    box_h = pad * 2 + line_h * len(lines)

    if text_box_corner == "TL":
        bx, by = int(W * 0.02), int(H * 0.03)
    elif text_box_corner == "TR":
        bx, by = W - box_w - int(W * 0.02), int(H * 0.03)
    elif text_box_corner == "BL":
        bx, by = int(W * 0.02), H - box_h - int(H * 0.03)
    elif text_box_corner == "BR":
        bx, by = W - box_w - int(W * 0.02), H - box_h - int(H * 0.03)
    else:
        raise ValueError(f"text_box_corner inválido: {text_box_corner}")

    # fundo da caixa
    draw.rectangle(
        (bx, by, bx + box_w, by + box_h),
        fill=COLOR_TEXT_BG,
        outline=COLOR_TEXT_BORDER,
        width=3,
    )
    for i, line in enumerate(lines):
        draw.text(
            (bx + pad, by + pad + i * line_h),
            line,
            font=font,
            fill=COLOR_TEXT_FG,
        )

    # ---- seta da caixa ao círculo ----
    if arrow:
        # ponto da caixa mais próximo do círculo
        bx_mid = bx + box_w // 2
        by_mid = by + box_h // 2
        # ponto na borda do círculo mais próximo da caixa
        import math

        dx, dy = cx - bx_mid, cy - by_mid
        d = max(1.0, math.hypot(dx, dy))
        end_x = cx - dx / d * r
        end_y = cy - dy / d * r
        start_x, start_y = bx_mid, by_mid

        draw.line(
            (start_x, start_y, end_x, end_y),
            fill=COLOR_CIRCLE,
            width=ARROW_W,
        )
        # cabeça de seta simples
        ah = ARROW_W * 3
        ux, uy = (end_x - start_x) / d, (end_y - start_y) / d
        nx, ny = -uy, ux
        p1 = (end_x - ux * ah + nx * ah * 0.6, end_y - uy * ah + ny * ah * 0.6)
        p2 = (end_x - ux * ah - nx * ah * 0.6, end_y - uy * ah - ny * ah * 0.6)
        draw.polygon([ (end_x, end_y), p1, p2 ], fill=COLOR_CIRCLE)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    dst = DST_DIR / src.name
    out.save(dst, optimize=True)
    return dst


# ---------------------------------------------------------------------------
# Especificação das 15 anotações
# ---------------------------------------------------------------------------
# Coordenadas em frações 0..1 (x da esquerda, y do topo).
# Onde possível, baseiam-se em heurísticas conservadoras do layout matplotlib
# padrão. Cesar deve revisar visualmente e ajustar se algum círculo cair fora.
ANNOTATIONS: dict[str, dict] = {
    "F18_heatmap_topo_sem_texto.png": dict(
        circle_cx_frac=0.78, circle_cy_frac=0.82, circle_r_frac=0.07,
        text="SAGE em UPFD-GossipCop com apenas duas features estruturais (B_estrutural): F1≈0,81 — o achado central do trabalho.",
        text_box_corner="TL",
    ),
    "F17_forest_plot_cohens_d.png": dict(
        circle_cx_frac=0.85, circle_cy_frac=0.40, circle_r_frac=0.07,
        text="GossipCop branching factor: d = +1,53 — único ponto acima do limiar de efeito grande (0,8).",
        text_box_corner="BL",
    ),
    "F14_rq3_multilingual_dist.png": dict(
        circle_cx_frac=0.30, circle_cy_frac=0.32, circle_r_frac=0.10,
        text="PT vs EN praticamente sobrepostos (d = −0,057, desprezível). DE difere por efeito médio.",
        text_box_corner="TR",
    ),
    "F3_textual_vs_topo_scatter.png": dict(
        circle_cx_frac=0.28, circle_cy_frac=0.28, circle_r_frac=0.12,
        text="Quadrante de discordância: textual diz REAL, topológico diz FAKE. Subgrupo onde o topológico inverte sistematicamente.",
        text_box_corner="BR",
    ),
    "F19_hop_importance.png": dict(
        circle_cx_frac=0.20, circle_cy_frac=0.38, circle_r_frac=0.08,
        text="Para FAKE, massa de importância está saturada em 1-hop (0,97–1,00): SAGE só olha grau da raiz.",
        text_box_corner="TR",
    ),
    "F25_distribuicao_log.png": dict(
        circle_cx_frac=0.28, circle_cy_frac=0.70, circle_r_frac=0.08,
        text="Cauda observada acima do ajuste lognormal: KS rejeita o fit (p < 1e-11) — heavy-tail mais agressivo.",
        text_box_corner="TR",
    ),
    "F26_lowess_acerto.png": dict(
        circle_cx_frac=0.20, circle_cy_frac=0.50, circle_r_frac=0.07,
        text="Cruzamento das curvas LOWESS: SAGE sobe com num_nodes; RF desce. Não existe um modelo globalmente superior.",
        text_box_corner="BR",
    ),
    "F27_f1_por_tercil.png": dict(
        circle_cx_frac=0.27, circle_cy_frac=0.55, circle_r_frac=0.07,
        text="SAGE cresce monotonicamente T1 → T3 em num_nodes; RF decresce. Gap concentra-se em grafos grandes.",
        text_box_corner="TR",
    ),
    "F29_outliers_acc.png": dict(
        circle_cx_frac=0.18, circle_cy_frac=0.85, circle_r_frac=0.09,
        text="fake_contido (N=31): SAGE acerta ZERO. real_viral (N=60): SAGE acerta 43% (abaixo da chance).",
        text_box_corner="TR",
    ),
    "F30_outliers_composicao.png": dict(
        circle_cx_frac=0.17, circle_cy_frac=0.50, circle_r_frac=0.12,
        text="real_viral: 0% de fakes verdadeiras, mas modelos predizem 57–77% como FAKE — assinatura da inversão distributiva.",
        text_box_corner="TR",
    ),
    "F28_distribuicao_profundidade.png": dict(
        circle_cx_frac=0.30, circle_cy_frac=0.55, circle_r_frac=0.08,
        text="30% dos grafos têm depth ≥ 3 (cauda até 9 hops). UPFD oficial NÃO é tudo estrela.",
        text_box_corner="TR",
    ),
    "F31_f1_por_depth.png": dict(
        circle_cx_frac=0.85, circle_cy_frac=0.28, circle_r_frac=0.10,
        text="Gap SAGE − RF cresce monotonicamente: +0,029 em depth=2, +0,383 em depth=7. Sinal multi-hop genuíno.",
        text_box_corner="BL",
    ),
    "F32_gap_subsets.png": dict(
        circle_cx_frac=0.80, circle_cy_frac=0.32, circle_r_frac=0.13,
        text="IC 95% bootstrap em depth ≥ 5 não cruza zero. Gap é 3,5× o agregado.",
        text_box_corner="BL",
    ),
    "F16_painel_f1_mestre.png": dict(
        circle_cx_frac=0.63, circle_cy_frac=0.72, circle_r_frac=0.05,
        text="SAGE topológico puro em UPFD-PolitiFact: F1 ≈ 0,33, abaixo da chance — colapso do mecanismo topológico onde Cohen's d < 0,5.",
        text_box_corner="TL",
    ),
    # F24 (fluxograma) e F33-35 (matrizes de confusão) não recebem anotação:
    # já são diagramas com leitura direta.
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--solo",
        help="Gera apenas a figura indicada (ex.: F31). Caso omitido, gera todas.",
    )
    args = p.parse_args()

    targets = list(ANNOTATIONS.items())
    if args.solo:
        targets = [(n, s) for n, s in targets if n.startswith(args.solo + "_")]
        if not targets:
            print(f"Nenhuma figura corresponde ao prefixo {args.solo!r}.")
            return 2

    done = 0
    for name, spec in targets:
        src = SRC_DIR / name
        if not src.exists():
            print(f"[skip] {name}: arquivo fonte não encontrado")
            continue
        dst = annotate(src, **spec)
        rel = dst.relative_to(REPO_ROOT)
        print(f"[ok]   {name}  ->  {rel}")
        done += 1

    print()
    print(f"Anotadas {done}/{len(targets)} figuras em {DST_DIR.relative_to(REPO_ROOT)}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
