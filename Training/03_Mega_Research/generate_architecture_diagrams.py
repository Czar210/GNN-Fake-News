"""
generate_architecture_diagrams.py  (v4 — colunas de nos + ativacoes explícitas)
--------------------------------------------------------------------------------
Estilo: colunas de circulos como diagramas educativos de NN (fiel a GNN).
Nivel de detalhe identico aos Correct:
  - Ativacoes (ReLU / ELU+Drop) aparecem como colunas proprias entre convs
  - Dropout(p=0.3) explicito para GAT entre cada GATConv
  - Subtitulos detalhados: concat=False, aggr=mean, W_l*h+W_r*h_N, sem ativacao
  - Conexoes 1-to-1 para ativacoes element-wise
  - Conexoes sparse (message passing) para camadas conv

Grafo exemplo (arvore de propagacao com 5 nos):
        0
       / \\
      1   2
     / \\
    3   4

Saida:  diagrams/{gcn,gat,sage}_architecture.{png,svg}
Uso:    python generate_architecture_diagrams.py
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Estrutura do grafo ────────────────────────────────────────────────────────
EDGES = [(0, 1), (0, 2), (1, 3), (1, 4)]
N = 5


def _neighbors(v):
    return {u for u, w in EDGES if w == v} | {u for u, w in EDGES if u == v}


MSG_SOURCES = {v: frozenset({v} | _neighbors(v)) for v in range(N)}


# ── Configuracoes dos modelos ─────────────────────────────────────────────────
# type: 'input' | 'conv' | 'act'
# cidx: indice na cmap (controla a cor do no)
# sub:  subtitulo tecnico (vazio para ativacoes simples)

MODELS = {
    'gcn': dict(
        title='GCN  ·  GCNClassifier',
        accent='#1A6FA8',
        cmap=['#D6EAF8', '#7FB3D3', '#2E86C1', '#1A5276'],
        act_color='#A9CCE3',
        pool_fc='#F39C12', pool_ec='#D68910',
        aggr='Agregação: soma normalizada por grau  (Kipf & Welling, 2017)',
        stages=[
            {'type': 'input', 'label': 'Entrada',    'sub': 'N × 768  (BERT)',                   'cidx': 0},
            {'type': 'conv',  'label': 'GCNConv 1',  'sub': '768→64  |  Σ norm. por grau',     'cidx': 1},
            {'type': 'act',   'label': 'ReLU',        'sub': '',                                   'cidx': 1},
            {'type': 'conv',  'label': 'GCNConv 2',  'sub': '64→64   |  Σ norm. por grau',     'cidx': 2},
            {'type': 'act',   'label': 'ReLU',        'sub': '',                                        'cidx': 2},
            {'type': 'conv',  'label': 'GCNConv 3',  'sub': '64→64   |  sem ativação',  'cidx': 3},
        ],
    ),
    'gat': dict(
        title='GAT  ·  GATClassifier',
        accent='#6C3483',
        cmap=['#E8DAEF', '#C39BD3', '#9B59B6', '#6C3483'],
        act_color='#D2B4DE',
        pool_fc='#E67E22', pool_ec='#CA6F1E',
        aggr='Agregação: atenção aprendida por softmax  (Veličković et al., 2018)',
        stages=[
            {'type': 'input', 'label': 'Entrada',        'sub': 'N × 768  (BERT)',                        'cidx': 0},
            {'type': 'conv',  'label': 'GATConv 1',      'sub': '768→64  |  4 heads  |  concat=False  |  atenção softmax', 'cidx': 1},
            {'type': 'act',   'label': 'ELU + Dropout',  'sub': 'p = 0.3',                                                        'cidx': 1},
            {'type': 'conv',  'label': 'GATConv 2',      'sub': '64→64   |  4 heads  |  concat=False  |  atenção softmax', 'cidx': 2},
            {'type': 'act',   'label': 'ELU + Dropout',  'sub': 'p = 0.3',                                     'cidx': 2},
            {'type': 'conv',  'label': 'GATConv 3',      'sub': '64→64   |  1 head  |  consolidação', 'cidx': 3},
        ],
    ),
    'sage': dict(
        title='GraphSAGE  ·  SAGEClassifier',
        accent='#B7770D',
        cmap=['#FDEBD0', '#F0B27A', '#E67E22', '#B7770D'],
        act_color='#FAD7A0',
        pool_fc='#2E86C1', pool_ec='#1A5276',
        aggr='Agregação: W_l·h_v + W_r·mean(N(v))  (Hamilton et al., 2017)',
        stages=[
            {'type': 'input', 'label': 'Entrada',    'sub': 'N × 768  (BERT)',                               'cidx': 0},
            {'type': 'conv',  'label': 'SAGEConv 1', 'sub': '768→64  |  aggr=mean  |  W_l·h + W_r·h_N  (indutivo)', 'cidx': 1},
            {'type': 'act',   'label': 'ReLU',        'sub': '',                                                   'cidx': 1},
            {'type': 'conv',  'label': 'SAGEConv 2', 'sub': '64→64   |  aggr=mean  |  W_l·h + W_r·h_N',             'cidx': 2},
            {'type': 'act',   'label': 'ReLU',        'sub': '',                                                   'cidx': 2},
            {'type': 'conv',  'label': 'SAGEConv 3', 'sub': '64→64   |  sem ativação',             'cidx': 3},
        ],
    ),
}


# ── Gerador principal ─────────────────────────────────────────────────────────

def make_diagram(model_key: str) -> plt.Figure:
    cfg    = MODELS[model_key]
    accent = cfg['accent']
    stages = [dict(s) for s in cfg['stages']]   # copia mutavel

    # ── Geometria ─────────────────────────────────────────────────────────────
    ns      = 1.0    # espaco vertical entre nos
    nr_conv = 0.24   # raio dos nos conv/input
    nr_act  = 0.17   # raio dos nos de ativacao (menores)
    gap_cc  = 2.4    # distancia entre duas colunas conv consecutivas
    gap_ca  = 1.15   # distancia conv -> ativacao
    gap_ac  = 1.25   # distancia ativacao -> prox conv

    ys      = [(N - 1) / 2.0 * ns - i * ns for i in range(N)]
    y_top   = ys[0]
    y_bot   = ys[-1]

    # Posicoes X de cada estagio
    x = 1.5
    for i, s in enumerate(stages):
        s['x'] = x
        if i < len(stages) - 1:
            nxt = stages[i + 1]
            if nxt['type'] == 'act':
                x += gap_ca
            elif s['type'] == 'act':
                x += gap_ac
            else:
                x += gap_cc

    last_x  = stages[-1]['x']
    pool_x  = last_x + 1.8
    pool_y  = 0.0
    pool_r  = 0.30

    drop_bx = pool_x + pool_r + 0.10     # borda esquerda da caixa dropout
    drop_w  = 0.54
    drop_cx = drop_bx + drop_w / 2.0     # centro da caixa dropout

    lin_bx  = drop_bx + drop_w + 0.12   # borda esquerda da caixa linear
    lin_w   = 0.60
    lin_cx  = lin_bx + lin_w / 2.0      # centro da caixa linear

    out_x   = lin_bx + lin_w + 1.35
    out_ys  = [0.42, -0.42]
    out_r   = 0.22

    total_w = out_x + 1.35
    y_lo    = y_bot - 2.0
    y_hi    = y_top + 1.65

    fig, ax = plt.subplots(figsize=(total_w, y_hi - y_lo))
    ax.set_xlim(0, total_w)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    # ── Titulo ────────────────────────────────────────────────────────────────
    ax.text(total_w / 2, y_top + 1.50, cfg['title'],
            ha='center', va='center', fontsize=13,
            fontweight='bold', color=accent)
    ax.plot([0.3, total_w - 0.3], [y_top + 1.23, y_top + 1.23],
            color=accent, lw=1.5, alpha=0.40)
    ax.text(total_w / 2, y_top + 0.99, cfg['aggr'],
            ha='center', va='center', fontsize=7.8,
            color='#7F8C8D', style='italic')

    # ── Arestas do grafo dentro das colunas conv/input (tracejado) ────────────
    for s in stages:
        if s['type'] != 'act':
            for u, v in EDGES:
                ax.plot([s['x'], s['x']], [ys[u], ys[v]],
                        color='#AABCC8', lw=1.0, alpha=0.60,
                        linestyle='--', zorder=1, dash_capstyle='round')

    # ── Conexoes entre estagios ───────────────────────────────────────────────
    for i in range(len(stages) - 1):
        s1, s2  = stages[i], stages[i + 1]
        nr1     = nr_act if s1['type'] == 'act' else nr_conv
        nr2     = nr_act if s2['type'] == 'act' else nr_conv
        x1, x2  = s1['x'], s2['x']

        if s2['type'] == 'act':
            # Ativacao e element-wise: conexoes 1-to-1
            for j in range(N):
                ax.plot([x1 + nr1, x2 - nr2], [ys[j], ys[j]],
                        color=cfg['act_color'], lw=0.7, alpha=0.55, zorder=1)
        else:
            # Conv: message passing esparso
            for v in range(N):
                for u in MSG_SOURCES[v]:
                    ax.plot([x1 + nr1, x2 - nr2], [ys[u], ys[v]],
                            color=accent, lw=0.55, alpha=0.20, zorder=1)

    # ── Conexoes ultima coluna -> pool ────────────────────────────────────────
    for i in range(N):
        ax.plot([last_x + nr_conv, pool_x - pool_r], [ys[i], pool_y],
                color=cfg['pool_fc'], lw=0.65, alpha=0.40, zorder=1)

    # ── Conexao dropout -> linear ─────────────────────────────────────────────
    ax.plot([drop_bx + drop_w + 0.03, lin_bx - 0.03], [pool_y, pool_y],
            color='#BDC3C7', lw=1.0, alpha=0.85, zorder=1)

    # ── Conexoes linear -> neuronios de saida ─────────────────────────────────
    for oy in out_ys:
        ax.plot([lin_bx + lin_w + 0.05, out_x - out_r], [pool_y, oy],
                color='#BDC3C7', lw=1.0, alpha=0.85, zorder=1)

    # ── Nos dos estagios ──────────────────────────────────────────────────────
    for s in stages:
        is_act = s['type'] == 'act'
        nr     = nr_act if is_act else nr_conv
        fc     = cfg['act_color'] if is_act else cfg['cmap'][s['cidx']]
        ec     = '#5D6D7E'  if is_act else '#2C3E50'
        lw     = 1.1        if is_act else 1.5
        for i in range(N):
            ax.add_patch(Circle((s['x'], ys[i]), nr,
                                facecolor=fc, edgecolor=ec,
                                linewidth=lw, zorder=3))

    # ── No de pooling ─────────────────────────────────────────────────────────
    ax.add_patch(Circle((pool_x, pool_y), pool_r,
                        facecolor=cfg['pool_fc'], edgecolor=cfg['pool_ec'],
                        linewidth=2.2, zorder=4))
    ax.text(pool_x, pool_y + 0.09, 'h', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=5, fontstyle='italic')
    ax.text(pool_x, pool_y - 0.11, 'graph', ha='center', va='center',
            fontsize=6.5, color='white', zorder=5)

    # ── Caixa de dropout (p=0.5) ──────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (drop_bx, pool_y - 0.24), drop_w, 0.48,
        boxstyle='round,pad=0.04',
        facecolor='#FDEDEC', edgecolor='#E74C3C', linewidth=1.4, zorder=3,
    ))
    ax.text(drop_cx, pool_y + 0.07, 'Drop',  ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#C0392B', zorder=4)
    ax.text(drop_cx, pool_y - 0.09, 'p=0.5', ha='center', va='center',
            fontsize=7.0, color='#C0392B', zorder=4)

    # ── Caixa linear (classificador) ─────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (lin_bx, pool_y - 0.24), lin_w, 0.48,
        boxstyle='round,pad=0.04',
        facecolor='#EAD7F7', edgecolor='#7D3C98', linewidth=1.4, zorder=3,
    ))
    ax.text(lin_cx, pool_y + 0.07, 'Linear', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#6C3483', zorder=4)
    ax.text(lin_cx, pool_y - 0.09, '64 → 2', ha='center', va='center',
            fontsize=7.0, color='#6C3483', zorder=4)

    # ── Neuronios de saida ────────────────────────────────────────────────────
    for oy, lbl, col in zip(out_ys, ['Real', 'Fake'], ['#27AE60', '#E74C3C']):
        ax.add_patch(Circle((out_x, oy), out_r,
                            facecolor=col, edgecolor='#2C3E50',
                            linewidth=1.5, zorder=4))
        ax.text(out_x, oy, lbl, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=5)

    # ── Rotulos abaixo de cada coluna ─────────────────────────────────────────
    label_y = y_bot - 0.54
    for s in stages:
        is_act = s['type'] == 'act'
        ax.text(s['x'], label_y, s['label'],
                ha='center', va='top',
                fontsize=7.8 if is_act else 8.5,
                fontweight='normal' if is_act else 'bold',
                color='#5D6D7E' if is_act else '#1A252F')
        if s['sub']:
            ax.text(s['x'], label_y - 0.26, s['sub'],
                    ha='center', va='top', fontsize=6.5,
                    color='#7F8C8D', style='italic')

    ax.text(pool_x,  label_y, 'Global\nMean Pool',
            ha='center', va='top', fontsize=8.5, fontweight='bold', color='#1A252F')
    ax.text(pool_x,  label_y - 0.26, '[N × 64] → [B × 64]',
            ha='center', va='top', fontsize=6.5, color='#7F8C8D', style='italic')
    ax.text(pool_x,  label_y - 0.46, '(embedding do grafo)',
            ha='center', va='top', fontsize=6.5, color='#7F8C8D', style='italic')

    ax.text(drop_cx, label_y, 'Dropout',
            ha='center', va='top', fontsize=7.8, color='#5D6D7E')
    ax.text(drop_cx, label_y - 0.26, 'p = 0.5',
            ha='center', va='top', fontsize=6.5, color='#7F8C8D', style='italic')

    ax.text(lin_cx,  label_y, 'Linear',
            ha='center', va='top', fontsize=8.5, fontweight='bold', color='#6C3483')
    ax.text(lin_cx,  label_y - 0.26, 'Classificador',
            ha='center', va='top', fontsize=6.5, color='#7F8C8D', style='italic')

    ax.text(out_x,   label_y, 'Saída',
            ha='center', va='top', fontsize=8.5, fontweight='bold', color='#1A252F')
    ax.text(out_x,   label_y - 0.26, 'logits [B × 2]',
            ha='center', va='top', fontsize=6.5, color='#7F8C8D', style='italic')

    # ── Legenda ───────────────────────────────────────────────────────────────
    leg_y = y_bot - 1.50
    lx = 0.55
    ax.plot([lx, lx + 0.48], [leg_y, leg_y], color=accent, lw=1.8, alpha=0.65)
    ax.text(lx + 0.63, leg_y,
            'Message passing (sparse — apenas vizinhos 1-hop)',
            va='center', fontsize=7.5, color='#444')
    lx += 7.8
    ax.plot([lx, lx + 0.48], [leg_y, leg_y],
            color='#AABCC8', lw=1.4, linestyle='--')
    ax.text(lx + 0.63, leg_y,
            'Arestas do grafo (intra-coluna)',
            va='center', fontsize=7.5, color='#444')

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


# ── Execucao ──────────────────────────────────────────────────────────────────

for key in ('gcn', 'gat', 'sage'):
    print(f"Gerando {key.upper()} ...", end=' ')
    fig  = make_diagram(key)
    base = os.path.join(OUTPUT_DIR, f'{key}_architecture')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    fig.savefig(base + '.svg', format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'OK -> diagrams/{key}_architecture.{{png,svg}}')

print('\nPronto! Arquivos em:', OUTPUT_DIR)
