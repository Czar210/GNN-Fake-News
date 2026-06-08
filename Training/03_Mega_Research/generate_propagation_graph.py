"""
generate_propagation_graph.py
------------------------------
Gera figura 2x2 de arvores de propagacao reais para a secao 2.2:
  PolitiFact fake | PolitiFact real
  GossipCop  fake | GossipCop  real

Nota UPFD: y=0 -> Fake, y=1 -> Real

Saida: diagrams/propagation_example.{pdf,png}
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from torch_geometric.datasets import UPFD
from torch_geometric.utils import to_networkx

# ── Caminhos ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, '..', '..')
MATERIAL   = os.path.normpath(os.path.join(REPO_ROOT, 'Material'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'diagrams')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Paleta ────────────────────────────────────────────────────────────────────
C_FAKE_NODE  = '#E74C3C'
C_FAKE_ROOT  = '#922B21'
C_FAKE_EDGE  = '#F1948A'
C_REAL_NODE  = '#2980B9'
C_REAL_ROOT  = '#1A5276'
C_REAL_EDGE  = '#85C1E9'


def pick_graph(dataset, label, min_nodes=10, max_nodes=30):
    """Retorna o grafo com mais nos no intervalo [min, max] e label dado."""
    candidates = [g for g in dataset
                  if g.y.item() == label and min_nodes <= g.num_nodes <= max_nodes]
    if not candidates:
        # fallback: qualquer tamanho
        candidates = [g for g in dataset if g.y.item() == label]
    candidates.sort(key=lambda g: g.num_nodes, reverse=True)
    return candidates[0]


def draw_tree(ax, data, title, color_node, color_root, color_edge):
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Raiz = no de maior grau
    degrees = dict(G.degree())
    root = max(degrees, key=degrees.get)

    # Layout hierarquico BFS
    try:
        pos = nx.bfs_layout(G, root, align='horizontal')
        pos = {n: (x, -y) for n, (x, y) in pos.items()}
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.0)

    n   = G.number_of_nodes()
    ns  = max(180, 1600 // n)

    node_colors = [color_root if v == root else color_node for v in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=color_edge, width=1.2,
                           alpha=0.65, arrows=False)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=ns,
                           linewidths=0.6, edgecolors='white')
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={root: 'raiz'},
                            font_size=6.5, font_color='white',
                            font_weight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.text(0.5, -0.03,
            f'{n} nos  |  {G.number_of_edges()} arestas',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, color='#666666')
    ax.axis('off')


# ── Carrega datasets ──────────────────────────────────────────────────────────
print("Carregando UPFD PolitiFact...", end=' ')
pf_train = UPFD(root=MATERIAL, name='politifact', feature='profile', split='train')
print(f"{len(pf_train)} grafos")

print("Carregando UPFD GossipCop...", end=' ')
gc_train = UPFD(root=MATERIAL, name='gossipcop', feature='profile', split='train')
print(f"{len(gc_train)} grafos")

# ── Seleciona exemplos ────────────────────────────────────────────────────────
pf_fake = pick_graph(pf_train, label=0)
pf_real = pick_graph(pf_train, label=1)
gc_fake = pick_graph(gc_train, label=0)
gc_real = pick_graph(gc_train, label=1)

print(f"PolitiFact fake: {pf_fake.num_nodes} nos | real: {pf_real.num_nodes} nos")
print(f"GossipCop  fake: {gc_fake.num_nodes} nos | real: {gc_real.num_nodes} nos")

# ── Figura 2x2 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                         facecolor='white',
                         gridspec_kw={'wspace': 0.08, 'hspace': 0.25})

draw_tree(axes[0, 0], pf_fake,
          'PolitiFact — Noticia Falsa',
          C_FAKE_NODE, C_FAKE_ROOT, C_FAKE_EDGE)

draw_tree(axes[0, 1], pf_real,
          'PolitiFact — Noticia Verdadeira',
          C_REAL_NODE, C_REAL_ROOT, C_REAL_EDGE)

draw_tree(axes[1, 0], gc_fake,
          'GossipCop — Noticia Falsa',
          C_FAKE_NODE, C_FAKE_ROOT, C_FAKE_EDGE)

draw_tree(axes[1, 1], gc_real,
          'GossipCop — Noticia Verdadeira',
          C_REAL_NODE, C_REAL_ROOT, C_REAL_EDGE)

# Legenda
p_raiz  = mpatches.Patch(color='#555555',
                         label='No raiz (publicacao original)')
p_falso = mpatches.Patch(color=C_FAKE_NODE,
                         label='No de repercussao — Falsa')
p_real  = mpatches.Patch(color=C_REAL_NODE,
                         label='No de repercussao — Verdadeira')
fig.legend(handles=[p_raiz, p_falso, p_real],
           loc='lower center', ncol=3,
           fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    'Arvores de propagacao reais — UPFD (PolitiFact e GossipCop)',
    fontsize=13, y=1.01, color='#222222')

plt.tight_layout()

# ── Salva ─────────────────────────────────────────────────────────────────────
base = os.path.join(OUTPUT_DIR, 'propagation_example')
fig.savefig(base + '.pdf', format='pdf', bbox_inches='tight', facecolor='white')
fig.savefig(base + '.png', dpi=300,     bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nOK -> diagrams/propagation_example.{{pdf,png}}")
