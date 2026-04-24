"""
15_analise_estrutural.py
------------------------
Compara propriedades topologicas entre os 3 datasets para entender por que
SAGE com features estruturais puras atinge 0.81 F1m no GossipCop mas so
0.55 no FakeNewsNet e ~0.42 no UPFD-PolitiFact.

Para cada dataset, computa por GRAFO:
  - num_nodes (tamanho)
  - num_edges
  - profundidade max da arvore (BFS desde a raiz)
  - largura max (max nos no mesmo nivel)
  - branching factor medio (filhos por no nao-folha)

Depois compara fake vs real:
  - histograma sobreposto
  - t-test em cada metrica
  - eta-squared (effect size)

Saidas em Execution/results/figuras_tcc/analise_estrutural/:
  estatisticas.csv          -- por (dataset, metrica): media_fake, media_real, t, p
  fig_distribuicoes.png     -- 4 metricas x 3 datasets, fake vs real sobrepostos
  fig_separabilidade.png    -- effect sizes (Cohen's d) por metrica/dataset

Tambem exporta amostras visualizaveis:
  amostra_<dataset>_FAKE_idxN.html   -- pyvis grafo interativo (fake)
  amostra_<dataset>_REAL_idxN.html   -- pyvis grafo interativo (real)

Uso:
  python 15_analise_estrutural.py
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import torch
from scipy import stats
from torch_geometric.datasets import UPFD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_folds import carregar_grafos as carregar_fnn

RAIZ     = Path(__file__).resolve().parent.parent.parent
MAT_DIR  = RAIZ / "Material"
OUT_DIR  = RAIZ / "Execution" / "results" / "figuras_tcc" / "analise_estrutural"


def metricas_grafo(g) -> dict:
    n = int(g.num_nodes)
    m = int(g.num_edges)
    if m == 0 or n == 1:
        return {"num_nodes": n, "num_edges": m,
                "depth_max": 0, "width_max": 1, "branching_avg": 0.0}

    # Adjacencia (tratamos como nao-direcionado pra BFS)
    adj = defaultdict(list)
    for s, t in g.edge_index.t().tolist():
        adj[s].append(t)
        adj[t].append(s)

    # BFS desde no 0 (raiz)
    profundidade = {0: 0}
    fila = deque([0])
    while fila:
        u = fila.popleft()
        for v in adj[u]:
            if v not in profundidade:
                profundidade[v] = profundidade[u] + 1
                fila.append(v)

    if not profundidade:
        return {"num_nodes": n, "num_edges": m, "depth_max": 0, "width_max": 1, "branching_avg": 0.0}

    depth_max = max(profundidade.values())
    largura = defaultdict(int)
    for d in profundidade.values():
        largura[d] += 1
    width_max = max(largura.values())

    # Branching factor: media de filhos por no nao-folha
    # Como temos arvore raiz->filhos: filhos = vizinhos com profundidade > propria
    filhos_count = []
    for u in profundidade:
        nf = sum(1 for v in adj[u] if profundidade.get(v, -1) > profundidade[u])
        if nf > 0:
            filhos_count.append(nf)
    branching = float(np.mean(filhos_count)) if filhos_count else 0.0

    return {"num_nodes": n, "num_edges": m,
            "depth_max": depth_max, "width_max": width_max,
            "branching_avg": branching}


def coletar_metricas(grafos, label_fake=0):
    rows = []
    for g in grafos:
        m = metricas_grafo(g)
        m["label"] = "fake" if g.y.item() == label_fake else "real"
        rows.append(m)
    return rows


def cohen_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    pooled = np.sqrt(((a.std()**2 * (len(a)-1)) + (b.std()**2 * (len(b)-1))) /
                     (len(a) + len(b) - 2)) if len(a) > 1 and len(b) > 1 else 1e-9
    return (a.mean() - b.mean()) / max(pooled, 1e-9)


def comparar_classes(linhas, metricas):
    out = {}
    for met in metricas:
        f = [r[met] for r in linhas if r["label"]=="fake"]
        r = [r[met] for r in linhas if r["label"]=="real"]
        t, p = stats.ttest_ind(f, r, equal_var=False)
        d = cohen_d(f, r)
        out[met] = {"media_fake": np.mean(f), "media_real": np.mean(r),
                    "std_fake": np.std(f), "std_real": np.std(r),
                    "t": float(t), "p": float(p), "cohen_d": float(d),
                    "n_fake": len(f), "n_real": len(r)}
    return out


def carregar_upfd(name):
    train = list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="train"))
    val   = list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="val"))
    test  = list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="test"))
    return train + val + test


def exportar_amostra_html(g, nome_arq, titulo):
    try:
        from pyvis.network import Network
    except ImportError:
        print("  [WARN] pyvis nao instalado -- pulando exportacao HTML.")
        return False
    net = Network(height="500px", width="100%", directed=False, notebook=False)
    n = g.num_nodes
    for i in range(n):
        cor = "#E07B54" if i == 0 else "#5BAD72"
        rotulo = "RAIZ" if i == 0 else f"u{i}"
        net.add_node(i, label=rotulo, color=cor, size=20 if i==0 else 10)
    for s, t in g.edge_index.t().tolist():
        net.add_edge(int(s), int(t))
    net.barnes_hut(spring_length=80)
    net.set_options('{"interaction": {"hover": true}}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / nome_arq
    try:
        net.write_html(str(out), notebook=False, open_browser=False)
        return True
    except Exception as e:
        print(f"  [WARN] erro pyvis: {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Analise Estrutural -- Por que GossipCop funciona com topologia pura?")
    print("=" * 72)

    print("\n[1/3] Carregando 3 datasets...")
    datasets = {
        "fnn":              carregar_fnn(),
        "upfd_politifact":  carregar_upfd("politifact"),
        "upfd_gossipcop":   carregar_upfd("gossipcop"),
    }
    for n, gs in datasets.items():
        fk = sum(1 for g in gs if g.y.item()==0)
        print(f"  {n}: {len(gs)} grafos | fake={fk} ({100*fk/len(gs):.1f}%)")

    print("\n[2/3] Computando metricas estruturais por grafo...")
    coletas = {n: coletar_metricas(gs) for n, gs in datasets.items()}

    metricas = ["num_nodes", "num_edges", "depth_max", "width_max", "branching_avg"]
    comparacoes = {n: comparar_classes(coletas[n], metricas) for n in datasets}

    # CSV consolidado
    csv_path = OUT_DIR / "estatisticas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "metrica", "media_fake", "std_fake", "media_real", "std_real",
                    "t", "p", "cohen_d", "n_fake", "n_real"])
        for ds, comp in comparacoes.items():
            for met, vals in comp.items():
                w.writerow([ds, met,
                            f"{vals['media_fake']:.4f}", f"{vals['std_fake']:.4f}",
                            f"{vals['media_real']:.4f}", f"{vals['std_real']:.4f}",
                            f"{vals['t']:.4f}", f"{vals['p']:.6f}",
                            f"{vals['cohen_d']:.4f}",
                            vals['n_fake'], vals['n_real']])
    print(f"  [OK] {csv_path.name}")

    # Print sumario
    print("\n  --- Sumario (valores significativos com p<0.05 destacados com **) ---")
    for ds, comp in comparacoes.items():
        print(f"\n  [{ds}]")
        for met, v in comp.items():
            sig = "**" if v["p"] < 0.05 else "  "
            print(f"    {sig} {met:<14} fake={v['media_fake']:7.2f}±{v['std_fake']:6.2f}  "
                  f"real={v['media_real']:7.2f}±{v['std_real']:6.2f}  "
                  f"d={v['cohen_d']:+.3f}  p={v['p']:.4f}")

    # Plot 1: distribuicoes (num_nodes, depth, width, branching) x 3 datasets
    print("\n[3/3] Gerando figuras...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    plot_metricas = ["num_nodes", "depth_max", "width_max", "branching_avg"]
    for i, ds in enumerate(datasets):
        rows = coletas[ds]
        for j, met in enumerate(plot_metricas):
            ax = axes[i][j]
            f = [r[met] for r in rows if r["label"]=="fake"]
            r = [r[met] for r in rows if r["label"]=="real"]
            # Bins log para num_nodes (FNN/Goss tem grafos enormes)
            if met == "num_nodes":
                lo, hi = max(min(min(f), min(r)), 1), max(max(f), max(r))
                bins = np.geomspace(lo, hi, 30)
                ax.set_xscale("log")
            else:
                bins = 30
            ax.hist(f, bins=bins, alpha=0.55, label=f"fake (n={len(f)})", color="#E07B54")
            ax.hist(r, bins=bins, alpha=0.55, label=f"real (n={len(r)})", color="#5BAD72")
            ax.set_title(f"{ds}\n{met}", fontsize=10)
            ax.legend(fontsize=8)
            d = comparacoes[ds][met]["cohen_d"]
            p = comparacoes[ds][met]["p"]
            ax.text(0.98, 0.95, f"d={d:+.2f}\np={p:.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.tight_layout()
    p1 = OUT_DIR / "fig_distribuicoes.png"
    plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  [OK] {p1.name}")

    # Plot 2: effect sizes barplot
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.25
    pos = np.arange(len(metricas))
    cores = {"fnn": "#4F86C6", "upfd_politifact": "#E07B54", "upfd_gossipcop": "#5BAD72"}
    for i, (ds, comp) in enumerate(comparacoes.items()):
        ds_d = [comp[m]["cohen_d"] for m in metricas]
        ax.bar(pos + i*width, ds_d, width, label=ds, color=cores[ds], alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.2,  color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(-0.2, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(pos + width)
    ax.set_xticklabels(metricas, rotation=15)
    ax.set_ylabel("Cohen's d (fake vs real)")
    ax.set_title("Separabilidade fake vs real por metrica estrutural\n"
                 "(|d|>=0.2 = pequeno; >=0.5 = medio; >=0.8 = grande)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = OUT_DIR / "fig_separabilidade.png"
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  [OK] {p2.name}")

    # Exportar amostras HTML (1 fake + 1 real por dataset)
    print("\n  Exportando grafos amostra (HTML interativo)...")
    for ds, gs in datasets.items():
        # pegar amostras de tamanho mediano
        gs_fake = sorted([g for g in gs if g.y.item()==0], key=lambda x: x.num_nodes)
        gs_real = sorted([g for g in gs if g.y.item()==1], key=lambda x: x.num_nodes)
        if gs_fake:
            g = gs_fake[len(gs_fake)//2]
            exportar_amostra_html(g, f"amostra_{ds}_FAKE_n{g.num_nodes}.html",
                                  f"{ds} FAKE (n={g.num_nodes})")
        if gs_real:
            g = gs_real[len(gs_real)//2]
            exportar_amostra_html(g, f"amostra_{ds}_REAL_n{g.num_nodes}.html",
                                  f"{ds} REAL (n={g.num_nodes})")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
