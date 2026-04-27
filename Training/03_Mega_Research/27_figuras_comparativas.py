"""
27_figuras_comparativas.py
--------------------------
Gera figuras comparativas adicionais (F16-F24) pedidas pelo orientador,
a partir dos CSVs ja produzidos pelos scripts anteriores. Nenhum
experimento novo; so plotagem.

Saidas em Execution/results/figuras_tcc/comparativas/:
  F16_painel_f1_mestre.png        Bar chart F1 por (dataset x modelo) -- todos experimentos
  F17_forest_plot_cohens_d.png    Forest plot Cohen's d por (dataset x metrica)
  F18_heatmap_topo_sem_texto.png  Heatmap (modelo x variant x dataset) -- topologia sem texto
  F19_hop_importance.png          Bar chart fracao de massa hop1/hop2/hop3+ por (classe x tamanho)
  F20_bluesky_matrizes.png        Cross-tab textual x topologico no Bluesky (proxy de matriz confusao)
  F23_bluesky_modelo_x_feed.png   Heatmap score medio por (feed x modelo) no Bluesky
  F24_fluxograma_regra_dual.png   Diagrama conceitual da regra dual de combinacao

Notas:
- F21 (F1 vs tempo de treino) e F22 (calibration plot) requerem dados nao
  persistidos pelos scripts anteriores; ficam fora deste passo.
- F20 e proxy: usa concordancia (textual concorda x discorda topologico)
  ao inves de matriz confusao real (sem labels no Bluesky).

Uso:
  python 27_figuras_comparativas.py
"""

import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RAIZ        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = RAIZ / "Execution" / "results"
FASE4       = RESULTS_DIR / "fase4_benchmarks"
FIGS_DIR    = RESULTS_DIR / "figuras_tcc"
OUT_DIR     = FIGS_DIR / "comparativas"


def ler_csv(path: Path) -> list:
    if not path.exists(): return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agg(rows, group_keys, value_key):
    grupos = {}
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        try: v = float(r[value_key])
        except Exception: continue
        grupos.setdefault(k, []).append(v)
    return grupos


# ─── F16. Painel mestre F1 por (dataset x modelo) ─────────────────────
def fig_painel_mestre():
    rows13 = ler_csv(FASE4 / "benchmark_upfd_oficial" / "resultados.csv")
    rows14 = ler_csv(FASE4 / "topologia_sem_texto" / "resultados.csv")

    # Conjunto: 3 datasets x [LogReg-textual, RF-estrutural?, GCN-topo, GAT-topo, SAGE-topo, GCN-completo, GAT-completo, SAGE-completo]
    # Simplifica: por dataset, para cada arquitetura, mostra MELHOR F1 alcancado
    # (com texto vs sem texto) -- 2 barras por arquitetura.

    datasets = ["fnn", "upfd_politifact", "upfd_gossipcop"]

    # textual baseline: do script 13 (LogReg_root) ou outro
    textual = {}
    g13 = agg([r for r in rows13 if r["modelo"] == "LogReg_root"],
              ["dataset"], "f1_macro")
    for (ds,), vals in g13.items():
        textual[ds] = float(np.mean(vals))
    # FNN: usa do baseline_textual da fase 2 (so tem FNN, sem coluna dataset)
    rows_fnn_text = ler_csv(RESULTS_DIR / "fase2_baselines" / "baseline_textual" / "resultados.csv")
    if rows_fnn_text:
        vals = [float(r["f1_macro"]) for r in rows_fnn_text if r.get("modelo","") == "LogReg"]
        if vals:
            textual["fnn"] = float(np.mean(vals))

    # GNN com texto: script 13 (UPFD com bert/content)
    gnn_com_texto = {}
    for ds in ("upfd_politifact", "upfd_gossipcop"):
        for mod in ("GCN", "GAT", "SAGE"):
            vals = [float(r["f1_macro"]) for r in rows13
                    if r["dataset"] == ds.replace("upfd_", "") and r["modelo"] == mod]
            if vals:
                gnn_com_texto[(ds, mod)] = float(np.mean(vals))

    # GNN sem texto (variant B_estrutural): script 14
    gnn_sem_texto = {}
    for ds in datasets:
        for mod in ("GCN", "GAT", "SAGE"):
            vals = [float(r["f1_macro"]) for r in rows14
                    if r["dataset"] == ds and r["modelo"] == mod
                    and r["variant"] == "B_estrutural"]
            if vals:
                gnn_sem_texto[(ds, mod)] = float(np.mean(vals))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    ds_labels = {"fnn": "FakeNewsNet", "upfd_politifact": "UPFD-PolitiFact",
                 "upfd_gossipcop": "UPFD-GossipCop"}

    for ax, ds in zip(axes, datasets):
        labels, valores, cores = [], [], []
        # Barra 0: LogReg textual
        if ds in textual:
            labels.append("LogReg\n(texto)")
            valores.append(textual[ds])
            cores.append("#3498db")
        for mod in ("GCN", "GAT", "SAGE"):
            if (ds, mod) in gnn_sem_texto:
                labels.append(f"{mod}\n(topo)")
                valores.append(gnn_sem_texto[(ds, mod)])
                cores.append("#2ecc71")
            if (ds, mod) in gnn_com_texto:
                labels.append(f"{mod}\n(texto+topo)")
                valores.append(gnn_com_texto[(ds, mod)])
                cores.append("#e74c3c")

        x = np.arange(len(labels))
        ax.bar(x, valores, color=cores, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="chance" if ds == datasets[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_title(f"{ds_labels[ds]}", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(valores):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    axes[0].set_ylabel("F1-macro (media sobre seeds)")
    fig.suptitle("F1 por dataset x modelo: textual (azul), topologico puro (verde), com texto (vermelho)",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "F16_painel_f1_mestre.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


# ─── F17. Forest plot Cohen's d ───────────────────────────────────────
def fig_forest_cohens_d():
    rows = ler_csv(FIGS_DIR / "analise_estrutural" / "estatisticas.csv")
    if not rows:
        print("  [SKIP] F17: estatisticas.csv ausente"); return

    metricas = ["num_nodes", "branching_avg", "width_max", "depth_max"]
    datasets = ["fnn", "upfd_politifact", "upfd_gossipcop"]
    cores    = {"fnn": "#9b59b6", "upfd_politifact": "#e67e22",
                "upfd_gossipcop": "#16a085"}

    fig, ax = plt.subplots(figsize=(11, 6))
    ypos = 0
    yticks, ylabels = [], []
    for met in metricas:
        for ds in datasets:
            row = next((r for r in rows if r["dataset"] == ds and r["metrica"] == met), None)
            if not row: continue
            d = float(row["cohen_d"])
            ax.plot([0, d], [ypos, ypos], color=cores[ds], linewidth=2)
            ax.scatter([d], [ypos], s=80, color=cores[ds], edgecolor="black",
                       zorder=3, label=ds if met == metricas[0] else None)
            ax.text(d + 0.05 * np.sign(d) if d != 0 else 0.05,
                    ypos, f"{d:+.2f}", va="center", fontsize=8,
                    ha="left" if d >= 0 else "right")
            yticks.append(ypos); ylabels.append(f"{met} | {ds}")
            ypos += 1
        ypos += 0.5  # gap entre metricas

    # Faixas de Cohen
    for thr, txt in [(0.2, "pequeno"), (0.5, "medio"), (0.8, "grande")]:
        ax.axvline(thr,  color="gray", linestyle=":", alpha=0.5)
        ax.axvline(-thr, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Cohen's $d$ (fake - real)")
    ax.set_xlim(-2, 2)
    ax.set_title("Tamanho de efeito por metrica estrutural e dataset\n"
                 "Linhas pontilhadas: limiares de Cohen (0.2 / 0.5 / 0.8)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right")
    plt.tight_layout()
    out = OUT_DIR / "F17_forest_plot_cohens_d.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


# ─── F18. Heatmap topologia sem texto ─────────────────────────────────
def fig_heatmap_topo():
    rows = ler_csv(FASE4 / "topologia_sem_texto" / "resultados.csv")
    if not rows:
        print("  [SKIP] F18"); return

    datasets = ["fnn", "upfd_politifact", "upfd_gossipcop"]
    variants = ["A_isroot", "B_estrutural", "C_posicional"]
    archs    = ["GCN", "GAT", "SAGE"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, datasets):
        M = np.zeros((len(archs), len(variants)))
        for i, a in enumerate(archs):
            for j, v in enumerate(variants):
                vals = [float(r["f1_macro"]) for r in rows
                        if r["dataset"] == ds and r["modelo"] == a and r["variant"] == v]
                M[i, j] = float(np.mean(vals)) if vals else np.nan
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=0.85, aspect="auto")
        ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, fontsize=8, rotation=20)
        ax.set_yticks(range(len(archs))); ax.set_yticklabels(archs)
        ax.set_title(ds, fontsize=11)
        for i in range(len(archs)):
            for j in range(len(variants)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="black", fontsize=10, fontweight="bold")
    fig.suptitle("F1-macro do GNN aplicado APENAS a features estruturais\n"
                 "(sem texto). Verde = modelo aprende; Vermelho = colapsa.", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025)
    out = OUT_DIR / "F18_heatmap_topo_sem_texto.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


# ─── F19. Hop importance bar chart ────────────────────────────────────
def fig_hop_importance():
    out_g = FIGS_DIR / "gnnexplainer" / "gossipcop" / "hop_importance.csv"
    out_p = FIGS_DIR / "gnnexplainer" / "politifact" / "hop_importance.csv"
    if not out_g.exists():
        print("  [SKIP] F19: hop_importance.csv ausente"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, path, titulo in [(axes[0], out_g, "UPFD-GossipCop (SAGE F1=0.81)"),
                              (axes[1], out_p, "UPFD-PolitiFact (SAGE F1=0.33)")]:
        if not path.exists():
            ax.text(0.5, 0.5, "sem dados (modelo nao classifica REAL corretamente)",
                    ha="center", transform=ax.transAxes)
            ax.set_title(titulo); continue
        rows = ler_csv(path)
        # Estratos
        estratos = sorted({(r["classe"], r["tam"]) for r in rows})
        x = np.arange(len(estratos))
        h1 = []; h2 = []; h3 = []
        labels = []
        for c, t in estratos:
            sub = [r for r in rows if r["classe"] == c and r["tam"] == t]
            h1.append(np.mean([float(r["frac_hop1"]) for r in sub]))
            h2.append(np.mean([float(r["frac_hop2"]) for r in sub]))
            h3.append(np.mean([float(r["frac_hop3plus"]) for r in sub]))
            labels.append(f"{c}\n{t}\n(N={len(sub)})")
        # Stacked bar
        ax.bar(x, h1, label="hop1 (raiz->filhos)",     color="#e74c3c")
        ax.bar(x, h2, bottom=h1, label="hop2",         color="#f39c12")
        ax.bar(x, h3, bottom=np.array(h1)+np.array(h2), label="hop3+", color="#3498db")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Fracao de massa de importancia")
        ax.set_ylim(0, 1.05)
        ax.set_title(titulo, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("GNNExplainer: onde o SAGE concentra a importancia das arestas\n"
                 "Assimetria FAKE vs REAL (achado central da \\S 4.7)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "F19_hop_importance.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


# ─── F20. Cross-tab Bluesky textual x topologico (proxy de matriz) ────
def fig_bluesky_crosstab():
    path = FIGS_DIR / "concordancia_bluesky" / "amostra_5k.csv"
    if not path.exists():
        print("  [SKIP] F20"); return
    rows = ler_csv(path)
    # Matriz 2x2: pred_text x pred_topo
    M = np.zeros((2, 2), dtype=int)
    for r in rows:
        pt = int(r["pred_text"])
        pk = int(r["pred_topo"])
        # ordem: linha 0 = pred=0 (FAKE), linha 1 = pred=1 (REAL)
        M[pt, pk] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="Blues", aspect="equal")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred topo: FAKE", "pred topo: REAL"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["pred texto: FAKE", "pred texto: REAL"])
    def _n(x): return f"{int(x):,}".replace(",", ".")
    for i in range(2):
        for j in range(2):
            cor = "white" if M[i, j] > M.max() / 2 else "black"
            ax.text(j, i, _n(M[i,j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=cor)
    total = M.sum()
    diag = M[0, 0] + M[1, 1]
    ax.set_title(f"Cross-tab classificadores no Bluesky (N={_n(total)})\n"
                 f"Concordam: {_n(diag)} ({100*diag/total:.1f}%) | "
                 f"Discordam: {_n(total-diag)} ({100*(total-diag)/total:.1f}%)\n"
                 f"NB: nao e matriz de confusao (Bluesky sem labels);\n"
                 f"e cross-tab textual vs topologico.")
    fig.colorbar(im, ax=ax, fraction=0.045)
    out = OUT_DIR / "F20_bluesky_crosstab.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


# ─── F23. Bluesky scores por feed x modelo ────────────────────────────
def fig_bluesky_modelo_x_feed():
    path = FIGS_DIR / "concordancia_bluesky" / "amostra_5k.csv"
    if not path.exists():
        print("  [SKIP] F23"); return
    rows = ler_csv(path)
    feeds = sorted({r["feed"] for r in rows})
    M = np.zeros((len(feeds), 2))
    for i, fd in enumerate(feeds):
        sub = [r for r in rows if r["feed"] == fd]
        if sub:
            M[i, 0] = np.mean([float(r["score_text"]) for r in sub])
            M[i, 1] = np.mean([float(r["score_topo"]) for r in sub])

    fig, ax = plt.subplots(figsize=(8, max(6, 0.4 * len(feeds))))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LogReg-BERT\n(texto)", "RF estrutural\n(topo)"])
    ax.set_yticks(range(len(feeds))); ax.set_yticklabels(feeds, fontsize=9)
    for i in range(len(feeds)):
        for j in range(2):
            cor = "white" if (M[i, j] < 0.25 or M[i, j] > 0.75) else "black"
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=cor)
    ax.set_title("Score medio 'fake-like' por feed x modelo (Bluesky, N=5k amostrado)\n"
                 "Vermelho = mais fake-like; Azul = mais real-like")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    out = OUT_DIR / "F23_bluesky_modelo_x_feed.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


# ─── F24. Fluxograma da regra dual de combinacao ──────────────────────
def fig_fluxograma_dual():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis("off")

    def caixa(x, y, w, h, txt, cor="#ecf0f1", fontsize=10, fontweight="normal"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=cor, edgecolor="black", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, wrap=True)

    def seta(x1, y1, x2, y2, txt=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle="->", mutation_scale=20,
                                      linewidth=1.5, color="#34495e"))
        if txt:
            ax.text((x1 + x2)/2 + 0.2, (y1 + y2)/2, txt, fontsize=8,
                    style="italic", color="#34495e")

    # Inputs
    caixa(0.5, 5.5, 2.5, 1.5, "Texto do post\n(string)", cor="#aed6f1")
    caixa(0.5, 0.5, 2.5, 1.5, "Topologia\n(num_nodes,\ngrau_root)", cor="#a9dfbf")

    # Modelos
    caixa(4, 5.5, 2.5, 1.5, "LogReg-BERT\n(texto)", cor="#3498db", fontsize=10, fontweight="bold")
    caixa(4, 0.5, 2.5, 1.5, "RF estrutural\n(topo)", cor="#27ae60", fontsize=10, fontweight="bold")

    seta(3, 6.25, 4, 6.25)
    seta(3, 1.25, 4, 1.25)

    # Scores
    caixa(7.5, 5.5, 2, 1.5, "score_T\n[0,1]", cor="#d6eaf8")
    caixa(7.5, 0.5, 2, 1.5, "score_K\n[0,1]", cor="#d4efdf")
    seta(6.5, 6.25, 7.5, 6.25)
    seta(6.5, 1.25, 7.5, 1.25)

    # Decisao
    caixa(7, 3.3, 3, 1.4, "concordam?\n(pred_T == pred_K)", cor="#fcf3cf", fontsize=10)
    seta(8.5, 5.5, 8.5, 4.7)
    seta(8.5, 2,   8.5, 3.3)

    # Saidas
    caixa(10.5, 4.5, 1.4, 1, "0.5 * T\n+ 0.5 * K", cor="#abebc6", fontsize=9)
    caixa(10.5, 1.5, 1.4, 1, "0.8 * T\n+ 0.2 * K", cor="#f5b7b1", fontsize=9)
    seta(10, 4.2, 10.5, 4.8, "sim")
    seta(10, 3.7, 10.5, 1.9, "nao")

    # Final score
    caixa(10.5, 7, 1.4, 0.8, "score final\n+ flag confianca", cor="#fadbd8", fontsize=8, fontweight="bold")
    seta(11.2, 5.5, 11.2, 7)
    seta(11.2, 2.5, 11.2, 7)

    ax.set_title("Regra dual de combinacao textual x topologico\n"
                 "(derivada post-hoc do experimento da \\S 4.6 -- script 20)", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "F24_fluxograma_regra_dual.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  [OK] {out.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  Figuras comparativas adicionais (F16-F24)")
    print("=" * 70)
    fig_painel_mestre()
    fig_forest_cohens_d()
    fig_heatmap_topo()
    fig_hop_importance()
    fig_bluesky_crosstab()
    fig_bluesky_modelo_x_feed()
    fig_fluxograma_dual()
    print(f"\n[OK] Saidas em: {OUT_DIR}")
    print("\nFiguras NAO geradas (requerem dados nao persistidos):")
    print("  F21 (F1 vs tempo de treino) - tempos nao persistidos pelos scripts")
    print("  F22 (calibration plot) - predict_proba do test set nao persistido")


if __name__ == "__main__":
    main()
