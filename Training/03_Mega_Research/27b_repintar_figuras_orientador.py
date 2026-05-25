"""
27b_repintar_figuras_orientador.py
-----------------------------------
Regenera 3 figuras com paletas semanticamente claras, segundo feedback do
orientador (resumo: "usar verde/vermelho conforme bom/ruim; guiar a atencao
do leitor").

Regenera SOMENTE estas tres (deixando o restante do 27 intacto):
  - F16_painel_f1_mestre.png        (paleta categorica + destaque colapso)
  - F20_bluesky_crosstab.png        (diagonal verde / off-diagonal vermelho)
  - F23_bluesky_modelo_x_feed.png   (RdYlGn_r: verde=real-like, vermelho=fake-like)

Saida sobrescreve em Material/GNN_TCC_atualizado/Imagens/ (alem do dir de
resultados, para o TCC pegar a versao nova diretamente).

Uso:
  python 27b_repintar_figuras_orientador.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAIZ      = Path(__file__).resolve().parent.parent.parent
RESULTS   = RAIZ / "Execution" / "results"
FIGS_DIR  = RESULTS / "figuras_tcc"
FASE4     = RESULTS / "fase4_benchmarks"
OUT_RES   = FIGS_DIR / "comparativas"
OUT_TCC   = RAIZ / "Material" / "GNN_TCC_atualizado" / "Imagens"


def ler_csv(p: Path):
    if not p.exists(): return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agg(rows, by_keys, val_key):
    out = {}
    for r in rows:
        k = tuple(r[k_] for k_ in by_keys)
        try: out.setdefault(k, []).append(float(r[val_key]))
        except Exception: pass
    return out


# ============================================================
# F16 painel mestre — paleta categorica + borda vermelha em colapso
# ============================================================
def fig_painel_mestre():
    rows13 = ler_csv(FASE4 / "benchmark_upfd_oficial" / "resultados.csv")
    rows14 = ler_csv(FASE4 / "topologia_sem_texto" / "resultados.csv")

    datasets = ["fnn", "upfd_politifact", "upfd_gossipcop"]

    textual = {}
    g13 = agg([r for r in rows13 if r["modelo"] == "LogReg_root"], ["dataset"], "f1_macro")
    for (ds,), vals in g13.items():
        textual[ds] = float(np.mean(vals))
    rows_fnn_text = ler_csv(RESULTS / "fase2_baselines" / "baseline_textual" / "resultados.csv")
    if rows_fnn_text:
        vals = [float(r["f1_macro"]) for r in rows_fnn_text if r.get("modelo","") == "LogReg"]
        if vals: textual["fnn"] = float(np.mean(vals))

    gnn_com_texto = {}
    for ds in ("upfd_politifact", "upfd_gossipcop"):
        for mod in ("GCN", "GAT", "SAGE"):
            vals = [float(r["f1_macro"]) for r in rows13
                    if r["dataset"] == ds.replace("upfd_", "") and r["modelo"] == mod]
            if vals: gnn_com_texto[(ds, mod)] = float(np.mean(vals))

    gnn_sem_texto = {}
    for ds in datasets:
        for mod in ("GCN", "GAT", "SAGE"):
            vals = [float(r["f1_macro"]) for r in rows14
                    if r["dataset"] == ds and r["modelo"] == mod
                    and r["variant"] == "B_estrutural"]
            if vals: gnn_sem_texto[(ds, mod)] = float(np.mean(vals))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    ds_labels = {"fnn": "FakeNewsNet", "upfd_politifact": "UPFD-PolitiFact",
                 "upfd_gossipcop": "UPFD-GossipCop"}

    # Paleta semantica clara
    C_TEXT = "#1f77b4"   # azul: baseline textual (referencia)
    C_TOPO = "#2ca02c"   # verde: topologico puro (o achado)
    C_FULL = "#ff7f0e"   # laranja: com texto (referencia da literatura)
    THR_COLAPSO = 0.55   # F1 abaixo disso = borda vermelha grossa (sinaliza colapso)

    for ax, ds in zip(axes, datasets):
        labels, valores, cores = [], [], []
        if ds in textual:
            labels.append("LogReg\n(texto)"); valores.append(textual[ds]); cores.append(C_TEXT)
        for mod in ("GCN", "GAT", "SAGE"):
            if (ds, mod) in gnn_sem_texto:
                labels.append(f"{mod}\n(topo)"); valores.append(gnn_sem_texto[(ds, mod)]); cores.append(C_TOPO)
            if (ds, mod) in gnn_com_texto:
                labels.append(f"{mod}\n(texto+topo)"); valores.append(gnn_com_texto[(ds, mod)]); cores.append(C_FULL)

        x = np.arange(len(labels))
        bars = ax.bar(x, valores, color=cores, edgecolor="black", linewidth=0.8)
        # Destaque vermelho em barras de colapso
        for b, v in zip(bars, valores):
            if v < THR_COLAPSO:
                b.set_edgecolor("#d62728")
                b.set_linewidth(2.5)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(0.02, 0.51, "chance", color="gray", fontsize=7, transform=ax.get_yaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_title(ds_labels[ds], fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(valores):
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold" if v >= 0.80 else "normal")

    axes[0].set_ylabel("F1-macro (media sobre seeds)")
    # Legenda manual
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_TEXT, edgecolor="black", label="Baseline textual (LogReg-BERT)"),
        Patch(facecolor=C_TOPO, edgecolor="black", label="Topologico puro (sem texto)"),
        Patch(facecolor=C_FULL, edgecolor="black", label="Com texto (referencia da literatura)"),
        Patch(facecolor="white", edgecolor="#d62728", linewidth=2.5,
              label=f"Borda vermelha = colapso (F1 < {THR_COLAPSO})"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Painel mestre: F1 por dataset x modelo", fontsize=12, y=1.07)
    plt.tight_layout()
    for out in (OUT_RES / "F16_painel_f1_mestre.png", OUT_TCC / "F16_painel_f1_mestre.png"):
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [OK] F16_painel_f1_mestre.png")


# ============================================================
# F20 cross-tab Bluesky — diagonal verde, off-diagonal vermelho
# ============================================================
def fig_bluesky_crosstab():
    p = FIGS_DIR / "concordancia_bluesky" / "amostra_5k.csv"
    if not p.exists():
        print("  [SKIP] F20: amostra_5k.csv ausente"); return
    rows = ler_csv(p)
    if not rows:
        print("  [SKIP] F20: csv vazio"); return

    # Cross-tab: pred_text x pred_topo
    mat = np.zeros((2, 2), dtype=int)  # linhas=text, cols=topo. 0=FAKE, 1=REAL
    for r in rows:
        try:
            t = int(float(r["pred_text"])); k = int(float(r["pred_topo"]))
            if t in (0, 1) and k in (0, 1):
                mat[t, k] += 1
        except Exception: pass

    N = mat.sum()
    concord = mat[0, 0] + mat[1, 1]
    discord = mat[0, 1] + mat[1, 0]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # Cor por celula: diagonal (concordam) = verde claro, off-diagonal (discordam) = vermelho claro
    cores = np.array([["#a8e6a8", "#f4a8a8"], ["#f4a8a8", "#a8e6a8"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, 1-i-0.5), 1, 1,
                                        facecolor=cores[i, j], edgecolor="black", linewidth=0.8))
            # Texto: contagem grande + porcentagem
            ax.text(j, 1-i+0.05, f"{mat[i,j]:,}".replace(",", "."),
                    ha="center", va="center", fontsize=20, fontweight="bold", color="#222")
            ax.text(j, 1-i-0.18, f"{100*mat[i,j]/N:.1f}%",
                    ha="center", va="center", fontsize=10, color="#444")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["FAKE", "REAL"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["REAL", "FAKE"], fontsize=11)  # invertido pq linha 0 fica em cima
    ax.set_xlabel("Predito pelo TOPOLOGICO (RF estrutural)", fontsize=11)
    ax.set_ylabel("Predito pelo TEXTUAL (LogReg-BERT)", fontsize=11)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"Cross-tab dos classificadores no Bluesky (N={N:,})\n"
                 f"Concordam: {concord:,} ({100*concord/N:.1f}%) | "
                 f"Discordam: {discord:,} ({100*discord/N:.1f}%)\n"
                 f"Verde = concordancia; vermelho = discordancia. NAO e matriz de confusao supervisionada.",
                 fontsize=10)
    ax.invert_yaxis()
    # Re-ajusta y labels apos invert
    ax.set_yticks([0, 1]); ax.set_yticklabels(["FAKE", "REAL"], fontsize=11)
    plt.tight_layout()
    for out in (OUT_RES / "F20_bluesky_crosstab.png", OUT_TCC / "F20_bluesky_crosstab.png"):
        plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [OK] F20_bluesky_crosstab.png")


# ============================================================
# F23 heatmap modelo x feed — RdYlGn_r (verde=real-like, vermelho=fake-like)
# ============================================================
def fig_bluesky_modelo_x_feed():
    p = FIGS_DIR / "concordancia_bluesky" / "amostra_5k.csv"
    if not p.exists():
        print("  [SKIP] F23: amostra_5k.csv ausente"); return
    rows = ler_csv(p)
    if not rows:
        print("  [SKIP] F23: csv vazio"); return

    feeds = sorted({r["feed"] for r in rows})
    mat = np.zeros((len(feeds), 2))  # cols: 0=textual, 1=topo
    cnt = np.zeros((len(feeds), 2))
    for r in rows:
        fi = feeds.index(r["feed"])
        try:
            mat[fi, 0] += float(r["score_text"]); cnt[fi, 0] += 1
            mat[fi, 1] += float(r["score_topo"]); cnt[fi, 1] += 1
        except Exception: pass
    media = np.divide(mat, cnt, out=np.zeros_like(mat), where=cnt>0)

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("RdYlGn_r")  # verde baixo, vermelho alto -> verde=real-like, vermelho=fake-like
    im = ax.imshow(media, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LogReg-BERT\n(texto)", "RF estrutural\n(topo)"], fontsize=10)
    ax.set_yticks(range(len(feeds))); ax.set_yticklabels(feeds, fontsize=9)
    for i in range(len(feeds)):
        for j in range(2):
            v = media[i, j]
            # cor do texto: branco em celulas escuras (extremos), preto no meio
            txt_color = "white" if (v > 0.75 or v < 0.25) else "#222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Score medio (0 = real-like; 1 = fake-like)")
    ax.set_title("Score medio 'fake-like' por feed x modelo no Bluesky\n"
                 "Verde = mais real-like; vermelho = mais fake-like", fontsize=11)
    plt.tight_layout()
    for out in (OUT_RES / "F23_bluesky_modelo_x_feed.png", OUT_TCC / "F23_bluesky_modelo_x_feed.png"):
        plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [OK] F23_bluesky_modelo_x_feed.png")


def main():
    print("Repintando F16, F20, F23 com paleta verde-vermelho semantica...")
    OUT_RES.mkdir(parents=True, exist_ok=True)
    fig_painel_mestre()
    fig_bluesky_crosstab()
    fig_bluesky_modelo_x_feed()
    print(f"\n[OK] Sobrescritas em {OUT_RES} e {OUT_TCC}")


if __name__ == "__main__":
    main()
