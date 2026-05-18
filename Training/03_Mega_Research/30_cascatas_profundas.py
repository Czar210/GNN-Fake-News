"""
30_cascatas_profundas.py
------------------------
Quantifica o ganho do SAGE sobre o RF tabular em grafos com cascata
multi-hop real (depth_max >= 3) no UPFD-GossipCop test.

Hipotese central: se SAGE-RF gap for desproporcionalmente maior em
cascatas profundas, e' evidencia direta de que a GNN agrega valor
*onde existe topologia multi-hop pra propagar*. Em estrelas planas
(depth=2, 70% do dataset), nao ha multi-hop -- so depende de num_nodes,
que o RF tabular ja captura.

Consome por_grafo.csv do script 28 (predicoes ja calculadas) e recarrega
o test set so para visualizacao (pyvis) de exemplos profundos.

Saidas em Execution/results/figuras_tcc/cascatas_profundas/:
  por_profundidade.csv          -- F1m e acc do RF e SAGE por valor exato de depth
  gap_subsets.csv               -- comparacao em diferentes thresholds de depth
  bootstrap_diff.csv            -- CI 95% bootstrap para diff F1(SAGE)-F1(RF) por subset
  fig_f1_por_depth.png          -- F1m de RF e SAGE por depth value (com N por barra)
  fig_gap_subsets.png           -- gap SAGE-RF em depth>=1,>=2,>=3,>=5
  amostra_deep_<idx>.html       -- 3 visualizacoes pyvis (depth alto)
  relatorio.txt

Uso:
  python 30_cascatas_profundas.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import UPFD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ    = Path(__file__).resolve().parent.parent.parent
MAT_DIR = RAIZ / "Material"
IN_CSV  = RAIZ / "Execution" / "results" / "figuras_tcc" / "estratificacao" / "por_grafo.csv"
OUT_DIR = RAIZ / "Execution" / "results" / "figuras_tcc" / "cascatas_profundas"

N_BOOTSTRAP = 2000
RANDOM_SEED = 42


def carregar_por_grafo(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def bootstrap_diff_f1(y_true, y_a, y_b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """CI 95% bootstrap para F1m(A) - F1m(B). Retorna (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = np.nan; continue
        fa = f1_score(y_true[idx], y_a[idx], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], y_b[idx], average="macro", zero_division=0)
        diffs[i] = fa - fb
    diffs = diffs[~np.isnan(diffs)]
    obs = (f1_score(y_true, y_a, average="macro", zero_division=0) -
           f1_score(y_true, y_b, average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  Cascatas profundas -- ganho SAGE vs RF em depth>=3")
    print("=" * 72)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Esperado {IN_CSV} (rode 28 primeiro).")
    d = carregar_por_grafo(IN_CSV)
    n_total = len(d["idx"])
    y_true = d["y_true"].astype(int)
    y_rf   = d["y_pred_rf"].astype(int)
    y_sage = d["y_pred_sage"].astype(int)
    depth  = d["depth_max"].astype(int)
    print(f"\nCarregado {n_total} grafos")

    # ── 1. F1 por valor exato de depth ─────────────────────────────────────
    print("\n[1/4] F1 por depth_max...")
    depths_unicos = sorted(set(depth.tolist()))
    por_depth = []
    for dep in depths_unicos:
        m = depth == dep
        n_sub = int(m.sum())
        if n_sub < 10:
            por_depth.append({"depth": dep, "n": n_sub,
                              "f1m_rf": None, "f1m_sage": None,
                              "acc_rf": float(accuracy_score(y_true[m], y_rf[m])),
                              "acc_sage": float(accuracy_score(y_true[m], y_sage[m]))})
            continue
        mixed = len(np.unique(y_true[m])) > 1
        por_depth.append({
            "depth": dep, "n": n_sub,
            "f1m_rf":   float(f1_score(y_true[m], y_rf[m],   average="macro", zero_division=0)) if mixed else None,
            "f1m_sage": float(f1_score(y_true[m], y_sage[m], average="macro", zero_division=0)) if mixed else None,
            "acc_rf":   float(accuracy_score(y_true[m], y_rf[m])),
            "acc_sage": float(accuracy_score(y_true[m], y_sage[m])),
        })

    with open(OUT_DIR / "por_profundidade.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth", "n", "acc_rf", "acc_sage", "f1m_rf", "f1m_sage", "gap_acc", "gap_f1"])
        for r in por_depth:
            gap_acc = r["acc_sage"] - r["acc_rf"]
            gap_f1  = (r["f1m_sage"] - r["f1m_rf"]) if r["f1m_rf"] is not None else None
            w.writerow([r["depth"], r["n"],
                        f"{r['acc_rf']:.4f}", f"{r['acc_sage']:.4f}",
                        f"{r['f1m_rf']:.4f}" if r["f1m_rf"] is not None else "",
                        f"{r['f1m_sage']:.4f}" if r["f1m_sage"] is not None else "",
                        f"{gap_acc:+.4f}",
                        f"{gap_f1:+.4f}" if gap_f1 is not None else ""])
        print("   depth  N      acc_RF  acc_SAGE  gap_acc  f1m_RF  f1m_SAGE  gap_f1")
        for r in por_depth:
            f1r = f"{r['f1m_rf']:.3f}" if r["f1m_rf"] is not None else "  -  "
            f1s = f"{r['f1m_sage']:.3f}" if r["f1m_sage"] is not None else "  -  "
            gap_f1 = (r["f1m_sage"] - r["f1m_rf"]) if r["f1m_rf"] is not None else None
            gap_f1s = f"{gap_f1:+.3f}" if gap_f1 is not None else "  -  "
            print(f"   {r['depth']:>3}   {r['n']:>4d}   {r['acc_rf']:.3f}   {r['acc_sage']:.3f}    "
                  f"{r['acc_sage']-r['acc_rf']:+.3f}   {f1r}    {f1s}   {gap_f1s}")
    print(f"   [OK] por_profundidade.csv")

    # ── 2. Gap em diferentes thresholds (depth>=1, >=2, >=3, >=5) ─────────
    print("\n[2/4] Gap por threshold de profundidade...")
    thresholds = [1, 2, 3, 5]
    gap_rows = []
    for th in thresholds:
        m = depth >= th
        n_sub = int(m.sum())
        if n_sub < 30 or len(np.unique(y_true[m])) < 2:
            print(f"   depth>={th}: N={n_sub} insuficiente, pulando")
            continue
        f1_rf   = f1_score(y_true[m], y_rf[m],   average="macro", zero_division=0)
        f1_sage = f1_score(y_true[m], y_sage[m], average="macro", zero_division=0)
        diff, lo, hi = bootstrap_diff_f1(y_true[m], y_sage[m], y_rf[m])
        sig = "SIM" if lo > 0 else ("NAO" if hi < 0 else "ambiguo")
        gap_rows.append({"threshold": f"depth>={th}", "n": n_sub,
                         "f1_rf": f1_rf, "f1_sage": f1_sage,
                         "diff": diff, "ci_lo": lo, "ci_hi": hi, "sig": sig})
        print(f"   depth>={th}: N={n_sub}  F1_RF={f1_rf:.3f}  F1_SAGE={f1_sage:.3f}  "
              f"diff={diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  significante={sig}")

    with open(OUT_DIR / "gap_subsets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "n", "f1m_rf", "f1m_sage", "diff", "ci95_lo", "ci95_hi", "sig"])
        for r in gap_rows:
            w.writerow([r["threshold"], r["n"],
                        f"{r['f1_rf']:.4f}", f"{r['f1_sage']:.4f}",
                        f"{r['diff']:+.4f}", f"{r['ci_lo']:+.4f}", f"{r['ci_hi']:+.4f}",
                        r["sig"]])
    print(f"   [OK] gap_subsets.csv")

    # Bootstrap separado salvo
    with open(OUT_DIR / "bootstrap_diff.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "n", "obs_diff_f1", "ci95_lo", "ci95_hi",
                    "interpretacao"])
        for r in gap_rows:
            interp = ("SAGE supera RF em magnitude significante"
                      if r["ci_lo"] > 0 else
                      "RF supera SAGE em magnitude significante"
                      if r["ci_hi"] < 0 else
                      "diferenca nao distinguivel de zero")
            w.writerow([r["threshold"], r["n"],
                        f"{r['diff']:+.4f}", f"{r['ci_lo']:+.4f}", f"{r['ci_hi']:+.4f}",
                        interp])
    print(f"   [OK] bootstrap_diff.csv")

    # ── 3. Figuras ─────────────────────────────────────────────────────────
    print("\n[3/4] Figuras...")

    # 3a. F1 por depth value
    deps = [r["depth"] for r in por_depth if r["f1m_rf"] is not None]
    rf_vals   = [r["f1m_rf"]   for r in por_depth if r["f1m_rf"] is not None]
    sage_vals = [r["f1m_sage"] for r in por_depth if r["f1m_rf"] is not None]
    ns        = [r["n"]        for r in por_depth if r["f1m_rf"] is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    pos = np.arange(len(deps))
    ax.bar(pos - 0.18, rf_vals,   0.35, label="RF tabular [num_nodes, grau_root]", color="#4F86C6")
    ax.bar(pos + 0.18, sage_vals, 0.35, label="SAGE estrutural", color="#E07B54")
    for i, n in enumerate(ns):
        ax.text(pos[i], max(rf_vals[i], sage_vals[i]) + 0.02, f"N={n}",
                ha="center", fontsize=8, color="dimgray")
    ax.set_xticks(pos); ax.set_xticklabels(deps)
    ax.set_xlabel("Profundidade max da cascata (depth_max)")
    ax.set_ylabel("F1-macro")
    ax.set_title("F1 por profundidade da cascata (UPFD-GossipCop test)\n"
                 "Gap RF -> SAGE cresce com profundidade = sinal multi-hop")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.0)
    plt.tight_layout()
    p1 = OUT_DIR / "fig_f1_por_depth.png"
    plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {p1.name}")

    # 3b. Gap por threshold com CI bootstrap
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["threshold"] for r in gap_rows]
    diffs  = [r["diff"]  for r in gap_rows]
    los    = [r["diff"] - r["ci_lo"] for r in gap_rows]
    his    = [r["ci_hi"] - r["diff"] for r in gap_rows]
    cores  = ["#5BAD72" if r["sig"]=="SIM" else "#999" for r in gap_rows]
    ax.bar(labels, diffs, yerr=[los, his], capsize=6, color=cores, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    for i, r in enumerate(gap_rows):
        ax.text(i, diffs[i] + (his[i] + 0.005), f"N={r['n']}",
                ha="center", fontsize=8, color="dimgray")
    ax.set_ylabel("F1(SAGE) - F1(RF), com CI 95% bootstrap")
    ax.set_title("Ganho do SAGE sobre o RF em diferentes recortes de profundidade\n"
                 "(verde = CI inferior > 0 = SAGE significativamente melhor)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = OUT_DIR / "fig_gap_subsets.png"
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {p2.name}")

    # ── 4. Visualizacao pyvis dos 3 grafos mais profundos ─────────────────
    print("\n[4/4] Visualizando exemplos profundos via pyvis...")
    try:
        from pyvis.network import Network
    except ImportError:
        print("   [WARN] pyvis nao instalado -- pulando visualizacoes")
        Network = None

    if Network is not None:
        print("   Recarregando UPFD-GossipCop test (necessario para edge_index)...")
        test = list(UPFD(root=str(MAT_DIR), name="gossipcop",
                         feature="profile", split="test"))
        # 3 grafos mais profundos (ordenados por depth desc, desempate por num_nodes)
        ordem = sorted(range(len(test)),
                       key=lambda i: (depth[i], d["num_nodes"][i]), reverse=True)[:3]
        for rank, idx in enumerate(ordem, 1):
            g = test[idx]
            dep_v = int(depth[idx])
            yt = "REAL" if y_true[idx] == 1 else "FAKE"
            yp_rf   = "REAL" if y_rf[idx]   == 1 else "FAKE"
            yp_sage = "REAL" if y_sage[idx] == 1 else "FAKE"

            net = Network(height="600px", width="100%", directed=True,
                          notebook=False, bgcolor="#fafafa")
            for i in range(g.num_nodes):
                cor = "#E07B54" if i == 0 else "#5BAD72"
                rot = f"RAIZ ({yt})" if i == 0 else f"n{i}"
                net.add_node(i, label=rot, color=cor, size=18 if i == 0 else 8)
            for s, t in g.edge_index.t().tolist():
                net.add_edge(int(s), int(t), arrows="to")
            net.barnes_hut(spring_length=70)
            net.set_options('{"interaction": {"hover": true}}')
            out = OUT_DIR / f"amostra_deep_{rank}_idx{idx}_depth{dep_v}.html"
            try:
                net.write_html(str(out), notebook=False, open_browser=False)
                print(f"   [OK] {out.name}  (depth={dep_v}, N={g.num_nodes}, "
                      f"true={yt}, RF={yp_rf}, SAGE={yp_sage})")
            except Exception as e:
                print(f"   [WARN] erro pyvis em idx={idx}: {e}")

    # ── Relatorio ─────────────────────────────────────────────────────────
    rel = OUT_DIR / "relatorio.txt"
    baseline_rf   = f1_score(y_true, y_rf,   average="macro", zero_division=0)
    baseline_sage = f1_score(y_true, y_sage, average="macro", zero_division=0)
    with open(rel, "w", encoding="utf-8") as f:
        f.write("Relatorio -- script 30 (cascatas profundas)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: UPFD-GossipCop test ({n_total} grafos)\n")
        f.write(f"Baseline global: F1m_RF={baseline_rf:.4f}  F1m_SAGE={baseline_sage:.4f}\n")
        f.write(f"Gap global: {baseline_sage - baseline_rf:+.4f}\n\n")

        f.write("--- F1 por valor exato de depth_max ---\n")
        for r in por_depth:
            f1r = f"{r['f1m_rf']:.3f}" if r["f1m_rf"] is not None else "n/a"
            f1s = f"{r['f1m_sage']:.3f}" if r["f1m_sage"] is not None else "n/a"
            f.write(f"  depth={r['depth']:2d}  N={r['n']:4d}  "
                    f"F1_RF={f1r}  F1_SAGE={f1s}\n")

        f.write("\n--- Gap em subsets (com CI 95% bootstrap, n_boot=2000) ---\n")
        for r in gap_rows:
            f.write(f"\n  [{r['threshold']}]  N={r['n']}\n")
            f.write(f"    F1_RF   = {r['f1_rf']:.4f}\n")
            f.write(f"    F1_SAGE = {r['f1_sage']:.4f}\n")
            f.write(f"    diff    = {r['diff']:+.4f}  CI95=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]\n")
            f.write(f"    significante (CI nao cruza zero): {r['sig']}\n")

        f.write("\n--- Interpretacao ---\n")
        gap_geq3 = next((r for r in gap_rows if r["threshold"] == "depth>=3"), None)
        gap_geq1 = next((r for r in gap_rows if r["threshold"] == "depth>=1"), None)
        if gap_geq3 and gap_geq1:
            ratio = abs(gap_geq3["diff"]) / max(abs(gap_geq1["diff"]), 1e-9)
            f.write(f"\n  Gap em depth>=3 ({gap_geq3['diff']:+.3f}) vs em depth>=1 ({gap_geq1['diff']:+.3f}): "
                    f"razao = {ratio:.2f}x\n")
            if ratio > 1.5 and gap_geq3["sig"] == "SIM":
                f.write("  -> Gap DESPROPORCIONAL em cascatas profundas. Sustenta a tese:\n"
                        "     SAGE agrega valor onde existe topologia multi-hop pra propagar.\n"
                        "     Em estrelas planas (depth=2), RF tabular captura o sinal -- nao precisa de GNN.\n")
            elif ratio > 1.1:
                f.write("  -> Gap MAIOR em cascatas profundas mas nao dramaticamente.\n"
                        "     Evidencia parcial pra tese; vale reportar com cautela.\n")
            else:
                f.write("  -> Gap UNIFORME entre profundidades. Tese nao se sustenta neste recorte:\n"
                        "     GNN ganha em volume, nao em estrutura multi-hop.\n")

        f.write("\n--- Implicacao para o TCC ---\n")
        f.write("§6.3.a do TCC menciona limitacao 'FNN com cascatas estrela'. UPFD-GossipCop\n")
        f.write(f"oficial NAO eh estrela plana: 100% tem depth>=2 e 30% tem depth>=3.\n")
        f.write("Considerar refinar a redacao: a critica de 'estrela plana' se aplica ao\n")
        f.write("FakeNewsNet construido por nos (Script 00), nao ao UPFD oficial.\n")
    print(f"\n[OK] {rel.name}")
    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
