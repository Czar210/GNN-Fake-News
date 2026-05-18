"""
28_distribuicao_e_estratificacao.py
-----------------------------------
Analise estratificada do erro do modelo ao longo de metricas estruturais
(num_nodes, depth_max, branching_avg) no UPFD-GossipCop test.

Responde 2 perguntas que faltam no TCC:
  (1) Como a distribuicao dos grafos se comporta? (fit lognormal sobre num_nodes)
  (2) Onde, dentro dessa distribuicao, o modelo erra mais? (LOWESS + tercis)

Tambem prepara terreno (sem analisar ainda) para 2 scripts futuros:
  - 29 (outliers estatisticos): por_grafo.csv contem tudo necessario
  - 30 (cascatas profundas): contagem por profundidade ja sumarizada

Modelos avaliados (carregados de Execution/weights/):
  - RF estrutural [num_nodes, grau_root]   F1m ~0.75 no test
  - SAGE estrutural [is_root, grau_norm]   F1m ~0.81 no test

Saidas em Execution/results/figuras_tcc/estratificacao/:
  por_grafo.csv             -- 1 linha por grafo do test (insumo de 29/30)
  bins_f1.csv               -- F1m por (metrica, tercil, modelo)
  distribuicao_fit.csv      -- parametros do fit lognormal por metrica
  fig_distribuicao_log.png  -- histograma + fit lognormal (eixo log)
  fig_lowess_acerto.png     -- P(acerto) suavizado vs num_nodes/depth/branching
  fig_f1_por_bin.png        -- bar chart F1 por tercil x modelo
  fig_profundidade.png      -- contagem por depth (preparacao p/ script 30)
  relatorio.txt             -- sumario numerico + decisoes p/ proximas rodadas

Uso:
  python 28_distribuicao_e_estratificacao.py
"""

import csv
import json
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_model import SAGEClassifier

RAIZ        = Path(__file__).resolve().parent.parent.parent
MAT_DIR     = RAIZ / "Material"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "estratificacao"


# ============================================================
# Metricas estruturais (mesma logica do script 15, isolada aqui)
# ============================================================
def metricas_grafo(g) -> dict:
    n = int(g.num_nodes)
    m = int(g.num_edges)
    grau_root = int((g.edge_index[0] == 0).sum().item()) if m else 0
    if m == 0 or n == 1:
        return {"num_nodes": n, "num_edges": m, "grau_root": grau_root,
                "depth_max": 0, "width_max": 1, "branching_avg": 0.0}

    adj = defaultdict(list)
    for s, t in g.edge_index.t().tolist():
        adj[s].append(t); adj[t].append(s)

    prof = {0: 0}
    fila = deque([0])
    while fila:
        u = fila.popleft()
        for v in adj[u]:
            if v not in prof:
                prof[v] = prof[u] + 1
                fila.append(v)

    depth_max = max(prof.values())
    largura = defaultdict(int)
    for d in prof.values():
        largura[d] += 1
    width_max = max(largura.values())

    filhos = []
    for u in prof:
        nf = sum(1 for v in adj[u] if prof.get(v, -1) > prof[u])
        if nf > 0: filhos.append(nf)
    branching = float(np.mean(filhos)) if filhos else 0.0

    return {"num_nodes": n, "num_edges": m, "grau_root": grau_root,
            "depth_max": depth_max, "width_max": width_max,
            "branching_avg": branching}


# ============================================================
# Carrega modelos persistidos e gera predicoes
# ============================================================
def predicoes_rf(grafos: list) -> np.ndarray:
    with open(WEIGHTS_DIR / "rf_struct_gossipcop.pkl", "rb") as f:
        bundle = pickle.load(f)
    rf = bundle["model"]
    X = np.array([[g.num_nodes,
                   int((g.edge_index[0] == 0).sum().item()) if g.edge_index.numel() else 0]
                  for g in grafos], dtype=np.float64)
    return rf.predict(X)


def features_sage(g) -> torch.Tensor:
    n = g.num_nodes
    is_root = torch.zeros(n, dtype=torch.float); is_root[0] = 1.0
    deg = torch.zeros(n, dtype=torch.float)
    if g.edge_index.numel():
        idx, counts = torch.unique(g.edge_index[0], return_counts=True)
        deg[idx] = counts.float()
    deg_norm = deg / max(deg.max().item(), 1.0)
    return torch.stack([is_root, deg_norm], dim=1)


def predicoes_sage(grafos: list) -> np.ndarray:
    ckpt = torch.load(WEIGHTS_DIR / "sage_struct_gossipcop.pth", map_location="cpu",
                      weights_only=False)
    sage = SAGEClassifier(num_node_features=ckpt["input_dim"], num_classes=2,
                          hidden_channels=ckpt["hidden_channels"])
    sage.load_state_dict(ckpt["state_dict"])
    sage.eval()

    transformados = [Data(x=features_sage(g), edge_index=g.edge_index, y=g.y)
                     for g in grafos]
    loader = DataLoader(transformados, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            out, _ = sage(batch.x, batch.edge_index, batch.batch)
            preds.extend(out.argmax(dim=1).tolist())
    return np.array(preds, dtype=np.int64)


# ============================================================
# LOWESS via rolling-mean sobre dados ordenados (sem statsmodels)
# ============================================================
def smoother_rolling(x: np.ndarray, y: np.ndarray, janela=0.10):
    """P(y=1 | x) suavizado por janela movel proporcional ao tamanho da amostra."""
    ord_idx = np.argsort(x)
    xs, ys = x[ord_idx], y[ord_idx]
    n = len(xs)
    w = max(int(janela * n), 30)
    suav = np.array([ys[max(0, i - w // 2): min(n, i + w // 2)].mean() for i in range(n)])
    return xs, suav


# ============================================================
# Fit lognormal: retorna (mu, sigma) + KS p-valor (>0.05 = bom fit)
# ============================================================
def fit_lognormal(x: np.ndarray):
    x_pos = x[x > 0]
    if len(x_pos) < 10:
        return None
    shape, loc, scale = stats.lognorm.fit(x_pos, floc=0)
    mu, sigma = np.log(scale), shape
    ks_stat, ks_p = stats.kstest(x_pos, "lognorm", args=(shape, loc, scale))
    return {"mu": float(mu), "sigma": float(sigma),
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "n": int(len(x_pos))}


# ============================================================
# Estratificacao por tercis
# ============================================================
def f1_por_tercil(metrica_vals: np.ndarray, y_true: np.ndarray,
                  y_pred: np.ndarray) -> list:
    """Retorna lista de dicts: [{tercil, lo, hi, n, f1m, acc}]."""
    q33, q66 = np.percentile(metrica_vals, [33.33, 66.67])
    bins = [(-np.inf, q33, "T1 (baixo)"),
            (q33, q66, "T2 (medio)"),
            (q66, np.inf, "T3 (alto)")]
    out = []
    for lo, hi, nome in bins:
        mask = (metrica_vals > lo) & (metrica_vals <= hi)
        if mask.sum() == 0:
            out.append({"tercil": nome, "lo": float(lo), "hi": float(hi),
                        "n": 0, "f1m": float("nan"), "acc": float("nan")})
            continue
        yt, yp = y_true[mask], y_pred[mask]
        f1m = f1_score(yt, yp, average="macro", zero_division=0)
        acc = accuracy_score(yt, yp)
        out.append({"tercil": nome,
                    "lo": float(lo) if np.isfinite(lo) else None,
                    "hi": float(hi) if np.isfinite(hi) else None,
                    "n": int(mask.sum()), "f1m": float(f1m), "acc": float(acc)})
    return out


# ============================================================
# Main
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  Distribuicao + estratificacao em UPFD-GossipCop test")
    print("=" * 72)

    # ── 1. Carregar test set ──────────────────────────────────────────────
    print("\n[1/6] Carregando UPFD-GossipCop test...")
    test = list(UPFD(root=str(MAT_DIR), name="gossipcop",
                     feature="profile", split="test"))
    print(f"   {len(test)} grafos")

    # ── 2. Metricas estruturais por grafo ──────────────────────────────────
    print("\n[2/6] Computando metricas estruturais...")
    mets = [metricas_grafo(g) for g in test]
    y_true = np.array([g.y.item() for g in test], dtype=np.int64)

    # ── 3. Predicoes dos modelos ──────────────────────────────────────────
    print("\n[3/6] Carregando modelos e predizendo...")
    y_rf   = predicoes_rf(test)
    print(f"   RF   F1m={f1_score(y_true, y_rf,   average='macro', zero_division=0):.4f}  "
          f"acc={accuracy_score(y_true, y_rf):.4f}")
    y_sage = predicoes_sage(test)
    print(f"   SAGE F1m={f1_score(y_true, y_sage, average='macro', zero_division=0):.4f}  "
          f"acc={accuracy_score(y_true, y_sage):.4f}")

    # ── 4. CSV por grafo (insumo de scripts 29/30) ─────────────────────────
    print("\n[4/6] Salvando por_grafo.csv (insumo p/ outliers e cascatas profundas)...")
    csv_g = OUT_DIR / "por_grafo.csv"
    cols = ["idx", "num_nodes", "num_edges", "grau_root",
            "depth_max", "width_max", "branching_avg",
            "y_true", "y_pred_rf", "y_pred_sage",
            "acerto_rf", "acerto_sage"]
    with open(csv_g, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for i, m in enumerate(mets):
            w.writerow([i, m["num_nodes"], m["num_edges"], m["grau_root"],
                        m["depth_max"], m["width_max"], f"{m['branching_avg']:.4f}",
                        int(y_true[i]), int(y_rf[i]), int(y_sage[i]),
                        int(y_true[i] == y_rf[i]), int(y_true[i] == y_sage[i])])
    print(f"   [OK] {csv_g.name} ({len(mets)} linhas)")

    # ── 5. Distribuicao: fit lognormal + figura ────────────────────────────
    print("\n[5/6] Fit lognormal + figura de distribuicao...")
    metricas_fit = ["num_nodes", "depth_max", "branching_avg"]
    fits = {}
    for met in metricas_fit:
        vals = np.array([m[met] for m in mets], dtype=np.float64)
        f = fit_lognormal(vals)
        if f is not None:
            fits[met] = f
            print(f"   {met:<14} lognormal mu={f['mu']:.3f} sigma={f['sigma']:.3f}  "
                  f"KS p={f['ks_p']:.4g}  (n={f['n']})")
        else:
            print(f"   {met:<14} -- amostras insuficientes para fit")

    with open(OUT_DIR / "distribuicao_fit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "n", "mu", "sigma", "ks_stat", "ks_p", "fit_bom_p>0.05"])
        for met, p in fits.items():
            w.writerow([met, p["n"], f"{p['mu']:.4f}", f"{p['sigma']:.4f}",
                        f"{p['ks_stat']:.4f}", f"{p['ks_p']:.6f}",
                        "SIM" if p["ks_p"] > 0.05 else "NAO"])

    # Figura: histograma + curva lognormal (eixo log onde fizer sentido)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, met in zip(axes, metricas_fit):
        vals = np.array([m[met] for m in mets], dtype=np.float64)
        vals_pos = vals[vals > 0]
        if met == "num_nodes":
            bins = np.geomspace(max(vals_pos.min(), 1), vals_pos.max(), 40)
            ax.set_xscale("log")
        else:
            bins = 30
        ax.hist(vals_pos, bins=bins, density=True, alpha=0.6, color="#4F86C6",
                label=f"empirico (n={len(vals_pos)})")
        if met in fits:
            f_ = fits[met]
            shape = f_["sigma"]; scale = np.exp(f_["mu"])
            xs = np.geomspace(vals_pos.min(), vals_pos.max(), 200) if met == "num_nodes" \
                 else np.linspace(vals_pos.min(), vals_pos.max(), 200)
            ax.plot(xs, stats.lognorm.pdf(xs, shape, loc=0, scale=scale),
                    color="#E07B54", lw=2,
                    label=f"lognorm fit\n(KS p={f_['ks_p']:.3g})")
        ax.set_title(met); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Distribuicao das metricas estruturais (UPFD-GossipCop test)", y=1.02)
    plt.tight_layout()
    p1 = OUT_DIR / "fig_distribuicao_log.png"
    plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {p1.name}")

    # ── 6. LOWESS + estratificacao por tercis + profundidade ───────────────
    print("\n[6/6] LOWESS + tercis + figura profundidade...")

    # 6a. LOWESS (rolling smoother) de P(acerto) vs metricas
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metricas_lowess = ["num_nodes", "depth_max", "branching_avg"]
    for ax, met in zip(axes, metricas_lowess):
        vals = np.array([m[met] for m in mets], dtype=np.float64)
        ac_rf   = (y_true == y_rf).astype(float)
        ac_sage = (y_true == y_sage).astype(float)
        xr, sr = smoother_rolling(vals, ac_rf)
        xs, ss = smoother_rolling(vals, ac_sage)
        ax.plot(xr, sr, color="#4F86C6", lw=1.8, label="RF tabular")
        ax.plot(xs, ss, color="#E07B54", lw=1.8, label="SAGE estrutural")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.5, label="chance")
        if met == "num_nodes":
            ax.set_xscale("log")
        ax.set_xlabel(met); ax.set_ylabel("P(acerto) suavizado")
        ax.set_title(met); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    plt.suptitle("LOWESS: prob de acerto vs metrica estrutural", y=1.02)
    plt.tight_layout()
    p2 = OUT_DIR / "fig_lowess_acerto.png"
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {p2.name}")

    # 6b. F1 por tercil (CSV + bar chart)
    bins_rows = []
    for met in metricas_lowess:
        vals = np.array([m[met] for m in mets], dtype=np.float64)
        for modelo, pred in [("RF", y_rf), ("SAGE", y_sage)]:
            for b in f1_por_tercil(vals, y_true, pred):
                bins_rows.append({"metrica": met, "modelo": modelo, **b})

    csv_b = OUT_DIR / "bins_f1.csv"
    with open(csv_b, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "modelo", "tercil", "lo", "hi", "n", "f1m", "acc"])
        for r in bins_rows:
            w.writerow([r["metrica"], r["modelo"], r["tercil"],
                        r["lo"], r["hi"], r["n"],
                        f"{r['f1m']:.4f}", f"{r['acc']:.4f}"])
    print(f"   [OK] {csv_b.name}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    tercis = ["T1 (baixo)", "T2 (medio)", "T3 (alto)"]
    for ax, met in zip(axes, metricas_lowess):
        rf_vals   = [next(r["f1m"] for r in bins_rows
                          if r["metrica"]==met and r["modelo"]=="RF"   and r["tercil"]==t)
                     for t in tercis]
        sage_vals = [next(r["f1m"] for r in bins_rows
                          if r["metrica"]==met and r["modelo"]=="SAGE" and r["tercil"]==t)
                     for t in tercis]
        pos = np.arange(len(tercis))
        ax.bar(pos - 0.18, rf_vals,   0.35, label="RF",   color="#4F86C6")
        ax.bar(pos + 0.18, sage_vals, 0.35, label="SAGE", color="#E07B54")
        ax.set_xticks(pos); ax.set_xticklabels(tercis, fontsize=9)
        ax.set_ylabel("F1-macro")
        ax.set_title(f"por tercil de {met}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)
    plt.suptitle("F1 por tercil estrutural (UPFD-GossipCop test)", y=1.02)
    plt.tight_layout()
    p3 = OUT_DIR / "fig_f1_por_bin.png"
    plt.savefig(p3, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {p3.name}")

    # 6c. Profundidade: contagem e F1 condicionado (preparacao p/ script 30)
    profundidades = np.array([m["depth_max"] for m in mets], dtype=int)
    cont = {int(d): int((profundidades == d).sum()) for d in sorted(set(profundidades))}
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = sorted(cont.keys()); ys = [cont[k] for k in xs]
    ax.bar(xs, ys, color="#5BAD72")
    for i, v in enumerate(ys):
        ax.text(xs[i], v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Profundidade max da cascata (depth_max)")
    ax.set_ylabel("Numero de grafos")
    ax.set_title("Distribuicao de profundidade -- UPFD-GossipCop test\n"
                 "(preparacao para analise de cascatas profundas, script 30)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p4 = OUT_DIR / "fig_profundidade.png"
    plt.savefig(p4, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {p4.name}")

    # ── Relatorio textual com decisoes ─────────────────────────────────────
    rel = OUT_DIR / "relatorio.txt"
    with open(rel, "w", encoding="utf-8") as f:
        f.write("Relatorio -- script 28 (distribuicao + estratificacao)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset: UPFD-GossipCop test (3826 grafos esperados)\n")
        f.write(f"  N efetivo carregado: {len(test)}\n\n")

        f.write("Modelos (carregados de Execution/weights/):\n")
        f.write(f"  RF   F1m={f1_score(y_true, y_rf,   average='macro', zero_division=0):.4f}\n")
        f.write(f"  SAGE F1m={f1_score(y_true, y_sage, average='macro', zero_division=0):.4f}\n\n")

        f.write("--- Distribuicao (fit lognormal) ---\n")
        for met, p in fits.items():
            ok = "BOM FIT" if p["ks_p"] > 0.05 else "fit ruim (heavy-tail diferente)"
            f.write(f"  {met:<14} mu={p['mu']:+.3f} sigma={p['sigma']:.3f}  "
                    f"KS p={p['ks_p']:.4g}  ({ok})\n")
        f.write("\n")

        f.write("--- F1m por tercil ---\n")
        for met in metricas_lowess:
            f.write(f"\n  [{met}]\n")
            for modelo in ["RF", "SAGE"]:
                f.write(f"    {modelo}:")
                for r in bins_rows:
                    if r["metrica"]==met and r["modelo"]==modelo:
                        f.write(f"  {r['tercil']}={r['f1m']:.3f} (n={r['n']})")
                f.write("\n")

        f.write("\n--- Profundidade (preparacao p/ script 30) ---\n")
        for d in sorted(cont):
            f.write(f"  depth={d}: {cont[d]} grafos ({100*cont[d]/len(test):.1f}%)\n")
        prof_geq2 = int((profundidades >= 2).sum())
        prof_geq3 = int((profundidades >= 3).sum())
        f.write(f"\n  depth>=2: {prof_geq2} ({100*prof_geq2/len(test):.1f}%)\n")
        f.write(f"  depth>=3: {prof_geq3} ({100*prof_geq3/len(test):.1f}%)\n")
        if prof_geq3 >= 30:
            f.write("  -> SUFICIENTE p/ analise estatistica em script 30\n")
        elif prof_geq3 >= 10:
            f.write("  -> POUCO. Script 30 sera principalmente qualitativo/visualizacao\n")
        else:
            f.write("  -> RARO. Script 30 vira anedota ilustrativa + reportar escassez\n")

        f.write("\n--- DECISOES PARA PROXIMAS RODADAS ---\n")
        f.write("Script 29 (outliers): usar por_grafo.csv. Criterio sugerido pos-hoc:\n")
        f.write("  - real_viral:   y_true=real (1) AND grau_root > p95\n")
        f.write("  - fake_contido: y_true=fake (0) AND grau_root < p5\n")
        f.write("  - cauda tamanho: num_nodes > p95 ou num_nodes <= p5\n")
        f.write("  Confirmar percentis empiricos olhando por_grafo.csv\n\n")
        f.write("Script 30 (cascatas profundas): focar em depth>=2 ou >=3 (ver acima).\n")
        f.write("  Comparacao chave: RF (que so ve num_nodes,grau_root) vs SAGE neste subset.\n")
        f.write("  Se ganho SAGE-RF for desproporcional aqui, e' evidencia direta de que\n")
        f.write("  a GNN agrega valor onde tem multi-hop pra propagar.\n")
    print(f"\n[OK] {rel.name}")

    # ── Sumario final ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Resumo")
    print("=" * 72)
    print(f"  Test set: {len(test)} grafos | RF={f1_score(y_true, y_rf, average='macro', zero_division=0):.4f} | "
          f"SAGE={f1_score(y_true, y_sage, average='macro', zero_division=0):.4f}")
    print(f"  Profundidade: depth>=2 em {int((profundidades>=2).sum())} grafos, "
          f"depth>=3 em {int((profundidades>=3).sum())} grafos")
    print(f"  Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
