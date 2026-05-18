"""
29_outliers.py
--------------
Analise de subgrupos extremos (outliers) em UPFD-GossipCop test.

Consome o por_grafo.csv produzido pelo script 28 -- nao re-carrega modelos.

Subgrupos definidos *post-hoc* a partir das distribuicoes do 28:
  1. real_viral     -- y_true=real ∩ grau_root > p95     (contra-exemplo regra "fake=viral")
  2. fake_contido   -- y_true=fake ∩ grau_root <= p5     (contra-exemplo regra "real=pouco eng.")
  3. cauda_alta_n   -- num_nodes > p95                    (grafos enormes)
  4. cauda_baixa_n  -- num_nodes <= p5                    (grafos minusculos)
  5. deep_extreme   -- depth_max >= 5                     (cascatas muito profundas)
  6. wide_extreme   -- branching_avg > p95                (super-spreaders)

Para cada subgrupo:
  - N total + composicao por classe
  - Acuracia do RF e do SAGE
  - F1-macro (so se subgrupo for mixed-class)
  - Taxa em que o modelo preve "fake"
  - Comparacao com baseline agregado

Saidas em Execution/results/figuras_tcc/outliers/:
  subgrupos.csv          -- 1 linha por subgrupo
  exemplos.csv           -- 5 grafos representativos de cada subgrupo
  fig_outliers_acc.png   -- bar chart acuracia por subgrupo (RF vs SAGE)
  fig_outliers_pred.png  -- matriz: subgrupo x [%fake_real, %correct_rf, %correct_sage]
  relatorio.txt

Uso:
  python 29_outliers.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ    = Path(__file__).resolve().parent.parent.parent
IN_CSV  = RAIZ / "Execution" / "results" / "figuras_tcc" / "estratificacao" / "por_grafo.csv"
OUT_DIR = RAIZ / "Execution" / "results" / "figuras_tcc" / "outliers"


def carregar_por_grafo(path: Path) -> dict:
    """Le por_grafo.csv como dict de arrays numpy."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = {}
    for k in rows[0].keys():
        try:
            cols[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            cols[k] = np.array([r[k] for r in rows])
    return cols


def avaliar_subgrupo(mask: np.ndarray, dados: dict, nome: str,
                     baseline_global: dict) -> dict:
    """Retorna metricas para um subgrupo definido pela mask."""
    n = int(mask.sum())
    if n == 0:
        return {"subgrupo": nome, "n": 0}

    y_true = dados["y_true"][mask].astype(int)
    y_rf   = dados["y_pred_rf"][mask].astype(int)
    y_sage = dados["y_pred_sage"][mask].astype(int)

    n_fake = int((y_true == 0).sum())
    n_real = int((y_true == 1).sum())
    mixed  = n_fake > 0 and n_real > 0

    res = {
        "subgrupo": nome, "n": n,
        "n_fake_real": n_fake, "n_real_real": n_real,
        "acc_rf":   float(accuracy_score(y_true, y_rf)),
        "acc_sage": float(accuracy_score(y_true, y_sage)),
        "pct_pred_fake_rf":   float((y_rf == 0).mean()),
        "pct_pred_fake_sage": float((y_sage == 0).mean()),
        # delta vs baseline global (negativo = pior que media)
        "delta_acc_rf":   float(accuracy_score(y_true, y_rf)   - baseline_global["acc_rf"]),
        "delta_acc_sage": float(accuracy_score(y_true, y_sage) - baseline_global["acc_sage"]),
    }
    if mixed:
        res["f1m_rf"]   = float(f1_score(y_true, y_rf,   average="macro", zero_division=0))
        res["f1m_sage"] = float(f1_score(y_true, y_sage, average="macro", zero_division=0))
    else:
        res["f1m_rf"] = res["f1m_sage"] = None  # nao faz sentido single-class
    return res


def amostrar_exemplos(mask: np.ndarray, dados: dict, nome: str,
                      k: int = 5) -> list:
    """Pega k grafos representativos do subgrupo (espacados pela mediana de num_nodes)."""
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []
    if len(idxs) <= k:
        sel = idxs
    else:
        ord_ = np.argsort(dados["num_nodes"][idxs])
        passo = max(1, len(ord_) // k)
        sel = idxs[ord_[::passo]][:k]
    return [{"subgrupo": nome, "idx": int(i),
             "num_nodes": int(dados["num_nodes"][i]),
             "grau_root": int(dados["grau_root"][i]),
             "depth_max": int(dados["depth_max"][i]),
             "branching_avg": float(dados["branching_avg"][i]),
             "y_true":     int(dados["y_true"][i]),
             "y_pred_rf":  int(dados["y_pred_rf"][i]),
             "y_pred_sage":int(dados["y_pred_sage"][i])}
            for i in sel]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Analise de outliers -- UPFD-GossipCop test")
    print("=" * 72)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Esperado {IN_CSV} (rode 28 primeiro).")
    dados = carregar_por_grafo(IN_CSV)
    n_total = len(dados["idx"])
    print(f"\nCarregado {n_total} grafos de {IN_CSV.name}")

    # ── Baseline global (para calcular delta dos subgrupos) ────────────────
    baseline = {
        "acc_rf":   float(accuracy_score(dados["y_true"], dados["y_pred_rf"])),
        "acc_sage": float(accuracy_score(dados["y_true"], dados["y_pred_sage"])),
    }
    print(f"Baseline global: acc_RF={baseline['acc_rf']:.4f}  acc_SAGE={baseline['acc_sage']:.4f}")

    # ── Percentis empiricos ────────────────────────────────────────────────
    grau_p5,  grau_p95 = np.percentile(dados["grau_root"], [5, 95])
    n_p5,     n_p95    = np.percentile(dados["num_nodes"], [5, 95])
    branch_p95         = np.percentile(dados["branching_avg"], 95)
    print(f"\nPercentis empiricos:")
    print(f"  grau_root     p5={grau_p5:.1f}  p95={grau_p95:.1f}")
    print(f"  num_nodes     p5={n_p5:.1f}   p95={n_p95:.1f}")
    print(f"  branching_avg p95={branch_p95:.2f}")

    # ── Definicao dos subgrupos ────────────────────────────────────────────
    is_fake = dados["y_true"] == 0
    is_real = dados["y_true"] == 1

    subgrupos = {
        "real_viral":     is_real & (dados["grau_root"]    >  grau_p95),
        "fake_contido":   is_fake & (dados["grau_root"]    <= grau_p5),
        "cauda_alta_n":             dados["num_nodes"]     >  n_p95,
        "cauda_baixa_n":            dados["num_nodes"]     <= n_p5,
        "deep_extreme":             dados["depth_max"]     >= 5,
        "wide_extreme":             dados["branching_avg"] >  branch_p95,
    }

    # ── Avaliar cada subgrupo ─────────────────────────────────────────────
    print("\n--- Resultados por subgrupo ---")
    resultados = []
    for nome, mask in subgrupos.items():
        r = avaliar_subgrupo(mask, dados, nome, baseline)
        resultados.append(r)
        if r["n"] == 0:
            print(f"  {nome:<18} N=0 (vazio)")
            continue
        f1_str = ""
        if r.get("f1m_rf") is not None:
            f1_str = f" | F1m: RF={r['f1m_rf']:.3f} SAGE={r['f1m_sage']:.3f}"
        print(f"  {nome:<18} N={r['n']:4d}  fake={r['n_fake_real']:3d} real={r['n_real_real']:3d}  "
              f"acc: RF={r['acc_rf']:.3f} (d={r['delta_acc_rf']:+.3f})  "
              f"SAGE={r['acc_sage']:.3f} (d={r['delta_acc_sage']:+.3f}){f1_str}")

    # ── CSV principal ─────────────────────────────────────────────────────
    csv_main = OUT_DIR / "subgrupos.csv"
    with open(csv_main, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subgrupo", "n", "n_fake", "n_real",
                    "acc_rf", "acc_sage", "delta_acc_rf", "delta_acc_sage",
                    "f1m_rf", "f1m_sage",
                    "pct_pred_fake_rf", "pct_pred_fake_sage"])
        for r in resultados:
            if r["n"] == 0:
                w.writerow([r["subgrupo"], 0, 0, 0, "", "", "", "", "", "", "", ""])
                continue
            w.writerow([r["subgrupo"], r["n"], r["n_fake_real"], r["n_real_real"],
                        f"{r['acc_rf']:.4f}", f"{r['acc_sage']:.4f}",
                        f"{r['delta_acc_rf']:+.4f}", f"{r['delta_acc_sage']:+.4f}",
                        f"{r['f1m_rf']:.4f}" if r.get("f1m_rf") is not None else "",
                        f"{r['f1m_sage']:.4f}" if r.get("f1m_sage") is not None else "",
                        f"{r['pct_pred_fake_rf']:.4f}", f"{r['pct_pred_fake_sage']:.4f}"])
    print(f"\n[OK] {csv_main.name}")

    # ── Exemplos ──────────────────────────────────────────────────────────
    todos_exemplos = []
    for nome, mask in subgrupos.items():
        todos_exemplos.extend(amostrar_exemplos(mask, dados, nome, k=5))
    csv_ex = OUT_DIR / "exemplos.csv"
    with open(csv_ex, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subgrupo", "idx", "num_nodes", "grau_root", "depth_max",
                    "branching_avg", "y_true", "y_pred_rf", "y_pred_sage"])
        for e in todos_exemplos:
            w.writerow([e["subgrupo"], e["idx"], e["num_nodes"], e["grau_root"],
                        e["depth_max"], f"{e['branching_avg']:.3f}",
                        e["y_true"], e["y_pred_rf"], e["y_pred_sage"]])
    print(f"[OK] {csv_ex.name}")

    # ── Figura 1: bar chart acuracia por subgrupo (RF vs SAGE) ────────────
    nomes_ord = [r["subgrupo"] for r in resultados if r["n"] > 0]
    acc_rf    = [r["acc_rf"]   for r in resultados if r["n"] > 0]
    acc_sage  = [r["acc_sage"] for r in resultados if r["n"] > 0]
    ns        = [r["n"]        for r in resultados if r["n"] > 0]

    fig, ax = plt.subplots(figsize=(11, 5))
    pos = np.arange(len(nomes_ord))
    ax.bar(pos - 0.18, acc_rf,   0.35, label="RF",   color="#4F86C6")
    ax.bar(pos + 0.18, acc_sage, 0.35, label="SAGE", color="#E07B54")
    ax.axhline(baseline["acc_rf"],   color="#4F86C6", lw=0.7, ls="--", alpha=0.6,
               label=f"baseline RF ({baseline['acc_rf']:.3f})")
    ax.axhline(baseline["acc_sage"], color="#E07B54", lw=0.7, ls="--", alpha=0.6,
               label=f"baseline SAGE ({baseline['acc_sage']:.3f})")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.4, label="chance")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{n}\n(N={ns[i]})" for i, n in enumerate(nomes_ord)],
                       fontsize=8, rotation=15)
    ax.set_ylabel("Acuracia")
    ax.set_title("Acuracia do modelo por subgrupo extremo (UPFD-GossipCop test)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p1 = OUT_DIR / "fig_outliers_acc.png"
    plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print(f"[OK] {p1.name}")

    # ── Figura 2: heatmap com pct fake real x pct fake predito ────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mat_data = []
    for r in resultados:
        if r["n"] == 0: continue
        pct_real_fake = r["n_fake_real"] / r["n"] if r["n"] else 0
        mat_data.append([pct_real_fake, r["pct_pred_fake_rf"], r["pct_pred_fake_sage"]])
    mat = np.array(mat_data)
    im = ax.imshow(mat.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(nomes_ord)))
    ax.set_xticklabels([f"{n}\n(N={ns[i]})" for i, n in enumerate(nomes_ord)],
                       fontsize=8, rotation=15)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["% fake (verdade)", "% predito fake (RF)", "% predito fake (SAGE)"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i, j, f"{mat[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i,j]-0.5)>0.3 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="proporcao")
    ax.set_title("Composicao real vs predicao do modelo por subgrupo\n"
                 "(linhas iguais = modelo bem calibrado; verde-vermelho = divergencia)")
    plt.tight_layout()
    p2 = OUT_DIR / "fig_outliers_pred.png"
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print(f"[OK] {p2.name}")

    # ── Relatorio ─────────────────────────────────────────────────────────
    rel = OUT_DIR / "relatorio.txt"
    with open(rel, "w", encoding="utf-8") as f:
        f.write("Relatorio -- script 29 (outliers)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: UPFD-GossipCop test ({n_total} grafos)\n")
        f.write(f"Baseline global: acc_RF={baseline['acc_rf']:.4f}  "
                f"acc_SAGE={baseline['acc_sage']:.4f}\n\n")

        f.write("Percentis empiricos:\n")
        f.write(f"  grau_root     p5={grau_p5:.1f}  p95={grau_p95:.1f}\n")
        f.write(f"  num_nodes     p5={n_p5:.1f}   p95={n_p95:.1f}\n")
        f.write(f"  branching_avg p95={branch_p95:.2f}\n\n")

        f.write("--- Resultados ---\n")
        for r in resultados:
            if r["n"] == 0:
                f.write(f"\n[{r['subgrupo']}] N=0 (vazio)\n")
                continue
            f.write(f"\n[{r['subgrupo']}]  N={r['n']}  "
                    f"(fake={r['n_fake_real']}, real={r['n_real_real']})\n")
            f.write(f"  RF:   acc={r['acc_rf']:.4f}  (delta vs baseline = {r['delta_acc_rf']:+.4f})  "
                    f"%pred_fake={r['pct_pred_fake_rf']:.3f}\n")
            f.write(f"  SAGE: acc={r['acc_sage']:.4f}  (delta vs baseline = {r['delta_acc_sage']:+.4f})  "
                    f"%pred_fake={r['pct_pred_fake_sage']:.3f}\n")
            if r.get("f1m_rf") is not None:
                f.write(f"  F1-macro (subgrupo mixed): RF={r['f1m_rf']:.4f}  SAGE={r['f1m_sage']:.4f}\n")

        f.write("\n--- Interpretacao chave ---\n")
        # Diagnostico automatico de "inversao distributiva"
        for nome in ["real_viral", "fake_contido"]:
            r = next((x for x in resultados if x["subgrupo"] == nome), None)
            if r is None or r["n"] == 0: continue
            acc_sage = r["acc_sage"]
            if acc_sage < 0.20:
                f.write(f"  [{nome}] SAGE acc={acc_sage:.3f} < 0.20 = INVERSAO confirmada.\n"
                        f"    Subgrupo onde a regra 'fake=alto grau' aprendida falha sistematicamente.\n"
                        f"    Sustenta numericamente a narrativa de inversao distributiva da §4.6.\n")
            elif acc_sage < 0.50:
                f.write(f"  [{nome}] SAGE acc={acc_sage:.3f} < chance = degradacao forte (mas nao inversao plena).\n")
            else:
                f.write(f"  [{nome}] SAGE acc={acc_sage:.3f} -- modelo resiste, sem inversao clara.\n")
    print(f"[OK] {rel.name}")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
