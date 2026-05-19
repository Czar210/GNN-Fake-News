"""
gerar_figura_heavy_tail.py
--------------------------
Helper one-off: gera figura educativa mostrando que as metricas
estruturais do UPFD-GossipCop sao heavy-tail (nao Gaussianas), justificando
o uso de percentil empirico (nao z-score) na definicao de outliers do script 29.

Saida: Photos/orientador_fase9/00_o_que_e_heavy_tail.png

Uso:
  python gerar_figura_heavy_tail.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ   = Path(__file__).resolve().parent.parent.parent
IN_CSV = RAIZ / "Execution" / "results" / "figuras_tcc" / "estratificacao" / "por_grafo.csv"
OUT    = RAIZ / "Photos" / "orientador_fase9" / "00_o_que_e_heavy_tail.png"


def carregar(metrica: str):
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        return np.array([float(r[metrica]) for r in csv.DictReader(f)])


def painel(ax, x, nome, log=False):
    """Histograma com Gaussiana ajustada sobreposta + linhas de percentil."""
    media = x.mean(); std = x.std()
    p50, p95, p99 = np.percentile(x, [50, 95, 99])
    g_p95 = media + 1.645 * std  # p95 esperado se fosse Gaussiana

    if log:
        bins = np.geomspace(max(x.min(), 1), x.max(), 50)
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        bins = np.linspace(x.min(), x.max(), 50)

    ax.hist(x, bins=bins, color="#4F86C6", alpha=0.75, edgecolor="white")

    xs = np.linspace(x.min(), x.max(), 400) if not log \
         else np.geomspace(max(x.min(), 1), x.max(), 400)
    gauss = stats.norm.pdf(xs, loc=media, scale=std) * len(x) * (bins[1] - bins[0]
            if not log else (np.log(bins[1]) - np.log(bins[0])) * bins[0])
    if not log:
        ax.plot(xs, gauss, color="#E07B54", lw=2.2, label="Gaussiana ajustada")

    ax.axvline(p50,   color="#5BAD72", ls="-",  lw=1.4, label=f"p50={p50:.1f}")
    ax.axvline(p95,   color="#5BAD72", ls="--", lw=1.4, label=f"p95={p95:.1f}")
    ax.axvline(g_p95, color="#E07B54", ls=":",  lw=2,
               label=f"$\\mu+1.645\\sigma$={g_p95:.1f}\n(p95 se fosse Gauss.)")

    razao = x.max() / max(p50, 0.001)
    ax.set_title(f"{nome}\nmax={x.max():.1f}, media={media:.2f}, "
                 f"max/mediana={razao:.1f}$\\times$" + ("  [log-log]" if log else ""),
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, which="both" if log else "major")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nn = carregar("num_nodes")
    dp = carregar("depth_max")
    br = carregar("branching_avg")
    gr = carregar("grau_root")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    painel(axes[0,0], nn, "num_nodes")
    painel(axes[0,1], gr, "grau_root  (engajamento da raiz)")
    painel(axes[1,0], br, "branching_avg")
    painel(axes[1,1], gr, "grau_root  (mesma metrica, escala log-log)", log=True)

    plt.suptitle(
        "Por que usamos PERCENTIL EMPIRICO e nao z-score pra definir outliers\n"
        "(distribuicoes do UPFD-GossipCop test, n=3826 grafos)",
        fontsize=13, y=1.00, fontweight="bold")

    msg = (
        "Gaussiana (laranja) tenta cobrir a distribuicao mas erra: "
        "(i) extende-se para valores negativos sem sentido fisico; "
        "(ii) subestima a cauda direita (p95 real >> $\\mu+1.645\\sigma$). "
        "z-score >2 = 'outlier' seria mal-definido. Percentil p95 e' invariante a forma da distribuicao.\n"
        "Painel inferior-direito: 'grau_root' em log-log revela linha quase reta = lei de potencia (assinatura de heavy-tail genuino)."
    )
    fig.text(0.5, -0.04, msg, ha="center", fontsize=9, style="italic",
             wrap=True, color="#444")

    plt.tight_layout()
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] {OUT}")

    print("\nNumeros chave (UPFD-GossipCop test):")
    for nome, x in [("num_nodes", nn), ("grau_root", gr),
                    ("depth_max", dp), ("branching_avg", br)]:
        p50 = np.percentile(x, 50); p95 = np.percentile(x, 95)
        g_p95 = x.mean() + 1.645 * x.std()
        print(f"  {nome:<14} max/mediana={x.max()/max(p50,0.001):6.1f}x  "
              f"p95={p95:6.1f}  vs Gauss-p95={g_p95:6.1f}  "
              f"razao={p95/max(g_p95,0.001):.2f}x")


if __name__ == "__main__":
    main()
