"""
18_analise_bluesky_crossfeed.py
-------------------------------
Replica a analise estrutural Cohen's d (Fase analise_estrutural) entre FEEDS do
Bluesky em vez de fake-vs-real. Como o Bluesky nao tem labels fake/real, usamos
"feed do post" como proxy (Political Science vs Science vs Blacksky etc.) para
verificar se a propagacao e estruturalmente distinta entre comunidades.

Por post (de feed_posts/*.jsonl):
  - reply_count, repost_count, like_count (campos do post)
  - thread_size: derivado de threads.txt (se thread_root for o post_id)

Comparacoes:
  - Cohen's d par a par entre feeds (em cada metrica)
  - Distribuicoes (boxplots por feed)
  - Top feeds por viralizacao

Saida em Execution/results/figuras_tcc/bluesky_crossfeed/:
  estatisticas_por_feed.csv     -- (feed, metrica): media, std, mediana, p95
  pares_cohens_d.csv            -- (feed_a, feed_b, metrica): cohen_d
  fig_distribuicoes.png         -- boxplot por feed para cada metrica
  fig_pares.png                 -- heatmap Cohen's d entre feeds

Uso:
  python 18_analise_bluesky_crossfeed.py
"""

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ        = Path(__file__).resolve().parent.parent.parent
BSKY_DIR    = RAIZ / "dados_bluesky"
FEED_DIR    = BSKY_DIR / "feed_posts"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "bluesky_crossfeed"

METRICAS = ["reply_count", "repost_count", "like_count"]


def carregar_feed(arq: Path) -> list:
    feed = arq.stem
    posts = []
    with open(arq, encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except Exception:
                continue
            posts.append({
                "post_id":      p.get("post_id"),
                "feed":         feed,
                "reply_count":  int(p.get("reply_count")  or 0),
                "repost_count": int(p.get("repost_count") or 0),
                "like_count":   int(p.get("like_count")   or 0),
                "is_reply":     p.get("reply_to") is not None,
            })
    return posts


def cohen_d(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((a.std(ddof=1)**2 * (len(a)-1)) + (b.std(ddof=1)**2 * (len(b)-1)))
                     / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / max(pooled, 1e-9))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Bluesky -- Analise estrutural cross-feed (Cohen's d)")
    print("=" * 72)

    feeds = sorted(FEED_DIR.glob("*.jsonl"))
    print(f"\n[1/3] Carregando {len(feeds)} feeds...")
    todos = []
    for arq in feeds:
        posts = carregar_feed(arq)
        todos += posts
        print(f"  {arq.stem:<25} {len(posts):>6} posts")

    print(f"\n  Total: {len(todos):,} posts")

    # Estatisticas por feed
    rows_csv = []
    feed_metrics = defaultdict(dict)  # feed_metrics[feed][metrica] = list
    feeds_unicos = sorted({p["feed"] for p in todos})
    for feed in feeds_unicos:
        sub = [p for p in todos if p["feed"] == feed]
        for met in METRICAS:
            vals = [p[met] for p in sub]
            feed_metrics[feed][met] = vals
            rows_csv.append({
                "feed": feed, "metrica": met, "n": len(vals),
                "media": round(np.mean(vals), 3),
                "std":   round(np.std(vals), 3),
                "mediana": round(np.median(vals), 1),
                "p95":   round(np.percentile(vals, 95), 1),
                "max":   int(np.max(vals)),
            })
    csv_stats = OUT_DIR / "estatisticas_por_feed.csv"
    with open(csv_stats, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feed","metrica","n","media","std","mediana","p95","max"])
        w.writeheader(); w.writerows(rows_csv)
    print(f"\n[2/3] Estatisticas: {csv_stats.name}")

    # Print sumario
    print(f"\n  --- Sumario (media de cada metrica por feed) ---")
    print(f"  {'feed':<25} {'reply':>10} {'repost':>10} {'like':>12}")
    print(f"  {'-'*60}")
    for feed in feeds_unicos:
        m = feed_metrics[feed]
        print(f"  {feed:<25} {np.mean(m['reply_count']):>10.2f} "
              f"{np.mean(m['repost_count']):>10.2f} "
              f"{np.mean(m['like_count']):>12.2f}")

    # Cohen's d par a par para cada metrica
    print(f"\n[3/3] Cohen's d par a par...")
    pares_csv = []
    for met in METRICAS:
        for i, fa in enumerate(feeds_unicos):
            for fb in feeds_unicos[i+1:]:
                d = cohen_d(feed_metrics[fa][met], feed_metrics[fb][met])
                pares_csv.append({"feed_a": fa, "feed_b": fb, "metrica": met,
                                  "cohen_d": round(d, 3)})
    csv_pares = OUT_DIR / "pares_cohens_d.csv"
    with open(csv_pares, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feed_a","feed_b","metrica","cohen_d"])
        w.writeheader(); w.writerows(pares_csv)
    print(f"   {csv_pares.name}")

    # Maiores Cohen's d
    print("\n  --- Top 10 maiores |d| (mais separabilidade) ---")
    top = sorted(pares_csv, key=lambda r: -abs(r["cohen_d"]))[:10]
    for r in top:
        print(f"  {r['feed_a']:<22} vs {r['feed_b']:<22} {r['metrica']:<14} d={r['cohen_d']:+.3f}")

    # Plot 1: distribuicoes (boxplot log) por feed para cada metrica
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, met in zip(axes, METRICAS):
        dados = [feed_metrics[f][met] for f in feeds_unicos]
        bp = ax.boxplot(dados, labels=feeds_unicos, showfliers=False, patch_artist=True)
        for patch, _ in zip(bp["boxes"], feeds_unicos):
            patch.set_facecolor("#5BAD72"); patch.set_alpha(0.6)
        ax.set_yscale("symlog")
        ax.set_title(f"{met} por feed (boxplot, escala symlog)")
        ax.set_ylabel(met)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = OUT_DIR / "fig_distribuicoes.png"
    plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\n  [OK] {p1.name}")

    # Plot 2: heatmap Cohen's d (so a metrica mais informativa: repost_count)
    n = len(feeds_unicos)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, met in zip(axes, METRICAS):
        M = np.zeros((n, n))
        for r in pares_csv:
            if r["metrica"] != met: continue
            i = feeds_unicos.index(r["feed_a"])
            j = feeds_unicos.index(r["feed_b"])
            M[i, j] = r["cohen_d"]
            M[j, i] = -r["cohen_d"]
        vmax = max(abs(M).max(), 0.1)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(feeds_unicos, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(feeds_unicos, fontsize=8)
        ax.set_title(f"Cohen's d cross-feed -- {met}")
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                            fontsize=6, color="black" if abs(M[i,j]) < 0.5*vmax else "white")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    p2 = OUT_DIR / "fig_pares.png"
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  [OK] {p2.name}")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
