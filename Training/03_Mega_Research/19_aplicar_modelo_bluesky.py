"""
19_aplicar_modelo_bluesky.py
----------------------------
DEMO de aplicacao dos modelos persistidos (Fase 17) sobre dados reais do
Bluesky. Como o Bluesky nao tem labels fake/real, esta etapa NAO mede F1 --
produz distribuicao de scores e ranking de posts mais "fake-like" por feed
para inspecao qualitativa.

Para cada post de cada feed (feed_posts/*.jsonl):
  - num_nodes  = 1 + reply_count + repost_count + (quote_count se houver)
  - grau_root  = num_nodes - 1
  - Aplicar RF estrutural (treinado em GossipCop) para obter score fake-like

Saida em Execution/results/figuras_tcc/bluesky_inferencia/:
  scores_por_post.csv         -- (feed, post_id, num_nodes, score, pred)
  resumo_por_feed.csv         -- por feed: media, mediana, p95 do score
  fig_distribuicao_scores.png -- KDE/hist de scores por feed
  top_suspeitos.txt           -- top 5 posts mais "fake-like" por feed (texto + score)

Tambem reconstroi 6 amostras visualizaveis (3 alta-score + 3 baixa-score) como
HTML PyVis em amostras_visualizacao/.

Uso:
  python 19_aplicar_modelo_bluesky.py
"""

import json
import csv
import pickle
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
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "bluesky_inferencia"
VIZ_DIR     = OUT_DIR / "amostras_visualizacao"


def carregar_rf():
    with open(WEIGHTS_DIR / "rf_struct_gossipcop.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["feature_names"], d["label_map"]


def construir_features_post(post: dict) -> tuple:
    """Retorna (num_nodes, grau_root) para um post Bluesky."""
    interacoes = (int(post.get("reply_count")  or 0)
                  + int(post.get("repost_count") or 0))
    # quote_count campo nao standard; vamos contar quotes via campo separado se houver
    if isinstance(post.get("quotes"), int):
        interacoes += post["quotes"]
    num_nodes = 1 + interacoes
    grau_root = interacoes
    return num_nodes, grau_root


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Bluesky -- Inferencia com modelo treinado em GossipCop (RF estrutural)")
    print("=" * 72)

    print("\n[1/4] Carregando RF estrutural persistido...")
    rf, feat_names, label_map = carregar_rf()
    print(f"  Modelo: RF({rf.n_estimators} arvores) | features={feat_names} | labels={label_map}")

    print("\n[2/4] Carregando posts do Bluesky e computando scores...")
    rows = []
    for arq in sorted(FEED_DIR.glob("*.jsonl")):
        feed = arq.stem
        n = 0
        with open(arq, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                num_nodes, grau_root = construir_features_post(p)
                rows.append({
                    "feed": feed, "post_id": p.get("post_id"),
                    "user_id": p.get("user_id"),
                    "num_nodes": num_nodes, "grau_root": grau_root,
                    "reply_count":  int(p.get("reply_count")  or 0),
                    "repost_count": int(p.get("repost_count") or 0),
                    "like_count":   int(p.get("like_count")   or 0),
                    "text": (p.get("text") or "")[:200],  # primeiros 200 chars
                })
                n += 1
        print(f"  {feed:<25} {n:>6} posts")

    print(f"\n  Total: {len(rows):,} posts")

    # Predicao em batch
    print("\n[3/4] Aplicando RF estrutural para obter score 'fake-like'...")
    X = np.array([[r["num_nodes"], r["grau_root"]] for r in rows], dtype=np.float64)
    probs = rf.predict_proba(X)
    # label 0 = fake, label 1 = real -> queremos prob de classe 0 (fake)
    classes = list(rf.classes_)
    idx_fake = classes.index(0) if 0 in classes else 0
    scores = probs[:, idx_fake]
    preds = rf.predict(X)

    for r, s, p_ in zip(rows, scores, preds):
        r["score_fake"] = round(float(s), 4)
        r["pred"]       = int(p_)

    # CSV completo
    csv_path = OUT_DIR / "scores_por_post.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["feed","post_id","user_id","num_nodes","grau_root",
                "reply_count","repost_count","like_count","score_fake","pred"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"  [OK] {csv_path.name} ({len(rows)} linhas)")

    # Resumo por feed
    feeds_unicos = sorted({r["feed"] for r in rows})
    resumo_csv = OUT_DIR / "resumo_por_feed.csv"
    with open(resumo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feed","n_posts","score_medio","score_mediana","score_p95",
                    "pct_pred_fake"])
        for feed in feeds_unicos:
            sub = [r for r in rows if r["feed"] == feed]
            sc = [r["score_fake"] for r in sub]
            pf = sum(1 for r in sub if r["pred"] == 0) / max(len(sub), 1)
            w.writerow([feed, len(sub),
                        round(np.mean(sc), 4),
                        round(np.median(sc), 4),
                        round(np.percentile(sc, 95), 4),
                        round(pf, 4)])
    print(f"  [OK] {resumo_csv.name}")

    # Print sumario
    print("\n  --- Score 'fake-like' medio por feed (RF treinado em GossipCop) ---")
    print(f"  {'feed':<25} {'n':>7} {'media':>8} {'mediana':>9} {'p95':>8} {'%pred=fake':>11}")
    print(f"  {'-'*72}")
    for feed in feeds_unicos:
        sub = [r for r in rows if r["feed"] == feed]
        sc = [r["score_fake"] for r in sub]
        pf = sum(1 for r in sub if r["pred"] == 0) / max(len(sub), 1)
        print(f"  {feed:<25} {len(sub):>7} {np.mean(sc):>8.4f} {np.median(sc):>9.4f} "
              f"{np.percentile(sc, 95):>8.4f} {100*pf:>10.1f}%")

    # Top-K mais suspeitos por feed (apenas top 5)
    print("\n  --- TOP 5 posts mais 'fake-like' por feed ---")
    top_path = OUT_DIR / "top_suspeitos.txt"
    with open(top_path, "w", encoding="utf-8") as f:
        for feed in feeds_unicos:
            sub = sorted([r for r in rows if r["feed"]==feed],
                         key=lambda x: -x["score_fake"])[:5]
            f.write(f"\n=== {feed} ===\n")
            for r in sub:
                f.write(f"  [score={r['score_fake']:.3f}  N={r['num_nodes']:>5}  "
                        f"likes={r['like_count']:>4}]  {r['text']}\n")
    print(f"  [OK] {top_path.name}")

    # Plot: distribuicao de scores por feed
    print("\n[4/4] Gerando figura de distribuicao de scores...")
    fig, ax = plt.subplots(figsize=(14, 7))
    cores = plt.cm.tab10(np.linspace(0, 1, len(feeds_unicos)))
    for feed, c in zip(feeds_unicos, cores):
        sub = [r["score_fake"] for r in rows if r["feed"]==feed]
        if len(sub) < 30: continue  # ignora feeds com poucos posts
        ax.hist(sub, bins=50, alpha=0.4, label=f"{feed} (n={len(sub)})",
                density=True, color=c)
    ax.set_xlabel("Score 'fake-like' (RF estrutural treinado em GossipCop)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuicao de scores por feed Bluesky\n"
                 "(NB: nao ha ground truth -- analise descritiva)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pfig = OUT_DIR / "fig_distribuicao_scores.png"
    plt.savefig(pfig, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  [OK] {pfig.name}")

    # Visualizacao: 3 mais alto-score + 3 medio-score (varios tamanhos) como PyVis
    print("\n  Exportando 6 amostras de grafos visualizaveis...")
    try:
        from pyvis.network import Network
    except ImportError:
        print("  [WARN] pyvis nao instalado; pulando visualizacao")
    else:
        # Filtra posts com num_nodes >= 5 para ter algo a mostrar
        candidatos = [r for r in rows if r["num_nodes"] >= 5]
        candidatos.sort(key=lambda x: -x["score_fake"])
        amostras = []
        if len(candidatos) >= 3:
            amostras += candidatos[:3]      # 3 mais "fake-like"
            amostras += candidatos[len(candidatos)//2:len(candidatos)//2+3]  # 3 medianos
        for r in amostras:
            net = Network(height="500px", width="100%", directed=False, notebook=False,
                          heading=f"{r['feed']} | post={r['post_id']} | "
                                  f"score={r['score_fake']:.3f} | N={r['num_nodes']}")
            net.add_node(0, label="RAIZ", color="#E07B54", size=25)
            for i in range(1, r["num_nodes"]):
                net.add_node(i, label=f"u{i}", color="#5BAD72", size=10)
                net.add_edge(0, i)
            net.barnes_hut(spring_length=80)
            out = VIZ_DIR / f"{r['feed']}_post{r['post_id']}_n{r['num_nodes']}_score{int(r['score_fake']*100):03d}.html"
            try:
                net.write_html(str(out), notebook=False, open_browser=False)
            except Exception:
                pass
        print(f"  [OK] {len(amostras)} HTMLs em {VIZ_DIR.name}/")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
