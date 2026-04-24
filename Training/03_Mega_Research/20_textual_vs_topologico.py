"""
20_textual_vs_topologico.py
---------------------------
Compara classificador TEXTUAL (LogReg sobre x[0]) vs TOPOLOGICO (RF estrutural)
em concordancia/discordancia. Responde a pergunta:

  "Os dois modelos veem a mesma coisa? Quando discordam, quem acerta?"

Roda no UPFD-GossipCop (caso forte: ambos tem labels, ambos atingem F1>0.7).
Tambem aplica no Bluesky para mostrar a distribuicao da discordancia em dados
sem ground truth.

Saida em Execution/results/figuras_tcc/textual_vs_topologico/:
  gossipcop_concordancia.csv     -- (post_id, score_text, score_topo, pred_text,
                                     pred_topo, label_real, concorda)
  gossipcop_metricas.txt         -- agreement, kappa, matriz 2x2, F1 quando discordam
  bluesky_concordancia.csv       -- por feed: distribuicao de discordancia
  fig_concordancia.png           -- scatter plot scores text vs topo
  casos_discordancia.txt         -- top 10 casos onde modelos mais divergem

Uso:
  python 20_textual_vs_topologico.py
"""

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              confusion_matrix, f1_score)
from torch_geometric.datasets import UPFD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ        = Path(__file__).resolve().parent.parent.parent
BSKY_DIR    = RAIZ / "dados_bluesky"
FEED_DIR    = BSKY_DIR / "feed_posts"
MAT_DIR     = RAIZ / "Material"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "textual_vs_topologico"

RANDOM_SEED = 42


def feats_topologicas(grafos):
    X = []
    for g in grafos:
        n = g.num_nodes
        gr = (g.edge_index[0] == 0).sum().item() if g.edge_index.numel() > 0 else 0
        X.append([n, gr])
    return np.array(X, dtype=np.float64)


def feats_textuais(grafos):
    """x[0] = embedding da raiz (notícia)."""
    import torch as _torch
    return _torch.stack([g.x[0] for g in grafos]).numpy()


def labels(grafos):
    return np.array([g.y.item() for g in grafos], dtype=np.int64)


def matriz_concordancia(p_text, p_topo, y_real=None):
    """
    Calcula matriz 2x2 de concordancia entre 2 classificadores.
    Quando y_real disponivel, mede F1 dos casos concordantes/discordantes.
    """
    n = len(p_text)
    concorda = (p_text == p_topo).sum()
    discorda = n - concorda
    matriz = np.zeros((2, 2), dtype=int)
    for pt, pk in zip(p_text, p_topo):
        matriz[int(pt), int(pk)] += 1
    out = {
        "n": int(n),
        "concorda": int(concorda),
        "discorda": int(discorda),
        "agreement_rate": round(concorda / max(n, 1), 4),
        "matriz_2x2": matriz.tolist(),  # [pred_text][pred_topo]
        "kappa": round(float(cohen_kappa_score(p_text, p_topo)), 4),
    }
    if y_real is not None:
        # F1 quando os 2 concordam vs quando discordam
        idx_c = np.where(p_text == p_topo)[0]
        idx_d = np.where(p_text != p_topo)[0]
        if len(idx_c):
            out["f1_quando_concordam"]  = round(float(f1_score(y_real[idx_c], p_text[idx_c], average="macro", zero_division=0)), 4)
        if len(idx_d):
            out["f1_text_quando_discordam"] = round(float(f1_score(y_real[idx_d], p_text[idx_d], average="macro", zero_division=0)), 4)
            out["f1_topo_quando_discordam"] = round(float(f1_score(y_real[idx_d], p_topo[idx_d], average="macro", zero_division=0)), 4)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Textual vs Topologico -- concordancia, discordancia, dominio de cada um")
    print("=" * 72)

    # ── 1. Carregar GossipCop ────────────────────────────────────────────
    print("\n[1/4] Carregando UPFD-GossipCop (com labels)...")
    train = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="content", split="train"))
    val   = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="content", split="val"))
    test  = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="content", split="test"))
    print(f"   Train+Val={len(train)+len(val)}  Test={len(test)}")

    Xt_tr = feats_textuais(train + val)
    Xt_te = feats_textuais(test)
    Xk_tr = feats_topologicas(train + val)
    Xk_te = feats_topologicas(test)
    y_tr  = labels(train + val)
    y_te  = labels(test)

    # Treinar 2 modelos no GossipCop
    print("\n[2/4] Treinando LogReg-textual e RF-topologico no GossipCop...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(Xt_tr, y_tr)
    p_text = lr.predict(Xt_te)
    s_text = lr.predict_proba(Xt_te)[:, list(lr.classes_).index(0)]  # prob fake

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(Xk_tr, y_tr)
    p_topo = rf.predict(Xk_te)
    s_topo = rf.predict_proba(Xk_te)[:, list(rf.classes_).index(0)]

    f1_text = f1_score(y_te, p_text, average="macro", zero_division=0)
    f1_topo = f1_score(y_te, p_topo, average="macro", zero_division=0)
    print(f"   F1m textual : {f1_text:.4f}")
    print(f"   F1m topologico : {f1_topo:.4f}")

    info = matriz_concordancia(p_text, p_topo, y_te)
    print(f"\n   --- Concordancia ---")
    print(f"   agreement = {info['agreement_rate']:.2%}  (kappa={info['kappa']})")
    print(f"   matriz 2x2 (rows=textual, cols=topologico):")
    print(f"     pred_text\\pred_topo   real(0)  fake(1)? olhar mapeamento")
    print(f"   {info['matriz_2x2']}")
    print(f"\n   F1 quando concordam : {info.get('f1_quando_concordam','-')}")
    print(f"   F1 textual (quando discordam) : {info.get('f1_text_quando_discordam','-')}")
    print(f"   F1 topologico (quando discordam): {info.get('f1_topo_quando_discordam','-')}")

    # CSV completo
    csv_g = OUT_DIR / "gossipcop_concordancia.csv"
    with open(csv_g, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","label_real","score_text","pred_text","score_topo","pred_topo",
                    "concorda","ambos_acertam","textual_acerta_solo","topo_acerta_solo"])
        for i, (yr, st, pt, sk, pk) in enumerate(zip(y_te, s_text, p_text, s_topo, p_topo)):
            concorda = bool(pt == pk)
            text_ok = bool(pt == yr)
            topo_ok = bool(pk == yr)
            w.writerow([i, int(yr),
                        round(float(st), 4), int(pt),
                        round(float(sk), 4), int(pk),
                        concorda,
                        text_ok and topo_ok,
                        text_ok and not topo_ok,
                        topo_ok and not text_ok])
    print(f"   [OK] {csv_g.name}")

    metrics_path = OUT_DIR / "gossipcop_metricas.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Textual vs Topologico -- UPFD-GossipCop\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"F1 macro textual:  {f1_text:.4f}\n")
        f.write(f"F1 macro topologico: {f1_topo:.4f}\n\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")

    # Casos top discordantes
    deltas = np.abs(s_text - s_topo)
    top_disc = np.argsort(-deltas)[:15]
    casos_path = OUT_DIR / "casos_discordancia.txt"
    with open(casos_path, "w", encoding="utf-8") as f:
        f.write("Top 15 casos onde textual e topologico mais divergem\n")
        f.write("(maior |score_text - score_topo|)\n")
        f.write("=" * 60 + "\n\n")
        for i in top_disc:
            f.write(f"idx={i:>4}  label={y_te[i]}  s_text={s_text[i]:.3f}  "
                    f"s_topo={s_topo[i]:.3f}  delta={deltas[i]:.3f}  "
                    f"text_pred={p_text[i]} topo_pred={p_topo[i]}  "
                    f"text_ok={p_text[i]==y_te[i]}  topo_ok={p_topo[i]==y_te[i]}\n")
    print(f"   [OK] {casos_path.name}")

    # Plot scatter scores
    print("\n[3/4] Gerando figura scatter textual vs topologico...")
    fig, ax = plt.subplots(figsize=(8, 8))
    cores_real = ["#5BAD72" if y == 1 else "#E07B54" for y in y_te]
    ax.scatter(s_text, s_topo, c=cores_real, alpha=0.4, s=20, edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="concordancia perfeita")
    ax.axhline(0.5, color="black", lw=0.5, alpha=0.5)
    ax.axvline(0.5, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("Score fake -- Textual (LogReg sobre x[0])")
    ax.set_ylabel("Score fake -- Topologico (RF [num_nodes, grau_root])")
    ax.set_title(f"Textual vs Topologico em UPFD-GossipCop\n"
                 f"agreement={info['agreement_rate']:.1%}  kappa={info['kappa']}")
    # legenda
    from matplotlib.lines import Line2D
    legend_el = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#E07B54', label='real label = fake (0)', markersize=8),
                 Line2D([0],[0], marker='o', color='w', markerfacecolor='#5BAD72', label='real label = real (1)', markersize=8),
                 Line2D([0],[0], linestyle='--', color='gray', label='concordancia perfeita')]
    ax.legend(handles=legend_el, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pfig = OUT_DIR / "fig_concordancia.png"
    plt.savefig(pfig, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {pfig.name}")

    # ── 4. Bluesky: aplicar ambos modelos sobre cascatas, ver discordancia ──
    print("\n[4/4] Aplicando ambos modelos no Bluesky (sem ground truth)...")
    print("   Carregando RF persistido (treinado em GossipCop) e treinando LogReg-content...")
    # LogReg precisa do BERT/embedding -- mas Bluesky nao tem embedding pronto.
    # Para a comparacao no Bluesky, usamos APENAS o RF estrutural (que treinamos
    # acima no content) e reportamos so a distribuicao de scores topologicos.
    # A comparacao plena texto-vs-topo no Bluesky precisaria gerar BERT em todos
    # os 168k posts, o que ja vimos que e custoso. Aqui apenas resumimos:

    # Conta posts por feed e calcula score topologico baseado em (reply+repost) como nodes
    bsky_rows = []
    for arq in sorted(FEED_DIR.glob("*.jsonl")):
        feed = arq.stem
        with open(arq, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                interacoes = int(p.get("reply_count") or 0) + int(p.get("repost_count") or 0)
                bsky_rows.append([feed, p.get("post_id"), 1 + interacoes, interacoes])

    X_bsky = np.array([[r[2], r[3]] for r in bsky_rows], dtype=np.float64)
    s_bsky = rf.predict_proba(X_bsky)[:, list(rf.classes_).index(0)]
    p_bsky = rf.predict(X_bsky)
    csv_b = OUT_DIR / "bluesky_concordancia.csv"
    with open(csv_b, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feed","post_id","num_nodes","grau_root","score_topo","pred_topo"])
        for r, s, pp in zip(bsky_rows, s_bsky, p_bsky):
            w.writerow(r + [round(float(s), 4), int(pp)])
    print(f"   [OK] {csv_b.name} ({len(bsky_rows)} linhas)")

    # Resumo
    print(f"\n   Bluesky scores topologicos (sem texto):")
    feeds_unicos = sorted({r[0] for r in bsky_rows})
    for feed in feeds_unicos:
        idx = [i for i, r in enumerate(bsky_rows) if r[0] == feed]
        sub = s_bsky[idx]
        pf = (p_bsky[idx] == 0).mean()
        print(f"     {feed:<25} n={len(idx):>6}  score_medio={sub.mean():.4f}  pred_fake_pct={100*pf:.1f}%")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
