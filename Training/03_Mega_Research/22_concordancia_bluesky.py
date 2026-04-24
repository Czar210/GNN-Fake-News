"""
22_concordancia_bluesky.py
--------------------------
Compara predicao TEXTUAL vs TOPOLOGICA em posts reais do Bluesky (sem ground
truth) e analisa OND eles concordam vs discordam.

Estrategia:
  1. Aplicar RF estrutural (treinado em GossipCop) em TODOS os 168k posts
  2. Amostrar 5k posts estratificados (por feed x bin de num_nodes)
  3. Gerar BERT desses 5k e aplicar LogReg-BERT (treinado em FNN)
  4. Cross-tab pred_textual x pred_topologico
  5. Agreement rate por:
     - bin de num_nodes (pequeno/medio/grande)
     - feed
     - bin de comprimento do texto
     - bin de engajamento (likes)

Saida em Execution/results/figuras_tcc/concordancia_bluesky/:
  amostra_5k.csv             -- (post_id, feed, num_nodes, text_len, score_text,
                                pred_text, score_topo, pred_topo, concorda)
  matriz_confusao.txt        -- matriz 2x2 + agreement + kappa
  cross_tab.csv              -- agreement_rate por bin/feed
  fig_matriz_confusao.png    -- heatmap pred_text vs pred_topo
  fig_concordancia_por_*.png -- agreement_rate vs num_nodes/text_len/feed

Uso:
  python 22_concordancia_bluesky.py
"""

import csv
import json
import pickle
import sys
import time
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
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "concordancia_bluesky"

RANDOM_SEED = 42
N_AMOSTRA   = 5000   # posts a embeddar com BERT
BINS_NOS    = [(1, 2), (2, 5), (5, 20), (20, 100), (100, 100000)]


def carregar_modelos():
    with open(WEIGHTS_DIR / "rf_struct_gossipcop.pkl", "rb") as f:
        rf = pickle.load(f)["model"]
    with open(WEIGHTS_DIR / "logreg_bert_fnn.pkl", "rb") as f:
        lr = pickle.load(f)["model"]
    return rf, lr


def carregar_posts():
    rows = []
    for arq in sorted(FEED_DIR.glob("*.jsonl")):
        feed = arq.stem
        with open(arq, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                interacoes = int(p.get("reply_count") or 0) + int(p.get("repost_count") or 0)
                rows.append({
                    "feed": feed,
                    "post_id": p.get("post_id"),
                    "text": (p.get("text") or "")[:500],
                    "text_len": len(p.get("text") or ""),
                    "num_nodes": 1 + interacoes,
                    "grau_root": interacoes,
                    "like_count": int(p.get("like_count") or 0),
                })
    return rows


def bin_idx(n: int, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= n < hi:
            return i
    return len(bins) - 1


def amostragem_estratificada(rows, n_alvo, bins, seed=RANDOM_SEED):
    """Amostra n_alvo posts estratificado por (feed, bin de num_nodes)."""
    rng = np.random.default_rng(seed)
    grupos = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["feed"], bin_idx(r["num_nodes"], bins))
        grupos[key].append(i)
    n_grupos = len(grupos)
    n_por_grupo = max(1, n_alvo // n_grupos)
    selecionados = []
    for key, idxs in grupos.items():
        k = min(n_por_grupo, len(idxs))
        sel = rng.choice(idxs, size=k, replace=False).tolist()
        selecionados.extend(sel)
    return selecionados


def kappa(a, b):
    """Cohen's kappa simplificado."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    po = (a == b).mean()
    p_a = np.bincount(a, minlength=2) / n
    p_b = np.bincount(b, minlength=2) / n
    pe = (p_a * p_b).sum()
    return (po - pe) / max(1 - pe, 1e-9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Concordancia Bluesky -- TEXTUAL vs TOPOLOGICO post a post")
    print("=" * 72)

    print("\n[1/5] Carregando modelos persistidos...")
    rf_topo, lr_text = carregar_modelos()
    print(f"   RF topologico: {rf_topo.n_estimators} arvores")
    print(f"   LogReg textual: input_dim={lr_text.coef_.shape[1]}")

    print("\n[2/5] Carregando todos posts Bluesky...")
    rows = carregar_posts()
    print(f"   {len(rows):,} posts")

    print("\n[3/5] Aplicando RF topologico em TODOS os posts...")
    X_topo_all = np.array([[r["num_nodes"], r["grau_root"]] for r in rows], dtype=np.float64)
    s_topo_all = rf_topo.predict_proba(X_topo_all)[:, list(rf_topo.classes_).index(0)]
    p_topo_all = (s_topo_all >= 0.5).astype(int)
    # 0=fake, 1=real (mantemos pred=0 quando score >=0.5)
    p_topo_all_label = np.where(p_topo_all == 1, 0, 1)  # se score>=0.5 -> classe fake (0)
    # ajuste: predict_proba devolve [:, idx_fake] = score fake; pred = 0 se score>=0.5
    p_topo_all = np.where(s_topo_all >= 0.5, 0, 1)
    print(f"   Score topologico medio: {s_topo_all.mean():.4f} | pred=fake={(p_topo_all==0).sum():,} ({100*(p_topo_all==0).mean():.1f}%)")

    print(f"\n[4/5] Amostragem estratificada de {N_AMOSTRA} posts...")
    sel = amostragem_estratificada(rows, N_AMOSTRA, BINS_NOS)
    print(f"   Amostrados: {len(sel)} posts")

    # BERT nos amostrados
    print(f"\n   Carregando BERT (sentence-transformers)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")
    textos = [rows[i]["text"] for i in sel]
    print(f"   Encodando {len(textos)} textos (batch=32)...")
    t0 = time.time()
    embs = model.encode(textos, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    print(f"   BERT pronto em {time.time()-t0:.1f}s | shape={embs.shape}")

    # Aplicar LogReg-BERT
    s_text = lr_text.predict_proba(embs)[:, list(lr_text.classes_).index(0)]
    p_text = np.where(s_text >= 0.5, 0, 1)
    print(f"   Score textual medio: {s_text.mean():.4f} | pred=fake={(p_text==0).sum()} ({100*(p_text==0).mean():.1f}%)")

    # Combinar com topologico nos amostrados
    s_topo = s_topo_all[sel]
    p_topo = p_topo_all[sel]

    # Salvar amostra_5k.csv
    csv_a = OUT_DIR / "amostra_5k.csv"
    with open(csv_a, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx_global","post_id","feed","num_nodes","text_len","like_count",
                    "score_text","pred_text","score_topo","pred_topo","concorda"])
        for j, i_g in enumerate(sel):
            r = rows[i_g]
            concorda = bool(p_text[j] == p_topo[j])
            w.writerow([i_g, r["post_id"], r["feed"], r["num_nodes"], r["text_len"],
                        r["like_count"],
                        round(float(s_text[j]),4), int(p_text[j]),
                        round(float(s_topo[j]),4), int(p_topo[j]),
                        concorda])
    print(f"   [OK] {csv_a.name}")

    # Matriz de confusao + agreement
    print(f"\n[5/5] Cross-tab e visualizacoes...")
    matriz = np.zeros((2, 2), dtype=int)
    for pt, pk in zip(p_text, p_topo):
        matriz[int(pt), int(pk)] += 1
    agreement = (p_text == p_topo).mean()
    k = kappa(p_text, p_topo)
    metrics_path = OUT_DIR / "matriz_confusao.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Concordancia TEXTUAL vs TOPOLOGICO em Bluesky (sem ground truth)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"N amostrado: {len(sel)}\n")
        f.write(f"Agreement rate: {agreement:.4f} ({100*agreement:.1f}%)\n")
        f.write(f"Cohen's kappa: {k:.4f}\n\n")
        f.write("Matriz de confusao (rows=textual, cols=topologico):\n")
        f.write(f"                 topo=fake(0)  topo=real(1)\n")
        f.write(f"  text=fake(0)   {matriz[0,0]:>8}     {matriz[0,1]:>8}\n")
        f.write(f"  text=real(1)   {matriz[1,0]:>8}     {matriz[1,1]:>8}\n\n")
        f.write(f"Distribuicao marginal:\n")
        f.write(f"  textual:    fake={(p_text==0).sum()}  real={(p_text==1).sum()}\n")
        f.write(f"  topologico: fake={(p_topo==0).sum()}  real={(p_topo==1).sum()}\n")
    print(f"   [OK] {metrics_path.name}")
    print(f"   Agreement: {100*agreement:.1f}% | kappa: {k:.3f}")

    # Plot heatmap matriz
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["fake (0)","real (1)"])
    ax.set_yticklabels(["fake (0)","real (1)"])
    ax.set_xlabel("Predicao TOPOLOGICA")
    ax.set_ylabel("Predicao TEXTUAL")
    ax.set_title(f"Matriz textual vs topologico (n={len(sel)})\n"
                 f"agreement={100*agreement:.1f}% | kappa={k:.3f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i,j]), ha="center", va="center",
                    fontsize=14, color="white" if matriz[i,j] > matriz.max()/2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    pcm = OUT_DIR / "fig_matriz_confusao.png"
    plt.savefig(pcm, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {pcm.name}")

    # Cross-tab agreement por bin de num_nodes
    print(f"\n   Agreement por bin de num_nodes:")
    bin_names = [f"[{lo},{hi})" if hi < 1000 else f">={lo}" for lo, hi in BINS_NOS]
    nums_amostra = np.array([rows[i]["num_nodes"] for i in sel])
    bin_per_post = np.array([bin_idx(n, BINS_NOS) for n in nums_amostra])
    agree_por_bin = []
    n_por_bin = []
    for b in range(len(BINS_NOS)):
        mask = bin_per_post == b
        if mask.sum() == 0:
            agree_por_bin.append(np.nan)
            n_por_bin.append(0)
            continue
        agree = (p_text[mask] == p_topo[mask]).mean()
        agree_por_bin.append(agree)
        n_por_bin.append(int(mask.sum()))
        print(f"     {bin_names[b]:<14} n={mask.sum():>5} agreement={100*agree:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bin_names, [100*a for a in agree_por_bin], color="#5BAD72", alpha=0.85)
    for i, (a, n) in enumerate(zip(agree_por_bin, n_por_bin)):
        if not np.isnan(a):
            ax.text(i, 100*a + 1, f"{100*a:.1f}%\n(n={n})", ha="center", fontsize=9)
    ax.set_ylabel("Agreement rate (%)")
    ax.set_xlabel("num_nodes (1 + interacoes)")
    ax.set_title(f"Agreement Textual vs Topologico por tamanho do post (Bluesky)\n"
                 f"baseline = agreement geral {100*agreement:.1f}%")
    ax.axhline(100*agreement, ls="--", color="gray", alpha=0.6, label=f"agreement geral")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    pf = OUT_DIR / "fig_concordancia_por_tamanho.png"
    plt.savefig(pf, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {pf.name}")

    # Cross-tab agreement por feed
    print(f"\n   Agreement por feed:")
    feeds_unicos = sorted({rows[i]["feed"] for i in sel})
    agree_feed = []
    n_feed = []
    for feed in feeds_unicos:
        mask = np.array([rows[i]["feed"] == feed for i in sel])
        if mask.sum() == 0:
            agree_feed.append(np.nan); n_feed.append(0); continue
        agree = (p_text[mask] == p_topo[mask]).mean()
        agree_feed.append(agree); n_feed.append(int(mask.sum()))
        print(f"     {feed:<25} n={mask.sum():>5} agreement={100*agree:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(feeds_unicos, [100*a for a in agree_feed], color="#4F86C6", alpha=0.85)
    for i, (a, n) in enumerate(zip(agree_feed, n_feed)):
        if not np.isnan(a):
            ax.text(i, 100*a + 1, f"{100*a:.1f}%\n(n={n})", ha="center", fontsize=8)
    ax.axhline(100*agreement, ls="--", color="gray", alpha=0.6, label=f"agreement geral")
    ax.set_ylabel("Agreement rate (%)")
    ax.set_title("Agreement Textual vs Topologico por feed (Bluesky)")
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.set_ylim(0, 105); ax.legend()
    plt.tight_layout()
    pff = OUT_DIR / "fig_concordancia_por_feed.png"
    plt.savefig(pff, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {pff.name}")

    # Cross-tab agreement por bin de text_len
    print(f"\n   Agreement por bin de comprimento do texto:")
    BINS_LEN = [(0, 50), (50, 150), (150, 280), (280, 100000)]
    text_lens = np.array([rows[i]["text_len"] for i in sel])
    bin_per_post = np.array([bin_idx(n, BINS_LEN) for n in text_lens])
    agree_len = []
    n_len = []
    for b in range(len(BINS_LEN)):
        mask = bin_per_post == b
        if mask.sum() == 0:
            agree_len.append(np.nan); n_len.append(0); continue
        a = (p_text[mask] == p_topo[mask]).mean()
        agree_len.append(a); n_len.append(int(mask.sum()))
        nome = f"[{BINS_LEN[b][0]},{BINS_LEN[b][1]})" if BINS_LEN[b][1] < 1000 else f">={BINS_LEN[b][0]}"
        print(f"     {nome:<14} n={mask.sum():>5} agreement={100*a:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    bin_names_len = [f"[{lo},{hi})" if hi<1000 else f">={lo}" for lo, hi in BINS_LEN]
    ax.bar(bin_names_len, [100*a for a in agree_len], color="#E07B54", alpha=0.85)
    for i, (a, n) in enumerate(zip(agree_len, n_len)):
        if not np.isnan(a):
            ax.text(i, 100*a + 1, f"{100*a:.1f}%\n(n={n})", ha="center", fontsize=9)
    ax.axhline(100*agreement, ls="--", color="gray", alpha=0.6, label=f"agreement geral")
    ax.set_ylabel("Agreement rate (%)")
    ax.set_xlabel("Comprimento do texto (chars)")
    ax.set_title("Agreement Textual vs Topologico por comprimento de texto")
    ax.set_ylim(0, 105); ax.legend()
    plt.tight_layout()
    pfl = OUT_DIR / "fig_concordancia_por_text_len.png"
    plt.savefig(pfl, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {pfl.name}")

    # Salvar cross-tab consolidado
    ct_path = OUT_DIR / "cross_tab.csv"
    with open(ct_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimensao","valor","n","agreement_rate"])
        for nm, a, n in zip(bin_names, agree_por_bin, n_por_bin):
            w.writerow(["num_nodes_bin", nm, n, round(float(a),4) if not np.isnan(a) else ""])
        for nm, a, n in zip(feeds_unicos, agree_feed, n_feed):
            w.writerow(["feed", nm, n, round(float(a),4) if not np.isnan(a) else ""])
        for nm, a, n in zip(bin_names_len, agree_len, n_len):
            w.writerow(["text_len_bin", nm, n, round(float(a),4) if not np.isnan(a) else ""])
    print(f"   [OK] {ct_path.name}")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
