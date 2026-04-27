"""
26_rq3_multilingual.py
----------------------
Experimento RQ3 (parte b): "topologia eh independente do idioma do post".

Hipotese: se o classificador estrutural (RF GossipCop, features
[num_nodes, grau_root]) for *puramente topologico*, a distribuicao
de scores em posts PT, DE, EN do Bluesky deve ser estatisticamente
indistinguivel.

Fonte: dados_bluesky/feed_posts/*.jsonl, campo `langs` nativo do AT Protocol.

Filtro: `langs == ('eng',)`, `('deu',)`, `('por',)` exclusivamente
(remove combos multilingues p/ nao confundir).

Para cada idioma:
  - extrai (num_nodes, grau_root) por post
  - aplica RF GossipCop -> score 'fake-like'
  - calcula media, mediana, std
  - histograma normalizado

Comparacao par-a-par (PT vs EN, DE vs EN):
  - Kolmogorov-Smirnov 2-sample test (D, p-value)
  - Cohen's d (effect size)
  - Mann-Whitney U (robusto a nao-normalidade)

Saida em Execution/results/figuras_tcc/rq3_multilingual/:
  scores_por_lang.csv      -- (lang, post_id, num_nodes, score, feed)
  comparacoes.csv          -- (par, KS_D, KS_p, cohens_d, MW_U, MW_p)
  fig_distribuicao.png     -- 3 histogramas sobrepostos
  fig_qq.png               -- Q-Q plots PT vs EN, DE vs EN
  relatorio.txt            -- interpretacao

Uso:
  python 26_rq3_multilingual.py
"""

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ        = Path(__file__).resolve().parent.parent.parent
BSKY_DIR    = RAIZ / "dados_bluesky"
FEED_DIR    = BSKY_DIR / "feed_posts"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "rq3_multilingual"

# Idiomas alvo: codigo ISO 639-3 conforme campo `langs` do Bluesky
LANGS_ALVO = {
    "eng": "Ingles",
    "deu": "Alemao",
    "por": "Portugues",
}

CORES = {"eng": "#3498db", "deu": "#e74c3c", "por": "#2ecc71"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size Cohen's d entre duas amostras (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12: return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def carregar_rf():
    with open(WEIGHTS_DIR / "rf_struct_gossipcop.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"]


def extrair_features(post: dict) -> tuple:
    """num_nodes, grau_root = 1 + interacoes, interacoes."""
    interacoes = (int(post.get("reply_count")  or 0)
                  + int(post.get("repost_count") or 0))
    if isinstance(post.get("quotes"), int):
        interacoes += post["quotes"]
    return 1 + interacoes, interacoes


def carregar_posts_por_lang() -> dict:
    """Retorna dict lang -> list of (post_id, num_nodes, grau_root, feed)."""
    por_lang = {k: [] for k in LANGS_ALVO}
    contagem = Counter()
    for arq in sorted(FEED_DIR.glob("*.jsonl")):
        feed = arq.stem
        with open(arq, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                langs = p.get("langs") or []
                if len(langs) != 1: continue   # apenas monolingue
                lg = langs[0]
                if lg not in LANGS_ALVO: continue
                nn, gr = extrair_features(p)
                por_lang[lg].append((p.get("post_id"), nn, gr, feed))
                contagem[lg] += 1
    print(f"  Contagens monolingues:")
    for k, n in contagem.items():
        print(f"    {k} ({LANGS_ALVO[k]}): {n:,} posts")
    return por_lang


def calcular_scores(rf, posts: list) -> np.ndarray:
    """Aplica RF -> retorna prob de classe 0 (fake)."""
    if not posts: return np.array([])
    X = np.array([[p[1], p[2]] for p in posts], dtype=np.float64)
    classes = list(rf.classes_)
    idx_fake = classes.index(0) if 0 in classes else 0
    return rf.predict_proba(X)[:, idx_fake]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  RQ3 (b) -- Topologia eh independente do idioma do post?")
    print("=" * 72)

    print("\n[1/5] Carregando RF estrutural (treinado em GossipCop)...")
    rf = carregar_rf()
    print(f"  RF: {rf.n_estimators} arvores | classes: {list(rf.classes_)}")

    print("\n[2/5] Lendo Bluesky e separando por idioma (monolingue puro)...")
    por_lang = carregar_posts_por_lang()

    print("\n[3/5] Aplicando RF e calculando scores...")
    scores_por_lang = {}
    for lg, posts in por_lang.items():
        s = calcular_scores(rf, posts)
        scores_por_lang[lg] = s
        if len(s):
            print(f"    {lg}: n={len(s):,} | media={s.mean():.4f} "
                  f"| mediana={np.median(s):.4f} | std={s.std():.4f}")

    # CSV completo
    csv_path = OUT_DIR / "scores_por_lang.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "post_id", "num_nodes", "grau_root", "feed", "score_fake"])
        for lg, posts in por_lang.items():
            for (pid, nn, gr, feed), s in zip(posts, scores_por_lang[lg]):
                w.writerow([lg, pid, nn, gr, feed, round(float(s), 4)])

    print("\n[4/5] Comparacoes estatisticas par-a-par...")
    pares = [("por", "eng"), ("deu", "eng"), ("por", "deu")]
    comp_rows = []
    rel = ["=" * 72,
           "  RQ3 (b) -- Topologia independente do idioma",
           "=" * 72,
           "",
           "Hipotese H0: distribuicao de score 'fake-like' eh igual entre idiomas.",
           "Se H0 nao eh rejeitada, classificador estrutural eh insensivel ao idioma.",
           "",
           "Notas:",
           "- Comparamos a distribuicao de scores; um efeito pequeno (Cohen's |d| < 0.2)",
           "  indica equivalencia pratica mesmo se p-value for significativo (n grande).",
           "- KS testa formato da distribuicao; Mann-Whitney testa diferenca de medianas.",
           "",
           f"  {'par':<15} {'n_a':>8} {'n_b':>8} {'mean_a':>9} {'mean_b':>9} "
           f"{'cohen_d':>8} {'KS_D':>8} {'KS_p':>10} {'MW_p':>10}",
           f"  {'-'*100}"]
    for a, b in pares:
        sa, sb = scores_por_lang[a], scores_por_lang[b]
        if len(sa) < 30 or len(sb) < 30:
            print(f"    {a} vs {b}: n insuficiente, pulando")
            continue
        d = cohens_d(sa, sb)
        ks_D, ks_p = stats.ks_2samp(sa, sb)
        try:
            mw_U, mw_p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        except Exception:
            mw_U, mw_p = float("nan"), float("nan")
        comp_rows.append({"par": f"{a}_vs_{b}", "n_a": len(sa), "n_b": len(sb),
                          "mean_a": float(np.mean(sa)), "mean_b": float(np.mean(sb)),
                          "cohens_d": d,
                          "KS_D": float(ks_D), "KS_p": float(ks_p),
                          "MW_U": float(mw_U), "MW_p": float(mw_p)})
        rel.append(f"  {a+'_vs_'+b:<15} {len(sa):>8} {len(sb):>8} "
                   f"{np.mean(sa):>9.4f} {np.mean(sb):>9.4f} "
                   f"{d:>8.3f} {float(ks_D):>8.3f} {float(ks_p):>10.2e} {float(mw_p):>10.2e}")
        print(f"    {a} vs {b}: d={d:.3f} | KS p={ks_p:.2e} | MW p={mw_p:.2e}")

    rel.append("")
    rel.append("Interpretacao (Cohen 1988):")
    rel.append("  |d| < 0.2  -> efeito desprezivel (idiomas equivalentes p/ topologia)")
    rel.append("  0.2 <= |d| < 0.5  -> pequeno")
    rel.append("  0.5 <= |d| < 0.8  -> medio")
    rel.append("  |d| >= 0.8  -> grande")
    rel.append("")

    # CSV comparacoes
    comp_path = OUT_DIR / "comparacoes.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["par","n_a","n_b","mean_a","mean_b",
                                           "cohens_d","KS_D","KS_p","MW_U","MW_p"])
        w.writeheader()
        for r in comp_rows:
            r2 = {**r,
                  "mean_a": round(r["mean_a"], 4), "mean_b": round(r["mean_b"], 4),
                  "cohens_d": round(r["cohens_d"], 4),
                  "KS_D": round(r["KS_D"], 4), "KS_p": float(f"{r['KS_p']:.4e}"),
                  "MW_U": round(r["MW_U"], 1), "MW_p": float(f"{r['MW_p']:.4e}")}
            w.writerow(r2)

    print("\n[5/5] Gerando figuras...")

    # Histograma sobreposto
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 41)
    def _n(x): return f"{int(x):,}".replace(",", ".")
    for lg in ["eng", "deu", "por"]:
        s = scores_por_lang[lg]
        if len(s) == 0: continue
        ax.hist(s, bins=bins, density=True, alpha=0.5,
                label=f"{LANGS_ALVO[lg]} (n={_n(len(s))}, mu={s.mean():.3f})",
                color=CORES[lg], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Score 'fake-like' (P(fake) do RF estrutural)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuicao de score topologico por idioma do post (Bluesky)\n"
                 "Se topologia eh independente de idioma, as curvas devem coincidir")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig_distribuicao.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [OK] {fig_path.name}")

    # Q-Q plot PT vs EN, DE vs EN
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (a, b) in zip(axes, [("por", "eng"), ("deu", "eng")]):
        sa, sb = scores_por_lang[a], scores_por_lang[b]
        if len(sa) < 30 or len(sb) < 30:
            ax.text(0.5, 0.5, "n insuficiente", ha="center", transform=ax.transAxes)
            continue
        # Q-Q manual: quantis equivalentes em [0.01, 0.99]
        qs = np.linspace(0.01, 0.99, 50)
        qa = np.quantile(sa, qs)
        qb = np.quantile(sb, qs)
        ax.scatter(qb, qa, s=20, color=CORES[a], edgecolor="black", linewidth=0.5)
        lim = [0, 1]
        ax.plot(lim, lim, "k--", alpha=0.4, label="y=x (idiomas equivalentes)")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"Quantis {LANGS_ALVO[b]}")
        ax.set_ylabel(f"Quantis {LANGS_ALVO[a]}")
        ax.set_title(f"Q-Q: {LANGS_ALVO[a]} vs {LANGS_ALVO[b]}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    qq_path = OUT_DIR / "fig_qq.png"
    plt.savefig(qq_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [OK] {qq_path.name}")

    # Relatorio
    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(rel) + "\n", encoding="utf-8")
    print(f"  [OK] {rel_path.name}")
    print(f"\n[OK] Saidas em {OUT_DIR}")


if __name__ == "__main__":
    main()
