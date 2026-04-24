"""
21_bloco_vs_mini.py
-------------------
Compara duas estrategias de modelagem:

  (a) BLOCO GRANDE: 1 modelo unico treinado em FNN + UPFD-Polit + UPFD-Goss
      concatenados (com dataset_id como feature extra)

  (b) MODELOS MINI: 1 modelo especializado por dataset

Pergunta central: vale a pena ter um modelo grande generalista?
  - Se BLOCO ~ MINI: generalizacao gratis (1 modelo, mesma performance)
  - Se BLOCO < MINI: especializacao vence (preciso de 1 modelo por dominio)
  - Se BLOCO > MINI: bloco aprende padrao geral (transferencia ajuda)

Foco: features TOPOLOGICAS [num_nodes, grau_root] -- universais entre datasets.
(Texto teria dim diferente entre BERT-768 do FNN vs content-310 do GossipCop.)

Modelo: RandomForest (rapido, comparavel ao usado no 17_persistir).
Validacao: split oficial de cada dataset; bloco usa concatenacao com dataset_id.

Saida em Execution/results/figuras_tcc/bloco_vs_mini/:
  resumo.csv          -- (dataset, modelo[mini|bloco], f1m, acc, f1fake)
  fig_comparacao.png  -- barplot lado a lado
  relatorio.txt       -- decisao + interpretacao

Uso:
  python 21_bloco_vs_mini.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import UPFD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_folds import carregar_grafos as carregar_fnn

RAIZ        = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = Path(__file__).resolve().parent / "data"
MAT_DIR     = RAIZ / "Material"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "bloco_vs_mini"

RANDOM_SEED = 42


def topo_feats(grafos):
    X = []
    for g in grafos:
        n = g.num_nodes
        gr = (g.edge_index[0] == 0).sum().item() if g.edge_index.numel() > 0 else 0
        X.append([n, gr])
    return np.array(X, dtype=np.float64)


def labels(grafos):
    return np.array([g.y.item() for g in grafos], dtype=np.int64)


def split_60_20_20(grafos, seed=RANDOM_SEED):
    """Para FNN, que veio em raw nao-stratificado. Retorna (train_idx, test_idx)."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(grafos))); rng.shuffle(idx)
    n_te = int(len(idx) * 0.20)
    test_idx = idx[:n_te]
    train_idx = idx[n_te:]
    return train_idx, test_idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Bloco grande vs Modelos mini (features topologicas)")
    print("=" * 72)

    # ── Carregar 3 datasets ──────────────────────────────────────────────
    print("\n[1/4] Carregando 3 datasets...")
    fnn_all = carregar_fnn()
    polit_train = list(UPFD(root=str(MAT_DIR), name="politifact", feature="profile", split="train"))
    polit_val   = list(UPFD(root=str(MAT_DIR), name="politifact", feature="profile", split="val"))
    polit_test  = list(UPFD(root=str(MAT_DIR), name="politifact", feature="profile", split="test"))
    goss_train  = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="train"))
    goss_val    = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="val"))
    goss_test   = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="test"))

    # FNN: split sintetico 80/20 (sem fold pareado para simplificar comparacao com UPFDs)
    fnn_tr_idx, fnn_te_idx = split_60_20_20(fnn_all, seed=RANDOM_SEED)
    fnn_train = [fnn_all[i] for i in fnn_tr_idx]
    fnn_test  = [fnn_all[i] for i in fnn_te_idx]
    print(f"   FNN          train={len(fnn_train)}  test={len(fnn_test)}")
    print(f"   UPFD-Polit   train+val={len(polit_train)+len(polit_val)}  test={len(polit_test)}")
    print(f"   UPFD-Goss    train+val={len(goss_train)+len(goss_val)}  test={len(goss_test)}")

    datasets = {
        "fnn":             {"tr": fnn_train,            "te": fnn_test,    "id": 0},
        "upfd_politifact": {"tr": polit_train+polit_val,"te": polit_test,  "id": 1},
        "upfd_gossipcop":  {"tr": goss_train+goss_val,  "te": goss_test,   "id": 2},
    }

    # ── (a) Modelos mini ─────────────────────────────────────────────────
    print("\n[2/4] Treinando modelos MINI (1 por dataset)...")
    resultados = []
    for nome, ds in datasets.items():
        Xtr = topo_feats(ds["tr"])
        ytr = labels(ds["tr"])
        Xte = topo_feats(ds["te"])
        yte = labels(ds["te"])
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)
        f1m = f1_score(yte, pred, average="macro", zero_division=0)
        f1f = f1_score(yte, pred, pos_label=0, zero_division=0)
        acc = accuracy_score(yte, pred)
        resultados.append({"dataset": nome, "modelo": "mini",
                           "f1_macro": round(f1m,4), "f1_fake": round(f1f,4),
                           "accuracy": round(acc,4),
                           "n_treino": len(Xtr), "n_teste": len(Xte)})
        print(f"   {nome:<22} F1m={f1m:.4f} (n_treino={len(Xtr)}, n_teste={len(Xte)})")

    # ── (b) Bloco grande ──────────────────────────────────────────────────
    print("\n[3/4] Treinando BLOCO GRANDE (3 datasets concatenados, +dataset_id)...")
    Xtr_big = []; ytr_big = []
    Xte_big_por_ds = {}
    for nome, ds in datasets.items():
        # treino: concat com dataset_id
        Xtr = topo_feats(ds["tr"])
        ytr = labels(ds["tr"])
        ids = np.full((len(Xtr), 1), ds["id"], dtype=np.float64)
        Xtr_big.append(np.concatenate([Xtr, ids], axis=1))
        ytr_big.append(ytr)
        # teste: separado por dataset
        Xte = topo_feats(ds["te"])
        yte = labels(ds["te"])
        ids_te = np.full((len(Xte), 1), ds["id"], dtype=np.float64)
        Xte_big_por_ds[nome] = (np.concatenate([Xte, ids_te], axis=1), yte)
    Xtr_big = np.concatenate(Xtr_big, axis=0)
    ytr_big = np.concatenate(ytr_big, axis=0)
    print(f"   Bloco treino total: n={len(Xtr_big)}, dim={Xtr_big.shape[1]}")

    rf_big = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf_big.fit(Xtr_big, ytr_big)
    print(f"   Treinado: {rf_big.n_estimators} arvores em {len(Xtr_big)} amostras")

    print(f"\n   Avaliando bloco grande no test set de cada dataset:")
    for nome, (Xte, yte) in Xte_big_por_ds.items():
        pred = rf_big.predict(Xte)
        f1m = f1_score(yte, pred, average="macro", zero_division=0)
        f1f = f1_score(yte, pred, pos_label=0, zero_division=0)
        acc = accuracy_score(yte, pred)
        resultados.append({"dataset": nome, "modelo": "bloco",
                           "f1_macro": round(f1m,4), "f1_fake": round(f1f,4),
                           "accuracy": round(acc,4),
                           "n_treino": len(Xtr_big), "n_teste": len(Xte)})
        print(f"   {nome:<22} F1m={f1m:.4f}")

    # ── 4. Salvar e plotar ───────────────────────────────────────────────
    csv_path = OUT_DIR / "resumo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","modelo","f1_macro","f1_fake",
                                          "accuracy","n_treino","n_teste"])
        w.writeheader(); w.writerows(resultados)
    print(f"\n[4/4] Saidas:")
    print(f"   [OK] {csv_path.name}")

    # Comparacao em tabela
    print(f"\n   --- Resumo (F1 macro) ---")
    print(f"   {'dataset':<22} {'mini':>10} {'bloco':>10} {'delta':>10}")
    print(f"   {'-'*52}")
    deltas = {}
    for nome in datasets:
        f1_mini  = next(r["f1_macro"] for r in resultados if r["dataset"]==nome and r["modelo"]=="mini")
        f1_bloco = next(r["f1_macro"] for r in resultados if r["dataset"]==nome and r["modelo"]=="bloco")
        d = f1_bloco - f1_mini
        deltas[nome] = d
        print(f"   {nome:<22} {f1_mini:>10.4f} {f1_bloco:>10.4f} {d:>+10.4f}")

    # Plot barplot lado a lado
    nomes = list(datasets.keys())
    f1_mini_list  = [next(r["f1_macro"] for r in resultados if r["dataset"]==n and r["modelo"]=="mini")  for n in nomes]
    f1_bloco_list = [next(r["f1_macro"] for r in resultados if r["dataset"]==n and r["modelo"]=="bloco") for n in nomes]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(nomes))
    w_bar = 0.35
    ax.bar(x - w_bar/2, f1_mini_list,  w_bar, label="MINI (1 por dataset)", color="#4F86C6", alpha=0.85)
    ax.bar(x + w_bar/2, f1_bloco_list, w_bar, label="BLOCO grande (3 concat + dataset_id)", color="#E07B54", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(nomes)
    ax.set_ylabel("F1 macro")
    ax.set_title("Bloco grande vs Modelos mini -- features topologicas [num_nodes, grau_root]")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (m, b) in enumerate(zip(f1_mini_list, f1_bloco_list)):
        ax.text(i - w_bar/2, m+0.01, f"{m:.3f}", ha="center", fontsize=9)
        ax.text(i + w_bar/2, b+0.01, f"{b:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig_comparacao.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [OK] {fig_path.name}")

    # Relatorio interpretativo
    rel = ["=" * 60, "  Bloco grande vs Modelos mini -- Decisao", "=" * 60, ""]
    media_delta = np.mean(list(deltas.values()))
    pior_delta  = min(deltas.values())
    rel.append(f"Delta medio (bloco - mini): {media_delta:+.4f}")
    rel.append(f"Maior queda: {pior_delta:+.4f}")
    rel.append("")
    if abs(media_delta) < 0.02 and pior_delta > -0.05:
        rel.append("DECISAO: BLOCO ~ MINI -- generalizacao funciona, vale 1 modelo so.")
    elif media_delta > 0.02:
        rel.append("DECISAO: BLOCO > MINI -- transferencia entre dominios ajuda.")
    else:
        rel.append("DECISAO: MINI > BLOCO -- especializacao vence; mantenha 1 por dataset.")
    rel.append("")
    rel.append("Detalhamento por dataset:")
    for nome, d in deltas.items():
        rel.append(f"  {nome:<22} delta={d:+.4f}")

    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(rel) + "\n", encoding="utf-8")
    print(f"   [OK] {rel_path.name}")

    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
