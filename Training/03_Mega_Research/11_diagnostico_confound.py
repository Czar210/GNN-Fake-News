"""
11_diagnostico_confound.py
--------------------------
Diagnostico do confound de tamanho do grafo (Fase 2C.1, Erro 3).

Quantifica quanto do sinal aprendido pelas GNNs pode ser explicado apenas
pelo numero de nos do grafo. Treina dois classificadores tabulares nos
mesmos folds da Fase 2A.1 / 2B.1:

  Variante A: X = [num_nodes]                  -- so o tamanho
  Variante B: X = [bert_raiz || num_nodes]     -- BERT + tamanho

Compara A com baseline textual (LogReg sobre BERT, Fase 2A.1) e Variante B
com Variante A para entender quanto N adiciona/explica.

Gate de Decisao:
  Se F1(Variante A) > 0.65 -> confound forte -> gerar dataset balanceado
  por subsampling pareado em bins de tamanho.
  Caso contrario, documentar e seguir.

Saidas em Execution/results/confound_diagnostico/:
  resultados.csv     -- f1 por fold para Variante A e B
  relatorio.txt      -- media +/- std + decisao do gate
  (condicional) data/fakenewsnet_balanceado.pt -- subsampling pareado

Uso:
  python gerar_folds.py             # uma vez
  python 11_diagnostico_confound.py
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_folds import iterar_folds, carregar_grafos

RAIZ     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR  = RAIZ / "Execution" / "results" / "fase2_baselines" / "confound_diagnostico"

RANDOM_SEED         = 42
GATE_F1_NUMNODES    = 0.65   # se Variante A passar disso, gera dataset balanceado
BINS_NOS = [(2, 10), (10, 20), (20, 50), (50, 101)]


def extrair(grafos: list) -> tuple:
    """Retorna (bert [N,768], num_nodes [N], y [N])."""
    bert      = torch.stack([g.x[0] for g in grafos]).numpy()
    num_nodes = np.array([g.num_nodes for g in grafos], dtype=np.float32)
    y         = np.array([g.y.item()  for g in grafos], dtype=np.int64)
    return bert, num_nodes, y


def fit_rf(X_tr, y_tr, X_te, y_te) -> dict:
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    return {
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1_fake":  float(f1_score(y_te, y_pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
    }


def gerar_dataset_balanceado(grafos: list, num_nodes: np.ndarray, y: np.ndarray) -> list:
    """
    Subsampling pareado em bins de tamanho. Para cada bin, amostra o mesmo
    numero de fake e real (minimo entre as duas classes do bin).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    selecionados = []
    for lo, hi in BINS_NOS:
        mask = (num_nodes >= lo) & (num_nodes < hi)
        idx_bin = np.where(mask)[0]
        if len(idx_bin) == 0:
            continue
        idx_fake = idx_bin[y[idx_bin] == 0]
        idx_real = idx_bin[y[idx_bin] == 1]
        n = min(len(idx_fake), len(idx_real))
        if n == 0:
            continue
        sel_fake = rng.choice(idx_fake, size=n, replace=False)
        sel_real = rng.choice(idx_real, size=n, replace=False)
        selecionados.extend(sel_fake.tolist())
        selecionados.extend(sel_real.tolist())
        print(f"    bin [{lo:3d}, {hi:3d}) | fake={len(idx_fake):3d} | real={len(idx_real):3d} | "
              f"amostrados={2*n}")

    selecionados.sort()
    return [grafos[i] for i in selecionados]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Diagnostico de Confound -- num_nodes vs BERT+num_nodes")
    print("=" * 62)

    print("\n[1/3] Carregando grafos e folds...")
    grafos = carregar_grafos()
    bert, num_nodes, y = extrair(grafos)
    print(f"  N = {len(grafos)} | num_nodes: min={num_nodes.min():.0f} "
          f"max={num_nodes.max():.0f} media={num_nodes.mean():.1f}")
    fakes = (y == 0).sum()
    print(f"  fake = {fakes} ({100*fakes/len(grafos):.1f}%)")

    folds = list(iterar_folds())

    print(f"\n[2/3] Treinando 2 variantes x {len(folds)} folds...")
    linhas_csv = []
    res_A, res_B = [], []

    for fold_idx, train_idx, test_idx in folds:
        # Variante A: so num_nodes
        XA_tr = num_nodes[train_idx].reshape(-1, 1)
        XA_te = num_nodes[test_idx].reshape(-1, 1)
        rA = fit_rf(XA_tr, y[train_idx], XA_te, y[test_idx])
        res_A.append(rA)
        linhas_csv.append({"variante": "A_num_nodes", "fold": fold_idx, **rA})

        # Variante B: BERT + num_nodes (concat)
        XB_tr = np.concatenate([bert[train_idx], num_nodes[train_idx].reshape(-1, 1)], axis=1)
        XB_te = np.concatenate([bert[test_idx],  num_nodes[test_idx].reshape(-1, 1)],  axis=1)
        rB = fit_rf(XB_tr, y[train_idx], XB_te, y[test_idx])
        res_B.append(rB)
        linhas_csv.append({"variante": "B_bert_nnodes", "fold": fold_idx, **rB})

        print(f"  fold {fold_idx:2d} | A: F1m={rA['f1_macro']:.4f} | "
              f"B: F1m={rB['f1_macro']:.4f}")

    # CSV
    csv_path = OUT_DIR / "resultados.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variante", "fold", "f1_macro", "f1_fake", "accuracy"])
        writer.writeheader()
        for row in linhas_csv:
            row = {**row, "f1_macro": round(row["f1_macro"], 4),
                   "f1_fake": round(row["f1_fake"], 4),
                   "accuracy": round(row["accuracy"], 4)}
            writer.writerow(row)

    # Resumo + Gate
    f1A = np.array([r["f1_macro"] for r in res_A])
    f1B = np.array([r["f1_macro"] for r in res_B])
    rel = [
        "=" * 62,
        "  Diagnostico de Confound -- Resumo",
        "=" * 62,
        f"  Variante A (so num_nodes): F1_macro = {f1A.mean():.4f} +/- {f1A.std():.4f}",
        f"  Variante B (BERT+num_nodes): F1_macro = {f1B.mean():.4f} +/- {f1B.std():.4f}",
        "",
        f"  Gate de Decisao: F1(A) > {GATE_F1_NUMNODES} ?",
    ]

    disparou_gate = bool(f1A.mean() > GATE_F1_NUMNODES)
    if disparou_gate:
        rel.append(f"  -> SIM ({f1A.mean():.4f} > {GATE_F1_NUMNODES}). Disparando subsampling pareado.")
    else:
        rel.append(f"  -> NAO ({f1A.mean():.4f} <= {GATE_F1_NUMNODES}). "
                   f"Confound presente mas fraco; nao gera dataset balanceado.")

    print()
    for l in rel:
        print(l)

    if disparou_gate:
        print("\n[3/3] Gerando dataset balanceado (subsampling pareado por bins de tamanho)...")
        bal = gerar_dataset_balanceado(grafos, num_nodes, y)
        out = DATA_DIR / "fakenewsnet_balanceado.pt"
        torch.save(bal, out)
        rel.append("")
        rel.append(f"  Dataset balanceado salvo em: {out}")
        rel.append(f"  Total balanceado: {len(bal)} grafos")
        print(f"  [OK] {out.name} -- {len(bal)} grafos")

        # Avaliacao rapida sobre dataset balanceado para checar se LogReg-BERT cai
        print("\n  Avaliando LogReg-BERT no dataset balanceado (k-fold interno)...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        bert_bal = np.stack([g.x[0].numpy() for g in bal])
        y_bal = np.array([g.y.item() for g in bal])
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        f1s = []
        for tr, te in skf.split(np.zeros(len(y_bal)), y_bal):
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            lr.fit(bert_bal[tr], y_bal[tr])
            f1s.append(f1_score(y_bal[te], lr.predict(bert_bal[te]), average="macro", zero_division=0))
        rel.append(f"  LogReg-BERT (balanceado): F1_macro = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  LogReg-BERT (balanceado): F1_macro = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    else:
        print("\n[3/3] Gate nao disparou -- dataset balanceado nao foi gerado.")

    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(rel) + "\n", encoding="utf-8")
    print(f"\n[OK] Relatorio salvo em {rel_path}")


if __name__ == "__main__":
    main()
