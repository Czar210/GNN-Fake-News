"""
10_baseline_textual.py
----------------------
Baseline textual oficial (Fase 2A.1, Erro 2 / Inc. 12).

Mede o "teto textual": qual F1 um classificador classico (Regressao Logistica,
Random Forest) atinge usando APENAS o embedding BERT do titulo do artigo
(`grafo.x[0]`). Esse numero e o baseline a ser superado pelas GNNs para que
a hipotese central do TCC se sustente -- se a GNN nao supera a regressao
sobre o texto, entao a topologia nao adiciona sinal discriminativo.

Roda nos MESMOS 10 folds de data/folds_fnn.pt (Fase 2B.1) para que
ttest_rel(GNN, baseline) seja pareado por fold na Fase 4.2.

Saidas em Execution/results/baseline_textual/:
  resultados.csv  -- 20 linhas (2 modelos x 10 folds): modelo, fold, f1_macro, f1_fake, accuracy
  relatorio.txt   -- resumo media +/- desvio por modelo

Uso:
  python gerar_folds.py            # uma vez, gera folds_fnn.pt
  python 10_baseline_textual.py
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_folds import iterar_folds, carregar_grafos

RAIZ    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = RAIZ / "Execution" / "results" / "fase2_baselines" / "baseline_textual"

RANDOM_SEED = 42


def extrair_features_e_labels(grafos: list) -> tuple:
    """
    Para cada grafo, X[i] = embedding BERT da raiz (grafo.x[0], 768-dim),
    y[i] = label (0=Fake, 1=Real).
    """
    X = torch.stack([g.x[0] for g in grafos]).numpy()
    y = np.array([g.y.item() for g in grafos], dtype=np.int64)
    return X, y


def treinar_e_avaliar(modelo, X_tr, y_tr, X_te, y_te) -> dict:
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    return {
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1_fake":  float(f1_score(y_te, y_pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Baseline Textual -- LogReg + RandomForest sobre BERT(titulo)")
    print("=" * 62)

    print("\n[1/3] Carregando grafos e folds...")
    grafos = carregar_grafos()
    X, y   = extrair_features_e_labels(grafos)
    print(f"  X.shape = {X.shape}  |  y.shape = {y.shape}  |  fake = {(y==0).sum()}")

    folds = list(iterar_folds())
    print(f"  {len(folds)} folds compartilhados (folds_fnn.pt)")

    print(f"\n[2/3] Treinando 2 modelos x {len(folds)} folds = "
          f"{2*len(folds)} fits...")

    csv_path = OUT_DIR / "resultados.csv"
    linhas_csv = []
    resumo = {"LogReg": [], "RandomForest": []}

    for fold_idx, train_idx, test_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        modelos = {
            "LogReg":       LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            "RandomForest": RandomForestClassifier(n_estimators=200,
                                                   random_state=RANDOM_SEED, n_jobs=-1),
        }

        for nome, modelo in modelos.items():
            t0  = time.time()
            res = treinar_e_avaliar(modelo, X_tr, y_tr, X_te, y_te)
            dt  = time.time() - t0
            resumo[nome].append(res)
            linhas_csv.append({
                "modelo":    nome,
                "fold":      fold_idx,
                "f1_macro":  round(res["f1_macro"], 4),
                "f1_fake":   round(res["f1_fake"],  4),
                "accuracy":  round(res["accuracy"], 4),
            })
            print(f"  fold {fold_idx:2d} | {nome:<13} | F1_macro={res['f1_macro']:.4f} | "
                  f"F1_fake={res['f1_fake']:.4f} | Acc={res['accuracy']:.4f} | "
                  f"{dt:.1f}s")

    print(f"\n[3/3] Salvando resultados em {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["modelo", "fold", "f1_macro", "f1_fake", "accuracy"])
        writer.writeheader()
        writer.writerows(linhas_csv)

    # Resumo agregado
    relatorio = ["=" * 62, "  Baseline Textual -- Resumo", "=" * 62]
    for nome, lista in resumo.items():
        f1m = [r["f1_macro"] for r in lista]
        f1f = [r["f1_fake"]  for r in lista]
        acc = [r["accuracy"] for r in lista]
        relatorio.append(
            f"  {nome:<13} | F1_macro = {np.mean(f1m):.4f} +/- {np.std(f1m):.4f} | "
            f"F1_fake = {np.mean(f1f):.4f} +/- {np.std(f1f):.4f} | "
            f"Acc = {np.mean(acc):.4f} +/- {np.std(acc):.4f}"
        )

    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(relatorio) + "\n", encoding="utf-8")

    print()
    for l in relatorio:
        print(l)
    print()
    print(f"[RESUMO] LogReg: F1={np.mean([r['f1_macro'] for r in resumo['LogReg']]):.3f} "
          f"+/- {np.std([r['f1_macro'] for r in resumo['LogReg']]):.3f} | "
          f"RandomForest: F1={np.mean([r['f1_macro'] for r in resumo['RandomForest']]):.3f} "
          f"+/- {np.std([r['f1_macro'] for r in resumo['RandomForest']]):.3f}")


if __name__ == "__main__":
    main()
