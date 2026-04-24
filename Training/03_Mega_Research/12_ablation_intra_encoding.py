"""
12_ablation_intra_encoding.py
-----------------------------
Ablation intra-encoding (Fase 3.1.bis).

Treina GCN em 3 variantes de features posicionais, sobre os mesmos folds:
  - pos-full:  is_root + grau + pos
  - pos-min:   is_root + pos       (grau zerado)
  - pos-grau:  is_root + grau      (pos zerada -- filhos identicos por design)

Compara contra o baseline textual (LogReg-BERT da Fase 2A.1) com ttest_rel
pareado por fold.

Aplica dois gates:
  1) Gate de Posicao:  F1(pos-min) > F1(baseline) com p<0.05
       -> ganho atribuivel a posicao ordinal
  2) Gate de Confound: F1(pos-grau) > F1(pos-min) + 0.01
       -> sinal dominante e grau (popularidade); confirma Erro 3 -> Fase 5B.2

Saidas em Execution/results/ablation_intra_encoding/:
  tabela.csv        -- f1 por (variante, fold)
  relatorio.txt     -- agregado + decisao dos gates

Uso:
  python 12_ablation_intra_encoding.py
  python 12_ablation_intra_encoding.py --epochs 30
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model   import GCNClassifier
from gerar_folds import iterar_folds

RAIZ      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = Path(__file__).resolve().parent / "data"
OUT_DIR   = RAIZ / "Execution" / "results" / "fase3_ablation" / "ablation_intra_encoding"
BASELINE_CSV = RAIZ / "Execution" / "results" / "fase2_baselines" / "baseline_textual" / "resultados.csv"

VARIANTES   = ["posfull", "posmin", "posgrau"]
BATCH_SIZE  = 32
PATIENCE    = 7

GATE_POSICAO_P_MAX  = 0.05
GATE_CONFOUND_DELTA = 0.01


def carregar_variante(suffix: str) -> list:
    splits = []
    for split in ("train", "val", "test"):
        p = DATA_DIR / f"fakenewsnet_{suffix}" / f"fakenewsnet_{split}.pt"
        if not p.exists():
            print(f"[ERRO] {p} nao encontrado.")
            print(f"       Execute: python 00_construir_grafos_fakenewsnet.py "
                  f"--feature-variant {'full' if suffix=='posfull' else 'pos-'+suffix[3:]} "
                  f"--output-suffix {suffix}")
            sys.exit(1)
        splits += torch.load(p, weights_only=False)
    return splits


def split_train_val(train_idx: list, grafos: list, val_frac: float = 0.10,
                    seed: int = 0) -> tuple:
    import random
    rng = random.Random(seed)
    fakes = [i for i in train_idx if grafos[i].y.item() == 0]
    reals = [i for i in train_idx if grafos[i].y.item() == 1]
    rng.shuffle(fakes); rng.shuffle(reals)
    n_val_fk = max(1, int(len(fakes) * val_frac))
    n_val_re = max(1, int(len(reals) * val_frac))
    val_idx   = fakes[:n_val_fk] + reals[:n_val_re]
    train_sub = fakes[n_val_fk:] + reals[n_val_re:]
    return [grafos[i] for i in train_sub], [grafos[i] for i in val_idx]


def avaliar_f1_macro(model, dataset: list, device) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch)
            preds  = out.argmax(dim=1).cpu().tolist()
            labels = data.y.squeeze().cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels if isinstance(labels, list) else [labels])
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def treinar_uma_vez(model, train_data, val_data, device, epochs, lr) -> torch.nn.Module:
    tr_loader  = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    best_val, best_state, patience_count = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for data in tr_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
        val_f1 = avaliar_f1_macro(model, val_data, device)
        scheduler.step(val_f1)
        if val_f1 > best_val:
            best_val, best_state, patience_count = val_f1, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def carregar_baseline_por_fold() -> dict:
    """Le baseline_textual/resultados.csv, retorna {fold_idx: f1_macro_LogReg}."""
    if not BASELINE_CSV.exists():
        print(f"[ERRO] {BASELINE_CSV} nao encontrado. Rode 10_baseline_textual.py.")
        sys.exit(1)
    out = {}
    with open(BASELINE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["modelo"] == "LogReg":
                out[int(row["fold"])] = float(row["f1_macro"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Ablation intra-encoding (Fase 3.1.bis).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr",     type=float, default=0.001)
    parser.add_argument("--cpu",    action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  Ablation Intra-Encoding -- GCN sobre posfull / posmin / posgrau")
    print("=" * 66)
    print(f"  Device: {device} | Epochs max: {args.epochs}")
    print()

    # Baseline por fold
    baseline_f1_por_fold = carregar_baseline_por_fold()
    print(f"  Baseline LogReg-BERT (Fase 2A.1): {len(baseline_f1_por_fold)} folds carregados")

    # Para cada variante, dataset compartilha indices via folds_fnn.pt
    folds = list(iterar_folds())
    print(f"  Folds compartilhados: {len(folds)}")
    print()

    # resultados[variante][fold_idx] = f1_macro
    resultados: dict = {v: {} for v in VARIANTES}
    linhas_csv = []

    for variante in VARIANTES:
        print(f"-- Variante: {variante} " + "-" * (60 - len(variante)))
        grafos = carregar_variante(variante)
        num_feats = grafos[0].x.shape[1]
        print(f"  num_features = {num_feats} | n_grafos = {len(grafos)}")

        for fold_idx, train_idx, test_idx in folds:
            torch.manual_seed(fold_idx); np.random.seed(fold_idx)
            train_sub, val_sub = split_train_val(train_idx, grafos, seed=fold_idx)
            test_sub = [grafos[i] for i in test_idx]

            t0 = time.time()
            model = GCNClassifier(num_feats, 2, seed=fold_idx).to(device)
            model = treinar_uma_vez(model, train_sub, val_sub, device,
                                    args.epochs, args.lr)
            f1m = avaliar_f1_macro(model, test_sub, device)
            dt = time.time() - t0

            resultados[variante][fold_idx] = f1m
            linhas_csv.append({"variante": variante, "fold": fold_idx,
                                "f1_macro": round(f1m, 4)})
            print(f"  fold {fold_idx:2d} | F1m={f1m:.4f} | {dt:.1f}s")
        print()

    # CSV
    csv_path = OUT_DIR / "tabela.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variante", "fold", "f1_macro"])
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"  [OK] {csv_path}")

    # Tabela pareada com t-test contra baseline
    folds_ord = sorted(baseline_f1_por_fold.keys())
    base = np.array([baseline_f1_por_fold[i] for i in folds_ord])
    rel = [
        "=" * 70,
        "  Ablation Intra-Encoding -- Resumo",
        "=" * 70,
        f"  Baseline LogReg-BERT: F1m = {base.mean():.4f} +/- {base.std():.4f}",
        "",
        f"  {'variante':<10} {'F1m_medio':>10} {'F1m_std':>10} {'p_vs_baseline':>15}",
        f"  {'-'*55}",
    ]

    pvals = {}
    medias = {}
    for v in VARIANTES:
        vals = np.array([resultados[v][i] for i in folds_ord])
        t, p = stats.ttest_rel(vals, base)
        pvals[v] = float(p)
        medias[v] = float(vals.mean())
        rel.append(
            f"  {v:<10} {vals.mean():>10.4f} {vals.std():>10.4f} {p:>15.6f}"
        )

    rel += ["", "=" * 70, "  Gates de Decisao", "=" * 70]

    # Gate de Posicao: pos-min supera baseline com p<0.05?
    gate_pos = (pvals["posmin"] < GATE_POSICAO_P_MAX) and (medias["posmin"] > base.mean())
    rel += [
        "",
        f"  [Gate Posicao] F1(pos-min) > baseline com p<{GATE_POSICAO_P_MAX}?",
        f"     F1m(pos-min) = {medias['posmin']:.4f} | baseline = {base.mean():.4f} | "
        f"p = {pvals['posmin']:.6f}",
        f"     -> {'SIM' if gate_pos else 'NAO'} -- "
        f"{'ganho atribuivel a posicao ordinal' if gate_pos else 'posicao por si so nao supera baseline'}",
    ]

    # Gate de Confound
    delta = medias["posgrau"] - medias["posmin"]
    gate_conf = delta > GATE_CONFOUND_DELTA
    rel += [
        "",
        f"  [Gate Confound] F1(pos-grau) > F1(pos-min) + {GATE_CONFOUND_DELTA}?",
        f"     F1m(pos-grau) = {medias['posgrau']:.4f} | F1m(pos-min) = {medias['posmin']:.4f} "
        f"| delta = {delta:+.4f}",
        f"     -> {'SIM' if gate_conf else 'NAO'} -- "
        f"{'sinal dominante e GRAU (popularidade), nao posicao -> documentar como confirmacao do Erro 3 (Fase 5B.2)' if gate_conf else 'sinal nao dominado por grau'}",
    ]
    rel.append("")
    rel.append("=" * 70)

    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(rel) + "\n", encoding="utf-8")

    print()
    for l in rel:
        print(l)


if __name__ == "__main__":
    main()
