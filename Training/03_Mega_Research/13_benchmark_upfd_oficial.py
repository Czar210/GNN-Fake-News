"""
13_benchmark_upfd_oficial.py
----------------------------
Benchmark cross-dataset (extensao da Fase 4.1) -- testa se topologia ajuda
em datasets onde features de no sao GENUINAMENTE distintas (UPFD oficial).

Diferente do nosso FakeNewsNet construido (Erro 1), o UPFD oficial usa:
  - raiz = notícia (BERT do titulo)
  - filhos = usuarios (BERT dos tweets/perfil HISTORICOS de cada usuario)
Cada no tem features proprias -- GCN nao colapsa por degeneracao.

Compara para cada (dataset, arquitetura):
  - Baseline textual: LogReg sobre x[0] (feature da raiz = noticia)
  - GCN, GAT, GraphSAGE sobre o grafo inteiro (multiplas seeds)

Usa o split oficial train/val/test do UPFD.

Saida em Execution/results/benchmark_upfd_oficial/:
  resultados.csv   -- por (dataset, modelo, seed): f1_macro, f1_fake, accuracy
  relatorio.txt    -- comparacao agregada (media+/-std) GCN/GAT/SAGE vs baseline

Uso:
  python 13_benchmark_upfd_oficial.py
  python 13_benchmark_upfd_oficial.py --epochs 30 --seeds 10
  python 13_benchmark_upfd_oficial.py --datasets politifact   # so um
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model  import GCNClassifier
from gat_model  import GATClassifier
from sage_model import SAGEClassifier

ARCHS = {
    "GCN":  GCNClassifier,
    "GAT":  GATClassifier,
    "SAGE": SAGEClassifier,
}

RAIZ     = Path(__file__).resolve().parent.parent.parent
MAT_DIR  = RAIZ / "Material"
OUT_DIR  = RAIZ / "Execution" / "results" / "fase4_benchmarks" / "benchmark_upfd_oficial"

BATCH_SIZE = 32
PATIENCE   = 7
RANDOM_SEED = 42

# (dataset, feature) -- GossipCop usa content (310d) para evitar OOM com BERT esparso (1.8GB denso)
CONFIGS = [
    ("politifact", "bert"),
    ("gossipcop",  "content"),
]


def baseline_textual(train, val, test) -> dict:
    """LogReg sobre x[0] (feature da raiz, i.e. da noticia)."""
    def feats(ds):
        X = torch.stack([d.x[0] for d in ds]).numpy()
        y = np.array([d.y.item() for d in ds], dtype=np.int64)
        return X, y

    X_tr, y_tr = feats(train)
    X_va, y_va = feats(val)
    X_te, y_te = feats(test)
    # Treino unificado em train+val (UPFD nao precisa de val pro LogReg)
    X_full = np.concatenate([X_tr, X_va], axis=0)
    y_full = np.concatenate([y_tr, y_va], axis=0)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_full, y_full)
    y_pred = clf.predict(X_te)
    return {
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1_fake":  float(f1_score(y_te, y_pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
    }


def avaliar_gcn(model, dataset, device) -> dict:
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
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_fake":  float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def treinar_gnn(arch_name, train_data, val_data, num_features, device, epochs, lr,
                 seed) -> torch.nn.Module:
    cls = ARCHS[arch_name]
    model = cls(num_features, 2, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    tr_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
        val_f1 = avaliar_gcn(model, val_data, device)["f1_macro"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--cpu",      action="store_true")
    parser.add_argument("--seeds",    type=int, default=5,
                        help="Numero de seeds para GNNs (default: 5)")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=["politifact", "gossipcop"], default=None,
                        help="Subset de datasets a rodar (default: ambos)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))
    cfgs = [c for c in CONFIGS if (args.datasets is None or c[0] in args.datasets)]

    print("=" * 72)
    print("  Benchmark UPFD Oficial -- GCN/GAT/SAGE vs Baseline Textual")
    print("=" * 72)
    print(f"  Device: {device} | Epochs max: {args.epochs} | Seeds: {seeds}")
    print(f"  Datasets: {[c[0] for c in cfgs]}")
    print()

    linhas_csv = []
    relatorio  = ["=" * 72, "  Benchmark UPFD Oficial -- Resumo", "=" * 72, ""]

    for dataset_name, feature in cfgs:
        print(f"\n-- {dataset_name} (feature={feature}) " + "-" * 30)
        train = UPFD(root=str(MAT_DIR), name=dataset_name, feature=feature, split="train")
        val   = UPFD(root=str(MAT_DIR), name=dataset_name, feature=feature, split="val")
        test  = UPFD(root=str(MAT_DIR), name=dataset_name, feature=feature, split="test")
        num_features = train[0].x.shape[1]
        print(f"   Train={len(train)} | Val={len(val)} | Test={len(test)} | dim_no={num_features}")

        # Baseline textual (LogReg sobre x[0]) -- deterministico
        t0 = time.time()
        res_log = baseline_textual(train, val, test)
        print(f"   LogReg(x[0]): F1m={res_log['f1_macro']:.4f} | F1fake={res_log['f1_fake']:.4f} "
              f"| Acc={res_log['accuracy']:.4f} | {time.time()-t0:.1f}s")
        linhas_csv.append({"dataset": dataset_name, "feature": feature,
                           "modelo": "LogReg_root", "seed": -1,
                           **{k: round(v, 4) for k, v in res_log.items()}})

        # GNNs com multiplas seeds
        gnn_results: dict = {arch: [] for arch in ARCHS}
        train_l, val_l, test_l = list(train), list(val), list(test)
        for arch in ARCHS:
            for seed in seeds:
                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                model = treinar_gnn(arch, train_l, val_l, num_features, device,
                                    args.epochs, args.lr, seed=seed)
                res = avaliar_gcn(model, test_l, device)
                dt = time.time() - t0
                gnn_results[arch].append(res)
                linhas_csv.append({"dataset": dataset_name, "feature": feature,
                                   "modelo": arch, "seed": seed,
                                   **{k: round(v, 4) for k, v in res.items()}})
                print(f"   {arch:<5} seed={seed} | F1m={res['f1_macro']:.4f} | "
                      f"F1fake={res['f1_fake']:.4f} | Acc={res['accuracy']:.4f} | {dt:.1f}s")

        # Relatorio agregado
        relatorio += [f"  [{dataset_name} / {feature}]",
                      f"    Baseline LogReg(x[0]):  F1m={res_log['f1_macro']:.4f}", ""]
        for arch in ARCHS:
            f1ms = np.array([r["f1_macro"] for r in gnn_results[arch]])
            f1fs = np.array([r["f1_fake"]  for r in gnn_results[arch]])
            accs = np.array([r["accuracy"] for r in gnn_results[arch]])
            delta = f1ms.mean() - res_log["f1_macro"]
            sinal = "+" if delta >= 0 else ""
            tag = ("AJUDA" if delta > 0.01 else
                   ("EQUIVALE" if abs(delta) <= 0.01 else "PIORA"))
            relatorio.append(
                f"    {arch:<5}: F1m={f1ms.mean():.4f}+/-{f1ms.std():.4f} "
                f"F1fake={f1fs.mean():.4f}+/-{f1fs.std():.4f} "
                f"Acc={accs.mean():.4f}+/-{accs.std():.4f} | "
                f"delta vs base={sinal}{delta:.4f} ({tag})"
            )
        relatorio.append("")

    csv_path = OUT_DIR / "resultados.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "feature", "modelo", "seed",
                                          "f1_macro", "f1_fake", "accuracy"])
        w.writeheader()
        w.writerows(linhas_csv)

    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(relatorio) + "\n", encoding="utf-8")

    print()
    for l in relatorio:
        print(l)
    print(f"\n[OK] CSV: {csv_path}")
    print(f"[OK] Relatorio: {rel_path}")


if __name__ == "__main__":
    main()
