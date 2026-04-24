"""
17_persistir_modelos_finais.py
------------------------------
Treina e salva os 3 modelos canonicos do TCC para uso na API/inferencia futura.

Modelos persistidos em Execution/weights/:
  1. logreg_bert_fnn.pkl       -- LogReg sobre x[0] (BERT da raiz) treinado em FNN
                                  (referencia textual; F1m ~0.86 no FNN)
  2. rf_struct_gossipcop.pkl   -- RandomForest com [num_nodes, grau_root]
                                  treinado em GossipCop (F1m ~0.75, leve, interpretavel)
  3. sage_struct_gossipcop.pth -- SAGE com [is_root, grau_norm] treinado em GossipCop
                                  (F1m ~0.81, mais pesado mas com GNNExplainer)
  4. metadata.json             -- contagens, scores, hiperparametros, paths

Uso:
  python 17_persistir_modelos_finais.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_model  import SAGEClassifier
from gerar_folds import carregar_grafos as carregar_fnn

RAIZ        = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = Path(__file__).resolve().parent / "data"
MAT_DIR     = RAIZ / "Material"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"

RANDOM_SEED = 42


def features_estruturais(g: Data) -> torch.Tensor:
    n = g.num_nodes
    is_root = torch.zeros(n, dtype=torch.float)
    is_root[0] = 1.0
    deg = torch.zeros(n, dtype=torch.float)
    if g.edge_index.numel() > 0:
        idx, counts = torch.unique(g.edge_index[0], return_counts=True)
        deg[idx] = counts.float()
    deg_norm = deg / max(deg.max().item(), 1.0)
    return torch.stack([is_root, deg_norm], dim=1)


def transformar(ds):
    return [Data(x=features_estruturais(g), edge_index=g.edge_index, y=g.y) for g in ds]


def feats_tabular(grafos: list) -> np.ndarray:
    X = []
    for g in grafos:
        n = g.num_nodes
        grau_root = (g.edge_index[0] == 0).sum().item() if g.edge_index.numel() > 0 else 0
        X.append([n, grau_root])
    return np.array(X, dtype=np.float64)


def avaliar_sage(model, dataset, device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    y_t, y_p = [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            out, _ = model(d.x, d.edge_index, d.batch)
            y_p.extend(out.argmax(dim=1).cpu().tolist())
            labels = d.y.squeeze().cpu().tolist()
            y_t.extend(labels if isinstance(labels, list) else [labels])
    return {
        "f1_macro": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "f1_fake":  float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y_t, y_p)),
    }


def treinar_sage_struct(train, val, num_features, device, epochs=30) -> SAGEClassifier:
    model = SAGEClassifier(num_features, 2, seed=RANDOM_SEED).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    crit  = torch.nn.CrossEntropyLoss()
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    tr    = DataLoader(train, batch_size=32, shuffle=True)
    best_v, best_state, pat = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for d in tr:
            d = d.to(device); opt.zero_grad()
            out, _ = model(d.x, d.edge_index, d.batch)
            crit(out, d.y.squeeze()).backward(); opt.step()
        v = avaliar_sage(model, val, device)["f1_macro"]
        sch.step(v)
        if v > best_v:
            best_v, best_state, pat = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 7: break
    if best_state: model.load_state_dict(best_state)
    return model


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 70)
    print("  Persistindo modelos finais para uso na API / inferencia")
    print("=" * 70)

    metadata = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": RANDOM_SEED, "modelos": {}}

    # ── 1. LogReg-BERT no FNN ────────────────────────────────────────────
    print("\n[1/3] LogReg-BERT (referencia textual) treinado em FakeNewsNet...")
    fnn = carregar_fnn()
    X = torch.stack([g.x[0] for g in fnn]).numpy()  # so a raiz, BERT 768d (legacy bugado)
    y = np.array([g.y.item() for g in fnn], dtype=np.int64)
    # FNN bugado tem dim 768 (sem features posicionais) -- e isso que LogReg precisa
    print(f"   X.shape={X.shape}  y.shape={y.shape}")

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X, y)
    pred = lr.predict(X)
    f1m_train = f1_score(y, pred, average="macro", zero_division=0)
    out_lr = WEIGHTS_DIR / "logreg_bert_fnn.pkl"
    with open(out_lr, "wb") as f:
        pickle.dump({"model": lr, "input_dim": X.shape[1], "label_map": {0: "fake", 1: "real"}}, f)
    print(f"   F1m (treino, sanidade)={f1m_train:.4f} | tempo={time.time()-t0:.1f}s")
    print(f"   [OK] {out_lr.name}")
    metadata["modelos"]["logreg_bert_fnn"] = {
        "tipo": "sklearn.LogisticRegression",
        "input_dim": int(X.shape[1]),
        "treinado_em": "FakeNewsNet (754 grafos, x[0]=BERT raiz)",
        "f1_macro_treino": round(f1m_train, 4),
        "path": str(out_lr.relative_to(RAIZ)),
        "uso": "classificador textual canonico; rapido; equivalente a GCN no FNN",
    }

    # ── 2. RandomForest estrutural no GossipCop ────────────────────────────
    print("\n[2/3] RandomForest estrutural [num_nodes, grau_root] em GossipCop...")
    train = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="train"))
    val   = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="val"))
    test  = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="test"))
    print(f"   GossipCop: train={len(train)} val={len(val)} test={len(test)}")

    Xtr = feats_tabular(train + val)
    ytr = np.array([g.y.item() for g in (train + val)], dtype=np.int64)
    Xte = feats_tabular(test)
    yte = np.array([g.y.item() for g in test], dtype=np.int64)

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    f1m_test = f1_score(yte, pred, average="macro", zero_division=0)
    acc_test = accuracy_score(yte, pred)
    out_rf = WEIGHTS_DIR / "rf_struct_gossipcop.pkl"
    with open(out_rf, "wb") as f:
        pickle.dump({"model": rf, "feature_names": ["num_nodes", "grau_root"],
                     "label_map": {0: "fake", 1: "real"}}, f)
    print(f"   F1m={f1m_test:.4f}  Acc={acc_test:.4f}  tempo={time.time()-t0:.1f}s")
    print(f"   [OK] {out_rf.name}")
    metadata["modelos"]["rf_struct_gossipcop"] = {
        "tipo": "sklearn.RandomForestClassifier",
        "n_estimators": 200,
        "input_features": ["num_nodes", "grau_root"],
        "treinado_em": "UPFD-GossipCop (1638 grafos train+val)",
        "avaliado_em":  "UPFD-GossipCop test (3826 grafos)",
        "f1_macro_test": round(f1m_test, 4),
        "accuracy_test": round(acc_test, 4),
        "path": str(out_rf.relative_to(RAIZ)),
        "uso": "classificador topologico leve e interpretavel; "
               "modelo principal de fallback quando texto nao esta disponivel",
    }

    # ── 3. SAGE estrutural no GossipCop ────────────────────────────────────
    print("\n[3/3] SAGE estrutural [is_root, grau_norm] em GossipCop (mais pesado)...")
    train_t = transformar(train)
    val_t   = transformar(val)
    test_t  = transformar(test)
    t0 = time.time()
    sage = treinar_sage_struct(train_t, val_t, num_features=2, device=device, epochs=30)
    res = avaliar_sage(sage, test_t, device)
    out_sage = WEIGHTS_DIR / "sage_struct_gossipcop.pth"
    torch.save({
        "state_dict": sage.state_dict(),
        "arch": "SAGEClassifier",
        "input_dim": 2,
        "feature_names": ["is_root", "grau_norm"],
        "hidden_channels": 64,  # default
        "label_map": {0: "fake", 1: "real"},
    }, out_sage)
    print(f"   F1m={res['f1_macro']:.4f}  F1fake={res['f1_fake']:.4f}  "
          f"Acc={res['accuracy']:.4f}  tempo={time.time()-t0:.1f}s")
    print(f"   [OK] {out_sage.name}")
    metadata["modelos"]["sage_struct_gossipcop"] = {
        "tipo": "torch_geometric.SAGEClassifier",
        "input_features": ["is_root", "grau_norm"],
        "hidden_channels": 64,
        "treinado_em": "UPFD-GossipCop (1092 train, 546 val)",
        "avaliado_em": "UPFD-GossipCop test (3826)",
        **{f"{k}_test": round(v, 4) for k, v in res.items()},
        "path": str(out_sage.relative_to(RAIZ)),
        "uso": "classificador topologico GNN; usado para visualizacao via "
               "GNNExplainer (arestas decisivas) na ferramenta",
    }

    # ── Metadata ──────────────────────────────────────────────────────────
    meta_path = WEIGHTS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] {meta_path.name}")
    print(f"\nResumo:")
    for nome, m in metadata["modelos"].items():
        f1key = next((k for k in m if "f1_macro" in k), None)
        print(f"  - {nome:<28} F1m={m.get(f1key, '-')}  -> {m['path']}")


if __name__ == "__main__":
    main()
