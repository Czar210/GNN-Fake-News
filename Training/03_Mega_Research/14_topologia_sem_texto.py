"""
14_topologia_sem_texto.py
-------------------------
Experimento "topologia SEM texto" -- responde: "se o texto nao estivesse
disponivel (post em lingua desconhecida, audio, post deletado), quanto sinal
sobra na propagacao?"

Substitui features textuais por features puramente estruturais e treina
GCN/GAT/SAGE em 3 datasets:
  - FakeNewsNet (nosso): reusa o data/fakenewsnet_posfull/ slicing as 3 ultimas
    dims (que ja sao [is_root, grau_norm, pos] -- ver Fase 3.1).
  - UPFD-PolitiFact: deriva [is_root, grau_norm, pos] do edge_index.
  - UPFD-GossipCop: idem.

3 variantes de feature por nó:
  - A_isroot:    [is_root]                       (1 dim)
  - B_estrutural:[is_root, grau_norm]            (2 dims)
  - C_posicional:[is_root, grau_norm, pos_norm]  (3 dims)

Comparacao:
  - vs chance teorica (~0.5 com balanco ~50/50)
  - vs baseline textual (Fase 2A.1: LogReg-BERT ~0.86 no FNN)
  - vs GCN com features completas (Fase 3.1.bis no FNN; Fase cross-dataset
    no UPFD)

Saida em Execution/results/fase4_benchmarks/topologia_sem_texto/:
  resultados.csv  -- por (dataset, variante, modelo, fold/seed)
  relatorio.txt   -- agregado media+/-std + interpretacao

Uso:
  python 14_topologia_sem_texto.py            # roda os 3 datasets
  python 14_topologia_sem_texto.py --datasets fnn upfd_politifact
  python 14_topologia_sem_texto.py --epochs 30
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model   import GCNClassifier
from gat_model   import GATClassifier
from sage_model  import SAGEClassifier
from gerar_folds import iterar_folds

RAIZ     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
MAT_DIR  = RAIZ / "Material"
OUT_DIR  = RAIZ / "Execution" / "results" / "fase4_benchmarks" / "topologia_sem_texto"

ARCHS    = {"GCN": GCNClassifier, "GAT": GATClassifier, "SAGE": SAGEClassifier}
VARIANTS = ["A_isroot", "B_estrutural", "C_posicional"]
DIMS     = {"A_isroot": 1, "B_estrutural": 2, "C_posicional": 3}

BATCH_SIZE = 32
PATIENCE   = 7


# ─── Construcao de features estruturais ──────────────────────────────────────

def features_estruturais_upfd(g: Data, variant: str) -> torch.Tensor:
    """
    Computa features estruturais para um grafo UPFD.
    Convencao UPFD: no 0 e a raiz (noticia), demais sao usuarios.
    """
    n = g.num_nodes
    is_root = torch.zeros(n, dtype=torch.float)
    is_root[0] = 1.0

    if variant == "A_isroot":
        return is_root.unsqueeze(1)

    # grau (somente origem, ja que UPFD eh DAG raiz->filhos)
    deg = torch.zeros(n, dtype=torch.float)
    if g.edge_index.numel() > 0:
        idx, counts = torch.unique(g.edge_index[0], return_counts=True)
        deg[idx] = counts.float()
    deg_max = max(deg.max().item(), 1.0)
    deg_norm = deg / deg_max

    if variant == "B_estrutural":
        return torch.stack([is_root, deg_norm], dim=1)

    # pos: ordem do no quando agregado a partir da raiz (BFS).
    # Se nao tiver caminho da raiz, usa indice do no como fallback.
    pos = torch.zeros(n, dtype=torch.float)
    visitado = {0}
    fila = [0]
    ordem = 0
    pos[0] = 0.0
    if g.edge_index.numel() > 0:
        # adjacencia
        adj: dict = {i: [] for i in range(n)}
        for s, t in g.edge_index.t().tolist():
            adj[s].append(t)
            adj[t].append(s)  # tratamos como nao-direcionado pra propagacao
        while fila:
            u = fila.pop(0)
            for v in adj[u]:
                if v not in visitado:
                    visitado.add(v)
                    ordem += 1
                    pos[v] = ordem
                    fila.append(v)
    pos_max = max(pos.max().item(), 1.0)
    pos_norm = pos / pos_max
    return torch.stack([is_root, deg_norm, pos_norm], dim=1)


def transformar_grafos_upfd(dataset_iterable, variant: str) -> list:
    novos = []
    for g in dataset_iterable:
        x_novo = features_estruturais_upfd(g, variant)
        novos.append(Data(x=x_novo, edge_index=g.edge_index, y=g.y))
    return novos


def carregar_fnn_topologia(variant: str) -> list:
    """Para FNN, usa as 3 ultimas dims do data/fakenewsnet_posfull/ ja prontas."""
    base = DATA_DIR / "fakenewsnet_posfull"
    grafos = []
    for split in ("train", "val", "test"):
        grafos += torch.load(base / f"fakenewsnet_{split}.pt", weights_only=False)
    n_take = DIMS[variant]
    out = []
    for g in grafos:
        # x original: [N, 771] = [BERT(768) | is_root | grau | pos]
        x_novo = g.x[:, 768:768 + n_take].contiguous().clone()
        out.append(Data(x=x_novo, edge_index=g.edge_index, y=g.y))
    return out


# ─── Treinamento ─────────────────────────────────────────────────────────────

def avaliar(model, dataset, device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
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


def treinar(arch_name, train_d, val_d, num_features, device, epochs, lr, seed):
    cls = ARCHS[arch_name]
    model = cls(num_features, 2, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss()
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    tr = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    best_val, best_state, pat = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for d in tr:
            d = d.to(device); opt.zero_grad()
            out, _ = model(d.x, d.edge_index, d.batch)
            crit(out, d.y.squeeze()).backward(); opt.step()
        v = avaliar(model, val_d, device)["f1_macro"]
        sch.step(v)
        if v > best_val:
            best_val, best_state, pat = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    return model


def split_train_val_from_indices(train_idx, grafos, val_frac=0.10, seed=0):
    import random
    rng = random.Random(seed)
    fakes = [i for i in train_idx if grafos[i].y.item() == 0]
    reals = [i for i in train_idx if grafos[i].y.item() == 1]
    rng.shuffle(fakes); rng.shuffle(reals)
    n_vf = max(1, int(len(fakes) * val_frac))
    n_vr = max(1, int(len(reals) * val_frac))
    val_idx   = fakes[:n_vf] + reals[:n_vr]
    train_sub = fakes[n_vf:] + reals[n_vr:]
    return [grafos[i] for i in train_sub], [grafos[i] for i in val_idx]


# ─── Loops por dataset ───────────────────────────────────────────────────────

def rodar_fnn(variants, archs, seeds_per_fold, epochs, lr, device) -> list:
    """FNN com k-fold (folds_fnn.pt). Retorna lista de dicts."""
    folds = list(iterar_folds())
    linhas = []
    for variant in variants:
        grafos = carregar_fnn_topologia(variant)
        num_feats = grafos[0].x.shape[1]
        print(f"\n  -- FNN / variant={variant} (dim={num_feats}) --")
        for fold_idx, train_idx, test_idx in folds:
            train_sub, val_sub = split_train_val_from_indices(train_idx, grafos, seed=fold_idx)
            test_sub = [grafos[i] for i in test_idx]
            for arch in archs:
                t0 = time.time()
                torch.manual_seed(fold_idx); np.random.seed(fold_idx)
                m = treinar(arch, train_sub, val_sub, num_feats, device, epochs, lr, seed=fold_idx)
                r = avaliar(m, test_sub, device)
                dt = time.time() - t0
                linhas.append({"dataset": "fnn", "variant": variant, "modelo": arch,
                               "fold_or_seed": fold_idx, **r})
                print(f"    fold {fold_idx:2d} | {arch:<5} | F1m={r['f1_macro']:.4f} | "
                      f"F1fake={r['f1_fake']:.4f} | Acc={r['accuracy']:.4f} | {dt:.1f}s")
    return linhas


def rodar_upfd(name, variants, archs, seeds, epochs, lr, device) -> list:
    """UPFD com split oficial + multiplas seeds."""
    print(f"\n  -- UPFD-{name} (carregando feature=profile pra topologia) --")
    train = list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="train"))
    val   = list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="val"))
    test  = list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="test"))
    print(f"     Train={len(train)} | Val={len(val)} | Test={len(test)}")

    linhas = []
    for variant in variants:
        train_t = transformar_grafos_upfd(train, variant)
        val_t   = transformar_grafos_upfd(val,   variant)
        test_t  = transformar_grafos_upfd(test,  variant)
        num_feats = train_t[0].x.shape[1]
        print(f"\n     variant={variant} (dim={num_feats})")
        for arch in archs:
            for seed in seeds:
                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                m = treinar(arch, train_t, val_t, num_feats, device, epochs, lr, seed=seed)
                r = avaliar(m, test_t, device)
                dt = time.time() - t0
                linhas.append({"dataset": f"upfd_{name}", "variant": variant, "modelo": arch,
                               "fold_or_seed": seed, **r})
                print(f"     {arch:<5} seed={seed} | F1m={r['f1_macro']:.4f} | "
                      f"F1fake={r['f1_fake']:.4f} | Acc={r['accuracy']:.4f} | {dt:.1f}s")
    return linhas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--cpu",      action="store_true")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["fnn", "upfd_politifact", "upfd_gossipcop"],
                        choices=["fnn", "upfd_politifact", "upfd_gossipcop"])
    parser.add_argument("--variants", type=str, nargs="+",
                        default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--archs",    type=str, nargs="+",
                        default=list(ARCHS.keys()), choices=list(ARCHS.keys()))
    parser.add_argument("--seeds",    type=int, default=3,
                        help="numero de seeds para UPFD (default 3)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Topologia SEM texto -- features estruturais puras")
    print("=" * 72)
    print(f"  Datasets: {args.datasets}")
    print(f"  Variants: {args.variants}")
    print(f"  Archs: {args.archs}")

    todas_linhas = []
    for ds in args.datasets:
        if ds == "fnn":
            todas_linhas += rodar_fnn(args.variants, args.archs, args.seeds,
                                      args.epochs, args.lr, device)
        elif ds == "upfd_politifact":
            todas_linhas += rodar_upfd("politifact", args.variants, args.archs,
                                       list(range(args.seeds)), args.epochs, args.lr, device)
        elif ds == "upfd_gossipcop":
            todas_linhas += rodar_upfd("gossipcop", args.variants, args.archs,
                                       list(range(args.seeds)), args.epochs, args.lr, device)

    # Salvar CSV
    csv_path = OUT_DIR / "resultados.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "modelo", "fold_or_seed",
                                          "f1_macro", "f1_fake", "accuracy"])
        w.writeheader()
        for r in todas_linhas:
            w.writerow({**r, "f1_macro": round(r["f1_macro"], 4),
                              "f1_fake":  round(r["f1_fake"],  4),
                              "accuracy": round(r["accuracy"], 4)})

    # Relatorio agregado
    rel = ["=" * 72, "  Topologia SEM texto -- Resumo (media +/- std)", "=" * 72, ""]
    chaves = sorted({(r["dataset"], r["variant"], r["modelo"]) for r in todas_linhas})
    rel.append(f"  {'dataset':<20} {'variant':<14} {'modelo':<6} {'F1m':>15} {'F1fake':>15} {'Acc':>15}")
    rel.append(f"  {'-'*92}")
    for ds, var, arch in chaves:
        vals = [r for r in todas_linhas if r["dataset"]==ds and r["variant"]==var and r["modelo"]==arch]
        f1m = np.array([v["f1_macro"] for v in vals])
        f1f = np.array([v["f1_fake"]  for v in vals])
        acc = np.array([v["accuracy"] for v in vals])
        rel.append(f"  {ds:<20} {var:<14} {arch:<6} "
                   f"{f1m.mean():.3f}+/-{f1m.std():.3f}   "
                   f"{f1f.mean():.3f}+/-{f1f.std():.3f}   "
                   f"{acc.mean():.3f}+/-{acc.std():.3f}")

    rel_path = OUT_DIR / "relatorio.txt"
    rel_path.write_text("\n".join(rel) + "\n", encoding="utf-8")

    print()
    for l in rel: print(l)
    print(f"\n[OK] {csv_path}")
    print(f"[OK] {rel_path}")


if __name__ == "__main__":
    main()
