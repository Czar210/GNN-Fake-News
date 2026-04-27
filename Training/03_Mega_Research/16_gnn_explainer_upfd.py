"""
16_gnn_explainer_upfd.py
------------------------
Aplica GNNExplainer aos modelos SAGE treinados com features estruturais puras
no UPFD-GossipCop (caso forte: F1m=0.81 sem texto) e UPFD-PolitiFact (caso
fraco: ~0.33). Mostra QUAIS arestas/nos foram decisivos para a classificacao
de cada amostra.

Para cada dataset:
  1. Treina SAGE com seed fixa nas features B_estrutural [is_root, grau_norm]
  2. Avalia no test set
  3. Seleciona 20 amostras estratificadas (resposta a feedback de banca):
     - 5 FAKE-pequena (n_nodes < mediana, classificadas corretamente, ranqueadas por confianca)
     - 5 FAKE-grande (n_nodes >= mediana, classificadas corretamente)
     - 5 REAL-pequena (n_nodes < mediana, classificadas corretamente)
     - 5 REAL-grande (n_nodes >= mediana, classificadas corretamente)
  4. Aplica GNNExplainer (edge_mask) em cada
  5. Calcula hop-distance importance: fracao de massa de importancia em
     arestas a 1-hop, 2-hop, 3+ hops da raiz (BFS distance).
  6. Exporta:
     - PNG: layout com arestas coloridas por importancia
     - HTML PyVis interativo com espessura/cor proporcional
     - hop_importance.csv: agregado por (classe, tamanho)

Saidas em Execution/results/figuras_tcc/gnnexplainer/:
  <dataset>_<idx>_<classe>_pred<p>.png/html
  resumo.txt    -- ranking de arestas decisivas + estatisticas

Uso:
  python 16_gnn_explainer_upfd.py
  python 16_gnn_explainer_upfd.py --dataset gossipcop
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_model import SAGEClassifier

RAIZ     = Path(__file__).resolve().parent.parent.parent
MAT_DIR  = RAIZ / "Material"
OUT_DIR  = RAIZ / "Execution" / "results" / "figuras_tcc" / "gnnexplainer"

BATCH_SIZE  = 32
PATIENCE    = 7
RANDOM_SEED = 42


# Wrapper sem retorno duplo (GNNExplainer espera apenas logits)
class _SAGEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out, _ = self.model(x, edge_index, batch)
        return out


def features_estruturais(g: Data) -> torch.Tensor:
    """[is_root, grau_normalizado] por nó."""
    n = g.num_nodes
    is_root = torch.zeros(n, dtype=torch.float)
    is_root[0] = 1.0
    deg = torch.zeros(n, dtype=torch.float)
    if g.edge_index.numel() > 0:
        idx, counts = torch.unique(g.edge_index[0], return_counts=True)
        deg[idx] = counts.float()
    deg_norm = deg / max(deg.max().item(), 1.0)
    return torch.stack([is_root, deg_norm], dim=1)


def transformar(dataset_iterable):
    return [Data(x=features_estruturais(g), edge_index=g.edge_index, y=g.y)
            for g in dataset_iterable]


def avaliar(model, dataset, device) -> tuple:
    """Retorna (f1_macro, predicoes_por_grafo: list of dict)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_t, y_p, probs = [], [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            out, _ = model(d.x, d.edge_index, d.batch)
            p = F.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu().tolist()
            y_p.extend(preds)
            probs.extend(p[:, 0].cpu().tolist())  # prob de classe 0 = fake
            labels = d.y.squeeze().cpu().tolist()
            y_t.extend(labels if isinstance(labels, list) else [labels])
    f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
    info = [{"y_true": yt, "y_pred": yp, "prob_fake": pf}
            for yt, yp, pf in zip(y_t, y_p, probs)]
    return f1, info


def treinar_sage(train, val, num_features, device, epochs=30, lr=0.001, seed=42):
    model = SAGEClassifier(num_features, 2, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss()
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    tr = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    best_val, best_state, pat = 0.0, None, 0
    for epoch in range(epochs):
        model.train()
        for d in tr:
            d = d.to(device); opt.zero_grad()
            out, _ = model(d.x, d.edge_index, d.batch)
            crit(out, d.y.squeeze()).backward(); opt.step()
        v = avaliar(model, val, device)[0]
        sch.step(v)
        if v > best_val:
            best_val, best_state, pat = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    return model


def selecionar_amostras(test, info, n_per_strata: int = 5) -> list:
    """
    Selecao estratificada (resposta a feedback de banca):
    n_per_strata amostras de cada um dos 4 estratos:
      (FAKE, pequena), (FAKE, grande), (REAL, pequena), (REAL, grande).

    "pequena" = num_nodes < mediana global do test set
    "grande"  = num_nodes >= mediana
    Apenas amostras CORRETAMENTE classificadas (y_true == y_pred).
    Dentro de cada estrato, ranqueia por confianca (prob_fake mais distante de 0.5).
    """
    sizes = np.array([test[i].num_nodes for i in range(len(test))])
    mediana = float(np.median(sizes))

    estratos = {
        ("FAKE", "pequena"): [],
        ("FAKE", "grande"):  [],
        ("REAL", "pequena"): [],
        ("REAL", "grande"):  [],
    }
    for i, x in enumerate(info):
        if x["y_true"] != x["y_pred"]:
            continue
        if test[i].num_nodes < 3 or test[i].edge_index.numel() == 0:
            continue
        classe = "FAKE" if x["y_true"] == 0 else "REAL"
        tam = "pequena" if test[i].num_nodes < mediana else "grande"
        # Score de confianca: |prob_fake - 0.5| -- maior = mais confiante
        confianca = abs(x["prob_fake"] - 0.5)
        estratos[(classe, tam)].append((i, x, confianca))

    sel = []
    for k, lista in estratos.items():
        # Top n_per_strata por confianca
        lista.sort(key=lambda t: -t[2])
        for i, ent, _ in lista[:n_per_strata]:
            sel.append((i, ent, k))   # carrega o estrato pra logging downstream

    print(f"   Mediana de num_nodes (corte pequena/grande): {mediana:.1f}")
    print(f"   Estratos: " + ", ".join(
        f"{k}={min(len(v), n_per_strata)}/{n_per_strata}" for k, v in estratos.items()))
    return sel


def hop_distance_da_raiz(num_nodes: int, edge_index: torch.Tensor) -> dict:
    """BFS a partir do no 0 (raiz). Retorna dict {edge_idx_no_edge_index: hop_dist}.
    hop_dist = min(d(raiz, src), d(raiz, tgt)) + 1 (a aresta esta naquele hop)."""
    if edge_index.numel() == 0:
        return {}
    adj = {i: [] for i in range(num_nodes)}
    for s, t in edge_index.t().tolist():
        adj[s].append(t)
        adj[t].append(s)
    dist = {0: 0}
    fila = [0]
    while fila:
        u = fila.pop(0)
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                fila.append(v)
    out = {}
    for k, (s, t) in enumerate(edge_index.t().tolist()):
        ds = dist.get(s, 99)
        dt = dist.get(t, 99)
        out[k] = min(ds, dt) + 1   # ex: aresta raiz->filho = hop 1
    return out


def analisar_hop_importance(g, edge_mask: torch.Tensor) -> dict:
    """Retorna {1: frac_1hop, 2: frac_2hop, '3+': frac_3plus} -- fracao da massa total."""
    hops = hop_distance_da_raiz(g.num_nodes, g.edge_index)
    total = float(edge_mask.sum().clamp(min=1e-9).item())
    massa = {1: 0.0, 2: 0.0, "3+": 0.0}
    for k, w in enumerate(edge_mask.tolist()):
        h = hops.get(k, 99)
        if h == 1: massa[1] += w
        elif h == 2: massa[2] += w
        else: massa["3+"] += w
    return {k: v / total for k, v in massa.items()}


def explicar(explainer, g, device) -> torch.Tensor:
    """Retorna edge_mask: tensor [num_edges] com importancia por aresta."""
    g = g.to(device)
    with torch.enable_grad():
        explanation = explainer(g.x, g.edge_index)
    return explanation.edge_mask.detach().cpu()


def visualizar_png(g, edge_mask, titulo, out_path: Path) -> None:
    """Layout estatico matplotlib: nos, arestas com cor/grossura por importancia."""
    n = g.num_nodes
    if g.edge_index.numel() == 0:
        return
    edges = g.edge_index.t().tolist()

    # Layout simples radial: raiz no centro, demais em circulo (ou camadas BFS)
    pos = {0: (0.0, 0.0)}
    # Filhos imediatos da raiz em circulo
    filhos_root = [t for s, t in edges if s == 0]
    if filhos_root:
        for k, v in enumerate(filhos_root):
            ang = 2 * np.pi * k / max(len(filhos_root), 1)
            pos[v] = (np.cos(ang), np.sin(ang))
    # Restantes: distribui em raio maior
    restantes = [v for v in range(n) if v not in pos]
    for k, v in enumerate(restantes):
        ang = 2 * np.pi * k / max(len(restantes), 1)
        pos[v] = (1.8 * np.cos(ang + 0.5), 1.8 * np.sin(ang + 0.5))

    fig, ax = plt.subplots(figsize=(10, 10))
    em_min, em_max = float(edge_mask.min()), float(edge_mask.max())
    rng = max(em_max - em_min, 1e-9)
    for (s, t), w in zip(edges, edge_mask.tolist()):
        if s not in pos or t not in pos: continue
        norm = (w - em_min) / rng
        x = [pos[s][0], pos[t][0]]
        y = [pos[s][1], pos[t][1]]
        ax.plot(x, y, color=plt.cm.Reds(0.3 + 0.7*norm),
                linewidth=0.5 + 4 * norm, alpha=0.6 + 0.4*norm, zorder=1)

    # Nos: raiz destacada
    for i in range(n):
        if i not in pos: continue
        x, y = pos[i]
        if i == 0:
            ax.scatter(x, y, s=400, c="#E07B54", edgecolors="black", lw=1.5, zorder=2)
            ax.text(x, y, "R", ha="center", va="center", fontweight="bold", fontsize=10)
        else:
            ax.scatter(x, y, s=80, c="#5BAD72", edgecolors="black", lw=0.7, zorder=2)

    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(titulo, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def visualizar_html(g, edge_mask, titulo, out_path: Path) -> None:
    """PyVis interativo: arestas com width/cor por importancia."""
    try:
        from pyvis.network import Network
    except ImportError:
        return
    n = g.num_nodes
    if g.edge_index.numel() == 0:
        return
    em_min, em_max = float(edge_mask.min()), float(edge_mask.max())
    rng = max(em_max - em_min, 1e-9)
    net = Network(height="600px", width="100%", directed=False, notebook=False,
                  heading=titulo)
    for i in range(n):
        if i == 0:
            net.add_node(0, label="RAIZ", color="#E07B54", size=25)
        else:
            net.add_node(i, label=f"u{i}", color="#5BAD72", size=10)
    for (s, t), w in zip(g.edge_index.t().tolist(), edge_mask.tolist()):
        norm = (w - em_min) / rng
        cor_int = int(255 * (0.3 + 0.7 * norm))
        cor = f"rgb({cor_int}, {255 - int(150 * norm)}, {255 - int(200 * norm)})"
        net.add_edge(int(s), int(t),
                     width=1 + 6 * norm,
                     color=cor,
                     title=f"importancia={w:.4f}")
    net.barnes_hut(spring_length=80)
    try:
        net.write_html(str(out_path), notebook=False, open_browser=False)
    except Exception:
        pass


def rodar_dataset(name: str, device, args) -> dict:
    OUT_DS = OUT_DIR / name
    OUT_DS.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  GNNExplainer / UPFD-{name}\n{'='*60}")
    print("[1] Carregando dataset (feature=profile, somente p/ topologia)...")
    train = transformar(list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="train")))
    val   = transformar(list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="val")))
    test  = transformar(list(UPFD(root=str(MAT_DIR), name=name, feature="profile", split="test")))
    print(f"   Train={len(train)} | Val={len(val)} | Test={len(test)}")

    print("[2] Treinando SAGE com features [is_root, grau_norm]...")
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    model = treinar_sage(train, val, num_features=2, device=device,
                          epochs=args.epochs, lr=args.lr, seed=RANDOM_SEED)
    f1, info = avaliar(model, test, device)
    print(f"   F1m={f1:.4f} | tempo={time.time()-t0:.1f}s")

    print("[3] Inicializando GNNExplainer...")
    explainer = Explainer(
        model=_SAGEWrapper(model),
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification",
                          task_level="graph", return_type="raw"),
    )

    print("[4] Selecionando 20 amostras estratificadas (5 por estrato) e explicando...")
    selecionados = selecionar_amostras(test, info, n_per_strata=5)
    sumario = []
    hop_rows = []   # para CSV agregado por estrato
    for idx, ent, estrato in selecionados:
        g = test[idx]
        em = explicar(explainer, g, device)
        edges = g.edge_index.t().tolist()
        ranked = sorted(zip(edges, em.tolist()), key=lambda kv: -kv[1])[:5]
        hop_imp = analisar_hop_importance(g, em)

        classe, tam = estrato
        ok = "OK"
        prob = ent["prob_fake"]
        titulo = (f"{name} idx={idx} | {classe}-{tam} (pred={ent['y_pred']}, "
                  f"p_fake={prob:.3f}) | hop1={hop_imp[1]:.2f} hop2={hop_imp[2]:.2f}")

        png_path = OUT_DS / f"idx{idx:04d}_{classe}_{tam}.png"
        html_path = OUT_DS / f"idx{idx:04d}_{classe}_{tam}.html"
        visualizar_png(g, em, titulo, png_path)
        visualizar_html(g, em, titulo, html_path)

        sumario.append({"idx": idx, "classe": classe, "tam": tam,
                        "pred": ent["y_pred"], "p_fake": prob, "ok": ok,
                        "n_nos": g.num_nodes, "n_arestas": len(edges),
                        "hop1": hop_imp[1], "hop2": hop_imp[2], "hop3plus": hop_imp["3+"],
                        "top5_arestas": ranked, "png": png_path.name})
        hop_rows.append({"classe": classe, "tam": tam, "idx": idx,
                         "n_nos": g.num_nodes,
                         "frac_hop1": hop_imp[1], "frac_hop2": hop_imp[2],
                         "frac_hop3plus": hop_imp["3+"]})
        print(f"   [OK] idx={idx} {classe}-{tam} hop1={hop_imp[1]:.2f}")

    # CSV de hop-importance
    import csv as _csv
    csv_path = OUT_DS / "hop_importance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["classe","tam","idx","n_nos",
                                            "frac_hop1","frac_hop2","frac_hop3plus"])
        w.writeheader()
        for r in hop_rows:
            r2 = {**r, "frac_hop1": round(r["frac_hop1"], 4),
                  "frac_hop2": round(r["frac_hop2"], 4),
                  "frac_hop3plus": round(r["frac_hop3plus"], 4)}
            w.writerow(r2)

    # Agregado por estrato
    estratos = sorted({(r["classe"], r["tam"]) for r in hop_rows})
    agg_lines = ["", "Agregado por estrato (media +/- std da fracao de massa):", ""]
    agg_lines.append(f"  {'estrato':<20} {'n':>3} {'hop1':>15} {'hop2':>15} {'hop3+':>15}")
    agg_lines.append(f"  {'-'*70}")
    for c, t in estratos:
        sub = [r for r in hop_rows if r["classe"] == c and r["tam"] == t]
        if not sub: continue
        h1 = np.array([r["frac_hop1"] for r in sub])
        h2 = np.array([r["frac_hop2"] for r in sub])
        h3 = np.array([r["frac_hop3plus"] for r in sub])
        agg_lines.append(f"  {c+'-'+t:<20} {len(sub):>3} "
                         f"{h1.mean():.3f}+/-{h1.std():.3f}   "
                         f"{h2.mean():.3f}+/-{h2.std():.3f}   "
                         f"{h3.mean():.3f}+/-{h3.std():.3f}")

    # Resumo textual
    rel_path = OUT_DS / "resumo.txt"
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(f"GNNExplainer / UPFD-{name}\n")
        f.write(f"SAGE com features [is_root, grau_normalizado], seed={RANDOM_SEED}\n")
        f.write(f"F1m no test: {f1:.4f}\n")
        f.write(f"N amostras explicadas (estratificadas): {len(sumario)}\n")
        for line in agg_lines: f.write(line + "\n")
        f.write("\n\n")
        for s in sumario:
            f.write(f"\n--- idx={s['idx']}  {s['classe']}-{s['tam']}  pred={s['pred']}  "
                    f"p_fake={s['p_fake']:.3f}  "
                    f"({s['n_nos']} nos, {s['n_arestas']} arestas) ---\n")
            f.write(f"   Hop importance: hop1={s['hop1']:.3f}  "
                    f"hop2={s['hop2']:.3f}  hop3+={s['hop3plus']:.3f}\n")
            f.write(f"   Top-5 arestas (importancia GNNExplainer):\n")
            for (src, tgt), imp in s["top5_arestas"]:
                f.write(f"     {src} -> {tgt}  importancia = {imp:.4f}\n")
    print(f"\n   Resumo: {rel_path}")
    print(f"   Hop CSV: {csv_path}")
    return {"f1": f1, "n_amostras": len(sumario), "out_dir": str(OUT_DS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--cpu",      action="store_true")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=["politifact", "gossipcop"],
                        default=["gossipcop", "politifact"])
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GNNExplainer aplicado a SAGE com features estruturais (UPFD)")
    print("=" * 70)

    resumo_global = {}
    for ds in args.datasets:
        r = rodar_dataset(ds, device, args)
        resumo_global[ds] = r

    print("\n" + "=" * 70)
    print("  Concluido")
    print("=" * 70)
    for ds, r in resumo_global.items():
        print(f"  UPFD-{ds}: F1m={r['f1']:.4f} | {r['n_amostras']} amostras explicadas | {r['out_dir']}")


if __name__ == "__main__":
    main()
