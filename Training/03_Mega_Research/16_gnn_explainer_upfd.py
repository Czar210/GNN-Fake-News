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
  3. Seleciona 5 amostras interessantes:
     - 2 FAKE corretamente classificadas (alta confianca)
     - 2 REAL corretamente classificadas (alta confianca)
     - 1 erro (baixa confianca) -- bordo/dificil
  4. Aplica GNNExplainer (edge_mask) em cada
  5. Exporta:
     - PNG: layout com arestas coloridas por importancia
     - HTML PyVis interativo com espessura/cor proporcional

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


def selecionar_amostras(test, info) -> list:
    """Seleciona ate 5 amostras interessantes."""
    # 2 fake corretas com alta confianca
    fake_corr = sorted(
        [(i, x) for i, x in enumerate(info) if x["y_true"] == 0 and x["y_pred"] == 0],
        key=lambda kv: -kv[1]["prob_fake"]
    )[:2]
    real_corr = sorted(
        [(i, x) for i, x in enumerate(info) if x["y_true"] == 1 and x["y_pred"] == 1],
        key=lambda kv: kv[1]["prob_fake"]   # menor prob_fake = real mais "real"
    )[:2]
    erro = sorted(
        [(i, x) for i, x in enumerate(info) if x["y_true"] != x["y_pred"]],
        key=lambda kv: -abs(kv[1]["prob_fake"] - 0.5)  # pega o "mais errado"
    )[:1]
    sel = fake_corr + real_corr + erro
    return [(i, info[i]) for i, _ in sel]


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

    print("[4] Selecionando amostras e gerando explicacoes...")
    selecionados = selecionar_amostras(test, info)
    sumario = []
    for idx, ent in selecionados:
        g = test[idx]
        # ignora grafos triviais
        if g.num_nodes < 3 or g.edge_index.numel() == 0:
            continue
        em = explicar(explainer, g, device)
        # rank top-5 arestas
        edges = g.edge_index.t().tolist()
        ranked = sorted(zip(edges, em.tolist()), key=lambda kv: -kv[1])[:5]

        classe = "FAKE" if ent["y_true"] == 0 else "REAL"
        ok = "OK" if ent["y_true"] == ent["y_pred"] else "ERRO"
        prob = ent["prob_fake"]
        titulo = f"{name} idx={idx} | {classe} (pred={ent['y_pred']}, p_fake={prob:.3f}) | {ok}"

        png_path = OUT_DS / f"idx{idx:04d}_{classe}_pred{ent['y_pred']}_{ok}.png"
        html_path = OUT_DS / f"idx{idx:04d}_{classe}_pred{ent['y_pred']}_{ok}.html"
        visualizar_png(g, em, titulo, png_path)
        visualizar_html(g, em, titulo, html_path)

        sumario.append({"idx": idx, "classe": classe, "pred": ent["y_pred"],
                        "p_fake": prob, "ok": ok, "n_nos": g.num_nodes,
                        "n_arestas": len(edges),
                        "top5_arestas": ranked,
                        "png": png_path.name})
        print(f"   [OK] idx={idx} {classe} {ok} -> {png_path.name}")

    # Resumo textual
    rel_path = OUT_DS / "resumo.txt"
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(f"GNNExplainer / UPFD-{name}\n")
        f.write(f"SAGE com features [is_root, grau_normalizado], seed={RANDOM_SEED}\n")
        f.write(f"F1m no test: {f1:.4f}\n\n")
        for s in sumario:
            f.write(f"\n--- idx={s['idx']}  {s['classe']}  pred={s['pred']}  "
                    f"p_fake={s['p_fake']:.3f}  {s['ok']}  "
                    f"({s['n_nos']} nos, {s['n_arestas']} arestas) ---\n")
            f.write(f"   Arestas mais decisivas (importancia GNNExplainer):\n")
            for (src, tgt), imp in s["top5_arestas"]:
                f.write(f"     {src} -> {tgt}  importancia = {imp:.4f}\n")
    print(f"\n   Resumo: {rel_path}")
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
