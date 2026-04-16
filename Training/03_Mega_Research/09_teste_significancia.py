"""
09_teste_significancia.py
--------------------------
Testes de significancia estatistica entre GCN, GAT e GraphSAGE.

Metodologia:
  - Treina cada arquitetura K vezes (default K=10) com seeds distintas
  - Coleta F1, Accuracy, Precision, Recall por execucao
  - Executa t-test pareado (scipy.stats.ttest_rel) entre todos os pares
    de modelos, pois cada run usa a mesma particao treino/teste
  - Reporta: media +/- desvio, t-estatistica, p-valor, nivel de sig.

Dataset padrao: FakeNewsNet (grafos reais, balanceado ~50/50).
  -> Usa os .pt pre-construidos por 00_construir_grafos_fakenewsnet.py.

Saidas em Execution/results/teste_significancia/:
  relatorio.txt       -- tabela completa de metricas e p-valores
  boxplot_f1.png      -- distribuicao F1 por arquitetura (K runs)
  violino_f1.png      -- violin plot + pontos individuais

Uso:
  python 09_teste_significancia.py
  python 09_teste_significancia.py --runs 10 --epochs 50
  python 09_teste_significancia.py --cpu
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from torch_geometric.loader import DataLoader

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR  = RAIZ / "Execution" / "results" / "teste_significancia"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model  import GCNClassifier
from gat_model  import GATClassifier
from sage_model import SAGEClassifier

BATCH_SIZE = 32
PATIENCE   = 10


def criar_modelo(nome: str, num_features: int, seed: int,
                 device: torch.device) -> torch.nn.Module:
    if nome == "GCN":
        return GCNClassifier(num_features, 2, seed=seed).to(device)
    if nome == "GAT":
        return GATClassifier(num_features, 2, seed=seed).to(device)
    if nome == "SAGE":
        return SAGEClassifier(num_features, 2, seed=seed).to(device)
    raise ValueError(f"Arquitetura desconhecida: {nome}")


# ─── Dados ────────────────────────────────────────────────────────────────────

def carregar_fakenewsnet() -> tuple:
    """Retorna (train_list, val_list, test_list, num_features)."""
    splits = {}
    for split in ("train", "val", "test"):
        p = DATA_DIR / f"fakenewsnet_{split}.pt"
        if not p.exists():
            print(f"[ERRO] {p} nao encontrado.")
            print("       Execute: python 00_construir_grafos_fakenewsnet.py")
            sys.exit(1)
        splits[split] = torch.load(p, weights_only=False)

    num_feats = splits["train"][0].x.shape[1]
    total = sum(len(v) for v in splits.values())
    fakes = sum(sum(1 for g in v if g.y.item() == 0) for v in splits.values())
    print(f"  FakeNewsNet: {total} grafos | fake={fakes} ({100*fakes/total:.1f}%) | "
          f"features={num_feats}")
    return splits["train"], splits["val"], splits["test"], num_feats


# ─── Treinamento ──────────────────────────────────────────────────────────────

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
            loss   = criterion(out, data.y.squeeze())
            loss.backward()
            optimizer.step()

        model.eval()
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for data in val_loader:
                data   = data.to(device)
                out, _ = model(data.x, data.edge_index, data.batch)
                preds  = out.argmax(dim=1).cpu().tolist()
                labels = data.y.squeeze().cpu().tolist()
                val_y_pred.extend(preds)
                val_y_true.extend(labels if isinstance(labels, list) else [labels])

        # Criterio de selecao: F1 (consistente com a metrica de avaliacao final)
        val_f1 = f1_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val:
            best_val       = val_f1
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─── Avaliacao ────────────────────────────────────────────────────────────────

@torch.no_grad()
def avaliar(model, test_data, device, pos_label_fake: int = 0) -> dict:
    model.eval()
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    y_true, y_pred = [], []
    for data in loader:
        data   = data.to(device)
        out, _ = model(data.x, data.edge_index, data.batch)
        preds  = out.argmax(dim=1).cpu().tolist()
        labels = data.y.squeeze().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels if isinstance(labels, list) else [labels])
    return {
        "f1":        round(f1_score(y_true, y_pred, pos_label=pos_label_fake, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=pos_label_fake, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    pos_label=pos_label_fake, zero_division=0), 4),
    }


# ─── Testes de significancia ──────────────────────────────────────────────────

def ttest_par(a: list, b: list) -> tuple:
    """t-test pareado. Retorna (t, p, sig_label)."""
    t, p = stats.ttest_rel(a, b)
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"
    return float(t), float(p), sig


# ─── Visualizacoes ────────────────────────────────────────────────────────────

CORES = {"GCN": "#4F86C6", "GAT": "#E07B54", "SAGE": "#5BAD72"}


def plot_boxplot(resultados: dict, metrica: str, out_dir: Path) -> None:
    arqs = list(resultados.keys())
    vals = [resultados[a][metrica] for a in arqs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(vals, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, arq in zip(bp["boxes"], arqs):
        patch.set_facecolor(CORES[arq])
        patch.set_alpha(0.75)

    # Pontos individuais (jitter)
    for i, (arq, v) in enumerate(zip(arqs, vals), start=1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v,
                   color=CORES[arq], alpha=0.7, s=30, zorder=3)

    ax.set_xticks(range(1, len(arqs) + 1))
    ax.set_xticklabels(arqs, fontsize=12)
    ax.set_ylabel(metrica.upper(), fontsize=12)
    ax.set_title(f"Distribuicao de {metrica.upper()} -- {len(vals[0])} runs\n"
                 f"(FakeNewsNet, label 0=Fake)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = out_dir / f"boxplot_{metrica}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {p.name}")


def plot_violino(resultados: dict, metrica: str, out_dir: Path) -> None:
    arqs = list(resultados.keys())
    vals = [resultados[a][metrica] for a in arqs]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(vals, positions=range(1, len(arqs) + 1),
                          showmedians=True, showextrema=True)
    for pc, arq in zip(parts["bodies"], arqs):
        pc.set_facecolor(CORES[arq])
        pc.set_alpha(0.6)

    for i, (arq, v) in enumerate(zip(arqs, vals), start=1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v,
                   color=CORES[arq], alpha=0.85, s=35, zorder=3)

    ax.set_xticks(range(1, len(arqs) + 1))
    ax.set_xticklabels(arqs, fontsize=12)
    ax.set_ylabel(metrica.upper(), fontsize=12)
    ax.set_title(f"Violin Plot -- {metrica.upper()} por Arquitetura\n"
                 f"({len(vals[0])} runs, FakeNewsNet)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = out_dir / f"violino_{metrica}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {p.name}")


# ─── Relatorio ────────────────────────────────────────────────────────────────

def salvar_relatorio(resultados: dict, pares_ttest: dict, runs: int,
                     out_dir: Path) -> None:
    arqs = list(resultados.keys())
    metricas = ["f1", "accuracy", "precision", "recall"]

    linhas = [
        "=" * 72,
        "  TESTES DE SIGNIFICANCIA -- GCN vs GAT vs GraphSAGE",
        "  Dataset: FakeNewsNet (politifact) | label 0=Fake",
        f"  Runs: {runs} | t-test pareado (scipy.stats.ttest_rel)",
        "=" * 72,
        "",
        "  METRICAS POR ARQUITETURA (media +/- std)",
        "",
        f"  {'Metrica':<12} " + " ".join(f"{a:>22}" for a in arqs),
        f"  {'-'*70}",
    ]

    for m in metricas:
        partes = []
        for a in arqs:
            vals = resultados[a][m]
            partes.append(f"{np.mean(vals):6.4f} +/- {np.std(vals):.4f}")
        linhas.append(f"  {m.upper():<12} " + " ".join(f"{p:>22}" for p in partes))

    linhas += [
        "",
        "=" * 72,
        "  T-TEST PAREADO -- F1 (pos_label=0, Fake)",
        "  Hipotese nula: as medias de F1 sao iguais",
        "  Nivel de significancia: * p<0.05  ** p<0.01  *** p<0.001  ns = nao significativo",
        "",
        f"  {'Par':<20} {'Media A':>9} {'Media B':>9} {'t-stat':>9} {'p-valor':>10} {'Sig':>5}",
        f"  {'-'*65}",
    ]

    for (a, b), (t, p, sig) in pares_ttest.items():
        ma = np.mean(resultados[a]["f1"])
        mb = np.mean(resultados[b]["f1"])
        linhas.append(
            f"  {a+' vs '+b:<20} {ma:>9.4f} {mb:>9.4f} {t:>9.4f} {p:>10.6f} {sig:>5}"
        )

    linhas += [
        "",
        "=" * 72,
        "  T-TEST PAREADO -- ACCURACY",
        "",
        f"  {'Par':<20} {'Media A':>9} {'Media B':>9} {'t-stat':>9} {'p-valor':>10} {'Sig':>5}",
        f"  {'-'*65}",
    ]

    for (a, b) in pares_ttest:
        t, p, sig = ttest_par(resultados[a]["accuracy"], resultados[b]["accuracy"])
        ma = np.mean(resultados[a]["accuracy"])
        mb = np.mean(resultados[b]["accuracy"])
        linhas.append(
            f"  {a+' vs '+b:<20} {ma:>9.4f} {mb:>9.4f} {t:>9.4f} {p:>10.6f} {sig:>5}"
        )

    linhas += [
        "",
        "=" * 72,
        "  INTERPRETACAO",
        "=" * 72,
        "",
        "  Resultados estatisticamente significativos (p < 0.05) indicam que",
        "  a diferenca de desempenho nao e fruto de variacao aleatoria.",
        "",
        "  Grafos FakeNewsNet possuem estrutura real de propagacao (media ~88 nos),",
        "  o que permite as arquiteturas explorar a topologia do grafo.",
        "  Diferentemente do Bluesky (grafos vazios, 1 no), aqui a comparacao",
        "  reflete genuinamente as capacidades arquiteturais.",
        "",
        "=" * 72,
    ]

    path = out_dir / "relatorio.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"  [OK] {path.name}")
    print()
    for l in linhas:
        sys.stdout.buffer.write((l + "\n").encode("utf-8", errors="replace"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Testes de significancia entre GCN, GAT e GraphSAGE.")
    parser.add_argument("--runs",   type=int, default=10,
                        help="Numero de execucoes por arquitetura (default: 10)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr",     type=float, default=0.001)
    parser.add_argument("--cpu",    action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  TESTES DE SIGNIFICANCIA -- GCN vs GAT vs GraphSAGE")
    print("=" * 72)
    print(f"  Device: {device} | Runs: {args.runs} | Epochs max: {args.epochs}")
    print()

    # ── Carregar dados ────────────────────────────────────────────────────
    print("[1/3] Carregando FakeNewsNet...")
    train_data, val_data, test_data, num_feats = carregar_fakenewsnet()
    print()

    # ── Executar K runs por arquitetura ───────────────────────────────────
    print(f"[2/3] Treinando {args.runs} runs x 3 arquiteturas "
          f"({args.runs * 3} treinos total)...")

    arqs = ["GCN", "GAT", "SAGE"]
    # resultados[arq][metrica] = lista de K valores
    resultados: dict = {a: {"f1": [], "accuracy": [], "precision": [], "recall": []}
                        for a in arqs}
    base_seed = 42

    for run in range(1, args.runs + 1):
        seed = base_seed + run * 7   # seeds bem espalhadas
        torch.manual_seed(seed)
        np.random.seed(seed)

        for arq in arqs:
            t0    = time.time()
            model = criar_modelo(arq, num_feats, seed, device)
            model = treinar_uma_vez(model, train_data, val_data,
                                    device, args.epochs, args.lr)
            res   = avaliar(model, test_data, device, pos_label_fake=0)
            elapsed = time.time() - t0

            for m in ("f1", "accuracy", "precision", "recall"):
                resultados[arq][m].append(res[m])

            print(f"  Run {run:02d}/{args.runs} | {arq:<5} | "
                  f"F1={res['f1']:.4f} | Acc={res['accuracy']:.4f} | "
                  f"{elapsed:.1f}s")

        print()

    # ── Testes t pareados ─────────────────────────────────────────────────
    print("[3/3] Calculando testes de significancia...")
    pares = [("GCN", "GAT"), ("GCN", "SAGE"), ("GAT", "SAGE")]
    pares_ttest = {}
    for a, b in pares:
        t, p, sig = ttest_par(resultados[a]["f1"], resultados[b]["f1"])
        pares_ttest[(a, b)] = (t, p, sig)
        print(f"  F1 {a} vs {b}: t={t:.4f}, p={p:.6f} {sig}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n[Plots] Salvando em {OUT_DIR}...")
    plot_boxplot(resultados, "f1", OUT_DIR)
    plot_violino(resultados, "f1", OUT_DIR)
    plot_boxplot(resultados, "accuracy", OUT_DIR)

    salvar_relatorio(resultados, pares_ttest, args.runs, OUT_DIR)

    print(f"\n[OK] Testes de significancia concluidos!")
    print(f"     Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
