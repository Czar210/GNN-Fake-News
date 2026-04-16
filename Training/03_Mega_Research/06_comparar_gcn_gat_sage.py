"""
06_comparar_gcn_gat_sage.py
---------------------------
Benchmark completo: treina GCN, GAT e GraphSAGE nas MESMAS condições e compara.

Extensão de 04_comparar_gcn_gat.py com GraphSAGE como terceiro modelo.

Saídas em Execution/results/comparativo_gcn_gat_sage/:
  matrizes_confusao.png   — 3 matrizes lado a lado (GCN | GAT | GraphSAGE)
  metricas_barras.png     — barras agrupadas (3 modelos × 4 métricas + tempo/época)
  relatorio.txt           — tabela completa com Δ vs GCN e ranking final

Uso:
  python 06_comparar_gcn_gat_sage.py
  python 06_comparar_gcn_gat_sage.py --epochs 30 --lr 0.005
  python 06_comparar_gcn_gat_sage.py --usar-pesos-salvos
  python 06_comparar_gcn_gat_sage.py --aggr max
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from torch_geometric.loader import DataLoader

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ        = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = Path(__file__).resolve().parent / "data"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
OUT_DIR     = RAIZ / "Execution" / "results" / "comparativo_gcn_gat_sage"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model  import GCNClassifier
from gat_model  import GATClassifier
from sage_model import SAGEClassifier


# ─── Carregamento de dados ────────────────────────────────────────────────────

BATCH_SIZE = 64

def carregar_dados(_device: torch.device) -> tuple:
    for split in ["train", "val", "test"]:
        p = DATA_DIR / f"grafos_bluesky_{split}.pt"
        if not p.exists():
            print(f"[ERRO] {p} não encontrado.")
            print("       Execute: python 02_construir_grafos_bluesky.py")
            sys.exit(1)

    train = torch.load(DATA_DIR / "grafos_bluesky_train.pt", weights_only=False)
    val   = torch.load(DATA_DIR / "grafos_bluesky_val.pt",   weights_only=False)
    test  = torch.load(DATA_DIR / "grafos_bluesky_test.pt",  weights_only=False)

    tr_loader  = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} grafos")

    return tr_loader, val_loader, test


# ─── Loop de treino unificado ─────────────────────────────────────────────────

def treinar(model, train_loader, val_loader, device, epochs, lr) -> tuple:
    """
    Retorna (model_treinado, historico_loss, historico_val_acc, tempo_medio_por_epoca).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    hist_loss, hist_val, tempos = [], [], []
    best_val   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.batch)
            loss   = criterion(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        ep_time = time.time() - t0
        tempos.append(ep_time)

        # Validação
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                data   = data.to(device)
                out, _ = model(data.x, data.edge_index, data.batch)
                preds  = out.argmax(dim=1)
                correct += int((preds == data.y.squeeze()).sum())
                total   += data.num_graphs

        val_acc  = correct / total if total > 0 else 0.0
        avg_loss = total_loss / sum(d.num_graphs for d in train_loader)
        scheduler.step(val_acc)

        hist_loss.append(avg_loss)
        hist_val.append(val_acc)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Val: {val_acc:.4f} | {ep_time:.2f}s/ep")

    if best_state:
        model.load_state_dict(best_state)

    return model, hist_loss, hist_val, float(np.mean(tempos))


# ─── Avaliação detalhada ──────────────────────────────────────────────────────

@torch.no_grad()
def avaliar_completo(model, test_data, device) -> dict:
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
        "y_true":    y_true,
        "y_pred":    y_pred,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ─── Visualizações ────────────────────────────────────────────────────────────

CORES = {
    "GCN":  "#4F86C6",   # azul
    "GAT":  "#E07B54",   # laranja
    "SAGE": "#5BAD72",   # verde
}


def plot_matrizes(resultados: dict, out_dir: Path) -> None:
    """Gera figura com as 3 matrizes de confusão lado a lado."""
    nomes   = list(resultados.keys())
    titulos = {"GCN": "GCN (Baseline)", "GAT": "GAT (4 heads)", "SAGE": "GraphSAGE (mean)"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Matrizes de Confusão — GCN vs GAT vs GraphSAGE (Dataset Bluesky)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, nome in zip(axes, nomes):
        res = resultados[nome]
        cm  = np.array(res["confusion"])
        im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set(
            title=titulos.get(nome, nome),
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            xlabel="Predição", ylabel="Realidade",
        )
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = out_dir / "matrizes_confusao.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def plot_metricas(resultados: dict, tempos: dict, _params: dict, out_dir: Path) -> None:
    """Gera barras agrupadas comparando as 3 arquiteturas."""
    metricas  = ["accuracy", "precision", "recall", "f1"]
    labels_m  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    nomes     = list(resultados.keys())

    x     = np.arange(len(metricas))
    width = 0.25
    offsets = [-width, 0, width]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Comparativo GCN vs GAT vs GraphSAGE — Dataset Bluesky",
                 fontsize=13, fontweight="bold")

    # ── Barras de métricas ────────────────────────────────────────────────
    ax = axes[0]
    for nome, offset in zip(nomes, offsets):
        vals = [resultados[nome][m] for m in metricas]
        bars = ax.bar(x + offset, vals, width,
                      label=nome, color=CORES[nome], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set(title="Métricas no Conjunto de Teste",
           xticks=x, xticklabels=labels_m,
           ylim=(0, 1.15), ylabel="Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Tempo por época ───────────────────────────────────────────────────
    ax2 = axes[1]
    max_t = max(tempos.values()) if tempos else 1.0
    for i, nome in enumerate(nomes):
        t = tempos.get(nome, 0.0)
        ax2.bar(nome, t, color=CORES[nome], alpha=0.88, width=0.5)
        ax2.text(i, t + max_t * 0.02, f"{t:.2f}s", ha="center", fontsize=11)

    ax2.set(title="Tempo Médio por Época (segundos)",
            ylabel="Segundos",
            ylim=(0, max_t * 1.3 if max_t > 0 else 1))
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = out_dir / "metricas_barras.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def salvar_relatorio(
    resultados: dict,
    tempos: dict,
    params: dict,
    epochs: int,
    out_dir: Path,
) -> None:
    """Gera relatório .txt com tabela completa e conclusão."""
    # Determina ranking por F1
    ranking = sorted(resultados.keys(), key=lambda n: resultados[n]["f1"], reverse=True)
    vencedor = ranking[0]



    linhas = [
        "=" * 65,
        "  RELATÓRIO COMPARATIVO — GCN vs GAT vs GraphSAGE",
        "=" * 65,
        f"  Dataset:  Bluesky (HuggingFace: Zaras210/bluesky-fake-news-dataset)",
        f"  Épocas:   {epochs}  |  Batch: {BATCH_SIZE}",
        "",
        f"  {'Metrica':<16} {'GCN':>8} {'GAT':>8} {'SAGE':>8} {'D(SAGE-GCN)':>12}",
        f"  {'-'*56}",
    ]

    for m in ["accuracy", "precision", "recall", "f1"]:
        r_gcn  = resultados.get("GCN",  {}).get(m, 0.0)
        r_gat  = resultados.get("GAT",  {}).get(m, 0.0)
        r_sage = resultados.get("SAGE", {}).get(m, 0.0)
        delta  = r_sage - r_gcn
        sinal  = "+" if delta >= 0 else ""
        linhas.append(
            f"  {m.capitalize():<16} {r_gcn:>8.4f} {r_gat:>8.4f} "
            f"{r_sage:>8.4f} {sinal}{delta:>10.4f}"
        )

    linhas += [
        "",
        f"  {'Parâmetros':<16} "
        f"{params.get('GCN', 0):>8,} {params.get('GAT', 0):>8,} {params.get('SAGE', 0):>8,}",
        f"  {'Tempo/época':<16} "
        f"{tempos.get('GCN', 0):>7.2f}s {tempos.get('GAT', 0):>7.2f}s {tempos.get('SAGE', 0):>7.2f}s",
        "",
        f"  -> Ranking em F1: {' > '.join(ranking)}",
        "",
        "=" * 65,
        "  CONCLUSÃO PARA A MONOGRAFIA",
        "=" * 65,
    ]

    r_sage = resultados.get("SAGE", {})
    r_gcn_m = resultados.get("GCN", {})
    delta_f1 = r_sage.get("f1", 0.0) - r_gcn_m.get("f1", 0.0)

    if vencedor == "SAGE":
        sinal = f"+{delta_f1:.4f}" if delta_f1 >= 0 else f"{delta_f1:.4f}"
        linhas.append(f"  GraphSAGE superou o GCN em F1-Score ({sinal}).")
        linhas.append("  A separação entre features do nó raiz (W_l) e dos vizinhos")
        linhas.append("  (W_r) no SAGEConv trouxe ganho de discriminação.")
        p_sage = params.get("SAGE", 0)
        p_gcn  = params.get("GCN",  1)
        linhas.append(f"  Custo: {p_sage/p_gcn:.1f}x mais parâmetros que o GCN.")
    elif vencedor == "GCN":
        linhas.append(f"  GCN mantém a melhor F1 entre os três modelos.")
        linhas.append("  Grafos pequenos/rasos (estrela, ≤50 nós) favorecem")
        linhas.append("  agregação simples normalizada por grau (GCN).")
        linhas.append("  GraphSAGE e GAT adicionam complexidade sem ganho claro.")
    else:
        linhas.append(f"  GAT obteve melhor F1. Resultado inesperado dado o")
        linhas.append("  experimento anterior — verifique variação de inicialização.")

    linhas += [
        "",
        "  Nota: GAT (F1=0.0000 no experimento anterior) falhou por colapso",
        "  de atenção em grafos homogêneos/rasos. GraphSAGE evita esse",
        "  problema por usar matrizes W_l e W_r fixas (sem softmax).",
        "=" * 65,
    ]

    path = out_dir / "relatorio.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    print(f"  [OK] {path.name}")
    print()
    import sys
    for l in linhas:
        sys.stdout.buffer.write((l + "\n").encode("utf-8", errors="replace"))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GCN vs GAT vs GraphSAGE.")
    parser.add_argument("--epochs",            type=int,   default=30)
    parser.add_argument("--lr",                type=float, default=0.005)
    parser.add_argument("--aggr",              type=str,   default="mean",
                        choices=["mean", "max", "lstm"],
                        help="Agregador do GraphSAGE (padrão: mean).")
    parser.add_argument("--usar-pesos-salvos", action="store_true",
                        help="Carrega pesos já treinados em vez de re-treinar.")
    parser.add_argument("--cpu",               action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 65)
    print("  Etapa 6: Benchmark Triplo GCN vs GAT vs GraphSAGE")
    print("=" * 65)
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Épocas:     {args.epochs}  |  LR: {args.lr}")
    print(f"  Agregador SAGE: {args.aggr}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Carregando dados...")
    train_loader, val_loader, test_data = carregar_dados(device)

    gcn  = GCNClassifier(768, 2).to(device)
    gat  = GATClassifier(768, 2).to(device)
    sage = SAGEClassifier(768, 2, aggr=args.aggr).to(device)

    params = {
        "GCN":  gcn.count_parameters(),
        "GAT":  gat.count_parameters(),
        "SAGE": sage.count_parameters(),
    }
    print(f"  GCN  parâmetros: {params['GCN']:,}")
    print(f"  GAT  parâmetros: {params['GAT']:,}")
    print(f"  SAGE parâmetros: {params['SAGE']:,}")

    tempos = {"GCN": 0.0, "GAT": 0.0, "SAGE": 0.0}

    if args.usar_pesos_salvos:
        # ── Carrega pesos já treinados ────────────────────────────────────
        print("\n[2/4] Carregando pesos salvos...")
        cargas = [
            (WEIGHTS_DIR / "pesos_gcn.pth",  gcn,  "GCN"),
            (WEIGHTS_DIR / "pesos_gat.pth",  gat,  "GAT"),
            (WEIGHTS_DIR / "pesos_sage.pth", sage, "SAGE"),
        ]
        for p, model, nome in cargas:
            if p.exists():
                model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                print(f"  [OK] {nome}: {p.name}")
            else:
                print(f"  [WARN] {p.name} não encontrado — treinando {nome} do zero.")
                _, _, _, t = treinar(model, train_loader, val_loader, device, args.epochs, args.lr)
                tempos[nome] = t
                torch.save(model.state_dict(), p)
                print(f"  [OK] Pesos salvos -> {p}")
    else:
        # ── Treina do zero nas mesmas condições ───────────────────────────
        print(f"\n[2/4] Treinando GCN ({args.epochs} épocas)...")
        gcn, _, _, tempos["GCN"] = treinar(gcn, train_loader, val_loader,
                                           device, args.epochs, args.lr)

        print(f"\n[3/4] Treinando GAT ({args.epochs} épocas)...")
        gat, _, _, tempos["GAT"] = treinar(gat, train_loader, val_loader,
                                           device, args.epochs, args.lr)

        print(f"\n[4/4] Treinando GraphSAGE ({args.epochs} épocas)...")
        sage, _, _, tempos["SAGE"] = treinar(sage, train_loader, val_loader,
                                             device, args.epochs, args.lr)

    # ── Avaliação ─────────────────────────────────────────────────────────
    print("\n[Avaliação] Calculando métricas no conjunto de teste...")
    resultados = {
        "GCN":  avaliar_completo(gcn,  test_data, device),
        "GAT":  avaliar_completo(gat,  test_data, device),
        "SAGE": avaliar_completo(sage, test_data, device),
    }

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n[Plots] Salvando visualizações em {OUT_DIR}...")
    plot_matrizes(resultados, OUT_DIR)
    plot_metricas(resultados, tempos, params, OUT_DIR)
    salvar_relatorio(resultados, tempos, params, args.epochs, OUT_DIR)

    print(f"\n[OK] Benchmark concluído!")
    print(f"     Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
