"""
04_comparar_gcn_gat.py
----------------------
Benchmark completo: treina GCN e GAT nas MESMAS condições e compara.

Saídas em Execution/results/comparativo_gcn_gat/:
  matrizes_confusao.png   — matrizes lado a lado (GCN | GAT)
  metricas_barras.png     — barras agrupadas (Acc / Prec / Recall / F1)
  relatorio.txt           — tabela completa + tempo de treino

Uso:
  python 04_comparar_gcn_gat.py
  python 04_comparar_gcn_gat.py --epochs 25 --usar-pesos-salvos
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
OUT_DIR     = RAIZ / "Execution" / "results" / "comparativo_gcn_gat"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat_model import GATClassifier
from gcn_model import GCNClassifier


# ─── Carregamento de dados ────────────────────────────────────────────────────

BATCH_SIZE = 64

def carregar_dados(device: torch.device) -> tuple:
    for split in ["train", "val", "test"]:
        p = DATA_DIR / f"grafos_bluesky_{split}.pt"
        if not p.exists():
            print(f"[ERRO] {p} não encontrado.")
            print("       Execute: python 02_construir_grafos_bluesky.py")
            sys.exit(1)

    train = torch.load(DATA_DIR / "grafos_bluesky_train.pt", weights_only=False)
    val   = torch.load(DATA_DIR / "grafos_bluesky_val.pt",   weights_only=False)
    test  = torch.load(DATA_DIR / "grafos_bluesky_test.pt",  weights_only=False)

    tr_loader  = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0)
    val_loader = DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return tr_loader, val_loader, test


# ─── Loop de treino unificado ─────────────────────────────────────────────────

def treinar(model, train_loader, val_loader, device, epochs, lr) -> tuple:
    """
    Retorna (model_treinado, historico_loss, historico_val_acc, tempo_por_epoca).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    hist_loss, hist_val, tempos = [], [], []
    best_val = 0.0
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
                data  = data.to(device)
                out, _ = model(data.x, data.edge_index, data.batch)
                preds  = out.argmax(dim=1)
                correct += int((preds == data.y.squeeze()).sum())
                total   += data.num_graphs

        val_acc = correct / total if total > 0 else 0.0
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
        data  = data.to(device)
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

def plot_matrizes(res_gcn: dict, res_gat: dict, out_dir: Path) -> None:
    """Gera figura com as duas matrizes de confusão lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrizes de Confusao - GCN vs GAT (Dataset Bluesky)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, res, titulo in zip(axes, [res_gcn, res_gat], ["GCN (Baseline)", "GAT (4 heads)"]):
        cm = np.array(res["confusion"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set(
            title=titulo,
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
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


def plot_metricas(res_gcn: dict, res_gat: dict,
                  tempo_gcn: float, tempo_gat: float, out_dir: Path) -> None:
    """Gera barras agrupadas comparando as métricas."""
    metricas = ["accuracy", "precision", "recall", "f1"]
    labels_m = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals_gcn = [res_gcn[m] for m in metricas]
    vals_gat = [res_gat[m] for m in metricas]

    x     = np.arange(len(metricas))
    width = 0.35
    cor_gcn = "#4F86C6"
    cor_gat = "#E07B54"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparativo GCN vs GAT - Dataset Bluesky",
                 fontsize=14, fontweight="bold")

    # -- Barras de metricas ------------------------------------------------
    ax = axes[0]
    bars_gcn = ax.bar(x - width/2, vals_gcn, width, label="GCN", color=cor_gcn, alpha=0.85)
    bars_gat = ax.bar(x + width/2, vals_gat, width, label="GAT", color=cor_gat, alpha=0.85)

    for bars, vals in [(bars_gcn, vals_gcn), (bars_gat, vals_gat)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set(title="Métricas no Conjunto de Teste",
           xticks=x, xticklabels=labels_m,
           ylim=(0, 1.12), ylabel="Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # -- Tempo por epoca ---------------------------------------------------
    ax2 = axes[1]
    ax2.bar(["GCN", "GAT"], [tempo_gcn, tempo_gat],
            color=[cor_gcn, cor_gat], alpha=0.85, width=0.5)
    ax2.set(title="Tempo Médio por Época (segundos)",
            ylabel="Segundos", ylim=(0, max(tempo_gcn, tempo_gat) * 1.3))
    for i, (modelo, t) in enumerate([("GCN", tempo_gcn), ("GAT", tempo_gat)]):
        ax2.text(i, t + max(tempo_gcn, tempo_gat) * 0.02,
                 f"{t:.2f}s", ha="center", fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = out_dir / "metricas_barras.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def salvar_relatorio(res_gcn, res_gat, tempo_gcn, tempo_gat,
                     params_gcn, params_gat, epochs, out_dir) -> None:
    vencedor = "GAT" if res_gat["f1"] >= res_gcn["f1"] else "GCN"
    delta_f1 = abs(res_gat["f1"] - res_gcn["f1"])

    linhas = [
        "=" * 58,
        "  RELATORIO COMPARATIVO - GCN vs GAT",
        "=" * 58,
        f"  Dataset:  Bluesky (HuggingFace: Zaras210/bluesky-fake-news-dataset)",
        f"  Épocas:   {epochs}  |  Batch: {BATCH_SIZE}",
        "",
        f"  {'Métrica':<16} {'GCN':>8} {'GAT':>8} {'Δ':>8}",
        f"  {'-'*44}",
    ]
    for m in ["accuracy", "precision", "recall", "f1"]:
        d = res_gat[m] - res_gcn[m]
        sinal = "+" if d >= 0 else ""
        linhas.append(f"  {m.capitalize():<16} {res_gcn[m]:>8.4f} {res_gat[m]:>8.4f} {sinal}{d:>7.4f}")

    linhas += [
        "",
        f"  {'Parâmetros':<16} {params_gcn:>8,} {params_gat:>8,}",
        f"  {'Tempo/época':<16} {tempo_gcn:>7.2f}s {tempo_gat:>7.2f}s",
        "",
        f"  → Vencedor em F1: {vencedor} (margem: {delta_f1:.4f})",
        "",
        "=" * 58,
        "  CONCLUSÃO PARA A MONOGRAFIA",
        "=" * 58,
    ]

    if res_gat["f1"] > res_gcn["f1"]:
        linhas.append(f"  O GAT superou o GCN em F1-Score (+{delta_f1:.4f}).")
        linhas.append("  O mecanismo de atenção diferenciou efetivamente nós")
        linhas.append("  influentes na propagação.")
        if tempo_gat > tempo_gcn * 1.5:
            linhas.append(f"  Custo: {tempo_gat/tempo_gcn:.1f}x mais lento por época.")
    elif res_gat["f1"] == res_gcn["f1"]:
        linhas.append("  Empate em F1. GAT não trouxe ganho claro neste dataset.")
    else:
        linhas.append(f"  GCN superou o GAT (margem: {delta_f1:.4f}).")
        linhas.append("  Possível causa: grafos pequenos/rasos não se beneficiam")
        linhas.append("  tanto do mecanismo de atenção.")

    linhas.append("=" * 58)

    path = out_dir / "relatorio.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    print(f"  [OK] {path.name}")
    print()
    for l in linhas:
        print(l)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GCN vs GAT.")
    parser.add_argument("--epochs",           type=int, default=30)
    parser.add_argument("--lr",               type=float, default=0.005)
    parser.add_argument("--usar-pesos-salvos", action="store_true",
                        help="Carrega pesos já treinados em vez de re-treinar.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 58)
    print("  GAT - Etapa 5: Benchmark Comparativo GCN x GAT")
    print("=" * 58)
    print(f"  Device:  {device}")
    if device.type == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"  Épocas: {args.epochs}  |  LR: {args.lr}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Carregando dados...")
    train_loader, val_loader, test_data = carregar_dados(device)

    gcn = GCNClassifier(768, 2).to(device)
    gat = GATClassifier(768, 2).to(device)
    print(f"  GCN parâmetros: {gcn.count_parameters():,}")
    print(f"  GAT parâmetros: {gat.count_parameters():,}")

    if args.usar_pesos_salvos:
        # ── Carrega pesos já treinados ────────────────────────────────────
        print("\n[2/4] Carregando pesos salvos...")
        p_gcn = WEIGHTS_DIR / "pesos_gcn.pth"
        p_gat = WEIGHTS_DIR / "pesos_gat.pth"
        for p, model, nome in [(p_gcn, gcn, "GCN"), (p_gat, gat, "GAT")]:
            if p.exists():
                model.load_state_dict(torch.load(p, map_location=device))
                print(f"  [OK] {nome}: {p}")
            else:
                print(f"  [WARN] {p} não encontrado - treinando do zero.")
        tempo_gcn = tempo_gat = 0.0
    else:
        # ── Treina do zero nas mesmas condições ───────────────────────────
        print(f"\n[2/4] Treinando GCN ({args.epochs} épocas)...")
        gcn, _, _, tempo_gcn = treinar(gcn, train_loader, val_loader, device, args.epochs, args.lr)

        print(f"\n[3/4] Treinando GAT ({args.epochs} épocas)...")
        gat, _, _, tempo_gat = treinar(gat, train_loader, val_loader, device, args.epochs, args.lr)

    # ── Avaliação ─────────────────────────────────────────────────────────
    print("\n[4/4] Avaliando nos dados de teste e gerando relatório...")
    res_gcn = avaliar_completo(gcn, test_data, device)
    res_gat = avaliar_completo(gat, test_data, device)

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"  Salvando gráficos em {OUT_DIR}...")
    plot_matrizes(res_gcn, res_gat, OUT_DIR)
    plot_metricas(res_gcn, res_gat, tempo_gcn, tempo_gat, OUT_DIR)
    salvar_relatorio(
        res_gcn, res_gat, tempo_gcn, tempo_gat,
        gcn.count_parameters(), gat.count_parameters(),
        args.epochs, OUT_DIR,
    )

    print(f"\n[OK] Benchmark concluido!")
    print(f"   Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
