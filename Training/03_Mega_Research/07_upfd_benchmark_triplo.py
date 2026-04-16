"""
07_upfd_benchmark_triplo.py
---------------------------
Benchmark triplo (GCN / GAT / GraphSAGE) no dataset UPFD PolitiFact.

O UPFD (User Preference-aware Fake News Detection) contém grafos de
propagacao reais do Twitter, com arvores de retweet de dezenas a centenas
de nos -- estrutura que o dataset Bluesky NAO tem (ver analise_benchmarks_gnn.md).

Esta comparacao e academicamente valida porque os modelos processam grafos
reais, revelando diferencas arquiteturais que os dados Bluesky nao podem
mostrar.

Datasets suportados: politifact | gossipcop
Features  suportadas: bert | spacy | content | profile

Saidas em Execution/results/upfd_benchmark_{dataset}/:
  matrizes_confusao.png
  metricas_barras.png
  curvas_treino.png
  relatorio.txt
  pesos_gcn_upfd_{dataset}.pth        (salvo em Execution/weights/)
  pesos_gat_upfd_{dataset}.pth
  pesos_sage_upfd_{dataset}.pth

Uso:
  python 07_upfd_benchmark_triplo.py
  python 07_upfd_benchmark_triplo.py --dataset gossipcop
  python 07_upfd_benchmark_triplo.py --epochs 50 --lr 0.001
  python 07_upfd_benchmark_triplo.py --usar-pesos-salvos
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
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ        = Path(__file__).resolve().parent.parent.parent
UPFD_ROOT   = RAIZ / "Material" / "upfd_data"           # cache do download
WEIGHTS_DIR = RAIZ / "Execution" / "weights"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model  import GCNClassifier
from gat_model  import GATClassifier
from sage_model import SAGEClassifier


# ─── Carregamento UPFD ───────────────────────────────────────────────────────

BATCH_SIZE = 32   # UPFD/FakeNewsNet tem poucos grafos — batch menor


def _inspecionar_dataset(nome, grafos):
    nos = [g.num_nodes for g in grafos]
    ar  = [g.num_edges for g in grafos]
    fk  = sum(1 for g in grafos if g.y.item() == 0)
    print(f"  {nome}: {len(grafos):,} grafos | "
          f"nos_med={sum(nos)/len(nos):.1f} | arestas_med={sum(ar)/len(ar):.1f} | "
          f"fake(0)={fk} ({100*fk/len(grafos):.1f}%)")


def carregar_fakenewsnet(_device: torch.device) -> tuple:
    """
    Carrega grafos FakeNewsNet pre-construidos pelo script 00_construir_grafos_fakenewsnet.py.
    Retorna (train_loader, val_loader, test_data, num_node_features).
    """
    DATA_DIR = Path(__file__).resolve().parent / "data"
    splits = {}
    for split in ["train", "val", "test"]:
        p = DATA_DIR / f"fakenewsnet_{split}.pt"
        if not p.exists():
            print(f"[ERRO] {p} nao encontrado.")
            print("       Execute: python 00_construir_grafos_fakenewsnet.py")
            import sys; sys.exit(1)
        splits[split] = torch.load(p, weights_only=False)
        _inspecionar_dataset(split, splits[split])

    tr_loader  = DataLoader(splits["train"], batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(splits["val"],   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    num_feats  = splits["train"][0].x.shape[1]
    return tr_loader, val_loader, splits["test"], num_feats


def carregar_upfd(dataset_name: str, feature: str, _device: torch.device) -> tuple:
    """
    Carrega UPFD via PyG. Se falhar (GDrive expirado), sugere alternativa.
    Retorna (train_loader, val_loader, test_data, num_node_features).
    """
    print(f"  Baixando/carregando UPFD '{dataset_name}' (feature='{feature}')...")
    UPFD_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        train_ds = UPFD(root=str(UPFD_ROOT), name=dataset_name, feature=feature, split="train")
        val_ds   = UPFD(root=str(UPFD_ROOT), name=dataset_name, feature=feature, split="val")
        test_ds  = UPFD(root=str(UPFD_ROOT), name=dataset_name, feature=feature, split="test")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar UPFD: {e}")
        print("       Os links do Google Drive expiraram.")
        print("       Use --dataset fakenewsnet (construido via 00_construir_grafos_fakenewsnet.py)")
        import sys; sys.exit(1)

    for split_name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        _inspecionar_dataset(split_name, list(ds))

    tr_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr_loader, val_loader, list(test_ds), train_ds.num_features


# ─── Loop de treino unificado ─────────────────────────────────────────────────

def treinar(model, train_loader, val_loader, device, epochs, lr) -> tuple:
    """
    Retorna (model, hist_loss, hist_val_acc, tempo_medio_por_epoca).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    hist_loss, hist_val, tempos = [], [], []
    best_val   = 0.0
    best_state = None
    patience_count = 0

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

        # Validacao
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
        val_f1   = f1_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
        avg_loss = total_loss / sum(d.num_graphs for d in train_loader)
        scheduler.step(val_f1)

        hist_loss.append(avg_loss)
        hist_val.append(val_f1)

        if val_f1 > best_val:
            best_val       = val_f1
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | {ep_time:.2f}s/ep")

        if patience_count >= 10:
            print(f"    [Early stopping] 10 epocas sem melhoria.")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, hist_loss, hist_val, float(np.mean(tempos))


# ─── Avaliacao detalhada ──────────────────────────────────────────────────────

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

    # No UPFD: label 0 = Fake, label 1 = Real
    # Para F1, usamos pos_label=0 (Fake) para medir deteccao de fake news
    return {
        "y_true":    y_true,
        "y_pred":    y_pred,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    pos_label=0, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        pos_label=0, zero_division=0), 4),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ─── Visualizacoes ────────────────────────────────────────────────────────────

CORES = {"GCN": "#4F86C6", "GAT": "#E07B54", "SAGE": "#5BAD72"}


def plot_matrizes(resultados: dict, dataset_name: str, out_dir: Path) -> None:
    nomes   = list(resultados.keys())
    titulos = {
        "GCN":  "GCN (Baseline)",
        "GAT":  "GAT (4 heads)",
        "SAGE": "GraphSAGE (mean)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Matrizes de Confusao -- GCN vs GAT vs GraphSAGE ({dataset_name.upper()})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nome in zip(axes, nomes):
        res = resultados[nome]
        cm  = np.array(res["confusion"])
        im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set(
            title=titulos.get(nome, nome),
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            xlabel="Predicao", ylabel="Realidade",
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


def plot_metricas(resultados: dict, tempos: dict, dataset_name: str, out_dir: Path) -> None:
    metricas = ["accuracy", "precision", "recall", "f1"]
    labels_m = ["Accuracy", "Precision", "Recall", "F1-Score"]
    nomes    = list(resultados.keys())

    x     = np.arange(len(metricas))
    width = 0.25
    offsets = [-width, 0, width]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Comparativo GCN vs GAT vs GraphSAGE -- {dataset_name.upper()} (UPFD)",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    for nome, offset in zip(nomes, offsets):
        vals = [resultados[nome][m] for m in metricas]
        bars = ax.bar(x + offset, vals, width, label=nome, color=CORES[nome], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set(title="Metricas no Conjunto de Teste (label 0 = Fake)",
           xticks=x, xticklabels=labels_m,
           ylim=(0, 1.15), ylabel="Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax2 = axes[1]
    max_t = max(tempos.values()) if any(tempos.values()) else 1.0
    for i, nome in enumerate(nomes):
        t = tempos.get(nome, 0.0)
        ax2.bar(nome, t, color=CORES[nome], alpha=0.88, width=0.5)
        if t > 0:
            ax2.text(i, t + max_t * 0.02, f"{t:.2f}s", ha="center", fontsize=11)

    ax2.set(title="Tempo Medio por Epoca (segundos)", ylabel="Segundos",
            ylim=(0, max(max_t * 1.3, 1)))
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = out_dir / "metricas_barras.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def plot_curvas(historicos: dict, out_dir: Path) -> None:
    """Plota curvas de treinamento (loss e val_acc) para os 3 modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Treinamento -- UPFD", fontsize=13, fontweight="bold")

    for nome, hist in historicos.items():
        epochs = range(1, len(hist["loss"]) + 1)
        axes[0].plot(epochs, hist["loss"], label=nome, color=CORES[nome])
        axes[1].plot(epochs, hist["val"],  label=nome, color=CORES[nome])

    axes[0].set(title="Loss de Treino", xlabel="Epoca", ylabel="CrossEntropyLoss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set(title="Acuracia de Validacao", xlabel="Epoca", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "curvas_treino.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def salvar_relatorio(
    resultados: dict,
    tempos: dict,
    params: dict,
    epochs: int,
    dataset_name: str,
    out_dir: Path,
) -> None:
    ranking  = sorted(resultados.keys(), key=lambda n: resultados[n]["f1"], reverse=True)
    vencedor = ranking[0]

    r_gcn  = resultados.get("GCN",  {})
    r_sage = resultados.get("SAGE", {})

    linhas = [
        "=" * 70,
        f"  RELATORIO COMPARATIVO -- GCN vs GAT vs GraphSAGE ({dataset_name.upper()})",
        "=" * 70,
        f"  Dataset:    UPFD {dataset_name} | Feature: BERT (768-dim)",
        f"  Epocas max: {epochs}  |  Batch: {BATCH_SIZE}",
        f"  Nota: label 0 = Fake, label 1 = Real (convencao UPFD)",
        "",
        f"  {'Metrica':<16} {'GCN':>8} {'GAT':>8} {'SAGE':>8} {'D(SAGE-GCN)':>12}",
        f"  {'-'*58}",
    ]

    for m in ["accuracy", "precision", "recall", "f1"]:
        r_g  = resultados.get("GCN",  {}).get(m, 0.0)
        r_a  = resultados.get("GAT",  {}).get(m, 0.0)
        r_s  = resultados.get("SAGE", {}).get(m, 0.0)
        d    = r_s - r_g
        sinal = "+" if d >= 0 else ""
        linhas.append(
            f"  {m.capitalize():<16} {r_g:>8.4f} {r_a:>8.4f} "
            f"{r_s:>8.4f} {sinal}{d:>10.4f}"
        )

    linhas += [
        "",
        f"  {'Parametros':<16} "
        f"{params.get('GCN', 0):>8,} {params.get('GAT', 0):>8,} {params.get('SAGE', 0):>8,}",
        f"  {'Tempo/epoca':<16} "
        f"{tempos.get('GCN', 0):>7.2f}s {tempos.get('GAT', 0):>7.2f}s {tempos.get('SAGE', 0):>7.2f}s",
        "",
        f"  -> Ranking em F1: {' > '.join(ranking)}",
        "",
        "=" * 70,
        "  CONCLUSAO",
        "=" * 70,
    ]

    f1_vals = {n: resultados[n]["f1"] for n in resultados}
    f1_spread = max(f1_vals.values()) - min(f1_vals.values())

    linhas.append(f"  Modelo com maior F1 nesta execucao: {vencedor}")
    linhas.append(f"  Amplitude das diferencas de F1 entre modelos: {f1_spread:.4f}")
    linhas.append("")
    linhas.append("  ATENCAO: este relatorio descreve UMA execucao com seed fixo.")
    linhas.append("  Diferencas pequenas (< ~0.02) podem ser variancia de inicializacao.")
    linhas.append("  Execute 09_teste_significancia.py para testes com multiplas runs.")
    linhas.append("  AVISO: se os grafos forem construidos com features nodais identicos")
    linhas.append("  (todos os nos com o mesmo embedding do artigo), as diferencas entre")
    linhas.append("  arquiteturas refletem apenas a transformacao linear sobre BERT,")
    linhas.append("  nao capacidades distintas de agregacao de grafo.")

    linhas += [
        "",
        "  Comparacao com Bluesky (grafos vazios):",
        f"    GCN  Bluesky F1=0.4128 | UPFD F1={r_gcn.get('f1', 0.0):.4f}",
        f"    GAT  Bluesky F1=0.0000 | UPFD F1={resultados.get('GAT', {}).get('f1', 0.0):.4f}",
        f"    SAGE Bluesky F1=0.2489 | UPFD F1={r_sage.get('f1', 0.0):.4f}",
        "  (Bluesky: todos os grafos tinham 1 no e 0 arestas -- ver analise_benchmarks_gnn.md)",
        "=" * 70,
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
    parser = argparse.ArgumentParser(description="Benchmark GCN vs GAT vs SAGE no UPFD.")
    parser.add_argument("--dataset",           type=str,   default="fakenewsnet",
                        choices=["politifact", "gossipcop", "fakenewsnet"])
    parser.add_argument("--feature",           type=str,   default="bert",
                        choices=["bert", "spacy", "content", "profile"])
    parser.add_argument("--epochs",            type=int,   default=50)
    parser.add_argument("--lr",                type=float, default=0.001)
    parser.add_argument("--usar-pesos-salvos", action="store_true")
    parser.add_argument("--cpu",               action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    OUT_DIR = RAIZ / "Execution" / "results" / f"upfd_benchmark_{args.dataset}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Benchmark Triplo GNN -- {args.dataset.upper()}")
    print("=" * 70)
    print(f"  Device:   {device}")
    if device.type == "cuda":
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print(f"  Epocas: {args.epochs} | LR: {args.lr}")
    print()

    # ── Dados ────────────────────────────────────────────────────────────────
    print("[1/3] Carregando dataset...")
    if args.dataset == "fakenewsnet":
        tr_loader, val_loader, test_data, num_features = carregar_fakenewsnet(device)
    else:
        tr_loader, val_loader, test_data, num_features = carregar_upfd(
            args.dataset, args.feature, device
        )

    gcn  = GCNClassifier(num_features, 2).to(device)
    gat  = GATClassifier(num_features, 2).to(device)
    sage = SAGEClassifier(num_features, 2).to(device)

    params = {
        "GCN":  gcn.count_parameters(),
        "GAT":  gat.count_parameters(),
        "SAGE": sage.count_parameters(),
    }
    print(f"\n  GCN  parametros: {params['GCN']:,}")
    print(f"  GAT  parametros: {params['GAT']:,}")
    print(f"  SAGE parametros: {params['SAGE']:,}")

    tempos     = {"GCN": 0.0, "GAT": 0.0, "SAGE": 0.0}
    historicos = {}

    # Nomes dos arquivos de pesos
    pesos = {
        "GCN":  WEIGHTS_DIR / f"pesos_gcn_upfd_{args.dataset}.pth",
        "GAT":  WEIGHTS_DIR / f"pesos_gat_upfd_{args.dataset}.pth",
        "SAGE": WEIGHTS_DIR / f"pesos_sage_upfd_{args.dataset}.pth",
    }

    if args.usar_pesos_salvos:
        # ── Carrega pesos ja treinados ────────────────────────────────────
        print("\n[2/3] Carregando pesos salvos...")
        modelos = {"GCN": gcn, "GAT": gat, "SAGE": sage}
        for nome, model in modelos.items():
            p = pesos[nome]
            if p.exists():
                model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                print(f"  [OK] {nome}: {p.name}")
            else:
                print(f"  [WARN] {p.name} nao encontrado -- treinando {nome} do zero.")
                m, hl, hv, t = treinar(model, tr_loader, val_loader, device,
                                       args.epochs, args.lr)
                tempos[nome] = t
                historicos[nome] = {"loss": hl, "val": hv}
                torch.save(m.state_dict(), p)
    else:
        # ── Treina do zero ────────────────────────────────────────────────
        print(f"\n[2/3] Treinando os 3 modelos ({args.epochs} epocas max)...")
        modelos_treino = [
            ("GCN",  gcn,  "[GCN]  "),
            ("GAT",  gat,  "[GAT]  "),
            ("SAGE", sage, "[SAGE] "),
        ]
        for nome, model, prefixo in modelos_treino:
            print(f"\n  {prefixo} Iniciando treinamento...")
            m, hl, hv, t = treinar(model, tr_loader, val_loader, device,
                                   args.epochs, args.lr)
            tempos[nome] = t
            historicos[nome] = {"loss": hl, "val": hv}
            torch.save(m.state_dict(), pesos[nome])
            print(f"  [OK] Pesos salvos -> {pesos[nome].name}")

    # ── Avaliacao ─────────────────────────────────────────────────────────
    print("\n[3/3] Avaliando no conjunto de teste...")
    resultados = {
        "GCN":  avaliar_completo(gcn,  test_data, device),
        "GAT":  avaliar_completo(gat,  test_data, device),
        "SAGE": avaliar_completo(sage, test_data, device),
    }

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n[Plots] Salvando em {OUT_DIR}...")
    plot_matrizes(resultados, args.dataset, OUT_DIR)
    plot_metricas(resultados, tempos, args.dataset, OUT_DIR)
    if historicos:
        plot_curvas(historicos, OUT_DIR)
    salvar_relatorio(resultados, tempos, params, args.epochs, args.dataset, OUT_DIR)

    print(f"\n[OK] Benchmark UPFD concluido!")
    print(f"     Resultados em: {OUT_DIR}")
    print(f"\n     Proximo passo: python 08_inferencia_cruzada.py")


if __name__ == "__main__":
    main()
