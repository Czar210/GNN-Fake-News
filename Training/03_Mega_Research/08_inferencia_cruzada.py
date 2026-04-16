"""
08_inferencia_cruzada.py
------------------------
Experimento de transferencia entre dominios (cross-dataset inference):
testa se um modelo treinado numa rede social consegue classificar
corretamente dados de outra rede social.

Experimentos:
  BS -> FNN  : modelo treinado no Bluesky aplicado ao FakeNewsNet
  FNN -> BS  : modelo treinado no FakeNewsNet aplicado ao Bluesky

Hipotese:
  BS -> FNN : degradacao esperada. O modelo Bluesky nunca viu grafos com
               estrutura real (os grafos Bluesky tinham 1 no, 0 arestas).
               Aprende apenas features BERT de texto, nao propagacao.
               No FakeNewsNet (grafos ricos), as camadas GNN processam
               estrutura desconhecida.

  FNN -> BS : degradacao parcial. O modelo FakeNewsNet aprendeu a usar
               propagacao real. No Bluesky, grafos degeneraram para 1 no
               -- o modelo usa apenas features proprias. A queda reflete
               diferenca de dominio (Twitter vs Bluesky).

Saidas em Execution/results/inferencia_cruzada/:
  relatorio.txt          -- metricas de todos os experimentos
  matrizes_cruzadas.png  -- grade de matrizes de confusao (3 arqs x 2 dirs)

Uso:
  python 08_inferencia_cruzada.py
  python 08_inferencia_cruzada.py --dataset fakenewsnet
  python 08_inferencia_cruzada.py --cpu
"""

import argparse
import sys
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
OUT_DIR     = RAIZ / "Execution" / "results" / "inferencia_cruzada"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model  import GCNClassifier
from gat_model  import GATClassifier
from sage_model import SAGEClassifier

BATCH_SIZE = 64


# ─── Carregamento de dados ────────────────────────────────────────────────────

def carregar_bluesky(device: torch.device) -> tuple:
    """Carrega o conjunto de teste do Bluesky."""
    test_path = DATA_DIR / "grafos_bluesky_test.pt"
    if not test_path.exists():
        print(f"[ERRO] {test_path} nao encontrado.")
        print("       Execute: python 02_construir_grafos_bluesky.py")
        sys.exit(1)

    test_data = torch.load(test_path, weights_only=False)
    fakes = sum(1 for g in test_data if g.y.item() == 1)
    print(f"  Bluesky test: {len(test_data):,} grafos | "
          f"Fake(1)={fakes} ({100*fakes/len(test_data):.2f}%) | "
          f"nos_medio={sum(g.num_nodes for g in test_data)/len(test_data):.1f}")
    return test_data


def carregar_fakenewsnet_test(_device: torch.device) -> list:
    """Carrega o conjunto de teste do FakeNewsNet (label 0=Fake, 1=Real)."""
    test_path = DATA_DIR / "fakenewsnet_test.pt"
    if not test_path.exists():
        print(f"[ERRO] {test_path} nao encontrado.")
        print("       Execute: python 00_construir_grafos_fakenewsnet.py")
        sys.exit(1)
    test_data = torch.load(test_path, weights_only=False)
    fakes = sum(1 for g in test_data if g.y.item() == 0)
    nos   = [g.num_nodes for g in test_data]
    print(f"  FakeNewsNet test: {len(test_data):,} grafos | "
          f"Fake(0)={fakes} ({100*fakes/len(test_data):.2f}%) | "
          f"nos_medio={sum(nos)/len(nos):.1f}")
    return test_data


# ─── Avaliacao ───────────────────────────────────────────────────────────────

@torch.no_grad()
def avaliar(model, test_data, device, pos_label_fake: int) -> dict:
    """
    Avalia o modelo no conjunto de teste.

    pos_label_fake: qual valor de y representa Fake?
      - Bluesky: 1 = Fake
      - UPFD:    0 = Fake
    """
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
        "precision": round(precision_score(y_true, y_pred,
                           pos_label=pos_label_fake, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,
                           pos_label=pos_label_fake, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,
                           pos_label=pos_label_fake, zero_division=0), 4),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─── Carregamento de pesos ────────────────────────────────────────────────────

def carregar_modelo(arquitetura: str, origem: str, num_features: int,
                    device: torch.device, dataset_name: str) -> torch.nn.Module:
    """
    Instancia e carrega os pesos de um modelo treinado.

    arquitetura: 'GCN' | 'GAT' | 'SAGE'
    origem:      'bluesky' | 'upfd'
    """
    if arquitetura == "GCN":
        model = GCNClassifier(num_features, 2).to(device)
    elif arquitetura == "GAT":
        model = GATClassifier(num_features, 2).to(device)
    elif arquitetura == "SAGE":
        model = SAGEClassifier(num_features, 2).to(device)
    else:
        raise ValueError(f"Arquitetura desconhecida: {arquitetura}")

    if origem == "bluesky":
        # Pesos treinados no Bluesky (baseline sem class weights)
        nome_arquivo = {
            "GCN":  "pesos_gcn.pth",
            "GAT":  "pesos_gat.pth",
            "SAGE": "pesos_sage.pth",
        }[arquitetura]
    else:  # upfd
        nome_arquivo = f"pesos_{arquitetura.lower()}_upfd_{dataset_name}.pth"

    path = WEIGHTS_DIR / nome_arquivo
    if not path.exists():
        print(f"  [WARN] {path.name} nao encontrado -- modelo nao carregado (zeros).")
        return model

    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print(f"  [OK] {arquitetura} ({origem}): {path.name}")
    return model


# ─── Visualizacao ─────────────────────────────────────────────────────────────

CORES = {"GCN": "#4F86C6", "GAT": "#E07B54", "SAGE": "#5BAD72"}


def plot_matrizes_cruzadas(todos_resultados: dict, dataset_name: str, out_dir: Path) -> None:
    """
    Grade 3x2: linhas = arquiteturas (GCN/GAT/SAGE), colunas = direcao (BS->UPFD, UPFD->BS).
    """
    arquiteturas = ["GCN", "GAT", "SAGE"]
    direcoes     = [
        (f"BS -> UPFD\n(treino Bluesky,\nteste {dataset_name})", "bs_upfd"),
        (f"UPFD -> BS\n(treino {dataset_name},\nteste Bluesky)",  "upfd_bs"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(
        f"Inferencia Cruzada -- Bluesky <-> UPFD ({dataset_name})\n"
        "Linhas: arquitetura | Colunas: direcao",
        fontsize=13, fontweight="bold",
    )

    for row, arq in enumerate(arquiteturas):
        for col, (titulo_col, chave) in enumerate(direcoes):
            ax  = axes[row][col]
            res = todos_resultados.get(arq, {}).get(chave, {})
            cm  = np.array(res.get("confusion", [[0, 0], [0, 0]]))

            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            fig.colorbar(im, ax=ax, shrink=0.8)

            f1  = res.get("f1", 0.0)
            acc = res.get("accuracy", 0.0)
            ax.set_title(f"{arq} | {titulo_col}\nF1={f1:.4f} Acc={acc:.4f}",
                         fontsize=9, pad=8)

            # Labels dependem da direcao (pos_label diferente)
            if chave == "bs_upfd":
                tick_labels = ["Fake(0)", "Real(1)"]
            else:
                tick_labels = ["Real(0)", "Fake(1)"]

            ax.set(
                xticks=[0, 1], yticks=[0, 1],
                xticklabels=tick_labels, yticklabels=tick_labels,
                xlabel="Predicao", ylabel="Realidade",
            )
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > (cm.max() or 1) / 2 else "black",
                            fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = out_dir / "matrizes_cruzadas.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def plot_comparativo_f1(
    resultados_bluesky: dict,
    resultados_upfd: dict,
    todos_cruzados: dict,
    dataset_name: str,
    out_dir: Path,
) -> None:
    """
    Grafico de barras comparando F1 em 4 cenarios para cada arquitetura:
    1. Treinado e testado no Bluesky (in-domain)
    2. Treinado e testado no UPFD    (in-domain)
    3. Treinado no Bluesky, testado no UPFD (cross)
    4. Treinado no UPFD, testado no Bluesky (cross)
    """
    arquiteturas = ["GCN", "GAT", "SAGE"]
    cenarios = [
        f"BS->BS\n(in-domain)",
        f"UPFD->UPFD\n(in-domain)",
        f"BS->UPFD\n(cross)",
        f"UPFD->BS\n(cross)",
    ]

    x     = np.arange(len(cenarios))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"F1-Score por Cenario -- Transferencia de Dominio (UPFD={dataset_name})",
        fontsize=13, fontweight="bold",
    )

    for arq, offset in zip(arquiteturas, offsets):
        vals = [
            resultados_bluesky.get(arq, {}).get("f1", 0.0),
            resultados_upfd.get(arq, {}).get("f1", 0.0),
            todos_cruzados.get(arq, {}).get("bs_upfd", {}).get("f1", 0.0),
            todos_cruzados.get(arq, {}).get("upfd_bs", {}).get("f1", 0.0),
        ]
        bars = ax.bar(x + offset, vals, width, label=arq, color=CORES[arq], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set(xticks=x, xticklabels=cenarios, ylim=(0, 1.15), ylabel="F1-Score (Fake)")
    ax.legend(title="Arquitetura")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(0.8, 1.08, "In-domain", ha="center", fontsize=10, color="gray")
    ax.text(2.2, 1.08, "Cross-domain", ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    path = out_dir / "comparativo_f1_cenarios.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


def salvar_relatorio(
    resultados_bluesky: dict,
    resultados_upfd: dict,
    todos_cruzados: dict,
    dataset_name: str,
    out_dir: Path,
) -> None:
    linhas = [
        "=" * 72,
        f"  RELATORIO DE INFERENCIA CRUZADA -- Bluesky <-> UPFD ({dataset_name})",
        "=" * 72,
        "",
        "  NOTA: Bluesky usa label 1=Fake; UPFD usa label 0=Fake.",
        "  F1 calculado sobre a classe Fake em ambos os casos.",
        "",
        f"  {'Cenario':<32} {'GCN':>8} {'GAT':>8} {'SAGE':>8}",
        f"  {'-'*60}",
    ]

    cenarios = [
        ("BS -> BS   (in-domain, Bluesky)",   resultados_bluesky),
        (f"UPFD -> UPFD (in-domain, {dataset_name})", resultados_upfd),
    ]
    for label, res_dict in cenarios:
        vals = [res_dict.get(a, {}).get("f1", 0.0) for a in ["GCN", "GAT", "SAGE"]]
        linhas.append(f"  {label:<32} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")

    linhas.append("")
    for chave, label in [("bs_upfd", f"BS -> UPFD (cross, treino Bluesky)"),
                         ("upfd_bs", f"UPFD -> BS (cross, treino {dataset_name})")]:
        vals = [todos_cruzados.get(a, {}).get(chave, {}).get("f1", 0.0)
                for a in ["GCN", "GAT", "SAGE"]]
        linhas.append(f"  {label:<32} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")

    linhas += [
        "",
        "=" * 72,
        "  ANALISE DE TRANSFERENCIA",
        "=" * 72,
        "",
    ]

    # Analise BS->FNN
    gcn_bs_upfd  = todos_cruzados.get("GCN", {}).get("bs_upfd", {}).get("f1", 0.0)
    gcn_upfd     = resultados_upfd.get("GCN", {}).get("f1", 0.0)
    gcn_bs       = resultados_bluesky.get("GCN", {}).get("f1", 0.0)
    linhas += [
        "  [BS -> FNN] Modelo treinado no Bluesky, testado no FakeNewsNet:",
        f"    GCN in-domain FNN: {gcn_upfd:.4f} vs cross BS->FNN: {gcn_bs_upfd:.4f}",
        f"    Queda GCN: {gcn_upfd - gcn_bs_upfd:.4f}",
        "    Interpretacao: o modelo Bluesky aprendeu classificacao textual (BERT),",
        "    ja que os grafos Bluesky tinham 0 arestas (bug de pipeline). O F1",
        "    residual no FNN (~0.65) reflete transferencia de sinal textual",
        "    entre dominios (ambos em ingles, PolitiFact/Twitter).",
        "",
    ]

    # Analise FNN->BS
    gcn_upfd_bs = todos_cruzados.get("GCN", {}).get("upfd_bs", {}).get("f1", 0.0)
    # Recuperar precision/recall cross para diagnostico de prevalencia
    gcn_cross_p = todos_cruzados.get("GCN", {}).get("upfd_bs", {}).get("precision", 0.0)
    gcn_cross_r = todos_cruzados.get("GCN", {}).get("upfd_bs", {}).get("recall",    0.0)
    linhas += [
        "  [FNN -> BS] Modelo treinado no FakeNewsNet, testado no Bluesky:",
        f"    GCN in-domain BS: {gcn_bs:.4f} vs cross FNN->BS: {gcn_upfd_bs:.4f}",
        f"    Queda GCN: {gcn_bs - gcn_upfd_bs:.4f}",
        f"    Precision={gcn_cross_p:.4f} | Recall={gcn_cross_r:.4f}",
        "    CAUSA PRINCIPAL: mismatch de prevalencia de classes.",
        "    FakeNewsNet (treino): ~50% fake. Bluesky (teste): ~0.58% fake.",
        "    Um modelo calibrado para 50% de prevalencia produz excesso de",
        "    falsos positivos num dataset com 0.58% de prevalencia, colapsando",
        "    a Precision e portanto o F1 -- independente da arquitetura GNN.",
        "    Causa secundaria: diferenca de dominio textual (Twitter vs Bluesky).",
        "    NOTA: para testar transferencia valida, os datasets precisam ter",
        "    prevalencias de classe comparaveis ou usar AUPRC em vez de F1.",
        "",
        "=" * 72,
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
    parser = argparse.ArgumentParser(description="Inferencia cruzada Bluesky <-> FakeNewsNet.")
    parser.add_argument("--dataset", type=str, default="fakenewsnet",
                        choices=["fakenewsnet"])
    parser.add_argument("--cpu",    action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  Inferencia Cruzada -- Bluesky <-> FakeNewsNet ({args.dataset})")
    print("=" * 72)
    print(f"  Device: {device}")
    print()

    # ── Carregar dados de teste ───────────────────────────────────────────
    print("[1/3] Carregando conjuntos de teste...")
    print("  Bluesky:")
    bs_test  = carregar_bluesky(device)
    print("  FakeNewsNet:")
    fnn_test = carregar_fakenewsnet_test(device)

    NUM_FEATURES = 768   # BERT -- mesmo em ambos os datasets

    # ── Carregar modelos ──────────────────────────────────────────────────
    print("\n[2/3] Carregando pesos salvos...")

    # Modelos treinados no Bluesky
    print("\n  Modelos treinados no Bluesky:")
    gcn_bs  = carregar_modelo("GCN",  "bluesky", NUM_FEATURES, device, args.dataset)
    gat_bs  = carregar_modelo("GAT",  "bluesky", NUM_FEATURES, device, args.dataset)
    sage_bs = carregar_modelo("SAGE", "bluesky", NUM_FEATURES, device, args.dataset)

    # Modelos treinados no FakeNewsNet
    print("\n  Modelos treinados no FakeNewsNet:")
    gcn_fnn  = carregar_modelo("GCN",  "upfd", NUM_FEATURES, device, args.dataset)
    gat_fnn  = carregar_modelo("GAT",  "upfd", NUM_FEATURES, device, args.dataset)
    sage_fnn = carregar_modelo("SAGE", "upfd", NUM_FEATURES, device, args.dataset)

    # ── Avaliacoes ────────────────────────────────────────────────────────
    print("\n[3/3] Calculando metricas...")

    # In-domain Bluesky (pos_label=1 para Fake)
    print("\n  In-domain Bluesky (Fake=1):")
    res_bs = {}
    for nome, model in [("GCN", gcn_bs), ("GAT", gat_bs), ("SAGE", sage_bs)]:
        r = avaliar(model, bs_test, device, pos_label_fake=1)
        res_bs[nome] = r
        print(f"    {nome}: Acc={r['accuracy']:.4f} | P={r['precision']:.4f} | "
              f"R={r['recall']:.4f} | F1={r['f1']:.4f}")

    # In-domain FakeNewsNet (pos_label=0 para Fake)
    print(f"\n  In-domain FakeNewsNet (Fake=0):")
    res_fnn = {}
    for nome, model in [("GCN", gcn_fnn), ("GAT", gat_fnn), ("SAGE", sage_fnn)]:
        r = avaliar(model, fnn_test, device, pos_label_fake=0)
        res_fnn[nome] = r
        print(f"    {nome}: Acc={r['accuracy']:.4f} | P={r['precision']:.4f} | "
              f"R={r['recall']:.4f} | F1={r['f1']:.4f}")

    # Cross BS -> FNN (modelo Bluesky, dados FakeNewsNet, pos_label=0)
    print(f"\n  Cross BS -> FNN (modelo Bluesky, dados FakeNewsNet, Fake=0):")
    res_cruzado = {arq: {} for arq in ["GCN", "GAT", "SAGE"]}
    for nome, model in [("GCN", gcn_bs), ("GAT", gat_bs), ("SAGE", sage_bs)]:
        r = avaliar(model, fnn_test, device, pos_label_fake=0)
        res_cruzado[nome]["bs_upfd"] = r
        print(f"    {nome}: Acc={r['accuracy']:.4f} | P={r['precision']:.4f} | "
              f"R={r['recall']:.4f} | F1={r['f1']:.4f}")

    # Cross FNN -> BS (modelo FakeNewsNet, dados Bluesky, pos_label=1)
    print(f"\n  Cross FNN -> BS (modelo FakeNewsNet, dados Bluesky, Fake=1):")
    for nome, model in [("GCN", gcn_fnn), ("GAT", gat_fnn), ("SAGE", sage_fnn)]:
        r = avaliar(model, bs_test, device, pos_label_fake=1)
        res_cruzado[nome]["upfd_bs"] = r
        print(f"    {nome}: Acc={r['accuracy']:.4f} | P={r['precision']:.4f} | "
              f"R={r['recall']:.4f} | F1={r['f1']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n[Plots] Salvando em {OUT_DIR}...")
    plot_matrizes_cruzadas(res_cruzado, args.dataset, OUT_DIR)
    plot_comparativo_f1(res_bs, res_fnn, res_cruzado, args.dataset, OUT_DIR)
    salvar_relatorio(res_bs, res_fnn, res_cruzado, args.dataset, OUT_DIR)

    print(f"\n[OK] Inferencia cruzada concluida!")
    print(f"     Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
