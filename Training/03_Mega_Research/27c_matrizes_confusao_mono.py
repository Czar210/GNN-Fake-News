"""
27c_matrizes_confusao_mono.py
-----------------------------
Gera matrizes de confusao reais (com ground truth) dos 3 modelos persistidos +
repinta a F20 (cross-tab Bluesky) em paleta monocromatica, onde 0 = pastel
(quase branco) e o maximo = cor saturada. Convencao pedida pelo orientador.

Saidas (em Imagens/ + Execution/results/figuras_tcc/comparativas/):
  F33_cm_logreg_bert_fnn.png        -- LogReg-BERT em FakeNewsNet (test split)
  F34_cm_rf_struct_gossipcop.png    -- RF estrutural em UPFD-GossipCop test
  F35_cm_sage_struct_gossipcop.png  -- SAGE estrutural em UPFD-GossipCop test
  F20_bluesky_crosstab.png          -- redo monocromatico (sobrescreve a versao verde-vermelho)

Uso:
  python 27c_matrizes_confusao_mono.py
"""

import csv
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_model import SAGEClassifier
from gerar_folds import carregar_grafos as carregar_fnn

RAIZ        = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
MAT_DIR     = RAIZ / "Material"
OUT_TCC     = RAIZ / "Material" / "GNN_TCC_atualizado" / "Imagens"
OUT_RES     = RAIZ / "Execution" / "results" / "figuras_tcc" / "comparativas"
AMOSTRA_BSK = RAIZ / "Execution" / "results" / "figuras_tcc" / "concordancia_bluesky" / "amostra_5k.csv"


# ─── Paleta monocromatica: 0 = pastel, max = saturada ────────────────────────
# Branco-quase-pastel -> azul medio. Tom unico, sem inversao de semantica.
CMAP_MONO = LinearSegmentedColormap.from_list(
    "mono_pastel",
    ["#f7fbff",   # quase branco/pastel para 0
     "#deebf7",
     "#c6dbef",
     "#9ecae1",
     "#6baed6",
     "#4292c6",
     "#2171b5"]  # azul medio-escuro para maximo
)


def plotar_matriz_mono(cm: np.ndarray, labels, titulo: str, subtitle: str,
                       out_paths, valor_total=None, mostrar_pct=True):
    """
    cm: matriz 2x2 com contagens.
    labels: lista [label_classe_0, label_classe_1]. Convencao: linhas=verdade, cols=predicao.
    """
    n = cm.sum() if valor_total is None else valor_total
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap=CMAP_MONO, vmin=0, vmax=cm.max() * 1.02 if cm.max() > 0 else 1)
    # Numero grande + porcentagem
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            txt_color = "white" if v > cm.max() * 0.55 else "#222"
            ax.text(j, i - 0.04, f"{v:,}".replace(",", "."),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=txt_color)
            if mostrar_pct:
                ax.text(j, i + 0.22, f"{100 * v / n:.1f}%",
                        ha="center", va="center", fontsize=10, color=txt_color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Verdade", fontsize=11)
    ax.set_title(f"{titulo}\n{subtitle}", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("contagem")
    plt.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()


# ─── F33: LogReg-BERT em FakeNewsNet ─────────────────────────────────────────
def cm_logreg_fnn():
    print("[1/4] LogReg-BERT em FakeNewsNet...")
    with open(WEIGHTS_DIR / "logreg_bert_fnn.pkl", "rb") as f:
        b = pickle.load(f)
    lr = b["model"]
    fnn = carregar_fnn()  # 754 grafos
    X = torch.stack([g.x[0] for g in fnn]).numpy()
    y = np.array([g.y.item() for g in fnn], dtype=np.int64)
    pred = lr.predict(X)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    acc = accuracy_score(y, pred)
    # Convencao UPFD: 0=fake, 1=real
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plotar_matriz_mono(
        cm, ["FAKE", "REAL"],
        "Matriz de confusao: LogReg-BERT em FakeNewsNet",
        f"N={len(y)} grafos | F1-macro={f1m:.3f} | Acc={acc:.3f}",
        [OUT_TCC / "F33_cm_logreg_bert_fnn.png",
         OUT_RES / "F33_cm_logreg_bert_fnn.png"],
    )
    print(f"   F1m={f1m:.4f} acc={acc:.4f}")


# ─── F34: RF estrutural em UPFD-GossipCop test ───────────────────────────────
def cm_rf_gossipcop():
    print("[2/4] RF estrutural em UPFD-GossipCop test...")
    with open(WEIGHTS_DIR / "rf_struct_gossipcop.pkl", "rb") as f:
        b = pickle.load(f)
    rf = b["model"]
    test = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="test"))
    X = np.array([[g.num_nodes,
                   int((g.edge_index[0] == 0).sum().item()) if g.edge_index.numel() else 0]
                  for g in test], dtype=np.float64)
    y = np.array([g.y.item() for g in test], dtype=np.int64)
    pred = rf.predict(X)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plotar_matriz_mono(
        cm, ["FAKE", "REAL"],
        "Matriz de confusao: RF estrutural em UPFD-GossipCop test",
        f"features [num_nodes, grau_root] | N={len(y)} | F1-macro={f1m:.3f} | Acc={acc:.3f}",
        [OUT_TCC / "F34_cm_rf_struct_gossipcop.png",
         OUT_RES / "F34_cm_rf_struct_gossipcop.png"],
    )
    print(f"   F1m={f1m:.4f} acc={acc:.4f}")


# ─── F35: SAGE estrutural em UPFD-GossipCop test ─────────────────────────────
def cm_sage_gossipcop():
    print("[3/4] SAGE estrutural em UPFD-GossipCop test...")
    ckpt = torch.load(WEIGHTS_DIR / "sage_struct_gossipcop.pth",
                      map_location="cpu", weights_only=False)
    sage = SAGEClassifier(num_node_features=ckpt["input_dim"], num_classes=2,
                          hidden_channels=ckpt["hidden_channels"])
    sage.load_state_dict(ckpt["state_dict"]); sage.eval()

    test = list(UPFD(root=str(MAT_DIR), name="gossipcop", feature="profile", split="test"))

    def feats_sage(g):
        n = g.num_nodes
        is_root = torch.zeros(n, dtype=torch.float); is_root[0] = 1.0
        deg = torch.zeros(n, dtype=torch.float)
        if g.edge_index.numel():
            idx, c = torch.unique(g.edge_index[0], return_counts=True)
            deg[idx] = c.float()
        deg_norm = deg / max(deg.max().item(), 1.0)
        return torch.stack([is_root, deg_norm], dim=1)

    transformados = [Data(x=feats_sage(g), edge_index=g.edge_index, y=g.y) for g in test]
    loader = DataLoader(transformados, batch_size=32, shuffle=False)
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            out, _ = sage(batch.x, batch.edge_index, batch.batch)
            y_pred.extend(out.argmax(dim=1).tolist())
            labels = batch.y.squeeze().tolist()
            y_true.extend(labels if isinstance(labels, list) else [labels])
    y = np.array(y_true); pred = np.array(y_pred)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plotar_matriz_mono(
        cm, ["FAKE", "REAL"],
        "Matriz de confusao: SAGE estrutural em UPFD-GossipCop test",
        f"features [is_root, grau_norm] | N={len(y)} | F1-macro={f1m:.3f} | Acc={acc:.3f}",
        [OUT_TCC / "F35_cm_sage_struct_gossipcop.png",
         OUT_RES / "F35_cm_sage_struct_gossipcop.png"],
    )
    print(f"   F1m={f1m:.4f} acc={acc:.4f}")


# ─── F20 redo: cross-tab Bluesky monocromatico ───────────────────────────────
def crosstab_bluesky_mono():
    print("[4/4] Cross-tab Bluesky (monocromatico, sobrescreve a F20)...")
    if not AMOSTRA_BSK.exists():
        print("  [SKIP] amostra_5k.csv ausente"); return
    with open(AMOSTRA_BSK, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mat = np.zeros((2, 2), dtype=int)  # linhas=text, cols=topo. 0=FAKE, 1=REAL
    for r in rows:
        try:
            t = int(float(r["pred_text"])); k = int(float(r["pred_topo"]))
            if t in (0, 1) and k in (0, 1):
                mat[t, k] += 1
        except Exception: pass
    N = mat.sum()
    concord = mat[0, 0] + mat[1, 1]
    discord = mat[0, 1] + mat[1, 0]
    plotar_matriz_mono(
        mat, ["FAKE", "REAL"],
        "Cross-tab dos classificadores no Bluesky",
        f"N={N:,} | Concordam: {concord:,} ({100*concord/N:.1f}%) | "
        f"Discordam: {discord:,} ({100*discord/N:.1f}%)\n"
        f"Linhas = predito pelo TEXTUAL (LogReg-BERT); Cols = predito pelo TOPOLOGICO (RF estrutural). "
        f"NAO e matriz de confusao supervisionada (Bluesky sem ground truth).".replace(",", "."),
        [OUT_TCC / "F20_bluesky_crosstab.png",
         OUT_RES / "F20_bluesky_crosstab.png"],
    )


def main():
    print("Gerando matrizes de confusao em paleta monocromatica (0 = pastel)...")
    cm_logreg_fnn()
    cm_rf_gossipcop()
    cm_sage_gossipcop()
    crosstab_bluesky_mono()
    print(f"\n[OK] Saidas em {OUT_TCC} e {OUT_RES}")


if __name__ == "__main__":
    main()
