"""
09_teste_significancia.py
--------------------------
Testes de significancia estatistica entre GCN, GAT e GraphSAGE.

Metodologia (Fase 2B.1, Erro 4):
  - Itera sobre os K=10 folds estratificados pre-gerados em
    data/folds_fnn.pt (ver gerar_folds.py).
  - Para cada fold, treina cada arquitetura no train_idx (com 10% reservado
    como val para early stopping) e avalia no test_idx.
  - Coleta F1, Accuracy, Precision, Recall por fold.
  - Executa t-test pareado (scipy.stats.ttest_rel) entre todos os pares
    de modelos -- agora a paridade do teste e sobre fold (variancia de
    generalizacao), nao sobre seed (variancia de inicializacao).

Dataset padrao: FakeNewsNet (grafos reais, balanceado ~50/50).
  -> Usa os .pt pre-construidos por 00_construir_grafos_fakenewsnet.py
     e os folds compartilhados gerados por gerar_folds.py.

Saidas em Execution/results/teste_significancia/:
  relatorio.txt       -- tabela completa de metricas e p-valores
  boxplot_f1.png      -- distribuicao F1 por arquitetura (K folds)
  violino_f1.png      -- violin plot + pontos individuais

Uso:
  python gerar_folds.py             # uma vez, gera folds_fnn.pt
  python 09_teste_significancia.py
  python 09_teste_significancia.py --epochs 50
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
OUT_DIR  = RAIZ / "Execution" / "results" / "fase4_benchmarks" / "teste_significancia"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcn_model  import GCNClassifier
from gat_model  import GATClassifier
from sage_model import SAGEClassifier
from gerar_folds import iterar_folds, carregar_grafos

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

def carregar_dataset_completo(data_suffix: str = None) -> tuple:
    """
    Retorna (grafos, num_features). Para uso com k-fold a partir de folds_fnn.pt.
    Se data_suffix for None, usa raiz data/ (legacy bugado). Caso contrario,
    carrega de data/fakenewsnet_<suffix>/.
    """
    folds_path = DATA_DIR / "folds_fnn.pt"
    if not folds_path.exists():
        print(f"[ERRO] {folds_path} nao encontrado.")
        print("       Execute: python gerar_folds.py")
        sys.exit(1)

    if data_suffix:
        sub = DATA_DIR / f"fakenewsnet_{data_suffix}"
        if not sub.exists():
            print(f"[ERRO] {sub} nao encontrado.")
            print(f"       Execute: python 00_construir_grafos_fakenewsnet.py "
                  f"--feature-variant <variant> --output-suffix {data_suffix}")
            sys.exit(1)
        grafos = []
        import torch as _torch
        for split in ("train", "val", "test"):
            grafos += _torch.load(sub / f"fakenewsnet_{split}.pt", weights_only=False)
        print(f"  Dataset: data/fakenewsnet_{data_suffix}/")
    else:
        grafos = carregar_grafos()
        print(f"  Dataset: data/ (raiz, legacy)")

    num_feats = grafos[0].x.shape[1]
    fakes = sum(1 for g in grafos if g.y.item() == 0)
    print(f"  FakeNewsNet: {len(grafos)} grafos | fake={fakes} "
          f"({100*fakes/len(grafos):.1f}%) | features={num_feats}")
    return grafos, num_feats


def split_train_val(train_idx: list, grafos: list, val_frac: float = 0.10,
                    seed: int = 0) -> tuple:
    """
    A partir do train_idx do fold, separa val_frac (estratificado por classe)
    para early stopping. Retorna (train_subset, val_subset).
    """
    import random
    rng = random.Random(seed)
    fakes = [i for i in train_idx if grafos[i].y.item() == 0]
    reals = [i for i in train_idx if grafos[i].y.item() == 1]
    rng.shuffle(fakes); rng.shuffle(reals)
    n_val_fk = max(1, int(len(fakes) * val_frac))
    n_val_re = max(1, int(len(reals) * val_frac))
    val_idx   = fakes[:n_val_fk] + reals[:n_val_re]
    train_sub = fakes[n_val_fk:] + reals[n_val_re:]
    return [grafos[i] for i in train_sub], [grafos[i] for i in val_idx]


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

        # Criterio de selecao: F1-macro (alinha com avaliacao final)
        val_f1 = f1_score(val_y_true, val_y_pred, average="macro", zero_division=0)
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
    ax.set_title(f"Distribuicao de {metrica.upper()} -- {len(vals[0])} folds\n"
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
                 f"({len(vals[0])} folds, FakeNewsNet)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = out_dir / f"violino_{metrica}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {p.name}")


# ─── Relatorio ────────────────────────────────────────────────────────────────

def salvar_relatorio(resultados: dict, pares_ttest: dict, n_folds: int,
                     out_dir: Path) -> None:
    arqs = list(resultados.keys())
    metricas = ["f1", "accuracy", "precision", "recall"]

    linhas = [
        "=" * 72,
        "  TESTES DE SIGNIFICANCIA -- GCN vs GAT vs GraphSAGE",
        "  Dataset: FakeNewsNet (politifact) | label 0=Fake",
        f"  Folds: {n_folds} (StratifiedKFold seed=42) | t-test pareado (ttest_rel)",
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
        description="Testes de significancia entre GCN, GAT e GraphSAGE (k-fold pareado).")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--cpu",         action="store_true")
    parser.add_argument("--data-suffix", type=str, default=None,
                        choices=["posfull", "posmin", "posgrau", "bugado_original"],
                        help="Sufixo do dataset em data/fakenewsnet_<suffix>/. "
                             "Default: legacy bugado em data/ (raiz).")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = (OUT_DIR.parent / f"teste_significancia_{args.data_suffix}") if args.data_suffix else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  TESTES DE SIGNIFICANCIA -- GCN vs GAT vs GraphSAGE (K-FOLD PAREADO)")
    print("=" * 72)
    print(f"  Device: {device} | Epochs max: {args.epochs}")
    print()

    # ── Carregar dados + folds ────────────────────────────────────────────
    print("[1/3] Carregando FakeNewsNet e folds compartilhados...")
    grafos, num_feats = carregar_dataset_completo(args.data_suffix)
    folds = list(iterar_folds())
    print(f"  Folds: {len(folds)} (compartilhados via folds_fnn.pt)")
    print()

    # ── Treinar 1 modelo por arquitetura por fold ─────────────────────────
    print(f"[2/3] Treinando {len(folds)} folds x 3 arquiteturas "
          f"({len(folds) * 3} treinos total)...")

    arqs = ["GCN", "GAT", "SAGE"]
    # resultados[arq][metrica] = lista de K valores (1 por fold)
    resultados: dict = {a: {"f1": [], "accuracy": [], "precision": [], "recall": []}
                        for a in arqs}

    for fold_idx, train_idx, test_idx in folds:
        torch.manual_seed(fold_idx)
        np.random.seed(fold_idx)

        train_sub, val_sub = split_train_val(train_idx, grafos, val_frac=0.10,
                                              seed=fold_idx)
        test_sub = [grafos[i] for i in test_idx]

        for arq in arqs:
            t0    = time.time()
            model = criar_modelo(arq, num_feats, seed=fold_idx, device=device)
            model = treinar_uma_vez(model, train_sub, val_sub,
                                    device, args.epochs, args.lr)
            res   = avaliar(model, test_sub, device, pos_label_fake=0)
            elapsed = time.time() - t0

            for m in ("f1", "accuracy", "precision", "recall"):
                resultados[arq][m].append(res[m])

            print(f"  Fold {fold_idx:02d}/{len(folds)-1} | {arq:<5} | "
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
    print(f"\n[Plots] Salvando em {out_dir}...")
    plot_boxplot(resultados, "f1", out_dir)
    plot_violino(resultados, "f1", out_dir)
    plot_boxplot(resultados, "accuracy", out_dir)

    salvar_relatorio(resultados, pares_ttest, len(folds), out_dir)

    # Tabela LaTeX p-valores
    tex_path = out_dir / "tabela_significancia.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Tabela de significancia (ttest_rel pareado por fold) -- GERADO AUTOMATICAMENTE\n")
        f.write("\\begin{tabular}{lrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Par & $\\bar{F1}_A$ & $\\bar{F1}_B$ & $t$ & $p$ & sig \\\\\n")
        f.write("\\midrule\n")
        for (a, b), (t, p, sig) in pares_ttest.items():
            ma = float(np.mean(resultados[a]['f1']))
            mb = float(np.mean(resultados[b]['f1']))
            f.write(f"{a} vs {b} & {ma:.4f} & {mb:.4f} & {t:.3f} & {p:.4f} & {sig} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  [OK] {tex_path.name}")

    # CSV por fold
    csv_path = out_dir / "por_fold.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["modelo", "fold", "f1_macro", "f1_fake", "accuracy", "precision", "recall"])
        for arq in resultados:
            for fold_idx in range(len(resultados[arq]["f1"])):
                w.writerow([arq, fold_idx,
                            resultados[arq]["f1"][fold_idx],
                            resultados[arq]["f1"][fold_idx],  # ja eh fake-only neste 09
                            resultados[arq]["accuracy"][fold_idx],
                            resultados[arq]["precision"][fold_idx],
                            resultados[arq]["recall"][fold_idx]])
    print(f"  [OK] {csv_path.name}")

    print(f"\n[OK] Testes de significancia concluidos!")
    print(f"     Resultados em: {out_dir}")


if __name__ == "__main__":
    main()
