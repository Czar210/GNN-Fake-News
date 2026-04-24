"""
03_treinar_gat.py
-----------------
Treina 4 variantes do GATClassifier nos grafos Bluesky construídos
pela Etapa 2 (02_construir_grafos_bluesky.py).

Variantes (mesmo esquema dos 4 GCNs do ensemble):
  ┌──────────────────────┬──────────────────┬───────────────────────────────┐
  │ Variante             │ Peso CE          │ Arquivo                       │
  ├──────────────────────┼──────────────────┼───────────────────────────────┤
  │ GAT Baseline         │ [1.0, 1.0]       │ pesos_gat.pth                 │
  │ GAT Controlado       │ [1.0, 1.0]       │ pesos_gat_bs_ctrl.pth         │
  │ GAT Cético           │ [1.0, 4.0]       │ pesos_gat_bs_cetico.pth       │
  │ GAT Extra Cético     │ [1.0, 6.0]       │ pesos_gat_bs_ex_cetico.pth    │
  └──────────────────────┴──────────────────┴───────────────────────────────┘

Uso:
  python 03_treinar_gat.py
  python 03_treinar_gat.py --epochs 50 --lr 0.003
  python 03_treinar_gat.py --variante baseline   # treina só uma variante
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ       = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat_model import GATClassifier

# ─── Configurações padrão ─────────────────────────────────────────────────────

EPOCHS_DEFAULT     = 30
LR_DEFAULT         = 0.005
WEIGHT_DECAY       = 5e-4
BATCH_SIZE         = 64
PATIENCE           = 7    # early stopping: épocas sem melhoria no val_f1 (macro)

VARIANTES = {
    "baseline":   {"peso_ce": [1.0, 1.0], "arquivo": "pesos_gat.pth"},
    "ctrl":       {"peso_ce": [1.0, 1.0], "arquivo": "pesos_gat_bs_ctrl.pth"},
    "cetico":     {"peso_ce": [1.0, 4.0], "arquivo": "pesos_gat_bs_cetico.pth"},
    "ex_cetico":  {"peso_ce": [1.0, 6.0], "arquivo": "pesos_gat_bs_ex_cetico.pth"},
}


# ─── Carregamento dos dados ───────────────────────────────────────────────────

def carregar_dados(device: torch.device) -> tuple:
    """Carrega os .pt gerados pela Etapa 2 e retorna (train_loader, val_loader, test_data)."""
    arquivos = {
        "train": DATA_DIR / "grafos_bluesky_train.pt",
        "val":   DATA_DIR / "grafos_bluesky_val.pt",
        "test":  DATA_DIR / "grafos_bluesky_test.pt",
    }

    for split, path in arquivos.items():
        if not path.exists():
            print(f"[ERRO] {path} não encontrado.")
            print("       Execute primeiro: python 02_construir_grafos_bluesky.py")
            sys.exit(1)

    train_data = torch.load(arquivos["train"], weights_only=False)
    val_data   = torch.load(arquivos["val"],   weights_only=False)
    test_data  = torch.load(arquivos["test"],  weights_only=False)

    print(f"  Train: {len(train_data):,} grafos")
    print(f"  Val:   {len(val_data):,} grafos")
    print(f"  Test:  {len(test_data):,} grafos")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_data


# ─── Loop de treinamento ──────────────────────────────────────────────────────

def treinar_epoca(model, loader, optimizer, criterion, device) -> tuple:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * data.num_graphs
        preds          = out.argmax(dim=1)
        total_correct += int((preds == data.y.squeeze()).sum())
        total         += data.num_graphs

    return total_loss / total, total_correct / total


@torch.no_grad()
def avaliar(model, loader, device) -> float:
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.batch)
        preds  = out.argmax(dim=1).cpu().tolist()
        labels = data.y.squeeze().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels if isinstance(labels, list) else [labels])
    if not y_true:
        return 0.0
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ─── Treinamento de uma variante ──────────────────────────────────────────────

def treinar_variante(
    nome: str,
    config: dict,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> GATClassifier:
    print(f"\n{'-'*55}")
    print(f"  Treinando: {nome.upper().replace('_', ' ')}")
    print(f"  Pesos CrossEntropy: {config['peso_ce']}")
    print(f"  Épocas: {epochs}  |  LR: {lr}  |  Early stopping: {PATIENCE}")
    print(f"{'-'*55}")

    model = GATClassifier(num_node_features=768, num_classes=2).to(device)
    print(f"  Parâmetros: {model.count_parameters():,}")

    peso_tensor = torch.tensor(config["peso_ce"], dtype=torch.float, device=device)
    criterion   = torch.nn.CrossEntropyLoss(weight=peso_tensor)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # LR scheduler: reduz pela metade se val_loss não melhora em 5 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_f1    = 0.0
    best_state     = None
    epochs_sem_melhora = 0
    t_inicio        = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = treinar_epoca(model, train_loader, optimizer, criterion, device)
        val_f1                = avaliar(model, val_loader, device)

        scheduler.step(val_f1)

        melhorou = val_f1 > best_val_f1
        if melhorou:
            best_val_f1        = val_f1
            best_state         = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1

        marker = " <- melhor" if melhorou else ""
        print(f"  Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val F1: {val_f1:.4f}{marker}")

        if epochs_sem_melhora >= PATIENCE:
            print(f"  [Early stopping] {PATIENCE} épocas sem melhoria. Parando.")
            break

    t_total = time.time() - t_inicio
    print(f"\n  Melhor Val F1 (macro): {best_val_f1:.4f} | Tempo total: {t_total:.1f}s")

    # Restaura os melhores pesos
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ─── Avaliação no conjunto de teste ──────────────────────────────────────────

@torch.no_grad()
def avaliar_teste(model, test_data: list, device: torch.device) -> dict:
    """Retorna métricas completas no conjunto de teste."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Treina as 4 variantes do GATClassifier.")
    parser.add_argument("--epochs",   type=int,   default=EPOCHS_DEFAULT)
    parser.add_argument("--lr",       type=float, default=LR_DEFAULT)
    parser.add_argument("--variante", type=str,   default="todas",
                        choices=list(VARIANTES.keys()) + ["todas"],
                        help="Qual variante treinar (padrão: todas).")
    parser.add_argument("--cpu",      action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 58)
    print("  GAT — Etapa 4: Treinamento das 4 Variantes")
    print("=" * 58)
    print(f"  Device:  {device}")
    if device.type == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
    print(f"  Épocas:  {args.epochs}")
    print(f"  LR:      {args.lr}")
    print()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] Carregando dados...")
    train_loader, val_loader, test_data = carregar_dados(device)

    variantes_a_treinar = (
        VARIANTES if args.variante == "todas"
        else {args.variante: VARIANTES[args.variante]}
    )

    print(f"\n[2/2] Treinando {len(variantes_a_treinar)} variante(s)...")

    resultados = {}
    for nome, config in variantes_a_treinar.items():
        model = treinar_variante(
            nome, config, train_loader, val_loader, device, args.epochs, args.lr
        )

        # Salvar pesos
        save_path = WEIGHTS_DIR / config["arquivo"]
        torch.save(model.state_dict(), save_path)
        print(f"  [OK] Pesos salvos -> {save_path}")

        # Avaliar no teste
        metricas = avaliar_teste(model, test_data, device)
        resultados[nome] = metricas
        print(f"  Teste - Acc: {metricas['accuracy']:.4f} | "
              f"P: {metricas['precision']:.4f} | R: {metricas['recall']:.4f} | "
              f"F1: {metricas['f1']:.4f}")

    # Relatório final
    print(f"\n{'='*58}")
    print("  RESULTADO FINAL — Métricas no Conjunto de Teste")
    print(f"{'='*58}")
    print(f"  {'Variante':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*50}")
    for nome, m in resultados.items():
        print(f"  {nome:<22} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f}")

    print(f"\n  Pesos salvos em: {WEIGHTS_DIR}")
    print("\n[OK] Concluido! Proximo passo: python 04_comparar_gcn_gat.py")


if __name__ == "__main__":
    main()
