"""
gerar_folds.py
--------------
Instancia StratifiedKFold(n_splits=10) UMA vez sobre o FakeNewsNet e salva os
indices em data/folds_fnn.pt. Todos os modelos (baseline textual, GNN bugado,
GNN Pos-Full, variantes) devem consumir exatamente os mesmos folds para que
ttest_rel pareie a comparacao (Erro 4, Fase 2B.1).

Uso:
  python gerar_folds.py
"""

import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_DIR   = Path(__file__).resolve().parent / "data"
N_SPLITS   = 10
RANDOM_SEED = 42


def carregar_grafos() -> list:
    grafos = []
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"fakenewsnet_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} nao existe. Rode primeiro:\n"
                f"  python 00_construir_grafos_fakenewsnet.py"
            )
        grafos += torch.load(path, weights_only=False)
    return grafos


def iterar_folds(folds_path: Path = None):
    """
    Generator que produz (fold_idx, train_idx, test_idx) — reutilizavel por
    baseline textual (2A.1), benchmark contrastivo (4.1) e testes de significancia.
    """
    if folds_path is None:
        folds_path = DATA_DIR / "folds_fnn.pt"
    folds = torch.load(folds_path, weights_only=False)
    for f in folds:
        yield f["fold_idx"], f["train_idx"], f["test_idx"]


def main():
    print("=" * 62)
    print(f"  Gerando {N_SPLITS}-fold estratificado (seed={RANDOM_SEED})")
    print("=" * 62)

    grafos = carregar_grafos()
    y = [g.y.item() for g in grafos]
    n_fake = sum(1 for v in y if v == 0)
    print(f"  Total: {len(grafos):,} grafos ({n_fake} fake, {len(grafos)-n_fake} real)")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(grafos)), y)):
        fakes_test = sum(1 for i in test_idx if y[i] == 0)
        print(f"  fold {fold_idx:2d} | train={len(train_idx):3d} | test={len(test_idx):3d} "
              f"(fake={fakes_test}, real={len(test_idx)-fakes_test})")
        folds.append({
            "fold_idx":  fold_idx,
            "train_idx": train_idx.tolist(),
            "test_idx":  test_idx.tolist(),
        })

    out_path = DATA_DIR / "folds_fnn.pt"
    torch.save(folds, out_path)

    # Sanidade: sem sobreposicao e cobertura total
    from itertools import chain
    soma_test = sum(len(f["test_idx"]) for f in folds)
    union_test = set(chain.from_iterable(f["test_idx"] for f in folds))
    assert soma_test == len(grafos), f"Soma dos testes ({soma_test}) != total ({len(grafos)})"
    assert len(union_test) == len(grafos), "Folds de teste se sobrepoem"

    print(f"\n[OK] {len(folds)} folds salvos em {out_path}")
    print(f"     Cobertura de teste: {len(union_test)}/{len(grafos)} (sem sobreposicao)")


if __name__ == "__main__":
    main()
