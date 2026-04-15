"""
02_construir_grafos_bluesky.py
------------------------------
Lê os CSVs produzidos por 01_baixar_dataset_hf.py e constrói grafos
PyTorch Geometric prontos para treinamento do GAT (e do GCN baseline).

Estrutura de grafo produzida (alinhada com a API atualizada):
  - Nó 0 = post raiz (texto original)
  - Nós 1..N = reposts/replies do post
  - Arestas = hierarquia real de propagação (usando reply_to quando disponível)
    → Fallback: estrela plana (raiz → todos)

Saída:
  Training/03_Mega_Research/data/grafos_bluesky_train.pt
  Training/03_Mega_Research/data/grafos_bluesky_val.pt
  Training/03_Mega_Research/data/grafos_bluesky_test.pt

Uso:
  python 02_construir_grafos_bluesky.py
  python 02_construir_grafos_bluesky.py --max-posts 5000 --batch-bert 128
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ       = Path(__file__).resolve().parent.parent.parent  # GNN-Fake-News/
OUTPUT_DIR = Path(__file__).resolve().parent / "data"       # 03_Mega_Research/data/
RAW_DIR    = RAIZ / "Training" / "01_BlueSky_Pipe" / "data" / "raw"

sys.path.append(str(RAIZ / "Training" / "01_BlueSky_Pipe" / "src"))

POSTS_CSV   = RAW_DIR / "posts_coletados.csv"
REPOSTS_CSV = RAW_DIR / "reposts_coletados.csv"

MAX_NOS_POR_GRAFO = 50   # limita grafos muito grandes (mais rápido no treino)
RANDOM_SEED       = 42


# ─── Embedding BERT em batch ──────────────────────────────────────────────────

def gerar_embeddings_batch(textos: list, batch_size: int = 64, device: str = "cuda") -> list:
    """
    Gera embeddings BERT para uma lista de textos usando sentence-transformers.
    Retorna lista de numpy arrays (768-dim).
    Usa GPU se disponível.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Carregando modelo BERT (paraphrase-multilingual-mpnet-base-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

    print(f"  Gerando {len(textos):,} embeddings (batch={batch_size}, device={device})...")
    embeddings = model.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.tolist()


# ─── Construção de grafos ─────────────────────────────────────────────────────

def _construir_grafo_hierarquico(
    post_id: str,
    x_raiz: torch.Tensor,
    reposts_do_post: pd.DataFrame,
    max_nos: int,
) -> Data:
    """
    Constrói um grafo de propagação hierárquico para um post.

    Estrutura:
      - Nó 0 = raiz (post original)
      - Nós 1..N = reposts (cada repost é um nó filho da raiz)
      - Arestas: raiz → cada repost (estrela plana, pois os CSVs de repost
        não têm hierarquia aninhada — somente a API tem a árvore completa)

    Nota: A API (main.py) agora usa _count_thread_nodes() para hierarquia
    real. Nos CSVs de treinamento só temos 1 nível de reposts, então
    usamos estrela plana — isso é consistente com os scripts GCN existentes.
    """
    num_filhos = min(len(reposts_do_post), max_nos - 1)
    num_nos    = 1 + num_filhos

    # Todos os nós recebem o embedding do post raiz (padrão do pipeline)
    x = torch.stack([x_raiz] * num_nos)

    if num_filhos > 0:
        src = torch.zeros(num_filhos, dtype=torch.long)               # [0,0,...,0]
        dst = torch.arange(1, 1 + num_filhos, dtype=torch.long)       # [1,2,...,N]
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def construir_grafos(
    df_posts: pd.DataFrame,
    df_reposts: pd.DataFrame,
    embeddings: list,
    max_nos: int,
) -> list:
    """
    Monta a lista de grafos Data do PyG.
    """
    # Índice de reposts por post_id (lookup O(1))
    if not df_reposts.empty and "post_id" in df_reposts.columns:
        reposts_idx = df_reposts.groupby("post_id")
    else:
        reposts_idx = None

    grafos = []
    for i, (_, row) in enumerate(tqdm(df_posts.iterrows(), total=len(df_posts),
                                       desc="  Construindo grafos")):
        x_raiz = torch.tensor(embeddings[i], dtype=torch.float)
        label  = int(row.get("label", 0))
        pid    = str(row.get("post_id", ""))

        if reposts_idx is not None and pid in reposts_idx.groups:
            reposts_do_post = reposts_idx.get_group(pid)
        else:
            reposts_do_post = pd.DataFrame()

        grafo = _construir_grafo_hierarquico(pid, x_raiz, reposts_do_post, max_nos)
        grafo.y = torch.tensor([label], dtype=torch.long)
        grafos.append(grafo)

    return grafos


# ─── Partição e salvamento ────────────────────────────────────────────────────

def particionar_e_salvar(grafos: list, output_dir: Path) -> None:
    """
    Particiona em 60/20/20 e salva como .pt.
    """
    import random
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(grafos)))
    rng.shuffle(indices)

    n     = len(indices)
    n_tr  = int(n * 0.60)
    n_val = int(n * 0.20)

    idx_train = indices[:n_tr]
    idx_val   = indices[n_tr:n_tr + n_val]
    idx_test  = indices[n_tr + n_val:]

    splits = {
        "grafos_bluesky_train": [grafos[i] for i in idx_train],
        "grafos_bluesky_val":   [grafos[i] for i in idx_val],
        "grafos_bluesky_test":  [grafos[i] for i in idx_test],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for nome, subset in splits.items():
        path = output_dir / f"{nome}.pt"
        torch.save(subset, path)
        fakes = sum(1 for g in subset if g.y.item() == 1)
        print(f"  [OK] {nome}.pt - {len(subset):,} grafos "
              f"({fakes} fakes / {len(subset)-fakes} reais)")

    print(f"\n  Arquivos salvos em: {output_dir}")


# ─── Relatório de sanidade ────────────────────────────────────────────────────

def verificar_grafos(grafos: list, nome: str = "") -> None:
    if not grafos:
        return
    nos    = [g.num_nodes for g in grafos]
    arestas = [g.num_edges for g in grafos]
    labels  = [g.y.item() for g in grafos]
    fakes   = sum(l == 1 for l in labels)

    print(f"  {nome} - {len(grafos):,} grafos | "
          f"nós médios={sum(nos)/len(nos):.1f} | "
          f"arestas médias={sum(arestas)/len(arestas):.1f} | "
          f"fakes={fakes} ({100*fakes/len(grafos):.1f}%)")

    # Verifica consistência: x deve ter shape [num_nodes, 768]
    for i, g in enumerate(grafos[:5]):
        assert g.x.shape == (g.num_nodes, 768), \
            f"Grafo {i} com x.shape={g.x.shape} inesperado (esperado [{g.num_nodes}, 768])"
    print("  [OK] Verificacao de sanidade OK (5 primeiros grafos)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Constrói grafos PyG do dataset Bluesky.")
    parser.add_argument("--max-posts",  type=int, default=None,
                        help="Limitar número de posts (para testes rápidos).")
    parser.add_argument("--batch-bert", type=int, default=128,
                        help="Batch size para geração de embeddings BERT (padrão: 128).")
    parser.add_argument("--max-nos",    type=int, default=MAX_NOS_POR_GRAFO,
                        help=f"Máximo de nós por grafo (padrão: {MAX_NOS_POR_GRAFO}).")
    parser.add_argument("--cpu", action="store_true",
                        help="Forçar uso de CPU mesmo com GPU disponível.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 58)
    print("  GAT — Etapa 2: Construção dos Grafos Bluesky (PyG)")
    print("=" * 58)
    print(f"  Device BERT: {device}")
    print(f"  Max nós/grafo: {args.max_nos}")
    if args.max_posts:
        print(f"  Posts limitados a: {args.max_posts:,}")
    print()

    # ── Verificar dependências ─────────────────────────────────────────────
    if not POSTS_CSV.exists():
        print(f"[ERRO] {POSTS_CSV} não encontrado.")
        print("       Execute primeiro: python 01_baixar_dataset_hf.py")
        sys.exit(1)

    # ── Carregar CSVs ─────────────────────────────────────────────────────
    print("[1/4] Carregando CSVs...")
    df_posts  = pd.read_csv(POSTS_CSV)
    df_reposts = pd.read_csv(REPOSTS_CSV) if REPOSTS_CSV.exists() else pd.DataFrame()

    df_posts["texto"] = df_posts["texto"].fillna("").astype(str)
    df_posts["label"] = df_posts["label"].fillna(0).astype(int)

    if args.max_posts:
        df_posts = df_posts.head(args.max_posts)

    print(f"  Posts: {len(df_posts):,}  |  Reposts: {len(df_reposts):,}")
    fakes = int(df_posts["label"].sum())
    print(f"  Labels: {fakes} fakes ({100*fakes/len(df_posts):.1f}%) / "
          f"{len(df_posts)-fakes} reais")

    # ── Verificar se grafos já existem ────────────────────────────────────
    trem_path = OUTPUT_DIR / "grafos_bluesky_train.pt"
    if trem_path.exists():
        resp = input(f"\n'{trem_path.name}' já existe. Reprocessar? [s/N] ").strip().lower()
        if resp != "s":
            print("Cancelado. Use os arquivos .pt existentes.")
            return

    # ── Gerar embeddings BERT ─────────────────────────────────────────────
    print("\n[2/4] Gerando embeddings BERT...")
    textos = df_posts["texto"].tolist()
    embeddings = gerar_embeddings_batch(textos, batch_size=args.batch_bert, device=device)

    # ── Construir grafos ──────────────────────────────────────────────────
    print("\n[3/4] Construindo grafos de propagação...")
    grafos = construir_grafos(df_posts, df_reposts, embeddings, args.max_nos)

    # ── Verificação rápida ────────────────────────────────────────────────
    print("\n  Verificando grafos construídos...")
    verificar_grafos(grafos, "Total")

    # ── Salvar ────────────────────────────────────────────────────────────
    print("\n[4/4] Particionando (60/20/20) e salvando...")
    particionar_e_salvar(grafos, OUTPUT_DIR)

    print("\n[OK] Concluido! Proximo passo: python 03_treinar_gat.py")


if __name__ == "__main__":
    main()
