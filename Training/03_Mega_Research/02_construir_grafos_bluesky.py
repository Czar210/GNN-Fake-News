"""
02_construir_grafos_bluesky.py
------------------------------
Constroi grafos PyG do Bluesky com features posicionais (Erro 1, Fase 3.1).

Estrutura por grafo:
  - Nó 0:     post raiz (BERT do texto)
  - Nós 1..N: reposts (cada um filho da raiz)
  - Features: 768 BERT + 3 posicionais [is_root, grau_norm, pos]  -> 771 dims

Variantes (--feature-variant):
  - full:     [BERT, is_root, grau_norm, pos]
  - pos-min:  [BERT, is_root, 0,         pos]
  - pos-grau: [BERT, is_root, grau_norm, 0]
  - notext:   [is_root, grau_norm, pos]                  -> 3 dims (sem BERT)

Filtragem:
  --min-reposts N   descarta grafos com menos de N reposts (default: 2)
                    Recomendado >= 2 para evitar grafos degenerados (1 no).

Saida:
  data/bluesky_<suffix>/bluesky_{train,val,test}.pt

Uso:
  python 02_construir_grafos_bluesky.py --feature-variant full --output-suffix posfull
  python 02_construir_grafos_bluesky.py --feature-variant notext --output-suffix notext --min-reposts 2
"""

import argparse
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
FEATURE_VARIANTS  = ("full", "pos-min", "pos-grau", "notext")
MIN_REPOSTS_DEFAULT = 2   # descarta grafos com poucos reposts (evita degeneracao)


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

def _construir_grafo_posicional(
    x_bert: torch.Tensor,
    num_filhos: int,
    max_nos: int,
    N_max_global: int,
    feature_variant: str,
) -> Data:
    """
    Constroi grafo em estrela com features posicionais (Erro 1, Fase 3.1).

    Layout dos 3 ultimos dims (apos BERT 768):
      raiz:    [is_root=1, grau_norm=N/N_max, pos=0]
      filho i: [is_root=0, grau_norm=0,        pos=i/num_filhos]

    feature_variant define quais dims posicionais ficam ativas; "notext" remove BERT.
    """
    num_filhos = min(num_filhos, max_nos - 1)
    num_nos    = 1 + num_filhos

    usa_grau = feature_variant != "pos-min"
    usa_pos  = feature_variant != "pos-grau"
    inclui_bert = feature_variant != "notext"

    grau_raiz = (num_filhos / max(N_max_global, 1)) if usa_grau else 0.0
    posicionais_raiz = torch.tensor([1.0, grau_raiz, 0.0], dtype=torch.float)

    if inclui_bert:
        x_raiz = torch.cat([x_bert, posicionais_raiz])  # [771]
    else:
        x_raiz = posicionais_raiz                        # [3]

    linhas = [x_raiz]
    for i in range(1, num_filhos + 1):
        pos_i = (i / num_filhos) if (usa_pos and num_filhos > 0) else 0.0
        posicionais_filho = torch.tensor([0.0, 0.0, pos_i], dtype=torch.float)
        if inclui_bert:
            linhas.append(torch.cat([x_bert, posicionais_filho]))
        else:
            linhas.append(posicionais_filho)

    x = torch.stack(linhas)

    if num_filhos > 0:
        src = torch.zeros(num_filhos, dtype=torch.long)
        dst = torch.arange(1, 1 + num_filhos, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Sanidade contra Erro 1 (se variante usa pos, esperamos num_nos vetores unicos)
    n_unique = x.unique(dim=0).shape[0]
    n_esperado = num_nos if usa_pos else min(num_nos, 2)
    assert n_unique >= min(n_esperado, 3) or num_nos < 2, (
        f"Features degeneradas: variant={feature_variant}, num_nos={num_nos}, unique={n_unique}"
    )
    return Data(x=x, edge_index=edge_index)


def construir_grafos(
    df_posts: pd.DataFrame,
    df_reposts: pd.DataFrame,
    embeddings: list,
    max_nos: int,
    min_reposts: int,
    N_max_global: int,
    feature_variant: str,
) -> list:
    if not df_reposts.empty and "post_id" in df_reposts.columns:
        reposts_idx = df_reposts.groupby("post_id")
    else:
        reposts_idx = None

    grafos, descartados = [], 0
    for i, (_, row) in enumerate(tqdm(df_posts.iterrows(), total=len(df_posts),
                                       desc="  Construindo grafos")):
        pid = str(row.get("post_id", ""))
        if reposts_idx is not None and pid in reposts_idx.groups:
            reposts_do_post = reposts_idx.get_group(pid)
            num_filhos_raw = len(reposts_do_post)
        else:
            num_filhos_raw = 0

        if num_filhos_raw < min_reposts:
            descartados += 1
            continue

        x_bert = torch.tensor(embeddings[i], dtype=torch.float)
        label  = int(row.get("label", 0))
        grafo = _construir_grafo_posicional(x_bert, num_filhos_raw, max_nos,
                                             N_max_global, feature_variant)
        grafo.y = torch.tensor([label], dtype=torch.long)
        grafos.append(grafo)

    print(f"  Grafos construidos: {len(grafos):,} | Descartados (<{min_reposts} reposts): {descartados}")
    return grafos


# ─── Partição e salvamento ────────────────────────────────────────────────────

def split_indices(n: int) -> tuple:
    import random
    rng = random.Random(RANDOM_SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    n_tr  = int(n * 0.60)
    n_val = int(n * 0.20)
    return indices[:n_tr], indices[n_tr:n_tr+n_val], indices[n_tr+n_val:]


def particionar_e_salvar(grafos: list, output_dir: Path) -> None:
    train_idx, val_idx, test_idx = split_indices(len(grafos))
    splits = {
        "bluesky_train": [grafos[i] for i in train_idx],
        "bluesky_val":   [grafos[i] for i in val_idx],
        "bluesky_test":  [grafos[i] for i in test_idx],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for nome, subset in splits.items():
        path = output_dir / f"{nome}.pt"
        torch.save(subset, path)
        fakes = sum(1 for g in subset if g.y.item() == 1)
        nos_med = sum(g.num_nodes for g in subset) / max(len(subset), 1)
        print(f"  [OK] {nome}.pt - {len(subset):,} grafos | "
              f"fakes={fakes} ({100*fakes/max(len(subset),1):.1f}%) | nos_med={nos_med:.1f}")
    print(f"\n  Arquivos salvos em: {output_dir}")


# ─── Relatório de sanidade ────────────────────────────────────────────────────

def verificar_grafos(grafos: list, esperado_dim: int, nome: str = "") -> None:
    if not grafos:
        print(f"  [WARN] {nome} sem grafos.")
        return
    nos    = [g.num_nodes for g in grafos]
    arestas = [g.num_edges for g in grafos]
    labels  = [g.y.item() for g in grafos]
    fakes   = sum(l == 1 for l in labels)

    print(f"  {nome} - {len(grafos):,} grafos | "
          f"nós medios={sum(nos)/len(nos):.1f} | "
          f"arestas medias={sum(arestas)/len(arestas):.1f} | "
          f"fakes={fakes} ({100*fakes/len(grafos):.1f}%)")

    for i, g in enumerate(grafos[:5]):
        assert g.x.shape == (g.num_nodes, esperado_dim), \
            f"Grafo {i} com x.shape={g.x.shape} inesperado (esperado [{g.num_nodes}, {esperado_dim}])"
    print(f"  [OK] dim por no = {esperado_dim} (5 primeiros grafos)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Constroi grafos PyG do Bluesky com features posicionais.")
    parser.add_argument("--max-posts",       type=int, default=None,
                        help="Limitar posts processados (para testes).")
    parser.add_argument("--batch-bert",      type=int, default=128)
    parser.add_argument("--max-nos",         type=int, default=MAX_NOS_POR_GRAFO)
    parser.add_argument("--min-reposts",     type=int, default=MIN_REPOSTS_DEFAULT,
                        help=f"Descarta grafos com menos de N reposts (default: {MIN_REPOSTS_DEFAULT}).")
    parser.add_argument("--feature-variant", type=str, default="full",
                        choices=list(FEATURE_VARIANTS))
    parser.add_argument("--output-suffix",   type=str, default=None,
                        help="Sufixo do diretorio em data/. Ex: posfull, notext.")
    parser.add_argument("--cpu",             action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUTPUT_DIR / f"bluesky_{args.output_suffix}" if args.output_suffix else OUTPUT_DIR
    esperado_dim = 3 if args.feature_variant == "notext" else 771

    print("=" * 62)
    print("  Bluesky -- Construcao de Grafos com Features Posicionais")
    print("=" * 62)
    print(f"  Device BERT:     {device}")
    print(f"  Min reposts:     {args.min_reposts}  (descartar abaixo)")
    print(f"  Max nos/grafo:   {args.max_nos}")
    print(f"  Feature variant: {args.feature_variant}  (dim por no = {esperado_dim})")
    print(f"  Output dir:      {out_dir}")
    print()

    if not POSTS_CSV.exists():
        print(f"[ERRO] {POSTS_CSV} nao encontrado. Rode 01_baixar_dataset_hf.py.")
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
    print(f"  Labels: {fakes} fakes ({100*fakes/len(df_posts):.2f}%) / {len(df_posts)-fakes} reais")

    # Estatistica de reposts por post para informar N_max_global
    reposts_por_post = df_reposts.groupby("post_id").size() if not df_reposts.empty else pd.Series(dtype=int)
    qualifica = (reposts_por_post >= args.min_reposts).sum()
    print(f"  Posts qualificados (>= {args.min_reposts} reposts): {qualifica:,}")

    # ── FILTRAGEM AGRESSIVA ANTES DO BERT ────────────────────────────────
    # Bluesky tem 166k posts mas so ~994 sao qualificados. Gerar BERT em todos
    # estoura memoria (OOM no Windows). Filtramos primeiro.
    posts_qualificados_ids = set(reposts_por_post[reposts_por_post >= args.min_reposts].index.astype(str))
    df_posts["post_id"] = df_posts["post_id"].astype(str)
    antes = len(df_posts)
    df_posts = df_posts[df_posts["post_id"].isin(posts_qualificados_ids)].reset_index(drop=True)
    print(f"  Filtrado: {antes:,} -> {len(df_posts):,} posts qualificados (BERT so nestes)")

    if len(df_posts) == 0:
        print("[ERRO] Nenhum post qualificado. Reduza --min-reposts.")
        sys.exit(1)

    if args.output_suffix and (out_dir / "bluesky_train.pt").exists():
        resp = input(f"\n'{out_dir.name}/bluesky_train.pt' ja existe. Reprocessar? [s/N] ").strip().lower()
        if resp != "s":
            print("Cancelado.")
            return

    # ── BERT (cache em OUTPUT_DIR / _bert_bluesky_cache_minN.pt) ──────────
    cache_path = OUTPUT_DIR / f"_bert_bluesky_cache_min{args.min_reposts}.pt"
    if args.feature_variant != "notext":
        if cache_path.exists():
            cache = torch.load(cache_path, weights_only=False)
            if len(cache) == len(df_posts):
                print(f"\n[2/4] Cache BERT carregado ({cache_path.name}, {len(cache)} embs)")
                embeddings = cache
            else:
                print(f"\n[2/4] Cache size mismatch -- regerando ({len(cache)} != {len(df_posts)})")
                embeddings = gerar_embeddings_batch(df_posts["texto"].tolist(),
                                                    batch_size=args.batch_bert, device=device)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(embeddings, cache_path)
        else:
            print(f"\n[2/4] Gerando embeddings BERT em {len(df_posts):,} posts qualificados...")
            embeddings = gerar_embeddings_batch(df_posts["texto"].tolist(),
                                                batch_size=args.batch_bert, device=device)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, cache_path)
            print(f"  [OK] cache em {cache_path.name}")
    else:
        # notext nao precisa de BERT, mas gerar zeros para nao quebrar interface
        print("\n[2/4] feature_variant=notext -- pulando BERT, usando zeros placeholder")
        embeddings = [[0.0] * 768 for _ in range(len(df_posts))]

    # ── N_max_global (somente sobre o subset filtrado) ────────────────────
    qualificados = reposts_por_post[reposts_por_post >= args.min_reposts]
    N_max_global = int(min(qualificados.max() if len(qualificados) > 0 else 1,
                           args.max_nos - 1))
    print(f"  N_max_global (capped a max_nos-1): {N_max_global}")

    # ── Construir grafos ──────────────────────────────────────────────────
    print(f"\n[3/4] Construindo grafos (variant={args.feature_variant})...")
    grafos = construir_grafos(df_posts, df_reposts, embeddings, args.max_nos,
                              args.min_reposts, N_max_global, args.feature_variant)

    if not grafos:
        print("[ERRO] Nenhum grafo construido. Reduza --min-reposts.")
        sys.exit(1)

    print("\n  Verificando grafos...")
    verificar_grafos(grafos, esperado_dim, "Total")

    print(f"\n[4/4] Particionando (60/20/20) e salvando em {out_dir}...")
    particionar_e_salvar(grafos, out_dir)

    print("\n[OK] Concluido!")


if __name__ == "__main__":
    main()
