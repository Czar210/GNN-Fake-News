"""
00_construir_grafos_fakenewsnet.py
----------------------------------
Constroi grafos de propagacao a partir do FakeNewsNet (PolitiFact).

Fonte: https://github.com/KaiDMML/FakeNewsNet
CSVs: politifact_fake.csv, politifact_real.csv

Estrutura de grafo produzida (equivalente ao UPFD):
  - No 0:     artigo de noticia (raiz) — features = BERT do titulo
  - Nos 1..N: usuarios que tweettaram/retweetaram (filhos)
              features = BERT do titulo (mesmo embedding da raiz, pois
              nao temos os tweets individuais — mesma estrategia do Bluesky)
  - Arestas:  estrela plana (raiz -> cada filho)
  - Label:    0 = Fake, 1 = Real

Saida:
  Training/03_Mega_Research/data/fakenewsnet_train.pt
  Training/03_Mega_Research/data/fakenewsnet_val.pt
  Training/03_Mega_Research/data/fakenewsnet_test.pt

Uso:
  python 00_construir_grafos_fakenewsnet.py
  python 00_construir_grafos_fakenewsnet.py --min-tweets 5 --max-nos 100
  python 00_construir_grafos_fakenewsnet.py --cpu
"""

import argparse
import io
import os
import random
import sys
import urllib.request
import ssl
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ       = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

FNN_BASE = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
FAKE_URL = f"{FNN_BASE}/politifact_fake.csv"
REAL_URL = f"{FNN_BASE}/politifact_real.csv"

RANDOM_SEED = 42
MAX_NOS_DEFAULT = 100   # limite de nos por grafo
MIN_TWEETS      = 2     # grafos com menos tweets sao descartados


# ─── Download ─────────────────────────────────────────────────────────────────

def _fetch_csv(url: str) -> pd.DataFrame:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r   = urllib.request.urlopen(req, timeout=60, context=ctx)
    return pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="replace")))


def carregar_fakenewsnet() -> pd.DataFrame:
    print("  Baixando FakeNewsNet PolitiFact (fake + real)...")
    df_fake = _fetch_csv(FAKE_URL)
    df_real = _fetch_csv(REAL_URL)

    df_fake["label"] = 0   # 0 = Fake (convencao UPFD)
    df_real["label"] = 1   # 1 = Real

    df = pd.concat([df_fake, df_real], ignore_index=True)
    df["tweet_ids"] = df["tweet_ids"].fillna("").astype(str)
    df["title"]     = df["title"].fillna("").astype(str)

    print(f"  Total: {len(df):,} artigos ({len(df_fake):,} fake + {len(df_real):,} real)")
    return df


# ─── Embeddings BERT ──────────────────────────────────────────────────────────

def gerar_embeddings(textos: list, batch_size: int, device: str) -> list:
    from sentence_transformers import SentenceTransformer
    print(f"  Carregando BERT (paraphrase-multilingual-mpnet-base-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
    print(f"  Gerando {len(textos):,} embeddings...")
    embs = model.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.tolist()


# ─── Construcao de grafos ──────────────────────────────────────────────────────

def construir_grafo(embedding: list, tweet_ids_str: str, label: int, max_nos: int) -> Data:
    """
    Constroi um grafo de propagacao em estrela para um artigo.

    No 0 = raiz (artigo)
    Nos 1..N = usuarios que tweetaram (sem features proprias — recebem embedding da raiz)
    """
    x_raiz = torch.tensor(embedding, dtype=torch.float)

    # Parsear tweet_ids (tab-separado)
    ids = [t.strip() for t in tweet_ids_str.split("\t") if t.strip()]
    num_filhos = min(len(ids), max_nos - 1)
    num_nos    = 1 + num_filhos

    # Todos os nos recebem o embedding do artigo (mesmo esquema do Bluesky)
    x = torch.stack([x_raiz] * num_nos)

    if num_filhos > 0:
        src = torch.zeros(num_filhos, dtype=torch.long)
        dst = torch.arange(1, 1 + num_filhos, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )


def construir_todos(df: pd.DataFrame, embeddings: list,
                    min_tweets: int, max_nos: int) -> list:
    grafos = []
    descartados = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                      desc="  Construindo grafos")):
        ids = [t.strip() for t in str(row["tweet_ids"]).split("\t") if t.strip()]
        if len(ids) < min_tweets:
            descartados += 1
            continue

        grafo = construir_grafo(embeddings[i], row["tweet_ids"], int(row["label"]), max_nos)
        grafos.append(grafo)

    print(f"  Grafos construidos: {len(grafos):,} | Descartados (<{min_tweets} tweets): {descartados}")
    return grafos


# ─── Particionar e salvar ─────────────────────────────────────────────────────

def particionar_e_salvar(grafos: list, output_dir: Path, prefixo: str = "fakenewsnet") -> None:
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(grafos)))
    rng.shuffle(indices)

    n     = len(indices)
    n_tr  = int(n * 0.60)
    n_val = int(n * 0.20)

    splits = {
        f"{prefixo}_train": [grafos[i] for i in indices[:n_tr]],
        f"{prefixo}_val":   [grafos[i] for i in indices[n_tr:n_tr + n_val]],
        f"{prefixo}_test":  [grafos[i] for i in indices[n_tr + n_val:]],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for nome, subset in splits.items():
        fakes = sum(1 for g in subset if g.y.item() == 0)
        nos_med = sum(g.num_nodes for g in subset) / len(subset)
        ar_med  = sum(g.num_edges for g in subset) / len(subset)
        path = output_dir / f"{nome}.pt"
        torch.save(subset, path)
        print(f"  [OK] {nome}.pt — {len(subset):,} grafos "
              f"(fake={fakes}, real={len(subset)-fakes}) | "
              f"nos_med={nos_med:.1f} | arestas_med={ar_med:.1f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Constroi grafos FakeNewsNet PolitiFact compatíveis com UPFD.")
    parser.add_argument("--min-tweets",  type=int, default=MIN_TWEETS)
    parser.add_argument("--max-nos",     type=int, default=MAX_NOS_DEFAULT)
    parser.add_argument("--batch-bert",  type=int, default=128)
    parser.add_argument("--cpu",         action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  FakeNewsNet — Construcao de Grafos de Propagacao")
    print("=" * 62)
    print(f"  Device BERT:     {device}")
    print(f"  Min tweets:      {args.min_tweets}")
    print(f"  Max nos/grafo:   {args.max_nos}")
    print()

    # Verificar se ja existem
    trem_path = OUTPUT_DIR / "fakenewsnet_train.pt"
    if trem_path.exists():
        resp = input(f"'{trem_path.name}' ja existe. Reprocessar? [s/N] ").strip().lower()
        if resp != "s":
            print("Cancelado.")
            return

    # Carregar dados
    print("[1/4] Baixando FakeNewsNet PolitiFact...")
    df = carregar_fakenewsnet()

    # Estatisticas de tweets por artigo
    tweet_counts = df["tweet_ids"].apply(
        lambda s: len([t for t in str(s).split("\t") if t.strip()])
    )
    print(f"\n  Distribuicao de tweets por artigo:")
    print(f"    min={tweet_counts.min()} | max={tweet_counts.max()} | "
          f"media={tweet_counts.mean():.1f} | mediana={tweet_counts.median():.0f}")
    print(f"    Artigos com >= {args.min_tweets} tweets: "
          f"{(tweet_counts >= args.min_tweets).sum():,}")
    print()

    # Gerar embeddings BERT dos titulos
    print("[2/4] Gerando embeddings BERT dos titulos...")
    embeddings = gerar_embeddings(
        df["title"].tolist(),
        batch_size=args.batch_bert,
        device=device,
    )

    # Construir grafos
    print("\n[3/4] Construindo grafos de propagacao...")
    grafos = construir_todos(df, embeddings, args.min_tweets, args.max_nos)

    if not grafos:
        print("[ERRO] Nenhum grafo construido. Reduza --min-tweets.")
        sys.exit(1)

    # Verificar balanceamento
    fakes_total = sum(1 for g in grafos if g.y.item() == 0)
    print(f"\n  Balanceamento total: {fakes_total} fake | "
          f"{len(grafos)-fakes_total} real "
          f"({100*fakes_total/len(grafos):.1f}% fake)")

    # Distribuicao de tamanhos
    nos = [g.num_nodes for g in grafos]
    print(f"  Nos por grafo: min={min(nos)} max={max(nos)} media={sum(nos)/len(nos):.1f}")

    # Salvar
    print("\n[4/4] Particionando (60/20/20) e salvando...")
    particionar_e_salvar(grafos, OUTPUT_DIR)

    print("\n[OK] Concluido! Proximo passo:")
    print("     python 07_upfd_benchmark_triplo.py --dataset fakenewsnet")


if __name__ == "__main__":
    main()
