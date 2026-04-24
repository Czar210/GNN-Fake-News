"""
00_construir_grafos_fakenewsnet.py
----------------------------------
Constroi grafos de propagacao a partir do FakeNewsNet (PolitiFact).

Fonte: https://github.com/KaiDMML/FakeNewsNet
CSVs: politifact_fake.csv, politifact_real.csv

Estrutura de grafo produzida (equivalente ao UPFD):
  - No 0:     artigo de noticia (raiz)
  - Nos 1..N: usuarios que tweettaram/retweetaram (filhos)
  - Arestas:  estrela plana (raiz -> cada filho)
  - Label:    0 = Fake, 1 = Real

Features nodais (Fase 3.1, Erro 1) -- 768 (BERT) + 3 = 771 dims:
  - is_root:    1 na raiz, 0 nos filhos
  - grau_norm:  N/N_max_global na raiz, 0 nos filhos
  - pos:        0 na raiz, i/num_filhos no filho i

Variantes (--feature-variant):
  - full:      [BERT, is_root, grau_norm, pos]
  - pos-min:   [BERT, is_root, 0, pos]              (grau zerado)
  - pos-grau:  [BERT, is_root, grau_norm, 0]       (pos zerada)

CAVEAT: o campo `tweet_ids` no CSV nao tem timestamp documentado. A
ordem usada para `pos=i/N` e a do CSV. Reportar como "ordem de aparicao"
nas conclusoes (Fase 5A.5), nao como "ordem cronologica".

Saida:
  Training/03_Mega_Research/data/fakenewsnet_<suffix>/fakenewsnet_train.pt
  Training/03_Mega_Research/data/fakenewsnet_<suffix>/fakenewsnet_val.pt
  Training/03_Mega_Research/data/fakenewsnet_<suffix>/fakenewsnet_test.pt

Uso:
  python 00_construir_grafos_fakenewsnet.py --feature-variant full --output-suffix posfull
  python 00_construir_grafos_fakenewsnet.py --feature-variant pos-min --output-suffix posmin
  python 00_construir_grafos_fakenewsnet.py --feature-variant pos-grau --output-suffix posgrau
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
FEATURE_VARIANTS = ("full", "pos-min", "pos-grau")


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

def construir_grafo(embedding: list, tweet_ids_str: str, label: int, max_nos: int,
                    N_max_global: int, feature_variant: str = "full") -> Data:
    """
    Constroi grafo de propagacao em estrela com features posicionais (Fase 3.1).

    No 0 (raiz):       [BERT(titulo) | is_root=1 | grau_norm=N/N_max_global | pos=0]
    Nos 1..N (filhos): [BERT(titulo) | is_root=0 | grau_norm=0              | pos=i/num_filhos]

    Variantes via `feature_variant`:
      - "full":     todos os 3 campos posicionais ativos
      - "pos-min":  grau_norm zerado (mantem is_root e pos)
      - "pos-grau": pos zerada (mantem is_root e grau_norm)
    """
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError(f"feature_variant invalido: {feature_variant!r}. "
                         f"Use um de {FEATURE_VARIANTS}.")

    x_bert = torch.tensor(embedding, dtype=torch.float)

    # Parsear tweet_ids (tab-separado). CAVEAT: ordem do CSV, nao cronologica.
    ids = [t.strip() for t in tweet_ids_str.split("\t") if t.strip()]
    num_filhos = min(len(ids), max_nos - 1)
    num_nos    = 1 + num_filhos

    # Mascaras das features posicionais por variante
    usa_grau = feature_variant != "pos-min"
    usa_pos  = feature_variant != "pos-grau"

    # Raiz
    grau_raiz = (num_filhos / N_max_global) if usa_grau else 0.0
    pos_raiz_extra = torch.tensor([1.0, grau_raiz, 0.0], dtype=torch.float)
    x_raiz = torch.cat([x_bert, pos_raiz_extra])           # [771]

    # Filhos
    linhas = [x_raiz]
    for i in range(1, num_filhos + 1):
        pos_filho = (i / num_filhos) if (usa_pos and num_filhos > 0) else 0.0
        pos_extra = torch.tensor([0.0, 0.0, pos_filho], dtype=torch.float)
        linhas.append(torch.cat([x_bert, pos_extra]))      # [771]

    x = torch.stack(linhas)                                 # [num_nos, 771]

    if num_filhos > 0:
        src = torch.zeros(num_filhos, dtype=torch.long)
        dst = torch.arange(1, 1 + num_filhos, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Sanidade contra o Erro 1 (features nodais identicas).
    # - variantes com `usa_pos`: cada filho tem pos diferente -> esperamos num_nos vetores unicos.
    # - "pos-grau" (sem pos): filhos sao identicos por design; basta raiz != filhos (2 unicos).
    n_unique  = x.unique(dim=0).shape[0]
    n_esperado = num_nos if usa_pos else min(num_nos, 2)
    assert n_unique >= min(n_esperado, 3) or num_nos < 2, (
        f"Features nodais degeneradas (label={label}, num_nos={num_nos}, "
        f"variant={feature_variant}, unicos={n_unique}, esperado>={n_esperado})"
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )


def construir_todos(df: pd.DataFrame, embeddings: list,
                    min_tweets: int, max_nos: int,
                    N_max_global: int, feature_variant: str) -> list:
    grafos = []
    descartados = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                      desc="  Construindo grafos")):
        ids = [t.strip() for t in str(row["tweet_ids"]).split("\t") if t.strip()]
        if len(ids) < min_tweets:
            descartados += 1
            continue

        grafo = construir_grafo(embeddings[i], row["tweet_ids"], int(row["label"]),
                                max_nos, N_max_global, feature_variant)
        grafos.append(grafo)

    print(f"  Grafos construidos: {len(grafos):,} | Descartados (<{min_tweets} tweets): {descartados}")
    return grafos


# ─── Particionar e salvar ─────────────────────────────────────────────────────

def split_indices(n: int) -> tuple:
    """Retorna (train_idx, val_idx, test_idx) com 60/20/20 e seed fixo."""
    rng = random.Random(RANDOM_SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    n_tr  = int(n * 0.60)
    n_val = int(n * 0.20)
    return (indices[:n_tr],
            indices[n_tr:n_tr + n_val],
            indices[n_tr + n_val:])


def particionar_e_salvar(grafos: list, output_dir: Path, prefixo: str = "fakenewsnet") -> None:
    train_idx, val_idx, test_idx = split_indices(len(grafos))

    splits = {
        f"{prefixo}_train": [grafos[i] for i in train_idx],
        f"{prefixo}_val":   [grafos[i] for i in val_idx],
        f"{prefixo}_test":  [grafos[i] for i in test_idx],
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
    parser.add_argument("--min-tweets",       type=int, default=MIN_TWEETS)
    parser.add_argument("--max-nos",          type=int, default=MAX_NOS_DEFAULT)
    parser.add_argument("--batch-bert",       type=int, default=128)
    parser.add_argument("--cpu",              action="store_true")
    parser.add_argument("--feature-variant",  type=str, default="full",
                        choices=list(FEATURE_VARIANTS),
                        help="Estrategia de features posicionais (Fase 3.1).")
    parser.add_argument("--output-suffix",    type=str, default=None,
                        help="Sufixo do diretorio de saida em data/. "
                             "Default: data/ (legacy bugado) -- USE um sufixo.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_suffix:
        out_dir = OUTPUT_DIR / f"fakenewsnet_{args.output_suffix}"
    else:
        out_dir = OUTPUT_DIR

    print("=" * 62)
    print("  FakeNewsNet — Construcao de Grafos de Propagacao")
    print("=" * 62)
    print(f"  Device BERT:        {device}")
    print(f"  Min tweets:         {args.min_tweets}")
    print(f"  Max nos/grafo:      {args.max_nos}")
    print(f"  Feature variant:    {args.feature_variant}")
    print(f"  Output dir:         {out_dir}")
    print()

    # Verificar se ja existem
    trem_path = out_dir / "fakenewsnet_train.pt"
    if trem_path.exists():
        resp = input(f"'{trem_path}' ja existe. Reprocessar? [s/N] ").strip().lower()
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

    # ── Calcular N_max_global SOMENTE sobre o treino (evita vazamento) ────────
    # Aplica filtro min_tweets primeiro para casar com o que vai virar grafo.
    df_filtrado = df[tweet_counts >= args.min_tweets].reset_index(drop=True)
    train_idx, _, _ = split_indices(len(df_filtrado))
    df_train = df_filtrado.iloc[train_idx]
    N_max_global = int(max(
        len([t for t in str(s).split("\t") if t.strip()])
        for s in df_train["tweet_ids"]
    ))
    # Cap pelo max_nos-1 (numero efetivo de filhos)
    N_max_global = min(N_max_global, args.max_nos - 1)
    print(f"  N_max_global (apenas treino, capped a max_nos-1): {N_max_global}")
    print()

    # Gerar embeddings BERT dos titulos (cache em disco para reuso entre variantes)
    cache_path = OUTPUT_DIR / "_bert_titulos_cache.pt"
    if cache_path.exists():
        print(f"[2/4] Carregando embeddings BERT do cache ({cache_path.name})...")
        cache = torch.load(cache_path, weights_only=False)
        # sanidade: tamanho bate com df?
        if len(cache) == len(df):
            embeddings = cache
        else:
            print(f"  [WARN] cache tem {len(cache)} embs, df tem {len(df)}. Regerando.")
            embeddings = gerar_embeddings(df["title"].tolist(),
                                          batch_size=args.batch_bert, device=device)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, cache_path)
    else:
        print("[2/4] Gerando embeddings BERT dos titulos (primeira vez)...")
        embeddings = gerar_embeddings(df["title"].tolist(),
                                      batch_size=args.batch_bert, device=device)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"  [OK] Cache salvo em {cache_path.name}")

    # Construir grafos
    print(f"\n[3/4] Construindo grafos (variant={args.feature_variant})...")
    grafos = construir_todos(df, embeddings, args.min_tweets, args.max_nos,
                              N_max_global, args.feature_variant)

    if not grafos:
        print("[ERRO] Nenhum grafo construido. Reduza --min-tweets.")
        sys.exit(1)

    # Sanidade da dimensao
    dim = grafos[0].x.shape[1]
    print(f"  Dimensao por no: {dim} (esperado: 771 = 768 BERT + 3 posicionais)")
    assert dim == 771, f"Dimensao inesperada: {dim}"

    # Verificar balanceamento
    fakes_total = sum(1 for g in grafos if g.y.item() == 0)
    print(f"\n  Balanceamento total: {fakes_total} fake | "
          f"{len(grafos)-fakes_total} real "
          f"({100*fakes_total/len(grafos):.1f}% fake)")

    # Distribuicao de tamanhos
    nos = [g.num_nodes for g in grafos]
    print(f"  Nos por grafo: min={min(nos)} max={max(nos)} media={sum(nos)/len(nos):.1f}")

    # Salvar
    print(f"\n[4/4] Particionando (60/20/20) e salvando em {out_dir}...")
    particionar_e_salvar(grafos, out_dir)

    print("\n[OK] Concluido! Proximo passo:")
    print(f"     python 07_upfd_benchmark_triplo.py --dataset fakenewsnet "
          f"--data-suffix {args.output_suffix or '<dir>'}")


if __name__ == "__main__":
    main()
