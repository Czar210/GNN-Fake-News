"""
01_baixar_dataset_hf.py
-----------------------
Faz o download seletivo do dataset Zaras210/bluesky-fake-news-dataset do
HuggingFace e produz dois CSVs compatíveis com o pipeline de treinamento:

  Training/01_BlueSky_Pipe/data/raw/posts_coletados.csv
  Training/01_BlueSky_Pipe/data/raw/reposts_coletados.csv

Estratégia de download (sem usar datasets.load_dataset — está quebrado para
este repo por colunas heterogêneas entre os CSVs):
  - graphs.tar.gz      → reposts_coletados.csv (arestas do grafo de prop.)
  - feed_posts/**      → posts_coletados.csv   (texto, metadados, labels)

Uso:
  python 01_baixar_dataset_hf.py
  python 01_baixar_dataset_hf.py --max-reposts 200000
  python 01_baixar_dataset_hf.py --dry-run        # lista arquivos, não baixa
"""

import argparse
import csv
import gzip
import json
import os
import random
import sys
import tarfile
from pathlib import Path

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ = Path(__file__).resolve().parent.parent.parent          # GNN-Fake-News/
CACHE_DIR = RAIZ / "Material" / "dados_bluesky" / "hf_cache"
OUTPUT_DIR = RAIZ / "Training" / "01_BlueSky_Pipe" / "data" / "raw"
OUTPUT_POSTS   = OUTPUT_DIR / "posts_coletados.csv"
OUTPUT_REPOSTS = OUTPUT_DIR / "reposts_coletados.csv"

DATASET_ID  = "Zaras210/bluesky-fake-news-dataset"
MAX_REPOSTS_DEFAULT = 300_000
RANDOM_SEED = 42

# ─── Heurística de rotulagem (mesma de extrator_amostra.py) ──────────────────

KEYWORDS_FAKE = [
    "conspiracy", "fake", "hoax", "mentira", "fraude", "scam",
    "desinforma", "fakenews", "fake news", "not true", "false claim",
    "misinformation", "disinformation",
]


def _label(texto: str, labels_api: list) -> int:
    """Retorna 1 (fake) ou 0 (real)."""
    for lbl in labels_api:
        val = lbl.get("val", "") if isinstance(lbl, dict) else str(lbl)
        if val in ("graphic-media", "porn", "spam", "!hide", "misleading"):
            return 1
    t = texto.lower()
    return 1 if any(k in t for k in KEYWORDS_FAKE) else 0


# ─── Listar arquivos do repositório HuggingFace ───────────────────────────────

def listar_arquivos_repo() -> list:
    """Retorna lista de paths de arquivos no repo HF."""
    try:
        from huggingface_hub import list_repo_files
        arquivos = list(list_repo_files(DATASET_ID, repo_type="dataset"))
        print(f"  {len(arquivos)} arquivos encontrados no repo.")
        return arquivos
    except Exception as e:
        print(f"[ERRO] Não foi possível listar arquivos do repo: {e}")
        sys.exit(1)


# ─── Download de arquivo individual ──────────────────────────────────────────

def baixar_arquivo(path_no_repo: str, force: bool = False) -> Path:
    """
    Baixa um arquivo do HF Hub para o cache local.
    Retorna o Path local do arquivo baixado.
    """
    from huggingface_hub import hf_hub_download
    local = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=path_no_repo,
            repo_type="dataset",
            cache_dir=str(CACHE_DIR),
            force_download=force,
        )
    )
    return local


# ─── Etapa 1A: Extrair posts dos feeds ───────────────────────────────────────

def _parse_jsonl_gz(fileobj) -> list:
    """Lê um .jsonl.gz e retorna lista de dicts de posts."""
    posts = []
    try:
        with gzip.open(fileobj, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                texto = obj.get("text", obj.get("texto", "")).strip()
                if not texto:
                    continue
                posts.append({
                    "post_id":      obj.get("post_id", obj.get("id", "")),
                    "autor":        obj.get("user_id", obj.get("author", "")),
                    "texto":        texto,
                    "data_criacao": obj.get("date", obj.get("created_at", "")),
                    "reposts":      int(obj.get("repost_count", 0) or 0),
                    "likes":        int(obj.get("like_count", 0) or 0),
                    "replies":      int(obj.get("reply_count", 0) or 0),
                    "reply_to":     obj.get("reply_to", "") or "",
                    "feed":         obj.get("feed", ""),
                    "label":        _label(texto, obj.get("labels") or []),
                })
    except Exception as e:
        print(f"    [WARN] Erro ao parsear JSONL.gz: {e}")
    return posts


def _parse_feed_csv_gz(local_path: Path, feed_name: str) -> list:
    """
    Tenta parsear CSV.gz de feed_posts/*.csv.gz.
    Formato esperado: colunas com nomes não-padronizados (post_id, user_id, ts).
    """
    posts = []
    try:
        with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return posts
            # Tenta detectar colunas por posição (feed, post_id, timestamp)
            for row in reader:
                if len(row) < 2:
                    continue
                post_id = row[1] if len(row) > 1 else ""
                posts.append({
                    "post_id":      post_id,
                    "autor":        "",
                    "texto":        f"[post bluesky {post_id}]",
                    "data_criacao": row[2] if len(row) > 2 else "",
                    "reposts":      0,
                    "likes":        0,
                    "replies":      0,
                    "reply_to":     "",
                    "feed":         feed_name,
                    "label":        0,
                })
    except Exception as e:
        print(f"    [WARN] Erro ao parsear CSV.gz {local_path.name}: {e}")
    return posts


def baixar_posts(arquivos_repo: list, force: bool = False) -> list:
    """
    Baixa e parseia todos os arquivos de feed_posts/.
    Prioriza .jsonl.gz; fallback para .csv.gz.
    """
    feed_files = [f for f in arquivos_repo if f.startswith("feed_posts/")
                  and not f.startswith("feed_posts_likes/")
                  and (f.endswith(".jsonl.gz") or f.endswith(".csv.gz") or f.endswith(".jsonl"))]

    if not feed_files:
        print("[WARN] Nenhum arquivo de posts encontrado em feed_posts/. Usando feed_posts_likes/.")
        feed_files = [f for f in arquivos_repo if f.startswith("feed_posts_likes/")
                      and f.endswith(".csv.gz")]

    print(f"\n[1/3] Baixando {len(feed_files)} arquivo(s) de posts...")
    todos_posts = []

    for fpath in sorted(feed_files):
        feed_name = Path(fpath).stem.replace(".jsonl", "").replace(".csv", "")
        print(f"  baixando {fpath}", end="", flush=True)
        try:
            local = baixar_arquivo(fpath, force=force)
            print(f"  ({local.stat().st_size // 1024} KB)")

            if fpath.endswith(".jsonl.gz"):
                novos = _parse_jsonl_gz(local.open("rb"))
            elif fpath.endswith(".csv.gz"):
                novos = _parse_feed_csv_gz(local, feed_name)
            else:
                with open(local, "r", encoding="utf-8", errors="replace") as f:
                    novos = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            texto = obj.get("text", "").strip()
                            if texto:
                                novos.append({
                                    "post_id": obj.get("post_id", ""),
                                    "autor": obj.get("user_id", ""),
                                    "texto": texto,
                                    "data_criacao": obj.get("date", ""),
                                    "reposts": 0, "likes": 0, "replies": 0,
                                    "reply_to": "", "feed": feed_name,
                                    "label": _label(texto, []),
                                })
                        except Exception:
                            pass

            # Adiciona feed_name se não preenchido
            for p in novos:
                if not p.get("feed"):
                    p["feed"] = feed_name
            todos_posts.extend(novos)
            print(f"    -> {len(novos)} posts")
        except Exception as e:
            print(f"\n    [WARN] Falhou: {e}")

    return todos_posts


# ─── Etapa 1B: Extrair reposts de graphs.tar.gz ──────────────────────────────

def baixar_reposts(max_linhas: int, force: bool = False) -> list:
    """
    Baixa graphs.tar.gz e extrai graphs/reposts.csv usando reservoir sampling.
    Formato das linhas: post_id,reposter_id,data_yyyymmdd (sem cabeçalho).
    """
    print(f"\n[2/3] Baixando graphs.tar.gz e extraindo até {max_linhas:,} reposts...")
    try:
        local = baixar_arquivo("graphs.tar.gz", force=force)
        print(f"  Arquivo: {local} ({local.stat().st_size // (1024*1024)} MB)")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar graphs.tar.gz: {e}")
        return []

    reposts = []
    random.seed(RANDOM_SEED)

    try:
        with gzip.open(local, "rb") as gz:
            with tarfile.open(fileobj=gz) as tar:
                # Procura graphs/reposts.csv dentro do tar
                membro = None
                for m in tar.getmembers():
                    if "reposts.csv" in m.name:
                        membro = m
                        break

                if membro is None:
                    print("[WARN] graphs/reposts.csv não encontrado no tar.")
                    return []

                fobj = tar.extractfile(membro)
                if fobj is None:
                    return []

                for i, linha_bytes in enumerate(fobj):
                    linha = linha_bytes.decode("utf-8", errors="replace").strip()
                    if not linha:
                        continue
                    partes = linha.split(",")
                    if len(partes) < 2:
                        continue
                    post_id    = partes[0]
                    reposter   = partes[1]
                    data       = partes[2] if len(partes) > 2 else ""

                    row = {
                        "post_id":        post_id,
                        "source":         reposter,
                        "target":         post_id,
                        "tipo_interacao": "repost",
                        "data":           data,
                    }

                    if i < max_linhas:
                        reposts.append(row)
                    else:
                        # Reservoir sampling
                        j = random.randint(0, i)
                        if j < max_linhas:
                            reposts[j] = row

                    if (i + 1) % 5_000_000 == 0:
                        print(f"    {i+1:,} linhas processadas...")

    except Exception as e:
        print(f"[ERRO] Ao extrair reposts: {e}")

    print(f"  -> {len(reposts):,} reposts extraídos.")
    return reposts


# ─── Salvar CSVs ─────────────────────────────────────────────────────────────

def salvar_posts(posts: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    campos = ["post_id", "autor", "texto", "data_criacao",
              "reposts", "likes", "replies", "reply_to", "feed", "label"]
    with open(OUTPUT_POSTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(posts)
    print(f"  [OK] Posts salvos: {len(posts):,} -> {OUTPUT_POSTS}")


def salvar_reposts(reposts: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    campos = ["post_id", "source", "target", "tipo_interacao", "data"]
    with open(OUTPUT_REPOSTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(reposts)
    print(f"  [OK] Reposts salvos: {len(reposts):,} -> {OUTPUT_REPOSTS}")


# ─── Relatório ────────────────────────────────────────────────────────────────

def relatorio(posts: list, reposts: list) -> None:
    total = len(posts)
    fakes = sum(1 for p in posts if p["label"] == 1)
    reais = total - fakes

    print("\n" + "─" * 55)
    print("  RELATÓRIO DA EXTRAÇÃO")
    print("-" * 55)
    print(f"  Posts totais:     {total:>8,}")
    print(f"  Labels reais (0): {reais:>8,}  ({100*reais/total:.1f}%)" if total else "")
    print(f"  Labels fake  (1): {fakes:>8,}  ({100*fakes/total:.1f}%)" if total else "")
    print(f"  Reposts:          {len(reposts):>8,}")
    if posts:
        feeds: dict = {}
        for p in posts:
            feeds[p.get("feed", "??")] = feeds.get(p.get("feed", "??"), 0) + 1
        print("\n  Posts por feed:")
        for feed, count in sorted(feeds.items(), key=lambda x: -x[1])[:10]:
            print(f"    {feed:<28} {count:>6,}")
    print("-" * 55)
    print("\n[OK] Concluido. Rode agora: python 02_construir_grafos_bluesky.py")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download dataset Bluesky do HuggingFace.")
    parser.add_argument("--max-reposts", type=int, default=MAX_REPOSTS_DEFAULT,
                        help=f"Máximo de reposts a amostrar (padrão: {MAX_REPOSTS_DEFAULT:,})")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescreve CSVs existentes sem perguntar.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas lista os arquivos do repo, sem baixar.")
    args = parser.parse_args()

    print("=" * 55)
    print("  GAT — Etapa 1: Download do Dataset Bluesky (HuggingFace)")
    print("=" * 55)
    print(f"  Dataset:    {DATASET_ID}")
    print(f"  Cache:      {CACHE_DIR}")
    print(f"  Saída:      {OUTPUT_DIR}")
    print(f"  MaxReposts: {args.max_reposts:,}")
    print()

    # Garante que huggingface_hub está disponível
    try:
        import huggingface_hub  # noqa
    except ImportError:
        print("[ERRO] huggingface_hub não encontrado. Execute: pip install huggingface_hub")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    arquivos = listar_arquivos_repo()

    if args.dry_run:
        print("\n--- Arquivos no repositório ---")
        for f in sorted(arquivos):
            print(f"  {f}")
        print(f"\nTotal: {len(arquivos)} arquivos. Nada baixado (--dry-run).")
        return

    # Se os CSVs já existem e têm conteúdo, pergunta se quer reprocessar
    if OUTPUT_POSTS.exists() and OUTPUT_POSTS.stat().st_size > 1000 and not args.force:
        resp = input(f"\n'{OUTPUT_POSTS.name}' ja existe. Reprocessar? [s/N] ").strip().lower()
        if resp != "s":
            print("Cancelado. Use os CSVs existentes ou apague-os manualmente.")
            return

    posts   = baixar_posts(arquivos, force=args.force)
    reposts = baixar_reposts(args.max_reposts, force=args.force)

    print("\n[3/3] Salvando CSVs...")
    salvar_posts(posts)
    salvar_reposts(reposts)

    relatorio(posts, reposts)


if __name__ == "__main__":
    main()
