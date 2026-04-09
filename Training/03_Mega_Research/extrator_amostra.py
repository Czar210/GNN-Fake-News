"""
extrator_amostra.py
-------------------
Lê o 14669616.zip em streaming (sem extrair tudo) e gera uma amostra
bem ajustada para o pipeline de treinamento.

Saída:
  Blue Sky/data/raw/posts_coletados.csv    — compatível com treinar_mega_dataset.py
  Blue Sky/data/raw/reposts_coletados.csv  — interações para construção dos grafos

Não extrai user_posts.tar.gz (18.5 GB) — não é necessário para treinar.
"""

import gzip
import json
import tarfile
import zipfile
import csv
import random
from pathlib import Path

# ─── Caminhos ─────────────────────────────────────────────────────────────────

RAIZ = Path(__file__).resolve().parent.parent
ZIP_PATH = RAIZ / "14669616.zip"
OUTPUT_DIR = RAIZ / "Training" / "01_BlueSky_Pipe" / "data" / "raw"
OUTPUT_POSTS = OUTPUT_DIR / "posts_coletados.csv"
OUTPUT_REPOSTS = OUTPUT_DIR / "reposts_coletados.csv"

# ─── Configuração da amostra ──────────────────────────────────────────────────

# Quantas interações pegar de graphs/reposts.csv (que tem 152 M linhas)
MAX_INTERACOES = 300_000

# Semente para reprodutibilidade
RANDOM_SEED = 42

# ─── Feeds que existem dentro de feed_posts.tar.gz ───────────────────────────

FEEDS_DE_NOTICIAS = {"News", "Science", "AcademicSky", "Political Science", "What's History"}
FEEDS_NEUTROS = {"Blacksky", "#UkrainianView", "GreenSky", "#Disability", "BookSky", "Game Dev"}


def _label_heuristica(texto: str, labels_api: list) -> int:
    """
    Retorna 1 (fake) ou 0 (real) com base em:
    1. Campo 'labels' do AT Protocol (mais confiável)
    2. Palavras-chave no texto
    3. Feed de origem
    """
    # Labels explícitas da plataforma
    if labels_api:
        for lbl in labels_api:
            val = lbl.get("val", "") if isinstance(lbl, dict) else str(lbl)
            if val in ("graphic-media", "porn", "spam", "!hide", "misleading"):
                return 1

    # Palavras-chave heurísticas
    keywords_fake = [
        "conspiracy", "fake", "hoax", "mentira", "fraude", "scam",
        "desinforma", "fakenews", "fake news", "not true", "false claim",
        "misinformation", "disinformation",
    ]
    texto_lower = texto.lower()
    if any(k in texto_lower for k in keywords_fake):
        return 1

    return 0


def extrair_posts(z: zipfile.ZipFile) -> list[dict]:
    """
    Extrai todos os posts de feed_posts.tar.gz (15 MB, já pequeno).
    Retorna lista de dicts prontos para o CSV.
    """
    posts = []

    with z.open("feed_posts.tar.gz") as raw:
        with gzip.open(raw) as gz:
            with tarfile.open(fileobj=gz) as tar:
                members = [m for m in tar.getmembers() if m.name.endswith(".jsonl.gz")]

                for member in members:
                    # Nome do feed: "feed_posts/News.jsonl.gz" → "News"
                    feed_name = Path(member.name).stem.replace(".jsonl", "")
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue

                    with gzip.open(fobj) as gf:
                        for line in gf:
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            texto = obj.get("text", "").strip()
                            if not texto:
                                continue

                            post_id = obj.get("post_id", "")
                            user_id = obj.get("user_id", "")
                            data = obj.get("date", "")
                            reposts = obj.get("repost_count", 0) or 0
                            likes = obj.get("like_count", 0) or 0
                            replies = obj.get("reply_count", 0) or 0
                            reply_to = obj.get("reply_to", "") or ""
                            labels_api = obj.get("labels") or []

                            label = _label_heuristica(texto, labels_api)

                            posts.append(
                                {
                                    "post_id": post_id,
                                    "autor": user_id,
                                    "texto": texto,
                                    "data_criacao": data,
                                    "reposts": reposts,
                                    "likes": likes,
                                    "replies": replies,
                                    "reply_to": reply_to,
                                    "feed": feed_name,
                                    "label": label,
                                }
                            )

    return posts


def extrair_interacoes(z: zipfile.ZipFile, max_linhas: int) -> list[dict]:
    """
    Lê graphs/reposts.csv de dentro de graphs.tar.gz em streaming.
    Formato: post_id, reposter_id, data_yyyymmdd  (sem cabeçalho)
    Retorna amostra de max_linhas linhas.
    """
    interacoes = []
    random.seed(RANDOM_SEED)

    # Reservoir sampling (k=max_linhas) para não carregar tudo na memória
    with z.open("graphs.tar.gz") as raw:
        with gzip.open(raw) as gz:
            with tarfile.open(fileobj=gz) as tar:
                fobj = tar.extractfile("graphs/reposts.csv")
                if fobj is None:
                    return interacoes

                for i, line_bytes in enumerate(fobj):
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) < 3:
                        continue

                    post_id, reposter_id, data = parts[0], parts[1], parts[2]

                    row = {
                        "post_id": post_id,
                        "source": reposter_id,
                        "target": post_id,
                        "tipo_interacao": "repost",
                        "data": data,
                    }

                    if i < max_linhas:
                        interacoes.append(row)
                    else:
                        # Reservoir sampling
                        j = random.randint(0, i)
                        if j < max_linhas:
                            interacoes[j] = row

    return interacoes


def salvar_posts(posts: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    campos = ["post_id", "autor", "texto", "data_criacao", "reposts", "likes", "replies", "reply_to", "feed", "label"]

    with open(OUTPUT_POSTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(posts)

    print(f"  Posts salvos: {len(posts):,} linhas -> {OUTPUT_POSTS}")


def salvar_interacoes(interacoes: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    campos = ["post_id", "source", "target", "tipo_interacao", "data"]

    with open(OUTPUT_REPOSTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(interacoes)

    print(f"  Interacoes salvas: {len(interacoes):,} linhas -> {OUTPUT_REPOSTS}")


def relatorio(posts: list[dict], interacoes: list[dict]) -> None:
    total = len(posts)
    fakes = sum(1 for p in posts if p["label"] == 1)
    reais = total - fakes

    print("\n--- Relatorio da Extracao -----------------------------------")
    print(f"  Posts totais:      {total:>8,}")
    print(f"  Labels reais (0):  {reais:>8,}  ({100*reais/total:.1f}%)")
    print(f"  Labels fake  (1):  {fakes:>8,}  ({100*fakes/total:.1f}%)")
    print(f"  Interacoes:        {len(interacoes):>8,}")

    feeds = {}
    for p in posts:
        feeds[p["feed"]] = feeds.get(p["feed"], 0) + 1
    print("\n  Posts por feed:")
    for feed, count in sorted(feeds.items(), key=lambda x: -x[1]):
        print(f"    {feed:<30} {count:>6,}")
    print("-------------------------------------------------------------")


def extrair(max_interacoes: int = MAX_INTERACOES) -> None:
    if not ZIP_PATH.exists():
        print(f"[ERRO] Arquivo não encontrado: {ZIP_PATH}")
        return

    print(f"Abrindo {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / 1e9:.1f} GB)...")
    print("  (Apenas feed_posts.tar.gz e graphs/reposts.csv serao lidos)")
    print()

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        print("[1/3] Extraindo posts dos feeds...")
        posts = extrair_posts(z)
        print(f"      {len(posts):,} posts carregados.")

        print(f"[2/3] Amostrando {max_interacoes:,} interacoes (reservoir sampling)...")
        interacoes = extrair_interacoes(z, max_interacoes)
        print(f"      {len(interacoes):,} interações selecionadas.")

        print("[3/3] Salvando CSVs...")
        salvar_posts(posts)
        salvar_interacoes(interacoes)

    relatorio(posts, interacoes)
    print("\nConcluido. Agora rode treinar_mega_dataset.py para treinar o modelo.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrai amostra do 14669616.zip para treinamento.")
    parser.add_argument(
        "--max-interacoes",
        type=int,
        default=MAX_INTERACOES,
        help=f"Máximo de interações a amostrar (padrão: {MAX_INTERACOES:,})",
    )
    args = parser.parse_args()
    extrair(max_interacoes=args.max_interacoes)
