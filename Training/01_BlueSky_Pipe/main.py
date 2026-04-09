# main.py

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import networkx as nx
import pandas as pd

from analysis import graph_analyzer
from collection import collect
from features import text_embedder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TERMO_DE_BUSCA = "Silk Song"
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logging.info("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS ---")

    posts_filepath = RAW_DATA_DIR / "posts_coletados.csv"
    reposts_filepath = RAW_DATA_DIR / "reposts_coletados.csv"

    if not posts_filepath.exists() or not reposts_filepath.exists():
        logging.info(f"Ficheiros não encontrados. Iniciando coleta para: '{TERMO_DE_BUSCA}'")
        try:
            client = collect.login_bluesky()
            posts_df, reposts_df = collect.coletar_dados_por_termo(
                client, TERMO_DE_BUSCA, limite_total_posts=500
            )

            if posts_df.empty:
                logging.warning("Nenhum post foi coletado. Pipeline interrompido.")
                return

            posts_df.to_csv(posts_filepath, index=False)
            reposts_df.to_csv(reposts_filepath, index=False)
            logging.info(f"Dados brutos salvos em '{RAW_DATA_DIR}'")

        except Exception as e:
            logging.error(f"Falha crítica na etapa de coleta: {e}")
            return
    else:
        logging.info("Ficheiros de dados brutos já existem. A carregar do disco.")
        posts_df = pd.read_csv(posts_filepath)
        reposts_df = pd.read_csv(reposts_filepath)

    logging.info("Iniciando a geração de features (embeddings)...")
    try:
        posts_com_embeddings_df = text_embedder.gerar_embeddings_de_texto(posts_df)
    except Exception as e:
        logging.error(f"Falha crítica na geração de embeddings: {e}")
        return

    try:
        processed_filepath = PROCESSED_DATA_DIR / "posts_com_embeddings.parquet"
        posts_com_embeddings_df.to_parquet(processed_filepath, index=False)
        logging.info(f"Dados processados (com embeddings) salvos em '{processed_filepath}'")
    except Exception as e:
        logging.error(f"Falha ao salvar os dados processados: {e}")
        return

    logging.info("Iniciando a construção e análise do grafo como validação...")
    if reposts_df.empty:
        logging.warning("Nenhum repost encontrado. Não é possível construir o grafo.")
    else:
        G = nx.from_pandas_edgelist(
            reposts_df, source="source", target="target", create_using=nx.DiGraph()
        )
        logging.info(
            f"Grafo construído com {G.number_of_nodes()} nós e "
            f"{G.number_of_edges()} arestas."
        )

        top_influencers = graph_analyzer.analisar_influencia(G)
        if top_influencers is not None:
            print("\n--- Top Usuários Mais Influentes (Mais Reposts Recebidos) ---")
            print(top_influencers.to_string(index=False))

        graph_analyzer.detectar_comunidades(G)

    logging.info("--- PIPELINE DE PROCESSAMENTO DE DADOS FINALIZADO COM SUCESSO ---")


if __name__ == "__main__":
    main()