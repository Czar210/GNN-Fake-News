# src/analysis/graph_analyzer.py

import logging

import networkx as nx
import pandas as pd
# A biblioteca para o algoritmo de Louvain é importada com um alias claro
import community as community_louvain

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def analisar_influencia(graph: nx.DiGraph, top_n: int = 10) -> pd.DataFrame | None:
    if graph.number_of_nodes() == 0:
        logging.warning("Grafo vazio. Análise de influência não pode ser realizada.")
        return None

    # O in-degree de um nó representa o número de arestas que apontam para ele.
    # No nosso contexto, é o número de reposts que um usuário recebeu.
    in_degree_centrality = dict(graph.in_degree())

    df_influencia = pd.DataFrame(
        in_degree_centrality.items(), columns=["usuario", "reposts_recebidos"]
    )

    df_influencia_ordenado = df_influencia.sort_values(
        by="reposts_recebidos", ascending=False
    )

    return df_influencia_ordenado.head(top_n)


def detectar_comunidades(graph: nx.DiGraph) -> dict | None:
    if graph.number_of_nodes() == 0:
        logging.warning("Grafo vazio. Detecção de comunidades não pode ser realizada.")
        return None

    # O algoritmo de Louvain funciona em grafos não-direcionados.
    # Convertemos o grafo para esta análise específica.
    graph_undirected = graph.to_undirected()

    partition = community_louvain.best_partition(graph_undirected)

    num_comunidades = len(set(partition.values()))
    logging.info(f"Detecção de comunidades encontrou {num_comunidades} grupos.")

    return partition