import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import sys
import traceback

# Importações do CS224W (PyTorch e PyG)
import torch
from torch_geometric.datasets import UPFD
from torch_geometric.utils import to_networkx

# --- Configurações ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'UPFD')

# =============================================================================
# ETAPA 1: CARREGAR O GRAFO (PyTorch Geometric)
# =============================================================================
def carregar_dataset_upfd():
    """
    Baixa e carrega o dataset UPFD (PolitiFact) usando PyTorch Geometric.
    """
    print(f"Baixando e carregando o dataset UPFD (PolitiFact) para: {DATA_PATH}")
    try:
        # =====================================================================
        # <<< AQUI ESTÁ A CORREÇÃO >>>
        # Mudei "PolitiFact" para "politifact" (tudo minúsculo)
        # =====================================================================
        dataset = UPFD(root=DATA_PATH, name="politifact", feature="bert")
        
        print("\n--- Dataset UPFD Carregado com Sucesso ---")
        print(f"Número total de grafos (notícias): {len(dataset)}")
        print(f"  - Notícias Reais (y=0): {(dataset.y == 0).sum()}")
        print(f"  - Notícias Falsas (y=1): {(dataset.y == 1).sum()}")
        return dataset
    
    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f">>> ERRO DETALHADO AO CARREGAR/BAIXAR O DATASET:")
        print(traceback.format_exc()) # Imprime o erro completo
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print("Se o erro for 'AssertionError', foi um erro de digitação no nome.")
        print("Se for um erro de instalação, tente os comandos pip novamente.")
        sys.exit(1)

# =============================================================================
# ETAPA 2: ESCOLHER, INSPECIONAR E CONVERTER (PyG -> NetworkX)
# =============================================================================
def extrair_e_converter_grafo(dataset, index_do_grafo=15):
    """
    Pega UM grafo de propagação do dataset e o converte para NetworkX.
    """
    print(f"\n--- Inspecionando Grafo de Propagação (Notícia) de Índice {index_do_grafo} ---")
    
    # Validação para garantir que o índice existe
    if index_do_grafo >= len(dataset):
        print(f"Erro: O índice {index_do_grafo} está fora do alcance. O dataset tem {len(dataset)} grafos.")
        print("Tentando o índice 0...")
        index_do_grafo = 0
        
    data = dataset[index_do_grafo]
    label_num = data.y.item() 
    label_str = "FAKE" if label_num == 1 else "REAL"
    
    print(f"  - Rótulo (Label): {label_str} (y={label_num})")
    print(f"  - Nós (usuários/tweets): {data.num_nodes}")
    print(f"  - Arestas (reposts): {data.num_edges}")

    print("\nConvertendo grafo de PyG para NetworkX...")
    G_nx = to_networkx(data, to_undirected=False)
    
    print("Conversão concluída.")
    return G_nx, label_str, index_do_grafo

# =============================================================================
# ETAPA 3: VISUALIZAR (NetworkX -> Pyvis)
# =============================================================================
def visualizar_grafo_propagacao(G, label, index, nome_arquivo_base="upfd_grafo_amostra"):
    """
    Cria uma visualização interativa do grafo de propagação com Pyvis.
    """
    output_filename = f"{nome_arquivo_base}_idx{index}_{label}.html"
    output_path = os.path.join(SCRIPT_DIR, output_filename)
    
    print(f"\nIniciando visualização com Pyvis...")
    print(f"Salvando em: {output_path}")

    net = Network(height="800px", width="100%", directed=True, 
                  notebook=False, cdn_resources='in_line')
    
    cor_noticia = '#FF0000' if label == 'FAKE' else '#008000' 
    cor_usuario = '#87CEEB'
    
    for node_id in G.nodes():
        if node_id == 0: # Nó Raiz (A Notícia)
            net.add_node(
                str(node_id), 
                label=f"NOTÍCIA ({label})", 
                color=cor_noticia, 
                size=25,
                title=f"Raiz da Propagação (Notícia)\nRótulo: {label}"
            )
        else: # Nós de Usuário/Repost
            net.add_node(
                str(node_id), 
                label=f"Usr {node_id}", 
                color=cor_usuario, 
                size=10,
                title=f"Nó de propagação {node_id}"
            )
            
    print("Adicionando arestas ao Pyvis...")
    for u, v in G.edges():
        net.add_edge(str(u), str(v))
    
    net.show_buttons(filter_=['physics'])
    
    try:
        html_content = net.generate_html(notebook=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Visualização salva! Abra o arquivo '{output_path}' no seu navegador.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo HTML: {e}")

# =============================================================================
# PONTO DE ENTRADA DO SCRIPT
# =============================================================================
if __name__ == "__main__":
    
    dataset_upfd = carregar_dataset_upfd()
    
    INDICE_PARA_VER = 50 
    G_networkx, G_label, G_index = extrair_e_converter_grafo(dataset_upfd, 
                                                             index_do_grafo=INDICE_PARA_VER)
    
    if G_networkx.number_of_nodes() > 0:
        visualizar_grafo_propagacao(G_networkx, G_label, G_index)
    else:
        print("O grafo selecionado está vazio. Encerrando.")