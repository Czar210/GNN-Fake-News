import os
import sys
import traceback
import numpy as np

# Importações do PyTorch / PyG (CS224W)
import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.models import GAE

# Importações para Análise de Cluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Configurações ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'UPFD')

# =============================================================================
# ETAPA 1: CARREGAR O DATASET (Completo)
# =============================================================================
def carregar_dataset_completo():
    """
    Carrega todos os 62 grafos do dataset UPFD.
    """
    print(f"Carregando dataset UPFD de: {DATA_PATH}")
    try:
        dataset = UPFD(root=DATA_PATH, name="politifact", feature="spacy")
        print(f"Dataset carregado com {len(dataset)} grafos (notícias).")
        return dataset
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        print(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# ETAPA 2: DEFINIR O MODELO (A GNN "NÃO ROTULADA")
# =============================================================================
class GraphEncoder(torch.nn.Module):
    """
    Este é o nosso "Encoder" (Codificador).
    Ele usa duas camadas GCN (como no Colab 0) para processar os nós.
    No final, 'global_mean_pool' comprime todos os nós em UM
    vetor de embedding para o grafo inteiro.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # O GCNConv é a camada GNN principal do Colab 0
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # x = features dos nós, edge_index = arestas
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # 'global_mean_pool' pega a média de todos os nós para
        # criar um único vetor (embedding) que representa o grafo.
        return global_mean_pool(x, batch)

def treinar_modelo_nao_supervisionado(dataset):
    """
    Treina um Graph Autoencoder (GAE) para aprender os embeddings
    de forma não supervisionada.
    """
    print("\nIniciando treinamento da GNN não supervisionada (GAE)...")
    
    # Prepara um 'DataLoader' para processar os grafos em lotes
    # (neste caso, 1 grafo por lote para o Autoencoder)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Define o modelo
    # data.num_features é o tamanho do embedding 'spacy' (ex: 300)
    # 32 é o tamanho do nosso embedding final
    encoder = GraphEncoder(dataset.num_features, 64, 32)
    
    # GAE é um modelo do PyG que junta o Encoder e um Decoder
    # O Decoder tentará reconstruir as arestas
    model = GAE(encoder) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loop de Treinamento
    model.train()
    total_loss = 0
    num_epochs = 20 # 20 passadas pelo dataset
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            
            # 1. O Encoder cria o embedding 'z'
            z = model.encode(data.x, data.edge_index, data.batch)
            
            # 2. O Decoder tenta reconstruir as arestas
            # 3. 'recon_loss' é a nossa perda (quão bem reconstruímos)
            loss = model.recon_loss(z, data.edge_index)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch: {epoch+1}/{num_epochs}, Perda (Loss) de Reconstrução: {total_loss/len(loader):.4f}")

    print("Treinamento concluído.")
    return model.encode # Retornamos APENAS o Encoder treinado

# =============================================================================
# ETAPA 3: EXTRAIR EMBEDDINGS
# =============================================================================
def extrair_embeddings_e_labels(dataset, encoder):
    """
    Passa todos os 62 grafos pelo Encoder treinado para obter
    os 62 vetores de embedding.
    """
    print("\nExtraindo embeddings de todos os 62 grafos...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Ordem importa
    
    embeddings = [] # Lista de vetores
    labels = []     # Lista de rótulos (0 ou 1)
    
    encoder.eval() # Coloca o modelo em modo "avaliação"
    with torch.no_grad(): # Desliga o cálculo de gradientes
        for data in loader:
            # Passamos o grafo pelo Encoder e pegamos o embedding 'z'
            z = encoder(data.x, data.edge_index, data.batch)
            
            embeddings.append(z.squeeze().cpu().numpy()) # Converte para NumPy
            labels.append(data.y.item())
            
    print(f"Embeddings extraídos. {len(embeddings)} vetores de {len(embeddings[0])} dimensões.")
    return np.array(embeddings), np.array(labels)

# =============================================================================
# ETAPA 4: PLOTAR OS CLUSTERS (t-SNE)
# =============================================================================
def plotar_clusters(embeddings, labels, output_filename="cluster_plot.png"):
    """
    Usa t-SNE para reduzir os embeddings para 2D e plota
    o gráfico de cluster.
    """
    print("\nRodando t-SNE para reduzir dimensionalidade...")
    
    # t-SNE é a técnica padrão para visualizar embeddings
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=200, 
                init='pca', n_iter=2500, random_state=42)
    
    # Transforma os embeddings de 32D para 2D
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print("Plotando o gráfico de clusters...")
    
    # Mapeia os labels (0, 1) para cores (verde, vermelho)
    cores = ['#008000', '#FF0000'] # Verde, Vermelho
    cores_map = [cores[label] for label in labels]
    
    plt.figure(figsize=(12, 8))
    
    # Separa os pontos para plotar com rótulos
    real_indices = (labels == 0)
    fake_indices = (labels == 1)
    
    plt.scatter(
        embeddings_2d[real_indices, 0], 
        embeddings_2d[real_indices, 1], 
        c=cores[0], label='Notícia Real (y=0)', alpha=0.7
    )
    plt.scatter(
        embeddings_2d[fake_indices, 0], 
        embeddings_2d[fake_indices, 1], 
        c=cores[1], label='Notícia Falsa (y=1)', alpha=0.7
    )
    
    plt.title('Análise de Cluster (t-SNE) de Embeddings de Grafos (Não Supervisionado)', fontsize=16)
    plt.xlabel('Dimensão t-SNE 1')
    plt.ylabel('Dimensão t-SNE 2')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(SCRIPT_DIR, output_filename)
    plt.savefig(output_path)
    
    print(f"\n--- SUCESSO! ---")
    print(f"Gráfico de cluster salvo em: {output_path}")

# =============================================================================
# PONTO DE ENTRADA DO SCRIPT
# =============================================================================
if __name__ == "__main__":
    
    # 1. Carregar os dados
    dataset_upfd = carregar_dataset_completo()
    
    # 2. Treinar o modelo não supervisionado
    encoder_treinado = treinar_modelo_nao_supervisionado(dataset_upfd)
    
    # 3. Extrair os 62 embeddings e rótulos
    embeddings_finais, labels_verdadeiros = extrair_embeddings_e_labels(dataset_upfd, encoder_treinado)
    
    # 4. Plotar o resultado
    plotar_clusters(embeddings_finais, labels_verdadeiros)