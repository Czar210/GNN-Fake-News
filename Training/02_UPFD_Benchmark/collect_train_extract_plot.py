import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- 1. Configuração e Modelo (Definido uma única vez!) ---

class GCNClassifier(torch.nn.Module):
    """
    Modelo GCN para classificação de grafos.
    Esta é a única fonte de verdade para a arquitetura da rede.
    """
    def __init__(self, num_node_features: int, num_classes: int, hidden_channels: int = 64):
        super(GCNClassifier, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Embeddings dos nós
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        # Readout (Global Pooling) -> Gera o vetor do cluster
        h = global_mean_pool(x, batch)

        # Classificação
        x = F.dropout(h, p=0.5, training=self.training)
        out = self.lin(x)
        
        return out, h

# --- 2. Funções de Treinamento ---

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch) # Ignoramos 'h' no treino
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def run_training(model, loader, device, epochs=20):
    """Gerencia o loop completo de treinamento."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    print(f"--- Iniciando Treinamento ({epochs} épocas) ---")
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Treinamento finalizado.")

# --- 3. Funções de Análise e Visualização ---

def get_embeddings(model, loader, device) -> pd.DataFrame:
    """Extrai os vetores latentes (h) para análise de clusters."""
    model.eval()
    embeddings_list = []
    labels_list = []

    print("--- Extraindo Embeddings para Visualização ---")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, h = model(data.x, data.edge_index, data.batch)
            embeddings_list.append(h.cpu().numpy())
            labels_list.append(data.y.cpu().numpy())

    X = np.concatenate(embeddings_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    df = pd.DataFrame(X)
    df['label'] = y
    df['Tipo'] = df['label'].map({0: 'Fake News', 1: 'Real News'})
    return df

def plot_clusters(df: pd.DataFrame):
    """Gera o gráfico t-SNE."""
    print("--- Gerando Gráfico t-SNE ---")
    # Remove colunas não numéricas
    features = df.drop(columns=['label', 'Tipo']).values
    
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    z = tsne.fit_transform(features)
    
    df['x_tsne'] = z[:, 0]
    df['y_tsne'] = z[:, 1]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.scatterplot(
        data=df, x='x_tsne', y='y_tsne', hue='Tipo', style='Tipo',
        palette={'Fake News': '#E74C3C', 'Real News': '#2ECC71'},
        s=100, alpha=0.8
    )
    plt.title("Clusters de Fake News vs Real News (Dataset UPFD)", fontsize=14)
    
    output_file = "resultado_final_clusters.png"
    plt.savefig(output_file)
    print(f"Gráfico salvo como: {output_file}")
    plt.show()

# --- 4. Pipeline Principal (Main) ---

if __name__ == "__main__":
    # Configurações Globais
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executando em: {device}")

    # A. Carga de Dados
    dataset = UPFD(root='.', name='politifact', feature='bert', split='train')
    # Um único loader serve para tudo se não precisarmos de validação separada agora
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # B. Instanciar Modelo
    model = GCNClassifier(dataset.num_features, dataset.num_classes).to(device)

    # C. Pipeline de Execução
    run_training(model, loader, device, epochs=25)
    
    # Para extração, usamos shuffle=False para garantir ordem (opcional, mas boa prática)
    analysis_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    df_results = get_embeddings(model, analysis_loader, device)
    
    plot_clusters(df_results)