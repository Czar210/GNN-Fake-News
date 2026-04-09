import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
from tqdm import tqdm

raiz_projeto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "src"))
from features import text_embedder

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, hidden_channels: int = 64):
        super(GCNClassifier, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        h = global_mean_pool(x, batch)
        x = F.dropout(h, p=0.5, training=self.training)
        out = self.lin(x)
        return out, h

def gerar_matriz():
    print("Gerador de Matriz de Confusão Restaurado. Iniciando inferências...")
    
    p_path = os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "data", "raw", "posts_coletados.csv")
    r_path = os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "data", "raw", "reposts_coletados.csv")
    
    df_p = pd.read_csv(p_path)
    df_i = pd.read_csv(r_path)
    
    keywords_fakes = ['conspiracy', 'fake', 'hoax', 'mentira', 'fraude', 'scam']
    
    # Caminhos
    peso_upfd = os.path.join(raiz_projeto, "Execution", "weights", "pesos_gcn.pth")
    peso_mega = os.path.join(raiz_projeto, "Execution", "results", "pesos_gcn_bluesky_ESCALA_MAIOR.pth")
    
    model_upfd = GCNClassifier(768, 2)
    model_mega = GCNClassifier(768, 2)
    
    if os.path.exists(peso_upfd):
        model_upfd.load_state_dict(torch.load(peso_upfd, map_location='cpu'))
    if os.path.exists(peso_mega):
        model_mega.load_state_dict(torch.load(peso_mega, map_location='cpu'))
        
    model_upfd.eval()
    model_mega.eval()

    y_true = []
    y_pred_upfd = []
    y_pred_mega = []

    for idx, row in tqdm(df_p.iterrows(), total=len(df_p)):
        texto = str(row['texto']).lower()
        label = 1 if any(k in texto for k in keywords_fakes) else 0
        
        # Embeddings
        emb = text_embedder.gerar_embeddings_de_texto(pd.DataFrame([{'texto': texto}]))
        x_raiz = torch.tensor(emb.iloc[0]['embedding'], dtype=torch.float)
        
        pid = row['post_id']
        interacoes = df_i[df_i['post_id'] == pid] if 'post_id' in df_i.columns else pd.DataFrame()
        
        num_nodos = 1 + len(interacoes)
        x = torch.stack([x_raiz for _ in range(num_nodos)])
        
        sources = [0] * len(interacoes)
        targets = list(range(1, num_nodos))
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        batch = torch.zeros(num_nodos, dtype=torch.long)
        
        with torch.no_grad():
            out_upfd, _ = model_upfd(x, edge_index, batch)
            out_mega, _ = model_mega(x, edge_index, batch)
            
            p_upfd = torch.argmax(out_upfd).item()
            p_mega = torch.argmax(out_mega).item()
            
        y_true.append(label)
        y_pred_upfd.append(p_upfd)
        y_pred_mega.append(p_mega)
        
    # Salvar matrizes
    out_dir = os.path.join(raiz_projeto, "Execution", "results", "matrizes_de_confusao")
    os.makedirs(out_dir, exist_ok=True)
    
    def plot_matrix(y_t, y_p, title, filename):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(title)
        plt.ylabel('Realidade (True Label)')
        plt.xlabel('Predição do Modelo')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()
        
    plot_matrix(y_true, y_pred_upfd, "Matriz de Confusão - Modelo Original (UPFD)", "matriz_upfd.png")
    plot_matrix(y_true, y_pred_mega, "Matriz de Confusão - Modelo Mega Dataset (Bluesky)", "matriz_mega.png")
    
    print("\nMatrizes geradas e salvas em:", out_dir)

if __name__ == "__main__":
    gerar_matriz()
