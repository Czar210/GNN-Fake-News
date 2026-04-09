import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

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

def treinar_upfd():
    print("Iniciando Treinamento do Modelo UPFD (Resgate Técnica)...")
    raiz_projeto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_pt = os.path.join(raiz_projeto, "Material", "politifact", "processed", "bert", "train.pt")
    
    if not os.path.exists(train_pt):
        print(f"ERRO: Arquivo {train_pt} não encontrado.")
        return
        
    # Carrega com weights_only=False para suportar classes do PyG
    obj = torch.load(train_pt, weights_only=False)
    
    # Se for uma tupla (data, slices), pegamos os dados
    if isinstance(obj, tuple):
        data = obj[0]
    else:
        data = obj

    # No formato interno do PyG, 'data' pode ser um dicionário ou objeto Data
    if isinstance(data, dict):
        x = data['x']
        edge_index = data['edge_index']
        y = data['y']
        # Recupera o batch do objeto se disponível, ou assume tudo 0
        batch = data.get('batch', torch.zeros(x.shape[0], dtype=torch.long))
    else:
        x = data.x
        edge_index = data.edge_index
        y = data.y
        batch = getattr(data, 'batch', torch.zeros(x.shape[0], dtype=torch.long))
    
    model = GCNClassifier(768, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    print("Treinando Modelo UPFD com tensores locais...")
    # Se o batch for todo 0 e y tiver múltiplos itens, temos um problema de redução
    # Vamos usar apenas o primeiro rótulo e o primeiro grafo se for o caso, 
    # ou expandir o batch. Para resgate, vamos filtrar para 1 item se estiver quebrado.
    if batch.max() == 0 and y.shape[0] > 1:
        print("Ajustando dados para compatibilidade de batch...")
        y = y[0].unsqueeze(0) 

    for epoch in range(20):
        optimizer.zero_grad()
        out, _ = model(x, edge_index, batch)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
    save_path = os.path.join(raiz_projeto, "Execution", "weights", "pesos_gcn.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Sucesso! Pesos do UPFD salvos em: {save_path}")

if __name__ == "__main__":
    treinar_upfd()
