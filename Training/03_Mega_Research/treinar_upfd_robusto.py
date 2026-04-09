import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

raiz_projeto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def load_local_upfd(split="train"):
    pt_file = os.path.join(raiz_projeto, "Material", "politifact", "processed", "bert", f"{split}.pt")
    
    obj = torch.load(pt_file, weights_only=False)
    data = obj[0] if isinstance(obj, tuple) else obj
    
    grafos = []
    
    if isinstance(data, dict):
        x = data['x']
        edge_index = data['edge_index']
        y = data['y']
    else:
        x = data.x
        edge_index = data.edge_index
        y = data.y
        
    num_graphs = len(y)
    total_nodes = len(x)
    chunk_size = total_nodes // num_graphs
    
    for i in range(num_graphs):
        start = i * chunk_size
        end = start + chunk_size if i < num_graphs - 1 else total_nodes
        
        x_sub = x[start:end]
        mask = (edge_index[0] >= start) & (edge_index[0] < end) & (edge_index[1] >= start) & (edge_index[1] < end)
        ei_sub = edge_index[:, mask]
        if ei_sub.numel() > 0:
            ei_sub = ei_sub - start
            
        grafos.append(Data(x=x_sub, edge_index=ei_sub, y=y[i].unsqueeze(0)))
    
    return grafos

def treinar_upfd():
    print("Carregando Dados UPFD...")
    train_data = load_local_upfd("train")
    val_data = load_local_upfd("val")
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    model = GCNClassifier(768, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Iniciando treinamento limpo UPFD...")
    
    best_val_acc = 0
    save_path = os.path.join(raiz_projeto, "Execution", "weights", "pesos_gcn.pth")
    
    for epoch in range(1, 31): # 30 epochs
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        # Validação
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                out, _ = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.num_graphs
                
        val_acc = correct / total
        print(f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_data):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc or epoch == 30:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"Treinamento Concluído. Melhor Acurácia de Validação: {best_val_acc:.4f}")
    print(f"Salvo em: {save_path}")

if __name__ == "__main__":
    treinar_upfd()
