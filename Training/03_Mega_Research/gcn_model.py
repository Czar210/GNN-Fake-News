"""
gcn_model.py
------------
Define o GCNClassifier — arquitetura Graph Convolutional Network — como
drop-in replacement do GATClassifier e SAGEClassifier existentes.

Interface identica:
  model = GCNClassifier(num_node_features=768, num_classes=2)
  out, h = model(x, edge_index, batch)

Pipeline forward:
  x [N, 768]
    -> GCNConv(768 -> 64)  -> ReLU
    -> GCNConv(64  -> 64)  -> ReLU
    -> GCNConv(64  -> 64)
    -> global_mean_pool -> [B, 64]
    -> Dropout(0.5)
    -> Linear(64 -> 2)
    -> (logits [B, 2], embedding [B, 64])

Referencia:
  Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
  Networks", ICLR 2017. https://arxiv.org/abs/1609.02907
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(torch.nn.Module):
    """
    GCN para classificacao binaria de grafos de propagacao.

    Args:
        num_node_features: Dimensao dos features de entrada por no (768 para BERT).
        num_classes:       Numero de classes (2: Real / Fake).
        hidden_channels:   Dimensao do espaco latente interno (padrao: 64).
        dropout:           Dropout antes do classificador final (padrao: 0.5).
        seed:              Semente para inicializacao de pesos (padrao: 12345).
    """

    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        dropout: float = 0.5,
        seed: int = 12345,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.dropout = dropout
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        h = global_mean_pool(x, batch)
        x = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(x), h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Teste rapido de sanidade ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testando GCNClassifier com grafo dummy...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GCNClassifier(num_node_features=768, num_classes=2).to(device)
    print(f"Parametros treinaveis: {model.count_parameters():,}")

    x          = torch.randn(5, 768, device=device)
    edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long, device=device)
    batch      = torch.zeros(5, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out, h = model(x, edge_index, batch)

    print(f"out.shape: {out.shape}  (esperado: [1, 2])")
    print(f"h.shape:   {h.shape}    (esperado: [1, 64])")
    print("[OK] GCNClassifier funcionando corretamente!")
