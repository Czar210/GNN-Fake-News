"""
gat_model.py
------------
Define o GATClassifier — arquitetura Graph Attention Network — como
drop-in replacement do GCNClassifier existente.

Interface idêntica ao GCN:
  model = GATClassifier(num_node_features=768, num_classes=2)
  out, h = model(x, edge_index, batch)

Diferenças chave vs GCNClassifier:
  ┌─────────────────┬──────────────────────────────┬────────────────────────────┐
  │ Aspecto         │ GCNClassifier                │ GATClassifier              │
  ├─────────────────┼──────────────────────────────┼────────────────────────────┤
  │ Agregação       │ Soma normalizada por grau    │ Atenção aprendida (softmax)│
  │ Multi-head      │ Não                          │ 4 heads (camadas 1 e 2)    │
  │ Ativação        │ ReLU                         │ ELU (padrão GAT)           │
  │ Dropout interno │ Só antes do classificador    │ Entre cada camada GAT      │
  │ Parâmetros      │ ~3x(in→hid) + (hid→2)       │ 4x mais por camada         │
  └─────────────────┴──────────────────────────────┴────────────────────────────┘

Pipeline forward:
  x [N, 768]
    → GATConv(768 → 64, heads=4, concat=False)  → ELU → Dropout
    → GATConv(64  → 64, heads=4, concat=False)  → ELU → Dropout
    → GATConv(64  → 64, heads=1)
    → global_mean_pool → [B, 64]
    → Dropout(0.5)
    → Linear(64 → 2)
    → (logits [B, 2], embedding [B, 64])
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool


class GATClassifier(torch.nn.Module):
    """
    Graph Attention Network para classificação binária de grafos.

    Args:
        num_node_features: Dimensão dos features de entrada por nó (768 para BERT).
        num_classes:       Número de classes (2: Real / Fake).
        hidden_channels:   Dimensão do espaço latente interno (padrão: 64).
        heads:             Número de cabeças de atenção nas camadas 1 e 2 (padrão: 4).
        dropout_attn:      Dropout aplicado internamente na atenção (padrão: 0.3).
        dropout_out:       Dropout antes do classificador final (padrão: 0.5).
    """

    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout_attn: float = 0.3,
        dropout_out: float = 0.5,
    ):
        super().__init__()
        torch.manual_seed(12345)

        self.dropout_attn = dropout_attn
        self.dropout_out  = dropout_out

        # Camada 1: 768 → 64 (média de 4 heads → concat=False mantém dim=64)
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            heads=heads,
            concat=False,           # output dim = hidden_channels (não heads*hidden)
            dropout=dropout_attn,
        )

        # Camada 2: 64 → 64 (mesma lógica)
        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout_attn,
        )

        # Camada 3: 64 → 64 (1 head — camada de consolidação)
        self.conv3 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=1,
            concat=True,            # com 1 head e concat=True → dim mantém 64
            dropout=dropout_attn,
        )

        # Classificador final
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ):
        """
        Args:
            x:          Features dos nós [N, num_node_features]
            edge_index: Arestas do grafo [2, E]
            batch:      Mapeamento nó→grafo [N]

        Returns:
            out: Logits de classificação [B, num_classes] (não-softmaxados)
            h:   Embeddings latentes do grafo [B, hidden_channels]
        """
        # ── Camada 1 ──────────────────────────────────────────────────────
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_attn, training=self.training)

        # ── Camada 2 ──────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_attn, training=self.training)

        # ── Camada 3 (consolidação) ────────────────────────────────────────
        x = self.conv3(x, edge_index)

        # ── Readout (Global Pooling) → embedding do grafo ──────────────────
        h = global_mean_pool(x, batch)

        # ── Classificação ─────────────────────────────────────────────────
        x = F.dropout(h, p=self.dropout_out, training=self.training)
        out = self.lin(x)

        return out, h

    def count_parameters(self) -> int:
        """Retorna o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Teste rápido de sanidade ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testando GATClassifier com grafo dummy...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GATClassifier(num_node_features=768, num_classes=2).to(device)
    print(f"Parâmetros treináveis: {model.count_parameters():,}")
    print(model)

    # Grafo dummy: 5 nós, 4 arestas (estrela simples), batch de 1 grafo
    x          = torch.randn(5, 768, device=device)
    edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long, device=device)
    batch      = torch.zeros(5, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out, h = model(x, edge_index, batch)

    print(f"\nout.shape: {out.shape}  (esperado: [1, 2])")
    print(f"h.shape:   {h.shape}    (esperado: [1, 64])")
    print(f"out: {out}")
    print(f"prob_fake: {torch.softmax(out, dim=1)[0, 1].item():.4f}")
    print("\n[OK] GATClassifier funcionando corretamente!")
