"""
sage_model.py
-------------
Define o SAGEClassifier — arquitetura GraphSAGE — como
drop-in replacement do GCNClassifier e GATClassifier existentes.

Interface idêntica ao GCN e GAT:
  model = SAGEClassifier(num_node_features=768, num_classes=2)
  out, h = model(x, edge_index, batch)

Diferenças chave vs GCNClassifier e GATClassifier:
  ┌─────────────────┬──────────────────────────────┬────────────────────────────┐
  │ Aspecto         │ GCNClassifier                │ SAGEClassifier             │
  ├─────────────────┼──────────────────────────────┼────────────────────────────┤
  │ Agregação       │ Soma normalizada por grau    │ Média (W_l·h + W_r·h_N)   │
  │ Matrizes/camada │ 1 (in→hid)                  │ 2 (W_l e W_r separadas)    │
  │ Indutivo        │ Não (transdutivo)            │ Sim (generaliza p/ nós novos)│
  │ Multi-head      │ Não                          │ Não                        │
  │ Ativação        │ ReLU                         │ ReLU (idem ao GCN)         │
  │ Dropout interno │ Só antes do classificador    │ Só antes do classificador  │
  │ Parâmetros      │ ~57,666                      │ ~115,010                   │
  └─────────────────┴──────────────────────────────┴────────────────────────────┘

Agregadores disponíveis via parâmetro `aggr`:
  'mean'  — média dos vizinhos (padrão, equivale ao SAGE original de Hamilton et al. 2017)
  'max'   — max-pooling dos embeddings de vizinhos
  'lstm'  — LSTM sobre a sequência de vizinhos (requer grafo com ordem estável)

Pipeline forward:
  x [N, 768]
    → SAGEConv(768 → 64, aggr='mean')  → ReLU
    → SAGEConv(64  → 64, aggr='mean')  → ReLU
    → SAGEConv(64  → 64, aggr='mean')
    → global_mean_pool → [B, 64]
    → Dropout(0.5)
    → Linear(64 → 2)
    → (logits [B, 2], embedding [B, 64])

Referência:
  Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017.
  https://arxiv.org/abs/1706.02216
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool


class SAGEClassifier(torch.nn.Module):
    """
    GraphSAGE para classificação binária de grafos de propagação.

    Args:
        num_node_features: Dimensão dos features de entrada por nó (768 para BERT).
        num_classes:       Número de classes (2: Real / Fake).
        hidden_channels:   Dimensão do espaço latente interno (padrão: 64).
        aggr:              Agregador de vizinhança — 'mean' | 'max' | 'lstm' (padrão: 'mean').
        dropout:           Dropout antes do classificador final (padrão: 0.5).
    """

    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        aggr: str = "mean",
        dropout: float = 0.5,
        seed: int = 12345,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.dropout = dropout

        # Camada 1: 768 → 64
        # SAGEConv aprende duas matrizes: W_l (self-loop) e W_r (vizinhos)
        self.conv1 = SAGEConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            aggr=aggr,
        )

        # Camada 2: 64 → 64
        self.conv2 = SAGEConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr=aggr,
        )

        # Camada 3: 64 → 64 (consolidação)
        self.conv3 = SAGEConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr=aggr,
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
        x = x.relu()

        # ── Camada 2 ──────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)
        x = x.relu()

        # ── Camada 3 (consolidação) ────────────────────────────────────────
        x = self.conv3(x, edge_index)

        # ── Readout (Global Pooling) → embedding do grafo ──────────────────
        h = global_mean_pool(x, batch)

        # ── Classificação ─────────────────────────────────────────────────
        x = F.dropout(h, p=self.dropout, training=self.training)
        out = self.lin(x)

        return out, h

    def count_parameters(self) -> int:
        """Retorna o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Teste rápido de sanidade ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testando SAGEClassifier com grafo dummy...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SAGEClassifier(num_node_features=768, num_classes=2).to(device)
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
    print("\n[OK] SAGEClassifier funcionando corretamente!")

    # ── Comparação de parâmetros com GCN e GAT ────────────────────────────
    print("\n-- Comparacao de complexidade --")
    print(f"  GCN  (referencia):  ~57,666 parametros")
    print(f"  SAGE (este modelo): {model.count_parameters():,} parametros")
    print(f"  GAT  (referencia):  ~218,562 parametros")
