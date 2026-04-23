import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Adiciona o path para o BERT embedder
raiz_projeto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "src"))
from features import text_embedder

from gcn_model import GCNClassifier

def treinar_com_dados_locais():
    print("Iniciando Treinamento com Dados Locais (Resgate)...")
    p_path = os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "data", "raw", "posts_coletados.csv")
    r_path = os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "data", "raw", "reposts_coletados.csv")
    
    df_p = pd.read_csv(p_path)
    df_i = pd.read_csv(r_path)
    
    # Heurística simples de rotulagem para o resgate (mesma lógica do original)
    keywords_fakes = ['conspiracy', 'fake', 'hoax', 'mentira', 'fraude', 'scam']
    
    grafos = []
    # Agrupa por post_id
    for idx, row in tqdm(df_p.iterrows(), total=len(df_p)):
        texto = str(row['texto']).lower()
        label = 1 if any(k in texto for k in keywords_fakes) else 0
        
        # Embeddings
        emb = text_embedder.gerar_embeddings_de_texto(pd.DataFrame([{'texto': texto}]))
        x_raiz = torch.tensor(emb.iloc[0]['embedding'], dtype=torch.float)
        
        # Filtra reposts deste post específico
        pid = row['post_id']
        interacoes = df_i[df_i['post_id'] == pid] if 'post_id' in df_i.columns else pd.DataFrame()
        
        # Construção básica do grafo (Análise de Propagação)
        num_nodos = 1 + len(interacoes)
        x = torch.stack([x_raiz for _ in range(num_nodos)])
        
        sources = [0] * len(interacoes)
        targets = list(range(1, num_nodos))
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        grafos.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))
        
    # Treino
    model = GCNClassifier(768, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(20):
        total_loss = 0
        for data in grafos:
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long))
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss/len(grafos):.4f}")
        
    out_path = os.path.join(raiz_projeto, "Execution", "results", "pesos_gcn_bluesky_ESCALA_MAIOR.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Sucesso! Pesos salvos em {out_path}")

if __name__ == "__main__":
    treinar_com_dados_locais()
