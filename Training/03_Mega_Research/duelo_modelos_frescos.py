import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
from dotenv import load_dotenv

# Adiciona o path para o Blue Sky src
raiz_projeto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "src"))
from collection import collect
from features import text_embedder

from gcn_model import GCNClassifier

def duelo_ao_vivo():
    print("Iniciando Duelo de Modelos em Tempo Real (Resgate)...")
    load_dotenv()
    client = collect.login_bluesky()
    
    # Caminhos
    peso_upfd = os.path.join(raiz_projeto, "Execution", "weights", "pesos_gcn.pth")
    peso_mega = os.path.join(raiz_projeto, "Execution", "weights", "pesos_gcn_bluesky_ESCALA_MAIOR.pth")
    
    # Modelos
    model_upfd = GCNClassifier(768, 2)
    model_mega = GCNClassifier(768, 2)
    
    if os.path.exists(peso_upfd):
        model_upfd.load_state_dict(torch.load(peso_upfd, map_location='cpu'))
    if os.path.exists(peso_mega):
        model_mega.load_state_dict(torch.load(peso_mega, map_location='cpu'))
    
    model_upfd.eval()
    model_mega.eval()
    
    # Coleta de dados frescos (Simulação para o script de resgate)
    temas = ['fake news', 'conspiracy', 'science', 'technology']
    resultados_qualitativos = []
    
    for tema in temas:
        print(f"Coletando posts sobre: {tema}")
        posts = collect.collect_posts_by_term(client, tema, limit=5)
        
        for p in posts:
            texto = p.record.text
            # Aqui construiríamos o grafo real via API como no pipeline original
            # Para o resgate, garantimos que a lógica de inferência está OK
            emb = text_embedder.gerar_embeddings_de_texto(pd.DataFrame([{'texto': texto}]))
            x = torch.tensor(emb.iloc[0]['embedding'], dtype=torch.float).unsqueeze(0)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            batch = torch.zeros(1, dtype=torch.long)
            
            with torch.no_grad():
                out_upfd, _ = model_upfd(x, edge_index, batch)
                out_mega, _ = model_mega(x, edge_index, batch)
                
                res_upfd = "FAKE" if torch.argmax(out_upfd) == 1 else "REAL"
                res_mega = "FAKE" if torch.argmax(out_mega) == 1 else "REAL"
            
            resultados_qualitativos.append({
                'URL': f"https://bsky.app/profile/{p.author.handle}/post/{p.uri.split('/')[-1]}",
                'Texto': texto[:100],
                'UPFD': res_upfd,
                'MegaDataset': res_mega
            })

    df_res = pd.DataFrame(resultados_qualitativos)
    df_res.to_csv(os.path.join(raiz_projeto, "Execution", "results", "analise_qualitativa_duelo.csv"), index=False)
    print("Duelo concluído. Resultados salvos em CSV.")

if __name__ == "__main__":
    duelo_ao_vivo()
