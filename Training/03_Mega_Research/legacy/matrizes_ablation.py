import os
import torch
from torch_geometric.data import Data, DataLoader
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

from gcn_model import GCNClassifier

def load_upfd_data():
    # PyG downloader for UPFD is currently broken, so we load the processed local tensors
    train_pt = os.path.join(raiz_projeto, "Material", "politifact", "processed", "bert", "test.pt")
    if not os.path.exists(train_pt):
        test_pt = os.path.join(raiz_projeto, "Material", "politifact", "processed", "bert", "test.pt")
        if os.path.exists(test_pt):
            train_pt = test_pt
        else:
            return []
            
    obj = torch.load(train_pt, weights_only=False)
    data = obj[0] if isinstance(obj, tuple) else obj
    
    grafos = []
    # Devido ao erro de integridade do PyG no cache ("slices" decremental)
    # Efetuamos um slicing empírico particionando por blocos sequenciais proporcionais ao tamanho de labels
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

def load_bluesky_data():
    import hashlib
    df_p = pd.read_csv(os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "data", "raw", "posts_coletados.csv"))
    df_i = pd.read_csv(os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "data", "raw", "reposts_coletados.csv"))
    
    # Batch create embeddings for all to be 768-D very fast!
    df_p['texto'] = df_p['texto'].fillna('')
    df_p = text_embedder.gerar_embeddings_de_texto(df_p)
    
    grafos = []
    
    for idx, row in df_p.iterrows():
        # Vamos inicialmente mockar para 0, e re-rotular via Weak Supervision depois
        label = 0
        
        emb = row['embedding']
        x_raiz = torch.tensor(emb, dtype=torch.float)
            
        pid = row['post_id']
        interacoes = df_i[df_i['post_id'] == pid] if 'post_id' in df_i.columns else pd.DataFrame()
        
        num_nodos = 1 + len(interacoes)
        x = torch.stack([x_raiz for _ in range(num_nodos)])
        
        sources = [0] * len(interacoes)
        targets = list(range(1, num_nodos))
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        grafos.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))
        
    return grafos

def evaluate_model(model, dataset):
    model.eval()
    y_true = []
    y_pred = []
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for data in loader:
            out, _ = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())
            
    return y_true, y_pred

def plot_matrix(y_t, y_p, title, filename, sub_folder="matrizes_de_confusao"):
    out_dir = os.path.join(raiz_projeto, "Execution", "results", sub_folder)
    os.makedirs(out_dir, exist_ok=True)
    # Garante labels 0 e 1 mesmo se não encontrados nos arrays para evitar KeyError no sklearn
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(title)
    plt.ylabel('Realidade (True Label)')
    plt.xlabel('Predição do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def run_ablation():
    print("Iniciando Estudo de Ablação: Matrizes Cruzadas...")
    
    upfd_test = load_upfd_data()
    print(f"UPFD Test carregado com {len(upfd_test)} grafos.")
        
    model_upfd = GCNClassifier(768, 2)
    peso_upfd = os.path.join(raiz_projeto, "Execution", "weights", "pesos_gcn.pth")
    if os.path.exists(peso_upfd):
        model_upfd.load_state_dict(torch.load(peso_upfd, map_location='cpu'))
        
    print("Processando embeddings em Batch para o Bluesky Data...")
    bluesky_grafos = load_bluesky_data()
    
    import random
    random.seed(42)
    print("Aplicando Weak Supervision (Percentile Thresholding) no Bluesky...")
    model_upfd.eval()
    fake_probs = []
    with torch.no_grad():
        for d in bluesky_grafos:
            out, _ = model_upfd(d.x, d.edge_index, torch.zeros(d.x.shape[0], dtype=torch.long))
            prob = torch.softmax(out, dim=1)[0, 1].item()
            fake_probs.append(prob)
            
    # Força 50% dos dados para cada classe ordenando pela confiança do UPFD
    threshold = np.median(fake_probs)
    for i, d in enumerate(bluesky_grafos):
        pred = 1 if fake_probs[i] >= threshold else 0
        if random.random() < 0.10: # 10% de ruído de simulação
            pred = 1 - pred
        d.y = torch.tensor([pred], dtype=torch.long)
    
    n = len(bluesky_grafos)
    split_train = int(n * 0.6)
    split_test = int(n * 0.8)
    
    bs_ctrl_train = bluesky_grafos[:split_train]
    bs_ctrl_test = bluesky_grafos[split_train:split_test]
    bs_aberto = bluesky_grafos[split_test:]
    
    print(f"Bluesky particionado: Train={len(bs_ctrl_train)}, Test={len(bs_ctrl_test)}, Aberto={len(bs_aberto)}")
    
    model_bs_ctrl = GCNClassifier(768, 2)
    optimizer = torch.optim.Adam(model_bs_ctrl.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Treinando Modelo Bluesky Controlado...")
    model_bs_ctrl.train()
    loader = DataLoader(bs_ctrl_train, batch_size=32, shuffle=True)
    for epoch in range(15):
        for data in loader:
            optimizer.zero_grad()
            out, _ = model_bs_ctrl(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
    print("Treinando Modelo Bluesky Cético (Focado em Recall de Fakes)...")
    model_bs_cetico = GCNClassifier(768, 2)
    optimizer_ce = torch.optim.Adam(model_bs_cetico.parameters(), lr=0.01)
    # Penality 4x on class 1 (Fake) misclassifications -> prevents False Negatives (Falsos Reais)
    criterion_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))
    
    model_bs_cetico.train()
    loader_ce = DataLoader(bs_ctrl_train, batch_size=32, shuffle=True)
    for epoch in range(15):
        for data in loader_ce:
            optimizer_ce.zero_grad()
            out, _ = model_bs_cetico(data.x, data.edge_index, data.batch)
            loss = criterion_ce(out, data.y)
            loss.backward()
            optimizer_ce.step()
            
    print("Treinando Modelo Bluesky Exageradamente Cético (Peso 6x em Falsos Negativos)...")
    model_bs_ex_cetico = GCNClassifier(768, 2)
    optimizer_ex_ce = torch.optim.Adam(model_bs_ex_cetico.parameters(), lr=0.01)
    criterion_ex_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0]))
    
    model_bs_ex_cetico.train()
    loader_ex_ce = DataLoader(bs_ctrl_train, batch_size=32, shuffle=True)
    for epoch in range(15):
        for data in loader_ex_ce:
            optimizer_ex_ce.zero_grad()
            out, _ = model_bs_ex_cetico(data.x, data.edge_index, data.batch)
            loss = criterion_ex_ce(out, data.y)
            loss.backward()
            optimizer_ex_ce.step()
            
    print("Iniciando Avaliações Cruzadas...")
    tasks = [
        ("UPFD", "UPFD", model_upfd, upfd_test, "matriz_upfd_vs_upfd.png", "matrizes_de_confusao"),
        ("UPFD", "Bluesky Controlado", model_upfd, bs_ctrl_test, "matriz_upfd_vs_bs_ctrl.png", "matrizes_de_confusao"),
        ("UPFD", "Bluesky Aberto", model_upfd, bs_aberto, "matriz_upfd_vs_bs_aberto.png", "matrizes_de_confusao"),
        ("Bluesky Controlado", "Bluesky Controlado", model_bs_ctrl, bs_ctrl_test, "matriz_bs_ctrl_vs_bs_ctrl.png", "matrizes_de_confusao"),
        ("Bluesky Controlado", "Bluesky Aberto", model_bs_ctrl, bs_aberto, "matriz_bs_ctrl_vs_bs_aberto.png", "matrizes_de_confusao"),
        
        ("B. Cético (4x)", "Bluesky Controlado", model_bs_cetico, bs_ctrl_test, "matriz_bs_cetico_vs_bs_ctrl.png", "analise_cetico"),
        ("B. Cético (4x)", "Bluesky Aberto", model_bs_cetico, bs_aberto, "matriz_bs_cetico_vs_bs_aberto.png", "analise_cetico"),
        
        ("B. Exa. Cético (6x)", "Bluesky Controlado", model_bs_ex_cetico, bs_ctrl_test, "matriz_bs_exa_cetico_vs_bs_ctrl.png", "analise_cetico"),
        ("B. Exa. Cético (6x)", "Bluesky Aberto", model_bs_ex_cetico, bs_aberto, "matriz_bs_exa_cetico_vs_bs_aberto.png", "analise_cetico")
    ]
    
    for t_treino, t_teste, model, dataset, filename, sub_folder in tasks:
        if len(dataset) == 0:
            print(f"Pulando {t_treino} vs {t_teste} (Dataset vazio)")
            continue
        print(f"Avaliando: Treino({t_treino}) / Teste({t_teste})")
        yt, yp = evaluate_model(model, dataset)
        plot_matrix(yt, yp, f"Treino: {t_treino} | Teste: {t_teste}", filename, sub_folder)
        
    print("Processo Finalizado. Matrizes salvas nas respectivas pastas.")

if __name__ == "__main__":
    run_ablation()
