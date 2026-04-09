import torch
import os
from torch_geometric.data import Data # Necessário para o unpickler encontrar a classe

path = r'c:\Users\cesar\Documents\GitHub\tcc-gnn-fake-news\politifact\processed\bert\train.pt'
obj = torch.load(path, weights_only=False)
print(f"Tipo do objeto: {type(obj)}")
if isinstance(obj, tuple):
    print(f"Tamanho da tupla: {len(obj)}")
    for i, item in enumerate(obj):
        print(f"Item {i}: {type(item)}")
elif isinstance(obj, list):
    print(f"Tamanho da lista: {len(obj)}")
    if len(obj) > 0:
        print(f"Tipo do primeiro elemento: {type(obj[0])}")
else:
    print(f"Objeto: {obj}")
