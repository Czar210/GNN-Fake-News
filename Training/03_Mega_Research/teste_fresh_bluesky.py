import os
import sys
import torch
import pandas as pd
from dotenv import load_dotenv

# Adiciona o path para o Blue Sky src
raiz_projeto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(raiz_projeto, "Training", "01_BlueSky_Pipe", "src"))
from collection import collect

def testar_fresh():
    print("Teste de Dados Frescos Restaurado.")
    # Lógica para coletar posts de HOJE e testar no modelo
    pass

if __name__ == "__main__":
    testar_fresh()
