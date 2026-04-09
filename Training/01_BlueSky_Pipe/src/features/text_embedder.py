from sentence_transformers import SentenceTransformer
import pandas as pd

# Lazy loading do modelo para economizar memória nos deploys
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return _model

def gerar_embeddings_de_texto(df, coluna='texto'):
    model = get_model()
    embeddings = model.encode(df[coluna].tolist())
    df['embedding'] = list(embeddings)
    return df