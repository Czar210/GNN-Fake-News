import os
import sys
import uuid
import json
import re

# PRIMEIRO: importar torch e torch_geometric da venv
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer

# DEPOIS: adicionar os caminhos do projeto
api_dir = os.path.dirname(os.path.abspath(__file__))
raiz = os.path.dirname(os.path.dirname(os.path.dirname(api_dir)))  # GNN Fake News/
sys.path.append(api_dir)
sys.path.append(os.path.join(raiz, "Training", "01_BlueSky_Pipe", "src"))

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd

from database import get_db, AnaliseHistory
from collection import collect
from features import text_embedder

app = FastAPI(title="Truth GNN Analytics - Backend", description="Motor de detecções em Real-time")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modelo GCN ───────────────────────────────────────────────────────────────

# Wrapper sem o return duplo (out, h) — necessário para o GNNExplainer
class _GCNWrapper(torch.nn.Module):
    def __init__(self, model: "GCNClassifier"):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        out, _ = self.model(x, edge_index, batch)
        return out


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

# ─── Carregamento dos modelos ─────────────────────────────────────────────────

model_upfd      = GCNClassifier(768, 2)
model_bs_ctrl   = GCNClassifier(768, 2)
model_bs_cetico = GCNClassifier(768, 2)
model_bs_ex_cetico = GCNClassifier(768, 2)

def load_models():
    specs = [
        (model_upfd,         "pesos_gcn.pth"),
        (model_bs_ctrl,      "pesos_bs_ctrl.pth"),
        (model_bs_cetico,    "pesos_bs_cetico.pth"),
        (model_bs_ex_cetico, "pesos_bs_ex_cetico.pth"),
    ]
    for model, fname in specs:
        path = os.path.join(raiz, "Execution", "weights", fname)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

load_models()

# ─── GNNExplainer (usa o modelo mais cético como referência — peso 0.40) ─────

_explainer: Explainer | None = None

def _init_explainer():
    global _explainer
    try:
        _explainer = Explainer(
            model=_GCNWrapper(model_bs_ex_cetico),
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )
    except Exception as e:
        print(f"[WARN] GNNExplainer não inicializado: {e}")

_init_explainer()

# ─── Word-level attribution (leave-one-out sobre embedding BERT) ──────────────

def _compute_word_importance(text: str, base_emb) -> list:
    """
    Para cada palavra remove ela do texto, re-embeda com BERT e mede a distância
    cosseno ao embedding original. Maior distância → palavra mais importante
    semanticamente para o modelo.
    Limitado a 20 palavras para não travar o background task.
    """
    import numpy as np
    try:
        model = text_embedder.get_model()
        words = text.split()
        if len(words) < 2:
            return []

        words = words[:20]
        base_vec = base_emb / (np.linalg.norm(base_emb) + 1e-9)

        result = []
        for i, word in enumerate(words):
            if len(word) < 3:          # ignora stopwords curtas
                result.append({"word": word, "importance": 0.0})
                continue
            masked = " ".join(words[:i] + words[i + 1:])
            if not masked.strip():
                continue
            masked_emb = model.encode(masked)
            masked_vec = masked_emb / (np.linalg.norm(masked_emb) + 1e-9)
            delta = float(1.0 - np.dot(base_vec, masked_vec))
            result.append({"word": word, "importance": round(delta, 5)})

        return result  # ordem original do texto (frontend ordena para exibir top-N)
    except Exception:
        return []

# ─── Cache de stats do dataset ────────────────────────────────────────────────

_stats_cache: dict | None = None

load_dotenv(os.path.join(raiz, ".env"))

try:
    bsky_client = collect.login_bluesky()
except Exception:
    bsky_client = None

# ─── Constantes do ensemble ───────────────────────────────────────────────────

WEIGHTS = [0.10, 0.20, 0.30, 0.40]
MODELS  = [model_upfd, model_bs_ctrl, model_bs_cetico, model_bs_ex_cetico]
NAMES   = ["UPFD_Baseline", "Bluesky_Controlado", "Bluesky_Cetico", "Bluesky_Exa_Cetico"]

# ─── Schemas ──────────────────────────────────────────────────────────────────

class RequestURL(BaseModel):
    url: str

# ─── Lógica de inferência (roda em background thread) ────────────────────────

def _run_inference(url: str, task_id: str) -> None:
    """
    Executa toda a pipeline pesada (scrape → BERT → GNN) e salva
    o resultado no banco de dados. Chamado via BackgroundTasks.
    """
    from database import SessionLocal

    db = SessionLocal()
    try:
        post_text = "Texto não extraído."
        interacoes = 5

        # Scrape via AT Protocol
        if bsky_client:
            try:
                match = re.search(r"profile/([^/]+)/post/([^/]+)", url)
                if not match:
                    raise ValueError("URL inválida")
                handle = match.group(1)
                rkey   = match.group(2)

                if not handle.startswith("did:"):
                    res = bsky_client.com.atproto.identity.resolve_handle({"handle": handle})
                    did = res.did
                else:
                    did = handle

                uri    = f"at://{did}/app.bsky.feed.post/{rkey}"
                thread = bsky_client.app.bsky.feed.get_post_thread({"uri": uri, "depth": 10})
                post_text = thread.thread.post.record.text

                if hasattr(thread.thread, "replies") and thread.thread.replies:
                    interacoes = len(thread.thread.replies)
            except Exception as e:
                post_text = f"Simulação local — erro na extração: {str(e)[:50]}"
        else:
            post_text = "Scraper inativo (credenciais BSKY ausentes). Modo Simulação."

        # BERT embedding
        df_temp = pd.DataFrame([{"texto": post_text}])
        df_temp = text_embedder.gerar_embeddings_de_texto(df_temp)
        emb = df_temp["embedding"].iloc[0]

        # Word importance (calcula antes de converter para tensor)
        word_importance = _compute_word_importance(post_text, emb)

        # Construção do grafo
        x_raiz   = torch.tensor(emb, dtype=torch.float)
        num_nodos = 1 + interacoes
        x        = torch.stack([x_raiz] * num_nodos)

        if interacoes > 0:
            edge_index = torch.tensor([[0] * interacoes, list(range(1, num_nodos))], dtype=torch.long)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, batch=torch.zeros(num_nodos, dtype=torch.long))

        # Inferência ensemble
        probs = []
        with torch.no_grad():
            for m in MODELS:
                out, _ = m(data.x, data.edge_index, data.batch)
                prob = torch.softmax(out, dim=1)[:, 1].item()
                probs.append(prob)

        ensemble_score = sum(w * p for w, p in zip(WEIGHTS, probs))
        is_fake        = bool(ensemble_score >= 0.5)

        # ── GNNExplainer ──────────────────────────────────────────────────────
        graph_explanation = None
        if _explainer is not None and data.edge_index.size(1) > 0:
            try:
                explanation = _explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                )
                raw_mask = explanation.edge_mask.detach().tolist()

                # Normaliza para [0, 1] para melhor visualização
                mn, mx = min(raw_mask), max(raw_mask)
                if mx > mn:
                    norm_mask = [(v - mn) / (mx - mn) for v in raw_mask]
                else:
                    norm_mask = [0.5] * len(raw_mask)

                src_nodes = data.edge_index[0].tolist()
                dst_nodes = data.edge_index[1].tolist()
                edges = [
                    {"from": int(s), "to": int(d), "importance": round(float(imp), 4)}
                    for s, d, imp in zip(src_nodes, dst_nodes, norm_mask)
                ]
                edges.sort(key=lambda e: e["importance"], reverse=True)

                graph_explanation = json.dumps({
                    "num_nodes": num_nodos,
                    "edges": edges,
                    "word_importance": word_importance,
                })
            except Exception as ex:
                graph_explanation = json.dumps({
                    "error": str(ex)[:120],
                    "word_importance": word_importance,
                })

        # Persiste no banco
        registro = db.query(AnaliseHistory).filter(AnaliseHistory.task_id == task_id).first()
        if registro:
            registro.status         = "done"
            registro.texto_resumo   = post_text[:200]
            registro.tamanho_grafo  = num_nodos
            registro.pred_upfd      = "Fake" if probs[0] >= 0.5 else "Real"
            registro.cert_upfd      = float(probs[0])
            registro.pred_bsky        = "Fake" if is_fake else "Real"
            registro.cert_bsky        = float(ensemble_score)
            registro.heuristica_final = float(ensemble_score)
            registro.graph_explanation = graph_explanation
            db.commit()

        # Salva fakes detectados em JSON
        if is_fake:
            fakes_file = os.path.join(api_dir, "fakes_testados.json")
            fakes_data = []
            if os.path.exists(fakes_file):
                with open(fakes_file, "r", encoding="utf-8") as f:
                    try:
                        fakes_data = json.load(f)
                    except Exception:
                        pass

            fakes_data.append({
                "url":            url,
                "texto":          post_text,
                "score_veredicto": round(ensemble_score, 4),
                "modelos":        {n: round(p, 4) for n, p in zip(NAMES, probs)},
            })
            with open(fakes_file, "w", encoding="utf-8") as f:
                json.dump(fakes_data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        # Em caso de erro inesperado, marca a task como falha
        registro = db.query(AnaliseHistory).filter(AnaliseHistory.task_id == task_id).first()
        if registro:
            registro.status       = "error"
            registro.texto_resumo = f"Erro interno: {str(e)[:200]}"
            db.commit()
    finally:
        db.close()

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_link(req: RequestURL, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Enfileira a análise e retorna um task_id imediatamente."""
    task_id = str(uuid.uuid4())

    # Cria registro no banco com status "processing"
    registro = AnaliseHistory(
        task_id   = task_id,
        status    = "processing",
        url_bsky  = req.url,
    )
    db.add(registro)
    db.commit()

    # Dispara inferência em background (thread pool — não bloqueia a UI)
    background_tasks.add_task(_run_inference, req.url, task_id)

    return {"status": "processing", "task_id": task_id}


@app.get("/api/result/{task_id}")
async def get_result(task_id: str, db: Session = Depends(get_db)):
    """Retorna o resultado quando pronto, ou status 'processing'/'error'."""
    registro = db.query(AnaliseHistory).filter(AnaliseHistory.task_id == task_id).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Task não encontrada")

    if registro.status != "done":
        return {"status": registro.status, "task_id": task_id}

    cert   = registro.cert_bsky or 0.0
    is_fake = (registro.heuristica_final or 0) >= 0.5

    exp = None
    if registro.graph_explanation:
        try:
            exp = json.loads(registro.graph_explanation)
        except Exception:
            pass

    return {
        "status":         "done",
        "task_id":        task_id,
        "url":            registro.url_bsky,
        "texto":          registro.texto_resumo,
        "is_fake":        is_fake,
        "ensemble_score": cert,
        "model_breakdown": [
            {"model": registro.pred_upfd or "—",  "prob_fake": registro.cert_upfd or 0,  "weight": 0.10},
            {"model": registro.pred_bsky or "—",  "prob_fake": cert,                      "weight": 0.90},
        ],
        "nodes":             registro.tamanho_grafo or 1,
        "graph_explanation": exp,
    }


@app.get("/api/stats")
async def get_dataset_stats():
    """Retorna estatísticas do dataset de treinamento (posts_coletados.csv)."""
    global _stats_cache
    if _stats_cache:
        return _stats_cache

    posts_path   = os.path.join(raiz, "Training", "01_BlueSky_Pipe", "data", "raw", "posts_coletados.csv")
    reposts_path = os.path.join(raiz, "Training", "01_BlueSky_Pipe", "data", "raw", "reposts_coletados.csv")

    if not os.path.exists(posts_path):
        return {"available": False}

    df = pd.read_csv(posts_path)
    total = len(df)
    fakes = int(df["label"].sum()) if "label" in df.columns else 0

    por_feed = []
    if "feed" in df.columns:
        por_feed = [
            {"feed": k, "count": int(v)}
            for k, v in df["feed"].value_counts().head(11).items()
        ]

    # Conta reposts sem carregar tudo na memória
    total_reposts = 0
    if os.path.exists(reposts_path):
        with open(reposts_path, encoding="utf-8") as f:
            total_reposts = sum(1 for _ in f) - 1  # desconta cabeçalho

    media_likes  = round(float(df["likes"].mean()), 1)  if "likes"   in df.columns else 0
    media_reposts = round(float(df["reposts"].mean()), 1) if "reposts" in df.columns else 0

    _stats_cache = {
        "available":      True,
        "total_posts":    total,
        "fakes":          fakes,
        "reais":          total - fakes,
        "pct_fake":       round(100 * fakes / total, 1) if total else 0,
        "total_reposts":  total_reposts,
        "media_likes":    media_likes,
        "media_reposts":  media_reposts,
        "por_feed":       por_feed,
    }
    return _stats_cache


@app.get("/api/history")
async def get_history(db: Session = Depends(get_db)):
    return db.query(AnaliseHistory).filter(
        AnaliseHistory.status == "done"
    ).order_by(AnaliseHistory.id.desc()).limit(10).all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
