import os
import sys
import uuid
import json
import pickle
import re

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

# Caminhos do projeto
api_dir = os.path.dirname(os.path.abspath(__file__))
raiz = os.path.dirname(os.path.dirname(os.path.dirname(api_dir)))  # GNN Fake News/
sys.path.append(api_dir)
sys.path.append(os.path.join(raiz, "Training", "01_BlueSky_Pipe", "src"))
sys.path.append(os.path.join(raiz, "Training", "03_Mega_Research"))

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd

from database import get_db, AnaliseHistory
from collection import collect
from features import text_embedder
from sage_model import SAGEClassifier  # da pasta Training/03_Mega_Research

app = FastAPI(title="Truth GNN Analytics - Backend",
              description="Detecção de fake news com classificador dual (textual + topológico)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Carregamento dos modelos canônicos persistidos ──────────────────────────
# Os 3 modelos abaixo foram treinados pelo script
# Training/03_Mega_Research/17_persistir_modelos_finais.py
WEIGHTS_DIR = os.path.join(raiz, "Execution", "weights")

# 1. LogReg-BERT (textual): treinado em FakeNewsNet, F1m sanidade ≈ 0.95
_lr_textual = None
try:
    with open(os.path.join(WEIGHTS_DIR, "logreg_bert_fnn.pkl"), "rb") as f:
        _lr_textual = pickle.load(f)["model"]
    print(f"[OK] LogReg-BERT textual carregado")
except Exception as e:
    print(f"[WARN] LogReg textual não carregado: {e}")

# 2. RandomForest estrutural (topológico leve): treinado em GossipCop,
#    features = [num_nodes, grau_root]; F1m ≈ 0.75
_rf_topologico = None
try:
    with open(os.path.join(WEIGHTS_DIR, "rf_struct_gossipcop.pkl"), "rb") as f:
        _rf_topologico = pickle.load(f)["model"]
    print(f"[OK] RF estrutural topológico carregado")
except Exception as e:
    print(f"[WARN] RF topológico não carregado: {e}")

# 3. SAGE estrutural (topológico GNN, usado para GNNExplainer):
#    features = [is_root, grau_norm]; F1m ≈ 0.81
_sage_topologico = None
try:
    state = torch.load(os.path.join(WEIGHTS_DIR, "sage_struct_gossipcop.pth"),
                       map_location="cpu", weights_only=False)
    _sage_topologico = SAGEClassifier(state["input_dim"], 2)
    _sage_topologico.load_state_dict(state["state_dict"])
    _sage_topologico.eval()
    print(f"[OK] SAGE estrutural carregado")
except Exception as e:
    print(f"[WARN] SAGE estrutural não carregado: {e}")


# ─── GNNExplainer (sobre o SAGE estrutural) ──────────────────────────────────
# Wrapper sem return duplo — GNNExplainer espera só logits
class _SAGEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        out, _ = self.model(x, edge_index, batch)
        return out


_explainer = None
if _sage_topologico is not None:
    try:
        _explainer = Explainer(
            model=_SAGEWrapper(_sage_topologico),
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
        print(f"[OK] GNNExplainer inicializado")
    except Exception as e:
        print(f"[WARN] GNNExplainer não inicializado: {e}")

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

# ─── Combinacao dos scores (regra do experimento 20) ─────────────────────────
# F1 quando os 2 modelos concordam = 0.97 (alta confianca).
# F1 quando discordam: textual = 0.91 vs topologico = 0.08 -> texto vence.
# Por isso quando discordam damos peso 0.8 ao textual.
PESO_TEXTUAL_QUANDO_DISCORDA = 0.8


def _features_topologicas_de_grafo(num_nodes: int, num_edges: int):
    """Para o RF estrutural: [num_nodes, grau_root]."""
    grau_root = num_edges  # em estrela, grau da raiz = numero de filhos
    return np.array([[num_nodes, grau_root]], dtype=np.float64)


def _features_estruturais_para_sage(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Para o SAGE: [is_root, grau_norm] por no."""
    is_root = torch.zeros(num_nodes, dtype=torch.float)
    is_root[0] = 1.0
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_index.numel() > 0:
        idx, counts = torch.unique(edge_index[0], return_counts=True)
        deg[idx] = counts.float()
    deg_norm = deg / max(deg.max().item(), 1.0)
    return torch.stack([is_root, deg_norm], dim=1)


def _combinar_scores(score_textual: float, score_topo: float) -> tuple:
    """
    Retorna (score_combinado, pred_combinado, concordam, peso_aplicado).

    Convenção: score = prob de FAKE (classe 0). pred = 0 (fake) se score>=0.5.
    """
    pred_t = score_textual >= 0.5
    pred_k = score_topo >= 0.5
    concordam = pred_t == pred_k
    if concordam:
        # media aritmetica simples
        score = 0.5 * score_textual + 0.5 * score_topo
        peso  = 0.5
    else:
        # textual vence (peso 0.8)
        score = PESO_TEXTUAL_QUANDO_DISCORDA * score_textual + (1 - PESO_TEXTUAL_QUANDO_DISCORDA) * score_topo
        peso  = PESO_TEXTUAL_QUANDO_DISCORDA
    pred = score >= 0.5
    return float(score), bool(pred), bool(concordam), float(peso)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class RequestURL(BaseModel):
    url: str

# ─── Lógica de inferência (roda em background thread) ────────────────────────

def _count_thread_nodes(thread_node) -> tuple[int, list]:
    """
    Percorre a árvore de replies recursivamente e retorna:
      - total de nós filhos (excluindo a raiz)
      - lista de arestas (parent_idx, child_idx) para o grafo
    """
    edges = []
    counter = [0]  # mutable counter para closure

    def _walk(node, parent_idx: int):
        if not hasattr(node, "replies") or not node.replies:
            return
        for reply in node.replies:
            counter[0] += 1
            child_idx = counter[0]
            edges.append((parent_idx, child_idx))
            _walk(reply, child_idx)

    _walk(thread_node, 0)
    return counter[0], edges


def _run_inference(url: str, task_id: str) -> None:
    """
    Executa toda a pipeline pesada (scrape → BERT → GNN) e salva
    o resultado no banco de dados. Chamado via BackgroundTasks.
    """
    from database import SessionLocal

    db = SessionLocal()
    try:
        post_text = "Texto não extraído."
        interacoes = 0
        tree_edges = []

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

                interacoes, tree_edges = _count_thread_nodes(thread.thread)
            except Exception as e:
                post_text = f"Simulação local — erro na extração: {str(e)[:50]}"
        else:
            post_text = "Scraper inativo (credenciais BSKY ausentes). Modo Simulação."

        # BERT embedding do texto do post (raiz)
        df_temp = pd.DataFrame([{"texto": post_text}])
        df_temp = text_embedder.gerar_embeddings_de_texto(df_temp)
        emb = df_temp["embedding"].iloc[0]
        emb_np = np.asarray(emb, dtype=np.float64)

        # Word importance (leave-one-out)
        word_importance = _compute_word_importance(post_text, emb)

        # Constrói o grafo estrutural (mesma estratégia dos scripts 14/19)
        num_nodos = 1 + interacoes
        if tree_edges:
            src = [e[0] for e in tree_edges]
            dst = [e[1] for e in tree_edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        elif interacoes > 0:
            edge_index = torch.tensor([[0] * interacoes, list(range(1, num_nodos))], dtype=torch.long)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        # ── 1. Modelo TEXTUAL (LogReg-BERT sobre emb da raiz) ──────────────
        score_textual = 0.5
        if _lr_textual is not None and emb_np.shape[0] == _lr_textual.coef_.shape[1]:
            try:
                proba = _lr_textual.predict_proba(emb_np.reshape(1, -1))[0]
                idx_fake = list(_lr_textual.classes_).index(0)
                score_textual = float(proba[idx_fake])
            except Exception as ex:
                print(f"[WARN] textual inference falhou: {ex}")

        # ── 2. Modelo TOPOLÓGICO leve (RF [num_nodes, grau_root]) ──────────
        score_topo_rf = 0.5
        if _rf_topologico is not None:
            try:
                X_topo = _features_topologicas_de_grafo(num_nodos, interacoes)
                proba = _rf_topologico.predict_proba(X_topo)[0]
                idx_fake = list(_rf_topologico.classes_).index(0)
                score_topo_rf = float(proba[idx_fake])
            except Exception as ex:
                print(f"[WARN] RF topologico falhou: {ex}")

        # ── 3. Modelo TOPOLÓGICO GNN (SAGE estrutural) — usado para GNNExplainer
        score_topo_sage = 0.5
        data_sage = None
        if _sage_topologico is not None and num_nodos >= 1:
            try:
                x_sage = _features_estruturais_para_sage(num_nodos, edge_index)
                batch  = torch.zeros(num_nodos, dtype=torch.long)
                data_sage = Data(x=x_sage, edge_index=edge_index, batch=batch)
                with torch.no_grad():
                    out, _ = _sage_topologico(data_sage.x, data_sage.edge_index, data_sage.batch)
                    prob = torch.softmax(out, dim=1)[0, 0].item()  # classe 0 = fake
                score_topo_sage = float(prob)
            except Exception as ex:
                print(f"[WARN] SAGE topologico falhou: {ex}")

        # ── Combinação dual: textual vs topologico (regra do exp 20) ───────
        score_combinado, pred_combinado, concordam, peso_aplicado = _combinar_scores(
            score_textual, score_topo_rf)
        is_fake = pred_combinado

        # ── GNNExplainer (sobre SAGE com features estruturais) ─────────────
        graph_explanation = None
        if _explainer is not None and data_sage is not None and data_sage.edge_index.size(1) > 0:
            try:
                explanation = _explainer(
                    x=data_sage.x,
                    edge_index=data_sage.edge_index,
                    batch=data_sage.batch,
                )
                raw_mask = explanation.edge_mask.detach().tolist()
                mn, mx = min(raw_mask), max(raw_mask)
                norm_mask = [(v - mn) / (mx - mn) for v in raw_mask] if mx > mn else [0.5] * len(raw_mask)

                src_nodes = data_sage.edge_index[0].tolist()
                dst_nodes = data_sage.edge_index[1].tolist()
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
        elif word_importance:
            graph_explanation = json.dumps({
                "num_nodes": num_nodos,
                "edges": [],
                "word_importance": word_importance,
            })

        # ── Persistência (reuso dos campos legados do schema) ──────────────
        registro = db.query(AnaliseHistory).filter(AnaliseHistory.task_id == task_id).first()
        if registro:
            registro.status         = "done"
            registro.texto_resumo   = post_text[:200]
            registro.tamanho_grafo  = num_nodos
            # pred_upfd/cert_upfd -> Modelo TEXTUAL
            registro.pred_upfd      = "Fake" if score_textual >= 0.5 else "Real"
            registro.cert_upfd      = float(score_textual)
            # pred_bsky/cert_bsky -> Modelo TOPOLÓGICO (RF)
            registro.pred_bsky      = "Fake" if score_topo_rf >= 0.5 else "Real"
            registro.cert_bsky      = float(score_topo_rf)
            # heuristica_final -> Score COMBINADO
            registro.heuristica_final = float(score_combinado)
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
                "url":             url,
                "texto":           post_text,
                "score_combinado": round(score_combinado, 4),
                "concordam":       concordam,
                "peso_textual":    peso_aplicado,
                "modelos":         {
                    "textual_logreg_bert":  round(score_textual, 4),
                    "topologico_rf":        round(score_topo_rf, 4),
                    "topologico_sage":      round(score_topo_sage, 4),
                },
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

    score_textual = float(registro.cert_upfd or 0.0)
    score_topo    = float(registro.cert_bsky or 0.0)
    score_combinado = float(registro.heuristica_final or 0.0)
    is_fake = score_combinado >= 0.5

    # Calcula concordancia + peso aplicado ja
    pred_t = score_textual >= 0.5
    pred_k = score_topo >= 0.5
    concordam = pred_t == pred_k
    peso_textual = 0.5 if concordam else PESO_TEXTUAL_QUANDO_DISCORDA

    exp = None
    if registro.graph_explanation:
        try:
            exp = json.loads(registro.graph_explanation)
        except Exception:
            pass

    return {
        "status":           "done",
        "task_id":          task_id,
        "url":              registro.url_bsky,
        "texto":            registro.texto_resumo,
        "is_fake":          is_fake,
        "ensemble_score":   score_combinado,
        "score_combinado":  score_combinado,
        "concordam":        concordam,
        "peso_textual_aplicado": peso_textual,
        "model_breakdown": [
            {"model": "Textual (LogReg-BERT, treinado em FakeNewsNet)",
             "prob_fake": score_textual,
             "pred": "Fake" if pred_t else "Real",
             "peso": peso_textual},
            {"model": "Topologico (RF estrutural, treinado em GossipCop)",
             "prob_fake": score_topo,
             "pred": "Fake" if pred_k else "Real",
             "peso": 1 - peso_textual},
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
