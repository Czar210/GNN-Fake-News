"""
25_publicar_huggingface.py
--------------------------
Sobe os pesos persistidos pelo `17_persistir_modelos_finais.py` para o
Hugging Face Hub, gerando tambem um model card README.md no padrao do
HF (com metadata YAML no frontmatter).

Pre-requisitos:
  pip install huggingface_hub
  echo "HF_TOKEN=hf_xxx" >> .env
  ou: huggingface-cli login

Uso:
  python 25_publicar_huggingface.py --repo-id seu-user/gnn-fake-news-tcc
  python 25_publicar_huggingface.py --repo-id org/repo --private

O script:
  1. valida que os 3 pesos + metadata.json existem
  2. gera README.md (model card) no diretorio temporario
  3. cria o repo (se nao existir) e faz upload de tudo
  4. imprime URL final pra apresentar
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ        = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
ENV_PATH    = RAIZ / ".env"

ARQUIVOS_OBRIGATORIOS = [
    "metadata.json",
    "logreg_bert_fnn.pkl",
    "rf_struct_gossipcop.pkl",
    "sage_struct_gossipcop.pth",
]


def carregar_token() -> str | None:
    """Le HF_TOKEN do .env (se existir) ou da env do shell."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def gerar_model_card(meta: dict, repo_id: str) -> str:
    """Gera README.md no formato do HF Hub (YAML frontmatter + markdown)."""
    m = meta["modelos"]
    f1_logreg = m["logreg_bert_fnn"]["f1_macro_treino"]
    f1_rf     = m["rf_struct_gossipcop"]["f1_macro_test"]
    f1_sage   = m["sage_struct_gossipcop"]["f1_macro_test"]
    acc_rf    = m["rf_struct_gossipcop"]["accuracy_test"]
    acc_sage  = m["sage_struct_gossipcop"]["accuracy_test"]

    return f"""---
license: cc-by-4.0
language:
- pt
- en
tags:
- fake-news-detection
- graph-neural-networks
- gnn
- bluesky
- upfd
- fakenewsnet
datasets:
- KaiDMML/FakeNewsNet
- upfd
metrics:
- f1
- accuracy
library_name: pytorch
pipeline_tag: graph-ml
---

# GNN Fake News Detection — TCC ({repo_id})

Pesos persistidos do TCC **"Deteccao de Fake News com Redes Neurais de Grafos
no Bluesky"**. Tres classificadores complementares para deteccao binaria
(real vs fake) de noticias em redes sociais.

## Modelos

| Modelo                       | Tipo                       | Input                                  | F1-macro | Acc    |
|------------------------------|----------------------------|----------------------------------------|----------|--------|
| `logreg_bert_fnn.pkl`        | sklearn.LogisticRegression | BERT 768d (raiz da arvore)            | {f1_logreg:.3f}    | --     |
| `rf_struct_gossipcop.pkl`    | sklearn.RandomForest       | `[num_nodes, grau_root]` (2d)         | {f1_rf:.3f}    | {acc_rf:.3f}  |
| `sage_struct_gossipcop.pth`  | torch_geometric GraphSAGE  | `[is_root, grau_norm]` por no (2d)    | {f1_sage:.3f}    | {acc_sage:.3f}  |

### Quando usar cada um

- **`logreg_bert_fnn`** — quando voce tem o **texto** da noticia. Roda em
  CPU em milissegundos. Equivalente a GCN textual no FakeNewsNet (a feature
  da raiz ja carrega quase todo o sinal). Treinado em FakeNewsNet (KaiDMML).

- **`rf_struct_gossipcop`** — quando voce SO tem a **topologia** (sem texto;
  ex: lingua desconhecida, post deletado, restricoes de privacidade). Captura
  ~88% do sinal topologico que um GNN mais pesado capturaria. Interpretavel:
  duas features apenas.

- **`sage_struct_gossipcop`** — versao GNN da topologia. F1 +6pp sobre o RF
  no UPFD-GossipCop. Use quando precisar de **GNNExplainer** para mostrar
  quais arestas/nos a rede usou pra decidir (modo "explicabilidade").

### Regra de combinacao (textual + topologico)

Quando ambos disponiveis, derivamos a regra empirica do experimento
`20_textual_vs_topologico.py` (TCC, capitulo de Aplicacao):

```python
def combinar(score_textual, score_topo):
    pred_t = score_textual >= 0.5
    pred_k = score_topo    >= 0.5
    if pred_t == pred_k:                   # concordam: media simples
        return 0.5 * score_textual + 0.5 * score_topo
    return 0.8 * score_textual + 0.2 * score_topo  # discordam: confiar mais no texto
```

Justificativa: quando os dois discordam, F1 do textual = 0.91 vs F1 do
topologico = 0.08 (no UPFD-GossipCop test). Texto vence quase sempre na
discordancia, mas o topologico ainda vale 0.2 como sinal de duvida.

## Carregar e usar

### LogReg textual (sklearn)

```python
import pickle
from sentence_transformers import SentenceTransformer

with open("logreg_bert_fnn.pkl", "rb") as f:
    blob = pickle.load(f)
clf = blob["model"]                                    # LogisticRegression
encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

texto = "Cientistas confirmam que vacinas contem chips 5G..."
emb   = encoder.encode([texto])                        # (1, 768)
prob_fake = clf.predict_proba(emb)[0, list(clf.classes_).index(0)]
print(f"P(fake) = {{prob_fake:.3f}}")
```

### RF estrutural (sklearn) — sem texto

```python
import numpy as np
import pickle

with open("rf_struct_gossipcop.pkl", "rb") as f:
    blob = pickle.load(f)
rf = blob["model"]                                     # RandomForestClassifier

# X = [[num_nodes_total, grau_da_raiz]]
X = np.array([[42, 41]], dtype=np.float64)
prob_fake = rf.predict_proba(X)[0, list(rf.classes_).index(0)]
print(f"P(fake) = {{prob_fake:.3f}}")
```

### SAGE estrutural (PyTorch Geometric) — para GNNExplainer

```python
import torch
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.nn import Linear

class SAGEClassifier(torch.nn.Module):
    def __init__(self, in_channels=2, hidden=64, num_classes=2):
        super().__init__()
        self.c1 = SAGEConv(in_channels, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.c3 = SAGEConv(hidden, hidden)
        self.lin = Linear(hidden, num_classes)
    def forward(self, x, edge_index, batch):
        h = self.c1(x, edge_index).relu()
        h = self.c2(h, edge_index).relu()
        h = self.c3(h, edge_index).relu()
        return self.lin(global_mean_pool(h, batch))

model = SAGEClassifier()
model.load_state_dict(torch.load("sage_struct_gossipcop.pth", map_location="cpu"))
model.eval()
# input por no: x = [[is_root_bool, grau_normalizado_pelo_max_global]]
```

## Datasets de treino

- **FakeNewsNet** (PolitiFact subset) — KaiDMML, ~750 grafos
  (https://github.com/KaiDMML/FakeNewsNet)
- **UPFD GossipCop** — Dou et al. SIGIR 2021, ~5.5k grafos
  (https://arxiv.org/abs/2104.12259)

## Limitacoes honestas

1. **PolitiFact UPFD: topologia NAO funciona** (Cohen's d ~0 entre fake/real
   nas metricas estruturais). Estes pesos so se aplicam a dominios onde fake
   news viralizam diferente — confirmamos no GossipCop, nao no PolitiFact.
2. **Sem labels para Bluesky.** Os modelos sao aplicados em modo
   demonstracao; o capitulo experimental do TCC valida apenas em datasets
   com ground-truth.
3. **Reprodutibilidade nao bit-exact.** Ver `README.md` do repo principal —
   resultados variam na 3a casa decimal entre maquinas (CUDA, cuDNN, CPU).
4. **Regra 0.8/0.2 derivada de UM experimento** (UPFD-GossipCop). Generaliza
   plausivelmente mas nao foi validada em outros dominios.

## Citacao

Se voce usar estes pesos em pesquisa, cite o TCC e os papers originais
dos datasets:

```bibtex
@misc{{sibila_tcc_gnn_fake_news_2026,
  author       = {{Sibila, Cesar Augusto}},
  title        = {{Deteccao de Fake News com Redes Neurais de Grafos no Bluesky (TCC)}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}

@inproceedings{{dou2021upfd,
  title     = {{User Preference-aware Fake News Detection}},
  author    = {{Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao}},
  booktitle = {{SIGIR}},
  year      = {{2021}}
}}

@article{{shu2018fakenewsnet,
  title   = {{FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media}},
  author  = {{Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan}},
  journal = {{arXiv:1809.01286}},
  year    = {{2018}}
}}
```

## Licenca

Os pesos sao distribuidos sob **CC-BY-4.0**. Os datasets originais
seguem as licencas dos respectivos autores (FakeNewsNet, UPFD).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True,
                   help="Ex: cesar-sibila/gnn-fake-news-tcc")
    p.add_argument("--private", action="store_true",
                   help="Cria repo privado (default: publico)")
    p.add_argument("--token", default=None,
                   help="HF token; default: le do .env ou env HF_TOKEN")
    p.add_argument("--dry-run", action="store_true",
                   help="So gera o model card local, nao sobe nada")
    args = p.parse_args()

    print("=" * 70)
    print("  Publicacao no Hugging Face Hub")
    print("=" * 70)

    print("\n[1/4] Validando arquivos persistidos...")
    faltando = [a for a in ARQUIVOS_OBRIGATORIOS if not (WEIGHTS_DIR / a).exists()]
    if faltando:
        print(f"   [ERRO] Faltando em {WEIGHTS_DIR}: {faltando}")
        print(f"          Rode primeiro: python 17_persistir_modelos_finais.py")
        sys.exit(1)
    print(f"   OK: {len(ARQUIVOS_OBRIGATORIOS)} arquivos presentes")

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text(encoding="utf-8"))

    print("\n[2/4] Gerando model card (README.md)...")
    card = gerar_model_card(meta, args.repo_id)

    if args.dry_run:
        out = WEIGHTS_DIR / "README_HF_preview.md"
        out.write_text(card, encoding="utf-8")
        print(f"   [DRY-RUN] Card escrito em: {out}")
        print(f"             Use este preview pra revisar antes de subir.")
        return

    token = args.token or carregar_token()
    if not token:
        print("   [ERRO] HF_TOKEN nao encontrado. Adicione no .env ou passe --token")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("   [ERRO] pip install huggingface_hub")
        sys.exit(1)

    print(f"\n[3/4] Criando repo {args.repo_id} (private={args.private})...")
    api = HfApi(token=token)
    create_repo(args.repo_id, token=token, private=args.private,
                repo_type="model", exist_ok=True)
    print(f"   OK")

    print(f"\n[4/4] Fazendo upload (pesos + metadata + README)...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # copia pesos
        for nome in ARQUIVOS_OBRIGATORIOS:
            shutil.copy(WEIGHTS_DIR / nome, tmp_path / nome)
        # escreve model card
        (tmp_path / "README.md").write_text(card, encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload pesos finais TCC GNN Fake News",
        )

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n[OK] Publicado em: {url}")
    print(f"     Confira o model card antes de divulgar.")


if __name__ == "__main__":
    main()
