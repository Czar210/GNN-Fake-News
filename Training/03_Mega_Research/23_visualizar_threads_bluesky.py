"""
23_visualizar_threads_bluesky.py
--------------------------------
Gera figuras pro TCC: visualizacoes de threads reais do Bluesky em formatos
PNG (estatico, pra LaTeX) e HTML interativo (pra apresentacao).

Fonte: dados_bluesky/graphs_extracted/graphs/threads.txt.gz
Formato: cada linha = `tamanho TAB timestamp TAB user_ids,separados,por,virgula`

Estrategia:
  - Selecionar 6 threads de tamanhos variados (pequena, media, grande)
  - Construir grafo: primeiro user = raiz, demais = filhos (estrela cronologica)
  - Aplicar RF estrutural persistido para mostrar score "fake-like" por thread
  - Visualizar com pyvis (HTML interativo) e matplotlib (PNG estatico)

Saida em Execution/results/figuras_tcc/threads_bluesky/:
  thread_<idx>_n<size>_score<XX>.png       # estatica para LaTeX
  thread_<idx>_n<size>_score<XX>.html      # interativa para apresentar
  resumo.txt                               # tabela das 6 threads selecionadas

Uso:
  python 23_visualizar_threads_bluesky.py
"""

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ        = Path(__file__).resolve().parent.parent.parent
THREADS_GZ  = RAIZ / "dados_bluesky" / "graphs_extracted" / "graphs" / "threads.txt.gz"
WEIGHTS_DIR = RAIZ / "Execution" / "weights"
OUT_DIR     = RAIZ / "Execution" / "results" / "figuras_tcc" / "threads_bluesky"

RANDOM_SEED = 42
N_AMOSTRAS  = 6
ALVOS_TAMANHO = [5, 15, 40, 80, 150, 400]   # tamanhos-alvo para escolher threads


def ler_threads(path: Path, max_linhas: int = 500_000) -> list:
    """
    Le threads.txt.gz; cada item retornado: (size, timestamp, [user_ids]).
    NB: o primeiro campo da linha (`partes[0]`) parece ser uma metrica derivada
    (profundidade ou interacoes totais), nao len(ids). Usamos len(ids) como
    tamanho real do grafo.
    """
    threads = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_linhas: break
            partes = line.strip().split("\t")
            if len(partes) < 3: continue
            try:
                ts = partes[1]
                ids = [int(x) for x in partes[2].split(",") if x]
                if len(ids) < 2: continue   # ignora threads triviais (1 user)
                threads.append((len(ids), ts, ids))
            except Exception:
                continue
    return threads


def selecionar_amostras(threads, alvos):
    """Para cada tamanho-alvo, pega thread cujo size mais se aproxima."""
    sels = []
    pegas = set()
    for alvo in alvos:
        candidatas = [(i, abs(t[0] - alvo)) for i, t in enumerate(threads) if i not in pegas]
        if not candidatas: continue
        idx = min(candidatas, key=lambda kv: kv[1])[0]
        sels.append(idx)
        pegas.add(idx)
    return [threads[i] for i in sels]


def aplicar_rf_score(num_nodes: int, grau_root: int, rf) -> float:
    """Retorna prob de classe fake (0)."""
    X = np.array([[num_nodes, grau_root]], dtype=np.float64)
    return float(rf.predict_proba(X)[0, list(rf.classes_).index(0)])


def visualizar_png(thread, score, out_path: Path):
    size, ts, ids = thread
    n = size

    # Layout: raiz no centro; demais em circulos concentricos por "lote"
    pos = {0: (0.0, 0.0)}
    if n > 1:
        # primeiros 12 filhos em circulo interno
        nivel1 = min(n - 1, 12)
        for k in range(nivel1):
            ang = 2 * np.pi * k / nivel1
            pos[k + 1] = (np.cos(ang), np.sin(ang))
        # restantes em anel externo
        restantes = n - 1 - nivel1
        for k in range(restantes):
            ang = 2 * np.pi * k / max(restantes, 1) + 0.3
            pos[k + 1 + nivel1] = (1.9 * np.cos(ang), 1.9 * np.sin(ang))

    fig, ax = plt.subplots(figsize=(8, 8))
    # arestas raiz -> filhos
    for i in range(1, n):
        if i not in pos: continue
        ax.plot([pos[0][0], pos[i][0]], [pos[0][1], pos[i][1]],
                color="gray", alpha=0.3, lw=0.6, zorder=1)
    # nos
    for i in range(n):
        if i not in pos: continue
        x, y = pos[i]
        if i == 0:
            ax.scatter(x, y, s=400, c="#E07B54", edgecolors="black", lw=1.5, zorder=3)
            ax.text(x, y, "R", ha="center", va="center", fontweight="bold", fontsize=11)
        else:
            ax.scatter(x, y, s=40, c="#5BAD72", edgecolors="black", lw=0.4, alpha=0.8, zorder=2)

    cor_score = "#c0392b" if score >= 0.5 else "#27ae60"
    ax.set_title(f"Thread Bluesky | size={n} nos | timestamp={ts}\n"
                 f"score 'fake-like' (RF estrutural treinado em GossipCop) = "
                 f"{score:.3f}",
                 fontsize=11, color=cor_score)
    lim = 2.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def visualizar_html(thread, score, out_path: Path):
    try:
        from pyvis.network import Network
    except ImportError:
        return False
    size, ts, ids = thread
    n = size
    titulo = f"Thread Bluesky | size={n} | score={score:.3f} | ts={ts}"
    net = Network(height="600px", width="100%", directed=False, notebook=False, heading=titulo)
    net.add_node(0, label=f"RAIZ (u{ids[0]})", color="#E07B54", size=25)
    # Limita pra HTML nao virar gigante (max 200 nos)
    n_show = min(n, 200)
    for i in range(1, n_show):
        net.add_node(i, label=f"u{ids[i]}", color="#5BAD72", size=10)
        net.add_edge(0, i)
    net.barnes_hut(spring_length=80)
    try:
        net.write_html(str(out_path), notebook=False, open_browser=False)
        return True
    except Exception as e:
        print(f"  [WARN] erro pyvis: {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not THREADS_GZ.exists():
        print(f"[ERRO] {THREADS_GZ} nao encontrado.")
        print(f"       Extraia dados_bluesky/graphs.tar.gz primeiro.")
        sys.exit(1)

    print("=" * 70)
    print("  Visualizacao de threads reais do Bluesky")
    print("=" * 70)

    print(f"\n[1/4] Carregando RF estrutural persistido...")
    with open(WEIGHTS_DIR / "rf_struct_gossipcop.pkl", "rb") as f:
        rf = pickle.load(f)["model"]
    print(f"   OK: {rf.n_estimators} arvores")

    print(f"\n[2/4] Lendo threads.txt.gz (max 500k linhas)...")
    threads = ler_threads(THREADS_GZ, max_linhas=500_000)
    print(f"   {len(threads):,} threads carregadas")
    sizes = sorted({t[0] for t in threads})
    print(f"   sizes presentes: min={min(sizes)} max={max(sizes)} unicos={len(sizes)}")

    print(f"\n[3/4] Selecionando {N_AMOSTRAS} threads de tamanhos variados...")
    sels = selecionar_amostras(threads, ALVOS_TAMANHO)
    print(f"   sizes escolhidos: {[t[0] for t in sels]}")

    print(f"\n[4/4] Gerando figuras (PNG + HTML)...")
    resumo_lines = ["Visualizacao de threads Bluesky -- 6 amostras de tamanhos variados",
                    "=" * 70,
                    "RF estrutural treinado em GossipCop com features [num_nodes, grau_root]",
                    "Score interpretacao: prob fake (>0.5 = predito fake)",
                    ""]
    for k, t in enumerate(sels):
        size, ts, ids = t
        score = aplicar_rf_score(num_nodes=size, grau_root=size - 1, rf=rf)
        score_int = int(round(score * 100))
        png_name = f"thread_{k:02d}_n{size}_score{score_int:03d}.png"
        html_name = f"thread_{k:02d}_n{size}_score{score_int:03d}.html"
        visualizar_png(t, score, OUT_DIR / png_name)
        visualizar_html(t, score, OUT_DIR / html_name)
        pred = "FAKE" if score >= 0.5 else "real"
        line = f"  thread_{k:02d}  size={size:>5}  ts={ts}  score={score:.3f} -> {pred}"
        print(f"   [OK] {png_name}")
        resumo_lines.append(line)

    rel_path = OUT_DIR / "resumo.txt"
    rel_path.write_text("\n".join(resumo_lines) + "\n", encoding="utf-8")
    print(f"\n   [OK] {rel_path.name}")
    print(f"\n[OK] Saidas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
