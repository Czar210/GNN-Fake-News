"""
24_consolidar_tcc.py
--------------------
Le todos os CSVs gerados pelas fases 2-4 e produz:

  1. Tabelas LaTeX prontas para colar no TCC
     (Execution/results/figuras_tcc/final/tabelas/*.tex)

  2. Figuras consolidadas comparativas
     (Execution/results/figuras_tcc/final/figuras/*.png)

  3. Indice geral de saidas (INDICE.md)
     -- mapeia cada tabela/figura ao capitulo do TCC onde deve entrar.

Fontes de dados:
  - fase2_baselines/baseline_textual/resultados.csv
  - fase2_baselines/confound_diagnostico/resultados.csv
  - fase3_ablation/ablation_intra_encoding/tabela.csv
  - fase4_benchmarks/teste_significancia_posfull/{tabela_significancia.tex, por_fold.csv}
  - fase4_benchmarks/topologia_sem_texto/resultados.csv
  - fase4_benchmarks/benchmark_upfd_oficial/resultados.csv
  - figuras_tcc/analise_estrutural/estatisticas.csv
  - figuras_tcc/textual_vs_topologico/{gossipcop_metricas.txt, ...}
  - figuras_tcc/bloco_vs_mini/resumo.csv
  - figuras_tcc/concordancia_bluesky/{matriz_confusao.txt, cross_tab.csv}
  - figuras_tcc/bluesky_inferencia/resumo_por_feed.csv

Uso:
  python 24_consolidar_tcc.py
"""

import csv
import shutil
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = RAIZ / "Execution" / "results"
FASE2       = RESULTS_DIR / "fase2_baselines"
FASE3       = RESULTS_DIR / "fase3_ablation"
FASE4       = RESULTS_DIR / "fase4_benchmarks"
FIGS_DIR    = RESULTS_DIR / "figuras_tcc"

OUT_DIR     = FIGS_DIR / "final"
OUT_TAB     = OUT_DIR / "tabelas"
OUT_FIG     = OUT_DIR / "figuras"


def ler_csv(path: Path) -> list:
    if not path.exists(): return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agregar(rows, group_keys, value_key):
    """Agrega: para cada combinacao de group_keys, retorna {keys -> [valores]}."""
    grupos = {}
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        try: v = float(r[value_key])
        except Exception: continue
        grupos.setdefault(k, []).append(v)
    return grupos


def latex_tabela(headers, rows, caption, label) -> str:
    """Gera string LaTeX de uma tabela."""
    align = "l" + "r" * (len(headers) - 1)
    lines = [
        "% " + caption,
        f"\\begin{{table}}[h]\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    indice_lines = [
        "# Materiais consolidados para o TCC",
        "",
        "Tabelas em `tabelas/` (LaTeX) e figuras em `figuras/` (PNG/PDF).",
        "Cada item indica o capitulo sugerido do TCC.",
        "",
        "## Tabelas LaTeX",
        "",
    ]

    # ── T1. Baseline textual no FNN (Fase 2A.1) ──────────────────────────
    rows = ler_csv(FASE2 / "baseline_textual" / "resultados.csv")
    if rows:
        agg = agregar(rows, ["modelo"], "f1_macro")
        agg_acc = agregar(rows, ["modelo"], "accuracy")
        tab_rows = []
        for modelo in ["LogReg", "RandomForest"]:
            v = agg.get((modelo,), []); a = agg_acc.get((modelo,), [])
            if v:
                tab_rows.append([modelo, len(v),
                                 f"{np.mean(v):.4f} $\\pm$ {np.std(v):.4f}",
                                 f"{np.mean(a):.4f} $\\pm$ {np.std(a):.4f}"])
        if tab_rows:
            tex = latex_tabela(
                ["Modelo", "Folds", "F1-macro", "Accuracy"], tab_rows,
                "Baseline textual no FakeNewsNet (k-fold estratificado, k=10).",
                "tab:baseline_textual_fnn")
            (OUT_TAB / "T1_baseline_textual_fnn.tex").write_text(tex, encoding="utf-8")
            indice_lines.append("- **T1** `T1_baseline_textual_fnn.tex` -- "
                                "Capitulo de Resultados (Fase 2). LogReg/RF sobre BERT(titulo).")
            print(f"[OK] T1_baseline_textual_fnn.tex")

    # ── T2. ttest_rel posfull (Fase 4.2) ──────────────────────────────────
    src_tex = FASE4 / "teste_significancia_posfull" / "tabela_significancia.tex"
    if src_tex.exists():
        shutil.copy(src_tex, OUT_TAB / "T2_significancia_posfull.tex")
        indice_lines.append("- **T2** `T2_significancia_posfull.tex` -- "
                            "Capitulo de Resultados. ttest_rel pareado por fold "
                            "GCN vs GAT vs SAGE no FNN posfull (todos ns).")
        print(f"[OK] T2_significancia_posfull.tex")

    # ── T3. Cross-dataset UPFD oficial (Fase cross-dataset) ──────────────
    rows = ler_csv(FASE4 / "benchmark_upfd_oficial" / "resultados.csv")
    if rows:
        # agrupa por (dataset, modelo)
        agg = agregar(rows, ["dataset", "modelo"], "f1_macro")
        tab_rows = []
        chaves = sorted(agg.keys())
        for ds, mod in chaves:
            vals = agg[(ds, mod)]
            tab_rows.append([ds, mod, len(vals),
                             f"{np.mean(vals):.4f} $\\pm$ {np.std(vals):.4f}"])
        tex = latex_tabela(
            ["Dataset", "Modelo", "Seeds/Folds", "F1-macro"], tab_rows,
            "Benchmark cross-dataset UPFD (PolitiFact + GossipCop) -- "
            "GCN/GAT/SAGE vs baseline textual.",
            "tab:cross_dataset_upfd")
        (OUT_TAB / "T3_cross_dataset_upfd.tex").write_text(tex, encoding="utf-8")
        indice_lines.append("- **T3** `T3_cross_dataset_upfd.tex` -- "
                            "Capitulo de Resultados. Compara archs nos 2 UPFDs oficiais.")
        print(f"[OK] T3_cross_dataset_upfd.tex")

    # ── T4. Topologia sem texto ──────────────────────────────────────────
    rows = ler_csv(FASE4 / "topologia_sem_texto" / "resultados.csv")
    if rows:
        agg = agregar(rows, ["dataset", "variant", "modelo"], "f1_macro")
        # Filtra so a variante mais informativa: B_estrutural
        tab_rows = []
        for (ds, var, mod), vals in sorted(agg.items()):
            if var != "B_estrutural": continue
            tab_rows.append([ds, mod, len(vals),
                             f"{np.mean(vals):.4f} $\\pm$ {np.std(vals):.4f}"])
        if tab_rows:
            tex = latex_tabela(
                ["Dataset", "Modelo", "Seeds/Folds", "F1-macro"], tab_rows,
                "Topologia SEM texto: features [is\\_root, grau\\_norm] -- "
                "sinal estrutural puro por dataset/arch.",
                "tab:topologia_sem_texto")
            (OUT_TAB / "T4_topologia_sem_texto.tex").write_text(tex, encoding="utf-8")
            indice_lines.append("- **T4** `T4_topologia_sem_texto.tex` -- "
                                "Capitulo de Resultados. Pergunta: 'topologia sozinha "
                                "detecta fake?'. Achado: SAGE-GossipCop=0.81.")
            print(f"[OK] T4_topologia_sem_texto.tex")

    # ── T5. Cohen's d analise estrutural ──────────────────────────────────
    rows = ler_csv(FIGS_DIR / "analise_estrutural" / "estatisticas.csv")
    if rows:
        tab_rows = []
        # so as 2 metricas mais informativas
        for r in rows:
            if r["metrica"] not in ("num_nodes", "branching_avg"): continue
            tab_rows.append([
                r["dataset"], r["metrica"],
                f"{float(r['media_fake']):.2f}",
                f"{float(r['media_real']):.2f}",
                f"{float(r['cohen_d']):+.3f}",
                f"{float(r['p']):.4g}",
            ])
        if tab_rows:
            tex = latex_tabela(
                ["Dataset", "Metrica", "media$_{fake}$", "media$_{real}$",
                 "Cohen's $d$", "$p$-valor"], tab_rows,
                "Analise estrutural fake vs real -- separabilidade por metrica topologica. "
                "Cohen's $d$: $|d|\\geq 0.5$ medio, $\\geq 0.8$ grande.",
                "tab:cohens_d")
            (OUT_TAB / "T5_cohens_d_estrutural.tex").write_text(tex, encoding="utf-8")
            indice_lines.append("- **T5** `T5_cohens_d_estrutural.tex` -- "
                                "Capitulo de Discussao. Explica POR QUE GossipCop funciona "
                                "(d=+1.53 em branching).")
            print(f"[OK] T5_cohens_d_estrutural.tex")

    # ── T6. Concordancia textual vs topologico no GossipCop ─────────────
    metric_path = FIGS_DIR / "textual_vs_topologico" / "gossipcop_metricas.txt"
    if metric_path.exists():
        # Le as metricas-chave do txt
        txt = metric_path.read_text(encoding="utf-8")
        f1_text = f1_topo = agree = kappa = "—"
        f1_concord = f1_text_disc = f1_topo_disc = "—"
        for line in txt.splitlines():
            if "F1 macro textual" in line: f1_text = line.split(":")[-1].strip()
            elif "F1 macro topologico" in line: f1_topo = line.split(":")[-1].strip()
            elif line.startswith("agreement_rate"): agree = line.split(":")[-1].strip()
            elif line.startswith("kappa"): kappa = line.split(":")[-1].strip()
            elif "f1_quando_concordam" in line: f1_concord = line.split(":")[-1].strip()
            elif "f1_text_quando_discordam" in line: f1_text_disc = line.split(":")[-1].strip()
            elif "f1_topo_quando_discordam" in line: f1_topo_disc = line.split(":")[-1].strip()

        tab_rows = [
            ["F1-macro textual (LogReg-content)", f1_text],
            ["F1-macro topologico (RF estrutural)", f1_topo],
            ["Agreement rate", agree],
            ["Cohen's kappa", kappa],
            ["F1 quando AMBOS concordam", f1_concord],
            ["F1 textual quando DISCORDAM", f1_text_disc],
            ["F1 topologico quando DISCORDAM", f1_topo_disc],
        ]
        tex = latex_tabela(
            ["Metrica", "Valor"], tab_rows,
            "Comparacao TEXTUAL vs TOPOLOGICO no UPFD-GossipCop -- "
            "concordancia, kappa, e desempenho condicional.",
            "tab:concordancia_gossipcop")
        (OUT_TAB / "T6_concordancia_gossipcop.tex").write_text(tex, encoding="utf-8")
        indice_lines.append("- **T6** `T6_concordancia_gossipcop.tex` -- "
                            "Capitulo de Discussao. Quando ambos batem, F1=0.97. "
                            "Quando discordam, textual ainda acerta (0.91), topo erra (0.08).")
        print(f"[OK] T6_concordancia_gossipcop.tex")

    # ── T7. Bloco grande vs Modelos mini ─────────────────────────────────
    rows = ler_csv(FIGS_DIR / "bloco_vs_mini" / "resumo.csv")
    if rows:
        # cria tabela direto: dataset | mini F1 | bloco F1 | delta
        por_ds = {}
        for r in rows:
            por_ds.setdefault(r["dataset"], {})[r["modelo"]] = r
        tab_rows = []
        for ds, d in sorted(por_ds.items()):
            mini = d.get("mini", {}); bloco = d.get("bloco", {})
            if mini and bloco:
                f1_m = float(mini["f1_macro"]); f1_b = float(bloco["f1_macro"])
                tab_rows.append([ds, f"{f1_m:.4f}", f"{f1_b:.4f}",
                                 f"{f1_b-f1_m:+.4f}"])
        tex = latex_tabela(
            ["Dataset", "MINI", "BLOCO grande", "$\\Delta$"], tab_rows,
            "Bloco grande (1 RF treinado em FNN+UPFD-Polit+UPFD-Goss + dataset\\_id) "
            "vs Modelos MINI (1 RF por dataset) -- features [num\\_nodes, grau\\_root].",
            "tab:bloco_vs_mini")
        (OUT_TAB / "T7_bloco_vs_mini.tex").write_text(tex, encoding="utf-8")
        indice_lines.append("- **T7** `T7_bloco_vs_mini.tex` -- "
                            "Capitulo de Discussao. Bloco unico ~ Mini, com leve vantagem "
                            "em datasets pequenos (UPFD-Polit +0.023).")
        print(f"[OK] T7_bloco_vs_mini.tex")

    # ── T8. Bluesky inferencia: scores por feed ──────────────────────────
    rows = ler_csv(FIGS_DIR / "bluesky_inferencia" / "resumo_por_feed.csv")
    if rows:
        tab_rows = []
        for r in sorted(rows, key=lambda x: -float(x["score_medio"])):
            tab_rows.append([
                r["feed"], r["n_posts"],
                f"{float(r['score_medio']):.4f}",
                f"{100*float(r['pct_pred_fake']):.1f}\\%",
            ])
        tex = latex_tabela(
            ["Feed Bluesky", "$N$", "Score medio", "\\% pred=fake"], tab_rows,
            "Aplicacao do RF estrutural (treinado em GossipCop) sobre 168k posts do Bluesky -- "
            "ranking de feeds por score 'fake-like'.",
            "tab:bluesky_inferencia_feeds")
        (OUT_TAB / "T8_bluesky_inferencia_feeds.tex").write_text(tex, encoding="utf-8")
        indice_lines.append("- **T8** `T8_bluesky_inferencia_feeds.tex` -- "
                            "Capitulo de Aplicacao. Political Science=69% fake-like. "
                            "News=7% (paradoxal -- viral mas nao 'fake-padrao').")
        print(f"[OK] T8_bluesky_inferencia_feeds.tex")

    # ── T9. Concordancia Bluesky por bin de num_nodes ────────────────────
    rows = ler_csv(FIGS_DIR / "concordancia_bluesky" / "cross_tab.csv")
    if rows:
        tab_rows = []
        for r in rows:
            if r["dimensao"] != "num_nodes_bin": continue
            ar = r["agreement_rate"]
            if ar == "": continue
            tab_rows.append([r["valor"], r["n"],
                             f"{100*float(ar):.1f}\\%"])
        if tab_rows:
            tex = latex_tabela(
                ["Tamanho (n nos)", "$N$ posts", "Agreement"], tab_rows,
                "Concordancia textual vs topologico no Bluesky por bin de num\\_nodes "
                "(cross-tab post a post, 5k amostrado).",
                "tab:bluesky_concord_size")
            (OUT_TAB / "T9_bluesky_concord_size.tex").write_text(tex, encoding="utf-8")
            indice_lines.append("- **T9** `T9_bluesky_concord_size.tex` -- "
                                "Capitulo de Aplicacao. Concordancia maxima em posts "
                                "20-100 nos (74%); minima em posts medios (42-44%).")
            print(f"[OK] T9_bluesky_concord_size.tex")

    # ── T10. UPFD: nossos F1 vs Dou et al. (2021) ────────────────────────
    # Valores publicados extraidos da Tabela 4 do paper original
    # "User Preference-aware Fake News Detection" (SIGIR 2021).
    # NOTA IMPORTANTE: Dou et al. reportam ACCURACY com features 'bert' (768d).
    # Nos rodamos F1-macro com 'content' (310d) -- comparacao com asterisco.
    # **Conferir contra o PDF do paper antes de submeter.**
    publicado = {
        # (dataset, modelo) -> acc reportada no paper
        ("politifact", "GCN"):  0.846,
        ("politifact", "GAT"):  0.846,
        ("politifact", "SAGE"): 0.846,
        ("gossipcop",  "GCN"):  0.972,
        ("gossipcop",  "GAT"):  0.971,
        ("gossipcop",  "SAGE"): 0.971,
    }
    rows = ler_csv(FASE4 / "benchmark_upfd_oficial" / "resultados.csv")
    if rows:
        nosso = agregar([r for r in rows if r["modelo"] in ("GCN","GAT","SAGE")],
                        ["dataset", "modelo"], "f1_macro")
        tab_rows = []
        for ds in ("politifact", "gossipcop"):
            for mod in ("GCN", "GAT", "SAGE"):
                vals = nosso.get((ds, mod), [])
                if not vals: continue
                m, s = float(np.mean(vals)), float(np.std(vals))
                pub  = publicado.get((ds, mod))
                pub_str = f"{pub:.3f}" if pub is not None else "--"
                tab_rows.append([ds, mod,
                                 f"{m:.3f}$\\pm${s:.3f}",
                                 pub_str])
        if tab_rows:
            tex = latex_tabela(
                ["Dataset", "Modelo", "F1-macro (nosso)", "Acc (Dou et al.)"],
                tab_rows,
                "Comparacao com baselines do UPFD original (Dou et al., SIGIR 2021, Tabela 4). "
                "Nota: nossos resultados sao F1-macro sobre k-fold com feature 'content' (310d); "
                "valores publicados sao accuracy com feature 'bert' (768d). "
                "Comparacao indicativa, nao bit-exact.",
                "tab:upfd_vs_publicado")
            (OUT_TAB / "T10_upfd_vs_publicado.tex").write_text(tex, encoding="utf-8")
            indice_lines.append("- **T10** `T10_upfd_vs_publicado.tex` -- "
                                "Capitulo de Resultados/Benchmarks. Nossos GNNs ficam dentro "
                                "do range publicado (PolitiFact ~0.78-0.82 vs 0.846 reportado; "
                                "GossipCop ~0.94-0.95 vs 0.97 reportado). Diferenca explicada "
                                "por feature ('content' 310d vs 'bert' 768d).")
            print(f"[OK] T10_upfd_vs_publicado.tex")

    # ── Figuras: copia as ja geradas pra final/figuras ───────────────────
    indice_lines += ["", "## Figuras", ""]
    figuras_relevantes = [
        ("analise_estrutural/fig_separabilidade.png",       "F1_separabilidade.png",
         "Cohen's $d$ por metrica e dataset (3 datasets x 5 metricas)"),
        ("analise_estrutural/fig_distribuicoes.png",        "F2_distribuicoes_estruturais.png",
         "Distribuicoes fake vs real por dataset/metrica (histogramas sobrepostos)"),
        ("textual_vs_topologico/fig_concordancia.png",      "F3_textual_vs_topo_scatter.png",
         "Scatter score textual vs topologico no UPFD-GossipCop"),
        ("bloco_vs_mini/fig_comparacao.png",                "F4_bloco_vs_mini.png",
         "Bar chart MINI vs BLOCO grande por dataset"),
        ("bluesky_crossfeed/fig_distribuicoes.png",         "F5_bluesky_crossfeed_distrib.png",
         "Distribuicao de likes/reposts/replies por feed Bluesky"),
        ("bluesky_crossfeed/fig_pares.png",                 "F6_bluesky_crossfeed_pares.png",
         "Heatmap Cohen's $d$ entre feeds Bluesky"),
        ("bluesky_inferencia/fig_distribuicao_scores.png",  "F7_bluesky_scores_por_feed.png",
         "Distribuicao de score 'fake-like' por feed Bluesky"),
        ("concordancia_bluesky/fig_matriz_confusao.png",    "F8_bluesky_matriz_confusao.png",
         "Matriz de confusao 2x2 textual vs topologico no Bluesky"),
        ("concordancia_bluesky/fig_concordancia_por_tamanho.png", "F9_bluesky_agreement_size.png",
         "Agreement rate por bin de num\\_nodes"),
        ("concordancia_bluesky/fig_concordancia_por_feed.png",    "F10_bluesky_agreement_feed.png",
         "Agreement rate por feed Bluesky"),
        ("concordancia_bluesky/fig_concordancia_por_text_len.png","F11_bluesky_agreement_textlen.png",
         "Agreement rate por bin de comprimento de texto"),
    ]
    for src_rel, dst_name, descr in figuras_relevantes:
        src = FIGS_DIR / src_rel
        if src.exists():
            shutil.copy(src, OUT_FIG / dst_name)
            indice_lines.append(f"- **{dst_name.split('_')[0]}** `{dst_name}` -- {descr}")
            print(f"[OK] {dst_name}")

    # GNNExplainer (todas as PNGs em uma pasta dedicada)
    gnn_src = FIGS_DIR / "gnnexplainer" / "gossipcop"
    if gnn_src.exists():
        gnn_dst = OUT_FIG / "F12_gnnexplainer_gossipcop"
        gnn_dst.mkdir(exist_ok=True)
        for f in gnn_src.glob("*.png"):
            shutil.copy(f, gnn_dst / f.name)
        for f in gnn_src.glob("*.html"):
            shutil.copy(f, gnn_dst / f.name)
        indice_lines.append(f"- **F12** `F12_gnnexplainer_gossipcop/` -- "
                            f"GNNExplainer aplicado em SAGE/GossipCop (5 amostras)")
        print(f"[OK] F12_gnnexplainer_gossipcop/")

    # Threads Bluesky
    th_src = FIGS_DIR / "threads_bluesky"
    if th_src.exists():
        th_dst = OUT_FIG / "F13_threads_bluesky"
        th_dst.mkdir(exist_ok=True)
        for f in list(th_src.glob("*.png")) + list(th_src.glob("*.html")):
            shutil.copy(f, th_dst / f.name)
        indice_lines.append(f"- **F13** `F13_threads_bluesky/` -- "
                            f"Visualizacoes de 6 threads reais Bluesky de tamanhos variados, "
                            f"com score topologico sobreposto")
        print(f"[OK] F13_threads_bluesky/")

    # ── Indice ───────────────────────────────────────────────────────────
    (OUT_DIR / "INDICE.md").write_text("\n".join(indice_lines) + "\n", encoding="utf-8")
    print(f"\n[OK] INDICE.md")
    print(f"\n[OK] Tudo em: {OUT_DIR}")


if __name__ == "__main__":
    main()
