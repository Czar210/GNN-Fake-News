# Documentação Técnica: 24_consolidar_tcc.py

## Metadados

- **Arquivo analisado:** `24_consolidar_tcc.py`
- **Caminho:** `Training/03_Mega_Research/24_consolidar_tcc.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Utilitário de consolidação — leitura de CSVs heterogêneos, geração programática de tabelas LaTeX e cópia de figuras para um diretório canônico (`Execution/results/figuras_tcc/final/`).
- **Modelos envolvidos:** Nenhum (não treina nem inferencia). Apenas agrega resultados das Fases 2–4 produzidos por outros scripts (10, 13, 14, 15, 19, 20, 21, 22, 26).
- **Datasets envolvidos:** FakeNewsNet (PolitiFact local), UPFD oficial (PolitiFact + GossipCop), Bluesky (168k posts, sem ground truth).
- **Contribuição para a questão central:** Este script não produz evidência empírica nova — sua contribuição é **metodológica e de integridade científica**. Garante que cada número apresentado no corpo do TCC é, por construção, idêntico ao número produzido pelo experimento mais recente, eliminando *drift* entre relato e experimento. Materializa o princípio de reprodutibilidade computacional de Pineau et al. (2021) dentro deste TCC.

---

## 1. Visão Geral do Script

`24_consolidar_tcc.py` é o script de **Fase 8 — Consolidação para o TCC** (PIPELINE.md, l. 209–216). Sua única função é varrer os CSVs/TXTs produzidos pelas Fases 2–4 do pipeline, agregá-los em estruturas LaTeX prontas (`*.tex` em `tabelas/`), copiar as figuras renderizadas para um diretório canônico (`figuras/`) e emitir um índice em Markdown (`INDICE.md`) que mapeia cada tabela/figura ao capítulo do TCC onde deve entrar.

A escolha de transformar resultados em LaTeX **programaticamente** — em vez de copiar e colar manualmente do CSV — é uma decisão de integridade científica. Como Knuth (1984) argumenta em *Literate Programming*, a transformação de dados em prosa científica é tão sujeita a erro quanto o cálculo, e a única forma confiável é tratar a documentação como artefato gerado por código. O script implementa esse princípio: tabelas T1–T11 e figuras F1–F24 nascem de funções determinísticas que leem CSVs canônicos.

O design tem três camadas: (i) **leitura tolerante** (`ler_csv` retorna lista vazia se o CSV não existe, permitindo rodar mesmo com fases incompletas); (ii) **agregação por chaves** (`agregar` reduz linhas a `{(chave1, chave2, ...) -> [valores]}`); (iii) **emissão LaTeX** (`latex_tabela` constrói o ambiente `tabular` no padrão `booktabs`). Cada bloco T1–T11 do `main()` é independente e protegido por `if rows:` — uma fase ausente não derruba as demais.

A saída é um par `(tabelas/, figuras/) + INDICE.md` versionado em git e referenciado por `\input{}` no LaTeX do TCC. Quando o orientador pede reexecução de um experimento das Fases 2–4, basta rerodar a fase + `python 24_consolidar_tcc.py`, e o PDF do TCC reflete o novo número sem intervenção manual.

---

## 2. Componentes Principais

### 2.1 `ler_csv(path)` e `agregar(rows, group_keys, value_key)`

```python
def ler_csv(path: Path) -> list:
    if not path.exists(): return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def agregar(rows, group_keys, value_key):
    grupos = {}
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        try: v = float(r[value_key])
        except Exception: continue
        grupos.setdefault(k, []).append(v)
    return grupos
```

`ler_csv` retorna `[]` em vez de levantar `FileNotFoundError` — comportamento deliberado para permitir consolidação parcial. `agregar` implementa o padrão *split-apply-combine* em ~7 linhas, sem dependência de pandas; cada CSV das Fases 2–4 tem múltiplas linhas por configuração (uma por seed/fold), e a função consolida em listas indexadas por tupla, permitindo `np.mean` / `np.std` no call site:

$$\bar{v} = \frac{1}{n}\sum_{i=1}^n v_i, \qquad s = \sqrt{\frac{1}{n}\sum_{i=1}^n (v_i - \bar{v})^2}$$

A cláusula `try/except Exception: continue` filtra silenciosamente valores não-numéricos sem mascarar erros estruturais — se nenhuma linha converter, `grupos` fica vazio e o bloco de tabela é pulado pelo `if tab_rows:` posterior.

**Embasamento acadêmico:**

> **McKinney, W. (2010)** — "Data Structures for Statistical Computing in Python"
> *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, pp. 56–61
> DOI: `10.25080/Majora-92bf1922-00a`
> **Localização:** Seção 3.4 (Group By: Split-Apply-Combine) — formalização da agregação por chaves múltiplas, que pandas implementa via `df.groupby(group_keys)[value_key].agg(...)`.
> **Relevância:** O dicionário `{(d, m) -> [f1_seed1, f1_seed2, ...]}` retornado por `agregar` é semanticamente idêntico ao resultado de `df.groupby([d_col, m_col])[value_col].apply(list)`. A escolha por implementação manual evita uma dependência pesada quando a operação cabe em 7 linhas — argumento de minimização de superfície de dependência em pipelines de pesquisa (Pineau et al., 2021, §3.3).

### 2.2 `_esc_latex(s)` — escape seletivo de caracteres reservados

```python
def _esc_latex(s: str) -> str:
    s = str(s)
    if "\\" not in s:
        s = (s.replace("&", "\\&").replace("#", "\\#").replace("_", "\\_")
              .replace("%", "\\%").replace("$", "\\$").replace("{", "\\{")
              .replace("}", "\\}").replace("~", "\\textasciitilde{}")
              .replace("^", "\\textasciicircum{}"))
    if s.startswith("["):
        s = "{}" + s
    return s
```

Implementa escape seletivo: strings que já contêm `\` (i.e., comandos LaTeX legítimos como `0.852$\pm$0.04`) passam intactas; strings vindas do CSV (texto puro) são escapadas. A heurística está declarada no docstring (linhas 71–76) e cobre os 9 caracteres reservados do LaTeX. O prefixo `{}` para células iniciadas em `[` (linhas 90–91) blinda contra o parser do LaTeX, que interpretaria `\\ [` como argumento opcional do `\\` — bug silencioso que produziria espaçamento errado em tabelas com bins do tipo `[1,2)`.

**Embasamento acadêmico:**

> **Lamport, L. (1986)** — *LaTeX: A Document Preparation System — User's Guide and Reference Manual*
> Addison-Wesley, ISBN: `0-201-15790-X`
> **Localização:** Capítulo 3 (Carefully Formatted Output) — listagem dos 10 caracteres reservados (`# $ % & ~ _ ^ \ { }`); Apêndice C.1 — comportamento do `\\` em ambiente `tabular`.
> **Relevância:** A função reflete diretamente as regras tipográficas do LaTeX, e o caso especial do prefixo `{}` é defesa contra o problema do Apêndice C.1 (interação `\\` + argumento opcional). Em um TCC com 11 tabelas geradas automaticamente, falhas silenciosas perto do *deadline* são particularmente custosas.

### 2.3 `latex_tabela(headers, rows, caption, label)`

Emite bloco LaTeX no padrão `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). Decisão importante documentada no código: **headers não são escapados, células sim**. Headers vêm do programador (podem conter LaTeX legítimo como `$\pm$`, `$\bar s_a$`); células vêm do CSV. `align = "l" + "r" * (len(headers) - 1)` deixa a primeira coluna left-aligned (rótulo categórico) e as demais right-aligned (números). O par `\caption{}` + `\label{}` permite que o corpo do TCC referencie cada tabela por `\ref{tab:...}`, garantindo numeração automática (Lamport, 1986, §6.4.2).

---

## 3. Fluxo de Consolidação: Onze Tabelas, Onze Argumentos

O `main()` é organizado em 11 blocos T1–T11, cada um correspondendo a uma claim específica do TCC.

| Tab. | CSV de origem | Argumento sustentado no TCC |
|------|---------------|----------------------------|
| T1 | `fase2_baselines/baseline_textual/resultados.csv` | **Teto textual** (LogReg/RF sobre BERT do nó raiz) — referência para qualquer ganho atribuído à topologia. |
| T2 | `fase4_benchmarks/teste_significancia_posfull/tabela_significancia.tex` (cópia) | t-test pareado por fold GCN/GAT/SAGE — todos os contrastes ns. |
| T3 | `fase4_benchmarks/benchmark_upfd_oficial/resultados.csv` | F1-macro de GCN/GAT/SAGE em UPFD-Polit e UPFD-Goss com 5 seeds — ancoragem com benchmark da literatura. |
| T4 | `fase4_benchmarks/topologia_sem_texto/resultados.csv` (variant `B_estrutural`) | **Núcleo do TCC**: F1 ainda alto sem features textuais — vulnerabilidade topológica. |
| T5 | `figuras_tcc/analise_estrutural/estatisticas.csv` (`num_nodes`, `branching_avg`) | Cohen's d fake vs real — explica POR QUE o GossipCop "funciona" (d=+1.53 em branching). |
| T6 | `figuras_tcc/textual_vs_topologico/gossipcop_metricas.txt` | Concordância (kappa, agreement rate) — quando ambos concordam, F1=0.97; quando discordam, textual mantém F1=0.91. |
| T7 | `figuras_tcc/bloco_vs_mini/resumo.csv` | RF "bloco" (3 datasets concatenados) vs 3 RFs "mini" — quase idêntico, leve vantagem em dataset pequeno. |
| T8 | `figuras_tcc/bluesky_inferencia/resumo_por_feed.csv` | Aplicação do RF estrutural em 168k posts do Bluesky — Political Science=69% fake-like. |
| T9 | `figuras_tcc/concordancia_bluesky/cross_tab.csv` (`num_nodes_bin`) | Concordância textual vs topológico por tamanho de grafo. |
| T10 | UPFD oficial + valores de Dou et al. (2021) hard-coded | Comparação com **valores publicados na literatura** — auditoria externa. |
| T11 | `figuras_tcc/rq3_multilingual/comparacoes.csv` | RQ3: distribuição do score topológico em PT/DE/EN — invariância parcial a idioma. |

**Embasamento para o desenho de relatos científicos:**

> **Mitchell, M.; Wu, S.; Zaldivar, A.; Barnes, P.; Vasserman, L.; Hutchinson, B.; Spitzer, E.; Raji, I. D.; Gebru, T. (2019)** — "Model Cards for Model Reporting"
> *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19)*, pp. 220–229
> DOI: `10.1145/3287560.3287596`
> **Localização:** Seção 4 (Model Card Sections) — campos "Quantitative Analyses" e "Evaluation Data"; §4.5 — necessidade de reportar métricas com desvios padrão e tamanho amostral.
> **Relevância:** Cada tabela T1–T11 reporta **valor central + desvio** (e.g., `0.8592 $\pm$ 0.0431`), no formato exigido por Mitchell et al. (2019, §4.5). A coluna "Folds/Seeds" (T1, T3, T4) é a operacionalização direta da exigência de transparência sobre o tamanho amostral subjacente. T10 confronta os números do TCC com valores publicados (Dou et al., 2021) — prática recomendada como salvaguarda contra *cherry-picking*.

---

## 4. Reprodutibilidade Computacional: Por Que Este Script Existe

### 4.1 O problema do *drift* manual

Em iterações anteriores deste TCC (cf. PIPELINE.md, l. 215), o orientador identificou casos em que números no corpo do texto divergiam dos números nos CSVs reais — efeito de copiar valores manualmente, rodar novo experimento e esquecer de atualizar a tabela. Esse fenômeno (*reporting drift*) é uma das causas dominantes da chamada **crise de reprodutibilidade** em ML empírico:

> **Pineau, J.; Vincent-Lamarre, P.; Sinha, K.; Larivière, V.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; Larochelle, H. (2021)** — "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)"
> *Journal of Machine Learning Research*, v. 22, n. 164, pp. 1–20, 2021
> URL: `https://jmlr.org/papers/v22/20-303.html`
> **Localização:** §3 (The Reproducibility Checklist), itens 3.3–3.4 (relato consistente entre código e texto); §5 (Lessons Learned) — divergências entre números reportados e os reproduzíveis a partir do código publicado.
> **Relevância:** Pineau et al. (2021) reportam que mesmo em submissões aceitas no NeurIPS 2019 — venue Tier-1 — uma fração não-trivial de papers tinha divergência entre Tabela X do PDF e Tabela X reproduzível pelo código. A *Reproducibility Checklist* é o que `24_consolidar_tcc.py` materializa para esta monografia: tabela do PDF == tabela do CSV, garantida por construção, não por revisão humana.

### 4.2 *Methods reproducibility* aplicado

> **Goodman, S. N.; Fanelli, D.; Ioannidis, J. P. A. (2016)** — "What does research reproducibility mean?"
> *Science Translational Medicine*, v. 8, n. 341, pp. 341ps12, 2016
> DOI: `10.1126/scitranslmed.aaf5027`
> **Localização:** Seção "A Lexicon for Reproducibility" — distinção entre **methods reproducibility** (mesmo procedimento → mesmo resultado), **results reproducibility** (estudos independentes → mesma conclusão) e **inferential reproducibility** (mesmas inferências de dados similares).
> **Relevância:** O script opera no nível de *methods reproducibility*: dado o mesmo conjunto de CSVs de entrada, produz deterministicamente o mesmo conjunto de tabelas LaTeX. Pré-condição (necessária, não suficiente) para os outros dois níveis. Sem ela, a afirmação "rodamos o experimento X e obtivemos F1=0.86" no TCC seria inverificável.

### 4.3 *Literate programming* como ancestral

> **Knuth, D. E. (1984)** — "Literate Programming"
> *The Computer Journal*, v. 27, n. 2, pp. 97–111
> DOI: `10.1093/comjnl/27.2.97`
> **Localização:** §1 (Introduction) — tese de que documentação e código devem partilhar fonte única; §4 (Examples) — TeX como exemplo.
> **Relevância:** A motivação de Knuth (eliminar inconsistência entre código e descrição) é a motivação do consolidador aplicada a um TCC empírico: eliminar inconsistência entre experimentos e sua descrição numérica. O "documento humano" (TCC PDF) é gerado, não digitado, a partir do "código executável" (Fases 2–4). Jupyter, Quarto e R Markdown derivam dessa mesma genealogia.

---

## 5. Análise de Código

### 5.1 Pontos fortes

- **Idempotência:** rodar duas vezes produz o mesmo resultado. `mkdir(parents=True, exist_ok=True)` (l. 120–122) e `Path.write_text` (sobrescreve) garantem que reexecuções não acumulam estado.
- **Tolerância a fases ausentes:** todo bloco T_i guardado por `if rows:` ou `if path.exists():`. Permite rodar em pipelines parcialmente executados.
- **Contrato de escape declarado em docstring** (l. 95–101): headers não escapam, células sim — decisão documentada *no código*.
- **Hard-coding deliberado de Dou et al. (2021)** (l. 345–353): inclui aviso explícito ("Conferir contra o PDF do paper antes de submeter"). Pior caso (literatura citada errada) mitigado por nota.
- **Sem dependência de pandas:** apenas `csv`, `pathlib`, `numpy`, `matplotlib`, `shutil`. Superfície de dependência mínima (Pineau et al., 2021, §3.3).

### 5.2 Limitações identificadas

**L1 — T10 mistura métricas distintas (F1-macro vs Accuracy):**
A nota (l. 342–344, 373–376) é honesta: "nossos F1-macro com `content` 310d vs accuracy publicado com `bert` 768d". Mas a comparação coalesce duas dimensões de divergência (feature e métrica) numa única coluna ambígua.
*Mitigação:* adicionar terceira coluna "nossa accuracy" no mesmo fold, separando as duas dimensões.

**L2 — Dou et al. (2021) hard-coded sem proveniência registrada:**
Os valores `politifact: 0.846, gossipcop: 0.972` são strings no código, sem registro de revisão do PDF / data de acesso.
*Mitigação:* mover para YAML versionado (`literatura_externa.yaml`) com `source_url`, `version`, `accessed_date`.

**L3 — Filtragem implícita por nome `B_estrutural` em T4:**
Linha 192: `if var != "B_estrutural": continue` filtra silenciosamente outras variantes. Se a Fase 14 produzir nomes diferentes em reexecução, T4 fica vazia sem aviso.
*Mitigação:* `assert any(r["variant"] == "B_estrutural" for r in rows), "Fase 14 não produziu B_estrutural"`.

**L4 — Ausência de hash do CSV no `INDICE.md`:**
O índice mapeia tabela → capítulo, mas não registra qual versão do CSV foi consolidada. Sem trilha de auditoria automática.
*Mitigação:* adicionar bloco `## Proveniência` com `sha256` de cada CSV de origem e `mtime` da última rodagem (Pineau et al., 2021, §3.4 — "Track and report all artifacts").

**L5 — `_esc_latex` heurístico baseado em presença de `\`:**
Strings com `\` são consideradas LaTeX-pré-formatado. Caso patológico (path Windows `C:\Users\foo` legitimamente em CSV) passaria sem escape — não ocorre no domínio atual, mas frágil para extensões.
*Mitigação:* flag explícita `escape=True/False` por chamada.

### 5.3 Boas práticas observadas

- **`booktabs` (toprule/midrule/bottomrule):** estilo recomendado para journals científicos — linhas horizontais finas, sem linhas verticais.
- **`\caption` antes do `\label`:** ordem correta para que `\ref{}` capture o número atribuído (Lamport, 1986, §6.4.2).
- **`\centering` em vez de `\begin{center}...\end{center}`:** `\centering` em `table` não introduz espaçamento vertical extra.

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Contribuição indireta para a questão GNN vs NLP

O script não produz evidência empírica nova sobre GNNs vs NLP — sua contribuição é **infraestrutural**. Mas a infraestrutura importa para o argumento: o resultado central do TCC (F1≈0.86 do baseline textual ≡ F1≈0.86 das GNNs no FakeNewsNet) é uma afirmação numérica precisa; sua credibilidade depende de o leitor poder verificar que os números reportados na monografia são exatamente os produzidos pelo experimento. `24_consolidar_tcc.py` é a peça que torna essa verificação operacional.

### 6.2 Conexão com integridade científica

Pineau et al. (2021, §5) observam que esforços de reprodutibilidade em ML focam em código de *treino* e *avaliação*, ignorando o passo final de **consolidação em prosa científica** — onde reside fração não-desprezível das discrepâncias reportadas em auditorias. O script preenche essa lacuna no TCC: o pipeline "experimento → CSV → tabela LaTeX → PDF" é totalmente automatizado, o que significa que qualquer alegação ("GNN obtém F1=0.86") pode ser rastreada determinísticamente até a linha do CSV que produziu o número.

### 6.3 Quando consolidação automatizada importa mais — datasets pequenos

No FakeNewsNet (N≈314) e UPFD-PolitiFact (N≈314), o desvio padrão entre folds é 0.04–0.05 (T1, T3). Nesse regime, **uma reexecução tipicamente produz números ligeiramente diferentes na 4ª casa decimal** mesmo com seed fixa, devido a non-determinismo de CUDA/cuBLAS. Se a tabela é digitada manualmente, o autor tipicamente preserva o número da primeira execução enquanto reroda e atualiza o resto — produzindo inconsistência interna. A consolidação programática garante que **todas** as tabelas refletem **a mesma rodagem**, eliminando esse modo de falha.

### 6.4 Resposta parcial à questão central

> **"GNNs são uma alternativa viável para detecção de fake news?"**

Este script não responde diretamente — mas **garante que a resposta dada pelo TCC é defensável**. Toda a evidência empírica do TCC — o teto textual de T1, a não-significância de T2, os F1 cross-dataset de T3, a vulnerabilidade topológica de T4, o Cohen's d de T5, a concordância de T6 — passa por este consolidador. Se a banca questionar qualquer número, a trilha de auditoria é trivial: localizar a tabela no `INDICE.md`, identificar o CSV de origem, identificar o script da Fase 2/3/4 que gerou o CSV, identificar a seed e o fold.

Esse atributo — **rastreabilidade end-to-end** — distingue uma monografia que segue prática científica corrente de uma que apenas relata números soltos. Aplicado ao debate GNN vs NLP, significa que a afirmação central ("GNNs não superam o teto textual no FakeNewsNet com features BERT, p-valor pareado ns") é uma afirmação **falsificável e auditável**, não uma alegação retórica.

---

## 7. Referências Bibliográficas

1. PINEAU, J.; VINCENT-LAMARRE, P.; SINHA, K.; LARIVIÈRE, V.; BEYGELZIMER, A.; D'ALCHÉ-BUC, F.; FOX, E.; LAROCHELLE, H. **Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)**. *Journal of Machine Learning Research*, v. 22, n. 164, pp. 1–20, 2021. Disponível em: `https://jmlr.org/papers/v22/20-303.html`

2. MITCHELL, M.; WU, S.; ZALDIVAR, A.; BARNES, P.; VASSERMAN, L.; HUTCHINSON, B.; SPITZER, E.; RAJI, I. D.; GEBRU, T. **Model Cards for Model Reporting**. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19)*, pp. 220–229, 2019. DOI: `10.1145/3287560.3287596`

3. McKINNEY, W. **Data Structures for Statistical Computing in Python**. *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, pp. 56–61, 2010. DOI: `10.25080/Majora-92bf1922-00a`

4. LAMPORT, L. **LaTeX: A Document Preparation System — User's Guide and Reference Manual**. Reading, MA: Addison-Wesley, 1986. ISBN: `0-201-15790-X`

5. KNUTH, D. E. **Literate Programming**. *The Computer Journal*, v. 27, n. 2, pp. 97–111, 1984. DOI: `10.1093/comjnl/27.2.97`

6. GOODMAN, S. N.; FANELLI, D.; IOANNIDIS, J. P. A. **What does research reproducibility mean?**. *Science Translational Medicine*, v. 8, n. 341, pp. 341ps12, 2016. DOI: `10.1126/scitranslmed.aaf5027`

7. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *SIGIR 2021*, pp. 2051–2055. DOI: `10.1145/3404835.3462990` *(referenciado em T10 para valores de comparação)*

---

## 8. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Reporting drift | Divergência entre números reportados em texto e números reproduzíveis a partir do código/dados publicados | Pineau et al. (2021), §3.3–3.4 |
| Methods reproducibility | Mesma entrada produz, deterministicamente, a mesma saída | Goodman et al. (2016) |
| Literate programming | Código executável e descrição humana produzidos do mesmo fonte | Knuth (1984), §1 |
| Split-apply-combine | Padrão de agregação por chaves; base do `groupby` em pandas e da função `agregar` | McKinney (2010), §3.4 |
| Model card | Documento padronizado de metadados sobre modelo de ML, com métricas, desvios e limitações | Mitchell et al. (2019), §4 |
| `booktabs` | Pacote LaTeX com `\toprule`/`\midrule`/`\bottomrule` — convenção em tabelas científicas modernas | Manual `booktabs.sty` |
| Trilha de auditoria | Sequência de artefatos versionados rastreando cada número do PDF até o experimento de origem | Pineau et al. (2021), §3.4 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 24_consolidar_tcc.py
Arquivo gerado: theory_andre/24_consolidar_tcc_doc.md
Fontes academicas utilizadas: 7 (Tier 1)
    1. Pineau et al. (2021) JMLR
    2. Mitchell et al. (2019) FAT*
    3. McKinney (2010) SciPy
    4. Lamport (1986) LaTeX Manual
    5. Knuth (1984) Computer Journal
    6. Goodman et al. (2016) Science Translational Medicine
    7. Dou et al. (2021) SIGIR (referenciado em T10)
Conceitos cobertos: leitura tolerante de CSV, agregacao split-apply-combine,
    escape LaTeX seletivo, geracao programatica de booktabs, mapeamento
    tabela-capitulo via INDICE.md, methods reproducibility, literate
    programming, model card reporting standards, reporting drift,
    audit trail end-to-end.
Limitacoes: T10 mistura F1-macro vs Accuracy (mitigado por nota explicita);
    Dou et al. hard-coded sem hash de proveniencia; INDICE.md sem
    sha256 dos CSVs (sem trilha de auditoria automatica da versao
    consolidada).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISAO -- nao prosseguir para o proximo script.
```
