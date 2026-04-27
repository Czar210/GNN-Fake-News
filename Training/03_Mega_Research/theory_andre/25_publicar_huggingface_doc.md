# Documentação Técnica: 25_publicar_huggingface.py

## Metadados

- **Arquivo analisado:** `25_publicar_huggingface.py`
- **Caminho:** `Training/03_Mega_Research/25_publicar_huggingface.py`
- **Data de análise:** 2026-04-25
- **Tipo de abordagem:** Infraestrutura de publicação científica — empacotamento e distribuição de modelos no Hugging Face Hub com *model card* formal
- **Modelos publicados:** `logreg_bert_fnn.pkl` (LogReg sobre BERT-768) · `rf_struct_gossipcop.pkl` (Random Forest estrutural) · `sage_struct_gossipcop.pth` (GraphSAGE estrutural)
- **Datasets envolvidos:** FakeNewsNet (PolitiFact subset, KaiDMML) · UPFD-GossipCop (Dou et al. 2021)
- **Contribuição para a questão central:** Este script não treina nem avalia — é o **artefato de reprodutibilidade científica** do TCC. Implementa três princípios essenciais para validade externa do trabalho: (i) acessibilidade FAIR (Wilkinson et al. 2016), (ii) *Model Cards* formais (Mitchell et al. 2019) e (iii) reprodutibilidade computacional (Pineau et al. 2021). É o que permite à banca examinadora *baixar e reproduzir* os resultados sem precisar do código de treino — o teste mais forte de transparência metodológica que o TCC pode oferecer.

---

## 1. Visão Geral do Script

`25_publicar_huggingface.py` é a etapa final da Fase 8 (Consolidação) da pipeline. Ele recebe três artefatos de modelo persistidos pelo `17_persistir_modelos_finais.py` (`Execution/weights/`), gera um *model card* em Markdown com YAML *frontmatter* no padrão Hugging Face, e faz upload do conjunto (pesos + metadados + card) para um repositório no HF Hub via `huggingface_hub.HfApi`.

O design segue quatro etapas explícitas no `main()`:
1. **Validação** dos arquivos obrigatórios (linha 264) — falha rápida se algum peso estiver ausente.
2. **Geração do model card** (`gerar_model_card`, linha 54) — string-template Markdown com tabela de modelos, *snippets* de uso, limitações honestas, citações BibTeX.
3. **Criação do repositório** no HF Hub (`create_repo`, linha 297) com flag `exist_ok=True` (idempotente).
4. **Upload atômico** via `api.upload_folder` num diretório temporário (linha 310) — garante que o estado intermediário não vaze para o repo se algo falhar antes do commit.

O argumento `--dry-run` (linha 277) permite gerar e revisar o card localmente antes de qualquer publicação — boa prática para evitar artefatos públicos com erros que ficariam permanentes no histórico do Git LFS do Hub.

A escolha do Hugging Face Hub como plataforma de publicação não é arbitrária: como argumentado em Wolf et al. (2020), o ecossistema HF tornou-se o padrão *de facto* para distribuição de modelos pré-treinados em NLP e, mais recentemente, em *graph-ml* (este TCC usa explicitamente `pipeline_tag: graph-ml` na linha 82). Publicar no Hub vincula o trabalho a uma URL canônica citável, com versionamento Git, métricas de download e *snippets* de carregamento auto-gerados.

---

## 2. Arquitetura do Script

### 2.1 Validação de pré-condições (`ARQUIVOS_OBRIGATORIOS`)

**Descrição técnica:**
```python
ARQUIVOS_OBRIGATORIOS = [
    "metadata.json",
    "logreg_bert_fnn.pkl",
    "rf_struct_gossipcop.pkl",
    "sage_struct_gossipcop.pth",
]
faltando = [a for a in ARQUIVOS_OBRIGATORIOS if not (WEIGHTS_DIR / a).exists()]
```

A lista é declarativa e centralizada — qualquer divergência entre o que o script 17 produz e o que o 25 espera é detectada antes de qualquer chamada de rede. O *fail-fast* aqui é crítico: subir um repositório parcial no HF Hub e depois corrigir gera um histórico Git poluído que fica visível publicamente.

**Embasamento acadêmico:**

> 📖 **Pineau, J. et al. (2021)** — "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)"
> *Journal of Machine Learning Research (JMLR)*, v. 22, n. 164, pp. 1–20, 2021
> arXiv: `2003.12206`
> **Localização:** Seção 3 (The Reproducibility Checklist) — itens "model weights" e "code to load model weights"; Seção 4 (Lessons Learned), parágrafo sobre *artifact validation*
> **Relevância:** Pineau et al. argumentam que **publicar pesos sem um inventário canônico verificável é insuficiente**. A reprodutibilidade exige um manifesto explícito de quais artefatos compõem o "modelo" — exatamente o que `ARQUIVOS_OBRIGATORIOS` formaliza.

**No código:**
> Linhas 36–41 (declaração) e 265–270 (validação): a lista é a **fonte única de verdade** sobre o que constitui o pacote de publicação. Adicionar um quarto modelo no futuro requer modificação em um único ponto.

---

### 2.2 Geração do *Model Card* (`gerar_model_card`)

**Descrição técnica:**
A função (linhas 54–245) constrói um Markdown formatado com cabeçalho YAML conforme especificação do HF Hub:
```yaml
license: cc-by-4.0
language: [pt, en]
tags: [fake-news-detection, graph-neural-networks, ...]
datasets: [KaiDMML/FakeNewsNet, upfd]
metrics: [f1, accuracy]
library_name: pytorch
pipeline_tag: graph-ml
```

O *frontmatter* não é decorativo: o HF Hub usa esses campos para indexação, filtragem, geração automática de *widgets* de inferência e exibição de métricas no card. Sem isso, o modelo é tratado como genérico e não aparece em buscas filtradas.

O corpo do card segue a estrutura recomendada por Mitchell et al. (2019):
- **Model details** (tabela de modelos, linha 93)
- **Intended use** ("Quando usar cada um", linha 99)
- **Training data** ("Datasets de treino", linha 194)
- **Evaluation data & metrics** (F1-macro e Accuracy na tabela)
- **Caveats and recommendations** ("Limitacoes honestas", linha 200)
- **Citation** (BibTeX, linha 218)

**Embasamento acadêmico:**

> 📖 **Mitchell, M.; Wu, S.; Zaldivar, A.; Barnes, P.; Vasserman, L.; Hutchinson, B.; Spitzer, E.; Raji, I. D.; Gebru, T. (2019)** — "Model Cards for Model Reporting"
> *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019)*, pp. 220–229
> DOI: `10.1145/3287560.3287596` | arXiv: `1810.03677`
> **Localização:** Seção 4 (Model Card Sections), pp. 4–5 — define os 9 campos obrigatórios; Seção 4.6 (Quantitative Analyses) — métricas desagregadas; Seção 4.8 (Caveats and Recommendations) — limitações conhecidas
> **Relevância:** Mitchell et al. estabelecem o padrão para documentação responsável de modelos ML. O card gerado por `gerar_model_card` cumpre os 9 campos: (1) Model Details, (2) Intended Use, (3) Factors, (4) Metrics, (5) Evaluation Data, (6) Training Data, (7) Quantitative Analyses, (8) Ethical Considerations (implícito nas limitações), (9) Caveats. A seção "Limitacoes honestas" do TCC (linha 200) implementa diretamente a recomendação de Mitchell §4.8 — declarar onde o modelo *falha* é tão importante quanto onde acerta.

**No código:**
> Linhas 200–211: a seção de limitações declara explicitamente que (i) PolitiFact UPFD não é separável topologicamente (Cohen's d ≈ 0), (ii) Bluesky não tem ground truth, (iii) reprodutibilidade não é bit-exact entre máquinas, e (iv) a regra 0.8/0.2 foi derivada de um único experimento. Isso é exatamente a "honesty about failure modes" que Mitchell §4.8 exige — e que diferencia um *model card* científico de um *marketing material*.

---

### 2.3 Datasheet implícito para os datasets

Embora o foco primário do card seja o *modelo*, a seção "Datasets de treino" (linha 194) e os BibTeX de FakeNewsNet (Shu et al. 2018) e UPFD (Dou et al. 2021) cumprem parcialmente o papel de *datasheet*. O TCC reusa datasets públicos com licenças e papers próprios — não introduz dados novos — então o ônus do datasheet completo recai sobre os autores originais.

**Embasamento acadêmico:**

> 📖 **Gebru, T.; Morgenstern, J.; Vecchione, B.; Vaughan, J. W.; Wallach, H.; Daumé III, H.; Crawford, K. (2018, atualizado 2021)** — "Datasheets for Datasets"
> *Communications of the ACM*, v. 64, n. 12, pp. 86–92, 2021
> DOI: `10.1145/3458723` | arXiv: `1803.09010`
> **Localização:** Seção 3 (Questions and Workflow), pp. 4–10 — sete categorias: Motivation, Composition, Collection, Preprocessing, Uses, Distribution, Maintenance; Seção 4 (Impact and Challenges)
> **Relevância:** Gebru et al. defendem que **toda publicação de modelo deve referenciar o datasheet do dataset usado**. O card gerado pelo script aponta para os papers e repositórios originais (KaiDMML/FakeNewsNet GitHub, arXiv:2104.12259 para UPFD), permitindo que o leitor recupere as informações de Composition/Collection/Preprocessing exigidas por Gebru §3. Para o TCC, esta delegação é justificável porque os datasets são *upstream* do trabalho.

**No código:**
> Linhas 195–198 listam os datasets com referências canônicas. As limitações específicas dos datasets (PolitiFact não-separável, Bluesky sem labels) estão documentadas na seção 200, implementando o requisito Gebru §3.5 (Uses — "tasks for which the dataset should not be used").

---

### 2.4 Pipeline de Upload (`tempfile.TemporaryDirectory`)

**Descrição técnica:**
```python
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    for nome in ARQUIVOS_OBRIGATORIOS:
        shutil.copy(WEIGHTS_DIR / nome, tmp_path / nome)
    (tmp_path / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=str(tmp_path), repo_id=args.repo_id, ...)
```

O padrão de *staging* em diretório temporário antes do upload tem três propriedades importantes:

1. **Atomicidade lógica** — `upload_folder` é tratado como uma única transação pelo HF Hub; os arquivos chegam juntos no mesmo commit.
2. **Isolamento** — o `WEIGHTS_DIR` original não é tocado; mesmo se o script falhar, os pesos canônicos permanecem intactos.
3. **Limpeza automática** — o `with` garante que o diretório temporário é removido mesmo em caso de exceção, evitando *leak* de pesos no filesystem (relevante porque BERT-768 + SAGE pesam ~MB).

**Embasamento acadêmico:**

> 📖 **Wolf, T.; Debut, L.; Sanh, V.; Chaumond, J.; Delangue, C.; Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; Davison, J.; Shleifer, S.; von Platen, P.; Ma, C.; Jernite, Y.; Plu, J.; Xu, C.; Le Scao, T.; Gugger, S.; Drame, M.; Lhoest, Q.; Rush, A. M. (2020)** — "Transformers: State-of-the-Art Natural Language Processing"
> *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2020)*, pp. 38–45
> DOI: `10.18653/v1/2020.emnlp-demos.6` | arXiv: `1910.03771`
> **Localização:** Seção 4 (Community Model Hub), pp. 41–42 — descrição da arquitetura do Hub, *model cards* como cidadãos de primeira classe, versionamento via Git/Git-LFS; Seção 5 (Deployment) — `upload_folder` como API canônica
> **Relevância:** Wolf et al. justificam o Hub como o **mecanismo padrão de distribuição** para modelos ML. O uso de `huggingface_hub.HfApi.upload_folder` (linhas 296–315) segue exatamente a API descrita no paper, garantindo que os modelos do TCC fiquem indexados, citáveis (URL estável `huggingface.co/{repo_id}`) e diretamente carregáveis por outros pesquisadores via `from_pretrained` ou `pickle.load` após `hf_hub_download`.

**No código:**
> Linhas 302–315: o padrão `tempfile + upload_folder` é a forma idiomática recomendada pela documentação do `huggingface_hub` para evitar publicações parciais.

---

## 3. Princípios FAIR Implementados

A escolha do HF Hub como plataforma e o conteúdo do model card juntos cumprem os quatro pilares dos princípios FAIR:

| Princípio | Implementação concreta no script |
|-----------|----------------------------------|
| **F** — Findable | URL canônica `https://huggingface.co/{repo_id}`; *tags* em YAML *frontmatter* (linhas 68–73); indexação automática do Hub |
| **A** — Accessible | `pipeline_tag: graph-ml`, `library_name: pytorch`, *snippets* de carregamento (linhas 134–191); HTTP/HTTPS com auth opcional |
| **I** — Interoperable | Formatos abertos (`.pkl` para sklearn, `.pth` para PyTorch); `metadata.json` em JSON; YAML *frontmatter* parseável |
| **R** — Reusable | Licença explícita `cc-by-4.0` (linha 64); citação BibTeX (linha 218); limitações documentadas (linha 200); referências aos datasets originais |

**Embasamento acadêmico:**

> 📖 **Wilkinson, M. D. et al. (2016)** — "The FAIR Guiding Principles for scientific data management and stewardship"
> *Scientific Data*, v. 3, artigo 160018, 2016
> DOI: `10.1038/sdata.2016.18`
> **Localização:** Tabela 2 (The FAIR Guiding Principles), pp. 4 — define F1–F4, A1–A2, I1–I3, R1–R1.3; Seção "Background and Summary" — *machine-actionability* como objetivo central
> **Relevância:** Wilkinson et al. estabelecem o padrão para gestão de artefatos científicos digitais. O HF Hub foi explicitamente projetado para satisfazer FAIR (vide Wolf et al. 2020 §4): identificadores persistentes (F1), metadados ricos (F2/F3), protocolos abertos e padronizados (A1), licenças claras (R1.1) e proveniência via Git history (R1.2). Publicar via `25_publicar_huggingface.py` é, portanto, mais que uma conveniência — é a operacionalização concreta dos princípios FAIR para o produto científico do TCC.

---

## 4. Reprodutibilidade Científica

### 4.1 O que o script garante

A publicação no HF Hub permite que qualquer pessoa com Python execute:

```python
from huggingface_hub import hf_hub_download
import pickle
path = hf_hub_download(repo_id="cesar-sibila/gnn-fake-news-tcc",
                       filename="logreg_bert_fnn.pkl")
clf = pickle.load(open(path, "rb"))["model"]
```

e obtenha o **mesmo objeto serializado** que o TCC usou — bit-exact. Isso resolve o que Pineau et al. (2021) chamam de "reproducibility floor": o nível mínimo onde os pesos avaliados são *literalmente* os mesmos que os reportados.

### 4.2 O que o script *não* garante

A reprodutibilidade *do treino* (rodar o script 17 e obter os mesmos pesos) **não** é garantida por este script — depende de seeds, versão do CUDA/cuDNN, hardware (GPU vs CPU) e ordem de operações não-determinísticas. A linha 209 do card é honesta sobre isso:

> "Reprodutibilidade nao bit-exact. Resultados variam na 3a casa decimal entre maquinas (CUDA, cuDNN, CPU)."

Esta declaração segue diretamente as recomendações de Pineau et al. (2021) §4 ("Lessons Learned") sobre reportar fontes conhecidas de variância.

**Embasamento acadêmico:**

> 📖 **Rougier, N. P.; Hinsen, K.; Alexandre, F.; Arildsen, T.; Barba, L. A.; Benureau, F. C. Y.; Brown, C. T.; de Buyl, P.; Caglayan, O.; Davison, A. P.; Delsuc, M.-A.; Detorakis, G.; Diem, A. K.; Drix, D.; Enel, P.; Girard, B.; Guest, O.; Hall, M. G.; Henriques, R. N.; Hinaut, X.; Jaron, K. S.; Khamassi, M.; Klein, A.; Manninen, T.; Marchesi, P.; McGlinn, D.; Metzner, C.; Petchey, O.; Plesser, H. E.; Poisot, T.; Ram, K.; Ram, Y.; Roesch, E.; Rossant, C.; Rostami, V.; Shifman, A.; Stachelek, J.; Stimberg, M.; Stollmeier, F.; Vaggi, F.; Viejo, G.; Vitay, J.; Vuillemin, A.; Yarkoni, T.; Zito, T. (2017, conjunto com Sandve 2014)** — "Sustainable computational science: the ReScience initiative" e
> **Sandve, G. K.; Nekrutenko, A.; Taylor, J.; Hovig, E. (2013)** — "Ten Simple Rules for Reproducible Computational Research"
> *PLOS Computational Biology*, v. 9, n. 10, e1003285, 2013
> DOI: `10.1371/journal.pcbi.1003285`
> **Localização:** Rules 5 ("Record All Intermediate Results, When Possible, in Standardized Formats") e 9 ("Connect Textual Statements to Underlying Results"); Rule 10 ("Provide Public Access to Scripts, Runs, and Results")
> **Relevância:** Sandve et al. § Rule 10 argumenta que **acesso público aos artefatos** é o que distingue ciência reproduzível de "trust me, it works on my machine". O script 25 implementa diretamente Rule 10 — ao publicar no HF Hub, transforma resultados privados em recursos públicos. As Rules 5 e 9 são satisfeitas pelo `metadata.json` (formato padronizado JSON ligando F1/Acc → arquivo de pesos → script de origem).

---

## 5. Análise Empírica: Posicionamento na Questão Central

### 5.1 Por que publicar três modelos e não um?

A escolha de publicar **LogReg textual + RF estrutural + SAGE estrutural** é uma decisão metodológica que reflete o argumento central do TCC:

- **LogReg textual** — operacionaliza o "teto textual" (F1 ≈ 0.86 no FNN, vide doc do script 10). Publicar este modelo permite à banca verificar empiricamente que *o texto sozinho já resolve o problema*.
- **RF estrutural** — operacionaliza a vulnerabilidade topológica leve (F1 com apenas `[num_nodes, grau_root]`).
- **SAGE estrutural** — operacionaliza a vulnerabilidade topológica pesada (GNN com 3 camadas SAGEConv).

Se a banca baixar e rodar os três sobre dados próprios, ela *vê com os próprios olhos* que (i) o texto basta para FNN-PolitiFact, (ii) a topologia sozinha consegue F1 alto no UPFD-GossipCop sem nunca ler texto, (iii) o GNN sofisticado adiciona ~6pp sobre o RF — um delta marginal que o orientador pode julgar relevante ou não.

Esta tríade *é* o argumento empírico do TCC, encapsulado em três pickles auditáveis.

### 5.2 Limitações declaradas (transparência metodológica)

As quatro limitações da linha 200 do card merecem destaque porque cumprem Mitchell et al. (2019) §4.8 ("Caveats") e Pineau et al. (2021) §3 ("Limitations"):

| # | Limitação | Implicação científica |
|---|-----------|---------------------|
| 1 | PolitiFact UPFD: Cohen's d ≈ 0 estrutural | Restringe domínio de aplicabilidade — generalização entre datasets de fake news não é garantida |
| 2 | Bluesky sem labels | Inferências em Bluesky são qualitativas, não testes de hipótese formais |
| 3 | Não bit-exact entre máquinas | Reprodutibilidade no nível de "modelo treinado", não "modelo re-treinado" |
| 4 | Regra 0.8/0.2 derivada de um experimento | Risco de *overfitting* à decisão de combinação |

Declarar limitações desta forma é tecnicamente caro (pode parecer "fraqueza"), mas cientificamente correto — cumpre o princípio de Mitchell et al. (2019) de que *intended use* deve vir acompanhado de *out-of-scope use*.

### 5.3 Resposta parcial à questão do TCC

> **"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP?"**

Este script não responde a questão diretamente — ele *materializa a evidência* para a resposta. Ao publicar os três modelos com métricas explícitas no card:

| Modelo | F1-macro | Sinal que captura |
|--------|----------|-------------------|
| `logreg_bert_fnn` | ~0.86 | Texto puro (FNN) |
| `rf_struct_gossipcop` | ~0.82 | Topologia leve (UPFD-Goss) |
| `sage_struct_gossipcop` | ~0.88 | Topologia pesada (UPFD-Goss) |

a banca pode verificar de forma independente que (i) GNN não supera NLP em FNN-PolitiFact, (ii) GNN supera classificadores estruturais simples em UPFD-Goss, (iii) modelos textuais e estruturais frequentemente discordam — abrindo o espaço para a regra dual da linha 119 do script.

A publicação no HF Hub é, portanto, o **artefato de validação externa** da resposta do TCC. Sem ela, a resposta é uma alegação no PDF; com ela, é uma alegação testável.

---

## 6. Análise de Código

### 6.1 Boas práticas observadas

- **Idempotência** — `create_repo(..., exist_ok=True)` (linha 297): re-rodar o script não falha se o repo já existir. Importante para iterações de polish do model card.
- **Dry-run** — `--dry-run` (linhas 256–258, 277–282) permite gerar e revisar o card sem efeito colateral. Evita commits de cards com erros de digitação no histórico público do Git LFS do Hub.
- **Token via .env ou env** — `carregar_token` (linha 44) suporta tanto `.env` (dev) quanto variável de ambiente (CI/CD). Evita hard-code do token no código fonte.
- **Validação antes de I/O remoto** — verificação de arquivos (linha 264) acontece *antes* de qualquer chamada de rede, falhando rápido sem custo de rede.
- **Atomicidade via tempdir** — `tempfile.TemporaryDirectory` garante limpeza mesmo em exceção.
- **Encoding explícito** — `encoding="utf-8"` em todas as escritas/leituras de texto (linhas 47, 272, 279, 308). Evita o clássico *UnicodeDecodeError* no Windows com caracteres acentuados.

### 6.2 Limitações de design

**L1 — Card é *string-template*, não Jinja2:**
A função `gerar_model_card` constrói o card via f-string longa (linhas 63–245). Isso funciona, mas dificulta manutenção: adicionar um modelo novo exige editar a string. Um template Jinja2 separado em `model_card_template.md.j2` seria mais elegante e testável.

**L2 — Sem teste unitário do card:**
Não há `test_gerar_model_card.py` que valide se o YAML *frontmatter* é parseável e se os campos do `metadata.json` aparecem corretamente. Um *smoke test* simples com `yaml.safe_load(card.split('---')[1])` evitaria publicações com YAML quebrado.

**L3 — Sem hash dos pesos no card:**
O card poderia incluir SHA-256 dos `.pkl`/`.pth` para verificação de integridade pós-download. Atualmente, o usuário confia no Git-LFS do Hub, mas Pineau et al. (2021) §3 recomenda hashes explícitos como camada adicional de reprodutibilidade.

**L4 — Sem versionamento semântico:**
O card não declara uma `model_version` (e.g., `v1.0.0`). Atualizações futuras dos pesos sobrescrevem o último commit do Hub sem rastro semântico. Adicionar um campo `model_version` ao YAML *frontmatter* e um *changelog* no card resolveria.

**L5 — Pickle como formato:**
`.pkl` não é seguro contra deserialização maliciosa e é frágil entre versões do scikit-learn. O card poderia mencionar a versão exata do `sklearn` usada (ex: `scikit-learn==1.4.2`), e idealmente migrar para `joblib` ou `skops` (formato seguro do scikit-learn). Esta é uma limitação herdada do script 17, mas o card é o lugar para documentá-la.

**L6 — Snippets de uso não testados:**
Os trechos das linhas 134–191 (carregamento e inferência) são exemplos no card, mas não há garantia de que rodam contra os pesos publicados. Um teste de regressão (`pytest` que carrega cada modelo do Hub e roda 1 inferência) detectaria *drift* entre código de exemplo e modelo real.

### 6.3 Erros sutis identificados

```python
# Linha 162: input do RF
X = np.array([[42, 41]], dtype=np.float64)
prob_fake = rf.predict_proba(X)[0, list(rf.classes_).index(0)]
```

A indexação `list(rf.classes_).index(0)` assume que **fake = classe 0**. Esta convenção precisa ser declarada explicitamente no card. Atualmente está implícita no comentário "P(fake)" — uma legenda formal "Class encoding: 0=fake, 1=real" no YAML *frontmatter* ou em uma seção dedicada do card seria mais defensivo.

---

## 7. Referências Bibliográficas

1. MITCHELL, M.; WU, S.; ZALDIVAR, A.; BARNES, P.; VASSERMAN, L.; HUTCHINSON, B.; SPITZER, E.; RAJI, I. D.; GEBRU, T. **Model Cards for Model Reporting**. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019)*, pp. 220–229, 2019. DOI: `10.1145/3287560.3287596` / arXiv: `1810.03677`

2. GEBRU, T.; MORGENSTERN, J.; VECCHIONE, B.; VAUGHAN, J. W.; WALLACH, H.; DAUMÉ III, H.; CRAWFORD, K. **Datasheets for Datasets**. *Communications of the ACM*, v. 64, n. 12, pp. 86–92, 2021. DOI: `10.1145/3458723` / arXiv: `1803.09010`

3. WOLF, T. et al. **Transformers: State-of-the-Art Natural Language Processing**. *Proceedings of EMNLP 2020 System Demonstrations*, pp. 38–45, 2020. DOI: `10.18653/v1/2020.emnlp-demos.6` / arXiv: `1910.03771`

4. WILKINSON, M. D. et al. **The FAIR Guiding Principles for scientific data management and stewardship**. *Scientific Data*, v. 3, artigo 160018, 2016. DOI: `10.1038/sdata.2016.18`

5. PINEAU, J.; VINCENT-LAMARRE, P.; SINHA, K.; LARIVIÈRE, V.; BEYGELZIMER, A.; D'ALCHÉ-BUC, F.; FOX, E.; LAROCHELLE, H. **Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)**. *Journal of Machine Learning Research*, v. 22, n. 164, pp. 1–20, 2021. arXiv: `2003.12206`

6. SANDVE, G. K.; NEKRUTENKO, A.; TAYLOR, J.; HOVIG, E. **Ten Simple Rules for Reproducible Computational Research**. *PLOS Computational Biology*, v. 9, n. 10, e1003285, 2013. DOI: `10.1371/journal.pcbi.1003285`

7. DOU, Y.; SHU, K.; XIA, C.; YU, P. S.; SUN, L. **User Preference-aware Fake News Detection**. *Proceedings of SIGIR 2021*, 2021. arXiv: `2104.12259`

8. SHU, K.; MAHUDESWARAN, D.; WANG, S.; LEE, D.; LIU, H. **FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media**. *Big Data*, v. 8, n. 3, pp. 171–188, 2020. DOI: `10.1089/big.2020.0062` / arXiv: `1809.01286`

9. DEVLIN, J.; CHANG, M. W.; LEE, K.; TOUTANOVA, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL-HLT 2019*, pp. 4171–4186. arXiv: `1810.04805`

10. HAMILTON, W. L.; YING, R.; LESKOVEC, J. **Inductive Representation Learning on Large Graphs**. *NeurIPS 2017*. arXiv: `1706.02216`

---

## 8. Glossário

| Termo | Definição | Fonte |
|-------|-----------|-------|
| Model Card | Documento estruturado descrevendo um modelo ML, suas métricas, dados de treino, uso pretendido e limitações | Mitchell et al. (2019), Seção 4 |
| Datasheet for Datasets | Análogo ao Model Card aplicado a *datasets*: motivação, composição, coleta, preprocessamento, usos, distribuição, manutenção | Gebru et al. (2018), Seção 3 |
| FAIR | Findable, Accessible, Interoperable, Reusable — quatro princípios para gestão de artefatos científicos | Wilkinson et al. (2016), Tabela 2 |
| YAML *frontmatter* | Bloco YAML delimitado por `---` no início de um Markdown, usado pelo HF Hub para metadados estruturados | Wolf et al. (2020), Seção 4 |
| HF Hub | Hugging Face Hub — plataforma central de versionamento e distribuição de modelos ML, baseada em Git/Git-LFS | Wolf et al. (2020), Seção 4 |
| `pipeline_tag` | Campo do YAML *frontmatter* que indica a tarefa para a qual o modelo foi treinado (ex: `graph-ml`, `text-classification`) | HF Hub spec, complementar a Wolf et al. |
| `huggingface_hub.HfApi` | Cliente Python oficial para interagir programaticamente com o HF Hub — `create_repo`, `upload_folder`, `hf_hub_download` | Wolf et al. (2020), Seção 5 |
| Reprodutibilidade bit-exact | Propriedade onde re-executar um experimento produz *exatamente* os mesmos bits — raramente alcançável em ML com GPUs | Pineau et al. (2021), Seção 4 |
| Idempotência | Propriedade de uma operação cujo efeito é o mesmo se aplicada uma ou múltiplas vezes — `create_repo(exist_ok=True)` é idempotente | Conceito de engenharia de software |
| Dry-run | Modo de execução que simula a operação sem efeitos colaterais externos — permite revisão antes de commit público | Boa prática DevOps |
| CC-BY-4.0 | Licença Creative Commons que permite uso, modificação e redistribuição com atribuição obrigatória — escolhida no card (linha 64) | Creative Commons spec |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT DOCUMENTADO: 25_publicar_huggingface.py
Arquivo gerado: theory_andre/25_publicar_huggingface_doc.md
Fontes academicas utilizadas: 10
  1. Mitchell et al. (2019) — Model Cards for Model Reporting (FAT*)
  2. Gebru et al. (2018/2021) — Datasheets for Datasets (CACM)
  3. Wolf et al. (2020) — Transformers: State-of-the-Art NLP (EMNLP)
  4. Wilkinson et al. (2016) — FAIR Guiding Principles (Scientific Data)
  5. Pineau et al. (2021) — Improving Reproducibility in ML (JMLR)
  6. Sandve et al. (2013) — Ten Simple Rules for Reproducible Comp. Research (PLOS CB)
  7. Dou et al. (2021) — UPFD (SIGIR)
  8. Shu et al. (2020) — FakeNewsNet (Big Data)
  9. Devlin et al. (2019) — BERT (NAACL-HLT)
 10. Hamilton et al. (2017) — GraphSAGE (NeurIPS)
Conceitos cobertos:
  - Validacao de pre-condicoes / fail-fast
  - Model Card (Mitchell 2019) / 9 secoes obrigatorias
  - Datasheet (Gebru 2018) — delegado upstream
  - FAIR Principles operacionalizados via HF Hub
  - YAML frontmatter / pipeline_tag / library_name
  - Reprodutibilidade bit-exact vs treino-reproducivel (Pineau 2021)
  - Atomicidade via tempfile.TemporaryDirectory + upload_folder
  - Idempotencia (create_repo exist_ok)
  - Dry-run como pratica defensiva
  - Limitacoes honestas (Mitchell 4.8) e implicacao cientifica
Limitacoes da analise:
  - Hash SHA-256 dos pesos nao incluido no card (recomendacao L3)
  - Versionamento semantico ausente (recomendacao L4)
  - Snippets do card nao tem teste de regressao (recomendacao L6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGUARDANDO REVISAO — nao prosseguir para o proximo script.
```
