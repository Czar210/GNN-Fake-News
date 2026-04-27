# CLAUDE.md — Protocolo de Análise TCC: GNNs vs NLP para Detecção de Fake News

> Este arquivo é lido automaticamente pelo Claude Code ao iniciar qualquer sessão neste projeto.
> Ele define o comportamento, protocolo e restrições para TODAS as tarefas neste repositório.

---

## CONTEXTO DO PROJETO

Este projeto é um TCC de análise empírica que investiga a seguinte questão central:

> **"GNNs são uma alternativa viável para detecção de fake news ou ainda estão muito atrás das metodologias tradicionais que usam NLP? Quais são seus prós e contras?"**

O projeto contém scripts que implementam diferentes abordagens (GNN, NLP, híbridas) para
detecção de fake news. A tarefa do Claude é analisar esses scripts empiricamente e produzir
documentação técnica acadêmica rigorosa.

---

## ESTRUTURA DE PASTAS RELEVANTE

```
C:\Users\André\OneDrive - Fundação São Paulo\Desktop\tcc-gnn-fake-news\
└── GNN-Fake-News\
    └── Training\
        └── 03_Mega_Research\          ← PASTA DE TRABALHO PRINCIPAL
            ├── CLAUDE.md              ← este arquivo
            ├── [script_1.py]          ← scripts a serem analisados
            ├── [script_2.py]
            ├── [script_N.py]
            └── theory\                ← DESTINO de todos os .md gerados
                ├── [script_1_doc.md]
                ├── [script_2_doc.md]
                └── ANALISE_COMPARATIVA_FINAL.md
```

**Regra de caminho:** Todos os arquivos `.md` produzidos devem ser salvos em:
`03_Mega_Research/theory/`

---

## PROTOCOLO OBRIGATÓRIO DE ANÁLISE

Ao receber instrução para analisar um script, Claude DEVE seguir EXATAMENTE estas 3 fases,
nesta ordem, sem pular etapas.

---

### ▶ FASE 1 — LEITURA E MAPEAMENTO INTERNO

1. Ler o script completo via filesystem antes de escrever qualquer linha de documentação
2. Construir mentalmente um inventário de todos os conceitos presentes:
   - Arquiteturas de modelos (GCN, GAT, GraphSAGE, GIN, BERT, RoBERTa, etc.)
   - Operações matemáticas (message passing, attention, softmax, pooling, etc.)
   - Métricas de avaliação (F1, AUC-ROC, Accuracy, Precision, Recall, MCC)
   - Datasets referenciados (FakeNewsNet, LIAR, ISOT, WELFake, POLITIFACT, GOSSIPCOP)
   - Técnicas de pré-processamento (tokenização, embedding, construção do grafo)
   - Funções de perda, otimizadores, schedulers, regularização
   - Hiperparâmetros e suas justificativas (ou ausência delas)
3. Identificar erros, ineficiências, code smells e boas práticas — anotar mas NÃO publicar ainda
4. Só avançar para Fase 2 após o inventário completo

---

### ▶ FASE 2 — COLETA DE MATERIAL ACADÊMICO

Para CADA conceito do inventário, buscar embasamento seguindo esta hierarquia:

#### FONTES ACEITAS (por ordem de preferência)

**Tier 1 — Obrigatório tentar primeiro:**
- Papers originais dos modelos via `web_search` + `web_fetch`
- ArXiv (cs.LG, cs.AI, cs.SI, cs.CL, cs.IR) — buscar por título exato
- Journals: IEEE Trans., ACM, Springer, Elsevier (Applied Soft Computing, IPM, etc.)
- Conferências top: NeurIPS, ICML, ICLR, ACL, EMNLP, WWW, KDD, AAAI, SIGIR

**Tier 2 — Material educacional de prestígio (quando Tier 1 não disponível):**
- Stanford CS224W (Machine Learning with Graphs): https://web.stanford.edu/class/cs224w/
- MIT OpenCourseWare
- CMU, Oxford, Cambridge course notes
- Distill.pub

**Tier 3 — Surveys e reviews (para contextualização ampla):**
- IEEE Transactions on Neural Networks and Learning Systems
- ACM Computing Surveys
- arXiv surveys com alta citação (>50 citas)

#### FONTES NUNCA ACEITAS como fonte primária:
- Medium, Towards Data Science, blogs pessoais
- Stack Overflow, Reddit, Quora, fóruns
- Documentação de bibliotecas (PyTorch, PyG, HuggingFace) isolada — podem complementar
- Wikipedia como fonte definitiva

#### ESTRATÉGIAS DE BUSCA POR CONCEITO:

| Conceito | Query de busca recomendada |
|----------|--------------------------|
| GCN | `"Kipf Welling semi-supervised classification graph convolutional networks ICLR 2017"` |
| GAT | `"Velickovic graph attention networks ICLR 2018"` |
| GraphSAGE | `"Hamilton inductive representation learning large graphs NeurIPS 2017"` |
| GIN | `"Xu how powerful graph neural networks ICLR 2019"` |
| BERT | `"Devlin BERT pre-training deep bidirectional transformers NAACL 2019"` |
| RoBERTa | `"Liu RoBERTa robustly optimized BERT pretraining 2019"` |
| Message Passing | `"Gilmer neural message passing quantum chemistry ICML 2017"` |
| GNN Fake News Survey | `"Gong fake news detection graph-based neural networks survey arXiv 2307.12639"` |
| GNN vs NLP benchmark | `"Krzywda comparative analysis GNN transformers fake news detection Electronics 2024"` |
| FakeNewsNet dataset | `"Shu FakeNewsNet data repository news content social context 2020"` |

**Para cada fonte encontrada, registrar obrigatoriamente:**
- Autores completos, Ano
- Título exato
- Venue (conferência/journal)
- DOI ou URL arXiv estável
- **Localização específica:** Seção X.Y, Equação N, Página P, Tabela T — NUNCA referência vaga

---

### ▶ FASE 3 — CRIAÇÃO DO ARQUIVO .md

Salvar em: `03_Mega_Research/theory/[nome_do_script]_doc.md`

O arquivo DEVE seguir esta estrutura completa:

```
# Documentação Técnica: [Nome do Script]

## Metadados
- **Arquivo analisado:** `[nome_exato_do_script.py]`
- **Caminho:** `03_Mega_Research/[nome_exato_do_script.py]`
- **Data de análise:** [data]
- **Tipo de abordagem:** GNN / NLP / Híbrido
- **Modelos principais:** [lista]
- **Datasets utilizados:** [lista]
- **Contribuição para a questão central:** [como este script informa o debate GNN vs NLP]

---

## 1. Visão Geral do Script
[3-4 parágrafos: o que o script faz, qual problema resolve,
qual metodologia usa, o que produz como saída]

---

## 2. Arquitetura e Componentes Principais

### 2.1 [Nome do Componente]
**Descrição técnica:**
[o que é, o que faz no contexto do script]

**Fundamento matemático:**
$$\text{Equação em LaTeX}$$
[Explicação variável por variável]

**Embasamento acadêmico:**
> 📖 **[Autor(es), Ano]** — "[Título completo]"
> *[Venue/Journal]* | DOI: `[doi]` ou arXiv: `[id]`
> **Localização:** Seção [X], Equação [N] / Página [P]
> **Relevância:** [por que este trecho específico embasa o conceito documentado]

**No código:**
> Linhas [X–Y]: `[trecho relevante]`
> [Explicação de como a teoria se manifesta na implementação]

---

## 3. Pipeline de Dados e Pré-processamento
[Mesma estrutura da Seção 2 para cada etapa do pipeline]

---

## 4. Construção do Grafo (se aplicável)
[Mesma estrutura — como nós, arestas e features são definidos]

---

## 5. Métricas de Avaliação

### 5.1 [Nome da Métrica]
**Fórmula:**
$$\text{Métrica} = \frac{\text{numerador}}{\text{denominador}}$$

**Interpretação no contexto de fake news:**
[por que esta métrica é adequada ou problemática para este problema]

**Embasamento acadêmico:**
> 📖 [referência]

---

## 6. Análise Empírica: Posicionamento na Questão Central

### 6.1 Pontos fortes desta abordagem
[análise crítica fundamentada em literatura]

### 6.2 Limitações identificadas
[análise crítica fundamentada em literatura]

### 6.3 Comparação com o estado da arte
| Abordagem | Accuracy | F1 | Dataset | Fonte |
|-----------|----------|-----|---------|-------|
| Este script | [val] | [val] | [dataset] | — |
| [Paper comparável] | [val] | [val] | [dataset] | [ref] |

> 📖 **Fonte da comparação:** [referência]

### 6.4 Resposta parcial à questão do TCC
[Como os resultados deste script especificamente contribuem para
a resposta da questão "GNNs são viáveis vs NLP?"]

---

## 7. Análise de Código

### 7.1 Erros identificados
```python
# ❌ Linha X — [descrição do erro]
[código problemático]

# ✅ Correção sugerida:
[código corrigido]
# Justificativa: [explicação técnica]
```

### 7.2 Ineficiências
[Análise de complexidade quando relevante: O(n), uso de memória, etc.]

### 7.3 Boas práticas observadas
[O que o script faz bem, com referência quando aplicável]

---

## 8. Referências Bibliográficas

[Lista numerada com TODAS as fontes usadas neste documento]
[Formato ABNT ou APA consistente, com DOI/URL para cada entrada]

1. AUTOR, Nome. **Título**. *Venue*, Ano. DOI: `xxx` / Disponível em: `url`
...

---

## 9. Glossário
| Termo | Definição | Fonte |
|-------|-----------|-------|
| [termo] | [definição precisa] | [ref] |
```

---

## REGRAS DE QUALIDADE — INEGOCIÁVEIS

1. **Zero conceitos sem fonte.** Se não encontrar Tier 1, usar Tier 2 e explicitar o tier.
2. **Localização específica obrigatória.** "Kipf & Welling, 2017" sem seção/equação = inválido.
3. **Toda equação explicada** variável por variável antes de conectar com o código.
4. **Toda afirmação empírica sobre performance** (ex: "GNNs têm accuracy inferior em X") deve
   citar paper com dados numéricos concretos.
5. **Conectar sempre** com a questão central do TCC na Seção 6.4 de cada documento.
6. **Um script por sessão.** Ao finalizar, emitir o sinal de conclusão e aguardar revisão.

---

## SINAL DE CONCLUSÃO OBRIGATÓRIO

Ao terminar cada script, Claude DEVE emitir exatamente este bloco antes de parar:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  SCRIPT DOCUMENTADO: [nome_do_script.py]
📄  Arquivo gerado: theory/[nome]_doc.md
📚  Fontes acadêmicas utilizadas: [N]
    [lista numerada de todas as referências]
🔍  Conceitos cobertos: [lista]
⚠️   Limitações: [o que não foi possível verificar com Tier 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴  AGUARDANDO REVISÃO — não prosseguir para o próximo script.
```

---

## PAPERS DE REFERÊNCIA CENTRAL (pré-validados)

Estes são os papers fundamentais do domínio. Usá-los como âncora inicial:

| # | Conceito | Referência | ID |
|---|----------|------------|----|
| 1 | GCN | Kipf & Welling (2017) "Semi-Supervised Classification with Graph Convolutional Networks" ICLR | arXiv:1609.02907 |
| 2 | GAT | Veličković et al. (2018) "Graph Attention Networks" ICLR | arXiv:1710.10903 |
| 3 | GraphSAGE | Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs" NeurIPS | arXiv:1706.02216 |
| 4 | GIN | Xu et al. (2019) "How Powerful are Graph Neural Networks?" ICLR | arXiv:1810.00826 |
| 5 | BERT | Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers" NAACL | arXiv:1810.04805 |
| 6 | RoBERTa | Liu et al. (2019) "RoBERTa: A Robustly Optimized BERT Pretraining Approach" | arXiv:1907.11692 |
| 7 | Message Passing | Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" ICML | arXiv:1704.01212 |
| 8 | GNN Fake News Survey | Gong et al. (2023) "Fake News Detection Through Graph-based Neural Networks: A Survey" | arXiv:2307.12639 |
| 9 | GNN Survey | Phan et al. (2023) "Fake news detection: A survey of GNN methods" Applied Soft Computing | DOI:10.1016/j.asoc.2023.110235 |
| 10 | GNN vs Transformer | Krzywda et al. (2024) "Comparative Analysis of GNNs and Transformers for Fake News Detection" Electronics | DOI:10.3390/electronics13234784 |
| 11 | FakeNewsNet | Shu et al. (2020) "FakeNewsNet: A Data Repository with News Content, Social Context..." Big Data | DOI:10.1089/big.2020.0062 |
| 12 | Graph Rep. Learning | Hamilton (2020) "Graph Representation Learning" Morgan & Claypool | Book |
