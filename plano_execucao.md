# Plano de Execução — Specs Detalhados (Fases 1–4)

**Propósito:** Spec operacional no estilo *speckit*. Para cada tarefa técnica das Fases 1 a 4 do [plano_correcao.md](plano_correcao.md), este documento define Objetivo, Entradas, Passos, Critério de Aceitação e Saída Esperada — o suficiente para que uma IA ou uma pessoa recém-chegada execute sem alucinar decisões.

**Escopo:** Fases 1, 2, 3 e 4 (código). A Fase 5 (redação) permanece no [plano_correcao.md](plano_correcao.md) sem expansão — texto não alucina da mesma forma que código.

**Convenções:**
- Todos os caminhos são relativos à raiz do repositório: `GNN-Fake-News/`.
- Comandos shell assumem execução no diretório `Training/03_Mega_Research/` salvo indicação contrária.
- `grep` é usado como verificação pós-condição.
- BERT usado em todo o projeto: `paraphrase-multilingual-mpnet-base-v2` (definido em `00_construir_grafos_fakenewsnet.py:88`).
- Dimensão de embedding BERT: **768**. Features posicionais adicionam 3 dims → total **771**.

---

## Fase 1 — Refatoração Base

### 1.1. Consolidar imports do `GCNClassifier` (Incoerência 17)

**Objetivo:** Eliminar reimplementações inline de `class GCNClassifier` em scripts de análise, centralizando em [gcn_model.py](Training/03_Mega_Research/gcn_model.py). Garante que uma mudança arquitetural futura afete todos os experimentos uniformemente.

**Entradas:**
- [gcn_model.py](Training/03_Mega_Research/gcn_model.py) já existe, já aceita parâmetro `seed` (default 12345).
- Arquivos que hoje têm `class GCNClassifier` inline (confirmado via `grep -l "class GCNClassifier" Training/03_Mega_Research/*.py`):
  - `04_comparar_gcn_gat.py`
  - `matrizes_ablation.py`
  - `analise_topologia.py`
  - `duelo_modelos_frescos.py`
  - `treinar_upfd_robusto.py`
  - `treinar_mega_dataset.py`
  - `restaurar_upfd.py`
  - `matriz_confusao.py`
- **NÃO tocar:** 06, 07, 08, 09 — já usam `from gcn_model import GCNClassifier`. Verificação prévia: `grep "class GCNClassifier" 06_*.py 07_*.py 08_*.py 09_*.py` deve retornar zero matches.

**Passos:**
1. Em cada arquivo da lista acima, localizar a definição inline `class GCNClassifier(torch.nn.Module):` (geralmente ~30 linhas).
2. Remover todo o bloco da classe inline.
3. Adicionar no topo do arquivo, junto aos outros imports: `from gcn_model import GCNClassifier`.
4. Verificar as chamadas ao construtor. Se algum arquivo instanciava sem `seed`, manter o default 12345 — não é necessário alterar chamadas.
5. Rodar cada script em modo de smoke test (comando definido em cada arquivo — tipicamente `python <arquivo>.py --help` se aceitar flags, senão execução curta) para confirmar import e ausência de `NameError`.

**Critério de aceitação:**
- `grep -l "class GCNClassifier" Training/03_Mega_Research/*.py` retorna **apenas** `gcn_model.py`.
- `python -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" <arquivos_afetados>` não levanta SyntaxError.
- F1 e accuracy reportados pelos scripts com seed 12345 permanecem idênticos ao snapshot anterior (diferença ≤ 1e-6). Se o arquivo não salva baseline, verificar que não lança exceção em execução curta.

**Saída esperada:**
- 8 arquivos modificados. Cada um perde ~30 linhas (classe inline) e ganha 1 linha (import).
- Commit: `refactor: consolida GCNClassifier em gcn_model.py (Inc. 17)`.

---

### 1.2. Mudar early stopping para F1-macro (Incoerência 16)

**Objetivo:** Alinhar o critério de seleção do melhor checkpoint com a métrica de avaliação final. Hoje o early stopping monitora `val_acc`; a avaliação reporta `f1_macro`. Em datasets desbalanceados, essa inconsistência pode selecionar o modelo errado.

**Entradas:**
- Arquivos com padrões de early stopping (via `grep -l "val_acc\|val_f1\|best_acc\|best_f1\|early_stop" Training/03_Mega_Research/*.py`):
  - `03_treinar_gat.py`
  - `04_comparar_gcn_gat.py`
  - `05_treinar_sage.py`
  - `06_comparar_gcn_gat_sage.py`
  - `07_upfd_benchmark_triplo.py`
  - `09_teste_significancia.py` (já usa F1 — apenas confirmar)
  - `treinar_upfd_robusto.py`
- Biblioteca: `sklearn.metrics.f1_score`.

**Passos:**
1. Em cada loop de treinamento, localizar a função de validação (tipicamente `def validar(...)` ou bloco `model.eval()`).
2. Substituir qualquer `val_acc = accuracy_score(...)` por:
   ```python
   val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
   ```
3. Substituir a comparação de early stopping `if val_acc > best_acc:` por `if val_f1 > best_f1:`.
4. Renomear variáveis de rastreamento: `best_acc → best_f1`, `best_state → melhor_state_f1` (ou equivalente consistente no arquivo).
5. **Preservar** a lógica de restauração do melhor checkpoint no fim do treino: `model.load_state_dict(best_state)` — apenas a métrica de referência muda.
6. Confirmar que os prints finais ainda reportam F1 (e opcionalmente accuracy também, como informação secundária).

**Critério de aceitação:**
- `grep "val_acc\|best_acc" Training/03_Mega_Research/{03,04,05,06,07}_*.py treinar_upfd_robusto.py` retorna zero matches (ou só em comentários).
- `grep "val_f1\|best_f1" ...` aparece em cada arquivo modificado.
- Smoke test: rodar `python 07_upfd_benchmark_triplo.py --dataset fakenewsnet` com configuração mínima (epochs reduzidas se houver flag). Log deve conter linha `val_f1 = 0.XXXX` durante o treino.
- Ao final do treino, o modelo restaurado tem `val_f1` igual ao maior registrado durante as épocas.

**Saída esperada:**
- 6-7 arquivos modificados.
- Nos logs de treino, todas as mensagens de progresso reportam F1-macro como métrica primária.
- Commit: `fix: early stopping baseado em f1_macro com best-checkpoint restore (Inc. 16)`.

---

### 1.3. Remover narrativas causais auto-geradas (Incoerência 15)

**Objetivo:** Eliminar frases interpretativas automáticas no relatório de [07_upfd_benchmark_triplo.py](Training/03_Mega_Research/07_upfd_benchmark_triplo.py) que afirmam "GAT vence" ou similar sem verificar significância estatística.

**Entradas:**
- [07_upfd_benchmark_triplo.py](Training/03_Mega_Research/07_upfd_benchmark_triplo.py) linhas 353, 354, 389, 399 (confirmado via grep). Lógica atual:
  - Linha 353: `ranking = sorted(resultados.keys(), key=lambda n: resultados[n]["f1"], reverse=True)`
  - Linha 354: `vencedor = ranking[0]`
  - Linha 389: `f"  -> Ranking em F1: {' > '.join(ranking)}"`
  - Linha 399: `f"  Modelo com maior F1 nesta execucao: {vencedor}"`
- Possíveis condicionais adicionais `if vencedor == "GAT": print("...")` — grepar por `if vencedor ==` para confirmar.

**Passos:**
1. Abrir [07_upfd_benchmark_triplo.py](Training/03_Mega_Research/07_upfd_benchmark_triplo.py).
2. Remover qualquer bloco condicional `if vencedor == "<modelo>":` que imprima texto interpretativo (ex.: "atenção diferencia super-spreaders").
3. Manter as linhas 353, 354, 389, 399 (ranking numérico é factual). Apenas trocar a frase da linha 399 para ser neutra:
   - Antes: `f"  Modelo com maior F1 nesta execucao: {vencedor}"`
   - Depois: `f"  Modelo com maior F1 nesta execução (sem teste de significância): {vencedor}"`
4. Rodar o script uma vez e inspecionar o arquivo de relatório gerado em `Execution/results/upfd_benchmark_fakenewsnet/relatorio.txt`. Garantir ausência de frases como "atenção diferencia super-spreaders", "GAT supera", etc.

**Critério de aceitação:**
- `grep -n "if vencedor" Training/03_Mega_Research/07_upfd_benchmark_triplo.py` retorna zero matches (ou só comentários).
- O relatório gerado contém tabela numérica + ranking, **sem** frases causais.
- Pipeline completo continua rodando sem erros.

**Saída esperada:**
- 1 arquivo modificado (07).
- Novo `relatorio.txt` sem narrativa automática.
- Commit: `fix: remove narrativas causais auto-geradas do benchmark (Inc. 15)`.

---

## Fase 2 — Baselines, Pareamento e Confound

### 2A.1. Baseline textual oficial (Erro 2 / Incoerência 12)

**Objetivo:** Medir o "teto textual" — qual F1 um classificador simples (Regressão Logística, Random Forest) atinge usando apenas o BERT do título do artigo? Esse número é o baseline a ser superado pelas GNNs para a hipótese central do TCC se sustentar.

**Entradas:**
- Arquivos `.pt` em [Training/03_Mega_Research/data/](Training/03_Mega_Research/data/):
  - `fakenewsnet_train.pt`, `fakenewsnet_val.pt`, `fakenewsnet_test.pt`.
- Cada grafo é `torch_geometric.data.Data` com `x [N, 768]`, onde `x[0]` é o embedding BERT do título (raiz).
- Dependência: **aguardar Fase 2B.1** para usar os mesmos folds — ou criar script aceitando os folds via arquivo externo.
- Bibliotecas: `sklearn.linear_model.LogisticRegression`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.metrics.f1_score`.

**Passos:**
1. Criar `Training/03_Mega_Research/10_baseline_textual.py`.
2. Script deve:
   ```python
   import torch
   from pathlib import Path
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import f1_score, accuracy_score
   from sklearn.model_selection import StratifiedKFold

   DATA_DIR = Path(__file__).parent / "data"
   # Carregar todos os grafos (unindo train+val+test para o k-fold)
   grafos = []
   for split in ["train", "val", "test"]:
       grafos += torch.load(DATA_DIR / f"fakenewsnet_{split}.pt")

   # Extrair feature de raiz e label de cada grafo
   X = torch.stack([g.x[0] for g in grafos]).numpy()   # [N, 768]
   y = torch.tensor([g.y.item() for g in grafos]).numpy()
   ```
3. Se Fase 2B.1 já produziu arquivo de folds (`folds_fnn.pt`), carregar e iterar sobre os 10 folds. Caso contrário, instanciar `StratifiedKFold(n_splits=10, random_state=42, shuffle=True)` diretamente aqui.
4. Para cada fold, treinar Regressão Logística (`max_iter=1000`, `random_state=42`) e Random Forest (`n_estimators=200`, `random_state=42`).
5. Computar F1-macro e accuracy por fold, agregar média ± desvio-padrão.
6. Exportar tabela em `Execution/results/baseline_textual/resultados.csv` com colunas `modelo, fold, f1_macro, f1_fake, accuracy`.

**Critério de aceitação:**
- Arquivo `10_baseline_textual.py` existe.
- Execução `python 10_baseline_textual.py` termina sem exceções.
- CSV `Execution/results/baseline_textual/resultados.csv` gerado com 20 linhas (2 modelos × 10 folds).
- Última linha do log termina com formato:
  ```
  [RESUMO] LogReg: F1=0.XXX ± 0.XXX | RandomForest: F1=0.XXX ± 0.XXX
  ```
- Valor de F1 da LogReg aproxima-se do histórico de 0.85 (erros_correcao.md linha 39).

**Saída esperada:**
- 1 arquivo novo: `10_baseline_textual.py`.
- CSV em `Execution/results/baseline_textual/resultados.csv`.
- Commit: `feat: baseline textual com LogReg e RandomForest (Inc. 12, Erro 2)`.

---

### 2B.1. Compartilhamento de folds via StratifiedKFold (Erro 4-A)

**Objetivo:** Instanciar os folds do K-fold estratificado **uma única vez** e salvá-los em disco, para que Baseline Textual (2A), GNN Original (controle bugado), GNN Estrutural (Pos-Full) e variantes (Pos-min, Pos-grau) rodem **exatamente nos mesmos splits**. Assim o `ttest_rel` mede diferenças entre modelos, não ruído de partição.

**Entradas:**
- Grafos .pt do FakeNewsNet (train + val + test unidos).
- `sklearn.model_selection.StratifiedKFold`.

**Passos:**
1. Criar script utilitário `Training/03_Mega_Research/gerar_folds.py` com:
   ```python
   import torch
   from pathlib import Path
   from sklearn.model_selection import StratifiedKFold

   DATA_DIR = Path(__file__).parent / "data"
   grafos = []
   for split in ["train", "val", "test"]:
       grafos += torch.load(DATA_DIR / f"fakenewsnet_{split}.pt")

   y = [g.y.item() for g in grafos]
   skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

   folds = []
   for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(grafos)), y)):
       folds.append({
           "fold_idx": fold_idx,
           "train_idx": train_idx.tolist(),
           "test_idx": test_idx.tolist(),
       })

   torch.save(folds, DATA_DIR / "folds_fnn.pt")
   print(f"[OK] {len(folds)} folds salvos em {DATA_DIR / 'folds_fnn.pt'}")
   ```
2. Rodar `python gerar_folds.py` uma vez.
3. Atualizar [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py) para carregar `folds_fnn.pt` em vez de instanciar novo KFold. Substituir o loop atual de runs por loop de folds.
4. Dentro de cada fold, instanciar todos os modelos com `seed=fold_idx` (garante que a variação entre modelos seja pareada na inicialização também).
5. Expor função `iterar_folds()` reutilizável por 2A.1 e por Fase 4.

**Critério de aceitação:**
- Arquivo `folds_fnn.pt` existe em `Training/03_Mega_Research/data/`.
- `folds_fnn.pt` ao ser carregado retorna lista de 10 dicts, cada um com chaves `fold_idx`, `train_idx`, `test_idx`.
- Soma dos tamanhos de `test_idx` em todos os folds = total de grafos (sem sobreposição entre folds de teste).
- `09_teste_significancia.py` executa usando os folds salvos; mesmo seed produz mesmos índices.

**Saída esperada:**
- 1 arquivo novo: `gerar_folds.py`.
- 1 arquivo modificado: `09_teste_significancia.py`.
- 1 dataset auxiliar: `data/folds_fnn.pt`.
- Commit: `feat: StratifiedKFold compartilhado entre modelos (Erro 4)`.

---

### 2C.1. Diagnóstico do confound de tamanho (Erro 3)

**Objetivo:** Quantificar quanto do sinal aprendido pelas GNNs pode ser explicado apenas pelo número de nós do grafo. Se um Random Forest usando só `num_nodes` como feature atinge F1 > 0.65, o confound é sério e exige mitigação (subamostragem pareada).

**Entradas:**
- Grafos .pt do FakeNewsNet.
- `folds_fnn.pt` da Fase 2B.1.
- `RandomForestClassifier`.

**Passos:**
1. Criar `Training/03_Mega_Research/11_diagnostico_confound.py`.
2. Para cada grafo, extrair `num_nodes = g.num_nodes`.
3. Treinar duas variantes no mesmo k-fold de `folds_fnn.pt`:
   - **Variante A (baseline de tamanho):** `X = num_nodes.reshape(-1, 1)` — só o número.
   - **Variante B (BERT + tamanho):** `X = concat(bert_raiz, num_nodes)` — verifica se `num_nodes` adiciona sinal sobre BERT.
4. Reportar F1-macro médio ± desvio-padrão por variante.
5. Aplicar **Gate de Decisão:**
   - Se F1(Variante A) > 0.65: disparar subamostragem pareada (passo 6).
   - Se F1(Variante A) ≤ 0.65: documentar resultado; confound existe mas é fraco. Pular passo 6.
6. **Subamostragem pareada (só se Gate ativar):**
   - Criar bins de tamanho: [2, 10), [10, 20), [20, 50), [50, 100).
   - Dentro de cada bin, amostrar igual número de fake e real (mínimo entre as duas classes).
   - Salvar `data/fakenewsnet_balanceado.pt` com grafos pareados.
   - Retreinar LogReg(BERT) sobre dataset balanceado; se F1 cair significativamente, confounder confirmado.

**Critério de aceitação:**
- Arquivo `11_diagnostico_confound.py` existe e executa.
- Relatório em `Execution/results/confound_diagnostico/relatorio.txt` contém:
  - F1 da Variante A por fold + média ± std.
  - F1 da Variante B por fold + média ± std.
  - Decisão do Gate ("disparar subamostragem: SIM/NÃO").
- Se Gate = SIM, existe `data/fakenewsnet_balanceado.pt` e relatório adicional com F1 sobre dataset balanceado.

**Saída esperada:**
- 1 arquivo novo: `11_diagnostico_confound.py`.
- 1 relatório: `Execution/results/confound_diagnostico/relatorio.txt`.
- Condicional: `data/fakenewsnet_balanceado.pt` se Gate disparou.
- Commit: `feat: diagnóstico quantitativo do confound de tamanho (Erro 3)`.

---

## Fase 3 — Positional Encodings

### 3.0. Arquivamento do baseline bugado

**Objetivo:** Preservar os arquivos `.pt` atuais (com features nodais idênticas) **antes** de regenerá-los, para que possam ser usados como grupo de controle na Fase 4.1.

**Entradas:**
- Diretório `Training/03_Mega_Research/data/` contendo `fakenewsnet_{train,val,test}.pt`.

**Passos:**
1. Criar diretório `Training/03_Mega_Research/data/fakenewsnet_bugado_original/`.
2. Copiar (não mover) os 3 arquivos: `fakenewsnet_train.pt`, `fakenewsnet_val.pt`, `fakenewsnet_test.pt`.
3. Criar arquivo `Training/03_Mega_Research/data/fakenewsnet_bugado_original/README.md` explicando o status:
   ```markdown
   # Grupo de Controle Bugado
   Estes arquivos contêm os grafos originais com features nodais idênticas 
   (Erro 1 do erros_correcao.md). Preservados para comparação contra a versão 
   corrigida com positional encodings (Fase 3.1).
   Data do arquivamento: <YYYY-MM-DD>
   ```
4. Verificar tamanhos: `ls -la data/fakenewsnet_bugado_original/` vs `ls -la data/*.pt`.

**Critério de aceitação:**
- Diretório `data/fakenewsnet_bugado_original/` contém 3 arquivos `.pt` + README.
- Arquivos são byte-idênticos aos originais: `sha256sum data/*.pt data/fakenewsnet_bugado_original/*.pt` mostra pares idênticos.

**Saída esperada:**
- Diretório e conteúdo preservados.
- Commit: `chore: arquiva grafos bugados para grupo de controle (Fase 3)`.

---

### 3.1. Design das features posicionais (Erro 1)

**Objetivo:** Reescrever o construtor de grafos para dar a cada nó **features diferentes** entre si, quebrando a degeneração matemática da GCN sobre features idênticas.

**⚠ Caveat sobre `pos=i/N`:** O campo `tweet_ids` no CSV do FakeNewsNet é uma string tab-separada de IDs de tweets. O script atual apenas divide por `\t` e usa a ordem resultante. Se a coleta original foi cronológica (o que é comum em datasets assim, mas **não está documentado**), `pos=i/N` aproxima ordem temporal. **Verificação prévia recomendada:** inspecionar 10 linhas do CSV e comparar tweet_ids com timestamps via API Twitter (se acessível); se não for possível, tratar `pos` como "ordem de aparição no CSV" nas conclusões do TCC (Fase 5A.5).

**Entradas:**
- [00_construir_grafos_fakenewsnet.py](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py) atual.
- Dataset CSV do FakeNewsNet (download já coberto pelo script).
- Fase 3.0 concluída (backup do bugado existe).

**Passos:**
1. Modificar a função `construir_grafo(...)` em [00_construir_grafos_fakenewsnet.py:101](Training/03_Mega_Research/00_construir_grafos_fakenewsnet.py#L101). Assinatura atualizada:
   ```python
   def construir_grafo(embedding, tweet_ids_str, label, max_nos, 
                       N_max_global, feature_variant="full"):
   ```
2. No início do `main()`, calcular `N_max_global` uma vez, **somente sobre o conjunto de treino** após o split (para evitar vazamento de teste):
   ```python
   N_max_global = max(len(ids.split("\t")) for ids in df_train["tweet_ids"])
   ```
3. Reescrever o bloco das features:
   ```python
   x_raiz = torch.tensor(embedding, dtype=torch.float)    # [768]
   num_filhos = min(len(ids), max_nos - 1)
   num_nos    = 1 + num_filhos

   # Raiz: is_root=1, grau_norm=N/N_max_global, pos=0
   pos_raiz = torch.tensor([1.0, num_filhos / N_max_global, 0.0])
   x_raiz_full = torch.cat([x_raiz, pos_raiz])           # [771]

   # Filhos: is_root=0, grau_norm=0, pos=i/N
   linhas_filhos = []
   for i in range(1, num_filhos + 1):
       pos_filho = torch.tensor([0.0, 0.0, i / num_filhos])
       linhas_filhos.append(torch.cat([x_raiz, pos_filho]))

   x = torch.stack([x_raiz_full] + linhas_filhos)        # [num_nos, 771]
   ```
4. Suportar 3 variantes via `feature_variant`:
   - `"full"`: [BERT, is_root, grau_norm, pos]  → 771 dims.
   - `"pos-min"`: [BERT, is_root, pos]  (grau zerado) → 771 dims com posição 769 sempre 0.
   - `"pos-grau"`: [BERT, is_root, grau_norm]  (pos zerada) → 771 dims com posição 770 sempre 0.
5. Adicionar assertion de sanidade no final de `construir_grafo`:
   ```python
   assert x.unique(dim=0).shape[0] >= min(num_nos, 3), \
       f"Features nodais ainda idênticas em grafo {label=}, {num_nos=}"
   ```
6. Adicionar flag CLI `--feature-variant {full,pos-min,pos-grau}` em `main()`. Passar para `construir_grafo`.
7. Adicionar flag CLI `--output-suffix` para gravar em diretórios distintos (ex.: `data/fakenewsnet_posfull/`, `data/fakenewsnet_posmin/`, `data/fakenewsnet_posgrau/`).

**Critério de aceitação:**
- `python 00_construir_grafos_fakenewsnet.py --feature-variant full` termina sem `AssertionError`.
- Arquivo gerado tem `data.x.shape[1] == 771` para qualquer grafo: `python -c "import torch; g=torch.load('data/fakenewsnet_posfull/fakenewsnet_train.pt')[0]; print(g.x.shape)"`.
- `assert` dentro do script nunca falha em run completo.
- Verificação manual de 1 grafo: `g.x[0, 768] == 1.0` (raiz marcada), `g.x[1:, 768].sum() == 0.0` (filhos desmarcados).

**Saída esperada:**
- 1 arquivo modificado: `00_construir_grafos_fakenewsnet.py`.
- 3 diretórios de saída (um por variante, gerados nas execuções de 3.2):
  - `data/fakenewsnet_posfull/`
  - `data/fakenewsnet_posmin/`
  - `data/fakenewsnet_posgrau/`
- Commit: `feat: features posicionais (is_root, grau_norm, pos) no construtor (Erro 1)`.

---

### 3.1.bis. Diagnóstico intra-encoding

**Objetivo:** Isolar de qual feature posicional vem o ganho (se houver): posição ordinal, grau do nó, ou ambos combinados.

**Entradas:**
- `00_construir_grafos_fakenewsnet.py` atualizado com `--feature-variant`.
- Fase 2B.1 (`folds_fnn.pt`) concluída.
- Modelos GCN, GAT, SAGE (via imports).

**Passos:**
1. Criar `Training/03_Mega_Research/12_ablation_intra_encoding.py`.
2. Para cada variante (`full`, `pos-min`, `pos-grau`):
   - Carregar grafos do diretório correspondente.
   - Treinar GCN (ou o modelo escolhido) sobre os 10 folds de `folds_fnn.pt`.
   - Coletar F1-macro por fold.
3. Aplicar **Gate de Posição:**
   - Se F1(pos-min) > F1(baseline textual) com `p < 0.05` no `ttest_rel`: ganho atribuível à posição ordinal.
4. Aplicar **Gate de Confound:**
   - Se F1(pos-grau) > F1(pos-min) + 0.01: o sinal dominante é grau (popularidade), não posição. **Documentar como confirmação do Erro 3 na Fase 5B.2** em vez de reportar como vitória topológica.
5. Exportar tabela pareada:
   ```
   variante   | F1_medio | F1_std | p-val vs Baseline
   ------------|----------|--------|------------------
   pos-min    | 0.XXX    | 0.XXX  | 0.XXX
   pos-grau   | 0.XXX    | 0.XXX  | 0.XXX
   pos-full   | 0.XXX    | 0.XXX  | 0.XXX
   ```

**Critério de aceitação:**
- Script executa sem exceções.
- Tabela final em `Execution/results/ablation_intra_encoding/tabela.csv`.
- Relatório `relatorio.txt` documenta qual gate disparou e a interpretação.

**Saída esperada:**
- 1 arquivo novo: `12_ablation_intra_encoding.py`.
- Tabela e relatório em `Execution/results/ablation_intra_encoding/`.
- Commit: `feat: ablation intra-encoding (pos vs grau vs full)`.

---

### 3.2. Regenerar arquivos .pt

**Objetivo:** Produzir os `.pt` da variante principal (`pos-full`) e das variantes de ablação, sem sobrescrever o backup da 3.0.

**Entradas:**
- Fase 3.0 concluída (backup existe).
- Fase 3.1 concluída (`00_construir_grafos_fakenewsnet.py` com flag `--feature-variant`).

**Passos:**
1. Rodar três vezes:
   ```
   python 00_construir_grafos_fakenewsnet.py --feature-variant full     --output-suffix posfull
   python 00_construir_grafos_fakenewsnet.py --feature-variant pos-min  --output-suffix posmin
   python 00_construir_grafos_fakenewsnet.py --feature-variant pos-grau --output-suffix posgrau
   ```
2. Verificar que `data/fakenewsnet_bugado_original/` permanece intacto (checksum).

**Critério de aceitação:**
- 3 diretórios criados em `data/`: `fakenewsnet_posfull/`, `fakenewsnet_posmin/`, `fakenewsnet_posgrau/`, cada um com `_train.pt`, `_val.pt`, `_test.pt`.
- Em cada diretório, `g.x.shape[1] == 771`.
- `data/fakenewsnet_bugado_original/` inalterado (sha256 igual ao da Fase 3.0).

**Saída esperada:**
- 9 arquivos `.pt` novos (3 variantes × 3 splits).
- Commit: `data: gera .pt para variantes posfull, posmin, posgrau (Fase 3)`.

---

## Fase 4 — Reavaliação

### 4.1. Benchmark contrastivo geral

**Objetivo:** Comparar, nos mesmos folds, três famílias de modelos: (a) GNN treinada sobre o dataset bugado original, (b) GNN sobre Pos-Full, (c) Baseline textual. O ganho da corrige sobre o bugado mede o efeito da correção; o ganho da corrige sobre o baseline textual mede se topologia carrega sinal.

**Entradas:**
- `data/fakenewsnet_bugado_original/` (3 arquivos).
- `data/fakenewsnet_posfull/` (3 arquivos).
- `folds_fnn.pt`.
- Modelos: GCN, GAT, SAGE + LogReg, RandomForest.

**Passos:**
1. Atualizar [07_upfd_benchmark_triplo.py](Training/03_Mega_Research/07_upfd_benchmark_triplo.py) para aceitar flag `--data-suffix {bugado_original,posfull}` e carregar o diretório correspondente.
2. Rodar 2 execuções completas:
   ```
   python 07_upfd_benchmark_triplo.py --data-suffix bugado_original --use-folds folds_fnn.pt
   python 07_upfd_benchmark_triplo.py --data-suffix posfull         --use-folds folds_fnn.pt
   ```
3. Rodar `10_baseline_textual.py` (já existente da Fase 2A.1) no mesmo `folds_fnn.pt`.
4. Coletar resultados em tabela unificada `Execution/results/benchmark_fase4/comparativo.csv`:
   ```
   modelo           | dataset          | F1_medio | F1_std
   -----------------|------------------|----------|-------
   LogReg-BERT      | textual          | 0.XXX    | 0.XXX
   RandomForest-BERT| textual          | 0.XXX    | 0.XXX
   GCN              | bugado_original  | 0.XXX    | 0.XXX
   GCN              | posfull          | 0.XXX    | 0.XXX
   GAT              | bugado_original  | 0.XXX    | 0.XXX
   GAT              | posfull          | 0.XXX    | 0.XXX
   SAGE             | bugado_original  | 0.XXX    | 0.XXX
   SAGE             | posfull          | 0.XXX    | 0.XXX
   ```
5. Gerar uma matriz de confusão por modelo×dataset (8 matrizes no total) e salvar como PNGs individuais em `Execution/results/benchmark_fase4/matrizes/`.

**Critério de aceitação:**
- `comparativo.csv` existe e tem 8 linhas numéricas.
- 8 PNGs de matriz de confusão existem.
- Nenhum valor é NaN; nenhuma execução crashou.

**Saída esperada:**
- 1 arquivo modificado: `07_upfd_benchmark_triplo.py` (adiciona flag `--data-suffix`).
- Resultados em `Execution/results/benchmark_fase4/`: `comparativo.csv` + `matrizes/*.png`.
- Commit: `feat: benchmark contrastivo bugado vs posfull vs textual (Fase 4.1)`.

---

### 4.2. K-fold pareado com ttest_rel

**Objetivo:** Aplicar `scipy.stats.ttest_rel` par a par sobre as métricas de F1 por fold. Reportar p-valor e decisão de não-rejeição de H0.

**Entradas:**
- `comparativo.csv` por fold (ampliar 4.1 para salvar por-fold, não só médias).
- `folds_fnn.pt`.
- `scipy.stats.ttest_rel`.

**Passos:**
1. Estender o CSV da Fase 4.1 para granularidade por fold: `Execution/results/benchmark_fase4/por_fold.csv` com colunas `modelo, dataset, fold, f1_macro, f1_fake`.
2. Atualizar [09_teste_significancia.py](Training/03_Mega_Research/09_teste_significancia.py) para:
   - Ler `por_fold.csv`.
   - Para cada par de modelos (ex.: GCN-posfull vs LogReg-BERT), aplicar `ttest_rel(f1_gcn, f1_logreg)` sobre os 10 folds.
   - Salvar tabela de p-valores.
3. Aplicar **Critério de não-rejeição de H0:**
   - Se `p-valor > 0.05` entre GCN-posfull e LogReg-BERT: Fase 5 encerra com conclusão negativa (5B.1 ramo `ns`).
   - Se `p-valor < 0.05` e F1(GCN-posfull) > F1(LogReg): Fase 5 encerra com conclusão positiva (5B.1 ramo positivo).
4. Exportar tabela em LaTeX para uso direto no TCC: `Execution/results/benchmark_fase4/tabela_significancia.tex`.

**Critério de aceitação:**
- `por_fold.csv` com 80 linhas (8 combinações × 10 folds).
- `tabela_significancia.tex` contém tabela LaTeX com colunas `Par de modelos | t | p | decisão`.
- Log do script imprime uma linha final explícita com a decisão principal (posfull vs baseline textual).

**Saída esperada:**
- Modificação em `09_teste_significancia.py`.
- 2 arquivos novos em `Execution/results/benchmark_fase4/`: `por_fold.csv`, `tabela_significancia.tex`.
- Commit: `feat: ttest_rel pareado entre modelos nos mesmos folds (Erro 4, Fase 4.2)`.

---

### 4.3. Prevalência em domínio transferido (Bluesky)

**Objetivo:** Medir AUPRC (não F1) do modelo treinado em FakeNewsNet quando aplicado ao Bluesky, que tem prevalência extrema (0.58% fake). F1 é enganoso sob desbalanceamento severo; AUPRC captura melhor a utilidade prática.

**Entradas:**
- Melhor modelo da Fase 4.1 (escolha: GCN-posfull ou o de maior F1 na 4.1), salvo em `Execution/weights/`.
- Grafos Bluesky: `data/grafos_bluesky_test.pt` (já gerados por [02_construir_grafos_bluesky.py](Training/03_Mega_Research/02_construir_grafos_bluesky.py)).
- **⚠ Caveat:** Muitos grafos Bluesky degeneraram para 1 nó, 0 arestas (documentado em [08_inferencia_cruzada.py:13](Training/03_Mega_Research/08_inferencia_cruzada.py#L13)). Avaliar também um subset filtrado com N ≥ 2.
- `sklearn.metrics.average_precision_score`.

**Passos:**
1. Atualizar ou criar [08_inferencia_cruzada.py](Training/03_Mega_Research/08_inferencia_cruzada.py) para:
   - Carregar melhor modelo treinado em FNN.
   - Fazer forward em grafos Bluesky para obter **scores de probabilidade** (não labels): `probs = torch.softmax(logits, dim=1)[:, 1]`  (prob de fake).
   - Calcular `AUPRC = average_precision_score(y_true, probs)`.
2. Gerar duas linhas de resultado: com todos os grafos, e apenas com grafos N ≥ 2.
3. Aplicar **Gate Secundário:**
   - Se `AUPRC < 0.10`: registrar explicitamente que "modelo não transfere".
   - Se `AUPRC ∈ [0.10, 0.30]`: transferência parcial.
   - Se `AUPRC ≥ 0.30`: transferência razoável.

**Critério de aceitação:**
- Arquivo `Execution/results/bluesky_auprc.csv` com colunas `modelo, subset, auprc, prevalencia`.
- Log imprime classificação do Gate Secundário.
- O TCC (Fase 5B.4) poderá citar o número diretamente.

**Saída esperada:**
- Modificação em `08_inferencia_cruzada.py`.
- 1 arquivo novo: `Execution/results/bluesky_auprc.csv`.
- Commit: `feat: AUPRC em Bluesky para inferência cruzada (Inc. 11, Fase 4.3)`.

---

## Matriz de dependências entre tarefas

| Tarefa | Depende de |
|---|---|
| 1.1, 1.2, 1.3 | Nada (pode paralelizar) |
| 2A.1 | 2B.1 (preferível) |
| 2B.1 | 1.2 (early stopping novo) |
| 2C.1 | 2B.1 |
| 3.0 | Nada (copia o que existe) |
| 3.1 | 3.0 (backup antes de alterar) |
| 3.1.bis | 3.1 + 2B.1 + 2A.1 |
| 3.2 | 3.1 |
| 4.1 | 3.2 + 2A.1 + 2B.1 |
| 4.2 | 4.1 |
| 4.3 | 4.1 (melhor modelo já treinado) |

## Ordem de execução recomendada

1. **Dia 1:** Fase 1 inteira (paralelo).
2. **Dia 2:** Fase 2B.1 (gerar folds) → Fase 2A.1 (baseline textual).
3. **Dia 3:** Fase 2C.1 (confound) em paralelo com Fase 3.0 (arquivamento).
4. **Dia 4-5:** Fase 3.1 (implementar), Fase 3.2 (gerar .pt).
5. **Dia 6:** Fase 3.1.bis (ablation).
6. **Dia 7:** Fase 4.1 (benchmark).
7. **Dia 8:** Fase 4.2 (significância) + Fase 4.3 (Bluesky).
8. **Dia 9+:** Fase 5 (redação, fora deste documento).
