import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import sys

# Pega o diretório do script (C:\...\LIAR\src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Constantes (caminhos completos)
LIAR_FILES = {
    'train': os.path.join(SCRIPT_DIR, 'train.tsv'),
    'test': os.path.join(SCRIPT_DIR, 'test.tsv'),
    'valid': os.path.join(SCRIPT_DIR, 'valid.tsv')
}
LIAR_COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 
                'state', 'party', 'barely_true_c', 'false_c', 'half_true_c', 
                'mostly_true_c', 'pants_on_fire_c', 'context']

CORES_LABEL = {
    'pants-fire': '#FF0000', 'false': '#FF6347', 'barely-true': '#FFA500',
    'half-true': '#FFFF00', 'mostly-true': '#9ACD32', 'true': '#008000'
}

# =====================================================================
# <<< NOVA FUNÇÃO (Regra 3 do Manual: Responsabilidade Única) >>>
# Esta função limpa o texto para que seja seguro para o HTML/JS
# =====================================================================
def sanitizar_html(texto):
    """Limpa uma string para ser usada com segurança em um 'title' de HTML/JS."""
    texto_str = str(texto) # Garante que é string
    texto_str = texto_str.replace('"', '&quot;') # Substitui aspas duplas
    texto_str = texto_str.replace("'", '&#39;')  # Substitui aspas simples
    texto_str = texto_str.replace('\n', ' ')     # Remove novas linhas
    texto_str = texto_str.replace('\r', ' ')     # Remove novas linhas
    return texto_str

def verificar_arquivos_locais():
    print("Verificando arquivos locais do dataset LIAR...")
    arquivo_faltando = False
    for key, filename in LIAR_FILES.items():
        if not os.path.exists(filename):
            print(f"Erro: Arquivo não encontrado: {filename}")
            arquivo_faltando = True
    
    if arquivo_faltando:
        print("\nErro: Os arquivos .tsv não foram encontrados na pasta do script.")
        sys.exit(1)
    print("Arquivos do LIAR encontrados.")

def criar_grafo_liar(amostra_n=100):
    print("Iniciando a criação do grafo LIAR...")
    
    try:
        df_full = pd.read_csv(
            LIAR_FILES['train'], 
            sep='\t', 
            header=None, 
            names=LIAR_COLUMNS
        )
        print(f"\nDataFrame lido com sucesso: {df_full.shape[0]} linhas, {df_full.shape[1]} colunas.\n")
    except Exception as e:
        print(f"Erro ao ler {LIAR_FILES['train']}: {e}")
        return nx.DiGraph()

    df_amostra = df_full.sample(n=min(amostra_n, len(df_full)), random_state=42)
    print(f"Processando {len(df_amostra)} declarações (amostra)...")
    
    G = nx.DiGraph()
    
    for _, row in df_amostra.iterrows():
        statement_id = row['id']
        label = row['label']
        speaker = str(row['speaker']).strip()
        party = str(row['party']).strip()
        subject = str(row['subject']).strip()

        # =====================================================================
        # <<< MUDANÇA: Sanitizando os títulos >>>
        # =====================================================================
        titulo_declaracao = sanitizar_html(f"Declaração ({label}): {row['statement'][:100]}...")
        titulo_orador = sanitizar_html(f"Orador: {speaker}")
        titulo_partido = sanitizar_html(f"Partido: {party}")

        G.add_node(
            statement_id, type='declaracao', label=label,
            title=titulo_declaracao # Usamos o título limpo
        )
        
        if speaker != 'nan' and speaker != 'None' and speaker:
            G.add_node(speaker, type='orador', title=titulo_orador)
            G.add_edge(speaker, statement_id)
            
            if party != 'nan' and party != 'None' and party:
                G.add_node(party, type='partido', title=titulo_partido)
                G.add_edge(speaker, party)
        
        if subject != 'nan' and subject != 'None' and subject:
            for subj in subject.split(','):
                subj_limpo = subj.strip()
                if subj_limpo:
                    titulo_assunto = sanitizar_html(f"Assunto: {subj_limpo}")
                    G.add_node(subj_limpo, type='assunto', title=titulo_assunto)
                    G.add_edge(statement_id, subj_limpo)
            
    print("--- Grafo LIAR (Amostrado) criado com sucesso! ---")
    print(f"Total de Nós: {G.number_of_nodes()}")
    print(f"Total de Arestas: {G.number_of_edges()}")
    
    return G

def visualizar_grafo_pyvis_liar(G, nome_arquivo="liar_grafo_amostra.html"):
    print(f"Iniciando visualização com Pyvis...")
    
    output_path = os.path.join(SCRIPT_DIR, nome_arquivo)
    print(f"Salvando em: {output_path}")

    net = Network(height="800px", width="100%", directed=True, 
                  notebook=False, cdn_resources='in_line')
    
    cores_tipo = {
        'declaracao': '#FFFFFF', 'orador': '#87CEEB',
        'partido': '#D3D3D3', 'assunto': '#90EE90'
    }
    tamanhos_tipo = {'declaracao': 25, 'orador': 15, 'partido': 10, 'assunto': 10}
    
    for node, attrs in G.nodes(data=True):
        if 'type' not in attrs: continue
        tipo = attrs['type']
        cor = cores_tipo[tipo]
        tamanho = tamanhos_tipo[tipo]
        
        # =====================================================================
        # <<< MUDANÇA: Usando o 'title' já sanitizado >>>
        # =====================================================================
        titulo_sanitizado = attrs.get('title', node) # Já vem limpo da função de criação
        
        if tipo == 'declaracao':
            cor = CORES_LABEL.get(attrs.get('label'), '#FFFFFF')
        
        net.add_node(str(node), label=str(node), color=cor, size=tamanho, title=titulo_sanitizado)
        
    print("Adicionando arestas ao Pyvis...")
    for u, v in G.edges():
        net.add_edge(str(u), str(v))
    
    net.show_buttons(filter_=['physics'])
    
    # Usando o método que comprovadamente resolve o erro de encoding:
    try:
        # Geramos o HTML na memória (agora com títulos seguros)
        html_content = net.generate_html(notebook=False)
        
        # Salvamos manualmente com UTF-8
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Visualização salva! Abra o arquivo '{output_path}' no seu navegador.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo HTML: {e}")

# =============================================================================
# PONTO DE ENTRADA DO SCRIPT
# =============================================================================
if __name__ == "__main__":
    
    N_AMOSTRA = 100 
    
    verificar_arquivos_locais()
    
    G_liar = criar_grafo_liar(amostra_n=N_AMOSTRA)
    
    if G_liar.number_of_nodes() > 0:
        visualizar_grafo_pyvis_liar(G_liar, nome_arquivo="liar_grafo_amostra.html")
    else:
        print("Grafo não foi criado (ou estava vazio). Encerrando.")