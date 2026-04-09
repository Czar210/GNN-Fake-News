-- =============================================================================
-- Migration: 20260331000002_match_documents_function.sql
-- Descrição:  Cria a função RPC match_documents para busca semântica.
--             Chamada pelo FastAPI via supabase-py: supabase.rpc("match_documents")
-- Operador:   <=> (distância de cosseno do pgvector)
-- =============================================================================


CREATE OR REPLACE FUNCTION match_documents(
    -- Vetor da query já embedado pelo multilingual-e5-large (1024 dims)
    query_embedding     vector(1024),

    -- Limiar mínimo de similaridade: 0.0 = aceita tudo, 1.0 = apenas idênticos
    -- Valor recomendado para guias de LoL: 0.5 (filtra ruído sem ser restritivo demais)
    match_threshold     float DEFAULT 0.5,

    -- Número máximo de resultados retornados
    match_count         int   DEFAULT 5
)
RETURNS TABLE (
    id          UUID,
    champion    TEXT,
    title       TEXT,
    author      TEXT,
    content     TEXT,
    similarity  float
)
LANGUAGE sql
STABLE  -- declara que a função não modifica o banco (permite otimizações pelo planner)
AS $$
    SELECT
        guides_gold.id,
        guides_gold.champion,
        guides_gold.title,
        guides_gold.author,
        guides_gold.content,

        -- Converte distância de cosseno em similaridade:
        -- <=> retorna 0 (idêntico) até 2 (oposto); 1 - distância = similaridade
        1 - (guides_gold.embedding <=> query_embedding) AS similarity

    FROM guides_gold

    -- Aplica o limiar: descarta chunks pouco relevantes antes de ordenar
    WHERE 1 - (guides_gold.embedding <=> query_embedding) >= match_threshold

    -- Os mais similares primeiro
    ORDER BY guides_gold.embedding <=> query_embedding

    -- Limita o número de resultados
    LIMIT match_count;
$$;


-- -----------------------------------------------------------------------------
-- Documentação da função
-- -----------------------------------------------------------------------------
COMMENT ON FUNCTION match_documents(vector, float, int) IS
'Busca semântica na Camada Ouro (guides_gold).
Parâmetros:
  query_embedding  – vetor 1024-d gerado pelo multilingual-e5-large para a pergunta do usuário
  match_threshold  – similaridade mínima (padrão 0.5); aumente para respostas mais precisas
  match_count      – nº máximo de chunks retornados (padrão 5)
Retorna: id, champion, title, author, content, similarity (0.0 a 1.0)
Uso via FastAPI: supabase.rpc("match_documents", {"query_embedding": [...], "match_threshold": 0.5, "match_count": 5})';
