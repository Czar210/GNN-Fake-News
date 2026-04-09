-- =============================================================================
-- Migration: 20260331000001_enable_pgvector_and_guides_gold.sql
-- Descrição:  Habilita a extensão pgvector e cria a Tabela Ouro (guides_gold)
--             para armazenar os guias de campeões processados pelo Metis.
-- Modelo de embedding: intfloat/multilingual-e5-large → 1024 dimensões
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Habilitar a extensão pgvector (transforma o Postgres em banco vetorial)
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;


-- -----------------------------------------------------------------------------
-- 2. Criar a Tabela Ouro: guides_gold
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS guides_gold (
    -- Identificador único gerado automaticamente
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Campeão ao qual o guia se refere (ex: "Jinx", "Thresh")
    champion    TEXT        NOT NULL,

    -- Título do guia (ex: "Guia Completo de Jinx - Patch 14.8")
    title       TEXT        NOT NULL,

    -- Autor do guia (nome de usuário ou fonte)
    author      TEXT,

    -- Conteúdo limpo do guia (texto após pré-processamento / chunking)
    content     TEXT        NOT NULL,

    -- Vetor de embedding gerado pelo multilingual-e5-large (1024 dimensões)
    embedding   vector(1024),

    -- Auditoria
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- -----------------------------------------------------------------------------
-- 3. Índice HNSW para busca semântica eficiente por similaridade de cosseno
--    (muito mais rápido que o índice IVFFlat para tabelas que crescem gradualmente)
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS guides_gold_embedding_hnsw_idx
    ON guides_gold
    USING hnsw (embedding vector_cosine_ops);


-- -----------------------------------------------------------------------------
-- 4. Comentários descritivos nas colunas (boa prática de documentação)
-- -----------------------------------------------------------------------------
COMMENT ON TABLE  guides_gold                IS 'Camada Ouro: guias de campeões processados e indexados para busca semântica pelo Metis.';
COMMENT ON COLUMN guides_gold.champion       IS 'Nome do campeão de LoL ao qual o guia pertence.';
COMMENT ON COLUMN guides_gold.title          IS 'Título original do guia.';
COMMENT ON COLUMN guides_gold.author         IS 'Autor ou fonte do guia (ex: OP.GG, u.gg, autor do guia).';
COMMENT ON COLUMN guides_gold.content        IS 'Texto limpo do guia após chunking e pré-processamento.';
COMMENT ON COLUMN guides_gold.embedding      IS 'Vetor de 1024 dimensões gerado pelo multilingual-e5-large.';
