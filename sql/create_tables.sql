-- 文档块表
CREATE TABLE document_chunks (
    chunk_id NUMBER GENERATED ALWAYS AS IDENTITY,
    doc_name VARCHAR2(200),
    chunk_content CLOB,
    chunk_vector VECTOR,
    metadata CLOB,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    PRIMARY KEY (chunk_id)
);

-- 创建向量索引
CREATE VECTOR INDEX doc_chunks_idx ON document_chunks(chunk_vector)
    PARAMETERS ('DIMENSION=384 DISTANCE_METRIC=COSINE'); 