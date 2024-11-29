import tiktoken
from openai import OpenAI
import os
import numpy as np
import PyPDF2
import docx
from .oracle_vector_store import OracleVectorStore
from sentence_transformers import SentenceTransformer

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-2D0EZSwcWUcD4c2K59353b7214854bBd8f35Ac131564EfBa",
    base_url="https://free.gpt.ge/v1"
)

# 初始化Oracle向量存储
vector_store = OracleVectorStore()

# 计算token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(string))

# 加载向量模型
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 文档向量化
def vectorize_document(file, max_tokens):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = file.getvalue().decode("utf-8")
    
    chunks = []
    current_chunk = ""
    for sentence in text.split('.'):
        if num_tokens_from_string(current_chunk + sentence) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)
    
    model = load_model()
    vectors = model.encode(chunks)
    
    # 将向量和文档添加到Oracle
    documents = [{
        'file_path': file.name,
        'content': chunk,
        'metadata': {'file_type': file.type}
    } for chunk in chunks]
    vector_store.add_vectors(vectors, documents)
    
    return chunks, vectors

# 保存索引和chunks
def save_index(file_name, chunks, vectors):
    # 不再需要本地保存,因为已经保存到Oracle中
    pass

# 加载所有保存的索引
def load_all_indices():
    # 不再需要本地加载,因为数据在Oracle中
    pass

# 删除索引
def delete_index(file_name):
    # TODO: 实现从Oracle中删除文档的功能
    pass

# 搜索相似文档
def search_similar_documents(query_text, top_k=5):
    model = load_model()
    query_vector = model.encode([query_text])[0]
    return vector_store.search_vectors(query_vector, top_k)
    