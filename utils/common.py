import tiktoken
from openai import OpenAI
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-2D0EZSwcWUcD4c2K59353b7214854bBd8f35Ac131564EfBa",
    base_url="https://free.gpt.ge/v1"
)

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
    index = faiss.IndexFlatL2(384)  # 384是向量维度
    index.add(vectors)
    return chunks, index

# 保存索引和chunks
def save_index(file_name, chunks, index):
    if not os.path.exists('indices'):
        os.makedirs('indices')
    with open(f'indices/{file_name}.pkl', 'wb') as f:
        pickle.dump((chunks, index), f)
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
    else:
        file_list = []
    if file_name not in file_list:
        file_list.append(file_name)
        with open(file_list_path, 'w') as f:
            f.write('\n'.join(file_list))

# 加载所有保存的索引
def load_all_indices():
    file_indices = {}
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        for file_name in file_list:
            file_path = f'indices/{file_name}.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    chunks, index = pickle.load(f)
                file_indices[file_name] = (chunks, index)
    return file_indices

# 删除索引
def delete_index(file_name):
    if os.path.exists(f'indices/{file_name}.pkl'):
        os.remove(f'indices/{file_name}.pkl')
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        if file_name in file_list:
            file_list.remove(file_name)
            with open(file_list_path, 'w') as f:
                f.write('\n'.join(file_list)) 