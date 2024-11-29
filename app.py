"""
AI知识问答系统 - Oracle Vector Store版本
"""

import streamlit as st
from utils.oracle_vector_store import OracleVectorStore
from sentence_transformers import SentenceTransformer
import openai
import PyPDF2
import docx
from io import BytesIO
import os
from dotenv import load_dotenv
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置oracle_vector_store的日志级别
logging.getLogger('utils.oracle_vector_store').setLevel(logging.INFO)

# 加载环境变量
load_dotenv()

# 初始化OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# 初始化向量模型
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

# 设置页面配置
st.set_page_config(
    page_title="AI知识问答系统 - by Huaiyuan Tan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 页面样式设置
st.markdown("<h6 style='text-align: right; color: gray;'>开发者: Huaiyuan Tan</h6>", unsafe_allow_html=True)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def read_file_content(uploaded_file) -> str:
    """读取上传文件的内容"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    content = ""
    
    try:
        if file_type == 'txt':
            content = uploaded_file.getvalue().decode('utf-8')
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        elif file_type == 'docx':
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            for para in doc.paragraphs:
                content += para.text + "\n"
        return content
    except Exception as e:
        st.error(f"文件读取错误: {str(e)}")
        return ""

def process_document(file_name: str, content: str) -> tuple:
    """处理文档内容，返回文档块和向量"""
    # 不再分割文档，直接使用整个内容
    chunks = [content]  # 改为单个块
    
    # 生成向量嵌入
    vectors = embeddings_model.encode(chunks)
    
    # 准备文档数据
    documents = [{
        'file_path': file_name,
        'content': content,
        'metadata': {'chunk_id': 0}  # 只有一个块，id为0
    }]
    
    return vectors, documents

def get_existing_documents():
    """获取已存在的文档列表"""
    with OracleVectorStore() as vector_store:
        try:
            documents = vector_store.list_documents()
            logger.info(f"获取到已存在文档: {len(documents)}个")
            return documents
        except Exception as e:
            logger.error(f"获取文档列表错误: {str(e)}")
            return []

def handle_file_upload():
    """处理文件上传"""
    # 显示已有文档
    st.subheader("已有文档:")
    existing_docs = get_existing_documents()
    if existing_docs:
        for doc in existing_docs:
            st.text(f"📄 {doc['file_path']} (分块数: {doc['chunk_count']})")
            logger.info(f"显示文档: {doc}")
    else:
        st.text("暂无文档")
        logger.warning("没有找到任何文档")
    
    st.subheader("上传新文档:")
    uploaded_file = st.file_uploader("选择文件", type=['txt', 'pdf', 'docx'], key="file_uploader")
    
    # 使用session_state来跟踪上传状态
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
        
    if uploaded_file and not st.session_state.file_processed:
        logger.info(f"收到文件上传: {uploaded_file.name}")
        with st.spinner("正在处理文档..."):
            content = read_file_content(uploaded_file)
            if content:
                vectors, documents = process_document(uploaded_file.name, content)
                
                # 存储到Oracle
                with OracleVectorStore() as vector_store:
                    try:
                        vector_store.init_schema()
                        # 先删除同名文件的所有记录
                        deleted = vector_store.delete_document(uploaded_file.name)
                        if deleted:
                            logger.info(f"删除已存在的文档: {uploaded_file.name}")
                        # 添加新的文档
                        vector_store.add_vectors(vectors, documents)
                        logger.info(f"成功添加文档: {uploaded_file.name}")
                        st.success(f"文档 {uploaded_file.name} 处理完成!")
                        st.session_state.file_processed = True
                        st.rerun()
                    except Exception as e:
                        logger.error(f"文档处理错误: {str(e)}", exc_info=True)
                        st.error(f"文档处理错误: {str(e)}")
    
    # 如果文件已处理，重置状态
    if not uploaded_file and st.session_state.file_processed:
        st.session_state.file_processed = False

def search_similar_documents(query: str, top_k: int = 3, preview_only: bool = False) -> list:
    """搜索相似文档"""
    logger.info(f"搜索文档，问题：{query}")
    
    # 生成查询向量
    query_vector = embeddings_model.encode([query])[0]
    
    # 在Oracle中搜索
    with OracleVectorStore() as vector_store:
        try:
            results = vector_store.search_vectors(
                query_vector=query_vector,
                top_k=top_k,
                preview_only=preview_only,
                similarity_threshold=0.99  # 放宽阈值，允许更多的匹配结果
            )
            return results
        except Exception as e:
            logger.error(f"搜索错误: {str(e)}")
            st.error(f"搜索错误: {str(e)}")
            return []

def generate_answer(query: str, context: str) -> str:
    """生成AI回答"""
    try:
        # 使用指定的API key和base_url
        client = openai.OpenAI(
            api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的助手。请基于提供的上下文信息回答问题。如果无法从上下文中找到答案，请明确说明。"},
                {"role": "user", "content": f"上下文信息:\n{context}\n\n问题: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"生成回答时发生错误: {str(e)}")
        return "抱歉，生成回答时出现错误。"

def main():
    st.title("电子病历问答系统")
    
    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    
    # 文件上传部分
    handle_file_upload()
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题:"):
        # 添加用户问题到历史记录
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("正在搜索相关内容..."):
                # 搜索相关文档
                results = search_similar_documents(prompt, top_k=1, preview_only=False)
                
                if not results:
                    error_message = "未找到任何相关文档。请确认：\n1. 文档是否已上传\n2. 患者姓名是否正确\n3. 问题是否准确"
                    st.warning(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    return
                
                # 获取最相关的文档
                result = results[0]
                similarity_score = 1 - result['similarity']
                
                # 显示匹配的文档信息
                st.info(f"匹配文档: {result['file_path']} (相似度: {similarity_score:.2%})")
                
                if similarity_score < 0.5:
                    st.warning("⚠️ 注意：当前文档的相似度较低，回答可能不够准确。")
                
                # 生成AI回答
                with st.spinner("正在生成回答..."):
                    answer = generate_answer(prompt, result['content'])
                    st.markdown(answer)
                    
                    # 保存助手回答到历史记录
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

if __name__ == "__main__":
    main()

