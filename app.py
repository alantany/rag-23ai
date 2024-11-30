"""
AI知识问答系统 - Oracle Vector Store版本
"""

import streamlit as st
from utils.oracle_vector_store import OracleVectorStore
from utils.oracle_json_store import OracleJsonStore
from utils.medical_record_parser import MedicalRecordParser
from sentence_transformers import SentenceTransformer
import os
import shutil
from pathlib import Path
import logging
import openai
import pdfplumber
import json
from typing import Dict, Any, List

# 配置日志和初始化模型
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

# 设置文档存储目录
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    """保存上传的文件到本地目录"""
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return str(file_path)

def read_file_content(file_path):
    """读取文件内容，使用 pdfplumber 处理 PDF"""
    try:
        # 获取文件扩展名
        file_extension = str(file_path).lower().split('.')[-1]
        
        # PDF文件处理
        if file_extension == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        
        # 文本文件处理
        else:
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"文件读取失败: {str(e)}")
        return None

def get_uploaded_files():
    """获取已上传的文件列表"""
    return list(UPLOAD_DIR.glob("*.*"))

def vectorize_document(file_path):
    """向量化文档"""
    try:
        content = read_file_content(file_path)
        vector = embeddings_model.encode([content])[0]
        documents = [{"file_path": str(file_path), "content": content}]
        
        with OracleVectorStore() as vector_store:
            vector_store.add_vectors([vector], documents)
        return True
    except Exception as e:
        logger.error(f"向量化失败: {str(e)}")
        return False

def parse_document_to_json(file_path):
    """解析文档为结构化JSON"""
    try:
        # 读取文件内容
        content = read_file_content(file_path)
        if not content:
            logger.error(f"无法读取文件内容: {file_path}")
            return None
            
        # 使用解析器处理
        parser = MedicalRecordParser()
        structured_data = parser.parse_medical_record(content)
        
        # 如果返回的是错误信息，返回 None
        if 'error' in structured_data:
            logger.error(f"解析失败: {structured_data['error']}")
            return None
            
        # 保存到 JSON 存储
        with OracleJsonStore() as json_store:
            json_store.add_document(
                file_path=str(file_path),
                content=content,
                structured_data=structured_data
            )
            
        return structured_data
        
    except Exception as e:
        logger.error(f"JSON解析失败: {str(e)}")
        return None

def search_similar_documents(query: str, top_k: int = 3):
    """向量搜索并使用 GPT 分析最相关的文档"""
    try:
        # 1. 向量搜索
        vector = embeddings_model.encode([query])[0]
        with OracleVectorStore() as vector_store:
            results = vector_store.search_vectors([vector], top_k=top_k)
        
        if not results:
            return []

        # 2. 使用 GPT 分析最相关的文档
        best_match = results[0]  # 取相似度最高的文档
        
        prompt = f"""
        基于以下医疗文档内容，回答问题：{query}

        文档内容：
        {best_match['content']}

        请提供详细的专业分析和答案。如果文档内容与问题无关，请明确指出。
        """

        # 使用新版 OpenAI API
        client = openai.OpenAI(
            api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的医疗助手，擅长分析医疗文档并提供准确的答案。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # 3. 将 GPT 分析结果添加到最相关的文档中
        best_match['gpt_analysis'] = response.choices[0].message.content
        
        # 返回所有检索结果，但只有最相关的文档包含 GPT 分析
        return results

    except Exception as e:
        logger.error(f"搜索文档时发生错误: {str(e)}")
        return []

def analyze_query_with_gpt(query: str) -> Dict[str, Any]:
    """使用 GPT 分析问题并提取搜索关键词"""
    try:
        client = openai.OpenAI(
            api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        prompt = f"""
        请分析以下医疗相关的问题，提取关键信息并构造数据库查询条件。
        
        问题：{query}
        
        请返回一个JSON格式的结果，包含以下字段：
        1. keywords: 搜索关键词列表
        2. fields: 应该在哪些字段中搜索（例如：basic_information, chief_complaint, diagnosis等）
        3. search_type: 搜索类型（exact精确匹配还是fuzzy模糊匹配）
        
        示例返回格式：
        {{
            "keywords": ["发烧", "咳嗽"],
            "fields": ["chief_complaint", "present_illness_history"],
            "search_type": "fuzzy"
        }}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的医疗信息检索专家，帮助构造精确的搜索条件。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        logger.error(f"GPT分析查询失败: {str(e)}")
        return None

def search_json_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """基于GPT分析结果搜索文档"""
    try:
        # 1. 使用GPT分析查询
        analysis = analyze_query_with_gpt(query)
        if not analysis:
            return []
            
        logger.info(f"GPT分析结果: {json.dumps(analysis, ensure_ascii=False)}")
        
        # 2. 构造Oracle JSON查询
        with OracleJsonStore() as json_store:
            search_conditions = []
            
            # 为每个字段构造查询条件
            for field in analysis['fields']:
                for keyword in analysis['keywords']:
                    if analysis['search_type'] == 'exact':
                        condition = f"JSON_EXISTS(doc_content, '$.structured_data.{field}?(@ == \"{keyword}\")')"
                    else:
                        condition = f"JSON_EXISTS(doc_content, '$.structured_data.{field}?(@ like_regex \"{keyword}\" flag \"i\")')"
                    search_conditions.append(condition)
            
            # 组合查询条件
            where_clause = " OR ".join(search_conditions)
            
            search_sql = f"""
                SELECT d.file_path,
                       d.doc_content,
                       1 as relevance
                FROM document_json d
                WHERE {where_clause}
                FETCH FIRST :1 ROWS ONLY
            """
            
            logger.info(f"执行SQL查询: {search_sql}")
            logger.info(f"查询参数: top_k={top_k}")
            
            # 先获取所有文档，看看数据库中有什么
            check_sql = "SELECT file_path, doc_content FROM document_json"
            all_docs = json_store.execute_search(check_sql, [])
            logger.info(f"数据库中的文档数量: {len(all_docs)}")
            for doc in all_docs:
                logger.info(f"文档路径: {doc['file_path']}")
                logger.info(f"文档内容: {json.dumps(doc['content'], ensure_ascii=False)}")
            
            # 执行实际查询
            results = json_store.execute_search(search_sql, [top_k])
            logger.info(f"查询结果数量: {len(results)}")
            return results
            
    except Exception as e:
        logger.error(f"JSON文档搜索失败: {str(e)}")
        return []

def main():
    st.title("医疗病历处理系统")
    
    # 侧边栏：功能选择
    with st.sidebar:
        st.header("功能选择")
        mode = st.radio(
            "选择功能",
            ["文档管理", "向量检索", "结构化检索"]
        )
    
    if mode == "文档管理":
        st.header("文档管理")
        
        # 批量文件上传部分
        uploaded_files = st.file_uploader(
            "上传病历文档（可多选）", 
            type=["txt", "docx", "pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"已选择 {len(uploaded_files)} 个文件")
            with col2:
                if st.button("保存所有文档"):
                    for uploaded_file in uploaded_files:
                        with st.spinner(f"正在保存 {uploaded_file.name}..."):
                            file_path = save_uploaded_file(uploaded_file)
                            st.success(f"已保存: {uploaded_file.name}")
        
        # 显示已上传的文件列表
        st.subheader("已上传的文件")
        files = get_uploaded_files()
        if not files:
            st.info("暂无上传的文件")
        else:
            # 添加批量操作按钮
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("批量向量化"):
                    for file in files:
                        with st.spinner(f"正在向量化 {file.name}..."):
                            if vectorize_document(file):
                                st.success(f"{file.name} 向量化完成")
                            else:
                                st.error(f"{file.name} 向量化失败")
            with col2:
                if st.button("批量结构化"):
                    for file in files:
                        with st.spinner(f"正在结构化 {file.name}..."):
                            structured_data = parse_document_to_json(file)
                            if structured_data:
                                st.success(f"{file.name} 结构化完成")
                            else:
                                st.error(f"{file.name} 结构化失败")
            
            # 显示文件列表
            for file in files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(file.name)
                with col2:
                    if st.button("向量化", key=f"vec_{file.name}"):
                        with st.spinner("正在向量化..."):
                            if vectorize_document(file):
                                st.success("向量化完成")
                            else:
                                st.error("向量化失败")
                with col3:
                    if st.button("结构化", key=f"json_{file.name}"):
                        with st.spinner("正在解析..."):
                            structured_data = parse_document_to_json(file)
                            if structured_data:
                                st.success("结构化完成")
                                with st.expander("查看结构化数据"):
                                    st.json(structured_data)
                            else:
                                st.error("结构化失败")
    
    elif mode == "向量检索":
        st.header("向量检索")
        
        # 显示已向量化的文档列表
        with st.expander("查看已向量化的文档", expanded=True):
            with OracleVectorStore() as vector_store:
                documents = vector_store.list_documents()
                if documents:
                    st.write("已向量化的文档：")
                    for doc in documents:
                        st.write(f"- {doc['file_path']} (向量数: {doc['chunk_count']})")
                else:
                    st.info("暂无向量化文档")
        
        # 搜索功能
        query = st.text_input("请输入搜索内容")
        if query:
            with st.spinner("正在搜索并分析..."):
                results = search_similar_documents(query, top_k=3)
                
                if results:
                    # 显示最相关文档的 GPT 分析结果
                    st.subheader("GPT 分析结果")
                    st.write(results[0]['gpt_analysis'])
                    
                    # 显示所有相关文档
                    st.subheader("相关文档")
                    for i, result in enumerate(results, 1):
                        similarity = 1 - result['similarity']
                        with st.expander(f"文档 {i}: {result['file_path']} (相似度: {similarity:.2%})"):
                            st.write(result['content'])
                else:
                    st.warning("未找到相关文档")
    
    else:  # 结构化检索
        st.header("结构化检索")
        query = st.text_input("请输入查询内容")
        
        if query:
            with st.spinner("正在分析查询并搜索..."):
                # 显示数据库中的所有文档
                with OracleJsonStore() as json_store:
                    check_sql = "SELECT file_path, doc_content FROM document_json"
                    all_docs = json_store.execute_search(check_sql, [])
                    st.write(f"数据库中共有 {len(all_docs)} 个文档")
                    
                    if all_docs:
                        with st.expander("查看所有文档"):
                            for doc in all_docs:
                                st.write(f"文档: {doc['file_path']}")
                                st.json(doc['content'])
                
                # 执行搜索
                results = search_json_documents(query, top_k=3)
                
                if results:
                    st.subheader("搜索结果")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"文档 {i}: {result['file_path']}"):
                            st.json(result['content'])
                else:
                    st.warning("未找到相关文档")

if __name__ == "__main__":
    main()

