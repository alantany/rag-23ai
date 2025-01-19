"""
AI知识问答系统 - Oracle Vector Store版本
"""

import streamlit as st
from utils.oracle_vector_store import OracleVectorStore
from utils.oracle_json_store import OracleJsonStore
from utils.oracle_graph_store import OracleGraphStore
from utils.oracle_property_graph import OraclePropertyGraph
from utils.medical_record_parser import MedicalRecordParser
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import shutil
from pathlib import Path
import logging
from openai import OpenAI
import pdfplumber
import json
import datetime
import hashlib
from typing import Dict, Any, List
from decimal import Decimal
from utils.medical_graph_parser import MedicalGraphParser
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import networkx as nx
from collections import defaultdict
import pandas as pd
import plotly.express as px
from utils.json_cache import JsonCache
import re

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# 设置文档存录
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

# 设置JSON缓存目录
CACHE_DIR = Path("json_cache")
CACHE_DIR.mkdir(exist_ok=True)

# 使用 st.cache_resource 缓存模型，并隐藏
@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    """加载向量化模型（使用本地缓存）"""
    logger.debug("加载向量化模型")
    return SentenceTransformer('all-MiniLM-L6-v2')

# 初始化模型
embeddings_model = load_embeddings_model()

def get_cache_path(file_path: str) -> Path:
    """根据文件路径生成缓存文件路径"""
    # 获取原始文件名（不含扩展名）
    original_name = Path(file_path).stem
    return CACHE_DIR / f"{original_name}.json"

def save_to_cache(file_name: str, data: dict):
    """保存数据到JSON缓存"""
    try:
        json_cache = JsonCache()
        cache_path = json_cache.get_cache_path(file_name)
        return json_cache.save_json(cache_path, data)
    except Exception as e:
        logger.error(f"保存到缓存失败: {str(e)}")
        return False

def load_from_cache(file_path: str) -> Dict:
    """从缓存加载结构化数据"""
    cache_path = get_cache_path(file_path)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"从缓存加载: {cache_path}")
        return data
    return None

def save_uploaded_file(uploaded_file):
    """保存上传的文件"""
    try:
        if not uploaded_file:
            logger.error("未提供上传文件")
            raise ValueError("未提供上传文件")
            
        # 获取文件名和扩展名
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()
        
        # 检查文件类型
        allowed_extensions = {'.pdf', '.txt', '.docx'}
        if file_ext not in allowed_extensions:
            logger.error(f"不支持的文件类型: {file_ext}")
            raise ValueError(f"不支持的文件类型: {file_ext}，仅支持 {', '.join(allowed_extensions)}")
            
        # 检查文件大小（限制为 10MB）
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if uploaded_file.size > max_size:
            logger.error(f"文件过大: {uploaded_file.size} bytes")
            raise ValueError(f"文件大小超过限制（最大 10MB）")
            
        # 构建保存路径
        save_path = UPLOAD_DIR / file_name
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        while save_path.exists():
            stem = Path(file_name).stem
            new_name = f"{stem}_{counter}{file_ext}"
            save_path = UPLOAD_DIR / new_name
            counter += 1
            
        # 保存文件
        try:
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"文件保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"保存文件时出错: {str(e)}")
            if save_path.exists():
                try:
                    save_path.unlink()
                    logger.info(f"删除失败的文件: {save_path}")
                except Exception as del_e:
                    logger.warning(f"删除失败的文件时出错: {str(del_e)}")
            raise
            
    except Exception as e:
        logger.error(f"处理上传文件失败: {str(e)}")
        return None

def read_file_content(file_obj):
    """读取文件内容，支持多种文件类型和编码"""
    try:
        # 处理文件路径
        if isinstance(file_obj, (str, Path)):
            file_path = Path(file_obj)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")
                
            file_type = file_path.suffix.lower()
            logger.info(f"处理文件: {file_path}, 类型: {file_type}")
            
            if file_type == '.pdf':
                try:
                    with pdfplumber.open(str(file_path)) as pdf:
                        text_content = []
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                text_content.append(text)
                        content = '\n'.join(text_content)
                        if not content:
                            logger.warning(f"PDF 文件内容为空: {file_path}")
                            raise ValueError(f"PDF 文件内容为空: {file_path}")
                        return content
                except Exception as e:
                    logger.error(f"读取 PDF 文件失败: {str(e)}")
                    raise
            else:
                # 尝试不同的编码读取文本文件
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            if content:
                                logger.info(f"成功使用 {encoding} 编码读取文件")
                                return content
                    except UnicodeDecodeError:
                        logger.debug(f"使用 {encoding} 编码读取失败，尝试下一个编码")
                        continue
                logger.error(f"无法使用已知编码读取文件: {', '.join(encodings)}")
                raise ValueError(f"无法使用已知编码读取文件: {', '.join(encodings)}")
        
        # 处理文件对象
        else:
            if not hasattr(file_obj, 'read'):
                logger.error("无效的文件对象：缺少 read 方法")
                raise ValueError("无效的文件对象：缺少 read 方法")
                
            # 获取文件类型
            file_type = Path(file_obj.name).suffix.lower() if hasattr(file_obj, 'name') else ''
            logger.info(f"处理文件对象，类型: {file_type}")
            
            # 处理 PDF 文件
            if file_type == '.pdf':
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(file_obj.read())
                        temp_path = temp_file.name
                        logger.debug(f"创建临时文件: {temp_path}")
                    
                    try:
                        with pdfplumber.open(temp_path) as pdf:
                            text_content = []
                            for page in pdf.pages:
                                text = page.extract_text()
                                if text:
                                    text_content.append(text)
                            content = '\n'.join(text_content)
                            if not content:
                                logger.warning("PDF 文件内容为空")
                                raise ValueError("PDF 文件内容为空")
                            return content
                    finally:
                        try:
                            os.unlink(temp_path)
                            logger.debug(f"删除临时文件: {temp_path}")
                        except Exception as e:
                            logger.warning(f"删除临时文件失败: {str(e)}")
                except Exception as e:
                    logger.error(f"处理 PDF 文件对象失败: {str(e)}")
                    raise
            
            # 处理其他文件
            else:
                try:
                    content = file_obj.read()
                    if isinstance(content, bytes):
                        # 尝试不同的编码解码
                        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                        for encoding in encodings:
                            try:
                                decoded = content.decode(encoding)
                                logger.info(f"成功使用 {encoding} 编码解码内容")
                                return decoded
                            except UnicodeDecodeError:
                                logger.debug(f"使用 {encoding} 编码解码失败，尝试下一个编码")
                                continue
                        logger.error(f"无法使用已知编码解码文件内容: {', '.join(encodings)}")
                        raise ValueError(f"无法使用已知编码解码文件内容: {', '.join(encodings)}")
                    return content
                except Exception as e:
                    logger.error(f"读取文件对象失败: {str(e)}")
                    raise
                
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        raise

def get_uploaded_files():
    """获取已上传的文件列表"""
    return list(UPLOAD_DIR.glob("*.*"))

def vectorize_document(file_path):
    """向量化文档
    
    Args:
        file_path: 文件路径，可以是字符串或 Path 对象
        
    Returns:
        bool: 向量化是否成功
    """
    try:
        # 读取文件内容
        logger.info(f"开始向量化文档: {file_path}")
        content = read_file_content(file_path)
        
        if not content:
            logger.error("文档内容为空")
            return False
            
        # 使用已加载的模型进行向量化
        try:
            # 将文档内容分割成较小的块
            chunks = split_text_into_chunks(content)
            if not chunks:
                logger.error("文档分块失败")
                return False
                
            # 对每个块进行向量化
            vectors = []
            documents = []
            for i, chunk in enumerate(chunks):
                try:
                    # 向量化文本块
                    vector = embeddings_model.encode([chunk])[0]
                    vectors.append(vector)
                    
                    # 准备文档信息
                    doc_info = {
                        "file_path": str(file_path),
                        "content": chunk,
                        "chunk_index": i
                    }
                    documents.append(doc_info)
                    
                except Exception as e:
                    logger.error(f"处理文本块 {i} 失败: {str(e)}")
                    continue
            
            if not vectors:
                logger.error("没有成功向量化的文本块")
                return False
                
            # 保存向量到数据库
            with OracleVectorStore() as vector_store:
                vector_store.add_vectors(vectors, documents)
                logger.info(f"成功保存 {len(vectors)} 个向量到数据库")
                
            return True
            
        except Exception as e:
            logger.error(f"向量化过程失败: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"向量化文档失败: {str(e)}")
        return False
        
def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    """将文本分割成较小的块
    
    Args:
        text: 要分割的文本
        chunk_size: 每个块的最大字符数
        overlap: 相邻块之间的重叠字符数
        
    Returns:
        list: 文本块列表
    """
    try:
        if not text:
            return []
            
        # 按句子分割文本
        sentences = re.split('[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果单个句子超过块大小，则按字符分割
            if sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 分割长句子
                for i in range(0, sentence_length, chunk_size - overlap):
                    chunk = sentence[i:i + chunk_size]
                    if chunk:
                        chunks.append(chunk)
                continue
            
            # 如果添加当前句子会超过块大小，保存当前块并开始新块
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
        
    except Exception as e:
        logger.error(f"分割文本失败: {str(e)}")
        return []

def init_database():
    """初始化数据库表"""
    try:
        with OracleJsonStore() as json_store:
            # 检查表是否存在
            check_table_sql = """
            SELECT table_name 
            FROM user_tables 
            WHERE table_name = 'DOCUMENT_JSON'
            """
            result = json_store.execute_search(check_table_sql)
            table_exists = len(result) > 0 if result else False
            
            if not table_exists:
                # 创建结构化文档表
                create_json_table_sql = """
                CREATE TABLE DOCUMENT_JSON (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    doc_info VARCHAR2(500),
                    doc_json JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                json_store.execute_sql(create_json_table_sql)
                logger.info("成功创建 DOCUMENT_JSON 表")
            else:
                logger.debug("DOCUMENT_JSON 表已存在，继续使用")

    except Exception as e:
        logger.error(f"初始化数据库失败: {str(e)}")
        st.error(f"初始化数据库失败: {str(e)}")
        raise

class MedicalRecordParser:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

    def parse_medical_record(self, text: str, doc_info: str) -> Dict[str, Any]:
        """解析医疗记录文本，生成结构化JSON数据和SQL插入语句"""
        try:
            prompt = f"""请将以下病历内容结构化为JSON格式，并生成对应的SQL插入语句。

病历内容：
{text}

要求：
1. JSON结构：
{{
    "患者姓名": "某某（保护隐私）",
    "性别": "性别",
    "年龄": "数字",
    "入院日期": "YYYY-MM-DD",
    "出院日期": "YYYY-MM-DD",
    "主诉": "主要症状",
    "现病史": ["症状1", "症状2"],
    "入诊": ["诊断1", "诊断2"],
    "出院诊": ["断1", "诊断2"],
    "生命体征": {{
        "体温": "包含单位",
        "血压": "包含单位"
    }},
    "生化指标": {{
        "指标名": "值和单位，异常标↑↓"
    }},
    "诊疗经过": "治疗过程描述",
    "出院医嘱": ["医嘱1", "医嘱2"]
}}

2. SQL要求：
- 表名：DOCUMENT_JSON
- 字段：(id NUMBER自增, doc_info VARCHAR2(500), doc_json JSON)
- doc_info值：{doc_info}
- 使用JSON_OBJECT或FORMAT JSON语法

请返回：
{{
    "structured_data": JSON格式的病历数据,
    "sql_statement": Oracle插入语句
}}"""

            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "你疗数据结构化专家，擅长解析病历文本并生成规范的JSON和SQL"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )

            # 解析返回的JSON
            result = json.loads(response.choices[0].message.content)
            
            # 验证结果格式
            if not isinstance(result, dict) or 'structured_data' not in result or 'sql_statement' not in result:
                raise ValueError("GPT返回数据格式不正确")

            # 添加元数据
            if 'metadata' not in result['structured_data']:
                result['structured_data']['metadata'] = {}
            result['structured_data']['metadata'].update({
                'import_time': datetime.datetime.now().isoformat(),
                'source_type': 'text',
                'last_updated': datetime.datetime.now().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"解析医疗记录失败: {str(e)}")
            return {"error": f"解析失败: {str(e)}"}

def parse_document_to_json(file_obj):
    """解析医疗文档并保存到数据库"""
    try:
        # 获取文件名（不含扩展名）
        file_name = Path(file_obj.name).stem
        
        # 检查缓存
        json_cache = JsonCache()
        cache_path = json_cache.get_cache_path(file_name)
        
        # 如果缓存存在，直接从缓存加载
        if os.path.exists(cache_path):
            try:
                cached_data = json_cache.load_json(cache_path)
                logger.info(f"从缓存加载文档: {file_name}")
                
                # 保存到数据库
                with OracleJsonStore() as json_store:
                    json_store.add_document(file_obj.name, cached_data)
                    logger.info(f"保存文档到数据库: {file_name}")
                
                return {"success": True, "message": "从缓存加载并保存到数据库成功"}
            except Exception as e:
                logger.error(f"从缓存加载失败: {str(e)}")
                # 继续尝试重新解析文档
        
        # 读取文档内容
        content = read_file_content(file_obj)
        if not content:
            raise ValueError("无法读取文档内容")
            
        # 使用 GPT 解析文档
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )
        
        # 构建提示词
        prompt = f"""
        请解析以下医疗文档，提取关键信息并以JSON格式返回：

        {content}

        请提取以下信息：
        1. 患者基本信息（姓名、性别、年龄等）
        2. 主诉
        3. 现病史
        4. 入院诊断和出院诊断
        5. 生命体征
        6. 生化指标
        7. 诊疗经过
        8. 出院医嘱

        返回格式示例：
        {{
            "患者姓名": "xxx",
            "性别": "男/女",
            "年龄": "xx岁",
            "主诉": "xxxx",
            "现病史": ["症状1", "症状2"],
            "入院诊断": ["诊断1", "诊断2"],
            "出院诊断": ["诊断1", "诊断2"],
            "生命体征": {{
                "体温": "xx℃",
                "血压": "xx/xxmmHg"
            }},
            "生化指标": {{
                "白细胞": "xx×10^9/L",
                "血红蛋白": "xx g/L"
            }},
            "诊疗经过": "xxxx",
            "出院医嘱": ["医嘱1", "医嘱2"]
        }}

        注意：
        1. 保持数据结构的一致性
        2. 对于缺失的信息使用空值或空列表
        3. 确保返回的是有效的JSON格式
        """
        
        # 调用 GPT 解析文档
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的医疗文档解析助手，擅长从医疗文档中提取结构化信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 使用较低的温度以获得更稳定的输出
        )
        
        # 解析 GPT 返回的结果
        result = response.choices[0].message.content
        if not result:
            raise ValueError("GPT 未返回解析结果")
            
        try:
            parsed_data = json.loads(result)
            if not isinstance(parsed_data, dict):
                raise ValueError("解析结果不是有效的 JSON 对象")
                
            # 保存到缓存
            if json_cache.save_json(cache_path, parsed_data):
                logger.info(f"保存到缓存成功: {cache_path}")
            
            # 保存到数据库
            with OracleJsonStore() as json_store:
                json_store.add_document(file_obj.name, parsed_data)
                logger.info(f"保存文档到数据库: {file_name}")
            
            return {"success": True, "message": "文档解析并保存成功"}
            
        except json.JSONDecodeError as e:
            logger.error(f"解析 GPT 返回的 JSON 失败: {str(e)}")
            raise ValueError("解析 GPT 返回的 JSON 失败")
            
    except Exception as e:
        logger.error(f"解析文档失败: {str(e)}")
        return {"error": f"解析文档失败: {str(e)}"}

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

        请提供详细专业分析和答案。如果文档内容与问题无关，请明确指出。
        """

        # 使用新 OpenAI API
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的医疗助手，擅长分析医疗文档并供准确的答案。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # 3. 将 GPT 分析结果添加到最相关的文档中
        best_match['gpt_analysis'] = response.choices[0].message.content
        
        # 返回所有检索结果，但只有相关的文包含 GPT 分析
        return results

    except Exception as e:
        logger.error(f"搜索文档时发生错误: {str(e)}")
        return []

def analyze_query_with_gpt(query_text):
    """使用 GPT 分析查询意图并生成查询条件"""
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )
        
        # 构建提示词
        prompt = f"""
        分析以下医疗查询，生成 Oracle JSON 查询条件：{query_text}

        文档结构示例：
        {{
            "患者姓名": "张某某",
            "主诉": "头痛、发热3天",
            "现病史": ["发热", "头痛", "咳嗽"],
            "入院诊断": ["上呼吸道感染", "病毒性感冒"],
            "出院诊断": ["病毒性感冒"],
            "生命体征": {{
                "体温": "38.5℃",
                "血压": "120/80mmHg"
            }},
            "生化指标": {{
                "白细胞": "10.5×10^9/L",
                "血红蛋白": "125g/L"
            }}
        }}

        示例查询和对应的条件：
        1. 查询患者张某某的信息
           {{"conditions": ["JSON_EXISTS(doc_json, '$.患者姓名?(@ == \"张某某\")')"]}}

        2. 查询患者张某某的主诉
           {{"conditions": ["JSON_EXISTS(doc_json, '$.患者姓名?(@ == \"张某某\")') AND JSON_EXISTS(doc_json, '$.主诉')" ]}}

        3. 查询主诉包含头痛的病例
           {{"conditions": ["JSON_EXISTS(doc_json, '$.主诉?(@ like_regex \"头痛\" flag \"i\")')"]}}

        4. 查询体温超过38度的病例
           {{"conditions": ["JSON_EXISTS(doc_json, '$.生命体征.体温?(@ like_regex \"3[8-9]|4[0-9]\" flag \"i\")')"]}}

        请分析查询并生成准确的 Oracle JSON 查询条件。注意：
        1. 确保路径与文档结构匹配
        2. 使用正确的 JSON_EXISTS 语法，@ 和 == 之间需要有空格
        3. 对于文本匹配，使用 like_regex 并添加 flag "i" 实现不区分大小写
        4. 对于数值比较，使用适当的比较运算符
        5. 查询特定患者的信息时，需要先匹配患者姓名
        
        请直接返回 JSON 格式的结果，不要添加任何额外的文本。
        """
        
        # 调用 GPT 分析查询
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的医疗数据分析助手，擅长将自然语言查询转换为数据库查询条件。请直接返回 JSON 格式的结果，不要添加任何额外的文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 使用较低的温度以获得更稳定的输出
            response_format={"type": "json_object"}  # 指定返回 JSON 格式
        )
        
        # 解析 GPT 返回的结果
        result = response.choices[0].message.content
        if not result:
            logger.warning("GPT 未返回有效的分析结果")
            return {"conditions": []}
            
        try:
            # 尝试解析 JSON
            analysis_result = json.loads(result)
            if not isinstance(analysis_result, dict) or 'conditions' not in analysis_result:
                logger.warning(f"GPT 返回的结果格式不正确: {result}")
                return {"conditions": []}
                
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"解析 GPT 返回的 JSON 失败: {str(e)}")
            # 返回一个默认的查询条件
            return {"conditions": []}
            
    except Exception as e:
        logger.error(f"分析查询失败: {str(e)}")
        return {"conditions": []}

# 配常量
TOP_K = 5  # 搜索结果返回的最大数量

def normalize_medical_term(query_text):
    """使用 GPT 将用询的标名称标准化"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
        
        messages = [
            {"role": "system", "content": """你是一个医疗指标名称标准化专家
请用中的指标名称为标的疗标称

规
1. 查询中包含某个检验指标的同义词或近义词，返回标准名称
2. 如果不确定，返回原始词语
3. 返回格式为 JSON：{"standard_term": "标准名称"}

示：
输入："淋巴细胞比例"
输出：{"standard_term": "淋巴细胞百分比"}

输入："白细胞计数"
输出：{"standard_term": "胞"}

输入："血红蛋白含量"
输出：{"standard_term": "血红蛋白"}"""},
            {"role": "user", "content": query_text}
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('standard_term', query_text)
        
    except Exception as e:
        logger.error(f"指标名称标准化: {str(e)}")
        return query_text

def search_documents(query_text):
    """基于GPT生成的查询条件搜索文档，支持结构化数据搜索"""
    try:
        # 使用GPT分析查询意图并生成查询条件
        analysis_result = analyze_query_with_gpt(query_text)
        logger.info(f"GPT分析结果: {json.dumps(analysis_result, ensure_ascii=False)}")
        
        if not analysis_result or 'conditions' not in analysis_result:
            logger.error("GPT分析结果格式错误")
            return []
            
        # 使用GPT生成的条件构建查询
        conditions = analysis_result.get('conditions', [])
        json_where = " OR ".join(conditions) if conditions else "1=1"
        
        # 构建完整的查询语句
        query = f"""
        SELECT doc_info, doc_json
        FROM DOCUMENT_JSON d
        WHERE {json_where}
        ORDER BY id DESC
        FETCH FIRST :1 ROWS ONLY
        """
        
        # 执行查询
        with OracleJsonStore() as json_store:
            results = json_store.execute_search(query, [TOP_K])
            if not results:
                logger.info("未找到匹配的文档")
                return []
            
            # 处理结果
            processed_results = []
            for result in results:
                try:
                    if not isinstance(result, dict) or 'doc_info' not in result or 'doc_json' not in result:
                        continue
                    
                    doc_json = result['doc_json']
                    if hasattr(doc_json, 'read'):
                        doc_json = json.loads(doc_json.read())
                    
                    if not isinstance(doc_json, dict):
                        continue
                    
                    processed_results.append({
                        'doc_info': result['doc_info'],
                        'doc_json': doc_json
                    })
                except Exception as e:
                    logger.error(f"处理搜索结果时出错: {str(e)}")
                    continue
            
            return processed_results
            
    except Exception as e:
        logger.error(f"搜索文档失败: {str(e)}")
        return []

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

def generate_answer(query_text, doc_json):
    """根据查询和文档生成答案"""
    try:
        # 使用 GPT 分析查询并生成答案
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )
        
        # 构建提示词
        prompt = f"""
        基于以下结构化医疗数据，回答问题：{query_text}

        文档数据：
        {json.dumps(doc_json, ensure_ascii=False, indent=2, cls=DecimalEncoder)}

        请提供详细专业的分析和答案。如果数据中没有相关信息，请明确指出。
        """
        
        # 调用 GPT 生成答案
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的医疗助手，擅长分析医疗数据并提供准确的答案。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        # 获取生成的答案
        answer = response.choices[0].message.content
        if not answer:
            logger.warning("GPT 未生成有效答案")
            return None
            
        return answer
        
    except Exception as e:
        logger.error(f"生成答案时出错: {str(e)}")
        return None

def display_search_results(query_text, results):
    """显示搜索结果"""
    if not results:
        st.warning("未找到相关文档")
        return
        
    try:
        # 尝试生成精确答案
        answer = generate_answer(query_text, results[0]['doc_json'])
        if answer:
            st.success(answer)
        else:
            st.warning("无法生成答案")
            
        # 显示所有匹配的文档
        st.subheader(f"📄 匹配的文档 ({len(results)} 个)")
        for result in results:
            try:
                doc_info = result['doc_info']
                data = result['doc_json']
                patient_name = data.get("患者姓名", Path(doc_info).stem)
                
                with st.expander(f"📋 {patient_name}", expanded=False):
                    # 创建标签页
                    tabs = st.tabs([
                        "基本信息", "主诉与诊断", "现病史", 
                        "生命体征", "生化指标", "诊疗经过"
                    ])
                    
                    with tabs[0]:
                        if "基本信息" in data:
                            st.json(data["基本信息"])
                        else:
                            st.info("未记录基本信息")
                    
                    with tabs[1]:
                        st.markdown("**主诉**")
                        st.write(data.get("主诉", "未记录"))
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**入院诊断**")
                            for diag in data.get("入院诊断", []):
                                st.write(f"- {diag}")
                        with col2:
                            st.markdown("**出院诊断**")
                            for diag in data.get("出院诊断", []):
                                st.write(f"- {diag}")
                    
                    with tabs[2]:
                        st.markdown("**现病史**")
                        for item in data.get("现病史", []):
                            st.write(f"- {item}")
                    
                    with tabs[3]:
                        if "生命体征" in data:
                            st.json(data["生命体征"])
                        else:
                            st.info("未记录生命体征")
                    
                    with tabs[4]:
                        if "生化指标" in data:
                            st.json(data["生化指标"])
                        else:
                            st.info("未记录生化指标")
                    
                    with tabs[5]:
                        st.markdown("**诊疗经过**")
                        st.write(data.get("诊疗经过", "未记录"))
                        if "出院医嘱" in data:
                            st.markdown("**出院医嘱**")
                            for advice in data["出院医嘱"]:
                                st.write(f"- {advice}")
                                
            except Exception as e:
                logger.error(f"显示文档时出错: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"显示搜索结果时出错: {str(e)}")
        st.error("显示搜索结果时出现错误")

def get_patient_metadata(patient_name: str) -> Dict[str, Any]:
    """获取患者的实际数据结构"""
    try:
        with OracleGraphStore() as graph_store:
            patient_info = graph_store.get_patient_info(patient_name)
            if not patient_info:
                return {}
            return patient_info
    except Exception as e:
        logger.error(f"获取患者元数据失败: {str(e)}")
        return {}

def analyze_graph_query(query_text: str) -> Dict[str, Any]:
    """使用大模型分析图数据查询意图"""
    try:
        # 从查询文本中提取患者姓名
        patient_name = None
        for name in ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"]:
            if name in query_text:
                patient_name = name
                break
        
        if not patient_name:
            logger.warning("未能从查询中识别出患者姓名")
            return {
                "query_type": "基本信息",
                "field": "all",
                "patient_name": None,
                "explanation": "未能识别患者姓名"
            }
            
        # 获取该患者的实际数据结构
        patient_data = get_patient_metadata(patient_name)
        if not patient_data:
            logger.warning(f"未找到患者 {patient_name} 的数据")
            return {
                "query_type": "基本信息",
                "field": "all",
                "patient_name": patient_name,
                "explanation": f"未找到患者 {patient_name} 的数据"
            }
            
        prompt = f"""
        请分析以下医疗查询，提取查询意图和关键信息。直接返回JSON对象，不要添加任何markdown格式或代码块标记。

        查询文本：{query_text}

        患者 {patient_name} 的实际数据结构下：
        {json.dumps(patient_data, ensure_ascii=False, indent=2)}

        你需要分析用户的查询意图，返回一个JSON对象（不要添加任何markdown格式或代码块标记），包含以下字段：
        - query_type: 查询类型，必以下之一：基本信息/主诉与诊断/现病史/生命体征/生化指标/诊疗经过
        - field: 具体查询的字段名，如果是查询整个类别，请返回"all"
        - patient_name: 患者姓名
        - explanation: 查询意图的解释

        如果是查询个人信息，应返回：
        {{
            "query_type": "基本信息",
            "field": "all",
            "patient_name": "{patient_name}",
            "explanation": "查询患者的所有基本信息"
        }}

        如果是查询个类别的所有信息（如"生化指标"、"主诉与诊断"等），请将field设置为"all"。
        如果是查询具体的指标或症状（如"白细胞"、"血压"等），请将field设置为具体的指标名称。

        请分析这个查询并返回JSON（不要添加markdown格式）
        """

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "你是一个专业的医疗查询分析助手。请直接返回JSON对象，不要添加任何markdown格式或代码块标记。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"OpenAI API返回内容: {content}")
        
        # 尝试解析返回内容
        try:
            # 首先尝试直接解析
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("直接解析JSON失败，尝试清理内容后重新解析")
                # 如果失败，尝试清理内容（去除可能的转义字符等）
                cleaned_content = content.replace('\n', '').replace('\r', '').strip()
                result = json.loads(cleaned_content)
            
            # 如果结果被包装在response字段中，提取内层JSON
            if isinstance(result, dict) and "response" in result:
                try:
                    inner_content = result["response"]
                    if isinstance(inner_content, str):
                        # 清理内层JSON字符串
                        inner_content = inner_content.replace('\n', '').replace('\r', '').strip()
                        result = json.loads(inner_content)
                    elif isinstance(inner_content, dict):
                        result = inner_content
                except json.JSONDecodeError as e:
                    logger.error(f"解析response字段失败: {str(e)}")
                    result = {
                        "query_type": "基本信息",
                        "field": "all",
                        "patient_name": patient_name,
                        "explanation": "解析查询意图时出现错误"
                    }
            
            # 验证结果格式
            if not isinstance(result, dict):
                raise ValueError("返回结果不是一个有效的JSON对象")
            
            # 验证必要字段
            required_fields = ["query_type", "field", "patient_name", "explanation"]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logger.warning(f"缺少必要字段: {missing_fields}")
                for field in missing_fields:
                    if field == "query_type":
                        result[field] = "基本信息"
                    elif field == "field":
                        result[field] = "all"
                    elif field == "patient_name":
                        result[field] = patient_name
                    elif field == "explanation":
                        result[field] = "查询患者信息"
            
            # 验证query_type是否为有效值
            valid_query_types = ["基本信息", "主诉与诊断", "现病史", "生命体征", "生化指标", "诊疗经过"]
            if result["query_type"] not in valid_query_types:
                logger.warning(f"无效query_type: {result['query_type']}, 使用默认值")
                result["query_type"] = "基本信息"
            
            # 确保patient_name与查中识别的一致
            if result["patient_name"] != patient_name:
                logger.warning(f"patient_name不匹配: {result['patient_name']} != {patient_name}")
                result["patient_name"] = patient_name
            
            # 处理field值
            if result["field"] == result["query_type"] or result["field"] in valid_query_types:
                logger.info(f"将field改为 {result['field']}")
                result["field"] = "all"
            
            logger.info(f"查询意图分析结果: {json.dumps(result, ensure_ascii=False)}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}, 原始内容: {content}")
            return {
                "query_type": "基本信息",
                "field": "all",
                "patient_name": patient_name,
                "explanation": "解析查询意图时出现错误"
            }
            
    except Exception as e:
        logger.error(f"分析查询意图失败: {str(e)}")
        return {
            "query_type": "基本信息",
            "field": "all",
            "patient_name": patient_name if 'patient_name' in locals() else None,
            "explanation": "分析查询意图时出现错误"
        }

def search_graph_data(query_text: str) -> List[Dict[str, Any]]:
    """基于图数据的搜索"""
    try:
        # 使用GPT分析查询意图
        analysis = analyze_graph_query(query_text)
        query_type = analysis.get("query_type")
        field = analysis.get("field")
        patient_name = analysis.get("patient_name")
        
        if not all([query_type, patient_name]):
            logger.error("查询意图分析结果不完整")
            return []
        
        # 使用图数据库搜索
        with OracleGraphStore() as graph_store:
            # 获取患者信息
            patient_info = graph_store.get_patient_info(patient_name)
            if not patient_info:
                return []
            
            # 根据查询类型返回结果
            if query_type == "基本信息":
                if field == "all":
                    # 返回所有信息
                    info = patient_info.get("患者", {}).get("基本信息", {})
                    if not info:
                        info = patient_info.get("基本信息", {})
                    if info:
                        result = []
                        for k, v in info.items():
                            result.append(f"{k}：{v}")
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的基本信息：\n" + "\n".join(result),
                            "explanation": analysis.get("explanation")
                        }}]
                else:
                    # 查询特定字段
                    value = None
                    # 先尝试从患者.基本信息中获取
                    info = patient_info.get("患者", {}).get("基本信息", {})
                    if not info:
                        # 如果没有，则从基本信息中获取
                        info = patient_info.get("基本信息", {})
                    value = info.get(field)
                    if value:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的{field}是{value}",
                            "explanation": analysis.get("explanation")
                        }}]
            
            elif query_type == "主诉与诊断":
                items = patient_info.get("主诉与诊断", [])
                if field == "all":
                    results = []
                    # 分类处理主诉和诊断
                    chief_complaints = []
                    admission_diagnoses = []
                    discharge_diagnoses = []
                    for item in items:
                        if item.get("类型") == "主诉":
                            chief_complaints.append(item.get("内容"))
                        elif item.get("类型") == "入院诊断":
                            admission_diagnoses.append(item.get("内容"))
                        elif item.get("类型") == "出院诊断":
                            discharge_diagnoses.append(item.get("内容"))
                    
                    # 组织返回内容
                    if chief_complaints:
                        results.append("主诉：")
                        results.extend([f"- {complaint}" for complaint in chief_complaints])
                    if admission_diagnoses:
                        results.append("\n入院诊断：")
                        results.extend([f"- {diagnosis}" for diagnosis in admission_diagnoses])
                    if discharge_diagnoses:
                        results.append("\n出院诊断：")
                        results.extend([f"- {diagnosis}" for diagnosis in discharge_diagnoses])
                        
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的主诉与诊断：\n" + "\n".join(results),
                            "explanation": analysis.get("explanation")
                        }}]
                else:
                    results = []
                    for item in items:
                        if field.lower() in item.get("类型", "").lower():
                            results.append(item.get("内容"))
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的{field}：\n" + "\n".join([f"- {r}" for r in results]),
                            "explanation": analysis.get("explanation")
                        }}]
            
            elif query_type == "现病史":
                items = patient_info.get("现病史", [])
                if field == "all":
                    results = []
                    for item in items:
                        symptom = item.get("症状", "")
                        description = item.get("描述", "")
                        if description:
                            results.append(f"- {symptom}：{description}")
                        else:
                            results.append(f"- {symptom}")
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的现病史：\n" + "\n".join(results),
                            "explanation": analysis.get("explanation")
                        }}]
                else:
                    results = []
                    for item in items:
                        if field.lower() in item.get("症状", "").lower():
                            description = item.get("描述", "")
                            if description:
                                results.append(f"{item.get('症状')}：{description}")
                            else:
                                results.append(item.get('症状'))
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的{field}：\n" + "\n".join(results),
                            "explanation": analysis.get("explanation")
                        }}]
            
            elif query_type == "生命体征":
                items = patient_info.get("生命体征", [])
                if field == "all":
                    results = []
                    for item in items:
                        results.append(f"- {item.get('指标')}：{item.get('数值')}{item.get('单位', '')}")
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的生命体征：\n" + "\n".join(results),
                            "explanation": analysis.get("explanation")
                        }}]
                else:
                    for item in items:
                        if field.lower() in item.get("指标", "").lower():
                            return [{"type": "answer", "data": {
                                "question": query_text,
                                "answer": f"{patient_name}的{field}是：{item.get('数值')}{item.get('单位', '')}",
                                "explanation": analysis.get("explanation")
                            }}]
            
            elif query_type == "生化指标":
                items = patient_info.get("生化指标", [])
                if field == "all":
                    results = []
                    # 按照参考范围分类
                    abnormal_items = []
                    normal_items = []
                    for item in items:
                        result_str = f"- {item.get('项目')}：{item.get('结果')}{item.get('单位', '')}"
                        if item.get('参考范围') == '异常':
                            abnormal_items.append(result_str + " (异常)")
                        else:
                            normal_items.append(result_str + " (正常)")
                    
                    # 组织返回内容
                    if abnormal_items:
                        results.append("异常指标：")
                        results.extend(abnormal_items)
                    if normal_items:
                        if results:  # 如果已经有异常指标，添加空行
                            results.append("")
                        results.append("正常指标：")
                        results.extend(normal_items)
                        
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的生化指标：\n" + "\n".join(results),
                            "explanation": analysis.get("explanation")
                        }}]
                else:
                    for item in items:
                        if field.lower() in item.get("项目", "").lower():
                            return [{"type": "answer", "data": {
                                "question": query_text,
                                "answer": f"{patient_name}的{field}是：{item.get('结果')}{item.get('单位', '')} ({item.get('参考范围', '')})",
                                "explanation": analysis.get("explanation")
                            }}]
            
            elif query_type == "诊疗经过":
                items = patient_info.get("诊疗经过", [])
                if field == "all":
                    results = []
                    # 分类处理诊疗经过和出院医嘱
                    diagnoses = []
                    advices = []
                    for item in items:
                        if item.get("类型") == "诊疗经过":
                            diagnoses.append(item.get("内容"))
                        elif item.get("类型") == "出院医嘱":
                            advices.append(item.get("内容"))
                    
                    # 组织返回内容
                    if diagnoses:
                        results.append("诊疗经过：")
                        results.extend([f"- {diagnosis}" for diagnosis in diagnoses])
                    if advices:
                        if results:  # 如果已经有诊疗经过，添加空行
                            results.append("")
                        results.append("出院医嘱：")
                        results.extend([f"- {advice}" for advice in advices])
                        
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的诊疗经过：\n" + "\n".join(results),
                            "explanation": analysis.get("explanation")
                        }}]
                else:
                    results = []
                    for item in items:
                        if field.lower() in item.get("类型", "").lower():
                            results.append(item.get("内容"))
                    if results:
                        return [{"type": "answer", "data": {
                            "question": query_text,
                            "answer": f"{patient_name}的{field}：\n" + "\n".join([f"- {r}" for r in results]),
                            "explanation": analysis.get("explanation")
                        }}]
            
            return []
            
    except Exception as e:
        logger.error(f"图数据搜索失败: {str(e)}")
        return []

def display_graph_results(results: List[Dict[str, Any]], query_text: str):
    """显示图数据搜索结果"""
    if not results:
        st.warning("未找到相关信息")
        return
        
    for result in results:
        if result["type"] == "answer":
            data = result["data"]
            st.write(data["answer"])

def process_graph_data(file_path: str, content: str) -> bool:
    """处理文档的图数据"""
    try:
        # 只处理图数据
        graph_parser = MedicalGraphParser()
        graph_result = graph_parser.parse_to_graph(content, file_path)
        
        if "error" in graph_result:
            return False
            
        return True
    except Exception as e:
        logger.error(f"处理图数据失败: {str(e)}")
        return False

def visualize_patient_graph(patient_info: Dict[str, Any]) -> str:
    """使用pyvis可视化患者的属性图"""
    try:
        # 创建网络图
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=False)
        
        # 设置物理布局选项
        net.set_options("""
        {
          "nodes": {
            "font": {
              "size": 14,
              "face": "Microsoft YaHei"
            }
          },
          "edges": {
            "color": {
              "color": "#666666",
              "highlight": "#000000"
            }
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.5
            }
          }
        }
        """)
        
        # 添加患者节点（中心节点）
        patient_name = patient_info.get('姓名', '未知患者')
        
        net.add_node(patient_name, 
                    label=patient_name,
                    color='#add8e6',  # lightblue
                    size=30,
                    shape='circle')
        
        # 添加基本信息节点
        basic_info = patient_info.get('基本信息', {})
        if basic_info:
            for key, value in basic_info.items():
                node_id = f'basic_{key}'
                net.add_node(node_id,
                            label=f'{key}：{value}',
                            color='#90EE90',  # lightgreen
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='基本信息')
        
        # 添加主诉与诊断节点
        if '主诉与诊断' in patient_info:
            for i, item in enumerate(patient_info['主诉与诊断']):
                node_id = f'diag_{i}'
                net.add_node(node_id,
                            label=f"{item.get('类型')}：{item.get('内容')}",
                            color='#FFB6C1',  # lightpink
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='主诉与诊断')
        
        # 添加现病史节点
        if '现病史' in patient_info:
            for i, item in enumerate(patient_info['现病史']):
                node_id = f'hist_{i}'
                net.add_node(node_id,
                            label=f"{item.get('症状')}：{item.get('描述')}",
                            color='#FFFFE0',  # lightyellow
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='现病史')

        # 添加生命体征节点
        if '生命体征' in patient_info:
            for i, item in enumerate(patient_info['生命体征']):
                node_id = f'vital_{i}'
                net.add_node(node_id,
                            label=f"{item.get('指标')}：{item.get('数值')}",
                            color='#F08080',  # lightcoral
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='生命体征')
        
        # 添加生化指标点
        if '生化标' in patient_info:
            for i, item in enumerate(patient_info['生化指标']):
                node_id = f'biochem_{i}'
                net.add_node(node_id,
                            label=f"{item.get('项目')}：{item.get('结果')}",
                            color='#DDA0DD',  # plum
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='生化指标')
        
        # 创建临时文件保存HTML
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
            try:
                net.save_graph(f.name)
                return f.name
            except Exception as e:
                raise
                
    except Exception as e:
        raise

def display_parsed_documents():
    """显示已解析的文档"""
    st.subheader("查看已解析的文档")
    
    try:
        with OracleGraphStore() as graph_store:
            patients = graph_store.get_all_patients()
            
            if not patients:
                st.info("📭 数据库中暂无结构化文档，请先在文档管理中上传并结构化文档")
                return
                
            st.write("已解析的文档中包含以下患者：")
            
            # 患列表
            for patient in patients:
                patient_name = patient.get('姓名', '未知患者')
                # 使用expander使每个患者的信息默认折叠
                with st.expander(f"📋 {patient_name}", expanded=False):
                    # 获取患者完整信息
                    patient_info = graph_store.get_patient_info(patient_name)
                    if patient_info:
                        # 创建两页标签
                        tab1, tab2 = st.tabs(["知识图谱", "完整数据"])
                        
                        with tab1:
                            try:
                                # 创建并显示交互式网络图
                                html_path = visualize_patient_graph(patient_info)
                                with open(html_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                components.html(html_content, height=600)
                                # 清理临时文件
                                os.unlink(html_path)
                            except Exception as e:
                                st.error(f"显示图形时出错: {str(e)}")
                        
                        with tab2:
                            st.json(patient_info)
                    else:
                        st.error("无获取患者详细信息")
            
    except Exception as e:
        logger.error(f"显示已解析文档失败: {str(e)}")
        st.error("显示已解析文档时出现错误")

def display_document_management():
    """显示文档管理界面"""
    st.header("文档管理")
    
    try:
        # 显示已上传的文档
        st.subheader("已上传的文档")
        files = list(UPLOAD_DIR.glob("*.*"))
        
        if files:
            # 创建文档列表
            for file in files:
                try:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(file.name)
                        
                    with col2:
                        if st.button("向量化", key=f"vec_{file.name}"):
                            with st.spinner("正在向量化..."):
                                try:
                                    if vectorize_document(file):
                                        st.success("向量化成功")
                                    else:
                                        st.error("向量化失败")
                                except Exception as e:
                                    logger.error(f"向量化文档失败: {str(e)}")
                                    st.error(f"向量化失败: {str(e)}")
                                    
                    with col3:
                        if st.button("结构化", key=f"struct_{file.name}"):
                            with st.spinner("正在结构化..."):
                                try:
                                    # 创建文件对象
                                    file_obj = FileObject(file)
                                    try:
                                        result = parse_document_to_json(file_obj)
                                        if result:
                                            if "error" in result:
                                                st.error(result["error"])
                                            else:
                                                st.success("结构化成功")
                                        else:
                                            st.error("结构化失败")
                                    finally:
                                        file_obj.close()
                                except Exception as e:
                                    logger.error(f"结构化文档失败: {str(e)}")
                                    st.error(f"结构化失败: {str(e)}")
                                    
                    with col4:
                        if st.button("图数据", key=f"graph_{file.name}"):
                            with st.spinner("正在处理图数据..."):
                                try:
                                    # 读取文件内容
                                    content = read_file_content(file)
                                    if content:
                                        if process_graph_data(file.name, content):
                                            st.success("图数据处理成功")
                                        else:
                                            st.error("图数据处理失败")
                                    else:
                                        st.error("无法读取文件内容")
                                except Exception as e:
                                    logger.error(f"处理图数据失败: {str(e)}")
                                    st.error(f"处理图数据失败: {str(e)}")
                                    
                    st.markdown("---")
                    
                except Exception as e:
                    logger.error(f"处理文档 {file.name} 时出错: {str(e)}")
                    continue
                    
        else:
            st.info("暂无上传的文档")
            
        # 文件上传部分
        uploaded_file = st.file_uploader("上传医疗文档", type=["pdf", "docx", "txt"])
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"已选择 {uploaded_file.name}")
            with col2:
                if st.button("保存文档"):
                    with st.spinner(f"正在保存 {uploaded_file.name}..."):
                        try:
                            file_path = save_uploaded_file(uploaded_file)
                            if file_path:
                                st.success(f"文件保存成功: {file_path}")
                            else:
                                st.error("文件保存失败")
                        except Exception as e:
                            logger.error(f"保存上传文件失败: {str(e)}")
                            st.error(f"文件保存失败: {str(e)}")
                            
    except Exception as e:
        logger.error(f"显示文档管理界面失败: {str(e)}")
        st.error(f"显示文档管理界面时出现错误: {str(e)}")

def display_vector_search():
    """显示向量检索界面"""
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

def display_structured_search():
    """显示结构化检索界面"""
    st.header("结构化检索")
    
    try:
        # 检查数据库中的结构化文档
        with OracleJsonStore() as json_store:
            # 首先检查表是否存在
            check_table_sql = """
            SELECT table_name 
            FROM user_tables 
            WHERE table_name = 'DOCUMENT_JSON'
            """
            result = json_store.execute_search(check_table_sql)
            if not result:
                st.warning("数据库未初始化，请先在文档管理中上传并结构化文档")
                return
            
            # 获取所有文档
            check_sql = """
            SELECT doc_info, doc_json
            FROM DOCUMENT_JSON 
            ORDER BY id DESC
            """
            all_docs = json_store.execute_search(check_sql)
            if not all_docs:
                st.info("📭 数据库暂无结构化文档，请先在文档管理中上传并结构化文档")
                return
            
            # 显示数据库中的所有文档详细信息
            st.subheader("📚 数据库中所有文档")
            st.write(f"📊 数据库中共有 {len(all_docs)} 个文档")
            
            # 处理每个文档
            for doc in all_docs:
                try:
                    if not isinstance(doc.get('doc_json'), dict):
                        logger.warning(f"跳过无效的文档格式: {doc.get('doc_info', '未知')}")
                        continue
                        
                    data = doc['doc_json']
                    patient_name = data.get("患者姓名", Path(doc['doc_info']).stem)
                    
                    # 使用expander为每个患者创建折叠面板
                    with st.expander(f"📋 {patient_name}", expanded=False):
                        # 创建标签页
                        tabs = st.tabs([
                            "基本信息", "主诉与诊断", "现病史", 
                            "生命体征", "生化指标", "诊疗经过"
                        ])
                        
                        with tabs[0]:
                            if "基本信息" in data:
                                st.json(data["基本信息"])
                            else:
                                st.info("未记录基本信息")
                        
                        with tabs[1]:
                            st.markdown("**主诉**")
                            st.write(data.get("主诉", "未记录"))
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**入院诊断**")
                                for diag in data.get("入院诊断", []):
                                    st.write(f"- {diag}")
                            with col2:
                                st.markdown("**出院诊断**")
                                for diag in data.get("出院诊断", []):
                                    st.write(f"- {diag}")
                        
                        with tabs[2]:
                            st.markdown("**现病史**")
                            for item in data.get("现病史", []):
                                st.write(f"- {item}")
                        
                        with tabs[3]:
                            if "生命体征" in data:
                                st.json(data["生命体征"])
                            else:
                                st.info("未记录生命体征")
                        
                        with tabs[4]:
                            if "生化指标" in data:
                                st.json(data["生化指标"])
                            else:
                                st.info("未记录生化指标")
                        
                        with tabs[5]:
                            st.markdown("**诊疗经过**")
                            st.write(data.get("诊疗经过", "未记录"))
                            if "出院医嘱" in data:
                                st.markdown("**出院医嘱**")
                                for advice in data["出院医嘱"]:
                                    st.write(f"- {advice}")
                except Exception as e:
                    logger.error(f"处理文档时出错: {str(e)}, 文档信息: {doc.get('doc_info', '未知')}")
                    continue
            
            # 搜索功能
            st.divider()
            st.subheader("🔍 智能搜索")
            query = st.text_input("请输入查询内容（支持结构化数据搜索）")
            
            if query:
                with st.spinner("正在分析查询并搜索..."):
                    try:
                        results = search_documents(query)
                        if results:
                            # 显示搜索结果
                            display_search_results(query, results)
                        else:
                            st.warning("未找到相关信息")
                    except Exception as e:
                        logger.error(f"执行搜索时出错: {str(e)}")
                        st.error("执行搜索时出现错误")
                        
    except Exception as e:
        logger.error(f"检索文档时发生错误: {str(e)}")
        st.error(f"检索文档时发生错误: {str(e)}")

def display_property_graph_search():
    """显示属性图检索界面"""
    st.header("属性图检索")
    
    # 创建标签页
    tab1, tab2 = st.tabs(["查询模板", "自定义查询"])
    
    with tab1:
        st.subheader("常用查询模板")
        query_type = st.selectbox(
            "选择查询类型",
            [
                "患者相似症状分析",
                "患者生化指标异常关联",
                "患者诊断关系网络",
                "患者用药关联分析",
                "患者治疗方案对比"
            ]
        )
        
        if query_type == "患者相似症状分析":
            patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
            if st.button("分析"):
                with st.spinner("正在分析相似症状..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 主查询
                            query = """
                            SELECT *
                            FROM GRAPH_TABLE ( MEDICAL_KG
                                MATCH (v1) -[e1]-> (s1), (v2) -[e2]-> (s2)
                                WHERE v1.ENTITY_TYPE = '患者'
                                AND v2.ENTITY_TYPE = '患者'
                                AND v1.ENTITY_NAME = :patient_name
                                AND v1.ENTITY_NAME != v2.ENTITY_NAME
                                AND e1.RELATION_TYPE = '现病史'
                                AND e2.RELATION_TYPE = '现病史'
                                COLUMNS (
                                    v1.ENTITY_NAME AS patient1,
                                    v2.ENTITY_NAME AS patient2,
                                    JSON_VALUE(s1.ENTITY_VALUE, '$.症状') AS symptom1,
                                    JSON_VALUE(s2.ENTITY_VALUE, '$.症状') AS symptom2
                                )
                            )
                            """
                            results = graph_store.execute_pgql(query, {"patient_name": patient_name})
                            if results:
                                # 构建用于分析的文本
                                analysis_text = []
                                target_symptoms = []
                                other_patients = {}
                                
                                # 整理症状数据
                                for result in results:
                                    if result['symptom1']:
                                        target_symptoms.append(result['symptom1'])
                                    if result['patient2'] not in other_patients:
                                        other_patients[result['patient2']] = set()
                                    if result['symptom2']:
                                        other_patients[result['patient2']].add(result['symptom2'])
                                
                                if target_symptoms:
                                    analysis_text.append(f"目标患者 {patient_name} 的症状：")
                                    for symptom in sorted(set(target_symptoms)):
                                        analysis_text.append(f"- {symptom}")
                                    
                                    analysis_text.append("\n其他患者的症状：")
                                    for p_name, symptoms in other_patients.items():
                                        if symptoms:
                                            analysis_text.append(f"\n{p_name} 的症状：")
                                            for symptom in sorted(symptoms):
                                                analysis_text.append(f"- {symptom}")
                                    
                                    # 调用OpenAI进行分析
                                    with st.spinner("正在分析症状相似度..."):
                                        try:
                                            client = OpenAI(
                                                api_key=os.getenv('OPENAI_API_KEY'),
                                                base_url=os.getenv('OPENAI_API_BASE')
                                            )
                                            
                                            # 构建提示词
                                            prompt = f"""请分析以下患者的症状信息，找出症状之间的相似性和可能的关联：

患者症状信息：
{chr(10).join(analysis_text)}

请从以下几个方面进行分析：
1. 各个患者与目标患者（{patient_name}）症状的相似度（用百分比表示）
2. 症状的相似性和关联性分析
3. 可能的共同病因
4. 需要注意的医学问题

请用中文回答，并尽可能专业和详细。对于症状相似度的分析，请给出具体的百分比数值。"""

                                            response = client.chat.completions.create(
                                                model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                                                messages=[
                                                    {"role": "system", "content": "你是一个专业的医学顾问，擅长分析患者症状的相似度。请从医学专业的角度分析症状的相关性和可能的病因。"},
                                                    {"role": "user", "content": prompt}
                                                ],
                                                temperature=0.7
                                            )
                                            
                                            analysis_result = response.choices[0].message.content
                                            st.write("### 症状分析结果")
                                            st.write(analysis_result)
                                        except Exception as e:
                                            st.error(f"分析失败: {str(e)}")
                                            logger.error(f"分析失败: {str(e)}", exc_info=True)
                                else:
                                    st.warning(f"未找到 {patient_name} 的症状记录")
                            else:
                                st.info("未找到任何症状记录")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        logger.error(f"分析失败: {str(e)}", exc_info=True)
                        
        elif query_type == "患者生化指标异常关联":
            patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
            if st.button("分析"):
                with st.spinner("正在分析生化指标异常关联..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 使用JSON_TABLE查询异常生化指标
                            query = """
                            SELECT v.ENTITY_NAME as patient,
                                   i.项目 as indicator,
                                   i.结果 as value,
                                   i.单位 as unit,
                                   i.参考范围 as reference
                            FROM MEDICAL_ENTITIES v,
                                 JSON_TABLE(v.ENTITY_VALUE, '$.生化指标[*]'
                                     COLUMNS (
                                         项目 VARCHAR2(100) PATH '$.项目',
                                         结果 VARCHAR2(100) PATH '$.结果',
                                         单位 VARCHAR2(100) PATH '$.单位',
                                         参考范围 VARCHAR2(100) PATH '$.参考范围'
                                     )
                                 ) i
                            WHERE v.ENTITY_TYPE = '患者'
                            AND v.ENTITY_NAME = :patient_name
                            AND i.参考范围 = '异常'
                            """
                            results = graph_store.execute_sql(query, {"patient_name": patient_name})
                            if results:
                                # 构建用于分析的文本
                                analysis_text = []
                                analysis_text.append(f"患者 {patient_name} 的异常生化指标：")
                                for result in results:
                                    analysis_text.append(f"- {result['indicator']}: {result['value']} {result['unit']}")
                                
                                # 构建提示词
                                prompt = f"""
                                请分析以下患者的异常化指，给出专业的医学分析意见。
                                请包含以下方面：
                                1. 异常指标的临床意义
                                2. 可能的病理生理机制
                                3. 需要关注的健康风险
                                4. 建议进一步检查的项目
                                5. 生活方式建议

                                {chr(10).join(analysis_text)}

                                请用专业但易懂的语言回答。
                                """

                                try:
                                    # 初始化OpenAI客户端
                                    client = OpenAI(
                                        api_key=os.getenv("OPENAI_API_KEY"),
                                        base_url=os.getenv("OPENAI_API_BASE")
                                    )
                                    
                                    # 调用OpenAI API进行分析
                                    response = client.chat.completions.create(
                                        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                        messages=[
                                            {"role": "system", "content": "你是一位经验丰富的临床医生，擅长解读各种生化指标。"},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.7
                                    )
                                    
                                    # 显示分析结果
                                    analysis = response.choices[0].message.content
                                    st.success(f"找到 {len(results)} 个异常指标")
                                    st.markdown(analysis)
                                except Exception as e:
                                    st.error(f"分析失败: {str(e)}")
                            else:
                                st.info("未找到异常生化指标记录")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        
        elif query_type == "患者诊断关系网络":
            if st.button("分析"):
                with st.spinner("正在分析诊断关系网络..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 使用JSON_TABLE从实体表中提取诊断信息
                            query = """
                            WITH DIAGNOSES AS (
                                SELECT 
                                    e.ENTITY_NAME as patient_name,
                                    d.类型 as diagnosis_type,
                                    d.内容 as diagnosis
                                FROM MEDICAL_ENTITIES e,
                                     JSON_TABLE(e.ENTITY_VALUE, '$.主诉与诊断[*]'
                                         COLUMNS (
                                             类型 VARCHAR2(100) PATH '$.类型',
                                             内容 VARCHAR2(1000) PATH '$.内容'
                                         )
                                     ) d
                                WHERE e.ENTITY_TYPE = '患者'
                                AND d.类型 IN ('入院诊断', '出院诊断')
                            )
                            SELECT DISTINCT 
                                d1.patient_name AS patient1,
                                d2.patient_name AS patient2,
                                d1.diagnosis_type AS diagnosis_type,
                                d1.diagnosis AS diagnosis_value
                            FROM DIAGNOSES d1
                            JOIN DIAGNOSES d2 ON d1.diagnosis = d2.diagnosis 
                                AND d1.diagnosis_type = d2.diagnosis_type
                                AND d1.patient_name < d2.patient_name
                            ORDER BY d1.patient_name, d2.patient_name, d1.diagnosis_type
                            """
                            logger.info("执行SQL查询: %s", query)
                            results = graph_store.execute_sql(query)
                            logger.info("查询结果: %r", results)
                            
                            if results:
                                # 创建网络图
                                net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                                
                                # 设置物理布局选项
                                net.set_options("""
                                {
                                    "nodes": {
                                        "font": {
                                            "size": 16,
                                            "face": "Microsoft YaHei"
                                        }
                                    },
                                    "edges": {
                                        "color": {
                                            "color": "#666666",
                                            "highlight": "#000000"
                                        },
                                        "font": {
                                            "size": 12,
                                            "face": "Microsoft YaHei"
                                        }
                                    },
                                    "physics": {
                                        "enabled": true,
                                        "solver": "forceAtlas2Based",
                                        "forceAtlas2Based": {
                                            "gravitationalConstant": -50,
                                            "centralGravity": 0.01,
                                            "springLength": 200,
                                            "springConstant": 0.08,
                                            "damping": 0.4,
                                            "avoidOverlap": 0.5
                                        }
                                    }
                                }
                                """)
                                
                                # 添加节点和边
                                nodes = set()
                                diagnosis_data = []
                                
                                for result in results:
                                    patient1 = result['patient1']
                                    patient2 = result['patient2']
                                    diagnosis_type = result['diagnosis_type']
                                    diagnosis_value = result['diagnosis_value']
                                    
                                    # 收集诊断数据用于分析
                                    diagnosis_data.append({
                                        'patient1': patient1,
                                        'patient2': patient2,
                                        'diagnosis_type': diagnosis_type,
                                        'diagnosis': diagnosis_value
                                    })
                                    
                                    if patient1 not in nodes:
                                        net.add_node(patient1, label=patient1, color='#add8e6', size=30)
                                        nodes.add(patient1)
                                    if patient2 not in nodes:
                                        net.add_node(patient2, label=patient2, color='#add8e6', size=30)
                                        nodes.add(patient2)
                                        
                                    net.add_edge(patient1, patient2, 
                                               title=f"{diagnosis_type}: {diagnosis_value}",
                                               label=diagnosis_value)
                                
                                # 保存并显示网络图
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
                                    net.save_graph(f.name)
                                    with open(f.name, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    components.html(html_content, height=600)
                                    os.unlink(f.name)
                                
                                # 使用大模型分析诊断关系
                                if diagnosis_data:
                                    with st.spinner("正在分析诊断关系..."):
                                        # 获取每个患者的完整诊断信息
                                        patient_diagnoses_query = """
                                        SELECT 
                                            e.ENTITY_NAME as patient_name,
                                            JSON_QUERY(e.ENTITY_VALUE, '$.主诉与诊断') as diagnoses
                                        FROM MEDICAL_ENTITIES e
                                        WHERE e.ENTITY_TYPE = '患者'
                                        """
                                        patient_diagnoses_results = graph_store.execute_sql(patient_diagnoses_query)
                                        
                                        # 构建分析文本
                                        analysis_text = []
                                        for result in patient_diagnoses_results:
                                            patient_name = result['patient_name']
                                            diagnoses = json.loads(result['diagnoses'])
                                            
                                            analysis_text.append(f"\n患者 {patient_name}:")
                                            admission_diagnoses = []
                                            discharge_diagnoses = []
                                            
                                            for diag in diagnoses:
                                                if diag['类型'] == '入院诊断':
                                                    admission_diagnoses.append(diag['内容'])
                                                elif diag['类型'] == '出院诊断':
                                                    discharge_diagnoses.append(diag['内容'])
                                            
                                            if admission_diagnoses:
                                                analysis_text.append("入院诊断：")
                                                for diag in admission_diagnoses:
                                                    analysis_text.append(f"- {diag}")
                                            
                                            if discharge_diagnoses:
                                                analysis_text.append("出院诊断：")
                                                for diag in discharge_diagnoses:
                                                    analysis_text.append(f"- {diag}")
                                        
                                        # 构建提示词
                                        prompt = f"""
                                        请分析以下患者群体的诊断关系网络，给出专业的医学分析意见。
                                        请包含以下方面：
                                        1. 患者群体的主要诊断类型分布
                                        2. 入院诊断和出院诊断的变化分析
                                        3. 诊断之间的关联性分析
                                        4. 可能的治疗路径和效果分析
                                        5. 对临床诊疗的建议

                                        患者诊断数据：
                                        {chr(10).join(analysis_text)}

                                        请用专业但易懂的语言回答。
                                        """

                                        try:
                                            # 初始化OpenAI客户端
                                            client = OpenAI(
                                                api_key=os.getenv("OPENAI_API_KEY"),
                                                base_url=os.getenv("OPENAI_API_BASE")
                                            )
                                            
                                            # 调用OpenAI API进行分析
                                            response = client.chat.completions.create(
                                                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                                messages=[
                                                    {"role": "system", "content": "你是一位经验丰富的临床医生，擅长分析患者诊断关系和治疗路径。"},
                                                    {"role": "user", "content": prompt}
                                                ],
                                                temperature=0.7
                                            )
                                            
                                            # 显示分析结果
                                            st.write("### 诊断关系网络分析")
                                            st.markdown(response.choices[0].message.content)
                                        except Exception as e:
                                            logger.error("大模型分析失败: %s", str(e))
                                            st.error(f"分析失败: {str(e)}")
                                else:
                                    st.info("未找到足够的诊断数据进行分析")
                            else:
                                st.info("未找到诊断关系")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        
        elif query_type == "患者用药关联分析":
            patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
            if st.button("分析"):
                with st.spinner("正在分析用药关联..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 使用JSON_TABLE从实体表中提取用药信息
                            query = """
                            WITH MEDICATIONS AS (
                                SELECT 
                                    e.ENTITY_NAME as patient_name,
                                    m.药品 as medication,
                                    m.用法 as usage,
                                    m.剂量 as dosage
                                FROM MEDICAL_ENTITIES e,
                                     JSON_TABLE(e.ENTITY_VALUE, '$.用药记录[*]'
                                         COLUMNS (
                                             药品 VARCHAR2(100) PATH '$.药品',
                                             用法 VARCHAR2(100) PATH '$.用法',
                                             剂量 VARCHAR2(100) PATH '$.剂量'
                                         )
                                     ) m
                                WHERE e.ENTITY_TYPE = '患者'
                            )
                            SELECT DISTINCT 
                                m1.patient_name AS patient1,
                                m2.patient_name AS patient2,
                                m1.medication AS medication_name,
                                m1.usage AS medication_usage,
                                m1.dosage AS medication_dosage
                            FROM MEDICATIONS m1
                            JOIN MEDICATIONS m2 ON m1.medication = m2.medication
                                AND m1.patient_name = :patient_name
                                AND m1.patient_name != m2.patient_name
                            ORDER BY m1.patient_name, m2.patient_name, m1.medication
                            """
                            results = graph_store.execute_sql(query, {"patient_name": patient_name})
                            
                            if results:
                                st.success(f"找到 {len(results)} 个用药关联")
                                
                                # 创建网络图
                                net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                                
                                # 设置物理布局选项
                                net.set_options("""
                                {
                                    "nodes": {
                                        "font": {
                                            "size": 16,
                                            "face": "Microsoft YaHei"
                                        }
                                    },
                                    "edges": {
                                        "color": {
                                            "color": "#666666",
                                            "highlight": "#000000"
                                        },
                                        "font": {
                                            "size": 12,
                                            "face": "Microsoft YaHei"
                                        }
                                    },
                                    "physics": {
                                        "enabled": true,
                                        "solver": "forceAtlas2Based",
                                        "forceAtlas2Based": {
                                            "gravitationalConstant": -50,
                                            "centralGravity": 0.01,
                                            "springLength": 200,
                                            "springConstant": 0.08,
                                            "damping": 0.4,
                                            "avoidOverlap": 0.5
                                        }
                                    }
                                }
                                """)
                                
                                # 添加节点和边
                                nodes = set()
                                medication_data = []
                                
                                for result in results:
                                    patient1 = result['patient1']
                                    patient2 = result['patient2']
                                    medication = result['medication_name']
                                    usage = result.get('medication_usage', '')
                                    dosage = result.get('medication_dosage', '')
                                    
                                    # 收集用药数据用于分析
                                    medication_data.append({
                                        'patient1': patient1,
                                        'patient2': patient2,
                                        'medication': medication,
                                        'usage': usage,
                                        'dosage': dosage
                                    })
                                    
                                    if patient1 not in nodes:
                                        net.add_node(patient1, label=patient1, color='#add8e6', size=30)
                                        nodes.add(patient1)
                                    if patient2 not in nodes:
                                        net.add_node(patient2, label=patient2, color='#add8e6', size=30)
                                        nodes.add(patient2)
                                        
                                    edge_label = f"{medication}"
                                    if usage and dosage:
                                        edge_title = f"{medication}\n用法：{usage}\n剂量：{dosage}"
                                    else:
                                        edge_title = medication
                                        
                                    net.add_edge(patient1, patient2, 
                                               title=edge_title,
                                               label=edge_label)
                                
                                # 保存并显示网络图
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
                                    net.save_graph(f.name)
                                    with open(f.name, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    components.html(html_content, height=600)
                                    os.unlink(f.name)
                                
                                # 使用大模型分析用药关联
                                if medication_data:
                                    with st.spinner("正在分析用药关联..."):
                                        # 获取每个患者的完整用药信息
                                        patient_medications_query = """
                                        SELECT 
                                            e.ENTITY_NAME as patient_name,
                                            JSON_QUERY(e.ENTITY_VALUE, '$.用药记录') as medications
                                        FROM MEDICAL_ENTITIES e
                                        WHERE e.ENTITY_TYPE = '患者'
                                        """
                                        patient_medications_results = graph_store.execute_sql(patient_medications_query)
                                        
                                        # 构建分析文本
                                        analysis_text = []
                                        for result in patient_medications_results:
                                            patient_name = result['patient_name']
                                            medications = json.loads(result['medications'] or '[]')
                                            
                                            if medications:
                                                analysis_text.append(f"\n患者 {patient_name} 的用药记录：")
                                                for med in medications:
                                                    med_text = f"- {med['药品']}"
                                                    if '用法' in med:
                                                        med_text += f"（{med['用法']}"
                                                        if '剂量' in med:
                                                            med_text += f"，{med['剂量']}"
                                                        med_text += "）"
                                                    analysis_text.append(med_text)
                                        
                                        # 构建提示词
                                        prompt = f"""
                                        请分析以下患者群体的用药关联情况，给出专业的医学分析意见。
                                        请包含以下方面：
                                        1. 患者群体的主要用药类型分布
                                        2. 药物之间的相互作用分析
                                        3. 用药方案的合理性分析
                                        4. 可能的用药风险提示
                                        5. 对临床用药的建议

                                        患者用药数据：
                                        {chr(10).join(analysis_text)}

                                        请用专业但易懂的语言回答。
                                        """

                                        try:
                                            # 初始化OpenAI客户端
                                            client = OpenAI(
                                                api_key=os.getenv("OPENAI_API_KEY"),
                                                base_url=os.getenv("OPENAI_API_BASE")
                                            )
                                            
                                            # 调用OpenAI API进行分析
                                            response = client.chat.completions.create(
                                                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                                messages=[
                                                    {"role": "system", "content": "你是一位经验丰富的临床药师，擅长分析患者用药方案和药物相互作用。"},
                                                    {"role": "user", "content": prompt}
                                                ],
                                                temperature=0.7
                                            )
                                            
                                            # 显示分析结果
                                            st.write("### 用药关联分析")
                                            st.markdown(response.choices[0].message.content)
                                        except Exception as e:
                                            logger.error("大模型分析失败: %s", str(e))
                                            st.error(f"分析失败: {str(e)}")
                                else:
                                    st.info("未找到足够的用药数据进行分析")
                            else:
                                st.info("未找到用药关联")
                    except Exception as e:
                        logger.error("分析失败: %s", str(e))
                        st.error(f"分析失败: {str(e)}")
                        
        elif query_type == "患者治疗方案对比":
            patient1 = st.selectbox("选择第一个患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
            patient2 = st.selectbox("选择第二个患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
            if patient1 != patient2 and st.button("对比"):
                with st.spinner("正在对比治疗方案..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 使用PGQL查询治疗方案
                            query = """
                            SELECT 
                                v.entity_name AS patient,
                                e.relation_type AS treatment_type,
                                t.entity_value AS treatment_detail
                            MATCH (v) -[e]-> (t)
                            WHERE v.entity_type = 'PATIENT'
                              AND v.entity_name IN (:patient1, :patient2)
                              AND e.relation_type = '诊疗经过'
                            """
                            results = graph_store.execute_pgql(query, {
                                "patient1": patient1,
                                "patient2": patient2
                            })
                            
                            if results:
                                # 按患者分组治疗方案
                                treatments = {}
                                for result in results:
                                    patient = result['patient']
                                    if patient not in treatments:
                                        treatments[patient] = []
                                    treatments[patient].append({
                                        'treatment': result['treatment_detail'],
                                        'effect': result['effect']
                                    })
                                
                                # 显示对比结果
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader(patient1)
                                    if patient1 in treatments:
                                        for t in treatments[patient1]:
                                            st.write(f"- {t['treatment']} (效果: {t['effect']})")
                                    else:
                                        st.info("无治疗方案记录")
                                        
                                with col2:
                                    st.subheader(patient2)
                                    if patient2 in treatments:
                                        for t in treatments[patient2]:
                                            st.write(f"- {t['treatment']} (效果: {t['effect']})")
                                    else:
                                        st.info("无治疗方案记录")
                            else:
                                st.info("未找到治疗方案记录")
                    except Exception as e:
                        st.error(f"对比失败: {str(e)}")
    
    with tab2:
        st.subheader("自定义PGQL查询")
        st.markdown("""
        您可以输入自定义的PGQL查询语句。以下是一些示例：
        
        1. 查询患者的所有症状：
        ```sql
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v) -[e]-> (s)
            WHERE v.ENTITY_TYPE = '患者'
            AND v.ENTITY_NAME = '张某某'
            AND e.RELATION_TYPE = '现病史'
            COLUMNS (
                v.ENTITY_NAME AS patient_name,
                JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS symptom
            )
        )
        ```
        
        2. 查询所有有发热症状的患者：
        ```sql
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v) -[e]-> (s)
            WHERE v.ENTITY_TYPE = '患者'
            AND e.RELATION_TYPE = '现病史'
            AND JSON_EXISTS(s.ENTITY_VALUE, '$.症状?(@ like_regex "发热" flag "i")')
            COLUMNS (
                v.ENTITY_NAME AS patient_name,
                JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS symptom
            )
        )
        ```
        
        3. 查询患者的异常生化指标：
        ```sql
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v) -[e]-> (s)
            WHERE v.ENTITY_TYPE = '患者'
            AND v.ENTITY_NAME = '张某某'
            AND e.RELATION_TYPE = '生化指标'
            AND JSON_EXISTS(s.ENTITY_VALUE, '$.参考范围?(@ == "异常")')
            COLUMNS (
                v.ENTITY_NAME AS patient_name,
                JSON_VALUE(s.ENTITY_VALUE, '$.项目') AS indicator,
                JSON_VALUE(s.ENTITY_VALUE, '$.结果') AS value,
                JSON_VALUE(s.ENTITY_VALUE, '$.单位') AS unit,
                JSON_VALUE(s.ENTITY_VALUE, '$.参考范围') AS reference
            )
        )
        ```
        
        4. 查询具有相似症状的患者：
        ```sql
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v1) -[e1]-> (s1), (v2) -[e2]-> (s2)
            WHERE v1.ENTITY_TYPE = '患者'
            AND v2.ENTITY_TYPE = '患者'
            AND v1.ENTITY_NAME != v2.ENTITY_NAME
            AND e1.RELATION_TYPE = '现病史'
            AND e2.RELATION_TYPE = '现病史'
            AND JSON_VALUE(s1.ENTITY_VALUE, '$.症状') = JSON_VALUE(s2.ENTITY_VALUE, '$.症状')
            COLUMNS (
                v1.ENTITY_NAME AS patient1,
                v2.ENTITY_NAME AS patient2,
                JSON_VALUE(s1.ENTITY_VALUE, '$.症状') AS common_symptom
            )
        )
        ```
        
        5. 查询患者的诊断和相关症状：
        ```sql
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v) -[e1]-> (d), (v) -[e2]-> (s)
            WHERE v.ENTITY_TYPE = '患者'
            AND v.ENTITY_NAME = '张某某'
            AND e1.RELATION_TYPE = '诊断'
            AND e2.RELATION_TYPE = '现病史'
            COLUMNS (
                v.ENTITY_NAME AS patient_name,
                JSON_VALUE(d.ENTITY_VALUE, '$.类型') AS diagnosis_type,
                JSON_VALUE(d.ENTITY_VALUE, '$.内容') AS diagnosis,
                JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS symptom
            )
        )
        ```
        """)
        
        query = st.text_area("输入PGQL查询语句", height=150)
        params = st.text_input("输入参数（JSON格式，可选）", "{}")
        
        if st.button("执行查询"):
            if query:
                with st.spinner("正在执行查询..."):
                    try:
                        # 解析参数
                        try:
                            params = json.loads(params) if params.strip() else {}
                        except json.JSONDecodeError:
                            st.error("参数格式错误，请使用有效的JSON格式")
                            return
                        
                        # 执行查询
                        with OracleGraphStore() as graph_store:
                            results = graph_store.execute_pgql(query, params)
                            if results:
                                st.success(f"查询成功，返回 {len(results)} 条结果")
                                st.json(results)
                            else:
                                st.info("查询未返回任何结果")
                    except Exception as e:
                        st.error(f"查询失败: {str(e)}")
            else:
                st.warning("请输入查询语句")

def analyze_medication():
    query = """
    WITH PATIENT_MEDS AS (
        SELECT 
            e.entity_name as patient_name,
            j.content as treatment_content
        FROM MEDICAL_ENTITIES e,
        JSON_TABLE(e.entity_value, '$.诊疗经过[*]' 
            COLUMNS (
                content PATH '$.内容'
            )
        ) j
        WHERE e.entity_type = '患者'
        AND REGEXP_LIKE(j.content, '给予|使用|服用|治疗|用药', 'i')
    )
    SELECT 
        patient_name,
        treatment_content,
        COUNT(*) OVER (PARTITION BY treatment_content) as usage_count
    FROM PATIENT_MEDS
    ORDER BY usage_count DESC, patient_name
    """
    
    results = execute_sql(query)
    
    if not results:
        st.warning("未找到用药记录数据。这可能是因为：\n1. 数据中没有记录用药信息\n2. 用药信息的记录格式需要标准化")
        return
        
    # 展示用药分析结果
    st.subheader("患者用药分析")
    
    # 创建数据表格
    df = pd.DataFrame(results, columns=['患者姓名', '用药记录', '使用频次'])
    
    # 显示详细数据表格
    st.dataframe(df)
    
    # 使用plotly创建可视化图表
    fig = px.bar(df, 
                 x='用药记录', 
                 y='使用频次',
                 title='用药频次分布',
                 color='使用频次',
                 hover_data=['患者姓名'])
    
    fig.update_layout(
        xaxis_title="用药记录",
        yaxis_title="使用频次",
        showlegend=True
    )
    
    st.plotly_chart(fig)
    
    # 添加AI分析见解
    st.subheader("AI分析见解")
    
    # 计算一些基本统计信息
    total_patients = len(df['患者姓名'].unique())
    total_medications = len(df['用药记录'].unique())
    most_common_med = df.loc[df['使用频次'].idxmax(), '用药记录']
    
    analysis_text = f"""
    根据数据分析，发现以下主要特点：
    
    1. 共有{total_patients}位患者的用药记录
    2. 记录中包含{total_medications}种不同的治疗方案
    3. 最常见的治疗方案是"{most_common_med}"
    
    建议：
    - 进一步细化用药记录的分类
    - 标准化用药记录的格式
    - 添加药物剂量、用药时间等详细信息
    """
    
    st.markdown(analysis_text)

def main():
    st.title("基于Oracle 23AI电子病历检索系统", anchor=False)
    st.markdown("<div style='text-align: right'>Developed by Huaiyuan Tan</div>", unsafe_allow_html=True)
    
    # 初始化数据库
    init_database()
    
    # 创建侧边栏菜单
    menu = st.sidebar.selectbox(
        "功能菜单",
        ["文档管理", "图数据检索", "向量检索", "结构化检索", "属性图检索"]
    )
    
    if menu == "文档管理":
        display_document_management()
    elif menu == "图数据检索":
        st.header("图数据检索")
        # 显示已解析的文档
        display_parsed_documents()
        
        # 添加搜索框
        query = st.text_input("请输入搜索内容（支持按患者姓名、症状、诊断等搜索）")
        if query:
            results = search_graph_data(query)
            display_graph_results(results, query)
    elif menu == "向量检索":
        display_vector_search()
    elif menu == "属性图检索":
        display_property_graph_search()
    else:  # 结构化检索
        display_structured_search()

class FileObject:
    """文件对象包装类，提供统一的文件操作接口"""
    def __init__(self, path):
        """初始化文件对象
        
        Args:
            path: 文件路径，可以是字符串或 Path 对象
        """
        self.path = Path(path)
        self.name = self.path.name
        self._file = None
        
        # 根据文件扩展名设置类型
        self.type = "application/pdf" if self.path.suffix.lower() == '.pdf' else "text/plain"
        
        # 验证文件是否存在
        if not self.path.exists():
            raise FileNotFoundError(f"文件不存在: {self.path}")
        
        # 验证文件是否可读
        if not os.access(self.path, os.R_OK):
            raise PermissionError(f"文件无法读取: {self.path}")
            
        logger.debug(f"创建文件对象: {self.path}, 类型: {self.type}")
    
    def read(self):
        """读取文件内容
        
        Returns:
            bytes: 文件的二进制内容
        """
        try:
            if self._file is None:
                self._file = open(self.path, 'rb')
            return self._file.read()
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            raise
    
    def seek(self, pos):
        """移动文件指针到指定位置
        
        Args:
            pos: 目标位置（字节偏移量）
        """
        try:
            if self._file is not None:
                self._file.seek(pos)
        except Exception as e:
            logger.error(f"移动文件指针失败: {str(e)}")
            raise
    
    def close(self):
        """关闭文件"""
        try:
            if self._file is not None:
                self._file.close()
                self._file = None
                logger.debug(f"关闭文件: {self.path}")
        except Exception as e:
            logger.error(f"关闭文件失败: {str(e)}")
            raise
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()

if __name__ == "__main__":
    main()

