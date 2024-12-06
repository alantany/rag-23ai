"""
AI知识问答系统 - Oracle Vector Store版本
"""

import streamlit as st
from utils.oracle_vector_store import OracleVectorStore
from utils.oracle_json_store import OracleJsonStore
from utils.medical_record_parser import MedicalRecordParser
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import shutil
from pathlib import Path
import logging
import openai
import pdfplumber
import json
import datetime
import hashlib
from typing import Dict, Any, List
from decimal import Decimal
from openai import OpenAI
from utils.oracle_graph_store import OracleGraphStore
from utils.medical_graph_parser import MedicalGraphParser
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import networkx as nx

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

def save_to_cache(file_path: str, data: Dict) -> None:
    """保存结构化数据到缓存"""
    cache_path = get_cache_path(file_path)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"缓存保存到: {cache_path}")

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
        # 使用已加载的模型进行向量化
        vector = embeddings_model.encode([content])[0]
        documents = [{"file_path": str(file_path), "content": content}]
        
        with OracleVectorStore() as vector_store:
            vector_store.add_vectors([vector], documents)
        return True
    except Exception as e:
        logger.error(f"向量化失败: {str(e)}")
        return False

def init_database():
    """初始化数据库表"""
    try:
        with OracleJsonStore() as json_store:
            # 检查表是否存在
            check_table_sql = """
            SELECT COUNT(*) as count
            FROM user_tables 
            WHERE table_name = 'DOCUMENT_JSON'
            """
            result = json_store.execute_search(check_table_sql)
            table_exists = result[0]['count'] > 0 if result else False
            
            if not table_exists:
                # 创建结构化文档表
                create_json_table_sql = """
                CREATE TABLE DOCUMENT_JSON (
                    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    doc_info VARCHAR2(500),
                    doc_json JSON
                )
                """
                json_store.execute_sql(create_json_table_sql)
                logger.info("成功创建DOCUMENT_JSON表")
            else:
                logger.debug("DOCUMENT_JSON表已存在")  # 改用debug级别的日志

    except Exception as e:
        logger.error(f"初始化数据库失败: {str(e)}")
        st.error(f"初始化数据库失败: {str(e)}")

class MedicalRecordParser:
    def __init__(self):
        self.client = openai.OpenAI(
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
                    {"role": "system", "content": "你是医疗数据结构化专家，擅长解析病历文本并生成规范的JSON和SQL。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )

            # 解析返回的JSON
            result = json.loads(response.choices[0].message.content)
            
            # 验证结果格式
            if not isinstance(result, dict) or 'structured_data' not in result or 'sql_statement' not in result:
                raise ValueError("GPT返回���数据格式不正确")

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

def parse_document_to_json(file):
    """解析文档为JSON格式"""
    try:
        content = read_file_content(file)
        
        # 解析为JSON
        parser = MedicalRecordParser()
        json_result = parser.parse_medical_record(content)
        
        # 解析为图数据
        graph_parser = MedicalGraphParser()
        graph_result = graph_parser.parse_to_graph(content, file.name)
        
        if "error" in json_result or "error" in graph_result:
            return False
            
        # 保存JSON结果
        with OracleJsonStore() as json_store:
            json_store.add_document(file.name, json_result)
            
        return True
    except Exception as e:
        logger.error(f"解析文档失败: {str(e)}")
        return False

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
        client = openai.OpenAI(
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
    """使用GPT分析查询意图并生成Oracle 23c JSON查询条件"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
        
        # 提供完整的JSON结构信息
        json_structure = """
{
    "患者姓名": "姓名",
    "性别": "男/女",
    "年龄": "数字",
    "入院日期": "YYYY-MM-DD",
    "院日期": "YYYY-MM-DD",
    "主诉": "主要状",
    "现病史": ["症状1", "症状2"],
    "入院诊断": ["诊断1", "诊断2"],
    "出院诊断": ["诊断1", "诊断2"],
    "命体征": {
        "体温": "包含单位",
        "血压": "包含单位"
    },
    "生化指标": {
        "天冬氨酸氨基转移酶": "值和单位",
        "丙氨酸氨基转移酶": "值和单位",
        "白细胞": "值和单位",
        "淋巴细��分比": "值单位",
        "中性粒细胞百分比": "值和单位",
        "血红蛋白": "值和单位",
        "血小板": "值和单位"
    },
    "诊疗经过": "治疗过程描述",
    "出院医嘱": ["医嘱1", "医2"]
}"""
        
        messages = [
            {"role": "system", "content": f"""你是一个Oracle 23c JSON查询专家。基于以下的JSON文档结构，帮助生成准确的Oracle JSON查询条件。

文档结构：
{json_structure}

数据库信息：
- 使用Oracle Database 23c
- 表名：DOCUMENT_JSON
- 字段：doc_info (VARCHAR2), doc_json (JSON)
- JSON数据存储在doc_json字段中

Oracle 23c JSON查询特：
1. 使用 JSON_EXISTS 进行条件匹配
2. 使用 JSON_VALUE 提取单个值
3. 使用 JSON_QUERY 提取JSON数组或对象
4. 支持点号访问嵌套属性
5. 支持组索引访问
6. 支持条件过滤：?(@ == "value")
7. 支持键名匹配：
   - 使用 $.生化指标.* 遍历所有指标
   - 使用 EXISTS 和 OR 组合多个条件

你的任务是：
1. 理解用户的自然语言查询
2. 根据档结构和Oracle 23c特性，生成最优的查询条件
3. 对于医学术语，考虑同义词和简写（如"转氨酶"可能指"天冬氨酸氨基转移酶"或"丙氨酸氨基转移酶"）
4. 返回格式为JSON：
{{
    "query_type": "查询类型",
    "conditions": ["Oracle JSON查询条件"],
    "fields": ["需要返回的段"],
    "keywords": ["键"]
}}

示例1：
输入："马某某的转氨酶指标"
输出：
{{
    "query_type": "检验结果",
    "conditions": [
        "JSON_EXISTS(doc_json, '$.患者姓名?(@ == \"马某某\")')",
        "(JSON_EXISTS(doc_json, '$.生化指标.天冬氨酸氨基转移酶') OR JSON_EXISTS(doc_json, '$.生化指标.丙氨酸氨基转移酶'))"
    ],
    "fields": ["生化标.天氨酸氨基转酶", "生化指标.丙氨酸氨基转移酶"],
    "keywords": ["马某某", "转氨酶"]
}}

示例2：
输入："查看马某某的淋巴细胞比例"
输出：
{{
    "query_type": "检验结果",
    "conditions": [
        "JSON_EXISTS(doc_json, '$.患姓名?(@ == \"马某某\")')",
        "JSON_EXISTS(doc_json, '$.生化指标.淋巴细胞百分比')"
    ],
    "fields": ["生化指标.淋巴细胞百分比"],
    "keywords": ["马某某", "淋巴细胞百分比"]
}}

示例3：
输入："马某某的血液相关指标"
输出：
{{
    "query_type": "检验结果",
    "conditions": [
        "JSON_EXISTS(doc_json, '$.患者姓名?(@ == \"马某某\")')",
        "(JSON_EXISTS(doc_json, '$.生化指标.血红蛋白') OR JSON_EXISTS(doc_json, '$.生化指标.血小板'))"
    ],
    "fields": [
        "生指标.白细胞",
        "生化指标.血红蛋白",
        "生化指标.血小板"
    ],
    "keywords": ["马某某"]
}}"""},
            {"role": "user", "content": query_text}
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        logger.error(f"GPT分析询失败: {str(e)}")
        return None

# 配置常量
TOP_K = 5  # 搜索结果返回的最大数量

def normalize_medical_term(query_text):
    """使用 GPT 将用�����询的标名称标准化"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
        
        messages = [
            {"role": "system", "content": """你是一个医疗指标名称标准化专家。
请将用户查询中的指标名称为标的疗标称

规则：
1. 查询中包含某个检验指标的同义词或近义词，返回标准名称
2. 如果不确定，返回原始词语
3. 返回格式为 JSON：{"standard_term": "标准名称"}

示：
输入："淋巴细胞比例"
输出：{"standard_term": "淋巴细胞百分比"}

输入："白细胞计数"
输出：{"standard_term": "白���胞"}

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
    """基于GPT生成的查询条件搜索文档，支持结构化数据和全文搜索"""
    try:
        # 使用GPT分析查询意图并生成查询条件
        analysis_result = analyze_query_with_gpt(query_text)
        logger.info(f"GPT分析结果: {json.dumps(analysis_result, ensure_ascii=False)}")
        
        if not analysis_result or 'conditions' not in analysis_result:
            logger.error("GPT分析结果格式错误")
            return []
            
        # 使用GPT生成的条件构建查询
        conditions = analysis_result.get('conditions', [])
        keywords = analysis_result.get('keywords', [])
        
        # 组合条件（使用AND连接姓名和其他条件）
        name_conditions = [c for c in conditions if '患者姓名' in c]
        other_conditions = [c for c in conditions if '患者姓名' not in c]
        
        # 构建JSON查条件
        if name_conditions and other_conditions:
            json_where = f"({' OR '.join(name_conditions)}) AND ({' OR '.join(other_conditions)})"
        else:
            json_where = " OR ".join(conditions) if conditions else "1=1"
            
        # 构建全文搜索条件（排除姓名关键词）
        content_conditions = []
        for keyword in keywords:
            if keyword not in ["马某某", "周某某"]:
                content_conditions.append(f"CONTAINS(content, '{keyword}') > 0")
        
        content_where = " OR ".join(content_conditions) if content_conditions else "1=1"
        
        # 构建完整的查询语句
        query = """
        SELECT d.doc_info,
               d.doc_json,
               d.content
        FROM DOCUMENT_JSON d
        WHERE 
            -- 首先匹配患者姓名（JSON或全文）
            (
                JSON_EXISTS(doc_json, '$.患者姓名?(@=="杨某某")') 
                OR CONTAINS(content, :1) > 0
            )
        ORDER BY id DESC
        FETCH FIRST :2 ROWS ONLY
        """
        
        # 获取姓名关键词（如果有）
        name_keyword = next((k for k in keywords if k in ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"]), None)
        
        # 执行查询
        with OracleJsonStore() as json_store:
            if name_keyword:
                # 构建带有具体姓名的查询
                actual_query = query.replace('杨某某', name_keyword)
                results = json_store.execute_search(actual_query, [name_keyword, TOP_K])
            else:
                results = json_store.execute_search(query, [TOP_K])
                
        if not results:
            logger.info("未找到匹配的文档")
            return []
                
        # 处理查询结果
        processed_results = []
        for row in results:
            try:
                doc_json = json.loads(json.dumps(row['doc_json'], cls=DecimalEncoder))
                processed_results.append({
                    'doc_info': row['doc_info'],
                    'doc_json': doc_json,
                    'content': row['content']
                })
            except Exception as e:
                logger.error(f"处理文档结果时出错: {str(e)}")
                continue
                    
        logger.info(f"数据库的文档数量: {len(processed_results)}")
        for result in processed_results:
            logger.info(f"文档路径: {result['doc_info']}")
                
        return processed_results
        
    except Exception as e:
        logger.error(f"JSON文档搜索失败: {str(e)}")
        return []

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

def generate_answer(query_text, doc_json, content=None):
    """根据查询意图生成答案，支持结构化数据和全文内容"""
    try:
        # 使用GPT分析查询意图
        analysis_result = analyze_query_with_gpt(query_text)
        if not analysis_result:
            return None
            
        query_type = analysis_result.get('query_type', '')
        fields = analysis_result.get('fields', [])
        keywords = analysis_result.get('keywords', [])
        
        # 获取患者姓名
        name = next((k for k in keywords if k in ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"]), "患者")
        
        # 如果查询包含症状或诊断相关的关键词，优先从全文内容中提取
        symptom_keywords = [k for k in keywords if k not in ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"]]
        if symptom_keywords and content:
            try:
                # 使用GPT从全文中提取相关信息
                extract_prompt = f"""
从以下医疗文档中提取与问题相关的信息：

问题：{query_text}
关注的症状/诊断：{', '.join(symptom_keywords)}

文档内容：
{content}

请提供简洁的答案，重点关注问题中提到的症状或诊断。如果找不到相关信息，请回复"未找到相关信息"。
"""
                
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "你是一个医疗信息提取专家。请从文档中准确提取信息，不要添加任何推测的内容。"},
                        {"role": "user", "content": extract_prompt}
                    ],
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content.strip()
                if answer != "未找到相关信息":
                    return f"{name}的情况：{answer}"
            except Exception as e:
                logger.error(f"GPT分析失败: {str(e)}")
                # 果GPT分析失败，继续尝试结构化数据
                
        # 如果没从全文中找到信息，或GPT分析失败，尝试从结构化数据中获取
        info = []
        for field in fields:
            if '.' in field:  # 理嵌套字段
                parent, child = field.split('.')
                if parent in doc_json and child in doc_json[parent]:
                    info.append(f"{child}是{doc_json[parent][child]}")
            else:  # 处理顶层字段
                if field in doc_json:
                    info.append(f"{field}是{doc_json[field]}")
        
        if info:
            if query_type:
                return f"{name}的{query_type}：" + "，".join(info)
            else:
                return f"{name}的信息：" + "，".join(info)
                
        return "未找到相关信息"
        
    except Exception as e:
        logger.error(f"生成答案失败: {str(e)}")
        return "抱歉，处理��的问题时出现错误"

def display_search_results(query_text, results):
    """显示搜索结果"""
    if not results:
        st.warning("未找到相关文档")
        return

    # 尝试生成精确答案
    for result in results:
        answer = generate_answer(query_text, result['doc_json'], result['content'])
        if answer and not answer.startswith("未找到相关信息"):
            st.success(answer)
        else:
            st.warning("未找到相关信息")

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

        患者 {patient_name} 的实际数据结构如下：
        {json.dumps(patient_data, ensure_ascii=False, indent=2)}

        你需要分析用户的查询意图，返回一个JSON对象（不要添加任何markdown格式或代码块标记），包含以下字段：
        - query_type: 查询类型，必���以下之一：基本信息/主诉与诊断/现病史/生命体征/生化指标/诊疗经过
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
                logger.warning(f"无效��query_type: {result['query_type']}, 使用默认值")
                result["query_type"] = "基本信息"
            
            # 确保patient_name与查询中识别的一致
            if result["patient_name"] != patient_name:
                logger.warning(f"patient_name不匹配: {result['patient_name']} != {patient_name}")
                result["patient_name"] = patient_name
            
            # 处理field值
            if result["field"] == result["query_type"] or result["field"] in valid_query_types:
                logger.info(f"将field从 {result['field']} 修改为 all")
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
                    # 返回所有基本信息
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

        # 添加生命体征���点
        if '生命体征' in patient_info:
            for i, item in enumerate(patient_info['生命体征']):
                node_id = f'vital_{i}'
                net.add_node(node_id,
                            label=f"{item.get('指标')}：{item.get('数值')}",
                            color='#F08080',  # lightcoral
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='生命体征')
        
        # 添��生化指标节点
        if '生化指标' in patient_info:
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
            
            # 显示患者列表
            for patient in patients:
                patient_name = patient.get('姓名', '未知患者')
                # 使用expander使每个患者的信息默认折叠
                with st.expander(f"📋 {patient_name}", expanded=False):
                    # 获取患者的完整信息
                    patient_info = graph_store.get_patient_info(patient_name)
                    if patient_info:
                        # 创建两个标签页
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
    
    # 显示已上传的文档
    st.subheader("已上传的文档")
    files = list(UPLOAD_DIR.glob("*.*"))
    if files:
        for file in files:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(file.name)
            with col2:
                if st.button("向量化", key=f"vec_{file.name}"):
                    with st.spinner("正在向量化..."):
                        if vectorize_document(file):
                            st.success("向量化成功")
                        else:
                            st.error("向量化失败")
            with col3:
                if st.button("结构化", key=f"struct_{file.name}"):
                    with st.spinner("正在结构化..."):
                        # 读取文件内容
                        content = read_file_content(file)
                        file_obj = type('FileObject', (), {
                            'name': file.name,
                            'type': 'application/pdf' if file.suffix == '.pdf' else 'text/plain',
                            'read': lambda: open(file, 'rb').read()
                        })()
                        if parse_document_to_json(file_obj):
                            st.success("结构化成功")
                        else:
                            st.error("结构化失败")
            with col4:
                if st.button("图数据", key=f"graph_{file.name}"):
                    with st.spinner("正在处理图数据..."):
                        # 读取文件内容
                        content = read_file_content(file)
                        if process_graph_data(file.name, content):
                            st.success("图数据处理成功")
                        else:
                            st.error("图数据处理失败")
            st.markdown("---")
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
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        st.success(f"文件保存成功: {file_path}")
                    else:
                        st.error("文件保存失败")

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
    
    # 检查数据库中的结构化文档
    with OracleJsonStore() as json_store:
        try:
            # 首先检查表是否存在
            check_table_sql = """
            SELECT COUNT(*) as count
            FROM user_tables 
            WHERE table_name = 'DOCUMENT_JSON'
            """
            result = json_store.execute_search(check_table_sql)
            table_exists = result[0]['count'] > 0 if result else False
            
            if not table_exists:
                st.warning("数据库未初始化，请先��文档管理中上传并结构化文档")
                return
            
            # 获取所有文档
            check_sql = """
            SELECT doc_info, doc_json, content 
            FROM DOCUMENT_JSON 
            ORDER BY id DESC
            """
            all_docs = json_store.execute_search(check_sql)
            
            # 显示数据库中的所有文档详细信息
            st.subheader("📚 数据库中所有文档")
            if all_docs:
                st.write(f"📊 数据库中共有 {len(all_docs)} 个文档")
                for doc in all_docs:
                    if isinstance(doc['doc_json'], dict):
                        data = doc['doc_json']
                        patient_name = data.get("患者名", Path(doc['doc_info']).stem)
                        
                        # 使用expander为每个患者创建折叠面板
                        with st.expander(f"📋 {patient_name}", expanded=False):
                            # 创建标签页
                            tabs = st.tabs([
                                "基本信息", "主诉与诊断", "现病史", 
                                "生命体征", "生化指标", "诊疗经过", "全文内容"
                            ])
                            
                            with tabs[0]:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**患者信息**")
                                    info = {
                                        "姓名": data.get("患者姓名", "未知"),
                                        "性别": data.get("性别", "未知"),
                                        "年龄": data.get("年龄", "未知"),
                                        "民族": data.get("民族", "未知"),
                                        "职业": data.get("职业", "未知"),
                                        "婚姻状况": data.get("婚姻状况", "未知")
                                    }
                                    st.json(info)
                                with col2:
                                    st.markdown("**住院信息**")
                                    info = {
                                        "入院日期": data.get("入院日期", "未知"),
                                        "出院日期": data.get("出院日期", "未知"),
                                        "住院天数": data.get("住院天数", "未知"),
                                        "出院情况": data.get("出院情况", "未知")
                                    }
                                    st.json(info)
                            
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
                                        
                            with tabs[6]:
                                st.markdown("**文档全文**")
                                if "content" in doc:
                                    st.text_area("", doc["content"], height=400)
                                else:
                                    st.info("未找到文档全文内容")
            else:
                st.info("📭 数据库暂无结构化文档，请先在文档管理中上传并结构化文档")
            
            # 搜索功能
            st.divider()
            st.subheader("🔍 智能搜索")
            query = st.text_input("请输入查询内容（支持结构化数据和全文搜索）")
            
            if query:
                with st.spinner("正在分析查询并搜索..."):
                    results = search_documents(query)
                    if results:
                        for result in results:
                            answer = generate_answer(query, result['doc_json'], result['content'])
                            if answer:
                                st.success(answer)
                                with st.expander("查看完整文档"):
                                    display_search_results(query, [result])
                    else:
                        st.warning("未找到相关信息")
                    
        except Exception as e:
            logger.error(f"检索文档时发生错误: {str(e)}")
            st.error(f"检索文档时发生错误: {str(e)}")

def display_property_graph_search():
    """��示属性图检索界面"""
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
                            # 使用PGQL查询相似症状
                            query = """
                            v1.entity_name AS patient1, 
                            v2.entity_name AS patient2,
                            e1.relation_type AS symptom1,
                            e2.relation_type AS symptom2
                            MATCH (v1) -[e1]-> () <-[e2]- (v2)
                            WHERE v1.entity_type = 'PATIENT' 
                              AND v2.entity_type = 'PATIENT'
                              AND v1.entity_name = :patient_name
                              AND v1.entity_name != v2.entity_name
                              AND e1.relation_type = '现病史'
                              AND e2.relation_type = '现病史'
                            """
                            results = graph_store.execute_pgql(query, {"patient_name": patient_name})
                            if results:
                                st.success(f"找到 {len(results)} 个相似症状")
                                for result in results:
                                    st.write(f"- {result['patient2']} 也有 '{result['symptom1']}' 症状")
                            else:
                                st.info("未找到相似症状")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        
        elif query_type == "患者生化指标异常关联":
            patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
            if st.button("分析"):
                with st.spinner("正在分析生化指标异常关联..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 使用PGQL查询异常生化指标
                            query = """
                            SELECT DISTINCT 
                                v.entity_name as patient,
                                e.indicator_name as indicator,
                                e.value as value,
                                e.unit as unit,
                                e.reference_range as reference
                            FROM (v) -[e:HAS_INDICATOR]-> ()
                            WHERE v.entity_type = 'PATIENT'
                            AND v.entity_name = :patient_name
                            AND e.reference_range = '异常'
                            """
                            results = graph_store.execute_pgql(query, {"patient_name": patient_name})
                            if results:
                                st.success(f"找到 {len(results)} 个异常指标")
                                for result in results:
                                    st.write(f"- {result['indicator']}: {result['value']}{result['unit']} (异常)")
                            else:
                                st.info("未找到异常指标")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        
        elif query_type == "患者诊断关系网络":
            if st.button("分析"):
                with st.spinner("正在分析诊断关系网络..."):
                    try:
                        with OracleGraphStore() as graph_store:
                            # 使用PGQL查询诊断关系
                            query = """
                            SELECT DISTINCT 
                                v1.entity_name AS patient1,
                                v2.entity_name AS patient2,
                                e1.relation_type AS diagnosis_type
                            MATCH (v1) -[e1]-> () <-[e2]- (v2)
                            WHERE v1.entity_type = 'PATIENT'
                              AND v2.entity_type = 'PATIENT'
                              AND v1.entity_name != v2.entity_name
                              AND e1.relation_type IN ('入院诊断', '出院诊断')
                              AND e2.relation_type = e1.relation_type
                            """
                            results = graph_store.execute_pgql(query)
                            if results:
                                # 创建网络图
                                net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                                
                                # 添加节点和边
                                nodes = set()
                                for result in results:
                                    patient1 = result['patient1']
                                    patient2 = result['patient2']
                                    diagnosis = result['diagnosis_type']
                                    
                                    if patient1 not in nodes:
                                        net.add_node(patient1, label=patient1, color='#add8e6')
                                        nodes.add(patient1)
                                    if patient2 not in nodes:
                                        net.add_node(patient2, label=patient2, color='#add8e6')
                                        nodes.add(patient2)
                                        
                                    net.add_edge(patient1, patient2, title=diagnosis)
                                
                                # 保存并显示网络图
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
                                    net.save_graph(f.name)
                                    with open(f.name, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    components.html(html_content, height=600)
                                    os.unlink(f.name)
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
                            # 使用PGQL查询用药关联
                            query = """
                            SELECT DISTINCT 
                                v1.entity_name AS patient1,
                                v2.entity_name AS patient2,
                                e1.relation_type AS medication_type
                            MATCH (v1) -[e1]-> () <-[e2]- (v2)
                            WHERE v1.entity_type = 'PATIENT'
                              AND v2.entity_type = 'PATIENT'
                              AND v1.entity_name = :patient_name
                              AND v1.entity_name != v2.entity_name
                              AND e1.relation_type = '用药记录'
                              AND e2.relation_type = e1.relation_type
                            """
                            results = graph_store.execute_pgql(query, {"patient_name": patient_name})
                            if results:
                                st.success(f"找到 {len(results)} 个用药关联")
                                for result in results:
                                    st.write(f"- {result['patient2']} 也使用了 '{result['medication']}'")
                            else:
                                st.info("未找到用药关联")
                    except Exception as e:
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
        SELECT v.entity_name, e.symptom
        FROM MATCH (v) -[e:HAS_SYMPTOM]-> ()
        WHERE v.entity_type = 'PATIENT'
        ```
        
        2. 查询特定症状的所有患者：
        ```sql
        SELECT v.entity_name
        FROM MATCH (v) -[e:HAS_SYMPTOM]-> ()
        WHERE v.entity_type = 'PATIENT'
        AND e.symptom = '发热'
        ```
        
        3. 查询患者的异常生化指标：
        ```sql
        SELECT v.entity_name, e.indicator_name, e.value
        FROM MATCH (v) -[e:HAS_INDICATOR]-> ()
        WHERE v.entity_type = 'PATIENT'
        AND e.reference_range = '异常'
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

def main():
    st.title("医疗文档智能检索系统")
    
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

if __name__ == "__main__":
    main()

