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

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# 设置文档存储目录
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

# 设置JSON缓存目录
CACHE_DIR = Path("json_cache")
CACHE_DIR.mkdir(exist_ok=True)

# 使用 st.cache_resource 缓存模型，并隐藏��
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
    "入院诊断": ["诊断1", "诊断2"],
    "出院诊断": ["断1", "诊断2"],
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
                raise ValueError("GPT返回的数据格式不正确")

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

def parse_document_to_json(file_path):
    """解析文档为结构化JSON并生成SQL插入语句"""
    try:
        # 读取文件内容
        content = read_file_content(file_path)
        if not content:
            logger.error(f"无法读取文件内容: {file_path}")
            return None
        
        # 检查缓存
        cached_data = load_from_cache(file_path)
        if cached_data:
            logger.info("使用缓存的结构化数据")
            structured_data = cached_data
        else:
            # 使用解析器处理
            parser = MedicalRecordParser()
            doc_info = str(file_path)
            result = parser.parse_medical_record(content, doc_info)
            
            # 如果返回的是错误信息，返回 None
            if 'error' in result:
                logger.error(f"解析失败: {result['error']}")
                return None
            
            structured_data = result['structured_data']
            # 保存到缓存
            save_to_cache(file_path, structured_data)
        
        # 执行SQL插入
        with OracleJsonStore() as json_store:
            try:
                # 检查文档是否已存在
                check_sql = """
                SELECT COUNT(*) as count 
                FROM DOCUMENT_JSON 
                WHERE doc_info = :1
                """
                result = json_store.execute_search(check_sql, [str(file_path)])
                doc_exists = result[0]['count'] > 0 if result else False
                
                # 将数据转为JSON字符串
                doc_json_str = json.dumps(structured_data, ensure_ascii=False)
                
                if doc_exists:
                    # 更新现有文档
                    update_sql = """
                    UPDATE DOCUMENT_JSON 
                    SET doc_json = JSON(:1)
                    WHERE doc_info = :2
                    """
                    # 记录实际执行的SQL和参数
                    logger.info("执行更新SQL: %s", update_sql)
                    logger.info("参数1 (doc_json): %s", doc_json_str)
                    logger.info("参数2 (doc_info): %s", str(file_path))
                    
                    json_store.execute_sql(update_sql, [doc_json_str, str(file_path)])
                    logger.info("数据更新成功")
                else:
                    # 插入新文档
                    insert_sql = """
                    INSERT INTO DOCUMENT_JSON (doc_info, doc_json) 
                    VALUES (:1, JSON(:2))
                    """
                    # 记录实际执行的SQL和参数
                    logger.info("执行插入SQL: %s", insert_sql)
                    logger.info("参数1 (doc_info): %s", str(file_path))
                    logger.info("参数2 (doc_json): %s", doc_json_str)
                    
                    json_store.execute_sql(insert_sql, [str(file_path), doc_json_str])
                    logger.info("数据插入成功")
                
                return structured_data
                
            except Exception as e:
                logger.error(f"SQL行败: {str(e)}")
                return None
        
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
        
        # 返回所有检索结果，但只有相关的文��包含 GPT 分析
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
    "生命体征": {
        "体温": "包含单位",
        "血压": "包含单位"
    },
    "生化指标": {
        "天冬氨酸氨基转移酶": "值和单位",
        "丙氨酸氨基转移酶": "值和单位",
        "白细胞": "值和单位",
        "淋巴细胞百分比": "值和单位",
        "中性粒细胞百分比": "值和单位",
        "血红蛋白": "值和单位",
        "血小板": "值和单位"
    },
    "诊疗经过": "治疗过程描述",
    "出院医嘱": ["医嘱1", "医嘱2"]
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

Oracle 23c JSON查询特性：
1. 使用 JSON_EXISTS 进行条件匹配
2. 使用 JSON_VALUE 提取单个值
3. 使用 JSON_QUERY 提取JSON数组或对象
4. 支持点号访问嵌套属性
5. 支持��组索引访问
6. 支持条件过滤：?(@ == "value")
7. 支持键名匹配：
   - 使用 $.生化指标.* 遍历所有指标
   - 使用 EXISTS 和 OR 组合多个条件

你的任务是：
1. 理解用户的自然语言查询
2. 根据文档结构和Oracle 23c特性，生成最优的查询条件
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
    "fields": ["生化指标.天冬氨酸氨基转移酶", "生化指标.丙氨酸氨基转移酶"],
    "keywords": ["马某某", "转氨酶"]
}}

示例2：
输入："查看马某某的淋巴细胞比例"
输出：
{{
    "query_type": "检验结果",
    "conditions": [
        "JSON_EXISTS(doc_json, '$.患���姓名?(@ == \"马某某\")')",
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
        "生化指标.白细胞",
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
    """使用 GPT 将用户查询的标名称标准化"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
        
        messages = [
            {"role": "system", "content": """你是一个医疗指标名称标准化专家。
请将用户查询中的指标名称转换为标准的医疗指标名称。

规则：
1. 如果查询中包含某个检验指标的同义词或近义词，返回标准名称
2. 如果不确定，返回原始词语
3. 返回格式为 JSON：{"standard_term": "标准名称"}

示：
输入："淋巴细胞比例"
输出：{"standard_term": "淋巴细胞百分比"}

输入："白细胞计数"
输出：{"standard_term": "白细胞"}

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
        logger.error(f"指标名称标准化失败: {str(e)}")
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
        
        # 构建JSON查询条件
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
                # 如果GPT分析失败，继续尝试结构化数据
                
        # 如果没有从全文中找到信息，或GPT分析失败，尝试从结构化数据中获取
        info = []
        for field in fields:
            if '.' in field:  # 处理嵌套字段
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
        return "抱歉，处理您的问题时出现错误"

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

def main():
    st.title("医疗病历处理系统")
    
    # 显示初始化进度
    with st.spinner("正在连接数据库..."):
        try:
            # 初始化数据库
            init_database()
        except Exception as e:
            st.error(f"数据库连接失败: {str(e)}")
            st.info("请检查：\n1. 数据库服务是否启动\n2. 网络连接是否正常\n3. 数据库配置是否正确")
            return
    
    # 加载模型（使用缓存，不会每次都下载）
    with st.spinner("正在加载模型（如果是首次运行，可能需要下载）..."):
        try:
            embeddings_model = load_embeddings_model()
            st.success("系统初始化完成！")
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.info("如果是网络问题，您可以：\n1. 检查网络连接\n2. 等待一会儿再试\n3. 或者手动下载模型到 ~/.cache/torch/sentence_transformers")
            return
    
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
                            result = parse_document_to_json(file)
                            if result:
                                st.success(f"{file.name} 结构化完成")
                                with st.expander("查看结构化数据"):
                                    st.json(result)
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
                    if st.button("��构化", key=f"struct_{file.name}"):
                        with st.spinner("正在结构化..."):
                            result = parse_document_to_json(file)
                            if result:
                                st.success("结构化完成")
                                with st.expander("查看结构化数据"):
                                    st.json(result)
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
        query = st.text_input("输搜索内容")
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
                    st.warning("数据库未初始化，请先在文档管理中上传并结构化文档")
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
                        st.markdown(f"### 📄 {Path(doc['doc_info']).name}")
                        if isinstance(doc['doc_json'], dict):
                            data = doc['doc_json']
                            
                            # 使用tabs来组织内容
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
                                        "职业": data.get("职业", "未"),
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
                                    st.info("未找到文档全文内容，请运行 update_content.py 更新全文内容")
                        
                        st.markdown("---")
                else:
                    st.info("📭 数据库中暂无结构化文档，请先在文档管���中上传并结构化文档")
                
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

if __name__ == "__main__":
    main()

