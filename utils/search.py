import json
from decimal import Decimal
from typing import Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """处理 Decimal 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def analyze_query_and_generate_sql(query: str, json_schema: Dict) -> Dict:
    """使用大模型分析查询并生成SQL"""
    try:
        prompt = f"""
        请分析以下查询，并生成合适的Oracle JSON查询语句。
        
        查询内容: {query}
        
        JSON数据结构:
        {json.dumps(json_schema, ensure_ascii=False, indent=2)}
        
        请返回以下JSON格式的结果:
        {{
            "analysis": {{
                "keywords": [],  # 查询关键词列表
                "fields": [],    # 相关字段列表
                "data_types": [] # 字段数据类型（number/text）
            }},
            "sql": "Oracle JSON查询语句",
            "explanation": "查询逻辑说明"
        }}
        
        注意：
        1. 对于数值类型字段，使用 JSON_VALUE 查询
        2. 对于文本类型字段，使用 JSON_EXISTS 查询
        3. 需要处理字段嵌套关系
        4. SQL中要包含 file_path, doc_content 和 relevance
        """
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "你是一个专业的数据库查询专家，擅长分析自然语言并生成Oracle JSON查询语句。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"GPT分析结果: {json.dumps(result, ensure_ascii=False)}")
        return result
        
    except Exception as e:
        logger.error(f"查询分析失败: {str(e)}")
        return None

def search_json_documents(query: str, top_k: int = 3):
    """搜索JSON文档"""
    try:
        # 获取JSON schema
        with OracleJsonStore() as json_store:
            schema = json_store.get_document_schema()
        
        # 使用大模型分析查询并生成SQL
        analysis_result = analyze_query_and_generate_sql(query, schema)
        if not analysis_result:
            return []
            
        # 执行查询
        with OracleJsonStore() as json_store:
            results = json_store.execute_search(
                analysis_result["sql"],
                {"top_k": top_k}
            )
            
            return results
            
    except Exception as e:
        logger.error(f"JSON文档搜索失败: {str(e)}")
        return [] 