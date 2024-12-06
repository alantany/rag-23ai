"""
Property Graph Query Parser

This module converts natural language queries into Property Graph queries.
"""

import json
import os
from typing import Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class GraphQueryParser:
    def __init__(self):
        """Initialize the Graph Query Parser."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # 定义查询模板
        self.query_templates = {
            # 基本查询模板
            "patient_info": """
                SELECT * FROM GRAPH_TABLE (medical_kg
                    MATCH
                    (v IS entity WHERE v.entity_name = :patient_name) -[e IS relation]-> (v2 IS entity)
                    COLUMNS (
                        e.relation_type AS record_type,
                        v2.entity_type AS info_type,
                        v2.entity_value AS detail
                    )
                )
            """,
            "patient_symptoms": """
                SELECT * FROM GRAPH_TABLE (medical_kg
                    MATCH
                    (v IS entity WHERE v.entity_name = :patient_name) 
                        -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
                    COLUMNS (
                        v.entity_name AS patient_name,
                        v2.entity_value AS symptoms
                    )
                )
            """,
            "treatment_process": """
                SELECT * FROM GRAPH_TABLE (medical_kg
                    MATCH
                    (v IS entity WHERE v.entity_name = :patient_name) 
                        -[e IS relation WHERE e.relation_type = '诊疗经过']-> (v2 IS entity)
                    COLUMNS (
                        v.entity_name AS patient_name,
                        v2.entity_value AS treatment_process
                    )
                )
            """,
            
            # 统计类查询模板
            "common_symptoms": """
                SELECT v2.entity_value AS symptom, COUNT(*) as count
                FROM GRAPH_TABLE (medical_kg
                    MATCH
                    (v IS entity) -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
                    COLUMNS (
                        v2.entity_value AS symptom
                    )
                )
                GROUP BY v2.entity_value
                ORDER BY count DESC
            """,
            
            # 医疗数据关联性分析模板
            "medical_data_correlation": """
                WITH 
                -- 获取病史数据
                history AS (
                    SELECT v2.entity_value AS symptom
                    FROM GRAPH_TABLE (medical_kg
                        MATCH (v IS entity WHERE v.entity_name = :patient_name)
                            -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
                        COLUMNS (v2.entity_value)
                    )
                ),
                -- 获取诊断数据
                diagnosis AS (
                    SELECT v2.entity_value AS diagnosis
                    FROM GRAPH_TABLE (medical_kg
                        MATCH (v IS entity WHERE v.entity_name = :patient_name)
                            -[e IS relation WHERE e.relation_type IN ('入院诊断', '出院诊断')]-> (v2 IS entity)
                        COLUMNS (v2.entity_value)
                    )
                ),
                -- 获取生命体征
                vitals AS (
                    SELECT v2.entity_value AS vital_sign
                    FROM GRAPH_TABLE (medical_kg
                        MATCH (v IS entity WHERE v.entity_name = :patient_name)
                            -[e IS relation WHERE e.relation_type = '生命体征']-> (v2 IS entity)
                        COLUMNS (v2.entity_value)
                    )
                ),
                -- 获取生化指标
                labs AS (
                    SELECT v2.entity_value AS lab_result
                    FROM GRAPH_TABLE (medical_kg
                        MATCH (v IS entity WHERE v.entity_name = :patient_name)
                            -[e IS relation WHERE e.relation_type = '生化指标']-> (v2 IS entity)
                        COLUMNS (v2.entity_value)
                    )
                )
                -- 组合所有数据
                SELECT 
                    :patient_name AS patient_name,
                    h.symptom,
                    d.diagnosis,
                    v.vital_sign,
                    l.lab_result
                FROM history h
                FULL OUTER JOIN diagnosis d ON 1=1
                FULL OUTER JOIN vitals v ON 1=1
                FULL OUTER JOIN labs l ON 1=1
                WHERE 
                    CASE 
                        WHEN :data_type1 = '现病史' AND :data_type2 = '诊断' THEN 
                            h.symptom IS NOT NULL AND d.diagnosis IS NOT NULL
                        WHEN :data_type1 = '现病史' AND :data_type2 = '生命体征' THEN 
                            h.symptom IS NOT NULL AND v.vital_sign IS NOT NULL
                        WHEN :data_type1 = '现病史' AND :data_type2 = '生化指标' THEN 
                            h.symptom IS NOT NULL AND l.lab_result IS NOT NULL
                        WHEN :data_type1 = '诊断' AND :data_type2 = '生命体征' THEN 
                            d.diagnosis IS NOT NULL AND v.vital_sign IS NOT NULL
                        WHEN :data_type1 = '诊断' AND :data_type2 = '生化指标' THEN 
                            d.diagnosis IS NOT NULL AND l.lab_result IS NOT NULL
                        WHEN :data_type1 = '生命体征' AND :data_type2 = '生化指标' THEN 
                            v.vital_sign IS NOT NULL AND l.lab_result IS NOT NULL
                        ELSE 1=1
                    END
            """
        }

    def parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        使用大模型解析自然语言查询意图。
        
        Args:
            query (str): 自然语言查询
            
        Returns:
            Tuple[str, Dict[str, Any]]: 查询模板名称和参数
        """
        try:
            # 从查询中提取患者姓名
            patient_name = None
            for name in ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"]:
                if name in query:
                    patient_name = name
                    break
                    
            if not patient_name:
                logger.warning("未能从查询中识别出患者姓名")
                return "patient_info", {}
                
            # 构建提示词
            prompt = f"""
            请分析以下医疗查询,返回查询意图和参数。请确保返回JSON格式的结果。

            查询文本: {query}

            可用的查询模板:
            1. patient_info - 查询患者的所有信息
            2. patient_symptoms - 查询患者的症状
            3. treatment_process - 查询患者的诊疗经过
            4. common_symptoms - 查询常见症状统计
            5. medical_data_correlation - 查询医疗数据关联性分析

            如果是查询数据关联性,请返回:
            {{
                "template": "medical_data_correlation",
                "params": {{
                    "patient_name": "{patient_name}",
                    "data_type1": "现病史/诊断/生命体征/生化指标",
                    "data_type2": "现病史/诊断/生命体征/生化指标"
                }}
            }}

            如果是其他查询,请返回:
            {{
                "template": "模板名称",
                "params": {{
                    "patient_name": "{patient_name}"
                }}
            }}
            """
            
            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专门解析医疗查询意图的AI助手。请确保返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # 解析返回结果
            result = json.loads(response.choices[0].message.content)
            logger.info(f"OpenAI API返回: {json.dumps(result, ensure_ascii=False)}")
            
            if "template" not in result or "params" not in result:
                raise ValueError("API返回格式错误")
                
            # 验证模板名称
            template = result["template"]
            if template not in self.query_templates:
                logger.warning(f"未知的查询模板: {template}")
                template = "patient_info"
                
            # 验证参数
            params = result["params"]
            if not isinstance(params, dict):
                logger.warning("参数格式错误")
                params = {"patient_name": patient_name}
                
            # 确保患者姓名参数存在
            if "patient_name" not in params:
                params["patient_name"] = patient_name
                
            # 对于数据关联性分析,验证数据类型参数
            if template == "medical_data_correlation":
                valid_types = ["现病史", "诊断", "生命体征", "生化指标"]
                if "data_type1" not in params or params["data_type1"] not in valid_types:
                    params["data_type1"] = "现病史"
                if "data_type2" not in params or params["data_type2"] not in valid_types:
                    params["data_type2"] = "诊断"
                    
            return template, params
            
        except Exception as e:
            logger.error(f"解析查询失败: {str(e)}")
            return "patient_info", {"patient_name": patient_name if patient_name else "未知患者"}

    def get_graph_query(self, template_name: str, params: Dict[str, Any]) -> str:
        """
        根据模板和参数生成图查询语句。
        
        Args:
            template_name (str): 查询模板名称
            params (Dict[str, Any]): 查询参数
            
        Returns:
            str: 图查询语句
        """
        if template_name not in self.query_templates:
            raise ValueError(f"未知的查询模板: {template_name}")
            
        query_template = self.query_templates[template_name]
        
        # 替换参数
        query = query_template
        for param_name, param_value in params.items():
            placeholder = f":{param_name}"
            if isinstance(param_value, str):
                param_value = f"'{param_value}'"
            query = query.replace(placeholder, str(param_value))
            
        return query.strip()

    def process_query(self, query: str) -> Tuple[str, Dict[str, Any], str]:
        """
        处理自然语言查询。
        
        Args:
            query (str): 自然语言查询
            
        Returns:
            Tuple[str, Dict[str, Any], str]: 模板名称、参数和最终查询语句
        """
        template_name, params = self.parse_query(query)
        graph_query = self.get_graph_query(template_name, params)
        return template_name, params, graph_query