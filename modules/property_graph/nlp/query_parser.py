"""
Property Graph Query Parser

This module converts natural language queries into Property Graph queries.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv

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
            
            # 复杂查询模板
            "patients_with_symptom": """
                SELECT DISTINCT v.entity_name AS patient_name, v2.entity_value AS symptom
                FROM GRAPH_TABLE (medical_kg
                    MATCH
                    (v IS entity) -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
                    WHERE v2.entity_value LIKE '%:symptom%'
                    COLUMNS (
                        v.entity_name,
                        v2.entity_value
                    )
                )
            """,
            "similar_symptoms_patients": """
                WITH patient_symptoms AS (
                    SELECT v2.entity_value AS symptom
                    FROM GRAPH_TABLE (medical_kg
                        MATCH
                        (v IS entity WHERE v.entity_name = :patient_name) 
                            -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
                        COLUMNS (
                            v2.entity_value AS symptom
                        )
                    )
                )
                SELECT DISTINCT v.entity_name AS similar_patient
                FROM GRAPH_TABLE (medical_kg
                    MATCH
                    (v IS entity) -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
                    WHERE v2.entity_value IN (SELECT symptom FROM patient_symptoms)
                        AND v.entity_name != :patient_name
                    COLUMNS (
                        v.entity_name
                    )
                )
            """
        }
        
        # 定义关键词到模板的映射
        self.keyword_templates = {
            "所有信息": "patient_info",
            "症状": "patient_symptoms",
            "诊疗经过": "treatment_process",
            "统计": "common_symptoms",
            "常见": "common_symptoms",
            "有相同症状": "similar_symptoms_patients",
            "查询有": "patients_with_symptom"
        }

    def extract_params(self, query: str, template_name: str) -> Dict[str, Any]:
        """
        Extract parameters from the query based on template.
        
        Args:
            query (str): Natural language query
            template_name (str): Name of the query template
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        params = {}
        
        # 提取患者姓名
        if template_name in ["patient_info", "patient_symptoms", "treatment_process", "similar_symptoms_patients"]:
            patient_match = re.search(r'([张李王赵马蒲周杨刘]某某)', query)
            if patient_match:
                params["patient_name"] = patient_match.group(1)
        
        # 提取症状
        if template_name == "patients_with_symptom":
            symptom_match = re.search(r'有(.+?)症状', query)
            if symptom_match:
                params["symptom"] = symptom_match.group(1)
        
        return params

    def parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse natural language query into a graph query.
        
        Args:
            query (str): Natural language query
            
        Returns:
            Tuple[str, Dict[str, Any]]: Query template name and parameters
        """
        try:
            # 首先尝试基于关键词匹配
            template_name = None
            for keyword, template in self.keyword_templates.items():
                if keyword in query:
                    template_name = template
                    break
            
            # 如果找到了模板，提取参数
            if template_name:
                params = self.extract_params(query, template_name)
                return template_name, params
            
            # 如果关键词匹配失败，使用OpenAI
            prompt = f"""
            请将以下自然语言查询转换为图数据库查询。
            
            可用的查询模板有：
            1. patient_info - 查询患者的所有信息
            2. patient_symptoms - 查询患者的症状
            3. treatment_process - 查询患者的诊疗经过
            4. common_symptoms - 查询常见症状统计
            5. patients_with_symptom - 查询具有特定症状的患者
            6. similar_symptoms_patients - 查询具有相似症状的患者
            
            自然语言查询：{query}
            
            请以JSON格式返回：
            {{
                "template": "模板名称",
                "params": {{
                    "参数名": "参数值"
                }},
                "explanation": "选择原因解释"
            }}
            """
            
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专门将自然语言转换为图数据库查询的AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # 解析返回结果
            result = json.loads(response.choices[0].message.content)
            
            # 验证模板是否存在
            if result["template"] not in self.query_templates:
                raise ValueError(f"Unknown template: {result['template']}")
                
            return result["template"], result["params"]
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            # 根据查询内容选择默认模板
            if "症状" in query:
                if "统计" in query or "常见" in query:
                    return "common_symptoms", {}
                elif "相同" in query or "相似" in query:
                    return "similar_symptoms_patients", self.extract_params(query, "similar_symptoms_patients")
                else:
                    return "patient_symptoms", self.extract_params(query, "patient_symptoms")
            elif "诊疗" in query or "治疗" in query:
                return "treatment_process", self.extract_params(query, "treatment_process")
            else:
                return "patient_info", self.extract_params(query, "patient_info")
            
    def get_graph_query(self, template_name: str, params: Dict[str, Any]) -> str:
        """
        Get the actual graph query based on template and parameters.
        
        Args:
            template_name (str): Name of the query template
            params (Dict[str, Any]): Query parameters
            
        Returns:
            str: The actual graph query
        """
        if template_name not in self.query_templates:
            raise ValueError(f"Unknown template: {template_name}")
            
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
        Process natural language query and return graph query.
        
        Args:
            query (str): Natural language query
            
        Returns:
            Tuple[str, Dict[str, Any], str]: Template name, parameters, and final query
        """
        template_name, params = self.parse_query(query)
        graph_query = self.get_graph_query(template_name, params)
        return template_name, params, graph_query