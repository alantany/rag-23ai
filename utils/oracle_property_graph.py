import oracledb
import logging
from typing import Dict, Any, List
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class OraclePropertyGraph:
    def __init__(self, graph_store):
        self.graph_store = graph_store
        self.graph_name = 'MEDICAL_KG'
            
    def get_patient_symptoms(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的所有症状"""
        query = """MATCH (v) -[e]-> (s)
        WHERE v.ENTITY_TYPE = '患者' 
        AND e.RELATION_TYPE = '现病史'
        COLUMNS (
            v.ENTITY_NAME AS patient_name,
            JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS symptom
        )"""
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})

    def get_patient_diagnosis(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的所有诊断"""
        query = """MATCH (v) -[e]-> (d)
        WHERE v.ENTITY_TYPE = '患者'
        AND v.ENTITY_NAME = :patient_name
        AND e.RELATION_TYPE = '入���诊断'
        COLUMNS (
            v.ENTITY_NAME AS patient,
            JSON_VALUE(d.ENTITY_VALUE, '$.诊断') AS diagnosis
        )"""
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})
        
    def get_patient_vitals(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的生命体征"""
        query = """MATCH (v) -[e]-> (vt)
        WHERE v.ENTITY_TYPE = '患者'
        AND v.ENTITY_NAME = :patient_name
        AND e.RELATION_TYPE = '生命体征'
        COLUMNS (
            v.ENTITY_NAME AS patient,
            JSON_VALUE(vt.ENTITY_VALUE, '$.名称') AS vital_sign,
            JSON_VALUE(vt.ENTITY_VALUE, '$.数值') AS value,
            JSON_VALUE(vt.ENTITY_VALUE, '$.单位') AS unit
        )"""
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})
        
    def get_patient_abnormal_labs(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的异常生化指标"""
        query = """
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v)
            WHERE v.ENTITY_TYPE = '患者'
            AND v.ENTITY_NAME = :patient_name
            COLUMNS (
                v.ENTITY_NAME AS patient,
                v.ENTITY_VALUE AS patient_data
            )
        )"""
        
        results = self.graph_store.execute_pgql(query, {'patient_name': patient_name})
        
        if not results:
            return []
            
        # 使用JSON_TABLE处理结果
        sql = """
        SELECT 
            i.项目 AS indicator,
            i.结果 AS value,
            i.单位 AS unit,
            i.参考范围 AS reference
        FROM JSON_TABLE(
            :patient_data,
            '$.生化指标[*]' 
            COLUMNS (
                项目 PATH '$.项目',
                结果 PATH '$.结果',
                单位 PATH '$.单位',
                参考范围 PATH '$.参考范围'
            )
        ) i
        WHERE i.参考范围 = '异常'
        """
        
        # 从PGQL查询结果中获取patient_data
        patient_data = results[0].get('patient_data') if results else None
        
        if not patient_data:
            return []
            
        # 执行JSON查询
        with self.graph_store.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(sql, {'patient_data': patient_data})
            
            # 获列名
            columns = [col[0].lower() for col in cursor.description]
            
            # 处理结果
            abnormal_labs = []
            for row in cursor:
                result = {}
                for i, value in enumerate(row):
                    result[columns[i]] = value if value is not None else ''
                abnormal_labs.append(result)
                
            return abnormal_labs

    def find_similar_patients(self, patient_name: str) -> List[Dict[str, Any]]:
        """查找所有患者的症状信息"""
        query = """SELECT *
FROM GRAPH_TABLE ( MEDICAL_KG
    MATCH (v) -[e]-> (s)
    WHERE v.ENTITY_TYPE = '患者'
    AND e.RELATION_TYPE = '现病史'
    COLUMNS (
        v.ENTITY_NAME AS patient_name,
        JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS symptom
    )
)"""
        return self.graph_store.execute_pgql(query)
        
    def find_diagnosis_by_symptom(self, symptom: str) -> List[Dict[str, Any]]:
        """查找与特定症状相关的诊断"""
        query = """MATCH (v) -[e1]-> (s), (v) -[e2]-> (d)
        WHERE v.ENTITY_TYPE = '患者'
        AND e1.RELATION_TYPE = '现病史'
        AND e2.RELATION_TYPE = '入院诊断'
        AND JSON_VALUE(s.ENTITY_VALUE, '$.症状') = :symptom
        COLUMNS (
            v.ENTITY_NAME AS patient,
            JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS symptom,
            JSON_VALUE(d.ENTITY_VALUE, '$.诊断') AS diagnosis
        )"""
        return self.graph_store.execute_pgql(query, {'symptom': symptom})