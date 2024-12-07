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
        AND e.RELATION_TYPE = '入院诊断'
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
        query = """MATCH (v) -[e]-> (i)
        WHERE v.ENTITY_TYPE = '患者'
        AND v.ENTITY_NAME = :patient_name
        AND e.RELATION_TYPE = '生化指标'
        AND JSON_VALUE(i.ENTITY_VALUE, '$.reference_range') = '异常'
        COLUMNS (
            v.ENTITY_NAME AS patient,
            JSON_VALUE(i.ENTITY_VALUE, '$.名称') AS indicator,
            JSON_VALUE(i.ENTITY_VALUE, '$.数值') AS value,
            JSON_VALUE(i.ENTITY_VALUE, '$.单位') AS unit
        )"""
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})

    def find_similar_patients(self, symptom: str) -> List[Dict[str, Any]]:
        """查找有相同症状的患者"""
        query = """MATCH (v1) -[e1]-> (s), (v2) -[e2]-> (s)
        WHERE v1.ENTITY_TYPE = '患者'
        AND v2.ENTITY_TYPE = '患者'
        AND v1.ENTITY_NAME != v2.ENTITY_NAME
        AND e1.RELATION_TYPE = '现病史'
        AND e2.RELATION_TYPE = '现病史'
        AND JSON_VALUE(s.ENTITY_VALUE, '$.症状') = :symptom
        COLUMNS (
            v1.ENTITY_NAME AS patient1,
            v2.ENTITY_NAME AS patient2,
            JSON_VALUE(s.ENTITY_VALUE, '$.症状') AS common_symptom
        )"""
        return self.graph_store.execute_pgql(query, {'symptom': symptom})
        
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