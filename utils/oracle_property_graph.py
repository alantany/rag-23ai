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
        query = """
        SELECT DISTINCT v.entity_name as patient, e.symptom_name as symptom, e.onset_time as time
        FROM MATCH (v) -[e:HAS_SYMPTOM]-> (s)
        WHERE v.entity_type = '患者'
        AND v.entity_name = :patient_name
        """
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})

    def get_patient_diagnosis(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的所有诊断"""
        query = """
        SELECT DISTINCT v.entity_name as patient, e.diagnosis_name as diagnosis, e.diagnosis_type as type, e.diagnosis_time as time
        FROM MATCH (v) -[e:HAS_DIAGNOSIS]-> (d)
        WHERE v.entity_type = '患者'
        AND v.entity_name = :patient_name
        """
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})
        
    def get_patient_vitals(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的生命体征"""
        query = """
        SELECT DISTINCT v.entity_name as patient, e.vital_name as vital_sign, e.value as value, e.unit as unit, e.measure_time as time
        FROM MATCH (v) -[e:HAS_VITAL]-> (vt)
        WHERE v.entity_type = '患者'
        AND v.entity_name = :patient_name
        """
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})
        
    def get_patient_abnormal_labs(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的异常生化指标"""
        query = """
        SELECT DISTINCT v.entity_name as patient, e.indicator_name as indicator, e.value as value, e.unit as unit, e.reference_range as reference
        FROM MATCH (v) -[e:HAS_INDICATOR]-> (i)
        WHERE v.entity_type = '患者'
        AND v.entity_name = :patient_name
        AND e.reference_range = '异常'
        """
        return self.graph_store.execute_pgql(query, {'patient_name': patient_name})

    def find_similar_patients(self, symptom: str) -> List[Dict[str, Any]]:
        """查找有相同症状的患者"""
        query = """
        SELECT DISTINCT v1.entity_name as patient1, v2.entity_name as patient2, e1.symptom_name as common_symptom
        FROM MATCH (v1) -[e1:HAS_SYMPTOM]-> (s) <-[e2:HAS_SYMPTOM]- (v2)
        WHERE v1.entity_type = '患者'
        AND v2.entity_type = '患者'
        AND v1.entity_name != v2.entity_name
        AND e1.symptom_name = :symptom
        """
        return self.graph_store.execute_pgql(query, {'symptom': symptom})
        
    def find_diagnosis_by_symptom(self, symptom: str) -> List[Dict[str, Any]]:
        """查找与特定症状相关的诊断"""
        query = """
        SELECT DISTINCT v.entity_name as patient, e1.symptom_name as symptom, e2.diagnosis_name as diagnosis
        FROM MATCH (v) -[e1:HAS_SYMPTOM]-> (s), (v) -[e2:HAS_DIAGNOSIS]-> (d)
        WHERE v.entity_type = '患者'
        AND e1.symptom_name = :symptom
        """
        return self.graph_store.execute_pgql(query, {'symptom': symptom})