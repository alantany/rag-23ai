import logging
from typing import Dict, Any, List
from utils.oracle_graph_store import OracleGraphStore

logger = logging.getLogger(__name__)

class OraclePropertyGraph:
    def __init__(self):
        self.graph_store = OracleGraphStore()

    def execute_pgql(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """execute_sql 的别名，用于兼容性"""
        return self.graph_store.execute_sql(query, params)

    def find_similar_patients(self, patient_name: str) -> List[Dict[str, Any]]:
        """查找所有患者的症状信息"""
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
        logger.info("查询字符串: %s", query)
        logger.info("参数: %s", {'patient_name': patient_name})
        result = self.execute_pgql(query, {'patient_name': patient_name})
        logger.info("查询结果: %r", result)
        return result

    def get_patient_symptoms(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的所有症状"""
        query = """
        SELECT 
            e1.ENTITY_NAME AS patient_name,
            e2.ENTITY_VALUE AS symptom
        FROM MEDICAL_ENTITIES e1
        JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
        JOIN MEDICAL_ENTITIES e2 ON e2.ENTITY_ID = r.TO_ENTITY_ID
        WHERE e1.ENTITY_TYPE = '患者'
        AND e1.ENTITY_NAME = :patient_name
        AND r.RELATION_TYPE = '现病史'
        """
        return self.execute_pgql(query, {'patient_name': patient_name})

    def get_patient_diagnosis(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的所有诊断"""
        query = """
        SELECT 
            e1.ENTITY_NAME AS patient,
            e2.ENTITY_VALUE AS diagnosis
        FROM MEDICAL_ENTITIES e1
        JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
        JOIN MEDICAL_ENTITIES e2 ON e2.ENTITY_ID = r.TO_ENTITY_ID
        WHERE e1.ENTITY_TYPE = '患者'
        AND e1.ENTITY_NAME = :patient_name
        AND r.RELATION_TYPE = '入院诊断'
        """
        return self.execute_pgql(query, {'patient_name': patient_name})

    def get_patient_abnormal_labs(self, patient_name: str) -> List[Dict[str, Any]]:
        """获取患者的异常生化指标"""
        query = """
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (v) -[e]-> (i)
            WHERE v.ENTITY_TYPE = 'PATIENT'
            AND v.ENTITY_NAME = :patient_name
            AND e.RELATION_TYPE = 'HAS_INDICATOR'
            AND i.REFERENCE_RANGE = '异常'
            COLUMNS (
                v.ENTITY_NAME AS patient,
                i.INDICATOR_NAME AS indicator,
                i.VALUE AS value,
                i.UNIT AS unit,
                i.REFERENCE_RANGE AS reference
            )
        )
        """
        logger.info("查询字符串: %s", query)
        logger.info("参数: %s", {'patient_name': patient_name})
        result = self.execute_pgql(query, {'patient_name': patient_name})
        logger.info("查询结果: %r", result)
        return result