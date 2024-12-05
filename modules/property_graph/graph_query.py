"""
Property Graph Query

This module provides query functionality for Oracle Property Graph.
"""

import oracledb
from typing import List, Dict, Any, Optional
from ..db_utils import get_connection
import logging

logger = logging.getLogger(__name__)

class PropertyGraphQuery:
    def __init__(self):
        """Initialize the Property Graph Query."""
        self.graph_name = "medical_kg"

    def _process_lob(self, value):
        """处理LOB对象"""
        if isinstance(value, oracledb.LOB):
            return value.read()
        return value

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询结果行"""
        return {k: self._process_lob(v) for k, v in row.items()}

    def execute_graph_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a graph query using GRAPH_TABLE.
        
        Args:
            query (str): The graph query to execute
            params (Optional[Dict]): Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    # 如果查询中有未替换的参数占位符，但没有提供参数，返回空列表
                    if ':' in query and not params:
                        logger.warning("Query contains parameters but no values provided")
                        return []
                        
                    # 处理参数
                    bind_params = {}
                    if params:
                        for key, value in params.items():
                            # 移除字符串值的引号(因为 execute 会自动处理)
                            if isinstance(value, str) and value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            bind_params[key] = value
                            
                    logger.info(f"执行查询: {query}")
                    logger.info(f"参数: {bind_params}")
                        
                    # 执行查询
                    cursor.execute(query, bind_params)
                    
                    # 获取列名
                    columns = [col[0].lower() for col in cursor.description]
                    
                    # 处理结果
                    results = []
                    for row in cursor:
                        # 将None值转换为更友好的显示
                        row_values = [self._process_lob(v) if v is not None else '' for v in row]
                        results.append(dict(zip(columns, row_values)))
                    
                    logger.info(f"查询返回 {len(results)} 条结果")
                    return results
                    
        except oracledb.Error as e:
            logger.error(f"Error executing graph query: {e}")
            return []
            
    def get_relation_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics about relation types.
        
        Returns:
            List[Dict[str, Any]]: Relation type statistics
        """
        query = """
        SELECT relation_type, COUNT(*) as count
        FROM GRAPH_TABLE (medical_kg
            MATCH
            (v IS entity) -[e IS relation]-> (v2 IS entity)
            COLUMNS (
                e.relation_type AS relation_type
            )
        )
        GROUP BY relation_type
        ORDER BY count DESC
        """
        return self.execute_graph_query(query)

    def get_patient_symptoms(self, patient_name: str) -> List[Dict[str, Any]]:
        """
        Get all symptoms for a specific patient.
        
        Args:
            patient_name (str): Name of the patient
            
        Returns:
            List[Dict[str, Any]]: Patient's symptoms
        """
        query = """
        SELECT * FROM GRAPH_TABLE (medical_kg
            MATCH
            (v IS entity WHERE v.entity_name = :patient_name) 
                -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
            COLUMNS (
                v.entity_name AS patient_name,
                v2.entity_value AS symptoms
            )
        )
        """
        return self.execute_graph_query(query, {"patient_name": patient_name})
