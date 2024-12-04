"""
Property Graph Query

This module provides query functionality for Oracle Property Graph.
"""

import oracledb
from typing import List, Dict, Any, Optional
from ..db_utils import get_connection

class PropertyGraphQuery:
    def __init__(self):
        """Initialize the Property Graph Query."""
        self.graph_name = "medical_kg"

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
                    cursor.execute(query, params or {})
                    columns = [col[0] for col in cursor.description]
                    results = []
                    for row in cursor:
                        results.append(dict(zip(columns, row)))
                    return results
        except oracledb.Error as e:
            print(f"Error executing graph query: {e}")
            return []

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """
        Get all entity relations in the graph.
        
        Returns:
            List[Dict[str, Any]]: List of relations
        """
        query = """
        SELECT * FROM GRAPH_TABLE (medical_kg
            MATCH
            (v IS entity) -[e IS relation]-> (v2 IS entity)
            COLUMNS (
                v.entity_name AS source_name,
                v.entity_type AS source_type,
                e.relation_type AS relation,
                v2.entity_name AS target_name
            )
        )
        """
        return self.execute_graph_query(query)

    def get_patient_relations(self, patient_name: str) -> List[Dict[str, Any]]:
        """
        Get all relations for a specific patient.
        
        Args:
            patient_name (str): Name of the patient
            
        Returns:
            List[Dict[str, Any]]: Patient's relations
        """
        query = """
        SELECT * FROM GRAPH_TABLE (medical_kg
            MATCH
            (v IS entity WHERE v.entity_name = :patient_name) -[e IS relation]-> (v2 IS entity)
            COLUMNS (
                e.relation_type AS record_type,
                v2.entity_type AS info_type,
                v2.entity_value AS detail
            )
        )
        """
        return self.execute_graph_query(query, {"patient_name": patient_name})

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

    def get_treatment_process(self, patient_name: str) -> List[Dict[str, Any]]:
        """
        Get treatment process for a specific patient.
        
        Args:
            patient_name (str): Name of the patient
            
        Returns:
            List[Dict[str, Any]]: Patient's treatment process
        """
        query = """
        SELECT * FROM GRAPH_TABLE (medical_kg
            MATCH
            (v IS entity WHERE v.entity_name = :patient_name) 
                -[e IS relation WHERE e.relation_type = '诊疗经过']-> (v2 IS entity)
            COLUMNS (
                v.entity_name AS patient_name,
                v2.entity_value AS treatment_process
            )
        )
        """
        return self.execute_graph_query(query, {"patient_name": patient_name})
