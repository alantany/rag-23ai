"""
Property Graph Manager

This module handles the creation and management of Oracle Property Graph.
"""

import os
import oracledb
from typing import Optional, Dict, Any
from ..db_utils import get_connection

class PropertyGraphManager:
    def __init__(self):
        """Initialize the Property Graph Manager."""
        self.graph_name = "medical_kg"
        
    def create_graph(self) -> bool:
        """
        Create the medical knowledge property graph.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create property graph
                    graph_sql = """
                    CREATE PROPERTY GRAPH medical_kg VERTEX TABLES (
                        medical_entities KEY (entity_id) LABEL entity PROPERTIES ALL COLUMNS
                    ) EDGE TABLES (
                        medical_relations KEY (relation_id)
                            SOURCE KEY (from_entity_id) REFERENCES medical_entities (entity_id)
                            DESTINATION KEY (to_entity_id) REFERENCES medical_entities (entity_id)
                            LABEL relation PROPERTIES ALL COLUMNS
                    )
                    """
                    cursor.execute(graph_sql)
                    return True
        except oracledb.Error as e:
            if "ORA-00955" in str(e):  # Object already exists
                return True
            print(f"Error creating property graph: {e}")
            return False
            
    def drop_graph(self) -> bool:
        """
        Drop the medical knowledge property graph.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"DROP PROPERTY GRAPH {self.graph_name}")
                    return True
        except oracledb.Error as e:
            print(f"Error dropping property graph: {e}")
            return False
            
    def check_graph_exists(self) -> bool:
        """
        Check if the property graph exists.
        
        Returns:
            bool: True if exists, False otherwise
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM USER_PROPERTY_GRAPHS 
                        WHERE GRAPH_NAME = :graph_name
                    """, graph_name=self.graph_name)
                    count = cursor.fetchone()[0]
                    return count > 0
        except oracledb.Error as e:
            print(f"Error checking property graph existence: {e}")
            return False
            
    def get_graph_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the property graph.
        
        Returns:
            Optional[Dict[str, Any]]: Graph information or None if error
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT * 
                        FROM USER_PROPERTY_GRAPHS 
                        WHERE GRAPH_NAME = :graph_name
                    """, graph_name=self.graph_name)
                    row = cursor.fetchone()
                    if row:
                        return {
                            "graph_name": row[0],
                            "graph_mode": row[1],
                            "all_tables": row[2],
                            "inmemory": row[3]
                        }
                    return None
        except oracledb.Error as e:
            print(f"Error getting property graph info: {e}")
            return None
