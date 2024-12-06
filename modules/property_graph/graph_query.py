"""
属性图查询模块
"""

from utils.oracle_graph_store import OracleGraphStore
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class PropertyGraphQuery:
    def __init__(self):
        """初始化属性图查询类"""
        self.graph_store = OracleGraphStore()
        
    def execute_graph_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行图查询
        
        Args:
            query (str): PGQL查询语句
            params (Optional[Dict]): 查询参数
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        try:
            # 执行PGQL查询
            results = self.graph_store.execute_pgql(query, params)
            return results
        except Exception as e:
            logger.error(f"执行图查询失败: {str(e)}")
            raise
