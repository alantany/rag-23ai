import oracledb
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json

# 加载环境变量
load_dotenv()
logger = logging.getLogger(__name__)

class OracleGraphStore:
    def __init__(self):
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        self.pool = None
        
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("请在.env文件中设置ORACLE_USER, ORACLE_PASSWORD和ORACLE_DSN")
        
        try:
            self.pool = oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=1,
                max=5,
                increment=1
            )
            logger.info("初始化OracleGraphStore成功")
        except Exception as e:
            logger.error(f"初始化OracleGraphStore失败: {str(e)}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            logger.info("关闭OracleGraphStore连接池")

    def execute_sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行SQL查询
        
        Args:
            query (str): SQL查询语句
            params (Optional[Dict]): 查询参数
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        try:
            with self.pool.acquire() as connection:
                cursor = connection.cursor()
                
                # 记录原始查询
                logger.info("执行SQL查询: %s", query)
                if params:
                    logger.info("参数: %s", params)
                
                # 执行SQL查询
                cursor.execute(query, params or {})
                
                # 获取列名
                columns = [col[0].lower() for col in cursor.description]
                
                # 处理结果
                results = []
                for row in cursor:
                    result = {}
                    for i, value in enumerate(row):
                        # 处理LOB对象
                        if isinstance(value, oracledb.LOB):
                            try:
                                result[columns[i]] = json.loads(value.read())
                            except json.JSONDecodeError:
                                result[columns[i]] = value.read()
                        else:
                            result[columns[i]] = value if value is not None else ''
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"执行SQL查询失败: {str(e)}")
            raise

    def execute_pgql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """execute_sql 的别名，用于兼容性"""
        return self.execute_sql(query, params)