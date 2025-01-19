import oracledb
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
from utils.medical_record_parser import MedicalRecordParser
import time

# 加载环境变量
load_dotenv()
logger = logging.getLogger(__name__)

class OracleJsonStore:
    def __init__(self):
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("请在.env文件中设置ORACLE_USER, ORACLE_PASSWORD和ORACLE_DSN")
        
        try:
            self.pool = oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=1,
                max=5,
                increment=1,
                getmode=oracledb.POOL_GETMODE_WAIT,
                wait_timeout=10,
                timeout=60,
                retry_count=3,
                retry_delay=2
            )
            logger.info("成功初始化OracleJsonStore连接池")
            
            # 初始化数据库表结构
            self.init_schema()
            
        except Exception as e:
            logger.error(f"初始化连接池失败: {str(e)}")
            raise

    def get_connection(self):
        """获取数据库连接，带重试机制"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                return self.pool.acquire()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"获取数据库连接失败: {str(e)}")
                    raise
                logger.warning(f"获取连接失败，第{attempt + 1}次重试")
                time.sleep(retry_delay)

    def init_schema(self):
        """初始化数据库表结构，保持向量化搜索表不变"""
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                # 检查表是否存在
                check_sql = """
                    SELECT COUNT(*) 
                    FROM user_tables 
                    WHERE table_name = 'DOCUMENT_JSON'
                """
                cursor.execute(check_sql)
                table_exists = cursor.fetchone()[0] > 0
                
                if not table_exists:
                    # 创建表，使用 JSON 类型
                    create_table_sql = """
                        CREATE TABLE DOCUMENT_JSON (
                            id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            doc_info VARCHAR2(1000),
                            doc_json JSON,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    cursor.execute(create_table_sql)
                    connection.commit()
                    logger.info("创建 DOCUMENT_JSON 表完成")

    def add_document(self, doc_info: str, doc_json: Dict):
        """添加文档到JSON存储，不影响向量化存储"""
        try:
            with self.pool.acquire() as connection:
                with connection.cursor() as cursor:
                    insert_sql = """
                        INSERT INTO DOCUMENT_JSON (doc_info, doc_json)
                        VALUES (:1, JSON(:2))
                    """
                    cursor.execute(insert_sql, [
                        doc_info,
                        json.dumps(doc_json, ensure_ascii=False)
                    ])
                    connection.commit()
                    logger.info(f"文档 {doc_info} 添加完成")
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索文档
        1. 从问题中提取患者姓名
        2. 根据患者姓名查找对应的文档
        3. 返回文档内容供后续处理
        """
        try:
            # 预定义的患者名字列表
            patient_names = ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"]
            
            # 从问题中识别患者姓名
            target_name = None
            for name in patient_names:
                if name in query:
                    target_name = name
                    break
            
            if not target_name:
                logger.warning(f"未从问题中识别出患者姓名: {query}")
                return []
            
            logger.info(f"从问题中识别出患者: {target_name}")
            
            # 根据患者姓名查找文档
            with self.pool.acquire() as connection:
                with connection.cursor() as cursor:
                    search_sql = """
                        SELECT d.doc_info, d.doc_json
                        FROM DOCUMENT_JSON d
                        WHERE d.doc_info LIKE :1
                        ORDER BY d.created_at DESC
                        FETCH FIRST 1 ROWS ONLY
                    """
                    doc_pattern = f"{target_name}.pdf"
                    logger.info(f"搜索文档: {doc_pattern}")
                    cursor.execute(search_sql, [doc_pattern])
                    
                    results = []
                    for row in cursor:
                        results.append({
                            'doc_info': row[0],
                            'doc_json': json.loads(row[1].read()) if hasattr(row[1], 'read') else row[1],
                            'query': query  # 保存原始问题，供后续生成答案使用
                        })
                    
                    logger.info(f"搜索完成，找到 {len(results)} 个结果")
                    return results
                
        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}")
            return []

    def close(self):
        """关闭连接池"""
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute_sql(self, sql: str, params: List[Any] = None) -> None:
        """执行SQL语句"""
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)
                    connection.commit()
                    logger.debug(f"SQL执行成功: {sql[:100]}...")
        except Exception as e:
            logger.error(f"SQL执行失败: {str(e)}")
            raise

    def execute_search(self, search_sql: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """执行JSON搜索查询，不影响向量化搜索功能"""
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    if params:
                        cursor.execute(search_sql, params)
                    else:
                        cursor.execute(search_sql)
                    
                    columns = [col[0].lower() for col in cursor.description]
                    results = []
                    
                    for row in cursor:
                        result = {}
                        for i, value in enumerate(row):
                            column_name = columns[i]
                            
                            # 处理JSON类型的数据
                            if column_name == 'doc_json' and value is not None:
                                if isinstance(value, str):
                                    result[column_name] = json.loads(value)
                                elif hasattr(value, 'read'):
                                    result[column_name] = json.loads(value.read())
                                else:
                                    result[column_name] = value
                            else:
                                result[column_name] = value
                        
                        results.append(result)
                    
                    return results
        except Exception as e:
            logger.error(f"查询执行失败: {str(e)}")
            raise 