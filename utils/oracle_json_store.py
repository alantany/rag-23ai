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
        """创建必要的表结构，使用 JSON 数据类型"""
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
                        CREATE TABLE document_json (
                            id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            file_path VARCHAR2(1000),
                            doc_content JSON,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    cursor.execute(create_table_sql)
                    connection.commit()
                    logger.info("创建 document_json 表完成")

    def add_document(self, file_path: str, content: str, structured_data: Dict = None):
        """添加文档到JSON存储"""
        logger.info(f"添加文档到JSON存储: {file_path}")
        
        if structured_data is None:
            # 如果没有提供结构化数据，使用解析器处理
            parser = MedicalRecordParser()
            structured_data = parser.parse_medical_record(content)
        
        # 添加文件信息和时间戳
        doc_json = {
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "structured_data": structured_data
        }
        
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                insert_sql = """
                    INSERT INTO document_json (file_path, doc_content)
                    VALUES (:1, JSON(:2))
                """
                cursor.execute(insert_sql, [
                    file_path,
                    json.dumps(doc_json, ensure_ascii=False)
                ])
                connection.commit()
                logger.info(f"文档 {file_path} 添加完成")

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索文档"""
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                # 使用 JSON_EXISTS 和 JSON_TEXTCONTAINS 进行搜索
                search_sql = """
                    SELECT d.file_path,
                           d.doc_content,
                           1 as relevance
                    FROM document_json d
                    WHERE JSON_TEXTCONTAINS(d.doc_content, '$.structured_data.*', :1)
                    FETCH FIRST :2 ROWS ONLY
                """
                cursor.execute(search_sql, [query, top_k])
                
                results = []
                for row in cursor:
                    results.append({
                        'file_path': row[0],
                        'content': json.loads(row[1].read()),
                        'relevance': row[2]
                    })
                
                return results

    def close(self):
        """关闭连接池"""
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute_search(self, search_sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        """执行JSON搜索查询"""
        with self.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(search_sql, params)
                
                results = []
                for row in cursor:
                    # 处理 JSON 数据
                    doc_content = row[1]
                    if isinstance(doc_content, str):
                        content = json.loads(doc_content)
                    elif hasattr(doc_content, 'read'):
                        content = json.loads(doc_content.read())
                    else:
                        content = doc_content  # 已经是字典类型
                    
                    results.append({
                        'file_path': row[0],
                        'content': content,
                        'relevance': row[2] if len(row) > 2 else 1.0
                    })
                
                return results 