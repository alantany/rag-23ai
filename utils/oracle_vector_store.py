import oracledb
import numpy as np
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import json
import logging

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class OracleVectorStore:
    def __init__(self, vector_dimension: int = 384):
        # 从环境变量获取数据库连接信息
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD") 
        self.dsn = os.getenv("ORACLE_DSN")
        self.vector_dimension = vector_dimension
        
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("请在.env文件中设置ORACLE_USER, ORACLE_PASSWORD和ORACLE_DSN")
        
        # 初始化数据库连接
        self.pool = oracledb.create_pool(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            min=1,
            max=5,
            increment=1
        )
        logger.info(f"初始化OracleVectorStore, 向量维度: {vector_dimension}")

    def init_schema(self):
        """创建必要的表"""
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                logger.info("检查表是否存在")
                check_sql = """
                    SELECT COUNT(*) 
                    FROM user_tables 
                    WHERE table_name = 'DOCUMENT_VECTORS'
                """
                logger.debug(f"执行SQL: {check_sql}")
                cursor.execute(check_sql)
                table_exists = cursor.fetchone()[0] > 0
                logger.info(f"表是否存在: {table_exists}")
                
                if not table_exists:
                    logger.info("创建表")
                    create_table_sql = f"""
                        CREATE TABLE document_vectors (
                            id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            file_path VARCHAR2(1000),
                            content CLOB,
                            vector VECTOR({self.vector_dimension}),
                            metadata CLOB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    logger.debug(f"执行SQL: {create_table_sql}")
                    cursor.execute(create_table_sql)
                    connection.commit()
                    logger.info("表创建完成")

    def batch_add_vectors(self, vectors: List[np.ndarray], documents: List[Dict[str, Any]], batch_size: int = 1000):
        """批量添加向量和文档"""
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                for i in range(0, len(vectors), batch_size):
                    batch_vectors = vectors[i:i + batch_size]
                    batch_docs = documents[i:i + batch_size]
                    
                    # 准备批量插入数据
                    data = []
                    for vector, doc in zip(batch_vectors, batch_docs):
                        if vector.shape[0] != self.vector_dimension:
                            raise ValueError(f"向量维度必须为{self.vector_dimension}")
                        
                        vector_str = '[' + ','.join(map(str, vector.tolist())) + ']'
                        data.append([
                            doc['file_path'],
                            doc['content'],
                            vector_str,
                            json.dumps(doc.get('metadata', {}), ensure_ascii=False)
                        ])
                    
                    # 执行批量插入
                    cursor.executemany("""
                        INSERT INTO document_vectors (file_path, content, vector, metadata)
                        VALUES (:1, :2, :3, :4)
                    """, data)
                    
                    connection.commit()

    def add_vectors(self, vectors: List[np.ndarray], documents: List[Dict[str, Any]]):
        """添加向量和文档到数据库"""
        logger.info(f"添加向量和文档: {len(vectors)}个")
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                for vector, doc in zip(vectors, documents):
                    vector_str = '[' + ','.join(map(str, vector.tolist())) + ']'
                    insert_sql = """
                        INSERT INTO document_vectors (file_path, content, vector, metadata)
                        VALUES (:1, :2, :3, :4)
                    """
                    logger.debug(f"执行SQL: {insert_sql}")
                    logger.debug(f"文件路径: {doc['file_path']}")
                    logger.debug(f"内容长度: {len(doc['content'])}")
                    cursor.execute(insert_sql, [
                        doc['file_path'],
                        doc['content'],
                        vector_str,
                        json.dumps(doc.get('metadata', {}))
                    ])
                connection.commit()
                logger.info("向量和文档添加完成")

    def search_vectors(self, query_vectors: List[np.ndarray], top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索与查询向量最相似的文档"""
        results = []
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                for query_vector in query_vectors:
                    # 确保 query_vector 是一个 NumPy 数组
                    if isinstance(query_vector, list):
                        query_vector = np.array(query_vector)
                    
                    query_vector_str = '[' + ','.join(map(str, query_vector.tolist())) + ']'
                    
                    search_sql = """
                        SELECT file_path, content, vector, 
                               (1 - (vector <=> :1)) AS similarity
                        FROM document_vectors
                        ORDER BY similarity DESC
                        FETCH FIRST :2 ROWS ONLY
                    """
                    cursor.execute(search_sql, [query_vector_str, top_k])
                    
                    for row in cursor:
                        results.append({
                            'file_path': row[0],
                            'content': row[1],
                            'similarity': row[3]
                        })
        return results

    def delete_document(self, file_path: str):
        """删除指定文档的所有向量"""
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                # 先查询是否存在该文件
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM document_vectors 
                    WHERE file_path = :1
                """, [file_path])
                
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # 存在则删除
                    cursor.execute("""
                        DELETE FROM document_vectors
                        WHERE file_path = :1
                    """, [file_path])
                    connection.commit()
                    return True
                return False

    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT file_path) as total_documents,
                        MIN(created_at) as earliest_document,
                        MAX(created_at) as latest_document
                    FROM document_vectors
                """)
                row = cursor.fetchone()
                return {
                    'total_chunks': row[0],
                    'total_documents': row[1],
                    'earliest_document': row[2],
                    'latest_document': row[3]
                }

    def close(self):
        """关闭数据库连接池"""
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def list_documents(self) -> List[Dict[str, Any]]:
        """获取所有文档列表"""
        logger.info("获取文档列表")
        with self.pool.acquire() as connection:
            with connection.cursor() as cursor:
                # 查询不同的文件及其块数
                sql = """
                    SELECT 
                        file_path,
                        COUNT(*) as chunk_count
                    FROM document_vectors 
                    GROUP BY file_path
                    ORDER BY file_path
                """
                logger.debug(f"执行SQL: {sql}")
                cursor.execute(sql)
                
                documents = []
                for row in cursor:
                    doc = {
                        'file_path': row[0],
                        'chunk_count': row[1]
                    }
                    logger.info(f"找到文档: {doc}")
                    documents.append(doc)
                
                logger.info(f"找到{len(documents)}个不同的文档")
                return documents