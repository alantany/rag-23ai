import os
from dotenv import load_dotenv
import oracledb
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

def init_database():
    """初始化数据库，创建所需的表和索引"""
    # 获取数据库连接信息
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")
    
    if not all([user, password, dsn]):
        raise ValueError("请在.env文件中设置ORACLE_USER, ORACLE_PASSWORD和ORACLE_DSN")
    
    try:
        # 创建数据库连接
        connection = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn
        )
        
        with connection.cursor() as cursor:
            logger.info("开始初始化数据库...")
            
            # 1. 创建向量表
            logger.info("检查向量表...")
            try:
                cursor.execute("""
                    CREATE TABLE document_vectors (
                        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        file_path VARCHAR2(1000),
                        content CLOB,
                        vector VECTOR(384),
                        metadata CLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("创建向量表成功")
                
                # 创建向量索引
                cursor.execute("""
                    CREATE INDEX vector_idx ON document_vectors(vector) 
                    INDEXTYPE IS VECTOR_INDEXTYPE 
                    PARAMETERS('VECTOR_DIM=384 DISTANCE_METRIC=COSINE')
                """)
                logger.info("创建向量索引成功")
            except oracledb.DatabaseError as e:
                if "ORA-00955" in str(e):  # 表已存在
                    logger.info("向量表已存在")
                else:
                    raise
            
            # 2. 创建JSON文档表
            logger.info("检查JSON文档表...")
            try:
                cursor.execute("""
                    CREATE TABLE DOCUMENT_JSON (
                        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        doc_info VARCHAR2(500),
                        doc_json JSON,
                        content CLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("创建JSON文档表成功")
            except oracledb.DatabaseError as e:
                if "ORA-00955" in str(e):
                    logger.info("JSON文档表已存在")
                else:
                    raise
            
            # 3. 创建全文索引
            logger.info("检查全文索引...")
            try:
                cursor.execute("""
                BEGIN
                    ctx_ddl.create_preference('my_lexer', 'CHINESE_VGRAM_LEXER');
                END;
                """)
                cursor.execute("""
                CREATE INDEX DOC_CONTENT_IDX ON DOCUMENT_JSON(content)
                INDEXTYPE IS CTXSYS.CONTEXT
                PARAMETERS('
                    LEXER my_lexer
                    SYNC (ON COMMIT)
                ')
                """)
                logger.info("创建全文索引成功")
            except oracledb.DatabaseError as e:
                if "ORA-00955" in str(e) or "ORA-20000" in str(e):
                    logger.info("全文索引已存在")
                else:
                    raise
            
            # 4. 创建图数据表
            logger.info("检查图数据表...")
            
            # 创建实体表
            try:
                cursor.execute("""
                    CREATE TABLE medical_entities (
                        entity_id NUMBER GENERATED ALWAYS AS IDENTITY,
                        entity_type VARCHAR2(50),
                        entity_name VARCHAR2(200),
                        entity_value VARCHAR2(200),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (entity_id)
                    )
                """)
                logger.info("创建实体表成功")
            except oracledb.DatabaseError as e:
                if "ORA-00955" in str(e):
                    logger.info("实体表已存在")
                else:
                    raise
            
            # 创建关系表
            try:
                cursor.execute("""
                    CREATE TABLE medical_relations (
                        relation_id NUMBER GENERATED ALWAYS AS IDENTITY,
                        from_entity_id NUMBER,
                        to_entity_id NUMBER,
                        relation_type VARCHAR2(50),
                        relation_time DATE,
                        doc_reference VARCHAR2(200),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (relation_id),
                        FOREIGN KEY (from_entity_id) REFERENCES medical_entities(entity_id),
                        FOREIGN KEY (to_entity_id) REFERENCES medical_entities(entity_id)
                    )
                """)
                logger.info("创建关系表成功")
            except oracledb.DatabaseError as e:
                if "ORA-00955" in str(e):
                    logger.info("关系表已存在")
                else:
                    raise
            
            connection.commit()
            logger.info("数据库初始化完成")
            
    except Exception as e:
        logger.error(f"初始化数据库失败: {str(e)}")
        raise
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    init_database() 