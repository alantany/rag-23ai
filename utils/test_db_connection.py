"""
Test Database Connection

This script tests the database connection and Property Graph functionality.
"""

import oracledb
import os
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """测试数据库连接"""
    # 加载环境变量
    load_dotenv()
    
    # 获取数据库配置
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")
    
    logger.info(f"测试连接到 Oracle 数据库:")
    logger.info(f"用户: {user}")
    logger.info(f"DSN: {dsn}")
    
    try:
        # 尝试建立连接
        connection = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn
        )
        
        logger.info("连接成功!")
        
        # 测试查询
        with connection.cursor() as cursor:
            cursor.execute("SELECT SYSDATE FROM DUAL")
            result = cursor.fetchone()
            logger.info(f"当前数据库时间: {result[0]}")
            
            # 测试图数据表是否存在
            cursor.execute("""
                SELECT COUNT(*) 
                FROM user_tables 
                WHERE table_name IN ('MEDICAL_ENTITIES', 'MEDICAL_RELATIONS')
            """)
            table_count = cursor.fetchone()[0]
            logger.info(f"找到 {table_count}/2 个图数据表")
            
            if table_count == 2:
                # 测试图数据
                cursor.execute("SELECT COUNT(*) FROM medical_entities")
                entity_count = cursor.fetchone()[0]
                logger.info(f"实体数量: {entity_count}")
                
                cursor.execute("SELECT COUNT(*) FROM medical_relations")
                relation_count = cursor.fetchone()[0]
                logger.info(f"关系数量: {relation_count}")
            
        connection.close()
        logger.info("数据库连接测试完成")
        return True
        
    except Exception as e:
        logger.error(f"连接失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()