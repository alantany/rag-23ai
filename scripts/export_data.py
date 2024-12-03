import os
import json
import logging
import oracledb
from datetime import datetime
from typing import List, Dict, Any
import base64
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseExporter:
    def __init__(self):
        """初始化数据库导出器"""
        self.db_config = {
            'user': os.getenv('ORACLE_USER'),
            'password': os.getenv('ORACLE_PASSWORD'),
            'dsn': os.getenv('ORACLE_DSN')
        }
        self.export_dir = "db_export"
        os.makedirs(self.export_dir, exist_ok=True)
        
        logger.info(f"数据库连接信息: dsn={self.db_config['dsn']}, user={self.db_config['user']}")

    def connect(self) -> oracledb.Connection:
        """创建数据库连接"""
        return oracledb.connect(
            user=self.db_config['user'],
            password=self.db_config['password'],
            dsn=self.db_config['dsn']
        )

    def export_table_data(self, table_name: str) -> List[Dict[str, Any]]:
        """导出表数据"""
        try:
            with self.connect() as connection:
                cursor = connection.cursor()
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [col[0].lower() for col in cursor.description]
                rows = []
                
                for row in cursor:
                    row_dict = {}
                    for i, value in enumerate(row):
                        if isinstance(value, oracledb.LOB):
                            value = value.read()
                        elif isinstance(value, datetime):
                            value = value.strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(value, bytes):
                            value = base64.b64encode(value).decode('utf-8')
                        row_dict[columns[i]] = value
                    rows.append(row_dict)
                
                return rows
        except Exception as e:
            logger.error(f"导出表 {table_name} 数据失败: {str(e)}")
            raise

    def generate_insert_statements(self, table_name: str, data: List[Dict[str, Any]]) -> str:
        """生成INSERT语句"""
        if not data:
            return ""

        statements = []
        for row in data:
            columns = list(row.keys())
            values = []
            for col in columns:
                value = row[col]
                if value is None:
                    values.append('NULL')
                elif isinstance(value, str):
                    # 处理CLOB和JSON类型
                    if col.lower() in ['doc_json', 'entity_value', 'content', 'metadata']:
                        values.append(f"""TO_CLOB('{value.replace("'", "''")}')""")
                    else:
                        values.append(f"""'{value.replace("'", "''")}'""")
                elif isinstance(value, (int, float)):
                    values.append(str(value))
                else:
                    values.append(f"""'{str(value)}'""")

            stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
            statements.append(stmt)

        return '\n'.join(statements)

    def export_table_to_file(self, table_name: str, timestamp: str):
        """为单个表生成SQL文件，包含建表语句和数据"""
        export_file = os.path.join(self.export_dir, f'{table_name.lower()}_{timestamp}.sql')
        
        # 表结构定义
        table_ddl = {
            "DOCUMENT_VECTORS": """
-- 创建向量表
CREATE TABLE DOCUMENT_VECTORS (
    ID NUMBER NOT NULL PRIMARY KEY,
    FILE_PATH VARCHAR2(1000),
    CONTENT CLOB,
    VECTOR VECTOR(384, *),
    METADATA CLOB,
    CREATED_AT TIMESTAMP(6)
);

-- 创建向量索引
CREATE INDEX vector_idx ON DOCUMENT_VECTORS(VECTOR) 
INDEXTYPE IS VECTOR_INDEXTYPE 
PARAMETERS('VECTOR_CONFIG=(DIMENSION(384), DISTANCE_METRIC(COSINE))');
""",
            "DOCUMENT_JSON": """
-- 创建JSON文档表
CREATE TABLE DOCUMENT_JSON (
    ID NUMBER NOT NULL PRIMARY KEY,
    DOC_INFO VARCHAR2(500),
    DOC_JSON JSON,
    CONTENT CLOB
);

-- 创建JSON索引
CREATE INDEX json_content_idx ON DOCUMENT_JSON(CONTENT)
INDEXTYPE IS CTXSYS.CONTEXT;
""",
            "MEDICAL_ENTITIES": """
-- 创建实体表
CREATE TABLE MEDICAL_ENTITIES (
    ENTITY_ID NUMBER NOT NULL PRIMARY KEY,
    ENTITY_TYPE VARCHAR2(100),
    ENTITY_NAME VARCHAR2(1000),
    ENTITY_VALUE CLOB,
    DOC_REF VARCHAR2(1000),
    CREATED_AT VARCHAR2(100)
);

-- 创建实体索引
CREATE INDEX entity_type_idx ON MEDICAL_ENTITIES(ENTITY_TYPE);
CREATE INDEX entity_name_idx ON MEDICAL_ENTITIES(ENTITY_NAME);
CREATE INDEX entity_doc_ref_idx ON MEDICAL_ENTITIES(DOC_REF);
""",
            "MEDICAL_RELATIONS": """
-- 创建关系表
CREATE TABLE MEDICAL_RELATIONS (
    RELATION_ID NUMBER NOT NULL PRIMARY KEY,
    FROM_ENTITY_ID NUMBER,
    TO_ENTITY_ID NUMBER,
    RELATION_TYPE VARCHAR2(100),
    DOC_REFERENCE VARCHAR2(1000),
    CREATED_AT VARCHAR2(100),
    FOREIGN KEY (FROM_ENTITY_ID) REFERENCES MEDICAL_ENTITIES(ENTITY_ID),
    FOREIGN KEY (TO_ENTITY_ID) REFERENCES MEDICAL_ENTITIES(ENTITY_ID)
);

-- 创建关系索引
CREATE INDEX relation_type_idx ON MEDICAL_RELATIONS(RELATION_TYPE);
CREATE INDEX relation_doc_ref_idx ON MEDICAL_RELATIONS(DOC_REFERENCE);
"""
        }

        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                # 写入文件头
                f.write(f"-- {table_name} 表导出脚本\n")
                f.write(f"-- 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-- 此脚本包含表结构、索引和数据\n\n")
                
                # 写入表结构和索引
                f.write(table_ddl[table_name])
                
                # 导出表数据
                logger.info(f"正在导出表 {table_name} 的数据")
                data = self.export_table_data(table_name)
                if data:
                    f.write(f"\n-- 插入 {table_name} 表数据\n")
                    f.write(self.generate_insert_statements(table_name, data))
                    f.write("\n")
                logger.info(f"成功导出表 {table_name} 的 {len(data)} 条记录")
                
                # 写入文件尾
                f.write("\nCOMMIT;\n")
                logger.info(f"导出完成，文件保存在: {export_file}")
                
        except Exception as e:
            logger.error(f"导出表 {table_name} 失败: {str(e)}")
            raise

    def export_all(self):
        """导出所有表的结构和数据"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 按顺序导出每个表（考虑外键依赖关系）
        tables = [
            "DOCUMENT_VECTORS",
            "DOCUMENT_JSON",
            "MEDICAL_ENTITIES",  # MEDICAL_RELATIONS 依赖此表
            "MEDICAL_RELATIONS"
        ]

        for table_name in tables:
            try:
                self.export_table_to_file(table_name, timestamp)
            except Exception as e:
                logger.error(f"导出表 {table_name} 失败: {str(e)}")
                continue

if __name__ == "__main__":
    try:
        exporter = DatabaseExporter()
        exporter.export_all()
    except Exception as e:
        logger.error(f"导出过程中发生错误: {str(e)}") 