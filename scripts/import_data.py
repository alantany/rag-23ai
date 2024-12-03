import os
import logging
import oracledb
from datetime import datetime
from typing import List
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseImporter:
    def __init__(self):
        """初始化数据库导入器"""
        self.db_config = {
            'user': os.getenv('ORACLE_USER'),
            'password': os.getenv('ORACLE_PASSWORD'),
            'dsn': os.getenv('ORACLE_DSN')
        }
        self.import_dir = "db_export"
        logger.info(f"数据库连接信息: dsn={self.db_config['dsn']}, user={self.db_config['user']}")

    def connect(self) -> oracledb.Connection:
        """创建数据库连接"""
        return oracledb.connect(
            user=self.db_config['user'],
            password=self.db_config['password'],
            dsn=self.db_config['dsn']
        )

    def get_sql_files(self) -> List[str]:
        """获取所有SQL文件并按正确顺序排序"""
        if not os.path.exists(self.import_dir):
            logger.error(f"导入目录不存在: {self.import_dir}")
            return []

        # 获取所有SQL文件
        sql_files = [f for f in os.listdir(self.import_dir) if f.endswith('.sql')]
        
        # 定义表的导入顺序
        table_order = {
            'document_vectors': 1,
            'document_json': 2,
            'medical_entities': 3,
            'medical_relations': 4
        }

        # 按表名和时间戳排序
        def sort_key(filename):
            # 从文件名中提取表名和时间戳
            parts = filename.split('_')
            table_name = '_'.join(parts[:-1])  # 表名可能包含下划线
            timestamp = parts[-1].replace('.sql', '')
            # 返回排序元组：(表顺序, 时间戳)
            return (table_order.get(table_name, 999), timestamp)

        return sorted(sql_files, key=sort_key)

    def execute_sql_file(self, sql_file: str) -> bool:
        """执行单个SQL文件"""
        file_path = os.path.join(self.import_dir, sql_file)
        logger.info(f"正在执行SQL文件: {sql_file}")

        try:
            # 读取SQL文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            # 分割SQL语句（按分号分割，但忽略CLOB内容中的分号）
            statements = []
            current_statement = []
            in_clob = False
            
            for line in sql_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('--'):
                    continue

                # 检查是否进入或退出CLOB内容
                if 'TO_CLOB' in line:
                    in_clob = True
                
                if in_clob:
                    current_statement.append(line)
                    if line.endswith("');"):
                        in_clob = False
                        if not any(keyword in ''.join(current_statement).upper() for keyword in ['CREATE', 'INSERT', 'DROP', 'ALTER']):
                            current_statement = []
                            continue
                        statements.append('\n'.join(current_statement))
                        current_statement = []
                else:
                    if ';' in line:
                        current_statement.append(line)
                        if not any(keyword in ''.join(current_statement).upper() for keyword in ['CREATE', 'INSERT', 'DROP', 'ALTER']):
                            current_statement = []
                            continue
                        statements.append('\n'.join(current_statement))
                        current_statement = []
                    else:
                        current_statement.append(line)

            # 执行SQL语句
            with self.connect() as connection:
                cursor = connection.cursor()
                
                for statement in statements:
                    if not statement.strip():
                        continue
                    
                    try:
                        logger.debug(f"执行SQL: {statement[:200]}...")  # 只显示前200个字符
                        cursor.execute(statement)
                        
                    except Exception as e:
                        logger.error(f"执行SQL语句失败: {str(e)}")
                        logger.error(f"问题SQL: {statement}")
                        raise

                connection.commit()
                logger.info(f"成功执行SQL文件: {sql_file}")
                return True

        except Exception as e:
            logger.error(f"处理SQL文件失败: {str(e)}")
            return False

    def import_all(self):
        """导入所有SQL文件"""
        sql_files = self.get_sql_files()
        if not sql_files:
            logger.error("没有找到SQL文件")
            return

        logger.info(f"找到以下SQL文件: {sql_files}")
        
        success_count = 0
        for sql_file in sql_files:
            try:
                if self.execute_sql_file(sql_file):
                    success_count += 1
            except Exception as e:
                logger.error(f"导入文件 {sql_file} 失败: {str(e)}")
                continue

        logger.info(f"导入完成: 成功 {success_count}/{len(sql_files)} 个文件")

if __name__ == "__main__":
    try:
        importer = DatabaseImporter()
        importer.import_all()
    except Exception as e:
        logger.error(f"导入过程中发生错误: {str(e)}") 