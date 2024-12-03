import click
import logging
from pathlib import Path
from utils.oracle_json_store import OracleJsonStore
from utils.json_cache import JsonCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.argument('json_files', nargs=-1, type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='强制覆盖已存在的文档')
def import_json(json_files, force):
    """从JSON文件导入文档到数据库
    
    支持指定多个JSON文件路径，例如：
    python import_json.py cache/doc1.json cache/doc2.json
    """
    json_cache = JsonCache()
    json_store = OracleJsonStore()
    
    for json_file in json_files:
        try:
            logger.info(f"正在导入: {json_file}")
            
            # 加载JSON数据
            data = json_cache.load_json(json_file)
            if not data:
                logger.error(f"无法加载JSON文件: {json_file}")
                continue
            
            # 获取原始文件路径
            file_path = Path(json_file).stem.split('_')[0]  # 移除时间戳部分
            
            # 检查文档是否存在
            if json_store.check_document_exists(file_path) and not force:
                logger.warning(f"文档已存在且未指定强制覆盖: {file_path}")
                continue
            
            # 添加到数据库
            if json_store.add_document(file_path, "", structured_data=data):
                logger.info(f"成功导入: {file_path}")
            else:
                logger.error(f"导入失败: {file_path}")
                
        except Exception as e:
            logger.error(f"处理文件时出错 {json_file}: {str(e)}")

if __name__ == '__main__':
    import_json() 