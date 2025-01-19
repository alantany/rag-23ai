import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JsonCache:
    """处理JSON文件的缓存"""
    
    def __init__(self):
        """初始化缓存目录"""
        self.cache_dir = Path("json_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_json(self, cache_path: str) -> dict:
        """从缓存加载JSON数据"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载缓存JSON失败: {str(e)}")
            return None
    
    def save_json(self, cache_path: str, data: dict) -> bool:
        """保存JSON数据到缓存"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存JSON到缓存失败: {str(e)}")
            return False
    
    def get_cache_path(self, file_name: str) -> str:
        """获取缓存文件路径"""
        return str(self.cache_dir / f"{Path(file_name).stem}.json") 