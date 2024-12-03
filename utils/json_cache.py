import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class JsonCache:
    def __init__(self, cache_dir: str = "json_cache"):
        """初始化 JSON 缓存管理器"""
        # 使用绝对路径
        self.cache_dir = Path(os.path.abspath(cache_dir))
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        logger.info(f"缓存目录: {self.cache_dir}")
        
    def save_json(self, file_name: str, data: Dict[str, Any]) -> Optional[str]:
        """保存 JSON 数据到缓存文件"""
        try:
            # 确保使用绝对路径
            cache_name = f"{Path(file_name).stem}.json"
            cache_path = self.cache_dir / cache_name
            
            logger.info(f"正在写入缓存文件: {cache_path}")
            
            # 直接写入新文件
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"缓存文件写入成功: {cache_path}")
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"保存JSON缓存失败: {str(e)}")
            return None
    
    def load_json(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """从缓存文件加载 JSON 数据"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"加载JSON缓存失败: {str(e)}")
            return None
    
    def list_cache_files(self) -> list:
        """列出所有缓存文件"""
        return list(self.cache_dir.glob("*.json"))
    
    def clear_cache(self, days: Optional[int] = None):
        """清理缓存文件
        
        Args:
            days: 如果指定，只清理超过指定天数的缓存
        """
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                if days is not None:
                    # 检查文件修改时间
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    age = (datetime.now() - file_time).days
                    if age <= days:
                        continue
                
                cache_file.unlink()
                logger.info(f"已删除缓存文件: {cache_file}")
                
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}") 