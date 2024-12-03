import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import pdfplumber
from utils.oracle_json_store import OracleJsonStore
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置文档存储目录
UPLOAD_DIR = root_dir / "uploaded_documents"

def read_file_content(file_path):
    """读取文件内容，使用 pdfplumber 处理 PDF"""
    try:
        # 获取文件扩展名
        file_extension = str(file_path).lower().split('.')[-1]
        
        # PDF文件处理
        if file_extension == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        
        # 文本文件处理
        else:
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"文件读取失败: {str(e)}")
        return None

def update_document_content():
    """更新数据库中的文档原始内容"""
    try:
        # 确保目录存在
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        # 获取所有PDF文件
        pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.info("未找到PDF文件")
            return
            
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 遍历处理每个PDF文件
        for pdf_file in pdf_files:
            try:
                logger.info(f"正在处理文件: {pdf_file}")
                
                # 读取PDF内容
                content = read_file_content(pdf_file)
                if not content:
                    logger.error(f"无法读取文件内容: {pdf_file}")
                    continue
                    
                logger.info(f"成功读取文件内容，长度: {len(content)} 字符")
                
                # 获取相对路径
                rel_path = pdf_file.relative_to(root_dir)
                logger.info(f"文档相对路径: {rel_path}")
                
                # 更新数据库
                with OracleJsonStore() as json_store:
                    # 先检查文档是否存在
                    check_sql = """
                    SELECT COUNT(*) as count 
                    FROM DOCUMENT_JSON 
                    WHERE doc_info = :1
                    """
                    result = json_store.execute_search(check_sql, [str(rel_path)])
                    doc_exists = result[0]['count'] > 0 if result else False
                    
                    if doc_exists:
                        update_sql = """
                        UPDATE DOCUMENT_JSON 
                        SET content = :1
                        WHERE doc_info = :2
                        """
                        json_store.execute_sql(update_sql, [content, str(rel_path)])
                        logger.info(f"更新文件内容成功: {rel_path}")
                    else:
                        logger.warning(f"文档不存在于数据库中: {rel_path}")
                    
            except Exception as e:
                logger.error(f"处理文件失败 {pdf_file}: {str(e)}")
                continue
                
        logger.info("所有文件处理完成")
        
    except Exception as e:
        logger.error(f"更新文档内容失败: {str(e)}")

if __name__ == "__main__":
    logger.info("开始更新文档内容...")
    update_document_content()
    logger.info("更新完成") 