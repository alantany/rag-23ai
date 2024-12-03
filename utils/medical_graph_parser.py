import openai
from typing import Dict, Any, List
import logging
import json
from datetime import datetime
from .oracle_graph_store import OracleGraphStore
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class MedicalGraphParser:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.graph_store = OracleGraphStore()

    def _clean_json_content(self, content: str) -> str:
        """清理和格式化JSON内容"""
        try:
            # 移除可能的markdown代码块标记
            lines = content.strip().split('\n')
            start_idx = 0
            end_idx = len(lines)
            
            # 处理开头
            for i, line in enumerate(lines):
                if line.strip() in ['```json', '```']:
                    start_idx = i + 1
                    break
            
            # 处理结尾
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '```':
                    end_idx = i
                    break
            
            # 提取JSON内容
            json_lines = lines[start_idx:end_idx]
            cleaned_content = '\n'.join(json_lines).strip()
            
            # 尝试解析JSON，如果失败则尝试修复常见问题
            try:
                json.loads(cleaned_content)
                return cleaned_content
            except json.JSONDecodeError as e:
                logger.warning(f"初次JSON解析失败，尝试修复: {str(e)}")
                # 记录原始内容
                logger.debug(f"原始内容:\n{cleaned_content}")
                
                # 1. 替换单引号为双引号
                cleaned_content = cleaned_content.replace("'", '"')
                
                # 2. 确保属性名使用双引号
                import re
                cleaned_content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned_content)
                
                # 3. 移除尾部逗号
                cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
                
                # 4. 修复可能的Unicode转义
                try:
                    cleaned_content = cleaned_content.encode('utf-8').decode('utf-8')
                except UnicodeError:
                    try:
                        cleaned_content = cleaned_content.encode('latin1').decode('utf-8')
                    except UnicodeError:
                        cleaned_content = cleaned_content.encode('utf-8', errors='ignore').decode('utf-8')
                
                # 5. 修复空值
                cleaned_content = re.sub(r':\s*,', ': null,', cleaned_content)
                cleaned_content = re.sub(r':\s*\}', ': null}', cleaned_content)
                cleaned_content = re.sub(r':\s*\]', ': null]', cleaned_content)
                
                # 6. 修复多余的逗号
                cleaned_content = re.sub(r',\s*,', ',', cleaned_content)
                
                # 7. 修复缺失的引号
                cleaned_content = re.sub(r':\s*([^"{}\[\],\s][^,}\]]*?)(\s*[,}\]])', r': "\1"\2', cleaned_content)
                
                # 8. 修复中文属性名
                def add_quotes_to_chinese(match):
                    key = match.group(2)
                    if any('\u4e00-\u9fff' in char for char in key):
                        return f'{match.group(1)}"{key}"{match.group(3)}'
                    return match.group(0)
                
                cleaned_content = re.sub(r'([{,]\s*)([^"{\[,\s]+)(\s*:)', add_quotes_to_chinese, cleaned_content)
                
                # 记录修复后的内容
                logger.debug(f"修复后的内容:\n{cleaned_content}")
                
                try:
                    # 再次验证JSON
                    json.loads(cleaned_content)
                    return cleaned_content
                except json.JSONDecodeError as e:
                    logger.error(f"JSON修复失败: {str(e)}")
                    logger.error(f"修复后的内容:\n{cleaned_content}")
                    # 尝试使用更激进的修复方式
                    try:
                        # 使用正则表达式清理不可见字符
                        cleaned_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_content)
                        # 修复可能的换行问题
                        cleaned_content = re.sub(r'\n\s*', ' ', cleaned_content)
                        # 修复可能的多余空格
                        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
                        # 修复可能的错误引号
                        cleaned_content = re.sub(r'(?<!\\)"', '"', cleaned_content)
                        # 修复可能的错误转义
                        cleaned_content = re.sub(r'\\([^"\\])', r'\1', cleaned_content)
                        
                        # 尝试解析
                        data = json.loads(cleaned_content)
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except Exception as e:
                        logger.error(f"激进修复失败: {str(e)}")
                        raise
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {str(e)}")
            logger.error(f"原始内容:\n{content}")
            raise
        except Exception as e:
            logger.error(f"清理JSON内容失败: {str(e)}")
            raise

    def parse_to_graph(self, content: str, doc_reference: str) -> Dict[str, Any]:
        """将医疗文档解析为图数据结构"""
        try:
            # 使用GPT提取实体和关系
            prompt = f"""
            请仔细分析以下电子病历，提取关键实体和关系。为了确保数据质量，请遵循以下规则：
            1. 每个诊断项单独列出，不要合并
            2. 时间格式统一为YYYY-MM-DD，如果没有具体时间则填写"null"
            3. 所有数值都需要用引号包围
            4. 对于过长的内容，请只保留最重要的前3-5项
            5. 确保JSON格式完整，所有字符串都正确闭合
            
            请按以下JSON格式返回（注意：所有属性名和值都要用双引号包围）：
            
            {{
                "患者": {{
                    "姓名": "患者姓名",
                    "基本信息": {{
                        "性别": "性别值",
                        "年龄": "年龄值",
                        "民族": "民族值",
                        "职业": "职业值",
                        "婚姻状况": "婚姻状况值"
                    }},
                    "住院信息": {{
                        "入院日期": "YYYY-MM-DD",
                        "出院日期": "YYYY-MM-DD",
                        "住院天数": "数值",
                        "出院情况": "描述"
                    }}
                }},
                "主诉与诊断": [
                    {{
                        "类型": "主诉/入院诊断/出院诊断",
                        "内容": "具体内容",
                        "时间": "YYYY-MM-DD"
                    }}
                ],
                "现病史": [
                    {{
                        "症状": "症状名称",
                        "时间": "YYYY-MM-DD",
                        "描述": "具体描述"
                    }}
                ],
                "生命体征": [
                    {{
                        "指标": "体温/脉搏/呼吸/血压",
                        "数值": "具体值",
                        "时间": "YYYY-MM-DD"
                    }}
                ],
                "生化指标": [
                    {{
                        "项目": "检验项目名称",
                        "结果": "检验结果值",
                        "时间": "YYYY-MM-DD"
                    }}
                ]
            }}

            病历内容如下：
            {content}
            """

            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗数据分析助手，擅长从病历文本中提取关键信息。请确保返回的JSON格式完整，数据简洁。对于过长的列表，只保留最重要的3-5项。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000  # 限制返回的token数量
            )
            
            # 记录API响应内容
            logger.info(f"API响应: {response}")
            
            # 清理和解析JSON
            content = self._clean_json_content(response.choices[0].message.content)
            
            # 验证JSON格式
            try:
                result = json.loads(content)
                logger.info("JSON解析成功")
                
                # 验证必要字段
                required_fields = ["患者", "主诉与诊断", "现病史", "生命体征", "生化指标"]
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"缺少必要字段: {field}")
                        result[field] = []
                
                # 将解析结果存入图数据库
                self._store_graph_data(result, doc_reference)
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}")
                logger.error(f"JSON内容:\n{content}")
                raise
            
        except Exception as e:
            logger.error(f"解析医疗记录失败: {str(e)}")
            raise

    def _store_graph_data(self, data: Dict[str, Any], doc_reference: str):
        """将解析后的数据存入图数据库"""
        try:
            # 直接使用store_medical_data方法存储所有数据
            self.graph_store.store_medical_data(data, doc_reference)
            logger.info(f"成功存储文档 {doc_reference} 的图数据")
        except Exception as e:
            logger.error(f"存储图数据失败: {str(e)}")
            raise