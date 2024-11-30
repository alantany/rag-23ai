import openai
from typing import Dict, Any
import logging
import json
import re

logger = logging.getLogger(__name__)

class MedicalRecordParser:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
            base_url="https://api.chatanywhere.tech/v1"
        )

    def parse_medical_record(self, content: str) -> Dict[str, Any]:
        """使用LLM解析医疗记录为结构化JSON"""
        try:
            prompt = f"""
            请仔细分析以下电子病历，提取关键信息并转换为结构化的JSON格式。
            
            需要提取的信息必须包含以下字段（如果信息不存在，请标注"未提供"）：
            
            {{
                "basic_information": {{
                    "name": "患者姓名",
                    "age": "年龄",
                    "gender": "性别",
                    "date_of_birth": "出生日期",
                    "occupation": "职业"
                }},
                "chief_complaint": "主诉",
                "present_illness_history": "现病史",
                "past_medical_history": "既往史",
                "physical_examination": "体格检查",
                "diagnosis": "诊断结果",
                "treatment_plan": "治疗方案"
            }}

            病历内容如下：
            {content}

            请严格按照上述JSON格式返回，确保每个字段都有值，没有的信息填写"未提供"。
            只返回JSON格式，不要有其他说明文字。
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗数据分析助手，擅长将病历文本转换为结构化的JSON格式。请只返回JSON数据，不要有任何其他文字。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # 如果解析失败，尝试从文本中提取JSON部分
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("无法从响应中提取有效的JSON")
            
            return result
            
        except Exception as e:
            logger.error(f"解析医疗记录失败: {str(e)}")
            return {"error": str(e)} 