"""
Oracle图数据存储类
"""

import oracledb
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json

# 加载环境变量
load_dotenv()
logger = logging.getLogger(__name__)

class OracleGraphStore:
    def __init__(self):
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        self.connection = None
        
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("请在.env文件中设置ORACLE_USER, ORACLE_PASSWORD和ORACLE_DSN")
        
        try:
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            logger.info("初始化OracleGraphStore成功")
        except Exception as e:
            logger.error(f"初始化OracleGraphStore失败: {str(e)}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            logger.info("关闭OracleGraphStore连接")

    def execute_sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行SQL查询
        
        Args:
            query (str): SQL查询语句
            params (Optional[Dict]): 查询参数
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        try:
            cursor = self.connection.cursor()
            
            # 记录原始查询
            logger.info("执行SQL查询: %s", query)
            if params:
                logger.info("参数: %s", params)
            
            # 执行SQL查询
            cursor.execute(query, params or {})
            
            # 获取列名
            columns = [col[0].lower() for col in cursor.description]
            
            # 处理结果
            results = []
            for row in cursor:
                result = {}
                for i, value in enumerate(row):
                    # 处理LOB对象
                    if isinstance(value, oracledb.LOB):
                        try:
                            result[columns[i]] = json.loads(value.read())
                        except json.JSONDecodeError:
                            result[columns[i]] = value.read()
                    else:
                        result[columns[i]] = value if value is not None else ''
                results.append(result)
            
            return results
                
        except Exception as e:
            logger.error(f"执行SQL查询失败: {str(e)}")
            raise

    def execute_pgql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """execute_sql 的别名，用于兼容性"""
        return self.execute_sql(query, params)

    def get_patients(self) -> List[Dict[str, Any]]:
        """获取所有患者列表"""
        try:
            # 获取所有患者实体
            sql = """
                SELECT 
                    e.ENTITY_NAME,
                    e.DOC_REF,
                    e.ENTITY_VALUE
                FROM MEDICAL_ENTITIES e
                WHERE e.ENTITY_TYPE = '患者'
            """
            logger.info(f"执行患者查询SQL: {sql}")
            results = self.execute_sql(sql)
            
            if not results:
                logger.info("没有找到任何患者数据")
                return []
            
            processed_results = []
            for row in results:
                try:
                    # 解析患者信息
                    entity_value = row['entity_value']
                    if isinstance(entity_value, str):
                        patient_info = json.loads(entity_value)
                    else:
                        patient_info = entity_value
                    
                    logger.info(f"解析的患者信息: {patient_info}")
                    
                    # 构建患者数据
                    patient_data = {
                        "姓名": patient_info.get("患者姓名", row['entity_name']),
                        "文档": row['doc_ref'],
                        "基本信息": {
                            "姓名": patient_info.get("患者姓名", "未知"),
                            "性别": patient_info.get("性别", "未知"),
                            "年龄": patient_info.get("年龄", "未知"),
                            "入院日期": patient_info.get("入院日期", "未知"),
                            "出院日期": patient_info.get("出院日期", "未知")
                        },
                        "主诉": patient_info.get("主诉", "未知"),
                        "现病史": patient_info.get("现病史", []),
                        "入院诊断": patient_info.get("入院诊断", []),
                        "出院诊断": patient_info.get("出院诊断", []),
                        "生命体征": patient_info.get("生命体征", {}),
                        "生化指标": patient_info.get("生化指标", {}),
                        "诊疗经过": patient_info.get("诊疗经过", ""),
                        "出院医嘱": patient_info.get("出院医嘱", []),
                        "metadata": patient_info.get("metadata", {})
                    }
                    
                    logger.info(f"处理后的患者数据: {patient_data}")
                    processed_results.append(patient_data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"解析患者信息失败: {str(e)}, 原始数据: {row['entity_value']}")
                    # 即使解析失败，也添加基本信息
                    processed_results.append({
                        "姓名": row['entity_name'],
                        "文档": row['doc_ref'],
                        "基本信息": {},
                        "主诉": "未知",
                        "现病史": [],
                        "入院诊断": [],
                        "出院诊断": [],
                        "生命体征": {},
                        "生化指标": {},
                        "诊疗经过": "",
                        "出院医嘱": [],
                        "metadata": {}
                    })
                except Exception as e:
                    logger.error(f"处理患者数据时出错: {str(e)}, 跳过此条记录")
                    continue
            
            logger.info(f"最终返回的患者数量: {len(processed_results)}")
            return processed_results
                
        except Exception as e:
            logger.error(f"获取患者列表失败: {str(e)}")
            return []

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """获取所有患者列表（get_patients 的别名）"""
        return self.get_patients()

    def get_patient_info(self, patient_name: str) -> Dict[str, Any]:
        """获取指定患者的详细信息"""
        try:
            # 获取患者实体信息
            sql = """
                SELECT 
                    e.ENTITY_NAME,
                    e.DOC_REF,
                    e.ENTITY_VALUE
                FROM MEDICAL_ENTITIES e
                WHERE e.ENTITY_TYPE = '患者'
                AND e.ENTITY_NAME = :1
            """
            logger.info(f"执行患者查询SQL: {sql}")
            results = self.execute_sql(sql, [patient_name])
            
            if not results:
                logger.info(f"未找到患者: {patient_name}")
                return {}
            
            row = results[0]
            try:
                # 解析患者信息
                entity_value = row['entity_value']
                if isinstance(entity_value, str):
                    patient_info = json.loads(entity_value)
                else:
                    patient_info = entity_value
                
                logger.info(f"解析的患者信息: {patient_info}")
                
                # 返回完整的患者信息
                return {
                    "姓名": row['entity_name'],
                    "文档": row['doc_ref'],
                    **patient_info  # 展开完整的患者信息
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"解析患者信息失败: {str(e)}, 原始数据: {row['entity_value']}")
                return {
                    "姓名": row['entity_name'],
                    "文档": row['doc_ref']
                }
            except Exception as e:
                logger.error(f"处理患者数据时出错: {str(e)}")
                return {
                    "姓名": row['entity_name'],
                    "文档": row['doc_ref']
                }
                
        except Exception as e:
            logger.error(f"获取患者信息失败: {str(e)}")
            return {}