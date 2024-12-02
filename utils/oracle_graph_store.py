import oracledb
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import time
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
        self.pool = None
        
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("请在.env文件中设置ORACLE_USER, ORACLE_PASSWORD和ORACLE_DSN")
        
        try:
            self.pool = oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=1,
                max=5,
                increment=1,
                getmode=oracledb.POOL_GETMODE_WAIT,
                wait_timeout=10,
                timeout=60,
                retry_count=3,
                retry_delay=2,
                encoding='UTF-8',
                nencoding='UTF-8'
            )
            logger.info("初始化OracleGraphStore成功")
            self.initialize_tables()
        except Exception as e:
            logger.error(f"初始化OracleGraphStore失败: {str(e)}")
            raise

    def initialize_tables(self):
        """初始化数据库表"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 检查实体表是否存在
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM USER_TABLES 
                    WHERE TABLE_NAME = 'MEDICAL_ENTITIES'
                """)
                if cursor.fetchone()[0] == 0:
                    # 创建实体表
                    cursor.execute("""
                        CREATE TABLE MEDICAL_ENTITIES (
                            ENTITY_ID NUMBER GENERATED ALWAYS AS IDENTITY,
                            ENTITY_TYPE VARCHAR2(100),
                            ENTITY_NAME VARCHAR2(1000),
                            ENTITY_VALUE VARCHAR2(1000),
                            ENTITY_TIME DATE,
                            DOC_REF VARCHAR2(1000),
                            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (ENTITY_ID)
                        )
                    """)
                    logger.info("成功创建实体表 MEDICAL_ENTITIES")
                else:
                    logger.info("实体表 MEDICAL_ENTITIES 已存在")
                
                # 检查关系表是否存在
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM USER_TABLES 
                    WHERE TABLE_NAME = 'MEDICAL_RELATIONS'
                """)
                if cursor.fetchone()[0] == 0:
                    # 创建关系表
                    cursor.execute("""
                        CREATE TABLE MEDICAL_RELATIONS (
                            RELATION_ID NUMBER GENERATED ALWAYS AS IDENTITY,
                            FROM_ENTITY_ID NUMBER,
                            TO_ENTITY_ID NUMBER,
                            RELATION_TYPE VARCHAR2(100),
                            RELATION_TIME DATE,
                            DOC_REFERENCE VARCHAR2(1000),
                            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (RELATION_ID),
                            FOREIGN KEY (FROM_ENTITY_ID) REFERENCES MEDICAL_ENTITIES(ENTITY_ID) ON DELETE CASCADE,
                            FOREIGN KEY (TO_ENTITY_ID) REFERENCES MEDICAL_ENTITIES(ENTITY_ID) ON DELETE CASCADE
                        )
                    """)
                    logger.info("成功创建关系表 MEDICAL_RELATIONS")
                else:
                    logger.info("关系表 MEDICAL_RELATIONS 已存在")
                
                connection.commit()
                logger.info("数据库表初始化完成")
                
        except Exception as e:
            logger.error(f"初始化数据库表失败: {str(e)}")
            raise

    def clear_data(self):
        """清空表中的所有数据"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 由于外键约束，先删除关系表数据
                cursor.execute("DELETE FROM MEDICAL_RELATIONS")
                rows_deleted = cursor.rowcount
                logger.info(f"已清除关系表数据: {rows_deleted}行")
                
                # 再删除实体表数据
                cursor.execute("DELETE FROM MEDICAL_ENTITIES")
                rows_deleted = cursor.rowcount
                logger.info(f"已清除实体表数据: {rows_deleted}行")
                
                connection.commit()
                logger.info("数据清理完成")
                
        except Exception as e:
            logger.error(f"清理数据失败: {str(e)}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            logger.info("关闭OracleGraphStore连接池")

    def close(self):
        """关闭连接池"""
        if self.pool:
            self.pool.close()
            logger.info("关闭OracleGraphStore连接池")

    def get_connection(self):
        """获取数据库连接，带重试机制"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                return self.pool.acquire()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"获取数据库连接失败: {str(e)}")
                    raise
                logger.warning(f"获取连接失败，第{attempt + 1}次重试")
                time.sleep(retry_delay)
                
    def get_document_data(self, doc_reference: str) -> Dict[str, Any]:
        """获取指定文档的所有数据"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 检查文是否存在
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM MEDICAL_ENTITIES 
                    WHERE DOC_REF = :1
                """, [doc_reference])
                count = cursor.fetchone()[0]
                logger.info(f"文档 {doc_reference} 的实体数量: {count}")
                
                # 获取患者信息
                cursor.execute("""
                    SELECT e.ENTITY_TYPE, e.ENTITY_NAME, e.ENTITY_VALUE, e.ENTITY_TIME
                    FROM MEDICAL_ENTITIES e
                    WHERE e.DOC_REF = :1
                    ORDER BY e.ENTITY_TYPE, e.ENTITY_TIME
                """, [doc_reference])
                
                entities = []
                for row in cursor:
                    entity = {
                        "类型": row[0],
                        "名称": row[1],
                        "值": row[2],
                        "时间": row[3].strftime('%Y-%m-%d') if row[3] else None
                    }
                    entities.append(entity)
                logger.info(f"找到的实体数量: {len(entities)}")
                
                # 获取关系信息
                cursor.execute("""
                    SELECT 
                        e1.ENTITY_TYPE as source_type,
                        e1.ENTITY_NAME as source_name,
                        r.RELATION_TYPE,
                        e2.ENTITY_TYPE as target_type,
                        e2.ENTITY_NAME as target_name,
                        e2.ENTITY_VALUE as target_value,
                        r.RELATION_TIME as target_time
                    FROM MEDICAL_RELATIONS r
                    JOIN MEDICAL_ENTITIES e1 ON r.FROM_ENTITY_ID = e1.ENTITY_ID
                    JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                    WHERE r.DOC_REFERENCE = :1
                    ORDER BY r.RELATION_TIME
                """, [doc_reference])
                
                relationships = []
                for row in cursor:
                    relation = {
                        "源类型": row[0],
                        "源名称": row[1],
                        "关系类型": row[2],
                        "目标类型": row[3],
                        "目标名称": row[4],
                        "目标值": row[5],
                        "时间": row[6].strftime('%Y-%m-%d') if row[6] else None
                    }
                    relationships.append(relation)
                logger.info(f"找到的关系数量: {len(relationships)}")
                
                result = {
                    "实体": entities,
                    "关系": relationships
                }
                logger.info(f"返回的数据结构: {json.dumps(result, ensure_ascii=False)}")
                return result
                
        except Exception as e:
            logger.error(f"获取文档数据失败: {str(e)}")
            raise

    def search_entities(self, entity_type: Optional[str] = None, 
                       entity_name: Optional[str] = None,
                       entity_value: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索实体"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if entity_type:
                    conditions.append("ENTITY_TYPE LIKE :type")
                    params.append(f"%{entity_type}%")
                
                if entity_name:
                    conditions.append("ENTITY_NAME LIKE :name")
                    params.append(f"%{entity_name}%")
                
                if entity_value:
                    conditions.append("ENTITY_VALUE LIKE :value")
                    params.append(f"%{entity_value}%")
                
                if start_date:
                    conditions.append("ENTITY_TIME >= TO_DATE(:start_date, 'YYYY-MM-DD')")
                    params.append(start_date)
                
                if end_date:
                    conditions.append("ENTITY_TIME <= TO_DATE(:end_date, 'YYYY-MM-DD')")
                    params.append(end_date)
                
                # 构建SQL语句
                sql = """
                    SELECT ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, ENTITY_TIME, DOC_REF
                    FROM MEDICAL_ENTITIES
                """
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY ENTITY_TIME DESC"
                
                # 执行查询
                cursor.execute(sql, params)
                
                results = []
                for row in cursor:
                    results.append({
                        "类型": row[0],
                        "名称": row[1],
                        "值": row[2],
                        "时间": row[3].strftime('%Y-%m-%d') if row[3] else None,
                        "文档引用": row[4]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"搜索实体失败: {str(e)}")
            raise

    def search_relationships(self, relationship_type: Optional[str] = None,
                           source_type: Optional[str] = None,
                           target_type: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索关系"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if relationship_type:
                    conditions.append("r.RELATION_TYPE LIKE :rel_type")
                    params.append(f"%{relationship_type}%")
                
                if source_type:
                    conditions.append("e1.ENTITY_TYPE LIKE :source_type")
                    params.append(f"%{source_type}%")
                
                if target_type:
                    conditions.append("e2.ENTITY_TYPE LIKE :target_type")
                    params.append(f"%{target_type}%")
                
                if start_date:
                    conditions.append("r.RELATION_TIME >= TO_DATE(:start_date, 'YYYY-MM-DD')")
                    params.append(start_date)
                
                if end_date:
                    conditions.append("r.RELATION_TIME <= TO_DATE(:end_date, 'YYYY-MM-DD')")
                    params.append(end_date)
                
                # 构建SQL语句
                sql = """
                    SELECT 
                        e1.ENTITY_TYPE as source_type,
                        e1.ENTITY_NAME as source_name,
                        r.RELATION_TYPE,
                        e2.ENTITY_TYPE as target_type,
                        e2.ENTITY_NAME as target_name,
                        e2.ENTITY_VALUE as target_value,
                        r.RELATION_TIME as relation_time,
                        r.DOC_REFERENCE
                    FROM MEDICAL_RELATIONS r
                    JOIN MEDICAL_ENTITIES e1 ON r.FROM_ENTITY_ID = e1.ENTITY_ID
                    JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                """
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY r.RELATION_TIME DESC NULLS LAST"
                
                # 记录SQL语句
                logger.info(f"执行关系查询SQL: {sql}")
                logger.info(f"参数: {params}")
                
                # 执行查询
                cursor.execute(sql, params)
                
                # 获取并记录原始数据
                raw_data = cursor.fetchall()
                logger.info(f"查询到的原始关系数据: {raw_data}")
                
                results = []
                for row in raw_data:
                    relation = {
                        "源类型": row[0],
                        "源名称": row[1],
                        "关系类型": row[2],
                        "目标类型": row[3],
                        "目标名称": row[4],
                        "目标值": row[5],
                        "时间": row[6].strftime('%Y-%m-%d') if row[6] else None,
                        "文档引用": row[7]
                    }
                    results.append(relation)
                
                logger.info(f"处理后的关系数据: {json.dumps(results, ensure_ascii=False)}")
                return results
                
        except Exception as e:
            logger.error(f"搜索关系失败: {str(e)}")
            raise

    def get_patients(self) -> List[Dict[str, Any]]:
        """获取所有患者列表"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
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
                cursor.execute(sql)
                
                # 获取并记录原始数据
                raw_data = cursor.fetchall()
                logger.info(f"查询到的原始患者数据: {raw_data}")
                
                if not raw_data:
                    logger.info("没有找到任何患者数据")
                    return []
                
                results = []
                for row in raw_data:
                    try:
                        # 解析患者信息
                        patient_info = json.loads(row[2]) if row[2] else {}
                        logger.info(f"解析的患者信息: {patient_info}")
                        
                        # 构建患者数据
                        patient_data = {
                            "姓名": row[0],  # 使用ENTITY_NAME作为姓名
                            "文档": row[1],  # 使用DOC_REF作为文档
                            "基本信息": patient_info.get("基本信息", {}),
                            "入院日期": patient_info.get("住院信息", {}).get("入院日期"),
                            "出院日期": patient_info.get("住院信息", {}).get("出院日期")
                        }
                        
                        # 确保基本信息是字典
                        if not isinstance(patient_data["基本信息"], dict):
                            patient_data["基本信息"] = {}
                        
                        logger.info(f"处理后的患者数据: {patient_data}")
                        results.append(patient_data)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"解析患者信息失败: {str(e)}, 原始数据: {row[2]}")
                        # 即使解析失败，也添加基本信息
                        results.append({
                            "姓名": row[0],
                            "文档": row[1],
                            "基本信息": {},
                            "入院日期": None,
                            "出院日期": None
                        })
                    except Exception as e:
                        logger.error(f"处理患者数据时出错: {str(e)}, 跳过此条记录")
                        continue
                
                logger.info(f"最终返回的患者数量: {len(results)}")
                logger.info(f"患者列表: {json.dumps(results, ensure_ascii=False)}")
                return results
                
        except Exception as e:
            logger.error(f"获取患者列表失败: {str(e)}")
            return []

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """获取所有患者列表（get_patients 的别名）"""
        return self.get_patients()

    def search_relations(self, query_type: Optional[str] = None, 
                        query_value: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索关系"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 构建SQL语句
                sql = """
                    SELECT DISTINCT
                        e2.ENTITY_NAME as entity_name,
                        e2.ENTITY_VALUE as entity_value
                    FROM MEDICAL_ENTITIES e1
                    JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
                    JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                    WHERE e1.ENTITY_TYPE = '患者'
                    AND e1.ENTITY_NAME = :1
                """
                
                params = [query_value] if query_value else []
                
                # 添加查询条件
                if query_type:
                    sql += " AND e2.ENTITY_TYPE = :2"
                    params.append(query_type)
                
                sql += " ORDER BY e2.ENTITY_NAME"
                
                # 记录SQL语句
                logger.info(f"执行关系查询SQL: {sql}")
                logger.info(f"参数: {params}")
                
                # 执行查询
                cursor.execute(sql, params)
                
                # 获取并记录原始数据
                raw_data = cursor.fetchall()
                logger.info(f"查询到的原始关系数据: {raw_data}")
                
                results = []
                for row in raw_data:
                    relation = {
                        "名称": row[0],
                        "值": row[1]
                    }
                    results.append(relation)
                
                logger.info(f"处理后的关系数据: {json.dumps(results, ensure_ascii=False)}")
                return results
                
        except Exception as e:
            logger.error(f"搜索关系失败: {str(e)}")
            return []

    def get_patient_timeline(self, doc_reference: str) -> List[Dict[str, Any]]:
        """获取患者的时间线数据"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 获取所有带时间的实体
                cursor.execute("""
                    SELECT 
                        e.ENTITY_TYPE,
                        e.ENTITY_NAME,
                        e.ENTITY_VALUE,
                        e.ENTITY_TIME,
                        r.RELATIONSHIP_TYPE
                    FROM MEDICAL_ENTITIES e
                    LEFT JOIN MEDICAL_RELATIONSHIPS r ON e.ENTITY_ID = r.TARGET_ID
                    WHERE e.DOC_REF = :1
                    AND e.ENTITY_TIME IS NOT NULL
                    ORDER BY e.ENTITY_TIME
                """, [doc_reference])
                
                timeline = []
                for row in cursor:
                    timeline.append({
                        "类型": row[0],
                        "名称": row[1],
                        "值": row[2],
                        "时间": row[3].strftime('%Y-%m-%d') if row[3] else None,
                        "关系": row[4]
                    })
                
                return timeline
                
        except Exception as e:
            logger.error(f"获取患者时间线失败: {str(e)}")
            raise

    def get_patient_summary(self, doc_reference: str) -> Dict[str, Any]:
        """获取患者摘要信息"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 获取患者基本信息
                cursor.execute("""
                    SELECT e.ENTITY_VALUE
                    FROM MEDICAL_ENTITIES e
                    WHERE e.DOC_REF = :1
                    AND e.ENTITY_TYPE = '患者'
                """, [doc_reference])
                
                row = cursor.fetchone()
                if not row:
                    return {}
                
                patient_info = json.loads(row[0]) if row[0] else {}
                
                # 获取主要诊断
                cursor.execute("""
                    SELECT e.ENTITY_NAME, e.ENTITY_VALUE, e.ENTITY_TIME
                    FROM MEDICAL_ENTITIES e
                    WHERE e.DOC_REF = :1
                    AND e.ENTITY_TYPE = '诊断'
                    ORDER BY e.ENTITY_TIME
                """, [doc_reference])
                
                diagnoses = []
                for row in cursor:
                    diagnoses.append({
                        "类型": row[0],
                        "内容": row[1],
                        "时间": row[2].strftime('%Y-%m-%d') if row[2] else None
                    })
                
                # 获取住院信息
                cursor.execute("""
                    SELECT e.ENTITY_NAME, e.ENTITY_VALUE, e.ENTITY_TIME
                    FROM MEDICAL_ENTITIES e
                    WHERE e.DOC_REF = :1
                    AND e.ENTITY_TYPE = '住院信息'
                    ORDER BY e.ENTITY_TIME
                """, [doc_reference])
                
                hospital_info = {}
                for row in cursor:
                    hospital_info[row[0]] = {
                        "值": row[1],
                        "时间": row[2].strftime('%Y-%m-%d') if row[2] else None
                    }
                
                return {
                    "基本信息": patient_info.get("基本信息", {}),
                    "住院信息": hospital_info,
                    "主要诊断": diagnoses
                }
                
        except Exception as e:
            logger.error(f"获取患者摘要失败: {str(e)}")
            raise

    def store_medical_data(self, data: dict, doc_reference: str):
        """存储医疗数据到图数据库"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 存储患者信息
                patient_data = data.get("患者", {})
                logger.info(f"开始处理患者数据: {doc_reference}")
                logger.info(f"患者数据内容: {json.dumps(patient_data, ensure_ascii=False)}")
                
                # 创建输出参数变量
                entity_id_var = cursor.var(oracledb.NUMBER)
                
                # 存储患者实体
                cursor.execute("""
                    INSERT INTO MEDICAL_ENTITIES 
                    (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF)
                    VALUES ('患者', :name, :value, :doc_ref)
                    RETURNING ENTITY_ID INTO :entity_id
                """, {
                    'name': patient_data.get("姓名"),
                    'value': json.dumps(patient_data, ensure_ascii=False),
                    'doc_ref': doc_reference,
                    'entity_id': entity_id_var
                })
                patient_id = entity_id_var.getvalue()[0]
                logger.info(f"存储患者信息成功: ID={patient_id}")

                # 存储主诉与诊断
                for diagnosis in data.get("主诉与诊断", []):
                    entity_id_var = cursor.var(oracledb.NUMBER)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF)
                        VALUES ('诊断', :name, :value, :doc_ref)
                        RETURNING ENTITY_ID INTO :entity_id
                    """, {
                        'name': str(diagnosis.get("类型", "")),
                        'value': str(diagnosis.get("内容", "")),
                        'doc_ref': doc_reference,
                        'entity_id': entity_id_var
                    })
                    diagnosis_id = entity_id_var.getvalue()[0]
                    
                    # 创建关系
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref)
                    """, {
                        'from_id': patient_id,
                        'to_id': diagnosis_id,
                        'rel_type': '诊断',
                        'doc_ref': doc_reference
                    })

                # 存储现病史
                for history in data.get("现病史", []):
                    entity_id_var = cursor.var(oracledb.NUMBER)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF)
                        VALUES ('现病史', :name, :value, :doc_ref)
                        RETURNING ENTITY_ID INTO :entity_id
                    """, {
                        'name': str(history.get("症状", "")),
                        'value': str(history.get("描述", "")),
                        'doc_ref': doc_reference,
                        'entity_id': entity_id_var
                    })
                    history_id = entity_id_var.getvalue()[0]
                    
                    # 创建关系
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref)
                    """, {
                        'from_id': patient_id,
                        'to_id': history_id,
                        'rel_type': '现病史',
                        'doc_ref': doc_reference
                    })

                # 存储生命体征
                for vital in data.get("生命体征", []):
                    entity_id_var = cursor.var(oracledb.NUMBER)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF)
                        VALUES ('生命体征', :name, :value, :doc_ref)
                        RETURNING ENTITY_ID INTO :entity_id
                    """, {
                        'name': str(vital.get("指标", "")),
                        'value': str(vital.get("数值", "")),
                        'doc_ref': doc_reference,
                        'entity_id': entity_id_var
                    })
                    vital_id = entity_id_var.getvalue()[0]
                    
                    # 创建关系
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref)
                    """, {
                        'from_id': patient_id,
                        'to_id': vital_id,
                        'rel_type': '生命体征',
                        'doc_ref': doc_reference
                    })

                # 存储生化指标
                for lab in data.get("生化指标", []):
                    entity_id_var = cursor.var(oracledb.NUMBER)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF)
                        VALUES ('生化指标', :name, :value, :doc_ref)
                        RETURNING ENTITY_ID INTO :entity_id
                    """, {
                        'name': str(lab.get("项目", "")),
                        'value': str(lab.get("结果", "")),
                        'doc_ref': doc_reference,
                        'entity_id': entity_id_var
                    })
                    lab_id = entity_id_var.getvalue()[0]
                    
                    # 创建关系
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref)
                    """, {
                        'from_id': patient_id,
                        'to_id': lab_id,
                        'rel_type': '生化指标',
                        'doc_ref': doc_reference
                    })
                
                connection.commit()
                logger.info(f"成功存储文档 {doc_reference} 的所有数据")
                
        except Exception as e:
            logger.error(f"存储图数据失败: {str(e)}")
            raise

    def get_patient_info(self, patient_name: str) -> Dict[str, Any]:
        """获取指定患者的详细信息"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
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
                cursor.execute(sql, [patient_name])
                
                # 获取并记录原始数据
                row = cursor.fetchone()
                logger.info(f"查询到的患者数据: {row}")
                
                if not row:
                    logger.info(f"未找到患者: {patient_name}")
                    return {}
                
                try:
                    # 解析患者信息
                    patient_info = json.loads(row[2]) if row[2] else {}
                    logger.info(f"解析的患者信息: {patient_info}")
                    
                    # 构建患者数据
                    result = {
                        "基本信息": patient_info.get("基本信息", {}),
                        "主诉与诊断": [],
                        "现病史": [],
                        "生命体征": [],
                        "生化指标": [],
                        "诊疗经过": []
                    }
                    
                    # 获取患者的诊断信息
                    sql = """
                        SELECT 
                            e2.ENTITY_NAME,
                            e2.ENTITY_VALUE,
                            e2.ENTITY_TIME
                        FROM MEDICAL_ENTITIES e1
                        JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
                        JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                        WHERE e1.ENTITY_TYPE = '患者'
                        AND e1.ENTITY_NAME = :1
                        AND e2.ENTITY_TYPE = '诊断'
                        ORDER BY e2.ENTITY_TIME
                    """
                    cursor.execute(sql, [patient_name])
                    
                    for diagnosis_row in cursor:
                        result["主诉与诊断"].append({
                            "类型": diagnosis_row[0],
                            "内容": diagnosis_row[1],
                            "时间": diagnosis_row[2].strftime('%Y-%m-%d') if diagnosis_row[2] else None
                        })
                    
                    # 获取现病史
                    sql = """
                        SELECT 
                            e2.ENTITY_NAME,
                            e2.ENTITY_VALUE,
                            e2.ENTITY_TIME
                        FROM MEDICAL_ENTITIES e1
                        JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
                        JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                        WHERE e1.ENTITY_TYPE = '患者'
                        AND e1.ENTITY_NAME = :1
                        AND e2.ENTITY_TYPE = '现病史'
                        ORDER BY e2.ENTITY_TIME
                    """
                    cursor.execute(sql, [patient_name])
                    
                    for history_row in cursor:
                        result["现病史"].append({
                            "症状": history_row[0],
                            "描述": history_row[1],
                            "时间": history_row[2].strftime('%Y-%m-%d') if history_row[2] else None
                        })
                    
                    # 获取生命体征
                    sql = """
                        SELECT 
                            e2.ENTITY_NAME,
                            e2.ENTITY_VALUE,
                            e2.ENTITY_TIME
                        FROM MEDICAL_ENTITIES e1
                        JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
                        JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                        WHERE e1.ENTITY_TYPE = '患者'
                        AND e1.ENTITY_NAME = :1
                        AND e2.ENTITY_TYPE = '生命体征'
                        ORDER BY e2.ENTITY_TIME
                    """
                    cursor.execute(sql, [patient_name])
                    
                    for vital_row in cursor:
                        result["生命体征"].append({
                            "指标": vital_row[0],
                            "数值": vital_row[1],
                            "时间": vital_row[2].strftime('%Y-%m-%d') if vital_row[2] else None
                        })
                    
                    # 获取生化指标
                    sql = """
                        SELECT 
                            e2.ENTITY_NAME,
                            e2.ENTITY_VALUE,
                            e2.ENTITY_TIME
                        FROM MEDICAL_ENTITIES e1
                        JOIN MEDICAL_RELATIONS r ON e1.ENTITY_ID = r.FROM_ENTITY_ID
                        JOIN MEDICAL_ENTITIES e2 ON r.TO_ENTITY_ID = e2.ENTITY_ID
                        WHERE e1.ENTITY_TYPE = '患者'
                        AND e1.ENTITY_NAME = :1
                        AND e2.ENTITY_TYPE = '生化指标'
                        ORDER BY e2.ENTITY_TIME
                    """
                    cursor.execute(sql, [patient_name])
                    
                    for lab_row in cursor:
                        result["生化指标"].append({
                            "项目": lab_row[0],
                            "结果": lab_row[1],
                            "时间": lab_row[2].strftime('%Y-%m-%d') if lab_row[2] else None
                        })
                    
                    logger.info(f"处理后的患者完整数据: {json.dumps(result, ensure_ascii=False)}")
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"解析患者信息失败: {str(e)}, 原始数据: {row[2]}")
                    return {
                        "基本信息": {},
                        "主诉与诊断": [],
                        "现病史": [],
                        "生命体征": [],
                        "生化指标": [],
                        "诊疗经过": []
                    }
                    
        except Exception as e:
            logger.error(f"获取患者信息失败: {str(e)}")
            return {}