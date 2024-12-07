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
        """初始化数据库表和属性图"""
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
                    logger.error("MEDICAL_ENTITIES 表不存在，请先创建数据库表")
                    raise ValueError("数据库表未创建")
                
                # 检查关系表是否存在
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM USER_TABLES 
                    WHERE TABLE_NAME = 'MEDICAL_RELATIONS'
                """)
                if cursor.fetchone()[0] == 0:
                    logger.error("MEDICAL_RELATIONS 表不存在，请先创建数据库表")
                    raise ValueError("数据库表未创建")
                
                # 检查属性图是否存在
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM USER_PROPERTY_GRAPHS 
                    WHERE GRAPH_NAME = 'medical_kg'
                """)
                if cursor.fetchone()[0] == 0:
                    logger.info("属性图不存在，开始创建...")
                    try:
                        # 创建属性图
                        cursor.execute("""
                        CREATE PROPERTY GRAPH medical_kg
                        VERTEX TABLES (
                            medical_entities
                            KEY (entity_id)
                            PROPERTIES (
                                entity_type,
                                entity_name,
                                entity_value,
                                doc_ref,
                                created_at
                            )
                        )
                        EDGE TABLES (
                            medical_relations
                            KEY (relation_id)
                            SOURCE KEY (from_entity_id) REFERENCES medical_entities(entity_id)
                            DESTINATION KEY (to_entity_id) REFERENCES medical_entities(entity_id)
                            PROPERTIES (
                                relation_type,
                                doc_reference,
                                created_at
                            )
                        )
                        """)
                        connection.commit()
                        logger.info("属性图创建成功")
                    except Exception as e:
                        if "ORA-00955" in str(e):  # 对象已存在
                            logger.info("属性图已存在")
                        else:
                            logger.error(f"创建属性图失败: {str(e)}")
                            raise
                else:
                    logger.info("属性图已存在")
                
                logger.info("数据库表和属性图检查完成")
                
        except Exception as e:
            logger.error(f"初始化数据库表和属性图失败: {str(e)}")
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

    def delete_document_data(self, doc_reference: str):
        """删除定文档的所有数据"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 由于外键约束，删除关系表数据
                cursor.execute("""
                    DELETE FROM MEDICAL_RELATIONS
                    WHERE DOC_REFERENCE = :1
                """, [doc_reference])
                relations_deleted = cursor.rowcount
                logger.info(f"已删除文档 {doc_reference} 的关系数据: {relations_deleted}行")
                
                # 再删除实体表数据
                cursor.execute("""
                    DELETE FROM MEDICAL_ENTITIES
                    WHERE DOC_REF = :1
                """, [doc_reference])
                entities_deleted = cursor.rowcount
                logger.info(f"已删除文档 {doc_reference} 的实体数据: {entities_deleted}行")
                
                connection.commit()
                logger.info(f"成功删除文档 {doc_reference} 的所有数据")
                
        except Exception as e:
            logger.error(f"删除文档数据失败: {str(e)}")
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
                        "目": row[3],
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
                
                # 执行查
                cursor.execute(sql, params)
                
                results = []
                for row in cursor:
                    results.append({
                        "类型": row[0],
                        "名称": row[1],
                        "值": row[2],
                        "时间": row[3].strftime('%Y-%m-%d') if row[3] else None,
                        "档引用": row[4]
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
                        "目名称": row[4],
                        "目标值": row[5],
                        "时间": row[6].strftime('%Y-%m-%d') if row[6] else None,
                        "档引用": row[7]
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
                
                # 获取有患者实体
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
                        # 读取CLOB内容
                        clob_data = row[2]
                        if hasattr(clob_data, 'read'):
                            entity_value = clob_data.read()
                        else:
                            entity_value = str(clob_data)
                        
                        # 解析患者信息
                        patient_info = json.loads(entity_value)
                        logger.info(f"解析的患者信息: {patient_info}")
                        
                        # 构建患者数据
                        patient_data = {
                            "姓名": patient_info.get("患者姓名", row[0]),
                            "文档": row[1],
                            "基本信息": {
                                "姓名": patient_info.get("患者姓名", "未知"),
                                "性别": patient_info.get("性别", "未知"),
                                "年龄": patient_info.get("年龄", "未知"),
                                "入院日期": patient_info.get("入院日期", "未知"),
                                "出院日": patient_info.get("出院日期", "未知")
                            },
                            "主诉": patient_info.get("主诉", "未知"),
                            "现病史": patient_info.get("现病史", []),
                            "院诊断": patient_info.get("入院诊断", []),
                            "出院诊断": patient_info.get("出院诊断", []),
                            "生命体征": patient_info.get("生命体征", {}),
                            "生化指标": patient_info.get("生化指标", {}),
                            "诊疗经过": patient_info.get("诊疗经过", ""),
                            "出院医嘱": patient_info.get("出院医嘱", []),
                            "metadata": patient_info.get("metadata", {})
                        }
                        
                        logger.info(f"处理后的患者数据: {patient_data}")
                        results.append(patient_data)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"解析患者信息失败: {str(e)}, 原始数据: {row[2]}")
                        # 即使解析失败，也添加基本信息
                        results.append({
                            "姓名": row[0],
                            "文档": row[1],
                            "基本信息": {},
                            "主诉": "未知",
                            "现病史": [],
                            "入院诊断": [],
                            "院诊断": [],
                            "生命体征": {},
                            "生化指标": {},
                            "诊疗经过": "",
                            "出院医嘱": [],
                            "metadata": {}
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
        """获取患者列表（get_patients 的别名）"""
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
                    AND e.ENTITY_TYPE = '住院息'
                    ORDER BY e.ENTITY_TIME
                """, [doc_reference])
                
                hospital_info = {}
                for row in cursor:
                    hospital_info[row[0]] = {
                        "值": row[1],
                        "时间": row[2].strftime('%Y-%m-%d') if row[2] else None
                    }
                
                return {
                    "本信息": patient_info.get("基本信息", {}),
                    "住信息": hospital_info,
                    "主诊断": diagnoses
                }
                
        except Exception as e:
            logger.error(f"获取患者摘要失败: {str(e)}")
            raise

    def store_medical_data(self, data: Dict[str, Any], doc_reference: str):
        """将解析后的数据存储到图数据库中"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 存储患者基本信息
                id_var = cursor.var(int)
                patient_name = str(data.get("患者姓名", ""))
                if not patient_name:
                    logger.error("患者姓名不能为空")
                    raise ValueError("患者姓名不能为空")
                    
                cursor.execute("""
                    INSERT INTO MEDICAL_ENTITIES 
                    (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                    VALUES ('患者', :name, :value, :doc_ref, :created_at)
                    RETURNING ENTITY_ID INTO :id
                """, {
                    'name': patient_name,  # 直接使用患者姓名，不设置默认值
                    'value': json.dumps(data, ensure_ascii=False),
                    'doc_ref': str(doc_reference),
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'id': id_var
                })
                patient_id = id_var.getvalue()[0]
                
                # 存储主诉
                id_var = cursor.var(int)
                cursor.execute("""
                    INSERT INTO MEDICAL_ENTITIES 
                    (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                    VALUES ('主诉', '主诉', :value, :doc_ref, :created_at)
                    RETURNING ENTITY_ID INTO :id
                """, {
                    'value': str(data.get("主诉", "未知")),
                    'doc_ref': str(doc_reference),
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'id': id_var
                })
                chief_complaint_id = id_var.getvalue()[0]
                
                cursor.execute("""
                    INSERT INTO MEDICAL_RELATIONS 
                    (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                    VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                """, {
                    'from_id': patient_id,
                    'to_id': chief_complaint_id,
                    'rel_type': "主诉",
                    'doc_ref': str(doc_reference),
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # 存储现病史
                for symptom in data.get("现病史", []):
                    id_var = cursor.var(int)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                        VALUES ('现病史', '症状', :value, :doc_ref, :created_at)
                        RETURNING ENTITY_ID INTO :id
                    """, {
                        'value': str(symptom),
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'id': id_var
                    })
                    symptom_id = id_var.getvalue()[0]
                    
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                    """, {
                        'from_id': patient_id,
                        'to_id': symptom_id,
                        'rel_type': "现病史",
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # 存储入院诊断
                for diagnosis in data.get("入院诊断", []):
                    id_var = cursor.var(int)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                        VALUES ('入院诊断', '断', :value, :doc_ref, :created_at)
                        RETURNING ENTITY_ID INTO :id
                    """, {
                        'value': str(diagnosis),
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'id': id_var
                    })
                    diagnosis_id = id_var.getvalue()[0]
                    
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                    """, {
                        'from_id': patient_id,
                        'to_id': diagnosis_id,
                        'rel_type': "入院诊断",
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # 存储出院诊断
                for diagnosis in data.get("出院诊断", []):
                    id_var = cursor.var(int)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                        VALUES ('出院诊断', '诊断', :value, :doc_ref, :created_at)
                        RETURNING ENTITY_ID INTO :id
                    """, {
                        'value': str(diagnosis),
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'id': id_var
                    })
                    diagnosis_id = id_var.getvalue()[0]
                    
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                    """, {
                        'from_id': patient_id,
                        'to_id': diagnosis_id,
                        'rel_type': "出院诊断",
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # 存储生体征
                vital_signs = data.get("生命体征", {})
                if isinstance(vital_signs, dict):
                    for key, value in vital_signs.items():
                        id_var = cursor.var(int)
                        cursor.execute("""
                            INSERT INTO MEDICAL_ENTITIES 
                            (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                            VALUES ('生命体征', :name, :value, :doc_ref, :created_at)
                            RETURNING ENTITY_ID INTO :id
                        """, {
                            'name': str(key),
                            'value': str(value),
                            'doc_ref': str(doc_reference),
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'id': id_var
                        })
                        vital_id = id_var.getvalue()[0]
                        
                        cursor.execute("""
                            INSERT INTO MEDICAL_RELATIONS 
                            (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                            VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                        """, {
                            'from_id': patient_id,
                            'to_id': vital_id,
                            'rel_type': "生命体征",
                            'doc_ref': str(doc_reference),
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                
                # 储生化指标
                biochemical_indicators = data.get("生化指标", {})
                if isinstance(biochemical_indicators, dict):
                    for key, value in biochemical_indicators.items():
                        id_var = cursor.var(int)
                        cursor.execute("""
                            INSERT INTO MEDICAL_ENTITIES 
                            (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                            VALUES ('生化指标', :name, :value, :doc_ref, :created_at)
                            RETURNING ENTITY_ID INTO :id
                        """, {
                            'name': str(key),
                            'value': str(value),
                            'doc_ref': str(doc_reference),
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'id': id_var
                        })
                        test_id = id_var.getvalue()[0]
                        
                        cursor.execute("""
                            INSERT INTO MEDICAL_RELATIONS 
                            (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                            VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                        """, {
                            'from_id': patient_id,
                            'to_id': test_id,
                            'rel_type': "生化指标",
                            'doc_ref': str(doc_reference),
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                
                # 存储诊疗经过
                if data.get("诊疗经过"):
                    id_var = cursor.var(int)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                        VALUES ('诊疗经过', '经过', :value, :doc_ref, :created_at)
                        RETURNING ENTITY_ID INTO :id
                    """, {
                        'value': str(data.get("诊疗经过")),
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'id': id_var
                    })
                    treatment_id = id_var.getvalue()[0]
                    
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                    """, {
                        'from_id': patient_id,
                        'to_id': treatment_id,
                        'rel_type': "诊疗经过",
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # 储出院医嘱
                for advice in data.get("出院医嘱", []):
                    id_var = cursor.var(int)
                    cursor.execute("""
                        INSERT INTO MEDICAL_ENTITIES 
                        (ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE, DOC_REF, CREATED_AT)
                        VALUES ('出院医嘱', '医嘱', :value, :doc_ref, :created_at)
                        RETURNING ENTITY_ID INTO :id
                    """, {
                        'value': str(advice),
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'id': id_var
                    })
                    advice_id = id_var.getvalue()[0]
                    
                    cursor.execute("""
                        INSERT INTO MEDICAL_RELATIONS 
                        (FROM_ENTITY_ID, TO_ENTITY_ID, RELATION_TYPE, DOC_REFERENCE, CREATED_AT)
                        VALUES (:from_id, :to_id, :rel_type, :doc_ref, :created_at)
                    """, {
                        'from_id': patient_id,
                        'to_id': advice_id,
                        'rel_type': "出院医嘱",
                        'doc_ref': str(doc_reference),
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                connection.commit()
                logger.info(f"成功存储文档 {doc_reference} 的数据")
                
        except Exception as e:
            logger.error(f"存储医疗数据失败: {str(e)}")
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
                    # 读取CLOB内容
                    clob_data = row[2]
                    if hasattr(clob_data, 'read'):
                        entity_value = clob_data.read()
                    else:
                        entity_value = str(clob_data)
                    
                    # 解析患者信息
                    patient_info = json.loads(entity_value)
                    logger.info(f"解析的患者信息: {patient_info}")
                    
                    # 返回完整的患者信息
                    return {
                        "姓名": row[0],
                        "文档": row[1],
                        **patient_info  # 展开完整的患者信息
                    }
                    
                except json.JSONDecodeError as e:
                    logger.error(f"解析患者息失败: {str(e)}, 原始数据: {entity_value}")
                    return {
                        "姓名": row[0],
                        "文档": row[1]
                    }
                except Exception as e:
                    logger.error(f"处理患者数据时出错: {str(e)}")
                    return {
                        "姓名": row[0],
                        "文档": row[1]
                    }
                
        except Exception as e:
            logger.error(f"获取患者信息失败: {str(e)}")
            return {}

    def check_graph_exists(self) -> bool:
        """检查属性图是否存在"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM USER_PROPERTY_GRAPHS 
                    WHERE GRAPH_NAME = 'medical_kg'
                """)
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"检查属性图状态失败: {str(e)}")
            return False

    def create_property_graph(self) -> bool:
        """创建属性图"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                try:
                    # 如果图已存在，先删除
                    cursor.execute("DROP PROPERTY GRAPH medical_kg FORCE")
                    connection.commit()
                    logger.info("已删除旧的属性图")
                except Exception as e:
                    if "ORA-00942" not in str(e):  # 不存在时忽略错误
                        logger.warning(f"删除旧图时出: {str(e)}")
                
                # 创建新的属性图
                cursor.execute("""
                CREATE PROPERTY GRAPH medical_kg
                VERTEX TABLES (
                    medical_entities
                    KEY (entity_id)
                    PROPERTIES (
                        entity_type,
                        entity_name,
                        entity_value,
                        doc_ref,
                        created_at
                    )
                )
                EDGE TABLES (
                    medical_relations
                    KEY (relation_id)
                    SOURCE KEY (from_entity_id) REFERENCES medical_entities(entity_id)
                    DESTINATION KEY (to_entity_id) REFERENCES medical_entities(entity_id)
                    PROPERTIES (
                        relation_type,
                        doc_reference,
                        created_at
                    )
                )
                """)
                connection.commit()
                logger.info("属性图创建成功")
                return True
        except Exception as e:
            logger.error(f"创建属性图失败: {str(e)}")
            return False

    def execute_pgql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行PGQL查询
        
        Args:
            query (str): PGQL查询语句
            params (Optional[Dict]): 查询参数
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # 处理查询字符串，移除多余的换行和空格
                final_query = ' '.join(query.strip().split())
                
                # 如果有参数，替换查询中的命名参数
                if params:
                    for key, value in params.items():
                        # 将命名参数替换为实际值
                        if isinstance(value, str):
                            final_query = final_query.replace(f":{key}", f"'{value}'")
                        else:
                            final_query = final_query.replace(f":{key}", str(value))
                
                # 准备PGQL查询
                pgql_query = f"""
                SELECT *
                FROM GRAPH_TABLE(MEDICAL_KG,
                    '{final_query.replace(chr(39), chr(39)+chr(39))}'
                )
                """
                
                # 记录完整的查询语句
                logger.info(f"执行PGQL查询: {pgql_query}")
                
                # 执行查询
                cursor.execute(pgql_query)
                
                # 获取列名
                columns = [col[0].lower() for col in cursor.description]
                
                # 处理结果
                results = []
                for row in cursor:
                    result = {}
                    for i, value in enumerate(row):
                        # 处理LOB对象
                        if isinstance(value, oracledb.LOB):
                            result[columns[i]] = value.read()
                        else:
                            result[columns[i]] = value if value is not None else ''
                    results.append(result)
                
                logger.info(f"PGQL查询返回 {len(results)} 条结果")
                return results
                
        except Exception as e:
            logger.error(f"执行PGQL查询失败: {str(e)}")
            raise