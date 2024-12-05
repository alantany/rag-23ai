"""
Test Property Graph Query Templates
"""

from .graph_query import PropertyGraphQuery
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_templates():
    """测试各种查询模板"""
    query_executor = PropertyGraphQuery()
    
    # 1. 测试基本信息查询
    print("\n1. 测试基本信息查询")
    query = """
    SELECT * FROM GRAPH_TABLE (medical_kg
        MATCH
        (v IS entity WHERE v.entity_name = :patient_name) -[e IS relation]-> (v2 IS entity)
        COLUMNS (
            e.relation_type AS record_type,
            v2.entity_type AS info_type,
            v2.entity_value AS detail
        )
    )
    """
    results = query_executor.execute_graph_query(query, {"patient_name": "周某某"})
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)
        
    # 2. 测试症状查询
    print("\n2. 测试症状查询")
    query = """
    SELECT * FROM GRAPH_TABLE (medical_kg
        MATCH
        (v IS entity WHERE v.entity_name = :patient_name) 
            -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
        COLUMNS (
            v.entity_name AS patient_name,
            v2.entity_value AS symptoms
        )
    )
    """
    results = query_executor.execute_graph_query(query, {"patient_name": "周某某"})
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)
        
    # 3. 测试诊疗经过查询
    print("\n3. 测试诊疗经过查询")
    query = """
    SELECT * FROM GRAPH_TABLE (medical_kg
        MATCH
        (v IS entity WHERE v.entity_name = :patient_name) 
            -[e IS relation WHERE e.relation_type = '诊疗经过']-> (v2 IS entity)
        COLUMNS (
            v.entity_name AS patient_name,
            v2.entity_value AS treatment_process
        )
    )
    """
    results = query_executor.execute_graph_query(query, {"patient_name": "周某某"})
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)
        
    # 4. 测试关系类型统计
    print("\n4. 测试关系类型统计")
    query = """
    SELECT relation_type, COUNT(*) as count
    FROM GRAPH_TABLE (medical_kg
        MATCH
        (v IS entity) -[e IS relation]-> (v2 IS entity)
        COLUMNS (
            e.relation_type AS relation_type
        )
    )
    GROUP BY relation_type
    ORDER BY count DESC
    """
    results = query_executor.execute_graph_query(query)
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)
        
    # 5. 测试患者数据统计
    print("\n5. 测试患者数据统计")
    query = """
    SELECT 
        patient_name,
        COUNT(CASE WHEN relation_type = '现病史' THEN 1 END) as symptom_count,
        COUNT(CASE WHEN relation_type IN ('入院诊断', '出院诊断') THEN 1 END) as diagnosis_count,
        COUNT(CASE WHEN relation_type = '生命体征' THEN 1 END) as vital_sign_count,
        COUNT(CASE WHEN relation_type = '生化指标' THEN 1 END) as lab_result_count
    FROM GRAPH_TABLE (medical_kg
        MATCH
        (v IS entity) -[e IS relation]-> (v2 IS entity)
        COLUMNS (
            v.entity_name AS patient_name,
            e.relation_type AS relation_type
        )
    )
    GROUP BY patient_name
    ORDER BY patient_name
    """
    results = query_executor.execute_graph_query(query)
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)

if __name__ == "__main__":
    test_templates() 