"""
Test Property Graph Query Templates
"""

from ..graph_query import PropertyGraphQuery
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_templates():
    """Test all query templates."""
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
        
    # 4. 测试常见症状统计
    print("\n4. 测试常见症状统计")
    query = """
    SELECT v2.entity_value AS symptom, COUNT(*) as count
    FROM GRAPH_TABLE (medical_kg
        MATCH
        (v IS entity) -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
        COLUMNS (
            v2.entity_value AS symptom
        )
    )
    GROUP BY v2.entity_value
    ORDER BY count DESC
    """
    results = query_executor.execute_graph_query(query)
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)
        
    # 5. 测试医疗数据关联性分析
    print("\n5. 测试医疗数据关联性分析")
    query = """
    WITH 
    -- 获取病史数据
    history AS (
        SELECT v2.entity_value AS symptom
        FROM GRAPH_TABLE (medical_kg
            MATCH (v IS entity WHERE v.entity_name = :patient_name)
                -[e IS relation WHERE e.relation_type = '现病史']-> (v2 IS entity)
            COLUMNS (v2.entity_value)
        )
    ),
    -- 获取诊断数据
    diagnosis AS (
        SELECT v2.entity_value AS diagnosis
        FROM GRAPH_TABLE (medical_kg
            MATCH (v IS entity WHERE v.entity_name = :patient_name)
                -[e IS relation WHERE e.relation_type IN ('入院诊断', '出院诊断')]-> (v2 IS entity)
            COLUMNS (v2.entity_value)
        )
    ),
    -- 获取生命体征
    vitals AS (
        SELECT v2.entity_value AS vital_sign
        FROM GRAPH_TABLE (medical_kg
            MATCH (v IS entity WHERE v.entity_name = :patient_name)
                -[e IS relation WHERE e.relation_type = '生命体征']-> (v2 IS entity)
            COLUMNS (v2.entity_value)
        )
    ),
    -- 获取生化指标
    labs AS (
        SELECT v2.entity_value AS lab_result
        FROM GRAPH_TABLE (medical_kg
            MATCH (v IS entity WHERE v.entity_name = :patient_name)
                -[e IS relation WHERE e.relation_type = '生化指标']-> (v2 IS entity)
            COLUMNS (v2.entity_value)
        )
    )
    -- 组合所有数据
    SELECT 
        :patient_name AS patient_name,
        h.symptom,
        d.diagnosis,
        v.vital_sign,
        l.lab_result
    FROM history h
    FULL OUTER JOIN diagnosis d ON 1=1
    FULL OUTER JOIN vitals v ON 1=1
    FULL OUTER JOIN labs l ON 1=1
    WHERE 
        CASE 
            WHEN :data_type1 = '现病史' AND :data_type2 = '诊断' THEN 
                h.symptom IS NOT NULL AND d.diagnosis IS NOT NULL
            WHEN :data_type1 = '现病史' AND :data_type2 = '生命体征' THEN 
                h.symptom IS NOT NULL AND v.vital_sign IS NOT NULL
            WHEN :data_type1 = '现病史' AND :data_type2 = '生化指标' THEN 
                h.symptom IS NOT NULL AND l.lab_result IS NOT NULL
            WHEN :data_type1 = '诊断' AND :data_type2 = '生命体征' THEN 
                d.diagnosis IS NOT NULL AND v.vital_sign IS NOT NULL
            WHEN :data_type1 = '诊断' AND :data_type2 = '生化指标' THEN 
                d.diagnosis IS NOT NULL AND l.lab_result IS NOT NULL
            WHEN :data_type1 = '生命体征' AND :data_type2 = '生化指标' THEN 
                v.vital_sign IS NOT NULL AND l.lab_result IS NOT NULL
            ELSE 1=1
        END
    """
    params = {
        "patient_name": "周某某",
        "data_type1": "现病史",
        "data_type2": "诊断"
    }
    results = query_executor.execute_graph_query(query, params)
    print(f"查询结果数量: {len(results)}")
    for row in results:
        print(row)

if __name__ == "__main__":
    test_templates() 