"""
Property Graph Demo Page
"""

import streamlit as st
from modules.property_graph.graph_query import PropertyGraphQuery
import json

def render_property_graph_demo():
    """渲染属性图演示页面"""
    st.title("医疗知识图谱查询")
    
    # 初始化查询执行器
    query_executor = PropertyGraphQuery()
    
    # 选择查询类型
    query_type = st.selectbox(
        "选择查询类型",
        [
            "患者基本信息查询",
            "患者症状查询",
            "患者诊疗经过查询",
            "关系类型统计",
            "患者数据统计"
        ]
    )
    
    # 根据查询类型显示不同的输入和执行不同的查询
    if query_type == "患者基本信息查询":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        if st.button("查询"):
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
            results = query_executor.execute_graph_query(query, {"patient_name": patient_name})
            if results:
                st.write(f"找到 {len(results)} 条记录:")
                for row in results:
                    # 尝试解析JSON字符串
                    try:
                        detail = json.loads(row['detail'])
                        row['detail'] = detail
                    except:
                        pass
                    st.json(row)
            else:
                st.warning("未找到相关记录")
                
    elif query_type == "患者症状查询":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        if st.button("查询"):
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
            results = query_executor.execute_graph_query(query, {"patient_name": patient_name})
            if results:
                st.write(f"找到 {len(results)} 条症状记录:")
                for row in results:
                    try:
                        symptoms = json.loads(row['symptoms'])
                        row['symptoms'] = symptoms
                    except:
                        pass
                    st.json(row)
            else:
                st.warning("未找到症状记录")
                
    elif query_type == "患者诊疗经过查询":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        if st.button("查询"):
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
            results = query_executor.execute_graph_query(query, {"patient_name": patient_name})
            if results:
                st.write(f"找到 {len(results)} 条诊疗记录:")
                for row in results:
                    try:
                        process = json.loads(row['treatment_process'])
                        row['treatment_process'] = process
                    except:
                        pass
                    st.json(row)
            else:
                st.warning("未找到诊疗记录")
                
    elif query_type == "关系类型统计":
        if st.button("统计"):
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
            if results:
                st.write("关系类型分布:")
                # 准备图表数据
                relation_types = [row['relation_type'] for row in results]
                counts = [row['count'] for row in results]
                # 绘制柱状图
                st.bar_chart(dict(zip(relation_types, counts)))
                # 显示详细数据
                st.write("详细统计:")
                st.table(results)
            else:
                st.warning("未找到任何关系数据")
                
    elif query_type == "患者数据统计":
        if st.button("统计"):
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
            if results:
                st.write("患者数据统计:")
                # 显示表格
                st.table(results)
                
                # 准备图表数据
                patients = [row['patient_name'] for row in results]
                symptom_counts = [row['symptom_count'] for row in results]
                diagnosis_counts = [row['diagnosis_count'] for row in results]
                vital_counts = [row['vital_sign_count'] for row in results]
                lab_counts = [row['lab_result_count'] for row in results]
                
                # 绘制多系列柱状图
                chart_data = {
                    "患者": patients,
                    "症状数": symptom_counts,
                    "诊断数": diagnosis_counts,
                    "生命体征数": vital_counts,
                    "生化指标数": lab_counts
                }
                st.bar_chart(chart_data)
            else:
                st.warning("未找到任何患者数据")

if __name__ == "__main__":
    render_property_graph_demo() 