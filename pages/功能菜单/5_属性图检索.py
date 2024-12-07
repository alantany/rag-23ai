"""
属性图检索功能模块
"""

import streamlit as st
from modules.property_graph.graph_query import PropertyGraphQuery
from utils.oracle_property_graph import OraclePropertyGraph
import json
import pandas as pd
from openai import OpenAI
from collections import defaultdict
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def analyze_symptoms_with_openai(prompt):
    """使用OpenAI API分析症状相似度"""
    try:
        # 获取所有必要的环境变量
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = os.getenv('OPENAI_API_BASE')
        model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        if not all([api_key, api_base]):
            st.error("缺少必要的OpenAI配置信息")
            return None
            
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的医学顾问，擅长分析患者症状的相似���。请从医学专业的角度分析症状的相关性和可能的病因。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"分析过程中出现错误：{str(e)}")
        return None

def render_property_graph_demo():
    """渲染属性图演示页面"""
    st.title("属性图检索")
    
    # 初始化查询执行器
    query_executor = PropertyGraphQuery()
    graph = OraclePropertyGraph(query_executor.graph_store)
    
    # 选择查询类型
    query_type = st.selectbox(
        "选择查询类型",
        [
            "患者基本信息查询",
            "患者症状查询",
            "患者诊疗经过查询",
            "关系类型统计",
            "患者数据统计",
            "相似患者分析"
        ]
    )
    
    # 根据查询类型显示不同的输入和执行不同的查询
    if query_type == "相似患者分析":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5)
        if st.button("分析"):
            # 获取所有患者的症状数据
            query = """SELECT DISTINCT v.entity_name as patient_name, e.relation_type as relation_type, v2.entity_value as entity_value, JSON_VALUE(v2.entity_value, '$.症状') as symptom_path1, JSON_VALUE(v2.entity_value, '$.主诉') as symptom_path2 FROM MATCH (v) -[e]-> (v2) WHERE v.entity_type = '患者' AND e.relation_type IN ('现病史', '主诉')"""
            
            results = query_executor.execute_graph_query(query)
            if results:
                # 处理数据用于展示
                processed_results = []
                patient_symptoms = defaultdict(lambda: defaultdict(list))
                
                for row in results:
                    # 获取症状（优先使用 symptom_path1，如果为空则使用 symptom_path2）
                    symptom = row.get('symptom_path1') or row.get('symptom_path2')
                    if symptom:
                        processed_results.append({
                            '患者姓名': row['patient_name'],
                            '关系类型': row['relation_type'],
                            '症状': symptom
                        })
                        # 为相似性分析收集数据
                        patient_symptoms[row['patient_name']][row['relation_type']].append(symptom)

                # 构建提示信息
                target_patient = patient_name
                prompt = f"""请分析以下患者的症状相似度。

目标患者 {target_patient}:
主诉: {', '.join(patient_symptoms[target_patient]['主诉']) if target_patient in patient_symptoms and patient_symptoms[target_patient]['主诉'] else '无'}
现病史: {', '.join(patient_symptoms[target_patient]['现病史']) if target_patient in patient_symptoms and patient_symptoms[target_patient]['现病史'] else '无'}

其他患者症状：
"""
                for name, symptoms in patient_symptoms.items():
                    if name != target_patient:
                        prompt += f"\n{name}:"
                        prompt += f"\n主诉: {', '.join(symptoms['主诉']) if symptoms['主诉'] else '无'}"
                        prompt += f"\n现病史: {', '.join(symptoms['现病史']) if symptoms['现病史'] else '无'}\n"

                prompt += "\n请分析这些患者中哪些与目标患者的症状相似，说明相似原因，并给出相似度评分（0-100分）。请从医学专业的角度进行分析。"

                # 调用OpenAI API进行分析
                analysis = analyze_symptoms_with_openai(prompt)
                if analysis:
                    st.write("### 症状相似度分析结果")
                    st.write(analysis)
            else:
                st.warning("未找到任���症状记录")
    
    elif query_type == "患者数据统计":
        if st.button("统计"):
            # 获取所有患者的异常指标
            abnormal_labs = {}
            patients = ["周某某", "蒲某某", "马某某"]  # 这里应该从数据库获取所有患者列表
            
            for patient in patients:
                results = graph.get_patient_abnormal_labs(patient)
                abnormal_labs[patient] = len(results) if results else 0
                
            # 获取症状和诊断统计
            query = """SELECT DISTINCT v.entity_name as patient, COUNT(DISTINCT CASE WHEN e.relation_type = '现病史' THEN v2.entity_id END) as symptom_count, COUNT(DISTINCT CASE WHEN e.relation_type = '入院诊断' THEN v2.entity_id END) as diagnosis_count FROM MATCH (v) -[e]-> (v2) WHERE v.entity_type = '患者' GROUP BY v.entity_name"""
            
            results = query_executor.execute_graph_query(query)
            if results:
                # 将结果转换为DataFrame
                df = pd.DataFrame(results)
                df.columns = ['患者姓名', '症状数量', '诊断数量']
                
                # 添加异常指标数量
                df['异常指标数量'] = df['患者姓名'].map(abnormal_labs)
                
                st.write("### 患者数据统计")
                st.dataframe(df)
            else:
                st.warning("未找到任何数据")
    
    elif query_type == "患者症状查询":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        if st.button("查询"):
            results = graph.get_patient_symptoms(patient_name)
            if results:
                st.write("### 患者症状")
                for row in results:
                    if row['symptom']:
                        st.write(f"- {row['symptom']}")
            else:
                st.warning("未找到任何症状记录")
    
    elif query_type == "患者诊疗经过查询":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        if st.button("查询"):
            query = """SELECT DISTINCT v.entity_name as patient, e.relation_type as relation_type, JSON_VALUE(v2.entity_value, '$.内容') as content FROM MATCH (v) -[e]-> (v2) WHERE v.entity_type = '患者' AND v.entity_name = :patient_name AND e.relation_type = '诊疗经过'"""
            
            results = query_executor.execute_graph_query(query, {'patient_name': patient_name})
            if results:
                st.write("### 诊疗经过")
                for row in results:
                    if row['content']:
                        st.write(f"- {row['content']}")
            else:
                st.warning("未找到任何诊疗记录")
    
    elif query_type == "关系类型统计":
        if st.button("统计"):
            query = """SELECT DISTINCT e.relation_type as relation_type, COUNT(*) as count FROM MATCH (v) -[e]-> (v2) WHERE v.entity_type = '患者' GROUP BY e.relation_type ORDER BY count DESC"""
            
            results = query_executor.execute_graph_query(query)
            if results:
                df = pd.DataFrame(results)
                df.columns = ['关系类型', '数量']
                st.write("### 关系类型统计")
                st.dataframe(df)
            else:
                st.warning("未找到任何关系数据")
    
    elif query_type == "患者基本信息查询":
        patient_name = st.text_input("请输入患者姓名", "周某某")
        if st.button("查询"):
            query = """SELECT DISTINCT v.entity_name as patient, JSON_VALUE(v.entity_value, '$.患者.基本信息.性别') as gender, JSON_VALUE(v.entity_value, '$.患者.基本信息.年龄') as age, JSON_VALUE(v.entity_value, '$.患者.基本信息.入院日期') as admission_date, JSON_VALUE(v.entity_value, '$.患者.基本信息.出院日期') as discharge_date FROM MATCH (v) WHERE v.entity_type = '患者' AND v.entity_name = :patient_name"""
            
            results = query_executor.execute_graph_query(query, {'patient_name': patient_name})
            if results:
                st.write("### 患者基本信息")
                for row in results:
                    st.write(f"姓名：{row['patient']}")
                    st.write(f"性别：{row['gender']}")
                    st.write(f"年龄：{row['age']}")
                    st.write(f"入院日期：{row['admission_date']}")
                    st.write(f"出院日期：{row['discharge_date']}")
            else:
                st.warning("未找到患者信息")

if __name__ == "__main__":
    render_property_graph_demo()