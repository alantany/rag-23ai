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

def analyze_symptoms_with_openai(symptoms_data):
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
        
        # 构建提示词
        prompt = f"""请分析以下患者的症状信息，找出症状之间的相似性和可能的关联：

患者症状信息：
{symptoms_data}

请从以下几个方面进行分析：
1. 症状的相似度和关联性
2. 可能的共同病因
3. 需要注意的医学问题

请用中文回答，并尽可能专业和详细。
"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的医学顾问，擅长分析患者症状的相似度。请从��学专业的角度分析症状的相关性和可能的病因。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"分析过程中出现错误：{str(e)}")
        return None

def render_property_graph_demo():
    st.title("医疗知识图谱检索")
    
    # 创建查询执行器
    query_executor = PropertyGraphQuery()
    graph = OraclePropertyGraph()
    
    # 查询类型选择
    query_type = st.selectbox(
        "选择查询类型",
        ["患者症状查询", "患者诊疗经过查询", "症状相似度分析"]
    )
    
    if query_type == "患者症状查询":
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
            query = """MATCH (v) -[e]-> (v2)
            WHERE v.ENTITY_TYPE = '患者'
            AND v.ENTITY_NAME = :patient_name
            AND e.RELATION_TYPE = '诊疗经过'
            COLUMNS (
                v.ENTITY_NAME AS patient,
                e.RELATION_TYPE AS relation_type,
                JSON_VALUE(v2.ENTITY_VALUE, '$.内容') AS content
            )"""
            
            results = query_executor.execute_graph_query(query, {'patient_name': patient_name})
            if results:
                st.write("### 诊疗经过")
                for row in results:
                    if row['content']:
                        st.write(f"- {row['content']}")
            else:
                st.warning("未找到任何诊疗记录")
                
    elif query_type == "症状相似度分析":
        if st.button("分析所有患者症状"):
            with st.spinner("正在获取症状数据..."):
                # 获取所有患者的症状
                results = graph.find_similar_patients("")
                
                if not results:
                    st.warning("未找到任何症状记录")
                    return
                
                # 按患者组织症状数据
                patient_symptoms = defaultdict(list)
                for row in results:
                    patient_symptoms[row['patient_name']].append(row['symptom'])
                
                # 构建用于分析的文本
                analysis_text = []
                for patient, symptoms in patient_symptoms.items():
                    analysis_text.append(f"{patient}的症状：")
                    for symptom in symptoms:
                        analysis_text.append(f"- {symptom}")
                    analysis_text.append("")  # 添加空行分隔
                
                # 调用OpenAI进行分析
                with st.spinner("正在分析症状相似度..."):
                    analysis_result = analyze_symptoms_with_openai("\n".join(analysis_text))
                    
                    if analysis_result:
                        st.write("### 症状分析结果")
                        st.write(analysis_result)
                    else:
                        st.error("症状分析失败")

if __name__ == "__main__":
    render_property_graph_demo()