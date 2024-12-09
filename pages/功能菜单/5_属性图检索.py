"""
属性图检索功能模块
"""

import streamlit as st
from utils.oracle_property_graph import OraclePropertyGraph
from utils.oracle_graph_store import OracleGraphStore
import json
import pandas as pd
from openai import OpenAI
from collections import defaultdict
import os
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

def analyze_with_openai(analysis_data, prompt_template):
    """使用OpenAI API进行分析"""
    try:
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
        
        prompt = prompt_template.format(data=analysis_data)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的医学顾问，擅长分析患者的医疗数据。请从医学专业的角度进行分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"分析过程中出现错误：{str(e)}")
        return None

def render_property_graph_demo():
    st.title("属性图检索")
    
    query_type = st.selectbox(
        "选择查询类型",
        ["症状相似度分析", "患者生化指标异常关联", "患者诊断关系网络"]
    )
    
    if query_type == "症状相似度分析":
        patient_name = st.text_input("请输入患者姓名", "马某某")
        if st.button("分析"):
            with st.spinner("正在获取症状数据..."):
                try:
                    graph = OraclePropertyGraph()
                    results = graph.find_similar_patients(patient_name)
                    
                    if not results:
                        st.warning("未找到任何症状记录")
                        return
                    
                    patient_symptoms = defaultdict(list)
                    for row in results:
                        try:
                            symptom1 = json.loads(row['symptom1'])
                            symptom1_text = symptom1.get('症状', '')
                            if symptom1_text:
                                patient_symptoms[row['patient1']].append(symptom1_text)
                                
                            symptom2 = json.loads(row['symptom2'])
                            symptom2_text = symptom2.get('症状', '')
                            if symptom2_text:
                                patient_symptoms[row['patient2']].append(symptom2_text)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    
                    analysis_text = []
                    analysis_text.append(f"目标患者 {patient_name} 的症状：")
                    if patient_name in patient_symptoms:
                        unique_symptoms = list(set(patient_symptoms[patient_name]))
                        for symptom in unique_symptoms:
                            analysis_text.append(f"- {symptom}")
                    
                    analysis_text.append("\n其他患者的症状：")
                    for p_name, symptoms in patient_symptoms.items():
                        if p_name != patient_name:
                            analysis_text.append(f"\n{p_name} 的症状：")
                            unique_symptoms = list(set(symptoms))
                            for symptom in unique_symptoms:
                                analysis_text.append(f"- {symptom}")
                    
                    prompt_template = """请分析以下患者的症状信息，找出症状之间的相似性和可能的关联：

{data}

请从以下几个方面进行分析：
1. 症状的相似度和关联性
2. 可能的共同病因
3. 需要注意的医学问题

请用中文回答，并尽可能专业和详细。"""
                    
                    with st.spinner("正在分析症状相似度..."):
                        analysis_result = analyze_with_openai("\n".join(analysis_text), prompt_template)
                        if analysis_result:
                            st.write("### 症状分析结果")
                            st.write(analysis_result)
                        else:
                            st.error("症状分析失败")
                except Exception as e:
                    st.error(f"分析过程中出现错误：{str(e)}")
                    logger.error("症状分析失败", exc_info=True)
                    
    elif query_type == "患者生化指标异常关联":
        patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
        if st.button("分析"):
            with st.spinner("正在分析生化指标异常关联..."):
                try:
                    with OracleGraphStore() as graph_store:
                        query = """
                        SELECT *
                        FROM GRAPH_TABLE ( MEDICAL_KG
                            MATCH (v) -[e]-> (i)
                            WHERE v.ENTITY_TYPE = 'PATIENT'
                            AND v.ENTITY_NAME = :patient_name
                            AND e.RELATION_TYPE = 'HAS_INDICATOR'
                            AND i.REFERENCE_RANGE = '异常'
                            COLUMNS (
                                v.ENTITY_NAME AS patient,
                                i.INDICATOR_NAME AS indicator,
                                i.VALUE AS value,
                                i.UNIT AS unit,
                                i.REFERENCE_RANGE AS reference
                            )
                        )
                        """
                        results = graph_store.execute_pgql(query, {'patient_name': patient_name})
                        
                        if results:
                            st.success(f"找到 {len(results)} 个异常生化指标")
                            
                            # 创建数据表格
                            df = pd.DataFrame(results)
                            st.write("### 异常生化指标列表")
                            st.dataframe(df)
                            
                            # 使用OpenAI分析生化指标
                            prompt_template = """请分析以下患者的异常生化指标：

{data}

请从以下几个方面进行分析：
1. 异常指标的分布情况
2. 可能的病理意义
3. 需要关注的健康风险
4. 建议的后续检查

请用中文回答，并尽可能专业和详细。"""
                            
                            with st.spinner("正在分析生化指标..."):
                                analysis_result = analyze_with_openai(df.to_string(), prompt_template)
                                if analysis_result:
                                    st.write("### 生化指标分析结果")
                                    st.write(analysis_result)
                                else:
                                    st.error("生化指标分析失败")
                        else:
                            st.info("未找到异常生化指标记录")
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
                    logger.error("生化指标分析失败", exc_info=True)
                    
    elif query_type == "患者诊断关系网络":
        patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
        if st.button("分析"):
            with st.spinner("正在分析诊断关系网络..."):
                try:
                    graph = OraclePropertyGraph()
                    results = graph.get_patient_diagnosis(patient_name)
                    
                    if results:
                        st.success(f"找到 {len(results)} 条诊断关系记录")
                        
                        # 创建数据表格
                        df = pd.DataFrame(results)
                        df.columns = ['患者1', '患者2', '诊断', '诊断时间']
                        
                        # 显示数据表格
                        st.write("### 诊断关系网络分析结果")
                        st.dataframe(df)
                        
                        # 使用OpenAI分析诊断关系
                        prompt_template = """请分析以下患者的诊断关系信息：

{data}

请从以下几个方面进行分析：
1. 共同诊断情况分析
2. 诊断的合理性评估
3. 需要注意的诊断相互作用
4. 诊疗建议和注意事项

请用中文回答，并尽可能专业和详细。重点关注诊断的共同点和差异点。"""
                        
                        with st.spinner("正在分析诊断关系..."):
                            analysis_result = analyze_with_openai(df.to_string(), prompt_template)
                            if analysis_result:
                                st.write("### 诊断分析结果")
                                st.write(analysis_result)
                            else:
                                st.error("诊断分析失败")
                    else:
                        st.info("未找到诊断关系记录")
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
                    logger.error("诊断关系分析失败", exc_info=True)

if __name__ == "__main__":
    render_property_graph_demo()