"""
属性图检索功能模块
"""

import streamlit as st
from utils.oracle_property_graph import OraclePropertyGraph
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
                {"role": "system", "content": "你是一个专业的医学顾问，擅长分析患者症状的相似度。请从医学专业的角度分析症状的相关性和可能的病因。"},
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
        ["症状相似度分析", "患者生化指标异常关联", "患者治疗方案对比"]
    )
    
    if query_type == "症状相似度分析":
        patient_name = st.text_input("请输入患者姓名", "马某某")
        if st.button("分析"):
            with st.spinner("正在获取症状数据..."):
                try:
                    graph = OraclePropertyGraph()
                    # 获取所有患者的症状
                    logger.info("开始调用find_similar_patients")
                    results = graph.find_similar_patients(patient_name)
                    logger.info("find_similar_patients返回结果: %r", results)
                    
                    if not results:
                        st.warning("未找到任何症状记录")
                        return
                    
                    # 按患者组织症状数据
                    patient_symptoms = defaultdict(list)
                    for row in results:
                        try:
                            # 解析目标患者的症状
                            symptom1 = json.loads(row['symptom1'])
                            symptom1_text = symptom1.get('症状', '')
                            if symptom1_text:
                                patient_symptoms[row['patient1']].append(symptom1_text)
                                
                            # 解析其他患者的症状
                            symptom2 = json.loads(row['symptom2'])
                            symptom2_text = symptom2.get('症状', '')
                            if symptom2_text:
                                patient_symptoms[row['patient2']].append(symptom2_text)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    
                    # 构建用分析的文本
                    analysis_text = []
                    analysis_text.append(f"目标患者 {patient_name} 的症状：")
                    if patient_name in patient_symptoms:
                        # 去重症状
                        unique_symptoms = list(set(patient_symptoms[patient_name]))
                        for symptom in unique_symptoms:
                            analysis_text.append(f"- {symptom}")
                    
                    analysis_text.append("\n其他患者的症状：")
                    for p_name, symptoms in patient_symptoms.items():
                        if p_name != patient_name:
                            analysis_text.append(f"\n{p_name} 的症状：")
                            # 去重症状
                            unique_symptoms = list(set(symptoms))
                            for symptom in unique_symptoms:
                                analysis_text.append(f"- {symptom}")
                    
                    # 调用OpenAI进行分析
                    with st.spinner("正在分析症状相似度..."):
                        analysis_result = analyze_symptoms_with_openai("\n".join(analysis_text))
                        
                        if analysis_result:
                            st.write("### 症状分析结果")
                            st.write(analysis_result)
                        else:
                            st.error("症状分析失败")
                except Exception as e:
                    st.error(f"分析过程中出现错误：{str(e)}")
                    raise  # 添加这行以便看到完整的错误堆栈
                    
    elif query_type == "患者生化指标异常关联":
        patient_name = st.selectbox("选择患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
        if st.button("分析"):
            with st.spinner("正在分析生化指标异常关联..."):
                try:
                    with OracleGraphStore() as graph_store:
                        # 使用PGQL查询异常生化指标
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
                            for result in results:
                                try:
                                    st.write(f"### {result['indicator']}")
                                    st.write(f"数值: {result['value']}")
                                    st.write(f"单位: {result['unit']}")
                                    st.write(f"参考范围: {result['reference']}")
                                except (KeyError, TypeError):
                                    continue
                        else:
                            st.info("未找到异常生化指标记录")
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
                    
    elif query_type == "患者治疗方案对比":
        patient1 = st.selectbox("选择第一个患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
        patient2 = st.selectbox("选择第二个患者", ["马某某", "周某某", "刘某某", "蒲某某", "杨某某"])
        if patient1 != patient2 and st.button("对比"):
            with st.spinner("正在对比治疗方案..."):
                try:
                    graph = OraclePropertyGraph()
                    # 获取两个患者的治疗方案
                    results1 = graph.get_patient_diagnosis(patient1)
                    results2 = graph.get_patient_diagnosis(patient2)
                    
                    if results1 or results2:
                        st.success("成功获取治疗方案")
                        
                        if results1:
                            st.write(f"### {patient1} 的治疗方案")
                            for result in results1:
                                try:
                                    diagnosis = json.loads(result['diagnosis'])
                                    st.write(f"诊断: {diagnosis.get('诊断', '')}")
                                except (json.JSONDecodeError, TypeError):
                                    continue
                                    
                        if results2:
                            st.write(f"### {patient2} 的治疗方案")
                            for result in results2:
                                try:
                                    diagnosis = json.loads(result['diagnosis'])
                                    st.write(f"诊断: {diagnosis.get('诊断', '')}")
                                except (json.JSONDecodeError, TypeError):
                                    continue
                    else:
                        st.info("未找到治疗方案记录")
                except Exception as e:
                    st.error(f"对比失败: {str(e)}")

if __name__ == "__main__":
    render_property_graph_demo()