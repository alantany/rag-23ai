"""
图数据检索功能模块
"""

import streamlit as st
from utils.oracle_graph_store import OracleGraphStore
import logging
import json
import os
import streamlit.components.v1 as components
from pathlib import Path
import tempfile
from pyvis.network import Network
from typing import Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

def visualize_patient_graph(patient_info: Dict[str, Any]) -> str:
    """使用pyvis可视化患者的属性图"""
    try:
        # 创建网络图
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=False)
        
        # 设置物理布局选项
        net.set_options("""
        {
          "nodes": {
            "font": {
              "size": 14,
              "face": "Microsoft YaHei"
            }
          },
          "edges": {
            "color": {
              "color": "#666666",
              "highlight": "#000000"
            }
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.5
            }
          }
        }
        """)
        
        # 添加患者节点（中心节点）
        patient_name = patient_info.get('姓名', '未知患者')
        st.write(f"正在处理患者: {patient_name}")  # 调试信息
        
        net.add_node(patient_name, 
                    label=patient_name,
                    color='#add8e6',  # lightblue
                    size=30,
                    shape='circle')
        
        # 添加基本信息节点
        basic_info = patient_info.get('基本信息', {})
        if basic_info:
            st.write("添加基本信息节点")  # 调试信息
            for key, value in basic_info.items():
                node_id = f'basic_{key}'
                net.add_node(node_id,
                            label=f'{key}：{value}',
                            color='#90EE90',  # lightgreen
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='基本信息')
        
        # 添加主诉与诊断节点
        if '主诉与诊断' in patient_info:
            st.write("添加主诉与诊断节点")  # 调试信息
            for i, item in enumerate(patient_info['主诉与诊断']):
                node_id = f'diag_{i}'
                net.add_node(node_id,
                            label=f"{item.get('类型')}：{item.get('内容')}",
                            color='#FFB6C1',  # lightpink
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='主诉与诊断')
        
        # 添加现病史节点
        if '现病史' in patient_info:
            st.write("添加现病史节点")  # 调试信息
            for i, item in enumerate(patient_info['现病史']):
                node_id = f'hist_{i}'
                net.add_node(node_id,
                            label=f"{item.get('症状')}：{item.get('描述')}",
                            color='#FFFFE0',  # lightyellow
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='现病史')
        
        # 添加生命体征节点
        if '生命体征' in patient_info:
            st.write("添加生命体征节点")  # 调试信息
            for i, item in enumerate(patient_info['生命体征']):
                node_id = f'vital_{i}'
                net.add_node(node_id,
                            label=f"{item.get('指标')}：{item.get('数值')}",
                            color='#F08080',  # lightcoral
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='生命体征')
        
        # 添加生化指标节点
        if '生化指标' in patient_info:
            st.write("添加生化指标节点")  # 调试信息
            for i, item in enumerate(patient_info['生化指标']):
                node_id = f'biochem_{i}'
                net.add_node(node_id,
                            label=f"{item.get('项目')}：{item.get('结果')}",
                            color='#DDA0DD',  # plum
                            size=20,
                            shape='box')
                net.add_edge(patient_name, node_id, title='生化指标')
        
        # 创建临时文件保存HTML
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
            try:
                net.save_graph(f.name)
                st.write(f"图形已保存到: {f.name}")  # 调试信息
                return f.name
            except Exception as e:
                st.error(f"保存图形时出错: {str(e)}")  # 调试信息
                raise
        
    except Exception as e:
        st.error(f"生成图形时出错: {str(e)}")  # 调试信息
        raise

def display_parsed_documents():
    """显示已解析的文档"""
    st.subheader("查看已解析的文档")
    
    try:
        with OracleGraphStore() as graph_store:
            patients = graph_store.get_all_patients()
            
            if not patients:
                st.info("📭 数据库中暂无结构化文档，请先在文档管理中上传并结构化文档")
                return
                
            st.write("已解析的文档中包含以下患者：")
            
            # 显示患者列表
            for patient in patients:
                patient_name = patient.get('姓名', '未知患者')
                # 使用expander使每个患者的信息默认折叠
                with st.expander(f"📋 {patient_name}", expanded=False):
                    # 获取患者的完整信息
                    patient_info = graph_store.get_patient_info(patient_name)
                    if patient_info:
                        # 创建两个标签页
                        tab1, tab2 = st.tabs(["知识图谱", "完整数据"])
                        
                        with tab1:
                            try:
                                # 创建并显示交互式网络图
                                html_path = visualize_patient_graph(patient_info)
                                with open(html_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                components.html(html_content, height=600)
                                # 清理临时文件
                                os.unlink(html_path)
                            except Exception as e:
                                st.error(f"显示图形时出错: {str(e)}")
                        
                        with tab2:
                            st.json(patient_info)
                    else:
                        st.error("无法获取患者详细信息")
                            
    except Exception as e:
        logger.error(f"显示已解析文档失败: {str(e)}")
        st.error(f"显示已解析文档时出现错误: {str(e)}")

def render_graph_search():
    """显示图数据检索界面"""
    st.title("图数据检索")
    
    # 显示已解析的文档
    display_parsed_documents()
    
    # 添加搜索框
    st.subheader("搜索")
    query_text = st.text_input(
        "请输入搜索内容（支持按患者姓名、症状、诊断等搜索）",
        placeholder="例如：查询张某的年龄"
    )
    
    # 执行搜索
    if query_text:
        try:
            with OracleGraphStore() as graph_store:
                # 根据查询内容搜索实体
                results = graph_store.search_entities(
                    entity_value=query_text
                )
                
                if results:
                    st.success(f"找到 {len(results)} 条相关记录")
                    st.json(results)
                else:
                    st.warning("未找到相关记录")
                    
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            st.error(f"搜索时出现错误: {str(e)}")

if __name__ == "__main__":
    render_graph_search() 