"""
结构化检索功能模块
"""

import streamlit as st
import logging
from utils.oracle_json_store import OracleJsonStore
from typing import Dict, Any, List

# 配置日志
logger = logging.getLogger(__name__)

def search_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """搜索结构化文档"""
    try:
        with OracleJsonStore() as json_store:
            results = json_store.search_documents(query, top_k)
            return results
    except Exception as e:
        logger.error(f"搜索文档失败: {str(e)}")
        return []

def display_search_results(query: str, results: List[Dict[str, Any]]):
    """显示搜索结果"""
    try:
        # 显示匹配的文档数量
        st.subheader(f"📄 找到 {len(results)} 个相关文档")
        
        # 显示每个文档的详细信息
        for doc in results:
            if isinstance(doc['doc_json'], dict):
                data = doc['doc_json']
                patient_name = data.get("患者姓名", "未知")
                
                # 使用expander为每个患者创建折叠面板
                with st.expander(f"📋 {patient_name}", expanded=False):
                    # 创建标签页
                    tabs = st.tabs([
                        "基本信息", "主诉与诊断", "现病史", 
                        "生命体征", "生化指标", "诊疗经过"
                    ])
                    
                    with tabs[0]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**患者信息**")
                            info = {
                                "姓名": data.get("患者姓名", "未知"),
                                "性别": data.get("性别", "未知"),
                                "年龄": data.get("年龄", "未知"),
                                "民族": data.get("民族", "未知"),
                                "职业": data.get("职业", "未知"),
                                "婚姻状况": data.get("婚姻状况", "未知")
                            }
                            st.json(info)
                        with col2:
                            st.markdown("**住院信息**")
                            info = {
                                "入院日期": data.get("入院日期", "未知"),
                                "出院日期": data.get("出院日期", "未知"),
                                "住院天数": data.get("住院天数", "未知"),
                                "出院情况": data.get("出院情况", "未知")
                            }
                            st.json(info)
                    
                    with tabs[1]:
                        st.markdown("**主诉**")
                        st.write(data.get("主诉", "未记录"))
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**入院诊断**")
                            for diag in data.get("入院诊断", []):
                                st.write(f"- {diag}")
                        with col2:
                            st.markdown("**出院诊断**")
                            for diag in data.get("出院诊断", []):
                                st.write(f"- {diag}")
                    
                    with tabs[2]:
                        st.markdown("**现病史**")
                        for item in data.get("现病史", []):
                            st.write(f"- {item}")
                    
                    with tabs[3]:
                        if "生命体征" in data:
                            st.json(data["生命体征"])
                        else:
                            st.info("未记录生命体征")
                    
                    with tabs[4]:
                        if "生化指标" in data:
                            st.json(data["生化指标"])
                        else:
                            st.info("未记录生化指标")
                    
                    with tabs[5]:
                        st.markdown("**诊疗经过**")
                        st.write(data.get("诊疗经过", "未记录"))
                        if "出院医嘱" in data:
                            st.markdown("**出院医嘱**")
                            for advice in data["出院医嘱"]:
                                st.write(f"- {advice}")
                    
    except Exception as e:
        st.error(f"显示搜索结果时发生错误: {str(e)}")
        logger.error("显示搜索结果失败", exc_info=True)

def render_structured_search():
    """显示结构化检索界面"""
    st.title("结构化检索")
    
    try:
        # 创建 OracleJsonStore 实例
        logger.info("开始初始化 OracleJsonStore...")
        with OracleJsonStore() as json_store:
            # 获取所有文档信息
            docs_sql = """
            SELECT doc_info, created_at 
            FROM DOCUMENT_JSON 
            ORDER BY created_at DESC
            """
            logger.info(f"执行SQL查询: {docs_sql}")
            docs = json_store.execute_search(docs_sql)
            logger.info(f"查询结果: {docs}")
            
            # 显示文档列表
            st.write("调试信息 - 文档列表:", docs)  # 临时调试输出
            st.subheader("📚 已导入的结构化文档")
            
            if docs:
                logger.info(f"找到 {len(docs)} 个文档")
                for doc in docs:
                    logger.info(f"处理文档: {doc}")
                    created_time = doc.get('created_at', '未知时间')
                    doc_info = doc.get('doc_info', '未知文档')
                    st.write(f"- {doc_info} (导入时间: {created_time})")
            else:
                logger.warning("未找到任何文档")
                st.warning("数据库中还没有结构化文档")
                return
                
            # 获取用户输入
            query = st.text_input("请输入查询内容（支持结构化数据搜索）")
            
            if not query:
                return
                
            # 搜索文档
            results = search_documents(query)
            
            if not results:
                st.warning("未找到相关信息")
                return
                
            # 显示搜索结果
            display_search_results(query, results)
                
    except Exception as e:
        logger.error(f"检索文档失败", exc_info=True)
        st.error(f"检索文档失败: {str(e)}")
        # 显示详细的错误信息
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    render_structured_search() 