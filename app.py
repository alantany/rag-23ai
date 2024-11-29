"""
AI知识问答系统
"""

import streamlit as st
from utils.common import load_all_indices
from tabs import rag_qa, web_qa, data_analysis
import os
import pickle

# 设置页面配置
st.set_page_config(
    page_title="AI知识问答系统 - by Huaiyuan Tan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 添加开发者信息
st.markdown("<h6 style='text-align: right; color: gray;'>开发者: Huaiyuan Tan</h6>", unsafe_allow_html=True)

# 隐藏 Streamlit 默认的菜单、页脚和 Deploy 按钮
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    st.title("AI知识问答系统")

    # 初始化 session state
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "web_messages" not in st.session_state:
        st.session_state.web_messages = []
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()

    # 使用radio来选择当前tab
    current_tab = st.radio("选择功能", ["RAG知识问答", "网络搜索问答", "AI数据分析"], horizontal=True, label_visibility="collapsed")
    st.write("当前功能:", current_tab)  # 调试信息

    # 用户输入
    prompt = st.chat_input("请输入您的问题:", key="chat_input")

    # 处理用户输入
    if prompt:
        st.write("收到输入:", prompt)  # 调试信息
        if current_tab == "RAG知识问答":
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            if st.session_state.file_indices:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("正在生成回答..."):
                        try:
                            relevant_docs = st.session_state.get('relevant_docs')
                            response, sources, relevant_excerpt = rag_qa.rag_qa(prompt, st.session_state.file_indices, relevant_docs)
                            st.markdown(response)
                            if sources:
                                st.markdown("**参考来源：**")
                                file_name, _ = sources[0]
                                st.markdown(f"**文件：** {file_name}")
                                if os.path.exists(f'indices/{file_name}.pkl'):
                                    with open(f'indices/{file_name}.pkl', 'rb') as f:
                                        file_content = pickle.load(f)[0]
                                    st.download_button(
                                        label="下载源文件",
                                        data='\n'.join(file_content),
                                        file_name=file_name,
                                        mime='text/plain',
                                        key=f"download_new_{len(st.session_state.rag_messages)}"
                                    )
                            if relevant_excerpt:
                                st.markdown(f"**相关原文：** <mark>{relevant_excerpt}</mark>", unsafe_allow_html=True)
                            else:
                                st.warning("未能提取到精确的相关原文，但找到相关信息。")
                            st.session_state.rag_messages.append({
                                "role": "assistant", 
                                "content": response, 
                                "sources": sources,
                                "relevant_excerpt": relevant_excerpt
                            })
                        except Exception as e:
                            st.error(f"生成回答时发生错误: {str(e)}")
            else:
                with st.chat_message("assistant"):
                    st.warning("请先上传文档。")
        elif current_tab == "网络搜索问答":
            st.write("进入网络搜索处理...")  # 调试信息
            st.session_state.web_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("正在搜索并生成回答..."):
                    try:
                        st.write(f"当前输入: {prompt}")  # 调试信息
                        st.write(f"是否以'搜索'开头: {prompt.startswith('搜索')}")  # 调试信息
                        if prompt.startswith("搜索"):
                            st.write("执行网络搜索...")  # 调试信息
                            search_query = prompt[2:].strip()
                            st.write(f"搜索关键词: {search_query}")  # 调试信息
                            response = web_qa.serpapi_search_qa(search_query)
                        else:
                            st.write("执行直接问答...")  # 调试信息
                            response = web_qa.direct_qa(prompt)
                        st.markdown(response)
                        st.session_state.web_messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"生成回答时发生错误: {str(e)}")
                        st.write(f"错误类型: {type(e)}")  # 调试信息
                        st.write(f"错误详情: {str(e)}")  # 调试信息

    # 根据当前tab显示相应内容
    if current_tab == "RAG知识问答":
        rag_qa.render()
    elif current_tab == "网络搜索问答":
        web_qa.render()
    else:
        data_analysis.render()

if __name__ == "__main__":
    main()

