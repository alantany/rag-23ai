import streamlit as st
from serpapi import GoogleSearch
from utils.common import client

def serpapi_search_qa(query, num_results=3):
    debug_container = st.empty()
    with debug_container:
        st.write("---调试信息---")
        st.write(f"开始搜索: {query}")
        params = {
            "engine": "google",
            "q": query,
            "api_key": "04fec5e75c6f477225ce29bc358f4cc7088945d0775e7f75721cd85b36387125",
            "num": num_results
        }
        st.write("搜索参数:", params)
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            st.write("搜索结果:", results)
            organic_results = results.get("organic_results", [])
            
            if not organic_results:
                return "没有找到相关结果。"
            
            snippets = [result.get("snippet", "") for result in organic_results]
            links = [result.get("link", "") for result in organic_results]
            
            search_results = "\n".join([f"{i+1}. {snippet} ({link})" for i, (snippet, link) in enumerate(zip(snippets, links))])
            st.write("处理后的搜索结果:", search_results)  # 调试信息
            
            prompt = f"""问题: {query}
搜索结果:
{search_results}

请根据上述搜索结果回答问题。如果搜索结果不足以回答问题，请说"根据搜索结果无法回答问题"。"""

            st.write("发送到OpenAI的提示:", prompt)  # 调试信息
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手，能够根据搜索结果回答问题。"},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content.strip()
            st.write("AI回答:", result)  # 调试信息
            return result
        except Exception as e:
            error_msg = f"搜索过程中出错: {str(e)}"
            st.error(error_msg)
            st.write(f"错误类型: {type(e)}")
            st.write(f"错误详情: {str(e)}")
            return error_msg

def direct_qa(query):
    debug_container = st.empty()
    with debug_container:
        st.write("---调试信息---")
        st.write(f"直接问答: {query}")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手，能够回答各种问题。请用中文回答。"},
                    {"role": "user", "content": query}
                ]
            )
            result = response.choices[0].message.content.strip()
            st.write("AI回答:", result)
            return result
        except Exception as e:
            error_msg = f"问答过程中出错: {str(e)}"
            st.error(error_msg)
            st.write(f"错误类型: {type(e)}")
            st.write(f"错误详情: {str(e)}")
            return error_msg

def render():
    st.header("网络搜索问答")
    
    # 添加使用说明
    st.info("""
    使用说明：
    1. 直接提问：输入任何问题，AI将直接回答
    2. 网络搜索：以"搜索"开头，如"搜索 最新的AI技术发展"
    """)

    # 创建一个容器来放置对话历史
    web_chat_container = st.container()

    with web_chat_container:
        for message in st.session_state.web_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"]) 