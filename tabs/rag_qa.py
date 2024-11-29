import streamlit as st
from utils.common import (
    vectorize_document, save_index, delete_index,
    client, num_tokens_from_string, load_model
)
import os
import jieba
from collections import Counter
import faiss
import pickle

# 初始化model
model = load_model()

def extract_keywords(text, top_k=5):
    words = jieba.cut(text)
    word_count = Counter(words)
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

def search_documents(keywords, file_indices):
    relevant_docs = []
    for file_name, (chunks, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

def rag_qa(query, file_indices, relevant_docs=None):
    keywords = extract_keywords(query)
    if relevant_docs is None:
        relevant_docs = search_documents(keywords, file_indices)
    
    if not relevant_docs:
        return "没有找到相关文档。请尝试使用不同的关键词。", [], ""

    all_chunks = []
    chunk_to_file = {}
    combined_index = faiss.IndexFlatL2(384)
    
    offset = 0
    for file_name in relevant_docs:
        if file_name in file_indices:
            chunks, index = file_indices[file_name]
            all_chunks.extend(chunks)
            for i in range(len(chunks)):
                chunk_to_file[offset + i] = file_name
            combined_index.add(index.reconstruct_n(0, index.ntotal))
            offset += len(chunks)

    if not all_chunks:
        return "没有找到相关信息。请确保已上传文档。", [], ""

    query_vector = model.encode([query])
    D, I = combined_index.search(query_vector, k=3)
    context = []
    context_with_sources = []
    for i in I[0]:
        if 0 <= i < len(all_chunks):
            chunk = all_chunks[i]
            context.append(chunk)
            file_name = chunk_to_file.get(i, "未知文件")
            context_with_sources.append((file_name, chunk))

    context_text = "\n".join(context)
    
    max_context_tokens = 3000
    while num_tokens_from_string(context_text) > max_context_tokens:
        context_text = context_text[:int(len(context_text)*0.9)]
    
    if not context_text:
        return "没有找到相关信息。", [], ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。"},
            {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
        ]
    )
    answer = response.choices[0].message.content
    
    if "相关原文：" in answer:
        answer_parts = answer.split("相关原文：", 1)
        main_answer = answer_parts[0].strip()
        relevant_excerpt = answer_parts[1].strip()
    else:
        main_answer = answer.strip()
        relevant_excerpt = ""
    
    if not relevant_excerpt and context:
        relevant_excerpt = context[0][:200] + "..."
    
    relevant_sources = []
    if relevant_excerpt:
        for file_name, chunk in context_with_sources:
            if relevant_excerpt in chunk:
                relevant_sources.append((file_name, chunk))
                break
    if not relevant_sources and context_with_sources:
        relevant_sources.append(context_with_sources[0])

    return main_answer, relevant_sources, relevant_excerpt

def render():
    st.header("RAG 问答")

    # 文档上传部分
    st.subheader("文档上传")
    max_tokens = 4096
    uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="rag_file_uploader_1")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                chunks, index = vectorize_document(uploaded_file, max_tokens)
                st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                save_index(uploaded_file.name, chunks, index)
            st.success(f"文档 {uploaded_file.name} 向量化并添加到索引中！")

    # 显示已处理的文件并添加删除按钮
    st.subheader("已处理文档:")
    for file_name in list(st.session_state.file_indices.keys()):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"• {file_name}")
        with col2:
            if st.button("删除", key=f"delete_{file_name}"):
                del st.session_state.file_indices[file_name]
                delete_index(file_name)
                st.success(f"文档 {file_name} 已删除！")
                st.rerun()

    # 添加关键词搜索功能
    st.subheader("关键词搜索")
    search_keywords = st.text_input("输入关键词（用空格分隔）", key="rag_search_keywords_1")
    if search_keywords:
        keywords = search_keywords.split()
        relevant_docs = search_documents(keywords, st.session_state.file_indices)
        if relevant_docs:
            st.write("相关文档：")
            for doc in relevant_docs:
                st.write(f"• {doc}")
            st.session_state.relevant_docs = relevant_docs
        else:
            st.write("没有找到相关文档。")
            st.session_state.relevant_docs = None

    # 对话部分
    st.subheader("对话")
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.rag_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    st.markdown("**参考来源：**")
                    file_name, _ = message["sources"][0]
                    st.markdown(f"**文件：** {file_name}")
                    if os.path.exists(f'indices/{file_name}.pkl'):
                        with open(f'indices/{file_name}.pkl', 'rb') as f:
                            file_content = pickle.load(f)[0]
                        st.download_button(
                            label="下载源文件",
                            data='\n'.join(file_content),
                            file_name=file_name,
                            mime='text/plain',
                            key=f"download_{i}"
                        )
                if "relevant_excerpt" in message:
                    st.markdown(f"**相关原文：** <mark>{message['relevant_excerpt']}</mark>", unsafe_allow_html=True) 