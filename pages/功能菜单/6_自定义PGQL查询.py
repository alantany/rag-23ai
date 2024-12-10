import streamlit as st
import pandas as pd
from utils.oracle_graph_store import OracleGraphStore
from utils.logger import logger

def render_custom_pgql_demo():
    st.title("自定义PGQL查询")
    
    # 创建两个选项卡：查询模板和自定义查询
    tab1, tab2 = st.tabs(["查询模板", "自定义查询"])
    
    with tab1:
        st.header("查询模板")
        # TODO: 添加常用查询模板
        
    with tab2:
        st.header("自定义PGQL查询")
        st.write("您可以输入自定义的PGQL查询语句，以下是一些示例：")
        
        # 示例1：查询患者的所有症状
        with st.expander("1. 查询患者的所有症状"):
            st.code("""
SELECT *
FROM GRAPH_TABLE ( MEDICAL_KG
    MATCH (v) -[e]-> (s)
    WHERE v.ENTITY_TYPE = '患者'
    AND e.RELATION_TYPE = '现病史'
    AND s.ENTITY_TYPE = '现病史'
    COLUMNS (
        v.ENTITY_NAME AS patient_name,
        s.ENTITY_VALUE AS symptom
    )
)
            """)
            
        # 示例2：查询特定症状的所有患者
        with st.expander("2. 查询特定症状的所有患者"):
            st.code("""
SELECT *
FROM GRAPH_TABLE ( MEDICAL_KG
    MATCH (v) -[e]-> (s)
    WHERE v.ENTITY_TYPE = '患者'
    AND e.RELATION_TYPE = '现病史'
    AND s.ENTITY_TYPE = '现病史'
    AND JSON_EXISTS(s.ENTITY_VALUE, '$.症状[*]?(@=="发热")')
    COLUMNS (
        v.ENTITY_NAME AS patient_name
    )
)
            """)
            
        # 示例3：查询患者的异常生化指标
        with st.expander("3. 查询患者的异常生化指标"):
            st.code("""
SELECT *
FROM GRAPH_TABLE ( MEDICAL_KG
    MATCH (v) -[e]-> (s)
    WHERE v.ENTITY_TYPE = '患者'
    AND JSON_EXISTS(v.ENTITY_VALUE, '$.生化指标[*]?(@.参考范围=="异常")')
    COLUMNS (
        v.ENTITY_NAME AS patient_name,
        JSON_QUERY(v.ENTITY_VALUE, '$.生化指标[*].项目' WITH ARRAY WRAPPER) AS indicator_name,
        JSON_QUERY(v.ENTITY_VALUE, '$.生化指标[*].结果' WITH ARRAY WRAPPER) AS value,
        JSON_QUERY(v.ENTITY_VALUE, '$.生化指标[*].单位' WITH ARRAY WRAPPER) AS unit,
        JSON_QUERY(v.ENTITY_VALUE, '$.生化指标[*].参考范围' WITH ARRAY WRAPPER) AS reference_range
    )
)
            """)
        
        # 用户输入查询
        query = st.text_area("请输入您的PGQL查询语句：", height=200)
        
        if st.button("执行��询"):
            if not query:
                st.warning("请输入查询语句")
            else:
                try:
                    with OracleGraphStore() as graph_store:
                        results = graph_store.execute_pgql(query)
                        
                        if results:
                            st.success(f"查询成功，找到 {len(results)} 条记录")
                            df = pd.DataFrame(results)
                            st.dataframe(df)
                        else:
                            st.info("未找到匹配的记录")
                except Exception as e:
                    st.error(f"查询执行失败: {str(e)}")
                    logger.error("PGQL查询执行失败", exc_info=True)

if __name__ == "__main__":
    render_custom_pgql_demo() 