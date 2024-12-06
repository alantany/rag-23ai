import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

def test_basic_graph_query():
    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN")
    )
    
    cursor = connection.cursor()
    
    try:
        # 检查MEDICAL_ENTITIES表中的数据
        print("检查MEDICAL_ENTITIES表中的数据:")
        query0 = """
        SELECT ENTITY_TYPE, ENTITY_NAME, ENTITY_VALUE
        FROM MEDICAL_ENTITIES
        WHERE ROWNUM <= 5
        """
        cursor.execute(query0)
        results = cursor.fetchall()
        if results:
            print("表中的实体数据示例:")
            for row in results:
                print(f"类型: {row[0]}, 名称: {row[1]}, 值: {row[2]}")
        else:
            print("MEDICAL_ENTITIES表中没有数据")
            
        print("\n查询所有患者节点:")
        query1 = """
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (a)
            WHERE a.ENTITY_TYPE = '患者'
            COLUMNS (
                a.ENTITY_NAME
            )
        )
        """
        cursor.execute(query1)
        results = cursor.fetchall()
        if results:
            print("找到的患者:")
            for row in results:
                print(f"患者名称: {row[0]}")
        else:
            print("未找到任何患者节点")
            
        print("\n查询周某某的相关节点:")
        query2 = """
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (a) -[e]- (b)
            WHERE a.ENTITY_TYPE = '患者' AND a.ENTITY_NAME = '周某某'
            COLUMNS (
                b.ENTITY_TYPE,
                b.ENTITY_NAME,
                e.RELATION_TYPE
            )
        )
        """
        cursor.execute(query2)
        results = cursor.fetchall()
        if results:
            print("查询结果:")
            for row in results:
                print(f"类型: {row[0]}, 名称: {row[1]}, 关系: {row[2]}")
        else:
            print("未找到相关节点")
            
        print("\n查询周某某的症状:")
        query3 = """
        SELECT *
        FROM GRAPH_TABLE ( MEDICAL_KG
            MATCH (a) -[e]- (b)
            WHERE a.ENTITY_TYPE = '患者' 
            AND a.ENTITY_NAME = '周某某'
            AND b.ENTITY_TYPE = '现病史'
            AND e.RELATION_TYPE = '现病史'
            COLUMNS (
                b.ENTITY_VALUE AS symptom_value
            )
        )
        """
        cursor.execute(query3)
        results = cursor.fetchall()
        if results:
            print("症状列表:")
            for row in results:
                symptom_value = row[0]
                if isinstance(symptom_value, str) and symptom_value.startswith('{'):
                    import json
                    try:
                        symptom_dict = json.loads(symptom_value)
                        if '症状' in symptom_dict:
                            print(f"- {symptom_dict['症状']}")
                    except json.JSONDecodeError:
                        print(f"- {symptom_value}")
                else:
                    print(f"- {symptom_value}")
        else:
            print("未找到症状记录")
        
    except Exception as e:
        print(f"查询执行错误: {str(e)}")
        return None
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    test_basic_graph_query() 