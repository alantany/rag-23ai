import oracledb
import os
from dotenv import load_dotenv
import numpy as np

def create_test_table(cursor):
    """创建测试表"""
    try:
        # 先删除已存在的表
        cursor.execute("""
            BEGIN
                EXECUTE IMMEDIATE 'DROP TABLE test_vector';
                EXCEPTION WHEN OTHERS THEN NULL;
            END;
        """)
        
        # 创建向量表 - 使用Oracle原生向量类型
        cursor.execute("""
            CREATE TABLE test_vector (
                id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name VARCHAR2(100),
                vector NUMBER VECTOR(3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建向量列索引
        cursor.execute("""
            CREATE INDEX vector_idx ON test_vector (vector) 
            INDEXTYPE IS VECTOR_INDEXTYPE 
            PARAMETERS('VECTOR_DIM=3 DISTANCE_METRIC=COSINE')
        """)
        print("成功创建测试向量表和索引")
    except Exception as e:
        print(f"创建表时出错: {str(e)}")

def insert_vector(cursor, name, vector):
    """插入向量"""
    try:
        # 使用 SDO_VECTOR_OP.CAST_NUMBER_VECTOR 转换向量
        cursor.execute("""
            INSERT INTO test_vector (name, vector)
            VALUES (:1, SDO_VECTOR_OP.CAST_NUMBER_VECTOR(:2))
        """, [name, vector.tolist()])
        print(f"成功插入向量: {name}")
    except Exception as e:
        print(f"插入向量时出错: {str(e)}")

def search_similar_vectors(cursor, query_vector, top_k=3):
    """搜索相似向量"""
    try:
        # 使用 SDO_VECTOR_OP.SIMILARITY_DISTANCE 计算相似度
        cursor.execute("""
            SELECT name, 
                   SDO_VECTOR_OP.SIMILARITY_DISTANCE(
                       vector, 
                       SDO_VECTOR_OP.CAST_NUMBER_VECTOR(:1)
                   ) as distance
            FROM test_vector
            ORDER BY 
                SDO_VECTOR_OP.SIMILARITY_DISTANCE(
                    vector, 
                    SDO_VECTOR_OP.CAST_NUMBER_VECTOR(:1)
                )
            FETCH FIRST :2 ROWS ONLY
        """, [query_vector.tolist(), top_k])
        
        results = cursor.fetchall()
        print("\n相似向量搜索结果:")
        for name, distance in results:
            similarity = 1 - float(distance)  # 转换距离为相似度
            print(f"名称: {name}, 相似度: {similarity:.4f}")
    except Exception as e:
        print(f"搜索向量时出错: {str(e)}")

def update_vector(cursor, name, new_vector):
    """更新向量"""
    try:
        cursor.execute("""
            UPDATE test_vector 
            SET vector = SDO_VECTOR_OP.CAST_NUMBER_VECTOR(:1)
            WHERE name = :2
        """, [new_vector.tolist(), name])
        print(f"成功更新向量: {name}")
    except Exception as e:
        print(f"更新向量时出错: {str(e)}")

def delete_vector(cursor, name):
    """删除向量"""
    try:
        cursor.execute("""
            DELETE FROM test_vector
            WHERE name = :1
        """, [name])
        print(f"成功删除向量: {name}")
    except Exception as e:
        print(f"删除向量时出错: {str(e)}")

def test_vector_operations():
    """测试向量操作"""
    # 加载环境变量
    load_dotenv()
    
    # 获取数据库连接信息
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")
    
    print("正在测试数据库连接...")
    print(f"连接信息:")
    print(f"用户名: {user}")
    print(f"DSN: {dsn}")
    
    try:
        # 创建连接
        connection = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn
        )
        
        print("\n连接成功!")
        
        # 获取数据库版本信息
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM V$VERSION")
            version = cursor.fetchone()
            print(f"\n数据库版本信息:")
            print(version[0])
            
            # 测试向量操作
            print("\n开始测试向量操作...")
            
            # 1. 创建表
            create_test_table(cursor)
            
            # 2. 插入测试向量
            vector1 = np.array([1.0, 0.0, 0.0])
            vector2 = np.array([0.0, 1.0, 0.0])
            vector3 = np.array([0.0, 0.0, 1.0])
            
            insert_vector(cursor, "向量1", vector1)
            insert_vector(cursor, "向量2", vector2)
            insert_vector(cursor, "向量3", vector3)
            
            # 3. 搜索相似向量
            query_vector = np.array([0.9, 0.1, 0.0])
            print("\n搜索与[0.9, 0.1, 0.0]最相似的向量:")
            search_similar_vectors(cursor, query_vector)
            
            # 4. 更新向量
            new_vector = np.array([0.5, 0.5, 0.0])
            update_vector(cursor, "向量1", new_vector)
            
            # 5. 再次搜索
            print("\n更新后再次搜索:")
            search_similar_vectors(cursor, query_vector)
            
            # 6. 删除向量
            delete_vector(cursor, "向量2")
            
            # 7. 最后一次搜索
            print("\n删除后再次搜索:")
            search_similar_vectors(cursor, query_vector)
            
            # 8. 清理
            cursor.execute("DROP TABLE test_vector")
            print("\n清理完成: 删除测试表")
            
            connection.commit()
        
        connection.close()
        print("\n测试完成!")
        
    except Exception as e:
        print(f"\n测试失败!")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    test_vector_operations() 