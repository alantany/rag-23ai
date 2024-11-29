import numpy as np
from utils.oracle_vector_store import OracleVectorStore

def test_vector_store():
    # 创建测试向量和文档
    test_vectors = [
        np.random.rand(384),  # 384维测试向量
        np.random.rand(384)
    ]
    
    test_docs = [
        {
            'file_path': 'test1.txt',
            'content': 'This is a test document 1',
            'metadata': {'type': 'test'}
        },
        {
            'file_path': 'test2.txt',
            'content': 'This is a test document 2',
            'metadata': {'type': 'test'}
        }
    ]
    
    # 使用上下文管理器测试
    with OracleVectorStore() as vector_store:
        # 初始化schema
        vector_store.init_schema()
        
        # 添加向量
        vector_store.add_vectors(test_vectors, test_docs)
        
        # 测试搜索
        query_vector = np.random.rand(384)
        results = vector_store.search_vectors(query_vector, top_k=2)
        
        # 验证结果
        assert len(results) > 0
        assert 'file_path' in results[0]
        assert 'content' in results[0]
        assert 'similarity' in results[0]

if __name__ == '__main__':
    test_vector_store()
    print("测试完成!") 