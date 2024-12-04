"""
Test Property Graph Query Parser
"""

from .query_parser import GraphQueryParser

def test_query_parser():
    """Test the query parser with different types of queries."""
    parser = GraphQueryParser()
    
    # 测试用例
    test_cases = [
        # 基本查询
        "查询马某某的所有信息",
        "马某某有什么症状",
        "查看马某某的诊疗经过",
        
        # 统计类查询
        "统计所有患者的常见症状",
        "查看最常见的症状",
        
        # 复杂查询
        "查询有发热症状的患者",
        "找出和马某某有相同症状的患者",
        "统计每种症状的患者数量"
    ]
    
    for query in test_cases:
        print(f"\n测试查询: {query}")
        try:
            template_name, params, graph_query = parser.process_query(query)
            print(f"模板名称: {template_name}")
            print(f"参数: {params}")
            print(f"生成的查询:\n{graph_query}")
        except Exception as e:
            print(f"处理查询时出错: {e}")

if __name__ == "__main__":
    test_query_parser() 