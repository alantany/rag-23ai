"""
Interactive Test for Property Graph Query Parser
"""

from .query_parser import GraphQueryParser
from ..graph_query import PropertyGraphQuery
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def interactive_test():
    """Interactive test for the query parser."""
    parser = GraphQueryParser()
    query_executor = PropertyGraphQuery()
    
    print("\n欢迎使用医疗知识图谱自然语言查询系统！")
    print("\n您可以用自然语言提问，比如：")
    print("- 查询周某某的基本信息")
    print("- 周某某现在有什么症状？")
    print("- 分析周某某的症状和诊断之间的关系")
    print("- 查看周某某的生命体征和生化指标的关联")
    print("- 分析周某某的诊断结果和化验指标之间的关系")
    print("\n您可以选择任意两种数据类型进行关联分析：")
    print("- 现病史(症状)")
    print("- 诊断结果")
    print("- 生命体征")
    print("- 生化指标")
    print("\n您可以像跟人聊天一样，用自然语言提问。")
    print("输入 'q' 或 'quit' 退出")
    
    while True:
        print("\n" + "="*50)
        query = input("\n请用自然语言提问：").strip()
        
        if query.lower() in ['q', 'quit', 'exit']:
            print("\n感谢使用！再见！")
            break
            
        if not query:
            continue
            
        try:
            logger.info(f"收到查询: {query}")
            
            # 解析查询
            template_name, params = parser.parse_query(query)
            logger.info(f"解析结果 - 模板: {template_name}, 参数: {params}")
            
            # 获取查询语句
            graph_query = parser.get_graph_query(template_name, params)
            logger.info(f"生成的查询语句: {graph_query}")
            
            # 执行查询
            results = query_executor.execute_graph_query(graph_query, params)
            logger.info(f"查询结果数量: {len(results) if results else 0}")
            
            if not results:
                print("\n没有找到相关信息。")
                continue
                
            print(f"\n查询结果：")
            
            # 根据不同的查询类型格式化输出
            if template_name == "patient_info":
                print(f"\n患者信息：")
                for row in results:
                    print(f"- {row['record_type']}: {row['detail']}")
                    
            elif template_name == "patient_symptoms":
                print(f"\n症状信息：")
                for row in results:
                    try:
                        import json
                        symptom_data = json.loads(row['symptoms'])
                        print(f"- {symptom_data.get('症状', row['symptoms'])}")
                    except:
                        print(f"- {row['symptoms']}")
                        
            elif template_name == "treatment_process":
                print(f"\n诊疗经过：")
                for row in results:
                    try:
                        import json
                        process_data = json.loads(row['treatment_process'])
                        print(f"- {process_data.get('内容', row['treatment_process'])}")
                    except:
                        print(f"- {row['treatment_process']}")
                        
            elif template_name == "common_symptoms":
                print(f"\n症状统计：")
                for row in results:
                    print(f"- {row['symptom']}: {row['count']}例")
                    
            elif template_name == "medical_data_correlation":
                print(f"\n医疗数据关联分析：")
                print(f"\n患者：{results[0]['patient_name']}")
                
                # 获取数据类型
                data_type1 = params.get('data_type1', '现病史')
                data_type2 = params.get('data_type2', '诊断')
                
                # 打印表头
                print(f"\n{data_type1}\t\t{data_type2}")
                print("-" * 50)
                
                # 打印关联数据
                for row in results:
                    # 根据数据类型选择要显示的字段
                    value1 = None
                    value2 = None
                    
                    if data_type1 == '现病史':
                        value1 = row.get('symptom')
                    elif data_type1 == '诊断':
                        value1 = row.get('diagnosis')
                    elif data_type1 == '生命体征':
                        value1 = row.get('vital_sign')
                    elif data_type1 == '生化指标':
                        value1 = row.get('lab_result')
                        
                    if data_type2 == '现病史':
                        value2 = row.get('symptom')
                    elif data_type2 == '诊断':
                        value2 = row.get('diagnosis')
                    elif data_type2 == '生命体征':
                        value2 = row.get('vital_sign')
                    elif data_type2 == '生化指标':
                        value2 = row.get('lab_result')
                        
                    if value1 and value2:
                        try:
                            # 尝试解析JSON
                            import json
                            try:
                                value1 = json.loads(value1).get('内容', value1)
                            except:
                                pass
                            try:
                                value2 = json.loads(value2).get('内容', value2)
                            except:
                                pass
                        except:
                            pass
                            
                        print(f"{value1}\t\t{value2}")
            
            else:
                print("\n查询结果：")
                for row in results:
                    print(row)
                    
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
            print(f"\n抱歉，处理您的问题时出现错误：{str(e)}")
            print("您可以换个方式问问看？")

if __name__ == "__main__":
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\n感谢使用！再见！")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"\n程序运行出错: {str(e)}") 