# 医疗文档智能检索系统

本系统提供了三种智能检索方式:

1. 基于向量的语义检索
2. 基于结构化JSON的精确检索(含全文索引)
3. 基于知识图谱的关系检索

## 系统架构

系统基于以下技术栈:
- Streamlit: 前端界面框架
- Oracle Database: 后端数据存储
- OpenAI API: 自然语言处理
- Sentence Transformers: 文本向量化

## 检索功能说明

### 1. 向量语义检索

对文档全文内容进行向量化,支持语义相似度搜索:

#### 技术方案
- 使用Sentence Transformers进行向量化
- 存储在Oracle向量列中
- 支持相似度排序和匹配
- 适合处理语义相似的自然语言查询

#### 数据模型
表结构设计:
```sql
CREATE TABLE document_vectors (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_path VARCHAR2(1000),
    content CLOB,
    vector VECTOR(384),  -- 使用Oracle原生向量类型
    metadata CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建向量索引
CREATE INDEX vector_idx ON document_vectors(vector) 
INDEXTYPE IS VECTOR_INDEXTYPE 
PARAMETERS('VECTOR_DIM=384 DISTANCE_METRIC=COSINE');
```

#### 向量化过程
1. 文档分块:
   - 按段落或固定长度分割文档
   - 保持语义完整性
   - 控制每块大小在模型最大输入范围内

2. 向量生成:
```python
# 使用Sentence Transformers生成向量
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = model.encode(text_chunks)
```

#### 查询示例
1. 语义相似度搜索:
```sql
SELECT file_path, content,
       (1 - (vector <=> :query_vector)) as similarity
FROM document_vectors
ORDER BY similarity DESC
FETCH FIRST :top_k ROWS ONLY;
```

### 2. 结构化JSON检索

将非结构化医疗文档转换为结构化JSON格式,支持两种查询方式:

#### 技术方案
- 使用GPT模型解析文档结构
- 存储在Oracle JSON字段中
- 支持JSON路径查询和全文索引查询
- 适合精确字段匹配和关键词搜索

#### 数据模型
表结构设计:
```sql
-- JSON文档表
CREATE TABLE DOCUMENT_JSON (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_info VARCHAR2(500),
    doc_json JSON,
    content CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建全文索引
BEGIN
    ctx_ddl.create_preference('my_lexer', 'CHINESE_VGRAM_LEXER');
END;
/

CREATE INDEX DOC_CONTENT_IDX ON DOCUMENT_JSON(content)
INDEXTYPE IS CTXSYS.CONTEXT
PARAMETERS('
    LEXER my_lexer
    SYNC (ON COMMIT)
');
```

#### JSON结构示例
```json
{
    "患者姓名": "张某",
    "性别": "男",
    "年龄": "45",
    "入院日期": "2023-01-01",
    "出院日期": "2023-01-10",
    "主诉": "发热3天",
    "现病史": [
        {
            "time": "2023-01-01",
            "content": "出现发热症状"
        }
    ],
    "入院诊断": [
        {
            "type": "主要诊断",
            "content": "上呼吸道感染"
        }
    ],
    "生命体征": {
        "体温": "38.5",
        "脉搏": "85",
        "呼吸": "20",
        "血压": "120/80"
    },
    "生化指标": {
        "血常规": {
            "白细胞": "10.5",
            "中性粒细胞": "75%"
        }
    }
}
```

#### 查询示例
1. JSON路径查询:
```sql
-- 查询特定患者的诊断信息
SELECT doc_info, doc_json
FROM DOCUMENT_JSON
WHERE JSON_EXISTS(doc_json, '$.患者姓名?(@=="张某")')
  AND JSON_EXISTS(doc_json, '$.入院诊断[*]?(@.type=="主要诊断")');
```

2. 全文索引查询:
```sql
-- 基于内容的全文搜索
SELECT doc_info, doc_json
FROM DOCUMENT_JSON
WHERE CONTAINS(content, '发热') > 0
ORDER BY SCORE(1) DESC;

-- 结合JSON条件和全文搜索
SELECT doc_info, doc_json
FROM DOCUMENT_JSON
WHERE JSON_EXISTS(doc_json, '$.患者姓名?(@=="张某")')
  AND CONTAINS(content, 'NEAR((发热, 咳嗽), 3)') > 0;
```

### 3. 知识图谱检索

基于文档中提取的实体关系进行图数据检索:

#### 技术方案
- 使用Oracle图数据库存储实体和关系
- 通过GPT模型从文档中提取患者相关的实体关系
- 使用图查询语言进行复杂关系检索

#### 数据模型
表结构设计:
```sql
-- 实体表
CREATE TABLE medical_entities (
    entity_id NUMBER GENERATED ALWAYS AS IDENTITY,
    entity_type VARCHAR2(50),  -- 实体类型:患者/症状/诊断/检查/用药等
    entity_name VARCHAR2(200), -- 实体名称
    entity_value VARCHAR2(200),-- 实体值（用于存储具体的指标值等）
    PRIMARY KEY (entity_id)
);

-- 关系表
CREATE TABLE medical_relations (
    relation_id NUMBER GENERATED ALWAYS AS IDENTITY,
    from_entity_id NUMBER,     -- 起始实体ID(患者)
    to_entity_id NUMBER,       -- 目标实体ID
    relation_type VARCHAR2(50),-- 关系类型
    relation_time DATE,        -- 关系发生时间
    doc_reference VARCHAR2(200),-- 文档引用
    PRIMARY KEY (relation_id),
    FOREIGN KEY (from_entity_id) REFERENCES medical_entities(entity_id),
    FOREIGN KEY (to_entity_id) REFERENCES medical_entities(entity_id)
);
```

#### 实体类型
1. 基本信息实体：
   - 性别
   - 年龄
   - 民族
   - 职业
   - 婚姻状况
   - 入院日期
   - 出院日期
   - 住院天数
   - 出院情况

2. 主诉与诊断实体：
   - 主诉症状
   - 入院诊断
   - 出院诊断

3. 现病史实体：
   - 症状
   - 时间点
   - 病情变化

4. 生命体征实体：
   - 体温
   - 脉搏
   - 呼吸
   - 血压

5. 生化指标实体：
   - 检验项目
   - 检验值
   - 参考范围

6. 诊疗经过实体：
   - 治疗措施
   - 用药情况
   - 检查结果

#### 关系类型
1. 基本信息关系：
   - HAS_GENDER(患者-性别)
   - HAS_AGE(患者-年龄)
   - HAS_ETHNICITY(患者-民族)
   - HAS_OCCUPATION(患者-职业)
   - HAS_MARRIAGE(患者-婚姻状况)
   - ADMISSION_DATE(患者-入院日期)
   - DISCHARGE_DATE(患者-出院日期)
   - HOSPITAL_STAY(患者-住院天数)
   - DISCHARGE_STATUS(患者-出院情况)

2. 主诉与诊断关系：
   - HAS_CHIEF_COMPLAINT(患者-主诉)
   - HAS_ADMISSION_DIAGNOSIS(患者-入院诊断)
   - HAS_DISCHARGE_DIAGNOSIS(患者-出院诊断)

3. 现病史关系：
   - HAS_SYMPTOM(患者-症状)
   - SYMPTOM_TIME(症状-时间)
   - SYMPTOM_CHANGE(症状-变化)

4. 生命体征关系：
   - HAS_VITAL_SIGN(患者-体征)
   - VITAL_SIGN_VALUE(体征-数值)
   - VITAL_SIGN_TIME(体征-时间)

5. 生化指标关系：
   - HAS_LAB_TEST(患者-检验)
   - LAB_TEST_VALUE(检验-结果)
   - LAB_TEST_TIME(检验-时间)
   - LAB_TEST_REFERENCE(检验-参考值)

6. 诊疗经过关系：
   - HAS_TREATMENT(患者-治疗)
   - HAS_MEDICATION(患者-用药)
   - HAS_EXAMINATION(患者-检查)
   - TREATMENT_TIME(治疗-时间)
   - MEDICATION_TIME(用药-时间)
   - EXAMINATION_TIME(检查-时间)

#### 图查询示例
1. 查询患者的所有生命体征变化：
```sql
SELECT p.entity_name as patient,
       v.entity_name as vital_sign,
       v.entity_value as value,
       r.relation_time
FROM medical_entities p
JOIN medical_relations r ON p.entity_id = r.from_entity_id
JOIN medical_entities v ON v.entity_id = r.to_entity_id
WHERE p.entity_type = '患者'
  AND r.relation_type = 'HAS_VITAL_SIGN'
ORDER BY r.relation_time;
```

2. 查询患者的诊断和相关症状：
```sql
SELECT p.entity_name as patient,
       d.entity_name as diagnosis,
       s.entity_name as symptom
FROM medical_entities p
JOIN medical_relations rd ON p.entity_id = rd.from_entity_id
JOIN medical_entities d ON d.entity_id = rd.to_entity_id
JOIN medical_relations rs ON p.entity_id = rs.from_entity_id
JOIN medical_entities s ON s.entity_id = rs.to_entity_id
WHERE p.entity_type = '患者'
  AND rd.relation_type = 'HAS_ADMISSION_DIAGNOSIS'
  AND rs.relation_type = 'HAS_SYMPTOM';
```

3. 查询患者的完整治疗过程：
```sql
SELECT p.entity_name as patient,
       t.entity_name as treatment,
       m.entity_name as medication,
       e.entity_name as examination,
       COALESCE(rt.relation_time, rm.relation_time, re.relation_time) as time
FROM medical_entities p
LEFT JOIN medical_relations rt ON p.entity_id = rt.from_entity_id AND rt.relation_type = 'HAS_TREATMENT'
LEFT JOIN medical_entities t ON t.entity_id = rt.to_entity_id
LEFT JOIN medical_relations rm ON p.entity_id = rm.from_entity_id AND rm.relation_type = 'HAS_MEDICATION'
LEFT JOIN medical_entities m ON m.entity_id = rm.to_entity_id
LEFT JOIN medical_relations re ON p.entity_id = re.from_entity_id AND re.relation_type = 'HAS_EXAMINATION'
LEFT JOIN medical_entities e ON e.entity_id = re.to_entity_id
WHERE p.entity_type = '患者'
  AND p.entity_name = :patient_name
ORDER BY time;
```

## 使用方法

1. 上传医疗文档
2. 系统自动进行三重处理:
   - 结构化解析
   - 向量化存储
   - 关系图谱构建
3. 在搜索框输入查询
4. 系统自动选择最佳检索方式:
   - 结构化字段匹配优先使用JSON检索
   - 语义相似度查询使用向量检索
   - 关系类查询使用图检索

## 部署说明

1. 环境要求:
   - Python 3.8+
   - Oracle Database 21c+
   - Streamlit 1.24+

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 配置数据库:
```bash
python setup_db.py
```

4. 启动应用:
```bash
streamlit run app.py
```
