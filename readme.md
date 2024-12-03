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
CREATE TABLE DOCUMENT_VECTORS (
    ID NUMBER NOT NULL PRIMARY KEY,
    FILE_PATH VARCHAR2(1000),
    CONTENT CLOB,
    VECTOR VECTOR(384, *),
    METADATA CLOB,
    CREATED_AT TIMESTAMP(6)
);

-- 创建向量索引
CREATE INDEX vector_idx ON DOCUMENT_VECTORS(VECTOR) 
INDEXTYPE IS VECTOR_INDEXTYPE 
PARAMETERS('VECTOR_CONFIG=(DIMENSION(384), DISTANCE_METRIC(COSINE))');
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
FROM DOCUMENT_VECTORS
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
    ID NUMBER NOT NULL PRIMARY KEY,
    DOC_INFO VARCHAR2(500),
    DOC_JSON JSON,
    CONTENT CLOB
);

-- 创建JSON索引
CREATE INDEX json_content_idx ON DOCUMENT_JSON(CONTENT)
INDEXTYPE IS CTXSYS.CONTEXT;
```

#### JSON结构示例
```json
{
    "患者姓名": "张某某",
    "性别": "男",
    "年龄": 45,
    "入院日期": "2024-01-01",
    "出院日期": "2024-01-10",
    "主诉": "发热3天",
    "现病史": [
        "发热",
        "咳嗽",
        "乏力"
    ],
    "入院诊断": [
        "上呼吸道感染",
        "病毒性感冒"
    ],
    "出院诊断": [
        "上呼吸道感染",
        "病毒性感冒",
        "支气管炎"
    ],
    "生命体征": {
        "体温": "38.5℃",
        "血压": "120/80mmHg"
    },
    "生化指标": {
        "白细胞": "10.5×10^9/L↑",
        "血红蛋白": "135g/L",
        "血小板": "223×10^9/L"
    },
    "诊疗经过": "入院后给予退热、止咳等对症治疗...",
    "出院医嘱": [
        "注意休息，避免受凉",
        "规律服药，定期复查"
    ],
    "metadata": {
        "import_time": "2024-12-02T13:50:59.020369",
        "source_type": "text",
        "last_updated": "2024-12-02T13:50:59.021829"
    }
}
```

#### 查询示例
1. JSON路径查询:
```sql
-- 查询特定患者的诊断信息
SELECT doc_info, doc_json
FROM DOCUMENT_JSON
WHERE JSON_EXISTS(doc_json, '$.患者姓名?(@=="张某某")')
  AND JSON_EXISTS(doc_json, '$.入院诊断[*]?(@=="上呼吸道感染")');
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
WHERE JSON_EXISTS(doc_json, '$.患者姓名?(@=="张某某")')
  AND CONTAINS(content, 'NEAR((发热, 咳嗽), 3)') > 0;
```

### 3. 知识图谱检索

基于文档中提取的实体关系进行图数据检索:

#### 技术方案
- 使用Oracle关系表存储实体和关系
- 通过GPT模型从文档中提取患者相关的实体关系
- 使用SQL查询进行复杂关系检索

#### 数据模型
表结构设计:
```sql
-- 实体表
CREATE TABLE MEDICAL_ENTITIES (
    ENTITY_ID NUMBER NOT NULL PRIMARY KEY,
    ENTITY_TYPE VARCHAR2(100),
    ENTITY_NAME VARCHAR2(1000),
    ENTITY_VALUE CLOB,
    DOC_REF VARCHAR2(1000),
    CREATED_AT VARCHAR2(100)
);

-- 创建实体索引
CREATE INDEX entity_type_idx ON MEDICAL_ENTITIES(ENTITY_TYPE);
CREATE INDEX entity_name_idx ON MEDICAL_ENTITIES(ENTITY_NAME);
CREATE INDEX entity_doc_ref_idx ON MEDICAL_ENTITIES(DOC_REF);

-- 关系表
CREATE TABLE MEDICAL_RELATIONS (
    RELATION_ID NUMBER NOT NULL PRIMARY KEY,
    FROM_ENTITY_ID NUMBER,
    TO_ENTITY_ID NUMBER,
    RELATION_TYPE VARCHAR2(100),
    DOC_REFERENCE VARCHAR2(1000),
    CREATED_AT VARCHAR2(100),
    FOREIGN KEY (FROM_ENTITY_ID) REFERENCES MEDICAL_ENTITIES(ENTITY_ID),
    FOREIGN KEY (TO_ENTITY_ID) REFERENCES MEDICAL_ENTITIES(ENTITY_ID)
);

-- 创建关系索引
CREATE INDEX relation_type_idx ON MEDICAL_RELATIONS(RELATION_TYPE);
CREATE INDEX relation_doc_ref_idx ON MEDICAL_RELATIONS(DOC_REFERENCE);
```

#### 实体类型
1. 基本信息实体：
   - 患者姓名
   - 性别
   - 年龄
   - 入院日期
   - 出院日期

2. 主诉与诊断实体：
   - 主诉症状
   - 入院诊断
   - 出院诊断

3. 现病史实体：
   - 症状
   - 病情描述

4. 生命体征实体：
   - 体温
   - 血压

5. 生化指标实体：
   - 检验项目
   - 检验值
   - 参考范围

6. 诊疗经过实体：
   - 诊疗过程
   - 出院医嘱

#### 关系类型
1. 基本信息关系：
   - HAS_GENDER(患者-性别)
   - HAS_AGE(患者-年龄)
   - ADMISSION_DATE(患者-入院日期)
   - DISCHARGE_DATE(患者-出院日期)

2. 主诉与诊断关系：
   - HAS_CHIEF_COMPLAINT(患者-主诉)
   - HAS_ADMISSION_DIAGNOSIS(患者-入院诊断)
   - HAS_DISCHARGE_DIAGNOSIS(患者-出院诊断)

3. 现病史关系：
   - HAS_SYMPTOM(患者-症状)

4. 生命体征关系：
   - HAS_VITAL_SIGN(患者-体征)
   - VITAL_SIGN_VALUE(体征-数值)

5. 生化指标关系：
   - HAS_LAB_TEST(患者-检验)
   - LAB_TEST_VALUE(检验-结果)
   - LAB_TEST_REFERENCE(检验-参考值)

6. 诊疗经过关系：
   - HAS_TREATMENT(患者-诊疗)
   - HAS_DISCHARGE_ADVICE(患者-医嘱)

#### 查询示例
1. 查询患者的所有生命体征：
```sql
SELECT p.entity_name as patient,
       v.entity_name as vital_sign,
       v.entity_value as value
FROM MEDICAL_ENTITIES p
JOIN MEDICAL_RELATIONS r ON p.entity_id = r.from_entity_id
JOIN MEDICAL_ENTITIES v ON v.entity_id = r.to_entity_id
WHERE p.entity_type = '患者'
  AND r.relation_type = 'HAS_VITAL_SIGN';
```

2. 查询患者的诊断和相关症状：
```sql
SELECT p.entity_name as patient,
       d.entity_name as diagnosis,
       s.entity_name as symptom
FROM MEDICAL_ENTITIES p
JOIN MEDICAL_RELATIONS rd ON p.entity_id = rd.from_entity_id
JOIN MEDICAL_ENTITIES d ON d.entity_id = rd.to_entity_id
JOIN MEDICAL_RELATIONS rs ON p.entity_id = rs.from_entity_id
JOIN MEDICAL_ENTITIES s ON s.entity_id = rs.to_entity_id
WHERE p.entity_type = '患者'
  AND rd.relation_type = 'HAS_ADMISSION_DIAGNOSIS'
  AND rs.relation_type = 'HAS_SYMPTOM';
```

## 数据导入导出

系统提供了完整的数据导入导出功能，支持将数据库中的表结构和数据导出为SQL文件，并能在新环境中重新导入。

### 导出功能

使用 `scripts/export_data.py` 可以导出数据：

```bash
python scripts/export_data.py
```

导出的文件将保存在 `db_export` 目录下，每个表生成一个独立的SQL文件：
- `document_vectors_时间戳.sql`
- `document_json_时间戳.sql`
- `medical_entities_时间戳.sql`
- `medical_relations_时间戳.sql`

每个SQL文件包含：
1. 表的创建语句
2. 索引的创建语句
3. 实际数据的INSERT语句

### 导入功能

使用 `scripts/import_data.py` 可以导入数据：

```bash
python scripts/import_data.py
```

导入工具会：
1. 按照正确的顺序导入SQL文件（考虑外键依赖）
2. 自动处理CLOB和JSON等特殊数据类型
3. 显示详细的导入进度和结果

## 环境要求

1. Python 3.11
2. Oracle Database 23ai
3. Streamlit 1.24+

## 安装部署

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置数据库连接：
创建 `.env` 文件并设置以下环境变量：
```bash
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password
ORACLE_DSN=host:port/service_name
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base
OPENAI_MODEL=gpt-3.5-turbo
```

3. 启动应用：
```bash
streamlit run app.py
```

## 使用说明

1. 上传医疗文档
2. 系统自动进行三重处理：
   - 向量化存储
   - JSON结构化解析
   - 知识图谱构建
3. 在搜索框输入查询
4. 系统自动选择最佳检索方式：
   - 语义相似度查询使用向量检索
   - 结构化字段匹配使用JSON检索
   - 关系类查询使用图检索
