-- MEDICAL_RELATIONS 表导出脚本
-- 导出时间: 2024-12-03 22:57:02
-- 此脚本包含表结构、索引和数据


-- 创建关系表
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

-- 插入 MEDICAL_RELATIONS 表数据
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (7, 8, 9, '主诉', '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (8, 8, 10, '现病史', '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (9, 8, 11, '现病史', '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (10, 8, 12, '现病史', '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (11, 8, 13, '现病史', '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (12, 8, 14, '诊疗经过', '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (22, 26, 27, '主诉', '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (23, 26, 28, '现病史', '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (24, 26, 29, '现病史', '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (25, 26, 30, '诊疗经过', '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (13, 15, 16, '主诉', '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (14, 15, 17, '现病史', '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (15, 15, 18, '现病史', '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (16, 15, 19, '现病史', '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (17, 15, 20, '现病史', '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (18, 15, 21, '诊疗经过', '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (26, 31, 32, '主诉', '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (27, 31, 33, '现病史', '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (28, 31, 34, '现病史', '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (29, 31, 35, '诊疗经过', '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (19, 22, 23, '主诉', '刘某某.pdf', '2024-12-03 22:26:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (20, 22, 24, '现病史', '刘某某.pdf', '2024-12-03 22:26:34');
INSERT INTO MEDICAL_RELATIONS (relation_id, from_entity_id, to_entity_id, relation_type, doc_reference, created_at) VALUES (21, 22, 25, '诊疗经过', '刘某某.pdf', '2024-12-03 22:26:34');

COMMIT;
