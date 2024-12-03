-- MEDICAL_ENTITIES 表导出脚本
-- 导出时间: 2024-12-03 22:57:02
-- 此脚本包含表结构、索引和数据


-- 创建实体表
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

-- 插入 MEDICAL_ENTITIES 表数据
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (8, '患者', '马某某', TO_CLOB('{"患者姓名": "马某某", "患者": {"基本信息": {"姓名": "马某某", "性别": "男", "年龄": "93", "入院日期": "2024-05-21", "出院日期": "2024-06-24"}}, "主诉与诊断": [{"类型": "主诉", "内容": "意识模糊"}, {"类型": "入院诊断", "内容": "多发性脑梗死"}, {"类型": "入院诊断", "内容": "呼吸衰竭"}, {"类型": "入院诊断", "内容": "意识障碍"}, {"类型": "入院诊断", "内容": "细菌性肺炎"}, {"类型": "入院诊断", "内容": "消化性溃疡伴出血"}, {"类型": "入院诊断", "内容": "冠状动脉粥样硬化性心脏病"}, {"类型": "入院诊断", "内容": "脑动脉硬化"}, {"类型": "入院诊断", "内容": "高血压病 3级（极高危险组）"}, {"类型": "入院诊断", "内容": "睡眠障碍"}, {"类型": "入院诊断", "内容": "前列腺增生"}, {"类型": "入院诊断", "内容": "胃食管反流"}, {"类型": "入院诊断", "内容": "认知障碍"}, {"类型": "入院诊断", "内容": "肝功能异常"}, {"类型": "入院诊断", "内容": "低蛋白血症"}, {"类型": "出院诊断", "内容": "多发性脑梗死"}, {"类型": "出院诊断", "内容": "呼吸衰竭"}, {"类型": "出院诊断", "内容": "意识障碍"}, {"类型": "出院诊断", "内容": "细菌性肺炎"}, {"类型": "出院诊断", "内容": "消化性溃疡伴出血"}, {"类型": "出院诊断", "内容": "冠状动脉粥样硬化性心脏病"}, {"类型": "出院诊断", "内容": "脑动脉硬化"}, {"类型": "出院诊断", "内容": "高血压病 3级（极高危险组）"}, {"类型": "出院诊断", "内容": "睡眠障碍"}, {"类型": "出院诊断", "内容": "前列腺增生"}, {"类型": "出院诊断", "内容": "胃食管反流"}, {"类型": "出院诊断", "内容": "认知障碍"}, {"类型": "出院诊断", "内容": "肝功能异常"}, {"类型": "出院诊断", "内容": "低蛋白血症"}, {"类型": "出院诊断", "内容": "泌尿系感染"}, {"类型": "出院诊断", "内容": "营养性贫血"}, {"类型": "出院诊断", "内容": "肾功能异常"}, {"类型": "出院诊断", "内容": "喉头水肿"}, {"类型": "出院诊断", "内容": "高尿酸血症"}], "现病史": [{"症状": "意识模糊"}, {"症状": "发热"}, {"症状": "乏力"}, {"症状": "饮食呛咳"}], "生命体征": [{"指标": "体温", "数值": "36.5", "单位": "℃"}, {"指标": "血压", "数值": "130/80", "单位": "mmHg"}], "生化指标": [{"项目": "C反应蛋白（快速）", "结果": "194.46", "单位": "mg/L", "参考范围": "异常"}, {"项目": "白细胞", "结果": "615.0", "单位": "/uL", "参考范围": "异常"}, {"项目": "淋巴细胞百分比", "结果": "7.1", "单位": "%", "参考范围": "异常"}, {"项目": "单核细胞百分比", "结果": "3.2", "单位": "%", "参考范围": "正常"}, {"项目": "中性粒细胞百分比", "结果": "89.5", "单位": "%", "参考范围": "异常"}, {"项目": "红细胞", "结果": "3.16", "单位": "*10^12/L", "参考范围": "异常"}, {"项目": "血红蛋白", "结果": "97", "单位": "g/L", "参考范围": "异常"}, {"项目": "血小板", "结果": "115", "单位": "*10^9/L", "参考范围": "异常"}, {"项目": "肌红蛋白", "结果": "301.3", "单位": "ng/mL", "参考范围": "异常"}, {"项目": "高敏肌钙蛋白 I", "结果": "74.5", "单位": "pg/mL", "参考范围": "异常"}, {"项目": "钙", "结果": "2.01", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "尿素", "结果": "38.1", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "尿酸", "结果": "543.2", "单位": "μmol/L", "参考范围": "异常"}, {"项目": "估算肾小球滤过率", "结果": "13.16", "单位": "mL/min/1.73m^2", "参考范围": "异常"}, {"项目": "葡萄糖", "结果": "6.25", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "胆碱酯酶", "结果": "3945.8", "单位": "U/L", "参考范围": "异常"}, {"项目": "乳酸", "结果": "2.56", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "天冬氨酸氨基转移酶", "结果": "80.6", "单位": "U/L", "参考范围": "异常"}, {"项目": "总蛋白", "结果": "55.1", "单位": "g/L", "参考范围": "异常"}, {"项目": "白蛋白", "结果": "30.1", "单位": "g/L", "参考范围": "异常"}, {"项目": "前白蛋白", "结果": "0.12", "单位": "g/L", "参考范围": "异常"}, {"项目": "D-二聚体", "结果": "950μ", "单位": "g/L", "参考范围": "异常"}, {"项目": "纤维蛋白原降解产物", "结果": "8.0", "单位": "mg/L", "参考范围": "异常"}, {"项目": "尿蛋白", "结果": "1+", "单位": "", "参考范围": "正常"}], "诊疗经过": [{"类型": "诊疗经过", "内容": "入院检验检查结合病史诊断为多发性脑梗死，意识障碍，呼吸衰竭等，血常规..."}, {"类型": "出院医嘱", "内容": "低盐低脂饮食，加强卧床护理"}, {"类型": "出院医嘱", "内容": "遵嘱服药，定期复查肺部 CT，呼吸内科随诊"}, {"类型": "出院医嘱", "内容": "不适神经内科随诊"}], "metadata": {"import_time": "2024-12-02T13:37:45.002014", "source_type": "text", "last_updated": "2024-12-02T13:37:45.002033"}}'), '马某某.pdf', '2024-12-03 21:03:26');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (9, '主诉', '主诉', TO_CLOB('未知'), '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (10, '现病史', '症状', TO_CLOB('{''症状'': ''意识模糊''}'), '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (11, '现病史', '症状', TO_CLOB('{''症状'': ''发热''}'), '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (12, '现病史', '症状', TO_CLOB('{''症状'': ''乏力''}'), '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (13, '现病史', '症状', TO_CLOB('{''症状'': ''饮食呛咳''}'), '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (14, '诊疗经过', '经过', TO_CLOB('[{''类型'': ''诊疗经过'', ''内容'': ''入院检验检查结合病史诊断为多发性脑梗死，意识障碍，呼吸衰竭等，血常规...''}, {''类型'': ''出院医嘱'', ''内容'': ''低盐低脂饮食，加强卧床护理''}, {''类型'': ''出院医嘱'', ''内容'': ''遵嘱服药，定期复查肺部 CT，呼吸内科随诊''}, {''类型'': ''出院医嘱'', ''内容'': ''不适神经内科随诊''}]'), '马某某.pdf', '2024-12-03 21:03:27');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (26, '患者', '蒲某某', TO_CLOB('{"患者姓名": "蒲某某", "患者": {"基本信息": {"姓名": "蒲某某", "性别": "女", "年龄": "86", "入院日期": "2024-06-16", "出院日期": "2024-06-22"}}, "主诉与诊断": [{"类型": "主诉", "内容": "间断意识丧失 1个月"}, {"类型": "入院诊断", "内容": "一过性意识丧失 待查"}, {"类型": "入院诊断", "内容": "慢性阻塞性肺疾病"}, {"类型": "入院诊断", "内容": "支气管哮喘"}, {"类型": "出院诊断", "内容": "脑血管供血不足"}, {"类型": "出院诊断", "内容": "体位性低血压"}, {"类型": "出院诊断", "内容": "慢性阻塞性肺疾病"}, {"类型": "出院诊断", "内容": "支气管哮喘"}], "现病史": [{"症状": "突发意识不清伴大汗、四肢无力"}, {"症状": "头晕、恶心、呕吐、四肢不自主抖动"}], "生命体征": [{"指标": "体温", "数值": "36.2", "单位": "℃"}, {"指标": "血压", "数值": "130/80", "单位": "mmHg"}], "生化指标": [{"项目": "尿白细胞", "结果": "1+", "单位": "", "参考范围": "正常"}, {"项目": "尿蛋白", "结果": "1+", "单位": "", "参考范围": "正常"}, {"项目": "尿胆红素", "结果": "1+", "单位": "", "参考范围": "正常"}, {"项目": "白细胞", "结果": "122.0", "单位": "/uL", "参考范围": "异常"}, {"项目": "鳞状上皮细胞", "结果": "107", "单位": "/uL", "参考范围": "异常"}, {"项目": "非鳞状上皮", "结果": "8.0", "单位": "/uL", "参考范围": "异常"}, {"项目": "粘液丝", "结果": "4671/LP", "单位": "", "参考范围": "异常"}, {"项目": "估算肾小球滤过率", "结果": "71.37", "单位": "mL/min/1.73m^2", "参考范围": "异常"}, {"项目": "乳酸", "结果": "3.78", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "丙酸氨基转移酶", "结果": "6.5", "单位": "U/L", "参考范围": "异常"}, {"项目": "总蛋白", "结果": "58.4", "单位": "g/L", "参考范围": "异常"}, {"项目": "白蛋白", "结果": "34.4", "单位": "g/L", "参考范围": "异常"}, {"项目": "载脂蛋白A1", "结果": "0.94", "单位": "g/L", "参考范围": "异常"}, {"项目": "肌酸激酶", "结果": "18.3", "单位": "U/L", "参考范围": "异常"}, {"项目": "三碘甲状腺原氨酸", "结果": "0.87nmol/L", "单位": "", "参考范围": "异常"}, {"项目": "单核细胞绝对值", "结果": "0.65×10^9/L", "单位": "", "参考范围": "异常"}, {"项目": "中性粒细胞绝对值", "结果": "6.67×10^9/L", "单位": "", "参考范围": "异常"}, {"项目": "白介素-6", "结果": "14.9", "单位": "pg/mL", "参考范围": "异常"}, {"项目": "N末端心房利钠肽", "结果": "207.1pg/ml", "单位": "", "参考范围": "异常"}], "诊疗经过": [{"类型": "诊疗经过", "内容": "入院后给与尿常规化学+有形成分分析，生化检查等，查出多发陈旧性脑梗死，进行进一步诊疗。"}, {"类型": "出院医嘱", "内容": "嘱其回家后注意休息"}, {"类型": "出院医嘱", "内容": "按时口服药物"}, {"类型": "出院医嘱", "内容": "有情况随诊"}], "metadata": {"import_time": "2024-12-02T15:39:23.056307", "source_type": "text", "last_updated": "2024-12-02T15:39:23.056361"}}'), '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (27, '主诉', '主诉', TO_CLOB('未知'), '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (28, '现病史', '症状', TO_CLOB('{''症状'': ''突发意识不清伴大汗、四肢无力''}'), '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (29, '现病史', '症状', TO_CLOB('{''症状'': ''头晕、恶心、呕吐、四肢不自主抖动''}'), '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (30, '诊疗经过', '经过', TO_CLOB('[{''类型'': ''诊疗经过'', ''内容'': ''入院后给与尿常规化学+有形成分分析，生化检查等，查出多发陈旧性脑梗死，进行进一步诊疗。''}, {''类型'': ''出院医嘱'', ''内容'': ''嘱其回家后注意休息''}, {''类型'': ''出院医嘱'', ''内容'': ''按时口服药物''}, {''类型'': ''出院医嘱'', ''内容'': ''有情况随诊''}]'), '蒲某某.pdf', '2024-12-03 22:26:37');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (15, '患者', '周某某', TO_CLOB('{"患者姓名": "周某某", "患者": {"基本信息": {"姓名": "周某某", "性别": "女", "年龄": "69", "入院日期": "2024-06-18", "出院日期": "2024-06-24"}}, "主诉与诊断": [{"类型": "主诉", "内容": "头晕"}, {"类型": "入院诊断", "内容": "脑血管供血不足"}, {"类型": "入院诊断", "内容": "多发腔隙性脑梗死"}, {"类型": "入院诊断", "内容": "高血压病"}, {"类型": "出院诊断", "内容": "脑血管供血不足"}, {"类型": "出院诊断", "内容": "多发腔隙性脑梗死"}, {"类型": "出院诊断", "内容": "脑动脉粥样硬化"}, {"类型": "出院诊断", "内容": "高血压病"}], "现病史": [{"症状": "头晕"}, {"症状": "天旋地转"}, {"症状": "恶心"}, {"症状": "胸闷心慌"}], "生命体征": [{"指标": "体温", "数值": "36.6", "单位": "℃"}, {"指标": "血压", "数值": "126/86", "单位": "mmHg"}], "生化指标": [{"项目": "白细胞", "结果": "7.27", "单位": "/HP", "参考范围": "异常"}, {"项目": "肌酐", "结果": "65.2", "单位": "μmol/L", "参考范围": "正常"}, {"项目": "血常规白细胞", "结果": "3.76", "单位": "*10^9/L", "参考范围": "正常"}, {"项目": "血常规血红蛋白", "结果": "132", "单位": "g/L", "参考范围": "正常"}, {"项目": "血常规血小板", "结果": "262", "单位": "*10^9/L", "参考范围": "正常"}], "诊疗经过": [{"类型": "诊疗经过", "内容": "给予对症止晕及改善循环药物治疗，查体各项指标正常，头颅MRI提示多发腔隙性脑梗塞等"}, {"类型": "出院医嘱", "内容": "低盐低脂饮食，测血压，注意休息"}, {"类型": "出院医嘱", "内容": "规律服药，定期复查"}], "metadata": {"import_time": "2024-12-02T13:50:59.020369", "source_type": "text", "last_updated": "2024-12-02T13:50:59.021829"}}'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (16, '主诉', '主诉', TO_CLOB('未知'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (17, '现病史', '症状', TO_CLOB('{''症状'': ''头晕''}'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (18, '现病史', '症状', TO_CLOB('{''症状'': ''天旋地转''}'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (19, '现病史', '症状', TO_CLOB('{''症状'': ''恶心''}'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (20, '现病史', '症状', TO_CLOB('{''症状'': ''胸闷心慌''}'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (21, '诊疗经过', '经过', TO_CLOB('[{''类型'': ''诊疗经过'', ''内容'': ''给予对症止晕及改善循环药物治疗，查体各项指标正常，头颅MRI提示多发腔隙性脑梗塞等''}, {''类型'': ''出院医嘱'', ''内容'': ''低盐低脂饮食，测血压，注意休息''}, {''类型'': ''出院医嘱'', ''内容'': ''规律服药，定期复查''}]'), '周某某.pdf', '2024-12-03 22:25:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (31, '患者', '杨某某', TO_CLOB('{"患者姓名": "杨某某", "患者": {"基本信息": {"姓名": "杨某某", "性别": "男", "年龄": "72", "入院日期": "2024-06-19", "出院日期": "2024-06-25"}}, "主诉与诊断": [{"类型": "主诉", "内容": "左侧肢体无力 5天，加重 2天"}, {"类型": "入院诊断", "内容": "脑梗死"}, {"类型": "入院诊断", "内容": "2型糖尿病"}, {"类型": "入院诊断", "内容": "高乳酸血症"}, {"类型": "出院诊断", "内容": "脑梗死"}, {"类型": "出院诊断", "内容": "颈动脉硬化"}, {"类型": "出院诊断", "内容": "锁骨下动脉粥样硬化"}, {"类型": "出院诊断", "内容": "维生素 B12缺乏"}, {"类型": "出院诊断", "内容": "颈椎退行性病变"}, {"类型": "出院诊断", "内容": "下肢动脉粥样硬化"}, {"类型": "出院诊断", "内容": "前列腺增生"}, {"类型": "出院诊断", "内容": "高同型半胱氨酸血症"}, {"类型": "出院诊断", "内容": "2型糖尿病"}, {"类型": "出院诊断", "内容": "高乳酸血症"}], "现病史": [{"症状": "左侧偏身力弱"}, {"症状": "2型糖尿病病史"}], "生命体征": [{"指标": "体温", "数值": "36.6", "单位": "℃"}, {"指标": "血压", "数值": "126/88", "单位": "mmHg"}], "生化指标": [{"项目": "同型半胱氨酸", "结果": "26.83", "单位": "μmol/L", "参考范围": "异常"}, {"项目": "血小板最大聚集率", "结果": "11.7", "单位": "%", "参考范围": "异常"}, {"项目": "维生素 B12", "结果": "130.00", "单位": "pg/mL", "参考范围": "异常"}, {"项目": "糖化血红蛋白", "结果": "6.60", "单位": "%", "参考范围": "异常"}], "诊疗经过": [{"类型": "诊疗经过", "内容": "完善相关实验室检验、完善相关检查、给予抗血小板、降脂稳定斑块、改善循环等治疗"}, {"类型": "出院医嘱", "内容": "平衡膳食，避免过多油脂摄入"}, {"类型": "出院医嘱", "内容": "重视生活方式调整，严格戒烟、戒酒"}, {"类型": "出院医嘱", "内容": "定时监测血糖，调整饮食习惯"}, {"类型": "出院医嘱", "内容": "定时监测血压，避免血压过低或过高"}, {"类型": "出院医嘱", "内容": "遵嘱规律服用抗血小板、降脂及补充维生素药物"}], "metadata": {"import_time": "2024-12-02T15:40:23.432119", "source_type": "text", "last_updated": "2024-12-02T15:40:23.432129"}}'), '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (32, '主诉', '主诉', TO_CLOB('未知'), '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (33, '现病史', '症状', TO_CLOB('{''症状'': ''左侧偏身力弱''}'), '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (34, '现病史', '症状', TO_CLOB('{''症状'': ''2型糖尿病病史''}'), '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (35, '诊疗经过', '经过', TO_CLOB('[{''类型'': ''诊疗经过'', ''内容'': ''完善相关实验室检验、完善相关检查、给予抗血小板、降脂稳定斑块、改善循环等治疗''}, {''类型'': ''出院医嘱'', ''内容'': ''平衡膳食，避免过多油脂摄入''}, {''类型'': ''出院医嘱'', ''内容'': ''重视生活方式调整，严格戒烟、戒酒''}, {''类型'': ''出院医嘱'', ''内容'': ''定时监测血糖，调整饮食习惯''}, {''类型'': ''出院医嘱'', ''内容'': ''定时监测血压，避免血压过低或过高''}, {''类型'': ''出院医嘱'', ''内容'': ''遵嘱规律服用抗血小板、降脂及补充维生素药物''}]'), '杨某某.pdf', '2024-12-03 22:26:40');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (22, '患者', '刘某某', TO_CLOB('{"患者姓名": "刘某某", "患者": {"基本信息": {"姓名": "刘某某", "性别": "男", "年龄": "93岁", "入院日期": "2024-03-28", "出院日期": "2024-06-25"}}, "主诉与诊断": [{"类型": "主诉", "内容": "突发左侧肢体无力"}, {"类型": "入院诊断", "内容": "急性脑梗死"}, {"类型": "入院诊断", "内容": "高血压 2级"}, {"类型": "入院诊断", "内容": "前列腺增生"}, {"类型": "入院诊断", "内容": "慢性胃炎"}, {"类型": "入院诊断", "内容": "慢性支气管炎"}, {"类型": "入院诊断", "内容": "支气管扩张"}, {"类型": "入院诊断", "内容": "高脂血症"}, {"类型": "出院诊断", "内容": "急性脑梗死"}, {"类型": "出院诊断", "内容": "高血压 2级"}, {"类型": "出院诊断", "内容": "前列腺增生"}, {"类型": "出院诊断", "内容": "慢性胃炎"}, {"类型": "出院诊断", "内容": "慢性支气管炎"}, {"类型": "出院诊断", "内容": "支气管扩张"}, {"类型": "出院诊断", "内容": "高脂血症"}, {"类型": "出院诊断", "内容": "心功能不全"}], "现病史": [{"症状": "左侧肢体无力"}], "生命体征": [{"指标": "体温", "数值": "36.4", "单位": "℃"}, {"指标": "血压", "数值": "126/70", "单位": "mmHg"}], "生化指标": [{"项目": "B-型钠尿肽", "结果": "181", "单位": "pg/mL", "参考范围": "异常"}, {"项目": "白细胞", "结果": "41.0", "单位": "/uL", "参考范围": "异常"}, {"项目": "嗜酸细胞百分比", "结果": "8.7", "单位": "%", "参考范围": "异常"}, {"项目": "红细胞", "结果": "3.17×10^12/L", "单位": "", "参考范围": "异常"}, {"项目": "血红蛋白", "结果": "95", "单位": "g/L", "参考范围": "异常"}, {"项目": "红细胞压积", "结果": "28.3", "单位": "%", "参考范围": "异常"}, {"项目": "国际标准化比值", "结果": "1.11", "单位": "", "参考范围": "异常"}, {"项目": "D-二聚体", "结果": "600μ", "单位": "g/L", "参考范围": "异常"}, {"项目": "阴离子间隙", "结果": "7.3", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "乳酸", "结果": "2.71", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "碱性磷酸酶", "结果": "41.4", "单位": "U/L", "参考范围": "异常"}, {"项目": "总胆固醇", "结果": "2.48", "单位": "mmol/L", "参考范围": "异常"}, {"项目": "低密度胆固醇", "结果": "1.19", "单位": "mmol/L", "参考范围": "正常"}], "诊疗经过": [{"类型": "诊疗经过", "内容": "根据病史、体征和影像学检查，明确诊断为急性脑梗死，予以抗血小板聚集、改善循环等药物治疗，积极肢体康复锻炼，观察血压变化。"}, {"类型": "出院医嘱", "内容": "注意休息，低盐低脂饮食"}, {"类型": "出院医嘱", "内容": "继续口服药物治疗"}, {"类型": "出院医嘱", "内容": "病情变化及时就诊"}], "metadata": {"import_time": "2024-12-02T15:21:19.581597", "source_type": "text", "last_updated": "2024-12-02T15:21:19.581616"}}'), '刘某某.pdf', '2024-12-03 22:26:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (23, '主诉', '主诉', TO_CLOB('未知'), '刘某某.pdf', '2024-12-03 22:26:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (24, '现病史', '症状', TO_CLOB('{''症状'': ''左侧肢体无力''}'), '刘某某.pdf', '2024-12-03 22:26:34');
INSERT INTO MEDICAL_ENTITIES (entity_id, entity_type, entity_name, entity_value, doc_ref, created_at) VALUES (25, '诊疗经过', '经过', TO_CLOB('[{''类型'': ''诊疗经过'', ''内容'': ''根据病史、体征和影像学检查，明确诊断为急性脑梗死，予以抗血小板聚集、改善循环等药物治疗，积极肢体康复锻炼，观察血压变化。''}, {''类型'': ''出院医嘱'', ''内容'': ''注意休息，低盐低脂饮食''}, {''类型'': ''出院医嘱'', ''内容'': ''继续口服药物治疗''}, {''类型'': ''出院医嘱'', ''内容'': ''病情变化及时就诊''}]'), '刘某某.pdf', '2024-12-03 22:26:34');

COMMIT;
