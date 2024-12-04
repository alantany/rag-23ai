-- DOCUMENT_JSON 表导出脚本
-- 导出时间: 2024-12-03 22:57:02
-- 此脚本包含表结构、索引和数据


-- 创建JSON文档表
CREATE TABLE DOCUMENT_JSON (
    ID NUMBER NOT NULL PRIMARY KEY,
    DOC_INFO VARCHAR2(500),
    DOC_JSON JSON,
    CONTENT CLOB
);

-- 创建JSON索引
CREATE INDEX json_content_idx ON DOCUMENT_JSON(CONTENT)
INDEXTYPE IS CTXSYS.CONTEXT;

-- 插入 DOCUMENT_JSON 表数据
INSERT INTO DOCUMENT_JSON (id, doc_info, doc_json, content) VALUES (1, 'uploaded_documents/马某某.pdf', '{'生命体征': {'体温': '36.5℃', '血压': '130/80mmHg'}, '性别': '男', '入院诊断': ['多发性脑梗死', '呼吸衰竭', '意识障碍', '细菌性肺炎', '消化性溃疡伴出血', '冠状动脉粥样硬化性心脏病', '脑动脉硬化', '高血压病 3级（极高危险组）', '睡眠障碍', '前列腺增生', '胃食管反流', '认知障碍', '肝功能异常', '低蛋白血症'], '生化指标': {'白细胞': '615.0/uL↑', '白细胞(高倍视野)': '111.82/HP↑', '白蛋白': '30.1g/L↓', 'C反应蛋白（快速）': '194.46mg/L↑', '纤维蛋白原降解产物': '8.0mg/L↑', '淋巴细胞百分比': '7.1%↓', '高敏肌钙蛋白 I': '74.5pg/mL↑', '单核细胞百分比': '3.2%', '尿素': '38.1mmol/L↑', '葡萄糖': '6.25mmol/L↑', '红细胞': '3.16*10^12/L↓', '血小板': '115*10^9/L↓', '肌红蛋白': '301.3ng/mL↑', '胆碱酯酶': '3945.8U/L↓', '尿酸': '543.2μmol/L↑', '钙': '2.01mmol/L↓', 'D-二聚体': '950μg/L↑', '估算肾小球滤过率': '13.16mL/min/1.73m^2↓', '乳酸': '2.56mmol/L↑', '尿蛋白': '1+', '血红蛋白': '97g/L↓', '总蛋白': '55.1g/L↓', '肌酐(酶法)': '329.5μmol/L↑', '天冬氨酸氨基转移酶': '80.6U/L↑', '前白蛋白': '0.12g/L↓', '中性粒细胞百分比': '89.5%↑'}, '患者姓名': '马某某', '主诉': '意识模糊', 'metadata': {'import_time': '2024-12-02T13:37:45.002014', 'source_type': 'text', 'last_updated': '2024-12-02T13:37:45.002033'}, '出院日期': '2024-06-24', '入院日期': '2024-05-21', '出院诊断': ['多发性脑梗死', '呼吸衰竭', '意识障碍', '细菌性肺炎', '消化性溃疡伴出血', '冠状动脉粥样硬化性心脏病', '脑动脉硬化', '高血压病 3级（极高危险组）', '睡眠障碍', '前列腺增生', '胃食管反流', '认知障碍', '肝功能异常', '低蛋白血症', '泌尿系感染', '营养性贫血', '肾功能异常', '喉头水肿', '高尿酸血症'], '现病史': ['意识模糊', '发热', '乏力', '饮食呛咳'], '年龄': Decimal('93'), '诊疗经过': '入院检验检查结合病史诊断为多发性脑梗死，意识障碍，呼吸衰竭等，血常规...', '出院医嘱': ['低盐低脂饮食，加强卧床护理', '遵嘱服药，定期复查肺部 CT，呼吸内科随诊', '不适神经内科随诊']}', TO_CLOB('2024年6月24日16时05分 出 院 记 录
姓名 马某某 性别 男 年龄 93岁 民族 汉族 职业 离休 婚姻 已婚 单位 退
休
住院日期 :2024年5月 21日-2024年6月 24 日，共34天手术日期
主 诉 :意识模糊 3天
入院时情况 :患者于入院 3天前突发意识模糊，伴有发热、乏力、饮食呛咳，
Tmax37.8℃，不能独立行走，自测新型冠状病毒抗原阳性，服用 Paxlvoid 及退
热剂，患者症状未改善，1天前患者体温升高，伴有寒战，Tmax37.6℃，症状
较前加重。急来我科门诊，急诊以“意识模糊待查”收入我科。患者自发病来精
神差，饮食减少，便秘，小便可。高血压病史 4年，平素口服缬沙坦 40mg 间
断联用施慧达 1.25mg/日，目前已停用降压药物，血压监测正常。2017年就诊
于我院诊断为“多发性脑梗死、脑动脉硬化、呼吸衰竭、意识障碍、 细菌性肺
炎、消化性溃疡伴出血、冠状动脉粥样硬化性心脏病冠状动脉粥样硬化性心脏病
等”，长期口服波立维。睡眠障碍 3年，口服富马酸喹硫平 75mg/晚+酒石酸
唑吡坦10mg/晚。认知障碍 1年，盐酸多奈哌齐 5mgqn。前列腺增生、尿频 1
年，近期尿频加重，赛洛多辛胶囊 4mgbid+非那雄胺 5mgqd+酒石酸托特罗
定片4mgqn。体温36.5℃ 脉搏 70次/分 呼吸18次/分 血压130/80mmHg
双肺呼吸音粗,双肺闻干湿性啰音,心律齐,腹部平坦,双下肢水肿。意识模糊，气管
切开，呼吸机辅助呼吸，双侧瞳孔等大等圆，直径约 3mm，直接及间接对光反
射灵敏，双侧面部感觉不合作，双侧额纹对称，双侧鼻唇沟对称，示齿口角不偏，
伸舌不偏，四肢肌张力适中，四肢肌力不合作，未见肌肉萎缩。双侧指鼻试验、
跟膝胫试验稳准、双手轮替试验不合作，双侧痛觉触觉、音叉振动觉、双侧位置
觉、运动觉不合作，双侧腱反射对称存在，双侧巴氏征及查多克征阴性。颈无抵
抗，布氏征及克氏征阴性。
入院诊断 :
多发性脑梗死
呼吸衰竭
意识障碍
细菌性肺炎
消化性溃疡伴出血
冠状动脉粥样硬化性心脏病
脑动脉硬化
高血压病 3级（极高危险组）
睡眠障碍
前列腺增生
胃食管反流
认知障碍
肝功能异常
低蛋白血症
诊疗经过 :患者入院检验检查结合病史诊断为多发性脑梗死，意识障碍，呼吸衰
竭等，血常规：C反应蛋白（快速）:194.46mg/L↑;*白细胞:9.31*10^9/L;淋巴
细胞百分比:7.1%↓;单核细胞百分比:3.2%;中性粒细胞百分比:89.5%↑;*红细
胞:3.16*10^12/L↓;*血红蛋白:97g/L↓;*血小板:115*10^9/L↓;心肌酶：肌红蛋
白:301.3ng/mL↑;高敏肌钙蛋白 I:74.5pg/mL↑;肝肾功：*钙:2.01mmol/L↓;*尿
素:38.1mmol/L↑;肌酐(酶法):329.5μmol/L↑;*尿酸:543.2μmol/L↑;估算肾小球
滤过率:13.16mL/min/1.73m^2↓;*葡萄糖:6.92mmol/L↑;胆碱酯酶:3945.8U/L
↓;乳酸:2.56mmol/L↑;*天冬氨酸氨基转移酶:80.6U/L↑;*总蛋白:55.1g/L↓;*白蛋
白:30.1g/L↓;前白蛋白:0.12g/L↓;患者昨日夜间体温升高，呼吸机辅助下指氧稳
定，继续美罗培南抗感染治疗。患者血小板偏低，考虑与感染相关，警惕出血风
险，注意动态复查。患者双下肢水肿，血清白蛋白降低，考虑低蛋白血症，予以
输入白蛋白，改善水肿情况。患者呼吸衰竭，持续呼吸机辅助呼吸，持续无法脱
机，予以行气管切开术，手术过程顺利。患者应用美罗培南抗感染治疗，复查常
规化验：全血细胞计数+5分类：*白细胞:9.97×10^9/L↑;单核细胞百分比:10.3%
↑;中性粒细胞百分比:75.7%↑;单核细胞绝对值:1.03×10^9/L↑;中性粒细胞绝对
值:7.55×10^9/L↑;*红细胞:2.65×10^12/L↓;*血红蛋白:76g/L↓; 凝血纤溶六项：
D-二聚体:950μg/L↑;纤维蛋白原降解产物:8.0mg/L↑;生化 36项：*葡萄
糖:6.25mmol/L↑。尿常规化学+有形成分分析：尿蛋白:1+;白细胞:615.0/uL↑;
白细胞(高倍视野):111.82/HP↑。患者肾功能不全，进行性加重，出现少尿，经
与家属沟通后予以行肾替代疗法，予以患者透析治疗，后患者全身水肿较前好转。
患者家属要求出院，告知患者家属患者目前病情尚不稳定及出院后相关风险，患
者家属坚持出院，准予今日办理出院手续。
出院诊断 :
多发性脑梗死 呼吸衰竭 意识障碍 细菌性肺炎 消化性溃疡伴出血 冠状动脉粥样硬化性心
脏病 脑动脉硬化 高血压病 3级（极高危险组） 睡眠障碍 前列腺增生 胃食管反流 认知障
碍 肝功能异常 低蛋白血症 泌尿系感染 营养性贫血 肾功能异常 喉头水肿 高尿酸血症
出院时情况 : 好转
出院医嘱 :1.低盐低脂饮食，加强卧床护理；2.遵嘱服药，定期复查肺部 CT，呼
吸内科随诊；3.不适神经内科随诊。
出院随访 : 是
签名'));
INSERT INTO DOCUMENT_JSON (id, doc_info, doc_json, content) VALUES (4, 'uploaded_documents/蒲某某.pdf', '{'生命体征': {'体温': '36.2℃', '血压': '130/80mmHg'}, '性别': '女', '入院诊断': ['一过性意识丧失 待查', '慢性阻塞性肺疾病', '支气管哮喘'], '生化指标': {'尿常规化学+有形成分分析': {'尿白细胞': '1+', '尿蛋白': '1+', '尿胆红素': '1+', '白细胞': '122.0/uL↑', '白细胞(高倍视野)': '22.18/HP↑', '鳞状上皮细胞': '107/uL↑', '非鳞状上皮': '8.0/uL↑', '粘液丝': '4671/LP↑'}, '生化 39项': {'同型半胱氨酸': '估算肾小球滤过率:71.37mL/min/1.73m^2↓', '乳酸': '3.78mmol/L↑', '丙酸氨基转移酶': '6.5U/L↓', '总蛋白': '58.4g/L↓', '白蛋白(溴甲酚绿法)': '34.4g/L↓', '载脂蛋白A1': '0.94g/L↓', '肌酸激酶': '18.3U/L↓'}, '甲功七项': {'三碘甲状腺原氨酸': '0.87nmol/L↓'}, '全血细胞计数+5分类': {'单核细胞绝对值': '0.65×10^9/L↑', '中性粒细胞绝对值': '6.67×10^9/L↑'}, '炎症四项': {'白介素-6': '14.9pg/mL↑'}, '血栓五项': {'N末端心房利钠肽': '207.1pg/ml↑'}}, '患者姓名': '蒲某某', '主诉': '间断意识丧失 1个月', 'metadata': {'import_time': '2024-12-02T15:39:23.056307', 'source_type': 'text', 'last_updated': '2024-12-02T15:39:23.056361'}, '出院日期': '2024-06-22', '入院日期': '2024-06-16', '出院诊断': ['脑血管供血不足', '体位性低血压', '慢性阻塞性肺疾病', '支气管哮喘'], '现病史': ['突发意识不清伴大汗、四肢无力', '头晕、恶心、呕吐、四肢不自主抖动'], '年龄': Decimal('86'), '诊疗经过': '入院后给与尿常规化学+有形成分分析，生化检查等，查出多发陈旧性脑梗死，进行进一步诊疗。', '出院医嘱': ['嘱其回家后注意休息', '按时口服药物', '有情况随诊']}', TO_CLOB('2024年6月22日08时29分 出 院 记 录
姓名 蒲某某 性别 女 年龄 86岁 民族 汉族 职业 离休 婚姻 婚姻 单位 工
作单位
住院日期 :2024年6月 16日-2024年6月 22 日，共6天手术日期
主 诉 :间断意识丧失 1个月。
入院时情况 :患者于1个月前如厕时突发意识不清，伴有大汗、四肢无力，症状
持续2分钟后自行缓解，当时测血压 70/40mmHg，无肢体抽搐，就诊于我院
急诊，考虑体位性低血压，住院治疗 2周后出院，今晨患者再次出现上述症状，
症状持续时间 3分钟左右，伴有头晕、恶心、呕吐，测血压 100/60mmHg，伴
有四肢不自主抖动，为求治疗就诊于我院，查头颅 CT 示多发陈旧性脑梗死，为
进一步诊疗收入我科。既往：哮喘 30年余，长期应用布地格福控制，慢阻肺病
史4年余，感冒后易复发，肠梗阻病史 30余年。查体：体温 36.2℃，脉搏 70
次/分，呼吸 18次/分，血压 130/80mmHg，神清、言语流利，双侧瞳孔正大
等圆，光反射灵敏，双侧瞳孔等大等圆，直径约3mm，直接及间接对光反射对
称，无眼震，双侧鼻唇沟对称，四肢肌力 5级，肌张力适中，双侧腱反射对称存
在，双侧巴氏征及查多克征阴性，颈软无抵抗，布氏征及克氏征阴性。双肺呼吸
音清,双肺未闻及干湿性啰音,心律齐,腹部平坦,双下肢未见水肿。
入院诊断 :
一过性意识丧失 待查
慢性阻塞性肺疾病
支气管哮喘
诊疗经过 :入院后给与尿常规化学+有形成分分析：尿白细胞:1+;尿蛋白:1+;尿胆
红素:1+;白细胞:122.0/uL↑;白细胞(高倍视野):22.18/HP↑;鳞状上皮细胞:107/uL
↑;非鳞状上皮:8.0/uL↑;粘液丝:4671/LP↑;生化 39项^同型半胱氨酸 ：估算肾小
球滤过率:71.37mL/min/1.73m^2↓;乳酸:3.78mmol/L↑;*丙酸氨基转移
酶:6.5U/L↓;*总蛋白:58.4g/L↓;*白蛋白(溴甲酚绿法):34.4g/L↓;载脂蛋白
A1:0.94g/L↓;*肌酸激酶:18.3U/L↓;甲功七项：*三碘甲状腺原氨酸:0.87nmol/L↓;
全血细胞计数+5分类：单核细胞绝对值:0.65×10^9/L↑;中性粒细胞绝对值:6.67
×10^9/L↑;炎症四项：白介素-6:14.9pg/mL↑;血栓五项：N末端心房利钠
肽:207.1pg/ml↑;头颅核磁示：多发腔隙性脑梗死；脑白质脱髓鞘改变。老年性
脑改变。颈部血管超声示：左侧颈动脉未见明显异常，双侧椎动脉血流阻力指数
增高，双侧锁骨下动脉可视部分未见明显异常。心脏超声示：静息状态下，二尖
瓣少量返流，主动脉瓣少量返流 。全腹部CT 示：对比 2024-05-18前片，大
致同前：肝左叶囊肿，肝内钙化灶，胰腺萎缩并脂肪化。脾门区结节灶伴环形钙
化，副脾伴钙化？脾动脉瘤不除外，建议增强扫描或 CTA 检查，左肾高密度囊
肿可能、左肾囊肿可能大。对比 2024-05-18前片：肠系膜脂膜炎，较前新见，
部分肠管扩张并积气、积液，较前有所好转，余大致同前：腹主动脉及其分支硬
化改变，腰背部皮下水肿，膀胱充盈欠佳，膀胱壁不厚，未见异常密度。乙状结
肠冗长并多发憩室，右侧附件区囊性病变，请结合超声，大致同前。胸部 CT 示：
与2024-6-16前片对比，大致同前：双肺支气管炎改变，双肺下叶支扩伴粘液
栓，周围多发慢性炎症，双肺间质改变，双肺陈旧病变，心脏增大，主动脉钙化，
甲状腺病变，请结合超声，给与改善循环、营养神经治疗，病情平稳后出院。
出院诊断 :
脑血管供血不足
体位性低血压
慢性阻塞性肺疾病
支气管哮喘
出院时情况 : 好转
出院时病员状况 : 患者意识丧失明显好转，查体：未见异常。
出院医嘱 :嘱其回家后注意休息，按时口服药物，有情况随诊。
出院随访 : 是
签名'));
INSERT INTO DOCUMENT_JSON (id, doc_info, doc_json, content) VALUES (2, 'uploaded_documents/周某某.pdf', '{'生命体征': {'体温': '36.6℃', '血压': '126/86mmHg'}, '性别': '女', '入院诊断': ['脑血管供血不足', '多发腔隙性脑梗死', '高血压病'], '生化指标': {'白细胞': '7.27/HP↑', '肌酐': '65.2μmol/L', '血常规': '白细胞: 3.76*10^9/L; 血红蛋白: 132g/L; 血小板: 262*10^9/L'}, '患者姓名': '周某某', '主诉': '头晕', 'metadata': {'import_time': '2024-12-02T13:50:59.020369', 'source_type': 'text', 'last_updated': '2024-12-02T13:50:59.021829'}, '出院日期': '2024-06-24', '入院日期': '2024-06-18', '出院诊断': ['脑血管供血不足', '多发腔隙性脑梗死', '脑动脉粥样硬化', '高血压病'], '现病史': ['头晕', '天旋地转', '恶心', '胸闷心慌'], '年龄': Decimal('69'), '诊疗经过': '给予对症止晕及改善循环药物治疗，查体各项指标正常，头颅MRI提示多发腔隙性脑梗塞等', '出院医嘱': ['低盐低脂饮食，测血压，注意休息', '规律服药，定期复查']}', TO_CLOB('2024年6月24日08时31分 出 院 记 录
姓名 周某某 性别 女 年龄 69岁 民族 汉族 职业 无 婚姻 已婚 单位 工作
单位
住院日期 :2024年6月 18日-2024年6月 24 日，共6天手术日期
主 诉 :头晕 4小时
入院时情况 :患者入院前 4小时午睡醒后无明显诱因出现头晕，自觉天旋地转，
伴恶心，周身湿冷，胸闷心慌等不适，无耳鸣及听力减退，无视物成双，无言语
不利，无口角歪斜，无肢体活动障碍，症状持续存在。既往有高血压病史。查体：
体温36.6℃，脉搏 76次/分，呼吸 20次/分，血压 126/86mmHg，双肺呼吸
音清,双肺未闻及干湿性啰音,心律齐,腹部平坦,双下肢未见水肿。意识清楚，言语
流利，双侧瞳孔等大等圆，直径约 3mm，直接及间接对光反射灵敏，双侧面部
感觉正常对称，双侧额纹对称，闭目有力，双侧鼻唇沟对称，示齿口角不偏，伸
舌不偏，四肢肌张力适中，四肢肌力 5级，未见肌肉萎缩。双侧指鼻试验、跟膝
胫试验稳准、双手轮替试验灵活，双侧痛觉触觉对称，音叉振动觉、双侧位置觉、
运动觉对称正常，双侧腱反射对称存在，双侧巴氏征及查多克征阴性。颈无抵抗，
布氏征及克氏征阴性。
入院诊断 :
脑血管供血不足
多发腔隙性脑梗死
高血压病
诊疗经过 :入院后根据患者发病特点考虑脑血管供血不足，给予对症止晕及改善
循环药物治疗。查尿常规化学+有形成分分析：白细胞(高倍视野):7.27/HP↑;粪
便常规+隐血试验：潜血试验(免疫法):阴性;凝血：凝血酶原活动度:120%;D-二
聚体:218μg/L;肌酐:65.2μmol/L;血常规：白细胞:3.76*10^9/L;*血红蛋
白:132g/L;*血小板:262*10^9/L;查眼震电图提示眼动系统（+），动态位置试
验（+），TCD 探及血管未见异常，脑电图无明显异常，查心脏超声提示静息状
态下提示三尖瓣少量返流，左室舒张功能减低。查双侧颈动脉中内膜局限性增厚
伴多发斑块形成，双侧椎动脉血流未见明显异常，右侧锁骨下动脉起始部斑块形
成，左侧锁骨下动脉可视部分未见明显异常。焦虑量表 9分，抑郁8分，MoCA22
分，MMSE24分，外周动脉阻塞检查提示外周动脉检测未见异常。查头颅 MRI
提示多发腔隙性脑梗塞，脑白质脱髓鞘改变。查头颅 MRA 提示双侧颈内动脉走
行迂曲，管腔粗细不均；椎动脉颅内段迂曲，未见明显狭窄或扩张；双侧大脑中
动脉及其分枝走行迂曲，管腔粗细欠均匀，M2段显影淡，分支未见明显减少；
双侧大脑前动脉及其分枝走行迂曲，A1-2段显影淡，分支未见明显减少；基底
动脉局部轻度狭窄；双侧大脑后动脉及其分枝走行迂曲，分支显示欠佳，右侧胚
胎型大脑后动脉。查动态心电图:1.主导节律为窦性:平均心室率 62次/min。最
快心室率86次/min;最慢心室率 48次/min。2.房性早搏总数 105次;3.完全性
右束支阻滞；4.T 波改变；查动态血压监测提示平均动脉压 116/83mmHg。经
治疗后患者头晕缓解，给予办理出院。
出院诊断 :
脑血管供血不足
多发腔隙性脑梗死
脑动脉粥样硬化
高血压病
出院时情况 : 好转
出院时病员状况 : 患者头晕缓解，无其他不适自诉。
出院医嘱 :1、低盐低脂饮食，测血压，注意休息，避免劳累受凉感冒；2、规律
服药，定期血常规、肝肾功能指标，复查头颅影像检查；3、神经科门诊随诊，
不适情况及时来院。
出院随访 : 是
签名'));
INSERT INTO DOCUMENT_JSON (id, doc_info, doc_json, content) VALUES (3, 'uploaded_documents/刘某某.pdf', '{'生命体征': {'体温': '36.4℃', '血压': '126/70mmHg'}, '性别': '男', '入院诊断': ['急性脑梗死', '高血压 2级', '前列腺增生', '慢性胃炎', '慢性支气管炎', '支气管扩张', '高脂血症'], '生化指标': {'白细胞': '41.0/uL↑', '白细胞(高倍视野)': '7.45/HP↑', '碱性磷酸酶': '41.4U/L↓', 'B-型钠尿肽': '181pg/mL↑', '红细胞': '3.17×10^12/L↓', '总胆固醇': '2.48mmol/L↓', '嗜酸细胞百分比': '8.7%↑', 'D-二聚体': '600μg/L↑', '乳酸': '2.71mmol/L↑', '国际标准化比值': '1.11↑', '红细胞压积': '28.3%↓', '血红蛋白': '95g/L↓', '阴离子间隙': '7.3mmol/L↓', '低密度胆固醇': '1.19mmol/L'}, '患者姓名': '刘某某', '主诉': '突发左侧肢体无力', 'metadata': {'import_time': '2024-12-02T15:21:19.581597', 'source_type': 'text', 'last_updated': '2024-12-02T15:21:19.581616'}, '出院日期': '2024-06-25', '入院日期': '2024-03-28', '出院诊断': ['急性脑梗死', '高血压 2级', '前列腺增生', '慢性胃炎', '慢性支气管炎', '支气管扩张', '高脂血症', '心功能不全'], '现病史': ['左侧肢体无力'], '年龄': '93岁', '诊疗经过': '根据病史、体征和影像学检查，明确诊断为急性脑梗死，予以抗血小板聚集、改善循环等药物治疗，积极肢体康复锻炼，观察血压变化。', '出院医嘱': ['注意休息，低盐低脂饮食', '继续口服药物治疗', '病情变化及时就诊']}', TO_CLOB('2024年6月25日08时47分 出 院 记 录
姓名 刘某某 性别 男 年龄 93岁 民族 汉族 职业 离休 婚姻 已婚 单位 航
天中心医院
住院日期 :2024年3月 28日-2024年6月 25 日，共89天手术日期
主 诉 :突发左侧肢体无力 4月余。
入院时情况 :患者于入院前 4月余无明显诱因出现左侧肢体无力，左上肢为主，
不能抬离床面，左手不能握拳，左下肢轻度力弱，抬腿略发沉，休息后未见好转。
后家属将其送至我院急诊，急查头颅 CT 示多发腔隙性脑梗死，脑白质变性，老
年性脑改变。查头颅 MRI示右侧中央前、后回、半卵圆中心、顶、枕叶新发急
性脑梗死，多发腔隙性脑梗死，脑白质脱髓鞘改变，脑萎缩。予以抗血小板聚集、
改善循环及脑保护等药物治疗后好转。1小时前自觉左上肢无力较前加重。既往
发现高血压 20余年，血压最高 170mmHg。发现前列腺增生 20余年。发现慢
性胃炎10余年。发现慢性支气管炎，支气管扩张 10余年。发现高脂血症 1月。
40年前因直肠癌在我院行手术治疗。对海鲜过敏。查体：体温 36.3℃，脉搏70
次/分，呼吸 16次/分，血压 130/75mmHg，双肺呼吸音粗，未闻及干湿性罗
音，心音有力，律齐，四肢无水肿。意识清楚，言语流利，双侧瞳孔等大同圆，
直径约3.0mm，直接、间接对光反射灵敏，双侧额纹对称，闭目有力，双侧鼻
唇沟对称，示齿口角不偏，伸舌居中，咽反射正常，双侧面部感觉对称，左上肢
肌力4-级，左下肢肌力 5-级，右侧肢体肌力 5级，肌张力适中，双侧肢体腱反
射对称存在，双侧肢体痛温觉对称正常，左侧指鼻试验、轮替动作不合作，双侧
跟膝胫试验稳准，左侧巴氏征弱阳性，颈无抵抗。
入院诊断 :
急性脑梗死
高血压 2级
前列腺增生
慢性胃炎
慢性支气管炎
支气管扩张
高脂血症
诊疗经过 :入院后根据病史、体征和影像学检查，经上级医师查房，明确诊断为
急性脑梗死，予以抗血小板聚集、改善循环等药物治疗，积极肢体康复锻炼，观
察血压变化。后患者咳嗽，咳痰较前加重，查 B-型钠尿肽:181pg/mL↑;全血细
胞计数+5分类^C-反应蛋白 ：*白细胞:5.70×10^9/L;嗜酸细胞百分比:8.7%↑;*
红细胞:3.17×10^12/L↓;*血红蛋白:95g/L↓;*红细胞压积:28.3%↓;凝血纤溶六项：
*国际标准化比值:1.11↑;D-二聚体:600μg/L↑;血生化：阴离子间隙:7.3mmol/L↓;
乳酸:2.71mmol/L↑;*碱性磷酸酶:41.4U/L↓;*总胆固醇:2.48mmol/L↓;*低密度胆
固醇:1.19mmol/L;尿常规化学+有形成分分析：白细胞:41.0/uL↑;白细胞(高倍视
野):7.45/HP↑;患者间断头晕不适，查头颅 MRI示多发腔隙性脑梗死，脑白质脱
髓鞘改变，双额颞部硬膜下积液，脑萎缩。腹部彩超示轻度脂肪肝，肝多发囊肿，
胆囊息肉样病变，双肾多发囊肿及肾盂旁囊肿（不除外同时伴有局部肾盏积水），
胰腺、脾脏未见明显异常。腹腔彩超示腹腔内未见腹水征象。经上述治疗后患者
患者病情较前好转。
出院诊断 :
急性脑梗死
高血压 2级
前列腺增生
慢性胃炎
慢性支气管炎
支气管扩张
高脂血症
心功能不全
出院时情况 : 好转
出院时病员状况 : 患者精神状态可，咳嗽，咳痰较前好转，左侧肢体力弱较前
好转，左上肢可抬离床面，左下肢轻度力弱。查体：体温 36.4℃，脉搏 72次/
分，血压 126/70mmHg，双肺呼吸音清，未闻及干湿性罗音，心音有力，心律
齐，四肢无水肿。意识清楚，言语流利，双侧瞳孔等大同圆，直径约 3.0mm，
直接、间接对光反射灵敏，双侧鼻唇沟对称，示齿口角不偏，伸舌居中，双侧面
部感觉对称，左上肢肌力 4-级，左下肢肌力 5-级，右侧肢体肌力 5级，肌张力
适中，双侧肢体腱反射对称存在，左侧指鼻试验、轮替动作欠稳准，双侧跟膝胫
试验稳准，左侧巴氏征阳性，颈无抵抗。
出院医嘱 :
1.注意休息，低盐低脂饮食；
2.继续口服药物治疗；
3.病情变化及时就诊。
出院随访 : 是
签名'));
INSERT INTO DOCUMENT_JSON (id, doc_info, doc_json, content) VALUES (5, 'uploaded_documents/杨某某.pdf', '{'生命体征': {'体温': '36.6℃', '血压': '126/88mmHg'}, '性别': '男', '入院诊断': ['脑梗死', '2型糖尿病', '高乳酸血症'], '生化指标': {'同型半胱氨酸': '26.83μmol/L↑', '血小板最大聚集率': '11.7%↓', '维生素 B12': '130.00pg/mL↓', '糖化血红蛋白': '6.60%↑'}, '患者姓名': '杨某某', '主诉': '左侧肢体无力 5天，加重 2天', 'metadata': {'import_time': '2024-12-02T15:40:23.432119', 'source_type': 'text', 'last_updated': '2024-12-02T15:40:23.432129'}, '出院日期': '2024-06-25', '入院日期': '2024-06-19', '出院诊断': ['脑梗死', '颈动脉硬化', '锁骨下动脉粥样硬化', '维生素 B12缺乏', '颈椎退行性病变', '下肢动脉粥样硬化', '前列腺增生', '高同型半胱氨酸血症', '2型糖尿病', '高乳酸血症'], '现病史': ['左侧偏身力弱', '2型糖尿病病史'], '年龄': Decimal('72'), '诊疗经过': '完善相关实验室检验、完善相关检查、给予抗血小板、降脂稳定斑块、改善循环等治疗', '出院医嘱': ['平衡膳食，避免过多油脂摄入', '重视生活方式调整，严格戒烟、戒酒', '定时监测血糖，调整饮食习惯', '定时监测血压，避免血压过低或过高', '遵嘱规律服用抗血小板、降脂及补充维生素药物']}', TO_CLOB('2024年6月25日10时50分 出 院 记 录
姓名 杨某某 性别 男 年龄 72岁 民族 汉族 职业 退休 婚姻 已婚 单位 工
作单位
住院日期 :2024年6月 19日-2024年6月 25 日，共6天手术日期
主 诉 :左侧肢体无力 5天，加重 2天
入院时情况 :患者于入院前 5天无明显诱因出现左侧偏身力弱，具体表现为左手
精细动作差，左上肢尚能持物及抬举，左下肢沉重发僵，尚能迈步及行走，无其
他伴随症状，上述症状持续不缓解，入院前 2天自觉上述症状有所加重，完善头
颅CT 未见高密度影。2型糖尿病病史。查体：体温 36.6℃，脉搏 67次/分，呼
吸18次/分，血压 126/88mmHg，双肺呼吸音清,双肺未闻及干湿性啰音,心律
齐,腹部平坦,双下肢未见水肿。意识清楚，言语流利，双侧瞳孔等大等圆，直径
约3mm，直接及间接对光反射灵敏，双侧面部感觉正常对称，双侧额纹对称，
闭目有力，双侧鼻唇沟对称，示齿口角不偏，伸舌不偏，四肢肌张力适中，左侧
轻瘫试验阳性，余肌力 5级，未见肌肉萎缩。双侧指鼻试验、跟膝胫试验稳准、
双手轮替试验灵活，双侧浅感觉、音叉振动觉、双侧位置觉、运动觉对称正常，
双侧腱反射对称存在，双侧病理征阴性。颈无抵抗，布氏征及克氏征阴性。
入院诊断 :
脑梗死 、2型糖尿病 、高乳酸血症
诊疗经过 :患者入院后完善相关实验室检验：：同型半胱氨酸 ^血脂 4项：血同
型半胱氨酸:26.83μmol/L↑;多参数血小板功能分析(肾上腺素)：血小板最大聚集
率:11.7%↓;血小板凝集实验:8.8%↓;血小板粘附率:4.6%↓;血小板有效抑制
率:88.3%↑;叶酸^维生素 B12：维生素 B12:130.00pg/mL↓;糖化血红蛋白 ：*
糖化血红蛋白:6.60%↑。完善相关检查：颈部血管超声：双侧颈动脉中内膜局限
性增厚伴多发斑块形成、双侧椎动脉血流阻力指数增高。锁骨下动脉超声：右侧
锁骨下动脉起始段斑块形成、左侧锁骨下动脉可视段未见明显异常。泌尿系超声：
前列腺增生、双肾、膀胱未见明显异常。
下肢动脉超声：双侧下肢动脉粥样硬化伴多发小斑块形成、双侧下肢静脉血流通
畅。心脏超声：三尖瓣少量返流、主动脉瓣少量返流 、左室舒张功能减低。腹
部超声：肝脏、胆囊、胰腺、脾脏未见明显异常。头颈部 CTA：颈部多发动脉
粥样硬化改变、右侧胚胎型大脑后动脉、左侧大脑后动脉 P1、2段粥样硬化改
变、CTP扫描未见明显异常。头部核磁：脑白质脱髓鞘改变、SWI 未见明确异
常出血信号及静脉血管畸形改变。颈椎核磁：颈椎退行性改变。C3-4、4-5、5-6、
6-7椎间盘病变并黄韧带及后纵韧带增厚，相应椎管及椎间孔变窄。颈 5-6水平
髓内变性改变不除外。无创心输出量测定：心室壁顺应性减低，舒张功能不良。
肢体节段性压力测试：外周动脉检查未见明显异常。眼震电图：眼动系统（+）
、动态位置试验（-）。脑电图：边缘状态脑电图。结合患者症状、体征及相关
检查结果，考虑脑梗死。给予抗血小板、降脂稳定斑块、改善循环等治疗后患者
症状较前明显改善，请示上级医师后准予带药出院。门诊随诊。
出院诊断 :
脑梗死 、颈动脉硬化 、锁骨下动脉粥样硬化 、维生素 B12缺乏 、颈椎退行性病变 、下肢动
脉粥样硬化 、前列腺增生 、高同型半胱氨酸血症 、2型糖尿病 、高乳酸血症
出院时情况 : 好转
出院时病员状况 : 患者左侧肢体无力症状改善，未诉其他特殊不适。查体：生
命体征平稳，内科查体未见明显异常。意识清楚，言语流利，双侧瞳孔等大等圆，
直径约3mm，直接及间接对光反射灵敏，双侧面部感觉正常对称，双侧额纹对
称，闭目有力，双侧鼻唇沟对称，示齿口角不偏，伸舌不偏，四肢肌张力适中，
左侧轻瘫试验阳性，余肌力 5级，未见肌肉萎缩。双侧指鼻试验、跟膝胫试验稳
准、双手轮替试验灵活，双侧浅感觉、音叉振动觉、双侧位置觉、运动觉对称正
常，双侧腱反射对称存在，双侧病理征阴性。颈无抵抗，布氏征及克氏征阴性。
出院医嘱 :
1.平衡膳食，避免过多油脂摄入，注意补充新鲜水果蔬菜及优质蛋白；
2.重视生活方式调整，严格戒烟、戒酒，避免二手烟、熬夜，注意休息，适当锻
炼，避免运动过度，避免大汗、腹泻、劳累等诱发因素；
3.定时监测血糖，调整饮食习惯，内分泌科就诊酌情药物使用；
4.定时监测血压，避免血压过低或过高，心内科门诊随诊调整降压药物使用；
5.遵嘱规律服用抗血小板、降脂及补充维生素药物，服用期间注意有无皮肤、粘
膜出血/黄染或血尿、黑便、肌肉酸痛等情况，定期复查肝肾功、凝血、血常规、
肌酶等实验室化验酌情调整药物使用；
阿司匹林肠溶片[0.1g*30片/盒]，1盒,0.1g，1次/日,口服
阿托伐他汀钙片[20mg*14片/盒[S-1]]，1盒,20mg，1次/晚,口服
脑脉利颗粒[10g*9袋/盒]，4盒,10g，3次/日,口服
叶酸片[5mg*100 片/瓶.]，1瓶,5mg，1次/日,口服
维生素B6片[10mg*100 片/瓶[S-3]]，1瓶,10mg，2次/日,口服
复合维生素 B片[复方*100片/瓶]，2瓶,2片，2次/日,口服
维生素B12片[25μg*100 片/瓶]，1瓶,50μg，1次/日,口服
苯磺酸氨氯地平片[5mg*7 片[未中标]]，4盒,5mg，1次/日,口服
胰岛素注射液[10ml:400iu*2 支/盒]，1支,4iu，立即,皮下注射
精蛋白锌重组赖脯胰岛素混合注射液（50R）[3ml:300iu/支[S-6]]，2支,14iu，
立即,皮下注射
6.定期神经内科门诊复诊，如遇严重头晕、行走不稳、肢体运动及感觉障碍、明
显呛咳、意识水平降低等特殊情况及时急诊就诊。
出院随访 : 是
签名'));

COMMIT;