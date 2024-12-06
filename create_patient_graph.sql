-- 创建患者表
CREATE TABLE patients (
    patient_id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
    name VARCHAR2(100),
    basic_info JSON,
    CONSTRAINT patient_pk PRIMARY KEY (patient_id)
);

-- 创建症状表
CREATE TABLE symptoms (
    symptom_id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
    name VARCHAR2(100),
    CONSTRAINT symptom_pk PRIMARY KEY (symptom_id)
);

-- 创建患者-症状关系表
CREATE TABLE patient_symptoms (
    relation_id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
    patient_id NUMBER,
    symptom_id NUMBER,
    record_date DATE,
    CONSTRAINT ps_pk PRIMARY KEY (relation_id),
    CONSTRAINT ps_patient_fk FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    CONSTRAINT ps_symptom_fk FOREIGN KEY (symptom_id) REFERENCES symptoms(symptom_id)
);

-- 创建Property Graph
CREATE PROPERTY GRAPH patient_graph
VERTEX TABLES (
    patients KEY (patient_id)
        LABEL patient
        PROPERTIES (patient_id, name, basic_info),
    symptoms KEY (symptom_id)
        LABEL symptom
        PROPERTIES (symptom_id, name)
)
EDGE TABLES (
    patient_symptoms AS has_symptom
        KEY (relation_id)
        SOURCE KEY (patient_id) REFERENCES patients(patient_id)
        DESTINATION KEY (symptom_id) REFERENCES symptoms(symptom_id)
        PROPERTIES (record_date)
); 