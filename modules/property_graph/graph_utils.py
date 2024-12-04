"""
Property Graph Utilities

This module provides utility functions for working with Oracle Property Graph.
"""

import json
from typing import List, Dict, Any, Optional
from .graph_query import PropertyGraphQuery

class PropertyGraphUtils:
    def __init__(self):
        """Initialize the Property Graph Utils."""
        self.query = PropertyGraphQuery()

    def format_patient_info(self, patient_name: str) -> Dict[str, Any]:
        """
        Format all information about a patient into a structured dictionary.
        
        Args:
            patient_name (str): Name of the patient
            
        Returns:
            Dict[str, Any]: Structured patient information
        """
        # Get all relations
        relations = self.query.get_patient_relations(patient_name)
        
        # Get symptoms
        symptoms = self.query.get_patient_symptoms(patient_name)
        
        # Get treatment process
        treatment = self.query.get_treatment_process(patient_name)
        
        # Format the information
        info = {
            "patient_name": patient_name,
            "symptoms": [],
            "treatment_process": None,
            "other_info": []
        }
        
        # Process symptoms
        for symptom in symptoms:
            if symptom.get("symptoms"):
                try:
                    symptom_data = json.loads(symptom["symptoms"])
                    if isinstance(symptom_data, dict) and "症状" in symptom_data:
                        info["symptoms"].append(symptom_data["症状"])
                except json.JSONDecodeError:
                    info["symptoms"].append(symptom["symptoms"])
        
        # Process treatment
        if treatment and treatment[0].get("treatment_process"):
            try:
                treatment_data = json.loads(treatment[0]["treatment_process"])
                if isinstance(treatment_data, list) and treatment_data:
                    info["treatment_process"] = treatment_data[0].get("内容")
            except json.JSONDecodeError:
                info["treatment_process"] = treatment[0]["treatment_process"]
        
        # Process other relations
        for relation in relations:
            if relation["record_type"] not in ["现病史", "诊疗经过"]:
                info["other_info"].append({
                    "type": relation["record_type"],
                    "value": relation.get("detail", "未知")
                })
        
        return info

    def analyze_common_symptoms(self) -> List[Dict[str, Any]]:
        """
        Analyze common symptoms across all patients.
        
        Returns:
            List[Dict[str, Any]]: List of symptoms and their frequencies
        """
        all_relations = self.query.get_all_relations()
        symptom_count = {}
        
        for relation in all_relations:
            if relation["relation"] == "现病史":
                try:
                    symptom_data = json.loads(relation.get("target_name", "{}"))
                    if isinstance(symptom_data, dict) and "症状" in symptom_data:
                        symptom = symptom_data["症状"]
                        symptom_count[symptom] = symptom_count.get(symptom, 0) + 1
                except json.JSONDecodeError:
                    continue
        
        return [
            {"symptom": symptom, "count": count}
            for symptom, count in sorted(
                symptom_count.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

    def find_similar_cases(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Find patients with similar symptoms.
        
        Args:
            symptoms (List[str]): List of symptoms to match
            
        Returns:
            List[Dict[str, Any]]: List of similar cases
        """
        all_relations = self.query.get_all_relations()
        patient_symptoms = {}
        
        # Collect symptoms for each patient
        for relation in all_relations:
            if relation["relation"] == "现病史":
                patient_name = relation["source_name"]
                try:
                    symptom_data = json.loads(relation.get("target_name", "{}"))
                    if isinstance(symptom_data, dict) and "症状" in symptom_data:
                        if patient_name not in patient_symptoms:
                            patient_symptoms[patient_name] = set()
                        patient_symptoms[patient_name].add(symptom_data["症状"])
                except json.JSONDecodeError:
                    continue
        
        # Calculate similarity
        similar_cases = []
        symptoms_set = set(symptoms)
        for patient, patient_symptom_set in patient_symptoms.items():
            common_symptoms = symptoms_set.intersection(patient_symptom_set)
            if common_symptoms:
                similarity = len(common_symptoms) / len(symptoms_set)
                similar_cases.append({
                    "patient_name": patient,
                    "common_symptoms": list(common_symptoms),
                    "similarity": similarity
                })
        
        # Sort by similarity
        return sorted(similar_cases, key=lambda x: x["similarity"], reverse=True)
