import streamlit as st
import requests
import os

st.title("Patient Readmission System")
data = {"numeric": [],
        "categorical": [],
        "textual": [],
        "sequential": []}

with st.form(key = "patient_data"):
    data["numeric"].append(st.number_input(label = "Days in hospital", min_value = 1, max_value = 14))
    data["numeric"].append(st.number_input(label = "Number of lab procedures"))
    data["numeric"].append(st.number_input(label = "Number of medications"))
    data["numeric"].append(st.number_input(label = "Number of diagnoses"))

    data["categorical"].append(st.selectbox(label = "Gender", options = ["Male", "Female"]))
    data["categorical"].append(st.selectbox(label = "Race", options = ["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian"]))
    data["categorical"].append(str(st.slider(label = "Admission type ID", min_value = 1, max_value = 8)))
    data["categorical"].append(str(st.slider(label = "Discharge disposition ID", min_value = 1, max_value = 28)))

    data["textual"].append(st.selectbox(label = "Diag 1", options = ['Circulatory', 'Diabetes', 'Respiratory', 'Digestive', 'Injury', 'Musculoskeletal', 'Neoplasms', 'Genitourinary', 'Other']))
    data["textual"].append(st.selectbox(label = "Diag 2", options = ['Circulatory', 'Diabetes', 'Respiratory', 'Digestive', 'Injury', 'Musculoskeletal', 'Neoplasms', 'Genitourinary', 'Other']))
    data["textual"].append(st.selectbox(label = "Diag 3", options = ['Circulatory', 'Diabetes', 'Respiratory', 'Digestive', 'Injury', 'Musculoskeletal', 'Neoplasms', 'Genitourinary', 'Other']))

    data["sequential"].append(st.selectbox(label = "metformin", options = ["No", "Steady", "Up", "Down"]))
    data["sequential"].append(st.selectbox(label = "insulin", options = ["No", "Steady", "Up", "Down"]))    
    data["sequential"].append(st.selectbox(label = "glipizide", options = ["No", "Steady", "Up", "Down"]))
    data["sequential"].append(st.selectbox(label = "glyburide", options = ["No", "Steady", "Up", "Down"]))
    data["sequential"].append(st.selectbox(label = "pioglitazone", options = ["No", "Steady", "Up", "Down"]))

    submit = st.form_submit_button(label = "Get Prediction")

if submit:
    if submit:
        text = data["textual"]
        data["textual"] = [" ".join(text)]

        try:
            url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
            response = requests.post(url, json=data)
            st.write(response.status_code)
            st.write(response.text)
            st.success(response.json())
        except Exception as e:
            st.error(f"API Error: {e}")