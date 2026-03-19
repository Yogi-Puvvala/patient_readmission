import streamlit as st
import requests
import os

st.title("Patient Readmission Prediction")

data = {
    "numeric"    : [],
    "categorical": [],
    "textual"    : [],
    "sequential" : []
}

with st.form(key="patient_data"):

    st.subheader("Hospital Stay")
    data["numeric"].append(st.number_input("Days in Hospital",        min_value=1,  max_value=14))
    data["numeric"].append(st.number_input("Number of Lab Procedures",min_value=1,  max_value=132))
    data["numeric"].append(st.number_input("Number of Medications",   min_value=1,  max_value=81))
    data["numeric"].append(st.number_input("Number of Diagnoses",     min_value=1,  max_value=16))

    st.subheader("Patient Info")
    data["categorical"].append(st.selectbox("Gender", ["Male", "Female"]))
    data["categorical"].append(st.selectbox("Race",   ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]))
    data["categorical"].append(st.selectbox("Admission Type",   ["Emergency", "Urgent", "Elective", "Not Available"]))
    data["categorical"].append(st.selectbox("Discharged To",    ["Home", "Transferred", "Nursing Facility", "Other"]))

    st.subheader("Diagnosis")
    diag_options = ['Circulatory','Diabetes','Respiratory','Digestive',
                    'Injury','Musculoskeletal','Neoplasms','Genitourinary','Other']
    data["textual"].append(st.selectbox("Primary Diagnosis",   diag_options))
    data["textual"].append(st.selectbox("Secondary Diagnosis", diag_options))
    data["textual"].append(st.selectbox("Third Diagnosis",     diag_options))

    st.subheader("Medications During Stay")
    med_options = ["No", "Steady", "Up", "Down"]
    data["sequential"].append(st.selectbox("Metformin",    med_options))
    data["sequential"].append(st.selectbox("Insulin",      med_options))
    data["sequential"].append(st.selectbox("Glipizide",    med_options))
    data["sequential"].append(st.selectbox("Glyburide",    med_options))
    data["sequential"].append(st.selectbox("Pioglitazone", med_options))

    submit = st.form_submit_button("Get Prediction")

if submit:
    # Combining textual into single string
    data["textual"] = [" ".join(data["textual"])]

    try:
        # API_URL  = os.getenv("API_URL", "http://127.0.0.1:8000")
        API_URL = os.getenv("API_URL", "http://localhost:8000")
        response = requests.post(f"{API_URL}/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result["Prediction"]

            # Displaying result 
            if prediction == "NO":
                st.success("Low Risk — Patient likely will NOT be readmitted")
            elif prediction == "<30":
                st.warning("High Risk — Patient may be readmitted within 30 days")
            else:
                st.info("Moderate Risk — Patient may be readmitted after 30 days")

            # Confidence scores 
            st.subheader("Confidence Scores")
            scores = result["Confidence_Scores"]
            for label, score in scores.items():
                st.progress(int(score), text=f"{label}: {score}%")

        else:
            st.error(f"API Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")