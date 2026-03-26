# Patient Readmission Prediction

A machine learning project that predicts whether a diabetic patient will be readmitted to hospital — and if so, whether it will happen within 30 days or after. Built end-to-end with a deep learning model, a REST API, and a Streamlit frontend, all containerized and deployed on Render.

**Live App:** https://patient-readmission-frontend.onrender.com

---

## What This Project Does

Hospital readmissions are a significant cost driver and a key indicator of care quality, particularly for diabetic patients. This project uses the Diabetes 130-US Hospitals dataset to build a classifier that predicts one of three outcomes: readmitted in less than 30 days, readmitted after 30 days, or not readmitted. The model is trained using TensorFlow/Keras and served through a FastAPI backend that the Streamlit frontend calls at inference time.

---

## Project Structure

```
patient_readmission/
├── api/                        # FastAPI application (app.py)
├── frontend/                   # Streamlit app (streamlit_app.py)
├── models/                     # Trained model artifact (model.h5)
├── src/                        # Preprocessing and feature engineering logic
├── Patient_Readmission.ipynb   # Exploratory analysis and model training notebook
├── temp.py                     # Utility script to convert model.pkl to model.h5
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yaml
├── requirements_api.txt
└── requirements_frontend.txt
```

---

## Model

The model is a neural network built with TensorFlow 2.19 and Keras 3.13, trained in a Jupyter notebook (`Patient_Readmission.ipynb`). The trained model is saved in HDF5 format (`models/model.h5`) and loaded by the API at startup.

The preprocessing logic lives in `src/` and is shared between the notebook and the API — the same transformations applied during training are applied at inference time.

One practical detail worth noting: TensorFlow takes a few seconds to fully load when the container starts. The Docker Compose health check accounts for this with a 60-second start period and a 20-second timeout per check, so the frontend only comes up after the API is genuinely ready.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | TensorFlow 2.19, Keras 3.13 |
| Preprocessing | Scikit-learn 1.8, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | Render |

The API and frontend have separate requirements files (`requirements_api.txt` and `requirements_frontend.txt`) and separate Dockerfiles, which keeps the images lean. TensorFlow is only installed in the API container.

---

## Running Locally

**Prerequisites:** Docker and Docker Compose installed.

```bash
git clone https://github.com/Yogi-Puvvala/patient_readmission.git
cd patient_readmission
docker-compose up --build
```

This starts two services:

- FastAPI backend at `http://localhost:8000`
- Streamlit frontend at `http://localhost:8501`

The frontend depends on the API being healthy before it starts. The first startup takes longer than usual because TensorFlow needs time to initialize and load the model.

---

## Training the Model

The model was trained in `Patient_Readmission.ipynb`. Open it in Jupyter or Google Colab to explore the data, run the preprocessing steps, train the model, and evaluate it. After training, the model is saved to `models/model.h5`.

## Deployment

The API and frontend are deployed as separate services on Render, each using its own Dockerfile. The frontend is pointed at the live API via the `API_URL` environment variable set at deploy time.

Because Render's free tier spins down idle services, the first request after inactivity may take a minute to respond while the container restarts and TensorFlow reloads the model.

---

## Dataset

This project uses the Diabetes 130-US Hospitals dataset, which covers over 100,000 hospital admissions for diabetic patients across 130 US hospitals between 1999 and 2008. The target variable has three classes: `<30` (readmitted within 30 days), `>30` (readmitted after 30 days), and `NO` (not readmitted).
