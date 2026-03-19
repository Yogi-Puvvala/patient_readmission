import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Absolute paths ──────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler    = pickle.load(open(os.path.join(BASE_DIR, "models/scaler.pkl"),    "rb"))
encoder   = pickle.load(open(os.path.join(BASE_DIR, "models/encoder.pkl"),   "rb"))
tokenizer = pickle.load(open(os.path.join(BASE_DIR, "models/tokenizer.pkl"), "rb"))
model = load_model(os.path.join(BASE_DIR, "models/model.h5"))
print("Model and artifacts loaded!")

MED_MAP = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}

def predict(data):

    # Numerical
    numeric_data        = np.array(data["numeric"]).reshape(1, -1)
    scaled_numeric_data = scaler.transform(numeric_data)
    # shape: (1, 4)

    # Categorical 
    categorical_data         = np.array(data["categorical"]).reshape(1, -1)
    encoded_categorical_data = encoder.transform(categorical_data)
    # shape: (1, 4)

    # Textual 
    tokenized = tokenizer.texts_to_sequences(data["textual"])
    tokenized_textual_data = pad_sequences(
        tokenized,
        maxlen=3,
        padding='post'
    )
    # shape: (1, 3)

    # Sequential 
    reshaped_sequential_data = np.array(
        [MED_MAP[val] for val in data["sequential"]]
    ).reshape(1, -1, 1)
    # shape: (1, 5, 1)

    # Predict 
    prediction = model.predict(
        {
            "numerical"  : scaled_numeric_data,
            "categorical": encoded_categorical_data,
            "text"       : tokenized_textual_data,
            "sequential" : reshaped_sequential_data
        },
        verbose=0
    )
    # shape: (1, 3)

    # Decode 
    labels          = ["<30", ">30", "NO"]
    predicted_index = np.argmax(prediction, axis=1)
    predicted_label = labels[predicted_index[0]]

    confidence_scores = {
        "<30": round(float(prediction[0][0]) * 100, 2),
        ">30": round(float(prediction[0][1]) * 100, 2),
        "NO" : round(float(prediction[0][2]) * 100, 2)
    }

    return {
        "Prediction"       : predicted_label,
        "Confidence_Scores": confidence_scores
    }