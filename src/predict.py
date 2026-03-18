import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import numpy as np

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(data):
    numeric_data = np.array(data["numeric"]).reshape(1, -1)
    categorical_data = np.array(data["categorical"]).reshape(1, -1)

    scaled_numeric_data = scaler.transform(numeric_data)
    encoded_categorical_data = encoder.transform(categorical_data)

    med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}
    reshaped_sequential_data = np.array(
        [med_map[num] for num in data["sequential"]]
    ).reshape(1, -1, 1)

    tokenized_textual_data = tokenizer.texts_to_sequences(data["textual"])
    tokenized_textual_data = np.array(tokenized_textual_data)

    prediction = model.predict(
        {
            "numerical": scaled_numeric_data,
            "categorical": encoded_categorical_data,
            "text": tokenized_textual_data,
            "sequential": reshaped_sequential_data
        },
        verbose = 0
    )

    labels = ["<30", ">30", "NO"]
    predicted_index = np.argmax(prediction, axis = 1)
    predicted_label = labels[predicted_index[0]]
    confidence_scores = {"<30": round(prediction[0][0]*100, 2), 
                         ">30": round(prediction[0][1]*100, 2),
                         "NO": round(prediction[0][2]*100, 2)}

    return {
        "Prediction": predicted_label,
        "Confidence_Scores": confidence_scores
    }