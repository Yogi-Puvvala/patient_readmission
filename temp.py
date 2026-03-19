import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import tensorflow
from tensorflow.keras.models import load_model

# Load original pkl model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Resave as h5
model.save("models/model.h5")
print("✅ model.h5 saved successfully!")
