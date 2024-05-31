import streamlit as st
import numpy as np
import joblib
import os

# File paths for the model and scalers
model_path = 'model.pkl'
scaler_path = 'standscaler.pkl'
minmax_scaler_path = 'minmaxscaler.pkl'

# Initialize variables
model, sc, ms = None, None, None

# Function to load the model and scalers
def load_model_and_scalers():
    global model, sc, ms
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("Model loaded successfully")
        else:
            st.error(f"Model file not found at {model_path}")

        if os.path.exists(scaler_path):
            sc = joblib.load(scaler_path)
            st.success("Standard Scaler loaded successfully")
        else:
            st.error(f"Standard Scaler file not found at {scaler_path}")

        if os.path.exists(minmax_scaler_path):
            ms = joblib.load(minmax_scaler_path)
            st.success("MinMax Scaler loaded successfully")
        else:
            st.error(f"MinMax Scaler file not found at {minmax_scaler_path}")

    except Exception as e:
        st.error(f"Error loading model or scalers: {e}")

# Load model and scalers at startup
load_model_and_scalers()

# Creating Streamlit app
st.title("Crop Predection System")

st.sidebar.header('Input Parameters')
N = st.sidebar.number_input('Nitrogen', min_value=0.0)
P = st.sidebar.number_input('Phosphorus', min_value=0.0)
K = st.sidebar.number_input('Potassium', min_value=0.0)
temp = st.sidebar.number_input('Temperature', min_value=0.0)
humidity = st.sidebar.number_input('Humidity', min_value=0.0)
ph = st.sidebar.number_input('Ph', min_value=0.0)
rainfall = st.sidebar.number_input('Rainfall', min_value=0.0)
soil_texture = st.sidebar.number_input('Soil Texture', min_value=0.0, max_value=100.0)
organic_matter = st.sidebar.number_input('Organic Matter', min_value=0.0, max_value=100.0)

if st.sidebar.button('Predict'):
    if model is None or sc is None or ms is None:
        st.error("Model or scalers are not loaded properly.")
    else:
        try:
            # Prepare feature list
            feature_list = [N, P, K, temp, humidity, ph, rainfall, soil_texture, organic_matter]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale features
            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)

            # Make prediction
            prediction = model.predict(final_features)

            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                         8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                         14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

            st.success(result)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
