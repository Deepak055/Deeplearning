import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import streamlit as st

# Load model and preprocessors
model = tf.keras.models.load_model('heartmodel5.h5')

with open('label_encode_sex2.pkl', 'rb') as file:
    label_encoder_sex = pickle.load(file)

with open('scaler_heart_disease2.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('❤️ Heart Disease Risk Prediction')

# Input fields
age = st.slider('Age', 18, 92)
sex = st.selectbox('Sex', label_encoder_sex.classes_)
cp = st.slider('Chest Pain Type (cp)', 0, 3)
trestbps = st.number_input('Resting Blood Pressure (trestbps)', 90, 200)
chol = st.number_input('Serum Cholesterol (chol)', 100, 600)
fbs = st.slider('Fasting Blood Sugar > 120 mg/dl (fbs)', 0, 1)
restecg = st.slider('Resting ECG Results (restecg)', 0, 2)
thalach = st.number_input('Max Heart Rate Achieved (thalach)', 60, 250)
exang = st.slider('Exercise Induced Angina (exang)', 0, 1)
oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 6.0, step=0.1)
slope = st.slider('Slope of ST Segment (slope)', 0, 2)
ca = st.slider('Number of Major Vessels (ca)', 0, 4)
thal = st.slider('Thalassemia (thal)', 0, 3)

# Encode categorical input
sex_encoded = label_encoder_sex.transform([sex])[0]

if st.button('Predict'):
    # Prepare input DataFrame with correct feature names
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Ensure column order matches training
    input_data = input_data[scaler.feature_names_in_]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
 

    # Display
    st.subheader(f"Heart Disease Probability: {prediction_proba:.2%}")
    if prediction_proba > 0.5:
        st.error('⚠️ High risk of heart disease!')
    else:
        st.success('✅ Low risk of heart disease.')
