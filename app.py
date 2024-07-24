# Streamlit Dashboard
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Function to extract surgery type
def extract_surgery_type(note):
    note = note.lower()
    if 'penetrative keratoplasty' in note or 'penetrative keratoplastie' in note:
        return 'PK'
    elif 'endothelial keratoplasty' in note:
        return 'EK'
    elif 'deep anterior lamellar keratoplasty' in note:
        return 'DALK'
    elif 'thpk' in note:
        return 'THPK'
    return 'Unknown'

# Load the model and label encoders (make sure to have these saved from your training process)
# Example: You can use pickle to save and load your trained model and encoders

import pickle

# Load model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Streamlit app
st.title('Keratoplasty Surgery Outcome Predictor')

# Input fields
doctor_free_text = st.text_area('Doctor Notes', '')
age = st.number_input('Age', min_value=0)
gender = st.selectbox('Gender', ['Male', 'Female'])
surgery_date = st.date_input('Surgery Date')

# Preprocess inputs
gender_encoded = label_encoders['Gender'].transform([gender])[0]
extracted_surgery_type = extract_surgery_type(doctor_free_text)
surgery_type_encoded = label_encoders['Surgery Type'].transform([extracted_surgery_type])[0]
days_since_surgery = (pd.to_datetime('today') - pd.to_datetime(surgery_date)).days

# Create input data frame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_encoded],
    'Surgery Type': [surgery_type_encoded],
    'Days since last visit': [days_since_surgery]
})

# Make prediction
if st.button('Predict Outcome'):
    prediction = model.predict(input_data)[0]
    predicted_outcome = label_encoders['Outcome'].inverse_transform([prediction])[0]
    st.write(f'Predicted Outcome: {predicted_outcome}')
