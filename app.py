import streamlit as st
import pandas as pd
import pickle

# Load pre-trained model and encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('labelencoder_gender.pkl', 'rb') as file:
    labelencoder_gender = pickle.load(file)

with open('labelencoder_surgery_type.pkl', 'rb') as file:
    labelencoder_surgery_type = pickle.load(file)

with open('labelencoder_outcome.pkl', 'rb') as file:
    labelencoder_outcome = pickle.load(file)

# Function to make a prediction
def predict_outcome(age, gender, surgery_type, days_since_last_visit):
    gender_encoded = labelencoder_gender.transform([gender])[0]
    surgery_type_encoded = labelencoder_surgery_type.transform([surgery_type])[0]
    input_data = pd.DataFrame([[age, gender_encoded, surgery_type_encoded, days_since_last_visit]],
                              columns=['Age', 'Gender', 'Surgery Type', 'Days since last visit'])
    input_data = input_data[['Age', 'Gender', 'Surgery Type', 'Days since last visit']]  # Ensure column order matches
    outcome_encoded = model.predict(input_data)
    return labelencoder_outcome.inverse_transform(outcome_encoded)[0]

# Streamlit UI
st.title("Medical Outcome Prediction")

age = st.number_input("Enter Age:", min_value=0, max_value=120, value=30)
gender = st.selectbox("Enter Gender:", options=labelencoder_gender.classes_)
surgery_type = st.selectbox("Enter Surgery Type:", options=labelencoder_surgery_type.classes_)
days_since_last_visit = st.number_input("Enter Days since last visit:", min_value=0, value=30)

if st.button("Predict"):
    result = predict_outcome(age, gender, surgery_type, days_since_last_visit)
    st.success(f'The predicted outcome is: {result}')
