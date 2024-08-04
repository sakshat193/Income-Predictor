import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the trained model and scaler
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'
label_encoder_path = 'label_encoders.pkl'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(label_encoder_path)

# Education level mapping
education_map = {
    2: '1st-4th',
    3: '5th-6th',
    4: '7th-8th',
    5: '9th',
    6: '10th',
    7: '11th',
    8: '12th',
    9: 'HS-grad',
    10: 'Some-college',
    11: 'Vocational ',
    12: 'Assoc Academic',
    13: 'Bachelors',
    14: 'Masters',
    15: 'Prof-school',
    16: 'Doctorate',
    1: 'No Formal Education'
}

# Workclass mapping
workclass_map = {
    'Federal Government': 0,
    'Local Government': 1,
    'Private Sector': 2,
    'Self-Employed (Incorporated)': 3,
    'Self-Employed (Not Incorporated)': 4,
    'State Government': 5,
    'Without Pay': 6
}

# Occupation mapping
occupation_map = {
    'Administrative/Clerical': 0,
    'Armed Forces': 1,
    'Craft/Repair': 2,
    'Executive/Managerial': 3,
    'Farming/Fishing': 4,
    'Handlers/Cleaners': 5,
    'Machine Operators/Inspectors': 6,
    'Other Service': 7,
    'Private Household Service': 8,
    'Professional Specialty': 9,
    'Protective Services': 10,
    'Sales': 11,
    'Technical Support': 12,
    'Transportation/Moving': 13
}

# Create a Streamlit app
st.title("Income Predictor")
st.write("This app predicts whether an individual's income is greater than $50,000 based on their demographic characteristics.")

# Add a loading animation
with st.spinner("Loading..."):
    # Input fields
    st.subheader("Demographic Information")
    education_level = st.selectbox("Education Level", [""] + list(education_map.values()))
    education_level_num = None
    if education_level!= "":
        education_level_num = [k for k, v in education_map.items() if v == education_level][0]

    workclass = st.selectbox("Workclass", [""] + list(workclass_map.keys()))
    workclass_num = None
    if workclass!= "":
        workclass_num = label_encoders['workclass'].transform([workclass])[0]

    occupation = st.selectbox("Occupation", [""] + list(occupation_map.keys()))
    occupation_num = None
    if occupation!= "":
        occupation_num = label_encoders['occupation'].transform([occupation])[0]

    age = st.slider("Age", min_value=0, max_value=200, value=None)
    hours_per_week = st.slider("Hours per Week", min_value=0, max_value=168, value=None)

    sex = st.selectbox("Sex", ["", "Male", "Female"])
    sex_num = None
    if sex!= "":
        sex_num = label_encoders['sex'].transform([sex])[0]

    # Create a prediction function
    def make_prediction(education_level_num, workclass_num, occupation_num, age, hours_per_week, sex_num):
        input_data = pd.DataFrame({
            "education.num": [education_level_num],
            "age": [age],
            "hours.per.week": [hours_per_week],
            "sex": [sex_num],
            "workclass": [workclass_num],
            "occupation": [occupation_num]
        })
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        if prediction[0] == 0:
            return "Less than $50,000"
        else:
            return "Greater than $50,000"


    # Create a prediction button
    if st.button("Make Prediction"):
        prediction = make_prediction(education_level_num, workclass_num, occupation_num, age, hours_per_week, sex_num)
        st.write("Predicted Income:", prediction)
        if education_level == "" or workclass == "" or occupation == "" or age is None or hours_per_week is None or sex == "":
            st.error("Please fill in all fields.")   
        else:
            if prediction == "Greater than $50,000":
                st.balloons()
                st.success("Congratulations! Your predicted income is greater than $50,000.")
            else:
                st.warning("Your predicted income is less than $50,000.")

    st.subheader("👋 Hello, I am Akshat Singh!\nWelcome to the Income Predictor App 🚀\nPredict your income with  over 80% accuracy!!")

    # Add some additional information
    if st.button("About the Model"):
        st.text("This model uses a XGBoost classifier to predict income based on demographic characteristics. The model was trained on a dataset of individuals and their corresponding income levels.")