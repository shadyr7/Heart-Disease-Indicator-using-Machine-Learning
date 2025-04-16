import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the Random Forest model
best_rf_model = joblib.load('rf_model_tuned.pkl')  # Replace with your actual model filename

# Create a title for the app
st.title("Heart Disease Indication Model")

# Create input fields for user inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)])  # Store values directly as integers
cp = st.selectbox("Chest Pain Type", options=[("Typical Angina", 0), ("Atypical Angina", 1), 
                                              ("Non-Anginal Pain", 2), ("Asymptomatic", 3)])  # Store values directly as integers
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)])
restecg = st.selectbox("Resting Electrocardiographic Results", options=[("Normal", 0), 
                                                                        ("Abnormal", 1), 
                                                                        ("Probable or definite left ventricular hypertrophy", 2)])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)])
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2), ("Unknown", 3)])

# Create a button to make the prediction
if st.button("Predict"):
    # Use the directly selected values
    sex = sex[1]
    cp = cp[1]
    fbs = fbs[1]
    restecg = restecg[1]
    exang = exang[1]
    slope = slope[1]
    thal = thal[1]

    # Preparing the input data as a numpy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    #Random Forest model
    prediction = best_rf_model.predict(input_data)

    #  output data for display
    result_df = pd.DataFrame({
        "Input Feature": [
            "Age", "Sex (0 = Female, 1 = Male)", "Chest Pain Type (0-3)", "Resting BP (mm Hg)", 
            "Cholesterol (mg/dl)", "Fasting Blood Sugar (0 = No, 1 = Yes)", 
            "Resting ECG Results (0-2)", "Max Heart Rate", "Exercise Induced Angina (0 = No, 1 = Yes)", 
            "Oldpeak", "Slope (0-2)", "No. of Major Vessels (0-3)", "Thalassemia (0-3)"
        ],
        "Your Input": [
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ],
        "Normal Range": [
            "18-120", "0-1", "0-3", "<120", "<200", "0 or 1", "0-2", ">=70", "0 or 1", "<5", "0-2", "0-3", "0-3"
        ]
    })

    #  prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")

    #  user inputs and normal values
    st.subheader("Your Inputs vs Normal Values")
    st.write(result_df)

    #  download results as CSV
    csv = result_df.to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "results.csv")
