import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

# Title
st.title("Bike Rentals Prediction App")

# User inputs
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)

weekday = st.selectbox("Weekday", [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])

weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"])

# Predict button
if st.button("Predict Rentals"):
    # Create dataframe for the input
    input_df = pd.DataFrame({
        "temperature": [temperature],
        "weekday": [weekday],
        "weather": [weather]
    })

    # One-hot encode to match training features
    input_encoded = pd.get_dummies(input_df)

    # Align with training columns (filling missing with 0)
    # NOTE: adjust to match the training columns from your model
    expected_cols = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)

    # Predict
    prediction = round(model.predict(input_encoded)[0])

    st.success(f"Predicted number of rentals: {prediction}")
