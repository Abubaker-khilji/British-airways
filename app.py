import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title('Customer Booking Prediction')
st.write("This app predicts whether a customer will complete a booking.")

# Create inputs for features
num_passengers = st.number_input('Number of Passengers', min_value=1, max_value=10, value=1)
purchase_lead = st.number_input('Purchase Lead (Days)', min_value=0, max_value=365, value=30)
length_of_stay = st.number_input('Length of Stay (Days)', min_value=1, max_value=30, value=5)
flight_hour = st.number_input('Flight Hour (24-hour)', min_value=0, max_value=23, value=12)
flight_day = st.selectbox('Flight Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Prepare input features for prediction
flight_day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
    'Saturday': 5, 'Sunday': 6
}

# Convert categorical flight_day into numeric
flight_day = flight_day_mapping[flight_day]

# Create a feature array
features = np.array([[num_passengers, purchase_lead, length_of_stay, flight_hour, flight_day]])

# Prediction button
if st.button('Predict Booking Completion'):
    prediction = model.predict(features)
    if prediction == 1:
        st.success("Booking Completed!")
    else:
        st.warning("Booking Not Completed.")

