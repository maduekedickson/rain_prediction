import streamlit as st
import pickle
import numpy as np

# Load the model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Add a title and an image
st.title("🌦️ Rainfall Prediction App")
st.write("This app predicts whether it will rain based on weather conditions. Enter the required parameters below to get started.")

# Add an image
st.image("rain.jpeg", caption="Predict Rainfall with AI")

# Input fields for features
st.subheader("Input Weather Conditions")
pressure = st.number_input("Pressure (hPa)", value=1010.0)
maxtemp = st.number_input("Maximum Temperature (°C)", value=25.0)
mintemp = st.number_input("Minimum Temperature (°C)", value=15.0)
dewpoint = st.number_input("Dewpoint (°C)", value=10.0)
humidity = st.number_input("Humidity (%)", value=50)
cloud = st.number_input("Cloud Coverage (%)", value=40)
sunshine = st.number_input("Sunshine (hours)", value=5.0)
winddirection = st.number_input("Wind Direction (°)", value=90.0)
windspeed = st.number_input("Wind Speed (km/h)", value=10.0)

# Prediction button
if st.button("Predict"):
    # Prepare the input as a 2D array
    input_data = np.array([[pressure, maxtemp, mintemp, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]])
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success("🌧️ It will rain!")
    else:
        st.info("☀️ It will not rain.")
