import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from datetime import datetime

# Load the models, scaler, and encoders
best_xgb_model = load('xgb_model.joblib')
best_svm_model = load('svm_model.joblib')
scaler = load('scaler.joblib')
# Load label encoders
airline_encoder = load('Airline_encoder.joblib')
source_encoder = load('Source_encoder.joblib')
destination_encoder = load('Destination_encoder.joblib')
total_stops_encoder = load('Total_Stops_encoder.joblib')
additional_info_encoder = load('Additional_Info_encoder.joblib')

def preprocess_airfare_data(airline, source, destination, total_stops, additional_info, date_of_journey, dep_time, arrival_time, duration):
    # Process each categorical feature using its corresponding LabelEncoder
    airline_encoded = airline_encoder.transform([airline])[0]
    source_encoded = source_encoder.transform([source])[0]
    destination_encoded = destination_encoder.transform([destination])[0]
    additional_info_encoded = additional_info_encoder.transform([additional_info])[0]
    total_stops_encoded = total_stops_encoder.transform([total_stops])[0]
    
    # Process date and time features
    journey_datetime = datetime.strptime(date_of_journey, '%Y-%m-%d')
    journey_day = journey_datetime.day
    journey_month = journey_datetime.month
    dep_hour, dep_minute = dep_time.hour, dep_time.minute
    arrival_hour, arrival_minute = arrival_time.hour, arrival_time.minute
    
    # Convert duration to total minutes
    duration_hours, duration_minutes = 0, 0
    if 'h' in duration:
        duration_hours = int(duration.split('h')[0])
        duration_minutes = int(duration.split('m')[0].split()[-1]) if 'm' in duration else 0
    else:  # if the duration is only in minutes (e.g., '50m')
        duration_minutes = int(duration.split('m')[0])
    duration_in_minutes = duration_hours * 60 + duration_minutes

    # Combine all features into a single array
    features = np.array([[
        airline_encoded, source_encoded, destination_encoded,
        journey_day, journey_month,
        dep_hour, dep_minute, arrival_hour, arrival_minute,
        duration_in_minutes, total_stops_encoded, additional_info_encoded
    ]])
    
    return features


def preprocess_rice_data(area, major_axis_length, minor_axis_length, eccentricity, extent):
    features = np.array([[area, major_axis_length, minor_axis_length, eccentricity, extent]])
    scaled_features = scaler.transform(features)
    return scaled_features

# Streamlit UI
st.sidebar.title('Choose a ML Application:')
project = st.sidebar.radio("", ('AIRFARE PRICE PREDICTION USING XGB REGRESSOR', 'RICE CLASSIFICATION USING SUPPORT VECTOR MACHINE'))

if project == 'AIRFARE PRICE PREDICTION USING XGB REGRESSOR':
    st.title('Airfare Price Prediction using XGB Regressor')

   # Dropdowns for categorical features using the original string labels
    airline = st.selectbox('Airline', options=airline_encoder.classes_)
    source = st.selectbox('Source', options=source_encoder.classes_)
    destination = st.selectbox('Destination', options=destination_encoder.classes_)
    total_stops = st.selectbox('Total Stops', options=total_stops_encoder.classes_)
    additional_info = st.selectbox('Additional Info', options=additional_info_encoder.classes_)
    

    date_of_journey = st.date_input('Date of Journey').strftime('%Y-%m-%d')
    dep_time = st.time_input('Departure Time')
    arrival_time = st.time_input('Arrival Time')
    duration = st.text_input('Duration', '2h 30m')
    if st.button('Predict Airfare'):
        data = preprocess_airfare_data(airline, source, destination, total_stops, additional_info, date_of_journey, dep_time, arrival_time, duration)
        prediction = best_xgb_model.predict(data)
        st.success(f'Predicted Airfare: ₹{prediction[0]:.2f}')

elif project == 'RICE CLASSIFICATION USING SUPPORT VECTOR MACHINE':
    st.title('Rice Classification using Support Vector Machine')
    area = st.number_input('Area', min_value=0)
    major_axis_length = st.number_input('Major Axis Length', min_value=0.0, format='%f')
    minor_axis_length = st.number_input('Minor Axis Length', min_value=0.0, format='%f')
    eccentricity = st.number_input('Eccentricity', min_value=0.0, format='%f')
    extent = st.number_input('Extent', min_value=0.0, format='%f')
    if st.button('Classify Rice'):
        data = preprocess_rice_data(area, major_axis_length, minor_axis_length, eccentricity, extent)
        prediction = best_svm_model.predict(data)
        st.success(f'Rice Class Prediction: {"Jasmine" if prediction[0] == 1 else "Gonen"}')