import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Load the models, scaler, and encoders
best_xgb_model = load('xgb_model.joblib')
best_rf_model = load('rf_model.joblib')
scaler = load('scaler.joblib')
# Load label encoders
airline_encoder = load('Airline_encoder.joblib')
source_encoder = load('Source_encoder.joblib')
destination_encoder = load('Destination_encoder.joblib')
total_stops_encoder = load('Total_Stops_encoder.joblib')
additional_info_encoder = load('Additional_Info_encoder.joblib')

# Function to generate time options at 5-minute intervals
def generate_time_options(interval_minutes=5):
    times = [(datetime.min + timedelta(minutes=i)).time().strftime('%H:%M')
             for i in range(0, 24*60, interval_minutes)]
    return times

time_options = generate_time_options()

# Function to check the input value range
def check_input(label, value, min_val, max_val):
    if not (min_val <= value <= max_val):
        st.error(f'Error: {label} must be between {min_val} and {max_val}')
        return False
    return True

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
    dep_time_obj = datetime.strptime(dep_time, '%H:%M')
    arrival_time_obj = datetime.strptime(arrival_time, '%H:%M')
    dep_hour, dep_minute = dep_time_obj.hour, dep_time_obj.minute
    arrival_hour, arrival_minute = arrival_time_obj.hour, arrival_time_obj.minute
    
    # Initialize duration in minutes
    duration_in_minutes = 0

    # Check if 'h' is in the duration string
    if 'h' in duration:
        parts = duration.split('h')
        hours = int(parts[0])
        minutes = int(parts[1].replace('m', '')) if 'm' in parts[1] else 0
        duration_in_minutes = hours * 60 + minutes
    elif 'm' in duration:
        # Only minutes are provided
        duration_in_minutes = int(duration.replace('m', ''))
    else:
        # Handle unexpected format, maybe default to 0 or raise an error
        st.error('Invalid duration format. Please enter the duration as "Xh Ym" or "Xm".')
        return None

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
    dep_time = st.selectbox('Departure Time', options=time_options)
    arrival_time = st.selectbox('Arrival Time', options=time_options)
    duration = st.text_input('Duration (e.g., "2h 30m" or "150m")')
    if st.button('Predict Airfare'):
        data = preprocess_airfare_data(airline, source, destination, total_stops, additional_info, date_of_journey, dep_time, arrival_time, duration)
        prediction = best_xgb_model.predict(data)
        st.success(f'Predicted Airfare: ₹{prediction[0]:.2f}')

elif project == 'RICE CLASSIFICATION USING SUPPORT VECTOR MACHINE':
    st.title('Rice Classification using Support Vector Machine')
    area = st.number_input('Area (Valid Range: 2500 - 11000)', min_value=2500, max_value=11000, value=2500, help="Enter a value between 2500 and 11000")
    major_axis_length = st.number_input('Major Axis Length (Valid Range: 70 - 190)', min_value=70, max_value=190, value=70, help="Enter a value between 70 and 190")
    minor_axis_length = st.number_input('Minor Axis Length (Valid Range: 30 - 90)', min_value=30, max_value=90, value=30, help="Enter a value between 30 and 90")
    eccentricity = st.number_input('Eccentricity (Valid Range: 0.6 - 1)', min_value=0.6, max_value=1.0, value=0.6, help="Enter a value between 0.6 and 1")
    extent = st.number_input('Extent (Valid Range: 0.3 - 1)', min_value=0.3, max_value=1.0, value=0.3, help="Enter a value between 0.3 and 1")

    if st.button('Classify Rice'):
        if all([
        check_input('Area', area, 2500, 11000),
        check_input('Major Axis Length', major_axis_length, 70, 190),
        check_input('Minor Axis Length', minor_axis_length, 30, 90),
        check_input('Eccentricity', eccentricity, 0.6, 1),
        check_input('Extent', extent, 0.3, 1)
        ]):
            # Preprocess and predict
            data = preprocess_rice_data(area, major_axis_length, minor_axis_length, eccentricity, extent)
            prediction = best_rf_model.predict(data)
            st.success(f'Rice Class Prediction: {"Jasmine" if prediction[0] == 1 else "Gonen"}')
