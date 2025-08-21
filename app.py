import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Load preprocessing objects (same as client.py)
@st.cache_resource
def load_preprocessing():
    encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return encoder, scaler


encoder, scaler = load_preprocessing()

# Recreate the CNN model architecture (must match client.py)
def create_cnn_model():
    """Must match the architecture from client.py exactly"""
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(13,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

@st.cache_resource
def load_model():
    # Create model architecture
    model = create_cnn_model()
    
    # Load saved weights
    with open("global_model_weights.pkl", "rb") as f:
        weights = pickle.load(f)
    
    # Set weights
    model.set_weights(weights)
    return model

# Load model once when app starts
model = load_model()

# Streamlit UI
st.title("Fraud Detection using Global Model")

# Input form
with st.form("transaction_form"):
    st.header("Transaction Details")
    
    # Categorical features (match your dataset)
    transaction_type = st.selectbox("Transaction Type", ["transfer", "withdrawal", "deposit"])
    source_account = st.text_input("Source Account (ID)")
    destination_account = st.text_input("Destination Account (ID)")
    transaction_mode = st.selectbox("Transaction Mode", ["mobile", "web", "atm"])
    currency = st.selectbox("Currency", ["USD", "EUR", "GBP"])
    device_type = st.selectbox("Device Type", ["mobile", "desktop", "tablet"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])
    
    # Numerical features
    timestamp = st.date_input("Date")
    transaction_amount = st.number_input("Amount", min_value=0.0)
    transaction_frequency = st.number_input("Frequency (txns/hour)", min_value=0)
    balance_before = st.number_input("Balance Before", min_value=0.0)
    balance_after = st.number_input("Balance After", min_value=0.0)
    average_transaction_value = st.number_input("Avg Transaction Value", min_value=0.0)
    
    submitted = st.form_submit_button("Check for Fraud")

# Preprocessing and prediction
if submitted:
    # Create DataFrame with input data
    input_data = pd.DataFrame([[
        transaction_type,
        source_account,
        destination_account,
        transaction_mode,
        currency,
        device_type,
        region,
        timestamp,
        transaction_amount,
        transaction_frequency,
        balance_before,
        balance_after,
        average_transaction_value
    ]], columns=[
        "transaction_type", "source_account", "destination_account", "transaction_mode",
        "currency", "device_type", "region", "timestamp", "transaction_amount",
        "transaction_frequency", "balance_before", "balance_after", "average_transaction_value"
    ])
    
    # Preprocess exactly like client.py
    # Convert timestamp to UNIX
    input_data["timestamp"] = (pd.to_datetime(input_data["timestamp"]).astype("int64") // 10**9)
    
    # Encode categoricals
    cat_cols = ["transaction_type", "source_account", "destination_account", 
               "transaction_mode", "currency", "device_type", "region"]
    for col in cat_cols:
        input_data[col] = encoder.fit_transform(input_data[col])
    
    # Scale numericals
    num_cols = ["timestamp", "transaction_amount", "transaction_frequency", 
               "balance_before", "balance_after", "average_transaction_value"]
    
    scaler = StandardScaler()
    input_data[num_cols] = scaler.fit(input_data[num_cols]).transform(input_data[num_cols])
     
    scaler.fit(input_data[num_cols])
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    

    
    # Predict
    prediction = model.predict(input_data.values)
    fraud_prob = prediction[0][0]
    
    # Display results
    st.subheader("Result")
    if fraud_prob > 0.5:
        st.error(f"⚠️ Fraud Detected (confidence: {fraud_prob:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (confidence: {1-fraud_prob:.2%})")
    
    # Show raw probability
    st.write(f"Fraud Probability: {fraud_prob:.4f}")