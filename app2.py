import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model
model = load_model('/Users/godholdalomenu/Desktop/ChurnPrediction/model3.h5')

# Load the trained scaler
scaler_path = '/Users/godholdalomenu/Desktop/ChurnPrediction/churn_scalerx.pkl'

try:
    loaded_scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    loaded_scaler = None

def preprocess_input(input_data):
    if loaded_scaler is None:
        st.error("Scaler not loaded successfully. Please check the file path and ensure the file is not corrupted.")
        return None

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Handle missing values, replace empty strings with NaN
    input_df.replace('', np.nan, inplace=True)

    # Convert numerical columns to float
    numerical_cols = ['TotalCharges']  # Add other numerical columns if needed
    input_df[numerical_cols] = input_df[numerical_cols].apply(pd.to_numeric, errors='coerce')

    # Select columns in the correct order
    input_df = input_df[['Partner', 'Dependents', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'ServiceFeatures']]

    # Label encode categorical variables
    label_encoder = LabelEncoder()
    categorical_cols = ['Contract', 'Partner', 'Dependents', 'PaperlessBilling', 'PaymentMethod', 'ServiceFeatures']
    for col in categorical_cols:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    # Use the loaded scaler to transform the input data
    input_scaled = loaded_scaler.transform(input_df)

    return input_scaled

def predict_churn(input_data):
    if model is None:
        st.error("Model not loaded successfully.")
        return None

    # Use your loaded model for prediction
    prediction_labels = model.predict(input_data)

    # Example assuming a binary classification model
    confidence = None
    if len(prediction_labels.shape) == 2 and prediction_labels.shape[1] == 1:
        # For binary classification, take the probability of the positive class
        confidence = prediction_labels.mean()

    return prediction_labels, confidence

def main():
    st.title("Churn Prediction")

    # Get input from the user
    contract_options = ['Month-to-month', 'One year', 'Two year']
    contract = st.selectbox("Contract", contract_options)

    partner_options = ['Yes', 'No']
    partner = st.radio("Partner", partner_options)

    dependents_options = ['Yes', 'No']
    dependents = st.radio("Dependents", dependents_options)

    paperless_billing_options = ['Yes', 'No']
    paperless_billing = st.radio("Paperless Billing", paperless_billing_options)

    payment_method_options = ['Cash', 'Electronic check']
    payment_method = st.selectbox("Payment Method", payment_method_options)

    total_charges = st.number_input("Total Charges", value=0.0)
    service_features = st.text_input("Service Features")

    # Load the trained scaler
    scaler_path = '/Users/godholdalomenu/Desktop/ChurnPrediction/churn_scalerx.pkl'
    
    try:
        loaded_scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        loaded_scaler = None

    if loaded_scaler is not None:
        # Use the loaded scaler to transform the input data
        input_scaled = preprocess_input({
            'Contract': contract,
            'Partner': partner,
            'Dependents': dependents,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'TotalCharges': total_charges,
            'ServiceFeatures': service_features
        })

        if input_scaled is not None:
            # Get churn prediction and confidence
            churn_prediction, confidence = predict_churn(input_scaled)

            if churn_prediction is not None:
                # Display prediction result
                st.header("Churn Prediction Result")
                st.write(f"Selected Contract: {contract}")
                st.write(f"Selected Partner: {partner}")
                st.write(f"Selected Dependents: {dependents}")
                st.write(f"Selected PaperlessBilling: {paperless_billing}")
                st.write(f"Selected PaymentMethod: {payment_method}")
                st.write(f"Selected TotalCharges: {total_charges}")
                st.write(f"Selected ServiceFeatures: {service_features}")
                predicted_label = 'Yes' if churn_prediction[0] == 1 else 'No'
                st.write(f"Predicted Churn: {predicted_label}")
                st.write(f"Confidence: {confidence}")

if __name__ == '__main__':
    main()

