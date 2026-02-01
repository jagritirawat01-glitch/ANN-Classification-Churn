import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Load the saved model and preprocessors
model = tf.keras.models.load_model('model.h5')

## Load one hot encoder and the scalar
with open('scalar.pkl', 'rb') as f:
    scalar = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

#Take user input
st.header("Input Customer Details")
CreditScore = st.number_input("Credit Score")
Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
Gender = st.selectbox("Gender", label_encoder_gender.classes_) 
Age = st.slider("Age", 18, 100)
Tenure = st.slider("Tenure", 0, 10)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
NumOfProducts = st.slider("Number of Products", 1, 4)
HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])
EstimatedSalary = st.number_input("Estimated Salary")

# Encode gender
Gender_encoded = label_encoder_gender.transform([Gender])[0]

# Create DataFrame from input data
input_data = pd.DataFrame([{
    'CreditScore': CreditScore,
    'Gender': Gender_encoded,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}])

# Encode geography
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate the dataframes
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input
input_scaled = scalar.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write("Probability of Churn: {:.2f}".format(prediction_prob))
if prediction_prob > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is unlikely to churn")