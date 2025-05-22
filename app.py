import streamlit as st
from model import predict
from sklearn.datasets import load_breast_cancer

# Title
st.title("🔬 Breast Cancer Prediction")

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names

# Input form
st.sidebar.header("Enter the following features:")
input_data = []
for feature in feature_names:
    value = st.sidebar.slider(f"{feature}", float(data.data[:, data.feature_names.tolist().index(feature)].min()),
                               float(data.data[:, data.feature_names.tolist().index(feature)].max()))
    input_data.append(value)

# Prediction
if st.button("Predict"):
    result = predict(input_data)
    st.success(f"🎉 The prediction is: **{result}**")
