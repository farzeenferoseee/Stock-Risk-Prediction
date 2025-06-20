import streamlit as st
import joblib
import pandas as pd

# Load model, label map, and feature names
try:
    model = joblib.load("lr_model.pkl")
    label_map = joblib.load("label_map.pkl")
    feature_names = joblib.load("feature_names.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'lr_model.pkl', 'label_map.pkl', and 'feature_names.pkl' are in the same directory.")
    st.stop()

# Title and instructions
st.title("Stock Risk Prediction")
st.write("Enter stock details, and we'll predict whether stock is overstocked or stockout.")

# These are the original features BEFORE one-hot encoding
categorical_base_features = ['season', 'item_category', 'supplier_reliability']

# Create input fields for numerical features
numerical_features = [col for col in feature_names if all(not col.startswith(f"{cat}_") for cat in categorical_base_features)]

user_input_dict = {}

for feature in numerical_features:
    user_input_dict[feature] = st.number_input(f"Enter {feature}", step=1.0)

# Dropdowns for categorical input
season_options = ['peak', 'off-peak']
item_category_options = ['essential', 'non-essential']
supplier_reliability_options = ['high', 'low']

categorical_input_dict = {
    'season': st.selectbox("Select season", season_options),
    'item_category': st.selectbox("Select item category", item_category_options),
    'supplier_reliability': st.selectbox("Select supplier reliability", supplier_reliability_options)
}

if st.button("Predict Stock Status"):
    # Merge numerical and categorical inputs
    user_input_combined = user_input_dict.copy()
    user_input_combined.update(categorical_input_dict)

    # Convert to DataFrame
    user_input_df = pd.DataFrame([user_input_combined])

    # One-hot encode categorical columns
    user_input_df = pd.get_dummies(user_input_df, columns=categorical_base_features)

    # Add missing dummy columns from training
    for col in feature_names:
        if col not in user_input_df.columns:
            user_input_df[col] = 0

    # Reorder to match training columns
    user_input_df = user_input_df[feature_names]

    # Predict
    prediction = model.predict(user_input_df)

    # Decode label
    predicted_status = list(label_map.keys())[list(label_map.values()).index(prediction[0])]

    # Output
    st.success(f"Predicted Stock Status: **{predicted_status}**")

    
