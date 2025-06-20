import streamlit as st
import joblib
import pandas as pd

# Load the saved model, label map, and feature names
try:
    model = joblib.load("lr_model.pkl")
    label_map = joblib.load("label_map.pkl")
    feature_names = joblib.load("feature_names.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'lr_model.pkl', 'label_map.pkl', and 'feature_names.pkl' are in the same directory.")
    st.stop()

# Streamlit UI
st.title("Stock Risk Prediction")
st.write("Enter stock details, and we'll predict whether stock is overstocked or stockout.")

# Define the original categorical features
original_categorical_features = ['season', 'item_category', 'supplier_reliability', 'item_category_essential', 'item_category_non-essential' , 'supplier_reliability_high', 'supplier_reliability_low']

# Create input fields for each feature
user_input_dict = {}

# Create input fields for numerical features
# Filter out the one-hot encoded columns from feature_names to get only the original numerical features
numerical_features = [col for col in feature_names if not any(col.startswith(cat + '_') for cat in original_categorical_features)]

for feature in numerical_features:
    user_input_dict[feature] = st.number_input(f"Enter {feature}")

# Create input fields for original categorical features with specific options
categorical_input_dict = {}
season_options = ['peak', 'off-peak']
item_category_options = ['essential', 'non-essential']
supplier_reliability_options = ['high', 'low']

categorical_input_dict['season'] = st.selectbox("Select season", season_options)
categorical_input_dict['item_category'] = st.selectbox("Select item category", item_category_options)
categorical_input_dict['supplier_reliability'] = st.selectbox("Select supplier reliability", supplier_reliability_options)


if st.button("Predict Stock Status"):
    # Combine numerical and categorical inputs
    user_input_combined = user_input_dict.copy()
    user_input_combined.update(categorical_input_dict)

    # Convert user input to a DataFrame
    user_input_df = pd.DataFrame([user_input_combined])

    # Perform one-hot encoding on user input
    user_input_df = pd.get_dummies(user_input_df, columns=original_categorical_features)

    # Ensure all columns from the training data are present in the user input DataFrame, add missing ones with 0
    for col in feature_names:
        if col not in user_input_df.columns:
            user_input_df[col] = 0

    # Reorder columns to match the training data
    user_input_df = user_input_df[feature_names]

    # Make prediction
    prediction = model.predict(user_input_df)

    # Map the prediction back to the original label
    predicted_status = list(label_map.keys())[list(label_map.values()).index(prediction[0])]

    # Display result
    st.success(f"Predicted Stock Status: **{predicted_status}**")

    st.markdown(f"""
        ### 📊 Your input:
    """)
    st.write(user_input_combined)
