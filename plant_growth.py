# %%
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title
st.title("🌱 Plant Growth Classification App")

# Load the trained model
try:
    model = joblib.load("plant_growth_model.pkl")
except FileNotFoundError:
    st.error("Model file 'plant_growth_model.pkl' not found. Please place it in the same directory.")
    st.stop()

# Sidebar for user input
st.sidebar.header("Enter Plant Details")

# Example input features (adjust these to your dataset’s actual features!)
height = st.sidebar.slider("Height (cm)", min_value=5.0, max_value=200.0, value=50.0, step=1.0)
width = st.sidebar.slider("Width (cm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
sunlight_hours = st.sidebar.slider("Sunlight Hours per Day", min_value=0, max_value=12, value=6, step=1)

# Function to preprocess inputs
def preprocess_input(height, width, soil_moisture, sunlight_hours):
    data = {
        "height": height,
        "width": width,
        "soil_moisture": soil_moisture,
        "sunlight_hours": sunlight_hours
    }
    df = pd.DataFrame([data])
    return df

# Prediction button
if st.sidebar.button("Predict"):
    input_df = preprocess_input(height, width, soil_moisture, sunlight_hours)
    
    try:
        prediction = model.predict(input_df)[0]
        
        st.subheader("🌿 Prediction Result")
        st.write(f"The predicted plant growth class is: **{prediction}**")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Instructions
st.write("""
### Instructions
1. Use the sidebar to enter plant details.
2. Adjust sliders for features like Height, Width, Soil Moisture, and Sunlight Hours.
3. Click **Predict** to see the predicted plant growth classification.
""")



