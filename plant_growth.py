```python
import streamlit as st
import pandas as pd
import joblib

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

# Numeric inputs
Sunlight_Hours = st.sidebar.slider("Sunlight Hours per Day", min_value=0, max_value=12, value=6, step=1)
Temperature = st.sidebar.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
Humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
Growth_Milestone = st.sidebar.slider("Growth Milestone (e.g., days or stage index)", min_value=0, max_value=100, value=10, step=1)

# Categorical inputs
Soil_Type = st.sidebar.selectbox("Soil Type", ["clay", "loam", "sandy"])
Water_Frequency = st.sidebar.selectbox("Water Frequency", ["bi-weekly", "daily", "weekly"])
Fertilizer_Type = st.sidebar.selectbox("Fertilizer Type", ["none", "chemical", "organic"])

# Function to preprocess inputs (manual one-hot encoding)
def preprocess_input(Sunlight_Hours, Temperature, Humidity, Growth_Milestone,
                     Soil_Type, Water_Frequency, Fertilizer_Type):
    data = {
        "Sunlight_Hours": Sunlight_Hours,
        "Temperature": Temperature,
        "Humidity": Humidity,
        "Growth_Milestone": Growth_Milestone,
        # Soil type one-hot
        "Soil_Type_clay": 1 if Soil_Type == "clay" else 0,
        "Soil_Type_loam": 1 if Soil_Type == "loam" else 0,
        "Soil_Type_sandy": 1 if Soil_Type == "sandy" else 0,
        # Water frequency one-hot
        "Water_Frequency_bi-weekly": 1 if Water_Frequency == "bi-weekly" else 0,
        "Water_Frequency_daily": 1 if Water_Frequency == "daily" else 0,
        "Water_Frequency_weekly": 1 if Water_Frequency == "weekly" else 0,
        # Fertilizer type one-hot
        "Fertilizer_Type_none": 1 if Fertilizer_Type == "none" else 0,
        "Fertilizer_Type_chemical": 1 if Fertilizer_Type == "chemical" else 0,
        "Fertilizer_Type_organic": 1 if Fertilizer_Type == "organic" else 0,
    }
    return pd.DataFrame([data])

# Prediction button
if st.sidebar.button("Predict"):
    input_df = preprocess_input(Sunlight_Hours, Temperature, Humidity, Growth_Milestone,
                                Soil_Type, Water_Frequency, Fertilizer_Type)
    
    try:
        prediction = model.predict(input_df)[0]
        
        st.subheader("🌿 Prediction Result")
        st.success(f"The predicted plant growth class is: **{prediction}**")
        
        # If model supports probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            st.write("### Prediction Probabilities")
            st.write(dict(zip(model.classes_, probs)))
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Instructions
st.write("""
### Instructions
1. Use the sidebar to enter plant details.
2. Provide numeric values for Sunlight Hours, Temperature, Humidity, and Growth Milestone.
3. Select Soil Type, Water Frequency, and Fertilizer Type from the dropdowns.
4. Click **Predict** to see the predicted plant growth classification.
""")
```
