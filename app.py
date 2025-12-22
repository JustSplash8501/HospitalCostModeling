# Database Contents License (DbCL)
# Copyright (C) 2025 JustSplash8501
# Licensed under the Database Contents License (DbCL).
# The contents are provided "as is" without warranty of any kind.

# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Hospital Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model():
    loaded = joblib.load('models/tuned_rf_model.pkl')
    # Check if it's a dictionary (model artifact) or direct model
    if isinstance(loaded, dict):
        return loaded['model']
    return loaded

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'tuned_rf_model.pkl' is in the 'models/' directory.")
    st.stop()

# Header
st.title("🏥 Hospital Insurance Cost Predictor")
st.markdown("""
Predict annual insurance charges based on patient demographics and health factors.  
""")

st.markdown("---")

# Input form in two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.slider("Age", min_value=18, max_value=100, value=35, step=1)
    sex = st.selectbox("Sex", options=["male", "female"])
    children = st.slider("Number of Children", min_value=0, max_value=5, value=0, step=1)
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

with col2:
    st.subheader("Health Information")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=27.5, step=0.1)
    smoker = st.selectbox("Smoking Status", options=["no", "yes"])

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Cost", type="primary", use_container_width=True):
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Make prediction with confidence interval
    try:
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence interval using individual tree predictions
        rf_regressor = model.named_steps['regressor']
        X_processed = model.named_steps['preprocess'].transform(input_data)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_processed)[0] for tree in rf_regressor.estimators_])
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bound = np.percentile(tree_predictions, lower_percentile)
        upper_bound = np.percentile(tree_predictions, upper_percentile)
        std_dev = np.std(tree_predictions)
        
        # Display prediction
        st.success(f"### 💰 Estimated Annual Insurance Cost: ${prediction:,.2f}")
        
        # Add basic stats
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Point Estimate", f"${prediction:,.2f}")
        
        with col4:
            st.metric("Lower Bound (95%)", f"${lower_bound:,.2f}")
        
        with col5:
            st.metric("Upper Bound (95%)", f"${upper_bound:,.2f}")
        
        # Additional stats
        col6, col7, col8 = st.columns(3)
        
        with col7:
            interval_width = upper_bound - lower_bound
            st.metric("Prediction Uncertainty", f"±${interval_width/2:,.2f}", 
                     help="Half-width of the 95% confidence interval")
        
        # Cost breakdown insights
        st.markdown("---")
        st.subheader("💡 Cost Insights")
        
        insights = []
        if smoker == "yes":
            insights.append("🚬 **Smoking status** is the strongest predictor of higher costs (typically +$23,000)")
        if bmi >= 30:
            insights.append("⚖️ **High BMI** (obese category) contributes to increased costs")
        if age >= 50:
            insights.append("👴 **Age over 50** increases expected healthcare costs")
        if children >= 3:
            insights.append("👨‍👩‍👧‍👦 **Multiple children** may increase family insurance costs")
        
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.markdown("✅ Your profile suggests lower-than-average insurance costs")
        
        # Additional information
        with st.expander("ℹ️ About this prediction"):
            st.markdown(f"""
            **Input Summary:**
            - Age: {age} years old
            - Sex: {sex.capitalize()}
            - BMI: {bmi:.1f}
            - Children: {children}
            - Smoker: {smoker.capitalize()}
            - Region: {region.capitalize()}
            
            **Prediction Details:**
            - Point Estimate: ${prediction:,.2f}
            - 95% Confidence Range: ±${(upper_bound - lower_bound)/2:,.2f}
            
            **Confidence Interval Interpretation:**
            We are 95% confident that the true cost for someone with your profile 
            falls between ${lower_bound:,.2f} and ${upper_bound:,.2f}. This interval 
            is calculated from the predictions of all {len(tree_predictions)} individual 
            trees in the Random Forest model.
            
            **Model Information:**s
            - Algorithm: Random Forest Regressor (Tuned)
            - Number of Trees: {len(tree_predictions)}
            - Training R²: 0.91
            - Test R²: 0.88
            - Test MAE: $2,481
            
            **Top Features by Importance:**
            1. Smoking status
            2. Age
            3. BMI
            4. Number of children
            5. Sex and region (minor influence)
            """)
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Random Forest Model | 
    <a href='https://github.com/JustSplash8501/hospital-cost-modeling'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts annual hospital insurance charges using a machine learning model 
    trained on demographic and health data.
    
    **Key Features:**
    - 88% variance explained (R²)
    - Mean error of ±$2,481
    """)
    
    st.markdown("---")
    st.header("Dataset Info")
    st.markdown("""
    - **Records:** 1,338 patients
    - **Features:** 6 (age, sex, BMI, children, smoker, region)
    - **Target:** Annual insurance charges
    - **Source:** Medical cost personal datasets
    """)
    
    st.markdown("---")
    st.markdown("**Model Version:** 1.0  \n**Last Updated:** December 2025")
