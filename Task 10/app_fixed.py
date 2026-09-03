import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection", page_icon="")

st.title(" Credit Card Fraud Detection")
st.markdown("Enter transaction details and click **Predict**")

# Load the NEW model and scaler
model = joblib.load('fraud_model_fixed.pkl')
scaler = joblib.load('scaler_fixed.pkl')

# Feature names
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

st.subheader(" Adjust Features")

# Create sliders
input_values = []
cols = st.columns(4)

for i, name in enumerate(feature_names):
    col = cols[i % 4]
    if name == 'Time':
        val = col.slider(f"{name}", -3.0, 5.0, 0.0, 0.1)
    elif name == 'Amount':
        val = col.slider(f"{name}", -3.0, 8.0, 0.0, 0.1)
    else:
        val = col.slider(f"{name}", -5.0, 5.0, 0.0, 0.1)
    input_values.append(val)

# Predict button
if st.button(" Predict", type="primary", use_container_width=True):
    # Convert to array
    input_array = np.array(input_values).reshape(1, -1)
    
    # Scale ALL features (NOW THIS WORKS!)
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    
    if prediction == 1:
        st.error(f" FRAUD DETECTED! (Confidence: {probability*100:.2f}%)")
        st.warning(" This transaction appears suspicious. Please investigate.")
    else:
        st.success(f" LEGITIMATE TRANSACTION (Confidence: {(1-probability)*100:.2f}%)")
        st.info(" This transaction appears normal.")
    
    # Show probability bar
    st.progress(probability)
    st.caption(f"Fraud Probability: {probability*100:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit | Model: Logistic Regression + SMOTE | Recall: 92%")