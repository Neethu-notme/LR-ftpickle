import os
import pickle
import streamlit as st
import pandas as pd

# ✅ Load model safely using pickle
model_path = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ logistic_model.pkl not found! Please ensure it's in the same folder as this script.")
else:
    with open(model_path, "rb") as f:
        #model = pickle.load(f)
        model = None
        st.warning("⚠️ Model not loaded. This is a test run.")

    # --- Streamlit UI ---
    st.title("🚢 Titanic Survival Predictor (Logistic Regression)")
    st.write("Predict the survival probability of a Titanic passenger using a trained Logistic Regression model.")

    # --- Inputs ---
    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # --- Data for prediction ---
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # --- Prediction ---
    if st.button("Predict Survival"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("🎯 Prediction Result:")
            st.write(f"**Prediction:** {'✅ Survived' if prediction == 1 else '❌ Did not survive'}")
            st.write(f"**Survival Probability:** {probability:.2f}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
            st.info("Make sure your logistic_model.pkl was trained using the same features.")

