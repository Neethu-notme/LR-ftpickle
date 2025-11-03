import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# --------------------------
# App Title
# --------------------------
st.title("🚢 Titanic Survival Prediction (Trained Inside Streamlit)")
st.write("This app trains a Logistic Regression model on the Titanic dataset and predicts survival.")

# --------------------------
# Load dataset
# --------------------------
uploaded_file = st.file_uploader("Upload Titanic dataset CSV (with columns like Pclass, Sex, Age, Fare, Embarked, Survived)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset loaded successfully!")
    st.write(df.head())

    # --------------------------
    # Preprocessing
    # --------------------------
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'].fillna('S', inplace=True)
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    df = df.fillna(df.mean(numeric_only=True))

    # --------------------------
    # Train model
    # --------------------------
    X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    y = df['Survived']

    model = LogisticRegression()
    model.fit(X, y)
    st.success("✅ Model trained successfully!")

    # --------------------------
    # Prediction inputs
    # --------------------------
    st.subheader("🔍 Predict Survival")
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
    embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

    # Encode user input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [le_sex.transform([sex])[0]],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [le_embarked.transform([embarked])[0]]
    })

    # --------------------------
    # Predict
    # --------------------------
    if st.button("Predict Survival"):
        pred_prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.write(f"**Prediction:** {'Survived ✅' if pred == 1 else 'Did not survive ❌'}")
        st.write(f"**Survival Probability:** {pred_prob:.2f}")

else:
    st.warning("👆 Please upload your Titanic dataset CSV to start.")
