import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("voting_model.pkl")

# Page config
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

# Title
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Fill in the passenger details to predict survival chances on the Titanic.")

# --- User Inputs ---
pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"{x} Class")
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid (in $)", min_value=0.0, max_value=600.0, value=50.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# --- Preprocessing Function ---
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    df = pd.DataFrame({
        'pclass': [pclass],
        'sex': [1 if sex == 'male' else 0],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked_Q': [1 if embarked == 'Q' else 0],
        'embarked_S': [1 if embarked == 'S' else 0]
    })
    return df

# --- Prediction ---
if st.button("🔍 Predict Survival"):
    input_df = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success(f"🎉 The passenger **would SURVIVE** with a probability of **{proba:.2%}**.")
    else:
        st.error(f"💀 The passenger **would NOT survive**. Survival chance: **{proba:.2%}**.")
