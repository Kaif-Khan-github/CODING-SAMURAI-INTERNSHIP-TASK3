import streamlit as st
import pandas as pd
import joblib

# Set Streamlit page config
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    layout="centered"
)

# Load trained VotingClassifier model
model = joblib.load("voting_model.pkl")

# App title & description
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details below to check if they would've survived the Titanic disaster.")

# --- Passenger Input Section ---
st.header("🧾 Passenger Information")

pclass = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x} Class")
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=50.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# --- Input Preprocessing ---
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    return pd.DataFrame({
        'pclass': [pclass],
        'sex': [0 if sex == 'male' else 1],  # ✅ Corrected: 0 = male, 1 = female
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked_Q': [1 if embarked == 'Q' else 0],
        'embarked_S': [1 if embarked == 'S' else 0]
        # 'embarked_C' is automatically implied when both Q and S are 0
    })

# --- Prediction Section ---
if st.button("🎯 Predict Survival"):
    input_df = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🧠 Prediction Result:")
    if prediction == 1:
        st.success(f"🎉 This passenger **would SURVIVE** with a survival probability of **{proba:.2%}**.")
    else:
        st.error(f"💀 Unfortunately, this passenger **would NOT survive**. Survival chance: **{proba:.2%}**.")
