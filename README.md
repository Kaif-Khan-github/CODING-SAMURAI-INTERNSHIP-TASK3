# 🚢 Titanic Survival Prediction App

This project is a machine learning web app that predicts whether a passenger would have survived the Titanic disaster, based on real historical data from the Titanic dataset.

Built as part of the **Coding Samurai Internship Challenge**, this app uses a voting ensemble model (Random Forest + XGBoost) and is deployed using **Streamlit**.

---

## 🎯 Features

- Takes user input like:
  - Passenger Class
  - Sex
  - Age
  - Number of Siblings/Spouses Aboard
  - Number of Parents/Children Aboard
  - Fare
  - Port of Embarkation
- Preprocesses input data
- Predicts survival probability in real-time
- Clean, interactive UI using Streamlit

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **scikit-learn**
- **XGBoost**
- **Streamlit**
- **Git & GitHub**

---

## 🤖 Model

- **RandomForestClassifier** with `max_depth=5`
- **XGBClassifier** with `max_depth=3`, `n_estimators=100`
- Combined using **VotingClassifier** with soft voting

---

## 🌐 Live App

👉https://coding-samurai-internship-task3-5cgu8qzvnw5g87px6np8ia.streamlit.app/

---

## 📂 Project Structure
├── Titanic_app.py # Streamlit frontend app
├── voting_model.pkl # Trained VotingClassifier model
├── requirements.txt # Python dependencies
├── README.md # Project documentation 


