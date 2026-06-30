# 🚢 Titanic Survival Predictor

An AI-powered web application that predicts whether a Titanic passenger would have survived, using K-Nearest Neighbors classification.

## 🚀 Live Demo
https://titanic-survival-prediction-ml-raafe.streamlit.app/

## 📌 Project Overview
This project uses machine learning to predict passenger survival probability based on the famous Titanic dataset. The model is trained on 891 passenger records including class, age, sex, fare, and family details.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Plotly

## ⚙️ Data Preprocessing
- Missing value imputation (median for Age, mode for Embarked)
- Label Encoding for Sex
- One-Hot Encoding for Embarked port
- Standard Scaling for feature normalization
- Dropped irrelevant columns (PassengerId, Name, Cabin, Ticket)

## 🤖 ML Model
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Tuned Parameters:** n_neighbors, weights, distance metric
- **Train/Test Split:** 80/20

## 📊 Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report
- Best parameters selected via cross-validation

## 🖥️ App Features
- Interactive passenger detail input form
- Real-time survival prediction
- Animated probability gauge (Plotly)
- Clean, professional light-themed dashboard UI

## 👨‍💻 Author
**Muhammad Raafe Memon**
Aspiring AI & Machine Learning Engineer
