import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f7f8fa;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .navbar {
        background: #ffffff;
        padding: 18px 32px;
        border-bottom: 1px solid #e5e7eb;
        margin: -1rem -4rem 2rem -4rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .navbar-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
    }
    .navbar-badge {
        background: #eff6ff;
        color: #2563eb;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        margin-left: auto;
    }

    .hero {
        text-align: center;
        padding: 8px 0 32px 0;
    }
    .hero h1 {
        font-size: 30px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .hero p {
        font-size: 15px;
        color: #64748b;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 2px;
    }

    .form-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 28px 28px 12px 28px;
        margin-bottom: 20px;
    }
    .form-card h3 {
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #475569 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        padding: 13px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 600;
        margin-top: 8px;
    }
    .stButton > button:hover {
        background: #1d4ed8;
    }

    .result-card-survived {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }
    .result-card-died {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }
    .result-icon { font-size: 36px; margin-bottom: 8px; }
    .result-title-survived { font-size: 20px; font-weight: 700; color: #15803d; }
    .result-title-died { font-size: 20px; font-weight: 700; color: #b91c1c; }
    .result-sub { font-size: 14px; color: #64748b; margin-top: 4px; }

    .gauge-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 16px 0 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    df = pd.read_csv("Titanic-Dataset.csv")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    x = df.drop(["PassengerId", "Name", "Cabin", "Ticket", "Survived"], axis=1)
    y = df["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
    grid.fit(x_train, y_train)

    acc = accuracy_score(y_test, grid.predict(x_test))
    return grid, scaler, x.columns.tolist(), acc, grid.best_params_


model, scaler, feature_cols, acc, best_params = train_model()

# --- Navbar ---
st.markdown("""
<div class="navbar">
    <span style="font-size:22px;">🚢</span>
    <span class="navbar-title">Titanic Survival Predictor</span>
    <span class="navbar-badge">ML Model · Live</span>
</div>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero">
    <h1>Will this passenger survive?</h1>
    <p>Enter passenger details below to get an instant survival prediction</p>
</div>
""", unsafe_allow_html=True)

# --- Stats Row ---
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{acc:.0%}</div>
        <div class="stat-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">K = {best_params['n_neighbors']}</div>
        <div class="stat-label">Optimal Neighbors</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">KNN</div>
        <div class="stat-label">Algorithm</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Form Card ---
st.markdown('<div class="form-card"><h3>Passenger details</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2, format_func=lambda x: f"Class {x}")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)

with col2:
    sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare paid ($)", min_value=0.0, max_value=600.0, value=32.0)

with col3:
    embarked = st.selectbox("Port of embarkation", ["Southampton", "Cherbourg", "Queenstown"])

st.markdown("</div>", unsafe_allow_html=True)

predict = st.button("Predict survival")

# --- Predict Logic ---
if predict:

    embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
    embarked_code = embarked_map[embarked]

    input_dict = {col: [0] for col in feature_cols}
    input_dict["Pclass"] = [pclass]
    input_dict["Sex"] = [1 if sex == "Male" else 0]
    input_dict["Age"] = [age]
    input_dict["SibSp"] = [sibsp]
    input_dict["Parch"] = [parch]
    input_dict["Fare"] = [fare]

    for col in feature_cols:
        if "Embarked_" in col:
            input_dict[col] = [1 if col == f"Embarked_{embarked_code}" else 0]

    input_df = pd.DataFrame(input_dict)[feature_cols]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card-survived">
                <div class="result-icon">🛟</div>
                <div class="result-title-survived">Survived</div>
                <div class="result-sub">{probability:.0%} survival probability</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-died">
                <div class="result-icon">⚠️</div>
                <div class="result-title-died">Did not survive</div>
                <div class="result-sub">{probability:.0%} survival probability</div>
            </div>""", unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": "#0f172a", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#16a34a" if probability > 0.5 else "#dc2626"},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#fef2f2"},
                    {"range": [40, 70], "color": "#fffbeb"},
                    {"range": [70, 100], "color": "#f0fdf4"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 2},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a"},
            height=240,
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
