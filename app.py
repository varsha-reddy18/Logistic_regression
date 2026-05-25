import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Titanic Survival Dashboard",
    page_icon="🚢",
    layout="wide"
)

# ======================================
# LOAD CSS
# ======================================

def load_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ======================================
# LOAD DATA
# ======================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/encoded_titanic.csv"
    )

    return df

df = load_data()

# ======================================
# LOAD MODEL & SCALER
# ======================================

model = joblib.load(
    "saved_models/logistic_model.pkl"
)

scaler = joblib.load(
    "saved_models/scaler.pkl"
)

# ======================================
# FEATURES & TARGET
# ======================================

X = df.drop("survived", axis=1)

y = df["survived"]

# ======================================
# SCALE FEATURES
# ======================================

X_scaled = scaler.transform(X)

# ======================================
# PREDICTIONS
# ======================================

y_pred = model.predict(X_scaled)

# ======================================
# ACCURACY
# ======================================

accuracy = accuracy_score(
    y,
    y_pred
)

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="main-header">

<h1>🚢 Titanic Survival Prediction Dashboard</h1>

<p>
Machine Learning Dashboard using Logistic Regression
</p>

</div>
""", unsafe_allow_html=True)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("📌 Navigation")

section = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Dataset Overview",
        "Visualizations",
        "Model Performance",
        "Survival Prediction"
    ]
)

# ======================================
# DASHBOARD
# ======================================

if section == "Dashboard":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📈 Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])

    c2.metric("Columns", df.shape[1])

    c3.metric(
        "Survival Rate",
        f"{df['survived'].mean()*100:.2f}%"
    )

    c4.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # HEATMAP
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="RdPu",
        linewidths=1
    )

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# DATASET OVERVIEW
# ======================================

elif section == "Dataset Overview":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10))

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # DATA INFO
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧾 Dataset Information")

    info_df = pd.DataFrame({

        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values

    })

    st.table(info_df)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# VISUALIZATIONS
# ======================================

elif section == "Visualizations":

    # ==================================
    # SURVIVAL COUNT
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🚢 Survival Count")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x=df["survived"],
        palette="RdPu"
    )

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # AGE DISTRIBUTION
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("👶 Age Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df["age"],
        kde=True,
        color="#ff4b91"
    )

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# MODEL PERFORMANCE
# ======================================

elif section == "Model Performance":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📈 Model Accuracy")

    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # CONFUSION MATRIX
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(
        y,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu"
    )

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # CLASSIFICATION REPORT
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📄 Classification Report")

    report = classification_report(
        y,
        y_pred
    )

    st.text(report)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# SURVIVAL PREDICTION
# ======================================

elif section == "Survival Prediction":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 Predict Survival")

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        [0, 1]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    fare = st.slider(
        "Fare",
        0.0,
        600.0,
        50.0
    )

    embarked = st.selectbox(
        "Embarked",
        [0, 1, 2]
    )

    # ==================================
    # PREDICTION
    # ==================================

    if st.button("Predict Survival 🚀"):

        input_data = np.array([[
            pclass,
            sex,
            age,
            fare,
            embarked
        ]])

        input_data = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_data
        )[0]

        if prediction == 1:

            st.markdown(
                """
                <div class="prediction-box">
                    🎉 Passenger Survived
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="prediction-box">
                    ❌ Passenger Did Not Survive
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)