# Sports Dataset Example - Linear & Logistic Regression (Streamlit Version)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import streamlit as st

# ------------------------
# 1. Create Sample Dataset
# ------------------------
np.random.seed(42)
n = 100

data = {
    "Age": np.random.randint(18, 35, n),
    "TrainingHours": np.random.randint(1, 10, n),
    "Height": np.random.randint(160, 200, n),
    "Weight": np.random.randint(55, 100, n),
    "MatchesPlayed": np.random.randint(0, 50, n)
}

df = pd.DataFrame(data)

# Target for Linear Regression (performance score = some function + noise)
df["PerformanceScore"] = (
    df["TrainingHours"] * 5 +
    df["MatchesPlayed"] * 2 +
    np.random.normal(0, 10, n)
).astype(int)

# Target for Logistic Regression (selection yes/no based on score threshold)
df["Selected"] = (df["PerformanceScore"] > 100).astype(int)

# ------------------------
# Streamlit UI
# ------------------------
st.title("🏏 Sports Dataset - Linear & Logistic Regression")
st.subheader("📊 Sample Data")
st.dataframe(df.head())

# ------------------------
# 2. Linear Regression
# ------------------------
X = df[["Age", "TrainingHours", "Height", "Weight", "MatchesPlayed"]]
y_linear = df["PerformanceScore"]

X_train, X_test, y_train, y_test = train_test_split(X, y_linear, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_linear = lin_model.predict(X_test)

st.subheader("📌 Linear Regression Results")
st.write("**Mean Squared Error (MSE):**", mean_squared_error(y_test, y_pred_linear))
st.write("**R² Score:**", r2_score(y_test, y_pred_linear))

# ------------------------
# 3. Logistic Regression
# ------------------------
y_logistic = df["Selected"]

X_train, X_test, y_train, y_test = train_test_split(X, y_logistic, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_logistic = log_model.predict(X_test)

st.subheader("📌 Logistic Regression Results")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred_logistic))
st.text("Classification Report:\n" + classification_report(y_test, y_pred_logistic))