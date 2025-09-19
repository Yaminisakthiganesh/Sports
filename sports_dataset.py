import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("sports_dataset.csv")

print("Sample Data:")
print(df.head())

# ------------------------
# Linear Regression
# ------------------------
X = df[["Age", "TrainingHours", "Height", "Weight", "MatchesPlayed"]]
y_linear = df["PerformanceScore"]

X_train, X_test, y_train, y_test = train_test_split(X, y_linear, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_linear = lin_model.predict(X_test)

print("\n📌 Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R² Score:", r2_score(y_test, y_pred_linear))

# ------------------------
# Logistic Regression
# ------------------------
y_logistic = df["Selected"]

X_train, X_test, y_train, y_test = train_test_split(X, y_logistic, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_logistic = log_model.predict(X_test)

print("\n📌 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Classification Report:\n", classification_report(y_test, y_pred_logistic))
