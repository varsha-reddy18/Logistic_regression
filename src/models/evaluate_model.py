import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    "../../data/processed/encoded_titanic.csv"
)

X = df.drop("survived", axis=1)

y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = joblib.load(
    "../../saved_models/scaler.pkl"
)

X_test = scaler.transform(X_test)

model = joblib.load(
    "../../saved_models/logistic_model.pkl"
)

y_pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    confusion_matrix(y_test, y_pred)
)

print(
    classification_report(y_test, y_pred)
)