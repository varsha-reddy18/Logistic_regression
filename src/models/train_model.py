import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

joblib.dump(
    model,
    "../../saved_models/logistic_model.pkl"
)

joblib.dump(
    scaler,
    "../../saved_models/scaler.pkl"
)

print("Model trained successfully.")