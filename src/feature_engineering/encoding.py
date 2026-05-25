import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    "../../data/cleaned/cleaned_titanic.csv"
)

le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])

df["embarked"] = le.fit_transform(
    df["embarked"]
)

df.to_csv(
    "../../data/processed/encoded_titanic.csv",
    index=False
)

print(df.head())