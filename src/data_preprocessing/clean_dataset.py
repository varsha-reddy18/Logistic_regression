import pandas as pd

# Load dataset
df = pd.read_csv("../../data/raw/titanic.csv")

# Select useful columns
df = df[[
    "survived",
    "pclass",
    "sex",
    "age",
    "fare",
    "embarked"
]]

# Handle missing values
df["age"].fillna(
    df["age"].median(),
    inplace=True
)

df["embarked"].fillna(
    df["embarked"].mode()[0],
    inplace=True
)

# Save cleaned dataset
df.to_csv(
    "../../data/cleaned/cleaned_titanic.csv",
    index=False
)

print("Data cleaned successfully.")