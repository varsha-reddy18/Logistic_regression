import seaborn as sns
import pandas as pd

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Save raw dataset
df.to_csv(
    "../../data/raw/titanic.csv",
    index=False
)

print(df.head())