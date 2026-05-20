import pandas as pd

csv_path = "analytics/misclassified/misclassified.csv"

df = pd.read_csv(csv_path)

# highest confidence wrong predictions first
df = df.sort_values(by="confidence_score", ascending=False)

top_n = 100

df.head(top_n).to_csv(
    "analytics/misclassified/top_confused.csv",
    index=False
)

print(f"Saved top {top_n} suspicious predictions")

