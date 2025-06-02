
import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment data
df = pd.read_csv("example_sentiment.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Plot sentiment score over time
plt.figure(figsize=(10, 5))
for source in df["source"].unique():
    subset = df[df["source"] == source]
    plt.plot(subset["timestamp"], subset["sentiment_score"], label=source)

plt.title("Sentiment Score Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sentiment_over_time.png")
plt.show()
