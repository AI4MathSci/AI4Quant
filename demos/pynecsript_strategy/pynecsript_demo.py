import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic price data
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=200)
prices = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)

# Generate synthetic sentiment (range -1 to 1)
sentiment = pd.Series(np.random.uniform(-1, 1, size=200), index=dates)

# Strategy Logic: Moving Average Crossover + Sentiment Filter
df = pd.DataFrame({'price': prices, 'sentiment': sentiment})
df['sma_fast'] = df['price'].rolling(window=5).mean()
df['sma_slow'] = df['price'].rolling(window=20).mean()

# Buy when fast MA crosses above slow MA and sentiment > 0
df['signal'] = 0
df.loc[(df['sma_fast'] > df['sma_slow']) & (df['sentiment'] > 0), 'signal'] = 1
df.loc[(df['sma_fast'] < df['sma_slow']) & (df['sentiment'] < 0), 'signal'] = -1

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df['price'], label='Price')
plt.plot(df['sma_fast'], label='SMA Fast (5)', linestyle='--')
plt.plot(df['sma_slow'], label='SMA Slow (20)', linestyle='--')
plt.fill_between(df.index, 0, df['sentiment'] * 10, color='gray', alpha=0.2, label='Sentiment (scaled)')

# Plot buy/sell signals
plt.scatter(df[df['signal'] == 1].index, df[df['signal'] == 1]['price'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(df[df['signal'] == -1].index, df[df['signal'] == -1]['price'], marker='v', color='red', label='Sell Signal', alpha=1)

plt.title("Pure Python Strategy: Moving Avg Crossover + Sentiment Filter")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pynescript_strategy_output.png")
plt.show()

# Save output to CSV
df.to_csv("pynescript_strategy_output.csv")

