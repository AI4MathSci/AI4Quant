# pynescript_strategy/strategy_runner.py

import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import (
    calculate_cumulative_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_loss_ratio
)

# ----- Load Sample Data (or generate synthetic) -----
# Simulated price data + random buy/hold/sell signals
df = pd.DataFrame({
    'price': 100 + pd.Series(range(100)).apply(lambda x: x + (x ** 0.5) * 2).sample(frac=1).reset_index(drop=True),
})
df['signal'] = pd.Series([0, 1, -1] * 33 + [0]).sample(frac=1).reset_index(drop=True)

# ----- Strategy Logic -----
# Apply signal to returns (basic simulation)
df['returns'] = df['price'].pct_change().fillna(0)
df['strategy_returns'] = df['returns'] * df['signal'].shift(1).fillna(0)

# ----- Metrics -----
cumulative = calculate_cumulative_returns(df['strategy_returns'])
sharpe = calculate_sharpe_ratio(df['strategy_returns'])
mdd = calculate_max_drawdown(cumulative)
wl = calculate_win_loss_ratio(df['signal'], df['returns'])

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {mdd:.2%}")
print(f"Win/Loss Ratio: {wl:.2f}")

# ----- Visualization -----
cumulative.plot(title="Cumulative Returns with Pynescript Strategy")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("pynescript_strategy/cumulative_returns.png")
plt.show()

