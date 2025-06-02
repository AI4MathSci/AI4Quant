import pandas as pd
from utils.metrics import (
    calculate_cumulative_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_loss_ratio
)

# Sample test dataframe
df = pd.read_csv('pynescript_strategy/sample_strategy.csv')  # assumes columns: 'price', 'signal'

# Compute metrics
df['returns'] = df['price'].pct_change().fillna(0)
cumulative = calculate_cumulative_returns(df['returns'])
sharpe = calculate_sharpe_ratio(df['returns'])
mdd = calculate_max_drawdown(cumulative)
wl = calculate_win_loss_ratio(df['signal'], df['returns'])

# Display
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {mdd:.2%}")
print(f"Win/Loss Ratio: {wl:.2f}")

