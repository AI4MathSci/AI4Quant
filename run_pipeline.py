# run_pipeline.py

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import (
    calculate_cumulative_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_loss_ratio
)

def simple_strategy(df: pd.DataFrame) -> pd.DataFrame:
    # Basic Moving Average Crossover
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    df['signal'] = 0
    df.loc[df['SMA_20'] > df['SMA_50'], 'signal'] = 1
    df.loc[df['SMA_20'] < df['SMA_50'], 'signal'] = -1
    return df

def evaluate_strategy(df: pd.DataFrame) -> dict:
    df['returns'] = df['price'].pct_change().fillna(0)
    cumulative = calculate_cumulative_returns(df['price'])
    return {
        'Sharpe Ratio': calculate_sharpe_ratio(df['returns']),
        'Max Drawdown': calculate_max_drawdown(cumulative),
        'Win/Loss Ratio': calculate_win_loss_ratio(df['signal'], df['returns'])
    }

def plot_strategy(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df['price'], label='Price')
    plt.plot(df['SMA_20'], label='SMA 20')
    plt.plot(df['SMA_50'], label='SMA 50')
    plt.title('Strategy with Moving Averages')
    plt.legend()
    plt.grid()
    plt.show()

def main(args):
    # Load or simulate data
    df = pd.read_csv('sentiment_demo/example_sentiment.csv')  # Replace with your own
    df = simple_strategy(df)
    metrics = evaluate_strategy(df)

    if args.plot:
        plot_strategy(df)

    if args.metrics:
        print("\nPerformance Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    if args.save_csv:
        df.to_csv('strategy_output.csv', index=False)
        print("\nSaved strategy_output.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategy pipeline")
    parser.add_argument('--plot', action='store_true', help='Plot strategy visualization')
    parser.add_argument('--metrics', action='store_true', help='Print strategy metrics')
    parser.add_argument('--save-csv', action='store_true', help='Save output to CSV')
    args = parser.parse_args()
    main(args)

