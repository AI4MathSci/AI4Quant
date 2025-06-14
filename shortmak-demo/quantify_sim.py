# shortmak-demo/quantify_sim.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --- Simulated Signal Generators (these are the "short reasoning chains") ---

def signal_sma_crossover(price):
    sma_short = price.rolling(window=5).mean()
    sma_long = price.rolling(window=20).mean()
    signal = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)
    return signal.fillna(0)

def signal_rsi(price):
    delta = price.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signal = ((rsi < 30).astype(int)) - ((rsi > 70).astype(int))
    return pd.Series(signal).fillna(0)

def signal_momentum(price):
    return np.sign(price.diff(periods=3)).fillna(0)

# --- Shortm@k Voting Logic ---

def shortmak_vote(signals, k=3):
    """
    signals: list of pandas Series (same length)
    k: number of signals to stop at for voting
    """
    vote_output = []
    length = len(signals[0])
    for i in range(length):
        votes = [s.iloc[i] for s in signals[:k]]
        vote = Counter(votes).most_common(1)[0][0]
        vote_output.append(vote)
    return pd.Series(vote_output)

# --- Main Execution ---

if __name__ == "__main__":
    # Generate fake price data
    np.random.seed(42)
    price = pd.Series(np.cumsum(np.random.randn(100)) + 100)

    # Run individual signals
    s1 = signal_sma_crossover(price)
    s2 = signal_rsi(price)
    s3 = signal_momentum(price)

    # Apply Shortm@k majority vote
    final_signal = shortmak_vote([s1, s2, s3], k=3)

    # Output results
    df = pd.DataFrame({
        'price': price,
        'sma_cross': s1,
        'rsi': s2,
        'momentum': s3,
        'final_signal': final_signal
    })

    df.to_csv("shortmak_quantify_output.csv", index=False)
    print("Saved: shortmak_quantify_output.csv")

    # Optional: Visualize
    plt.figure(figsize=(10, 4))
    plt.plot(price, label="Price")
    plt.plot(df[df['final_signal'] == 1].index, price[df['final_signal'] == 1], '^', color='g', label="Buy")
    plt.plot(df[df['final_signal'] == -1].index, price[df['final_signal'] == -1], 'v', color='r', label="Sell")
    plt.title("Shortm@k Final Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("shortmak_quantify_plot.png")
    plt.show()

