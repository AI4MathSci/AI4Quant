import random
from collections import Counter
from utils.config import CONFIG
import matplotlib.pyplot as plt
import pandas as pd

def simulate_shortmak(signals):
    return Counter(signals).most_common(1)[0][0]

def generate_mock_signals(true_signal="buy", num_chains=7, noise=0.3):
    options = ["buy", "hold", "sell"]
    return [
        random.choice([o for o in options if o != true_signal]) if random.random() < noise else true_signal
        for _ in range(num_chains)
    ]

def run_experiment():
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = []

    for noise in noise_levels:
        signals = generate_mock_signals(
    true_signal=CONFIG["true_signal"],
    num_chains=CONFIG["num_reasoning_chains"],
    noise=CONFIG["noise_level"]
)
        voted = simulate_shortmak(signals)
        results.append({
            "noise": noise,
            "voted": voted,
            "signals": signals
        })

    df = pd.DataFrame(results)
    print(df[["noise", "voted"]])
    plot_results(df)

def plot_results(df):
    mapping = {"buy": 1, "hold": 0, "sell": -1}
    plt.plot(df["noise"], df["voted"].map(mapping), marker="o")
    plt.title("Shortm@k Signal vs. Noise Level")
    plt.xlabel("Noise")
    plt.ylabel("Signal (1=Buy, 0=Hold, -1=Sell)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()

