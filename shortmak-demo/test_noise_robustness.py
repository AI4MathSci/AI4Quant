import numpy as np
from collections import Counter

def generate_noisy_signals(true_signal, n=7, noise_level=0.3):
    signals = []
    options = ["buy", "hold", "sell"]
    for _ in range(n):
        if np.random.rand() < noise_level:
            noisy = np.random.choice([s for s in options if s != true_signal])
            signals.append(noisy)
        else:
            signals.append(true_signal)
    return signals

def majority_vote(signals):
    return Counter(signals).most_common(1)[0][0]

# Test robustness
true_decision = "buy"
for noise in [0.1, 0.3, 0.5, 0.7]:
    signals = generate_noisy_signals(true_decision, noise_level=noise)
    voted = majority_vote(signals)
    print(f"Noise Level: {noise:.0%} | Signals: {signals} | Vote: {voted}")

