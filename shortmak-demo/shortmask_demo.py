import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Simulate votes from 5 short reasoning chains
def simulate_votes():
    options = ['BUY', 'HOLD', 'SELL']
    votes = [random.choices(options, weights=[0.4, 0.3, 0.3])[0] for _ in range(5)]
    return votes

# Perform majority vote
def majority_vote(votes):
    vote_count = Counter(votes)
    return vote_count.most_common(1)[0][0]

# Run the demo
if __name__ == "__main__":
    votes = simulate_votes()
    result = majority_vote(votes)

    # Save to CSV
    df = pd.DataFrame({'Strategy': [f'Strategy_{i+1}' for i in range(5)], 'Vote': votes})
    df.to_csv("strategy_votes.csv", index=False)

    # Save final decision
    with open("final_decision.txt", "w") as f:
        f.write(f"Majority Vote Decision: {result}")

    # Print and plot
    print("Votes:", votes)
    print("Majority Decision:", result)

    vote_distribution = df['Vote'].value_counts()
    vote_distribution.plot(kind='bar', title='Strategy Vote Distribution')
    plt.ylabel('Count')
    plt.xlabel('Decision')
    plt.tight_layout()
    plt.savefig("vote_distribution.png")
    plt.show()

