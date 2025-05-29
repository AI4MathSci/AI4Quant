# 🧠 Short-m@k Strategy Voting Demo

This demo implements the **Short-m@k** method — a lightweight ensemble technique that uses short reasoning chains and majority voting to select trading signals.

## 📈 What is Short-m@k?

Instead of relying on one long and complex model output, we generate multiple **short reasoning chains** (e.g., 5 quick models or strategies), then:

1. **Early stopping** after a few (e.g., 3) chains complete.
2. **Majority vote** among them to decide:
   - `BUY`, `SELL`, or `HOLD`

## 💡 Why It Matters

- Faster & cheaper inference than long-chain reasoning
- Avoids overfitting by promoting diverse short models
- Demonstrated better accuracy in recent LLM research

## 🧪 What's in This Demo?

- `shortmak_demo.py`: Simulates 5 short strategies with slight randomness
- Applies majority voting to determine the final trading action
- Outputs:
  - CSV of raw votes
  - Final decision
  - Simple matplotlib plot (optional)

## 🚀 How to Run

1. Clone the repo and navigate to the folder:
   ```bash
   git clone https://github.com/AI4MathSci/AI4Quant.git
   cd AI4Quant/demos/shortmak_demo

