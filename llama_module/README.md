# LlamaIndex Pipeline Module

This module implements the integration of LlamaIndex (GPT-powered document parsing) with our quant-finance research tools.

## 🧠 What It Does

- Loads financial documents (PDF, TXT, HTML)
- Runs LlamaIndex to extract sentiment or relevance
- Outputs structured results for use in BackTrader or analysis

## 📂 Files

- `llama_pipeline.py`: Main orchestration script (load → analyze → export)
- `utils.py`: Helper functions (normalization, summaries)
- `sentiment_output.csv`: Sample output file

## ▶️ Example Usage

```bash
python llama_pipeline.py --input data/news_docs/earnings_call.txt --output outputs/sentiment_output.csv

