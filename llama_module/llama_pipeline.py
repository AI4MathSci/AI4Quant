# llama_pipeline.py

"""
This module retrieves documents using LlamaIndex, analyzes them with an LLM,
and outputs sentiment scores to be used in a quant trading strategy.
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data/news_docs"  # Folder with your news or analysis documents
MODEL_NAME = "gpt-4"         # or "gpt-3.5-turbo"
TOPIC_QUERY = "What is the current sentiment about Apple stock (AAPL)?"

# --- INITIALIZATION ---
llm = OpenAI(model=MODEL_NAME)
documents = SimpleDirectoryReader(DATA_DIR).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)

# --- RUN QUERY ---
response = query_engine.query(TOPIC_QUERY)
summary_text = str(response)

print("\n--- Summary Output ---\n")
print(summary_text)

# --- MOCKED SENTIMENT SCORE ---
# You can implement your own logic to parse the summary_text.
sentiment_score = 0.65  # Example: confidence-scaled bullish score (0 to 1)

# --- EXPORT TO CSV FOR STRATEGY ---
output_df = pd.DataFrame([{
    "timestamp": datetime.utcnow().isoformat(),
    "query": TOPIC_QUERY,
    "summary": summary_text,
    "sentiment_score": sentiment_score
}])

os.makedirs("outputs", exist_ok=True)
output_df.to_csv("outputs/sentiment_output.csv", index=False)

print("\nSaved to outputs/sentiment_output.csv")

