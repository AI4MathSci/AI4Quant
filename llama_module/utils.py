# llama_module/utils.py

import numpy as np
import pandas as pd

def normalize_scores(scores, method="zscore"):
    """
    Normalize sentiment or relevance scores.
    
    Parameters:
        scores (list or np.array): Raw scores
        method (str): Normalization method ("zscore", "minmax")

    Returns:
        np.array: Normalized scores
    """
    scores = np.array(scores)
    
    if method == "zscore":
        return (scores - scores.mean()) / scores.std()
    elif method == "minmax":
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        raise ValueError("Unsupported normalization method.")

def summarize_df(df, n=5):
    """
    Quick summary of top-scoring rows in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'score' or similar column
        n (int): Number of top entries to return
    
    Returns:
        pd.DataFrame: Top n rows
    """
    if 'score' not in df.columns:
        raise ValueError("Expected column 'score' in DataFrame.")
    
    return df.sort_values("score", ascending=False).head(n)

