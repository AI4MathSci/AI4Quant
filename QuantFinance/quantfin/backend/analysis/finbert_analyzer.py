from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FinBERTAnalyzer:
    """
    Finance-specific sentiment analysis using FinBERT model.
    Provides high-quality sentiment analysis as fallback when OpenAI API is not available.
    """
    
    def __init__(self):
        """Initialize FinBERT model and tokenizer"""
        try:
            logger.info("Initializing FinBERT sentiment analyzer...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()  # Set to evaluation mode
            logger.info("FinBERT sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT: {e}")
            raise
    
    def analyze_sentiment(self, text: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text using FinBERT.
        
        Args:
            text: Financial text to analyze
            confidence_threshold: Minimum confidence required to enable the result
            
        Returns:
            Dictionary with sentiment analysis results matching existing format
        """
        try:
            if not text or not text.strip():
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,  # Zero confidence for empty input
                    "enabled": False,    # Always disabled
                    "source": "finbert_fallback",
                    "reasoning": "Empty text provided",
                    "model": "ProsusAI/finbert"
                }
            
            # Tokenize and truncate if needed
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map predictions to sentiment
            sentiment_map = {0: "bearish", 1: "neutral", 2: "bullish"}
            sentiment_idx = predictions.argmax().item()
            confidence = predictions[0][sentiment_idx].item()
            
            # Calculate sentiment score (positive for bullish, negative for bearish)
            if sentiment_map[sentiment_idx] == "bullish":
                score = confidence
            elif sentiment_map[sentiment_idx] == "bearish":
                score = -confidence
            else:
                score = 0.0
            
            return {
                "sentiment": sentiment_map[sentiment_idx],
                "score": score,
                "confidence": confidence,
                "enabled": confidence >= confidence_threshold,
                "source": "finbert",
                "reasoning": f"FinBERT analysis: {sentiment_map[sentiment_idx]} with {confidence:.3f} confidence",
                "model": "ProsusAI/finbert"
            }
            
        except Exception as e:
            logger.error(f"FinBERT sentiment analysis error: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,  # Zero confidence for failures
                "enabled": False,    # Always disabled
                "source": "finbert_fallback",
                "reasoning": f"Analysis failed: {str(e)}",
                "model": "ProsusAI/finbert"
            }    