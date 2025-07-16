import logging
from typing import Dict, Any, List
from quantfin.backend.config.config import SENTIMENT_CLASSIFICATION_THRESHOLD

logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """Keyword-based sentiment analyzer as ultimate fallback"""
    
    def __init__(self):
        # Financial sentiment keywords (reuse existing from llamaindex_engine.py)
        self.bullish_keywords = [
            "bullish", "positive", "growth", "profit", "gain", "rise", "increase", 
            "strong", "outperform", "beat", "exceed", "higher", "up", "buy", "hold",
            "recommend", "target", "upgrade", "positive outlook", "strong fundamentals"
        ]
        self.bearish_keywords = [
            "bearish", "negative", "decline", "loss", "fall", "decrease", "weak",
            "underperform", "miss", "lower", "down", "sell", "downgrade", "risk",
            "concern", "warning", "negative outlook", "weak fundamentals"
        ]
    
    def analyze_sentiment(self, text: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze sentiment using keyword matching
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence required to enable the result
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Case 1: Empty text
            if not text or not text.strip():
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                    "enabled": False,  # Always disabled
                    "source": "keyword_analysis",
                    "reasoning": "Empty text provided",
                    "error": None,
                    "model": "keyword_matcher",
                    "keyword_count": 0
                }
            
            # Case 2: Normal analysis
            text_lower = text.lower()
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
            
            total_keywords = bullish_count + bearish_count
            
            # Case 3: No keywords found
            if total_keywords == 0:
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                    "enabled": False,  # Always disabled
                    "source": "keyword_analysis",
                    "reasoning": "No financial keywords detected in text",
                    "error": None,
                    "model": "keyword_matcher",
                    "keyword_count": 0
                }
            
            # Case 4: Valid analysis with keywords
            score = (bullish_count - bearish_count) / total_keywords
            confidence = min(0.8, total_keywords * 0.1)  # Cap confidence at 0.8
            
            if score > SENTIMENT_CLASSIFICATION_THRESHOLD:
                sentiment = "bullish"
            elif score < -SENTIMENT_CLASSIFICATION_THRESHOLD:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "score": score,
                "confidence": confidence,
                "enabled": confidence >= confidence_threshold,  # Dynamic based on actual confidence
                "source": "keyword_analysis",
                "reasoning": f"Found {bullish_count} bullish and {bearish_count} bearish keywords. Score: {score:.3f}",
                "error": None,
                "model": "keyword_matcher",
                "keyword_count": total_keywords
            }
            
        # Case 5: Exception handling
        except Exception as e:
            logger.error(f"Keyword analysis error: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "enabled": False,  # Always disabled for errors
                "source": "keyword_analysis",
                "reasoning": f"Analysis failed: {str(e)}",
                "error": str(e),
                "model": "keyword_matcher",
                "keyword_count": 0
            }