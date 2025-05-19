from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import logging
import os
import asyncpraw
from quantfin.backend.config.config import config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = timedelta(hours=1)
        # Initialize Reddit client
        if not all([config.reddit_client_id, config.reddit_client_secret, config.reddit_user_agent]):
            logger.error("Missing Reddit credentials. Please check your .env file.")
            raise ValueError("Missing Reddit credentials. Please check your .env file.")
            
        self.reddit = asyncpraw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent
        )

    async def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyzes news sentiment for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to look back
            
        Returns:
            Dict containing sentiment scores and analysis
        """
        try:
            # Get news from Yahoo Finance
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return {
                    "sentiment_score": 0,
                    "sentiment_magnitude": 0,
                    "message": "No news found"
                }

            # Analyze sentiment for each news item
            sentiments = []
            for item in news:
            #    if 'title' in item:
            #        text = item['title']
            #        if 'summary' in item:
            #            text += " " + item['summary']
                if 'content' in item and 'title' in item['content']:
                    text = item['content']['title']
                    if 'summary' in item['content']:
                        text += " " + item['content']['summary']
                    
                    # Calculate sentiment using TextBlob
                    analysis = TextBlob(text)
                    sentiments.append(analysis.sentiment.polarity)

            if not sentiments:
                return {
                    "sentiment_score": 0,
                    "sentiment_magnitude": 0,
                    "message": "No valid news content found"
                }

            # Calculate aggregate sentiment metrics
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_magnitude = sum(abs(s) for s in sentiments) / len(sentiments)
            return {
                "sentiment_score": round(avg_sentiment, 3),
                "sentiment_magnitude": round(sentiment_magnitude, 3),
                "message": "News sentiment analysis successful"
            }

        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {e}")
            return {
                "sentiment_score": 0,
                "sentiment_magnitude": 0,
                "message": f"Error in sentiment analysis: {str(e)}"
            }

    async def get_social_media_sentiment(self, symbol: str, platform: str = "reddit") -> Dict[str, Any]:
        """
        Analyzes social media sentiment for a given stock symbol using Reddit.
        
        Args:
            symbol (str): Stock symbol
            platform (str): Social media platform (defaults to "reddit")
            
        Returns:
            Dict containing sentiment scores and analysis
        """
        if platform.lower() != "reddit":
            return {
                "sentiment_score": 0,
                "sentiment_magnitude": 0,
                "message": f"Platform {platform} not supported"
            }
            
        try:
            # Search for posts about the stock
            search_query = f"${symbol} OR {symbol}"
            subreddits = ['stocks', 'investing', 'wallstreetbets', 'stockmarket']
            
            sentiments = []
            for subreddit_name in subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.search(search_query, limit=50, time_filter='week'):
                        # Analyze post title
                        title_sentiment = TextBlob(post.title).sentiment.polarity
                        sentiments.append(title_sentiment)
                        
                        # Analyze post text if available
                        if post.selftext:
                            text_sentiment = TextBlob(post.selftext).sentiment.polarity
                            sentiments.append(text_sentiment)
                        
                        # Analyze top comments
                        try:
                            # Load comments first
                            await post.load()
                            # Get comments list
                            comments = await post.comments.list()
                            # Process top 10 comments
                            for comment in list(comments)[:10]:
                                if hasattr(comment, 'body'):
                                    comment_sentiment = TextBlob(comment.body).sentiment.polarity
                                    sentiments.append(comment_sentiment)
                        except Exception as comment_error:
                            logger.error(f"Error processing comments for post {post.id}: {str(comment_error)}")
                            continue
                            
                except Exception as subreddit_error:
                    logger.error(f"Error accessing subreddit {subreddit_name}: {str(subreddit_error)}")
                    continue
            
            if not sentiments:
                return {
                    "sentiment_score": 0,
                    "sentiment_magnitude": 0,
                    "message": "No Reddit content found"
                }
                
            # Calculate aggregate sentiment metrics
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_magnitude = sum(abs(s) for s in sentiments) / len(sentiments)
            
            return {
                "sentiment_score": round(avg_sentiment, 3),
                "sentiment_magnitude": round(sentiment_magnitude, 3),
                "message": "Reddit sentiment analysis successful"
            }
            
        except Exception as e:
            logger.error(f"Error in Reddit sentiment analysis for {symbol}: {str(e)}", exc_info=True)
            return {
                "sentiment_score": 0,
                "sentiment_magnitude": 0,
                "message": f"Error in Reddit sentiment analysis: {str(e)}"
            }

    async def get_combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Combines sentiment from multiple sources.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict containing combined sentiment analysis
        """
        news_sentiment = await self.get_news_sentiment(symbol)
        social_sentiment = await self.get_social_media_sentiment(symbol)

        # Combine sentiments (simple average for now)
        combined_score = (news_sentiment["sentiment_score"] + social_sentiment["sentiment_score"]) / 2
        combined_magnitude = (news_sentiment["sentiment_magnitude"] + social_sentiment["sentiment_magnitude"]) / 2
        
        return {
            "combined_sentiment_score": round(combined_score, 3),
            "combined_sentiment_magnitude": round(combined_magnitude, 3),
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "message": "Combined sentiment analysis successful"
        } 