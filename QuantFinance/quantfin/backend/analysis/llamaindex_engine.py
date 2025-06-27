from typing import Dict, Any, List, Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import SimpleWebPageReader
import logging
import os
import aiohttp
import json
import re
import feedparser
from datetime import datetime, timedelta
import pandas as pd

from quantfin.backend.config.config import config

logger = logging.getLogger(__name__)

class QuantFinLlamaEngine:
    """Optimized LlamaIndex-powered financial analysis engine"""
    
    def __init__(self):
        if not config.has_openai_key:
            raise ValueError("OpenAI API key required for LlamaIndex")
        
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.openai_model,
            temperature=0.1
        )
        Settings.embed_model = OpenAIEmbedding(
            api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        self.index = None
        self.query_engine = None
        self.document_count = 0
    
    async def build_financial_knowledge_base(self, symbol: str, scope: str = "news", days_back: int = 30, analysis_date: str = None) -> None:
        """Build knowledge base based on analysis scope with date-specific context"""
        documents = []
        
        try:
            if analysis_date:
                logger.info(f"Building knowledge base for {symbol} with scope '{scope}' as of {analysis_date} (last {days_back} days)")
            else:
                logger.error("analysis_date is required for temporal sentiment analysis")
                raise ValueError("analysis_date parameter is required for temporal sentiment analysis")
            
            if scope in ["news", "comprehensive"]:
                # Try Alpha Vantage historical news first
                alpha_news = await self._get_alpha_vantage_news(symbol, analysis_date, days_back)
                if alpha_news:
                    documents.extend(alpha_news)
                    logger.info(f"Added {len(alpha_news)} Alpha Vantage news articles")
                
                # Get financial news from Yahoo RSS with date context
                news_docs = await self._get_financial_news(symbol, days_back, analysis_date)
                documents.extend(news_docs)
                logger.info(f"Added {len(news_docs)} news documents")
                
                # Only use SEC 8-K as backup if no news found AND scope is "news" only
                if len(news_docs) == 0 and scope == "news":
                    logger.info("No news found, fetching SEC 8-K filings as news substitute")
                    sec_news_docs = await self._get_sec_filings(symbol, analysis_date)
                    # Filter for 8-K filings only (current reports = news-like)
                    filtered_docs = [doc for doc in sec_news_docs if '8-K' in doc.metadata.get('form_type', '')]
                    documents.extend(filtered_docs)
                    logger.info(f"Added {len(filtered_docs)} SEC 8-K filings as news substitute")
            
            if scope in ["filings", "comprehensive"]:
                # Include SEC filings for comprehensive analysis
                sec_docs = await self._get_sec_filings(symbol, analysis_date)
                documents.extend(sec_docs)
                logger.info(f"Added {len(sec_docs)} SEC filing documents")
            
            if scope == "comprehensive":
                # Include earnings transcripts for comprehensive analysis
                earnings_docs = await self._get_earnings_transcripts(symbol)
                documents.extend(earnings_docs)
                logger.info(f"Added {len(earnings_docs)} earnings documents")
            
            if documents:
                logger.info(f"Creating vector index from {len(documents)} documents...")
                self.index = VectorStoreIndex.from_documents(documents)
                self.query_engine = self.index.as_query_engine()
                self.document_count = len(documents)
                logger.info(f"Knowledge base built successfully with {len(documents)} total documents for {symbol}")
            else:
                logger.warning(f"No documents found for {symbol} with scope {scope}")
                self.index = None
                self.query_engine = None
                self.document_count = 0
                
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
            self.index = None
            self.query_engine = None
            self.document_count = 0
            raise
    
    async def get_sentiment_data(self, symbol: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis"""
        if not self.query_engine:
            return self._fallback_sentiment("No knowledge base available")
        
        try:
            # Construct intelligent sentiment query
            sentiment_query = f"""
            Analyze the overall market sentiment and outlook for {symbol} based on available information.
            
            Please provide:
            1. Overall sentiment (bullish/bearish/neutral)
            2. Confidence level (0-1 scale)
            3. Key supporting factors
            4. Main concerns or risks
            5. Short-term vs long-term outlook
            
            Focus on recent developments, market conditions, and fundamental factors.
            Be specific and cite key information from the documents.
            """
            
            logger.info(f"Querying LlamaIndex for sentiment analysis of {symbol}")
            response = self.query_engine.query(sentiment_query)
            
            # Parse the response into structured sentiment
            sentiment_analysis = await self._parse_llm_sentiment(str(response))
            
            # Apply confidence threshold
            if sentiment_analysis["confidence"] < confidence_threshold:
                logger.info(f"Sentiment confidence {sentiment_analysis['confidence']} below threshold {confidence_threshold}, returning neutral")
                sentiment_analysis["sentiment"] = "neutral"
                sentiment_analysis["score"] = 0.0
            
            return {
                "sentiment": sentiment_analysis["sentiment"],
                "score": sentiment_analysis["score"],
                "confidence": sentiment_analysis["confidence"],
                "enabled": True,
                "reasoning": sentiment_analysis.get("reasoning", ""),
                "key_factors": sentiment_analysis.get("key_factors", []),
                "concerns": sentiment_analysis.get("concerns", []),
                "outlook": sentiment_analysis.get("outlook", "neutral"),
                "source_count": self.document_count
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_sentiment(str(e))
    
    async def get_risk_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive risk analysis"""
        if not self.query_engine:
            return {"risks": [], "overall_risk": "unknown", "error": "No knowledge base"}
        
        risk_query = f"""
        Identify and analyze key risks for {symbol} including:
        1. Company-specific risks (financial, operational, competitive)
        2. Industry and sector risks
        3. Market and economic risks
        4. Regulatory and policy risks
        
        For each risk, provide severity level (low/medium/high/critical) and likelihood.
        Focus on recent developments and concrete concerns mentioned in the documents.
        """
        
        try:
            logger.info(f"Performing risk analysis for {symbol}")
            response = self.query_engine.query(risk_query)
            return await self._parse_risk_analysis(str(response))
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {"risks": [], "overall_risk": "unknown", "error": str(e)}
    
    async def get_catalyst_analysis(self, symbol: str) -> Dict[str, Any]:
        """Identify potential price catalysts"""
        if not self.query_engine:
            return {"catalysts": [], "error": "No knowledge base"}
        
        catalyst_query = f"""
        Identify potential positive and negative catalysts for {symbol} including:
        1. Upcoming events (earnings, product launches, regulatory decisions)
        2. Market trends and industry developments
        3. Technical factors and chart patterns
        4. Fundamental changes or announcements
        
        Classify each catalyst by:
        - Impact level (low/medium/high)
        - Probability (low/medium/high)
        - Timeframe (short-term/medium-term/long-term)
        
        Focus on specific events and developments mentioned in the documents.
        """
        
        try:
            logger.info(f"Performing catalyst analysis for {symbol}")
            response = self.query_engine.query(catalyst_query)
            return await self._parse_catalyst_analysis(str(response))
        except Exception as e:
            logger.error(f"Catalyst analysis error: {e}")
            return {"catalysts": [], "error": str(e)}
    
    def _fallback_sentiment(self, error_msg: str) -> Dict[str, Any]:
        """Fallback sentiment when analysis fails"""
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "enabled": False,
            "reasoning": f"Analysis unavailable: {error_msg}",
            "error": error_msg,
            "source_count": 0
        }
    
    # Parsing methods (enhanced but still simplified)
    async def _parse_llm_sentiment(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured sentiment"""
        response_lower = response.lower()
        
        # First, check for sentiment statements from the LLM
        sentiment = None
        confidence = 0.7  # Default confidence for statements
        
        # Look for overall sentiment declarations
        if "overall sentiment: bullish" in response_lower or "sentiment: bullish" in response_lower:
            sentiment = "bullish"
        elif "overall sentiment: bearish" in response_lower or "sentiment: bearish" in response_lower:
            sentiment = "bearish"
        elif "overall sentiment: neutral" in response_lower or "sentiment: neutral" in response_lower:
            sentiment = "neutral"
        
        # Look for confidence level in the response
        confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_lower)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # If confidence is given as percentage (>1), convert to decimal
                if confidence > 1:
                    confidence = confidence / 100
            except:
                pass
        
        # If we found sentiment, calculate score
        if sentiment:
            if sentiment == "bullish":
                score = min(0.8, 0.3 + confidence * 0.5)
            elif sentiment == "bearish":
                score = max(-0.8, -0.3 - confidence * 0.5)
            else:  # neutral
                score = 0.0
        else:
            # Fallback to keyword-based sentiment detection
            bullish_indicators = ["bullish", "positive", "optimistic", "strong outlook", "growth potential", "buy", "uptrend"]
            bearish_indicators = ["bearish", "negative", "pessimistic", "weak outlook", "declining", "sell", "downtrend"]
            
            bullish_score = sum(1 for indicator in bullish_indicators if indicator in response_lower)
            bearish_score = sum(1 for indicator in bearish_indicators if indicator in response_lower)
            
            if bullish_score > bearish_score:
                sentiment = "bullish"
                score = min(0.8, 0.3 + (bullish_score - bearish_score) * 0.1)
            elif bearish_score > bullish_score:
                sentiment = "bearish" 
                score = max(-0.8, -0.3 - (bearish_score - bullish_score) * 0.1)
            else:
                sentiment = "neutral"
                score = 0.0
            
            # Confidence based on response length and specificity
            confidence = min(0.9, 0.4 + len(response.split()) / 1000)
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "reasoning": response[:400],  # First 400 chars
            "key_factors": [],
            "concerns": [],
            "outlook": sentiment
        }
    
    async def _parse_risk_analysis(self, response: str) -> Dict[str, Any]:
        """Parse risk analysis response"""
        return {
            "risks": [],
            "overall_risk": "medium",
            "analysis": response[:300]
        }
    
    async def _parse_catalyst_analysis(self, response: str) -> Dict[str, Any]:
        """Parse catalyst analysis response"""
        return {
            "catalysts": [],
            "analysis": response[:300]
        }
    
    # Data source methods (stub implementations for now)
    async def _get_financial_news(self, symbol: str, days_back: int, analysis_date: str) -> List:
        """Get financial news documents from Yahoo Finance RSS with date-specific filtering"""
        try:
            logger.info(f"Fetching Yahoo Finance RSS news for {symbol} as of {analysis_date} (last {days_back} days)")
            
            # Yahoo Finance RSS feed for the symbol
            rss_url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
            
            # Add proper browser headers to avoid rate limiting
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(rss_url, headers=headers, timeout=15) as response:
                    logger.info(f"Yahoo RSS response status: {response.status} for {symbol}")
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse RSS feed
                        feed = feedparser.parse(content)
                        documents = []
                        
                        # Calculate date range based on analysis_date
                        analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
                        start_date = analysis_dt - timedelta(days=days_back)
                        end_date = analysis_dt
                        
                        logger.info(f"=== TEMPORAL FILTERING DEBUG ===")
                        logger.info(f"Analysis Date: {analysis_date}")
                        logger.info(f"Days Back: {days_back}")
                        logger.info(f"Start Date (inclusive): {start_date.strftime('%Y-%m-%d')}")
                        logger.info(f"End Date (inclusive): {end_date.strftime('%Y-%m-%d')}")
                        logger.info(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                        logger.info(f"Total Days in Range: {(end_date - start_date).days + 1}")
                        logger.info(f"=== END TEMPORAL FILTERING DEBUG ===")
                        
                        logger.info(f"Found {len(feed.entries)} total RSS entries for {symbol}")
                        
                        for entry in feed.entries[:20]:  # Limit to 20 most recent articles
                            try:
                                # Parse published date
                                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                    pub_date = datetime(*entry.published_parsed[:6])
                                elif hasattr(entry, 'published'):
                                    # Fallback: try to parse published string
                                    pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=None)
                                else:
                                    # If no date, exclude it (don't assume recent)
                                    logger.warning(f"Article '{getattr(entry, 'title', 'No title')[:50]}...' has no publish date, excluding")
                                    continue
                                
                                # Filter by date range - only include news available up to analysis_date
                                if pub_date < start_date or pub_date > end_date:
                                    continue
                                
                                # Extract title and summary
                                title = getattr(entry, 'title', 'No title')
                                summary = getattr(entry, 'summary', '')
                                link = getattr(entry, 'link', '')
                                
                                # Create document content
                                content_text = f"Title: {title}\n\nSummary: {summary}"
                                
                                # Create LlamaIndex document
                                from llama_index.core import Document
                                doc = Document(
                                    text=content_text,
                                    metadata={
                                        "source": "Yahoo Finance RSS",
                                        "symbol": symbol,
                                        "published": entry.published if hasattr(entry, 'published') else str(pub_date),
                                        "url": link,
                                        "news_type": "financial_news",
                                        "analysis_date": analysis_date,
                                        "date_range_start": start_date.strftime('%Y-%m-%d'),
                                        "date_range_end": end_date.strftime('%Y-%m-%d'),
                                        "days_back": days_back
                                    }
                                )
                                documents.append(doc)
                                
                            except Exception as e:
                                logger.warning(f"Error processing RSS entry: {e}")
                                continue
                        
                        # Summary statistics
                        logger.info(f"=== NEWS FILTERING SUMMARY ===")
                        logger.info(f"Total RSS entries found: {len(feed.entries)}")
                        logger.info(f"Entries checked: {len(feed.entries[:20])}")
                        logger.info(f"Entries within date range: {len(documents)}")
                        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                        logger.info(f"Analysis date: {analysis_date}")
                        logger.info(f"=== END NEWS FILTERING SUMMARY ===")
                        
                        logger.info(f"Retrieved {len(documents)} relevant news articles for {symbol} (within {days_back} days of {analysis_date})")
                        return documents
                    else:
                        logger.warning(f"Yahoo RSS returned status {response.status} for {symbol}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance RSS news for {symbol}: {e}")
            return []
    
    async def _get_sec_filings(self, symbol: str, analysis_date: str = None) -> List:
        """Get SEC filing documents for the company with date filtering"""
        try:
            logger.info(f"Fetching SEC filings for {symbol} as of {analysis_date}")
            
            # Get company CIK (Central Index Key) from ticker
            cik = await self._get_company_cik(symbol)
            if not cik:
                logger.warning(f"Could not find CIK for symbol {symbol}")
                return []
            
            logger.info(f"Found CIK {cik} for {symbol}")
            
            # Get recent filings (10-K, 10-Q, 8-K) with date filtering
            filings_data = await self._fetch_recent_filings(cik, analysis_date)
            
            # Convert filings to LlamaIndex documents
            documents = []
            for filing in filings_data:
                logger.info(f"Processing filing: {filing}")
                doc_content = await self._fetch_filing_content(filing, cik)
                if doc_content:
                    from llama_index.core import Document
                    doc = Document(
                        text=doc_content,
                        metadata={
                            "source": f"SEC {filing['form']} Filing",
                            "symbol": symbol,
                            "filing_date": filing['filingDate'],
                            "form_type": filing['form'],
                            "analysis_date": analysis_date
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} SEC filings for {symbol}")
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching SEC filings for {symbol}: {e}")
            return []
    
    async def _get_earnings_transcripts(self, symbol: str) -> List:
        """Get earnings call transcripts"""
        logger.info(f"Fetching earnings transcripts for {symbol}")
        # Stub implementation - would integrate with earnings APIs
        return []
    
    async def _get_company_cik(self, symbol: str) -> str:
        """Get company CIK from ticker symbol"""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {
                'User-Agent': 'QuantFin Research Tool contact@quantfin.com'  # Required by SEC
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Search for company by ticker
                        for entry in data.values():
                            if entry.get('ticker', '').upper() == symbol.upper():
                                return str(entry['cik_str']).zfill(10)  # Pad with zeros
            return None
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")
            return None

    async def _fetch_recent_filings(self, cik: str, analysis_date: str = None) -> List[Dict]:
        """Fetch recent SEC filings for a company"""
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = {
                'User-Agent': 'QuantFin Research Tool contact@quantfin.com'
            }
            
            logger.info(f"Fetching from URL: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    logger.info(f"SEC API response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"SEC API response keys: {list(data.keys())}")
                        
                        filings = data.get('filings', {}).get('recent', {})
                        logger.info(f"Filings keys: {list(filings.keys())}")
                        
                        if 'form' in filings:
                            total_forms = len(filings['form'])
                            logger.info(f"Total forms found: {total_forms}")
                            if total_forms > 0:
                                logger.info(f"Sample forms: {filings['form'][:5]}")
                        
                        # Filter for important filing types
                        important_forms = ['10-K', '10-Q', '8-K', '10-K/A', '10-Q/A']
                        recent_filings = []
                        
                        for i, form in enumerate(filings.get('form', [])):
                            if form in important_forms:
                                recent_filings.append({
                                    'form': form,
                                    'filingDate': filings['filingDate'][i],
                                    'accessionNumber': filings['accessionNumber'][i],
                                    'primaryDocument': filings['primaryDocument'][i]
                                })
                        
                        logger.info(f"Filtered filings count: {len(recent_filings)}")
                        
                        # Sort by date (most recent first)
                        recent_filings.sort(key=lambda x: x['filingDate'], reverse=True)
                        
                        # Apply date filtering if analysis_date provided
                        if analysis_date:
                            analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
                            # Filter filings up to analysis_date
                            relevant_filings = [f for f in recent_filings if datetime.strptime(f['filingDate'], '%Y-%m-%d') <= analysis_dt]
                            # Sort by proximity to analysis_date (closest first)
                            relevant_filings.sort(key=lambda x: abs((datetime.strptime(x['filingDate'], '%Y-%m-%d') - analysis_dt).days))
                            
                            logger.info(f"Found {len(relevant_filings)} filings up to {analysis_date} out of {len(recent_filings)} total")
                            return relevant_filings[:5]  # Return 5 closest to analysis_date
                        else:
                            return recent_filings[:10]  # Return 10 most recent
                    else:
                        logger.error(f"SEC API returned status {response.status}")
            return []
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []

    async def _fetch_filing_content(self, filing: Dict, cik: str) -> str:
        """Fetch the actual content of a SEC filing"""
        try:
            # Construct URL for the filing document
            # SEC EDGAR URL format: /Archives/edgar/data/CIK/ACCESSION_NO_DASHES/FILENAME
            acc_num_no_dashes = filing['accessionNumber'].replace('-', '')
            cik_short = cik.lstrip('0')  # Remove leading zeros
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_short}/{acc_num_no_dashes}/{filing['primaryDocument']}"
            
            logger.info(f"Fetching filing content from: {url}")
            
            headers = {
                'User-Agent': 'QuantFin Research Tool contact@quantfin.com'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    logger.info(f"Filing content response status: {response.status}")
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Raw content length: {len(content)}")
                        # Extract text from HTML/XBRL (simplified)
                        clean_text = self._clean_sec_filing(content)
                        logger.info(f"Clean content length: {len(clean_text)}")
                        if len(clean_text) > 0:
                            logger.info(f"Content preview: {clean_text[:200]}...")
                        return clean_text[:10000]  # Limit size for LlamaIndex
                    else:
                        logger.warning(f"Failed to fetch filing content, status: {response.status}")
            return ""
        except Exception as e:
            logger.warning(f"Could not fetch filing content for {filing.get('form', 'unknown')}: {e}")
            return ""

    def _clean_sec_filing(self, html_content: str) -> str:
        """Clean SEC filing HTML to extract readable text"""
        try:
            # Remove HTML tags
            clean = re.sub(r'<[^>]+>', ' ', html_content)
            # Remove extra whitespace
            clean = re.sub(r'\s+', ' ', clean)
            # Remove common HTML entities
            clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            # Extract key sections (simplified)
            return clean.strip()
        except Exception as e:
            logger.warning(f"Error cleaning SEC filing content: {e}")
            return html_content  # Return raw content if cleaning fails

    async def _get_alpha_vantage_news(self, symbol: str, analysis_date: str, days_back: int = 30) -> List:
        """Get historical financial news from Alpha Vantage API"""
        # Initialize Alpha Vantage config if not already done
        if not hasattr(self, 'alpha_vantage_api_key'):
            self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        
        if not self.alpha_vantage_api_key:
            logger.warning("Alpha Vantage API key not configured, skipping historical news")
            return []
            
        try:
            # Calculate date range
            analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
            start_date = analysis_dt - timedelta(days=days_back)
            
            logger.info(f"Fetching Alpha Vantage news for {symbol} from {start_date.strftime('%Y-%m-%d')} to {analysis_date}")
            
            # Alpha Vantage News API call with correct parameters
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_api_key,
                'time_from': start_date.strftime('%Y%m%dT0000'),
                'time_to': analysis_dt.strftime('%Y%m%dT2359'),
                'limit': 50,
                'sort': 'LATEST'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.alpha_vantage_base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if 'Error Message' in data:
                            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                            return []
                        
                        if 'Information' in data:
                            logger.warning(f"Alpha Vantage API info: {data['Information']}")
                            return []
                        
                        if 'feed' in data and data['feed']:
                            logger.info(f"Alpha Vantage: Found {len(data['feed'])} news articles for {symbol}")
                            
                            documents = []
                            for article in data['feed']:
                                from llama_index.core import Document
                                
                                # Create document text from title and summary
                                doc_text = f"Title: {article.get('title', '')}\n\nSummary: {article.get('summary', '')}"
                                
                                doc = Document(
                                    text=doc_text,
                                    metadata={
                                        "source": "Alpha Vantage News",
                                        "symbol": symbol,
                                        "published": article.get('time_published', ''),
                                        "url": article.get('url', ''),
                                        "news_type": "financial_news",
                                        "analysis_date": analysis_date,
                                        "sentiment_score": article.get('overall_sentiment_score', 0),
                                        "sentiment_label": article.get('overall_sentiment_label', 'neutral'),
                                        "authors": article.get('authors', []),
                                        "topics": [topic.get('topic', '') for topic in article.get('topics', [])]
                                    }
                                )
                                documents.append(doc)
                            
                            return documents
                        else:
                            logger.info(f"Alpha Vantage: No news articles found for {symbol}")
                            return []
                    else:
                        logger.warning(f"Alpha Vantage API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return [] 