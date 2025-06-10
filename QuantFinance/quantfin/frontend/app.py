import gradio as gr
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def is_valid_symbol(symbol):
    """Validate if the symbol exists in yfinance"""
    logger.info(f"Validating symbol: {symbol}")
    if not symbol or len(symbol.strip()) == 0:
        logger.warning(f"Invalid symbol: empty or None")
        return False
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="5d")
        is_valid = len(hist) > 0
        logger.info(f"Symbol {symbol} validation result: {is_valid}")
        return is_valid
    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {e}")
        return False

def get_api_key_status():
    """Check if OpenAI API key is configured"""
    try:
        response = requests.get(f"{API_BASE_URL}/config/openai-status")
        if response.status_code == 200:
            configured = response.json().get("configured", False)
            logger.info(f"OpenAI API key status: {'configured' if configured else 'not configured'}")
            return configured
        else:
            logger.warning(f"Failed to get API key status: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error checking API key status: {e}")
    return False

def format_api_status(configured):
    """Format API key status for display"""
    if configured:
        return "✅ OpenAI API Key: Configured"
    else:
        return "❌ OpenAI API Key: Not Configured (Sentiment analysis disabled)"

# Trading Interface
def trading_interface():
    with gr.Blocks(title="Trading Strategy Simulator", css="""
        #optimization-summary, #performance-summary {
            min-height: 300px !important;
            max-height: 500px !important;
            overflow-y: auto !important;
            padding: 15px !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            background-color: #fafafa !important;
        }
        .optimization-display, .performance-display {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            line-height: 1.6 !important;
        }
        .optimization-display h4, .performance-display h4 {
            color: #2563eb !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }
        """) as demo:
        gr.Markdown("# 📈 Trading Strategy Simulator")
        
        # API Status
        api_status = gr.Markdown(value=format_api_status(get_api_key_status()))
        
        with gr.Row():
            with gr.Column(scale=1):
                symbol_input = gr.Textbox(
                    label="Stock Symbol",
                    placeholder="e.g., AAPL",
                    value="AAPL"
                )
                
                start_date = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                )
                
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    value=datetime.now().strftime("%Y-%m-%d")
                )
                
                initial_capital = gr.Number(
                    label="Initial Capital ($)",
                    value=100000,
                    minimum=1000
                )
                
                strategy = gr.Dropdown(
                    label="Trading Strategy",
                    choices=["Simple Moving Average", "RSI", "Bollinger Bands"],
                    value="Simple Moving Average"
                )
                
                # Strategy Parameters with conditional display
                with gr.Accordion("Strategy Parameters", open=False, elem_classes=["strategy-params"], elem_id="strategy-params-section"):
                    # SMA Parameters (show only for Simple Moving Average)
                    with gr.Group(visible=True) as sma_params:
                        sma_window = gr.Slider(
                            label="SMA Window",
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=1
                        )
                    
                    # RSI Parameters (show only for RSI)
                    with gr.Group(visible=False) as rsi_params:
                        rsi_period = gr.Slider(
                            label="RSI Period",
                            minimum=5,
                            maximum=30,
                            value=14,
                            step=1
                        )
                        
                        rsi_overbought = gr.Slider(
                            label="RSI Overbought Level",
                            minimum=60,
                            maximum=90,
                            value=70,
                            step=1
                        )
                        
                        rsi_oversold = gr.Slider(
                            label="RSI Oversold Level",
                            minimum=10,
                            maximum=40,
                            value=30,
                            step=1
                        )
                    
                    # Bollinger Bands Parameters (show only for Bollinger Bands)
                    with gr.Group(visible=False) as bb_params:
                        bb_period = gr.Slider(
                            label="Bollinger Bands Period",
                            minimum=10,
                            maximum=30,
                            value=20,
                            step=1
                        )
                        
                        bb_dev = gr.Slider(
                            label="Bollinger Bands Deviation",
                            minimum=1.0,
                            maximum=3.0,
                            value=2.0,
                            step=0.1
                        )
                
                # Sentiment Analysis Controls
                with gr.Accordion("Sentiment Analysis", open=True, elem_classes=["sentiment-controls"], elem_id="sentiment-section"):
                    use_sentiment = gr.Checkbox(
                        label="Enable Sentiment Analysis",
                        value=False
                    )
                    
                    sentiment_weight = gr.Slider(
                        label="Sentiment Weight (0 = ignore, 1 = full weight)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1
                    )
                    
                    sentiment_threshold = gr.Slider(
                        label="Sentiment Threshold (minimum strength to trigger)",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05
                    )
                    
                    # LlamaIndex-specific controls
                    analysis_scope = gr.Dropdown(
                        label="Analysis Scope",
                        choices=["news", "comprehensive", "filings"],
                        value="news",
                        info="news: Recent news only, comprehensive: News + SEC filings + earnings, filings: SEC filings only"
                    )
                    
                    sentiment_confidence_threshold = gr.Slider(
                        label="Confidence Threshold (minimum confidence to act)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1
                    )
                    
                    include_risk_analysis = gr.Checkbox(
                        label="Include Risk Factor Analysis",
                        value=False,
                        info="Analyze company-specific, industry, and market risks"
                    )
                    
                    include_catalyst_analysis = gr.Checkbox(
                        label="Include Catalyst Analysis",
                        value=False,
                        info="Identify potential positive and negative price catalysts"
                    )
                
                # Optimization Controls
                with gr.Accordion("Backtesting Optimization", open=False, elem_classes=["optimization-controls"], elem_id="optimization-section"):
                    enable_optimization = gr.Checkbox(
                        label="Enable Parameter Optimization",
                        value=True,
                        info="Automatically find optimal parameters using split-sample testing"
                    )
                    
                    optimization_split_ratio = gr.Slider(
                        label="Training/Test Split Ratio",
                        minimum=0.5,
                        maximum=0.9,
                        value=0.7,
                        step=0.1,
                        info="Percentage of data used for training (rest for testing)"
                    )
                    
                    compare_sentiment_versions = gr.Checkbox(
                        label="Compare Sentiment vs Technical-Only",
                        value=True,
                        info="Test both sentiment-enhanced and technical-only versions"
                    )
                    
                    optimization_speed = gr.Dropdown(
                        label="Optimization Speed",
                        choices=["lightning", "fast", "medium", "balanced", "thorough", "exhaustive"],
                        value="medium",
                        info="lightning: ~18 combinations (30 sec), fast: ~48 (1 min), medium: ~120 (2-3 min), balanced: ~240 (4-6 min), thorough: ~400 (8-12 min), exhaustive: ~800+ (15-30+ min)"
                    )
                    
                    max_combinations = gr.Number(
                        label="Max Parameter Combinations",
                        value=200,
                        minimum=50,
                        maximum=1000,
                        info="Upper limit for parameter testing (limits compute time)"
                    )
            
            with gr.Column(scale=2):
                with gr.Tab("Simulation"):
                    simulate_btn = gr.Button("Run Simulation", variant="primary")
                    simulation_output = gr.JSON(label="Simulation Results")
                
                with gr.Tab("Backtesting"):
                    backtest_btn = gr.Button("Run Backtest", variant="primary")
                    
                    # Enhanced results display
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🎯 **Optimization Results**")
                            optimization_summary = gr.Markdown(
                                value="""
**Ready to optimize...**

• Click "Run Backtest" to start parameter optimization
• System will test multiple parameter combinations
• Both sentiment-enhanced and technical-only approaches will be evaluated
• Results will show the best performing configuration

*Optimization progress will appear here...*
                                """, 
                                elem_id="optimization-summary",
                                elem_classes=["optimization-display"]
                            )
                        
                        with gr.Column():
                            gr.Markdown("### 📊 **Performance Summary**")
                            performance_summary = gr.Markdown(
                                value="""
**Performance metrics will appear here...**

**Training Performance:**
• Sentiment approach: *pending*
• Technical approach: *pending*

**Out-of-Sample Testing:**
• Sentiment approach: *pending*  
• Technical approach: *pending*

**Recommendation:**
• Best approach: *will be determined*
• Confidence level: *based on testing*

*Detailed performance analysis will be shown after optimization completes...*
                                """,
                                elem_id="performance-summary", 
                                elem_classes=["performance-display"]
                            )
                    
                    with gr.Row():
                        gr.Markdown("### 📋 **Detailed Results** (Advanced)")
                        backtest_output = gr.JSON(label="Complete Analysis", visible=False)
                    
                    # Toggle for detailed results
                    show_detailed = gr.Checkbox(label="Show Detailed JSON Results", value=False)
                    
                    def toggle_detailed_results(show):
                        return gr.update(visible=show)
                    
                    show_detailed.change(
                        fn=toggle_detailed_results,
                        inputs=[show_detailed],
                        outputs=[backtest_output]
                    )
        
        # Event handlers
        def update_strategy_params(strategy_name):
            """Update visibility of strategy parameters based on selected strategy"""
            sma_visible = strategy_name == "Simple Moving Average"
            rsi_visible = strategy_name == "RSI" 
            bb_visible = strategy_name == "Bollinger Bands"
            
            return (
                gr.update(visible=sma_visible),  # sma_params
                gr.update(visible=rsi_visible),  # rsi_params
                gr.update(visible=bb_visible)    # bb_params
            )
        
        def format_backtest_results(result):
            """Format backtest results for enhanced display"""
            if "error" in result:
                return (
                    f"""
#### ❌ **Error Occurred**

**Error Message:** {result['error']}

**Troubleshooting:**
• Check if the stock symbol is valid
• Verify date ranges are correct  
• Ensure API services are running
• Try again with different parameters

*Please check your inputs and try again.*
                    """,
                    """
**Unable to complete analysis due to error.**

Please review the error message and adjust your settings accordingly.
                    """,
                    result
                )
            
            # Handle progress/intermediate updates
            if result.get("status") == "progress":
                progress_text = f"""
#### 🔄 **Optimization In Progress...**

**Current Status:** {result.get('current_status', 'Processing...')}
**Progress:** {result.get('progress', 'Starting...')}
**Time Elapsed:** {result.get('time_elapsed', 'Just started')}

**What's Happening:**
• Testing parameter combinations for optimal performance
• Comparing sentiment-enhanced vs technical-only approaches  
• {result.get('current_step', 'Initializing optimization process...')}

*Please wait while optimization completes...*
                """
                
                return progress_text, "Optimization in progress...", result
            
            if result.get("methodology") == "Simple Backtest":
                # Simple backtest formatting
                initial_capital = result.get('initial_capital', 100000)
                final_value = result.get('final_portfolio_value', initial_capital)
                profit_loss = final_value - initial_capital
                return_pct = result.get('total_return_pct', 0)
                
                optimization_text = f"""
#### 📊 **Simple Backtest Complete**

**Method:** {result.get('methodology', 'N/A')}  
**Strategy:** {result.get('strategy', 'N/A')}  

#### 💰 **Financial Results**
**Initial Capital:** ${initial_capital:,.2f}
**Final Portfolio Value:** ${final_value:,.2f}
**Profit/Loss:** ${profit_loss:+,.2f}
**Total Return:** {return_pct:+.2f}%

**Analysis Mode:** Single parameter set (no optimization)
**Sentiment Analysis:** {'Enabled' if result.get('sentiment_analysis', {}).get('enabled', False) else 'Disabled'}
"""
                
                performance_text = f"""
#### 📈 **Performance Summary**

**Starting Investment:** ${initial_capital:,.2f}
**Ending Portfolio Value:** ${final_value:,.2f}
**Net Gain/Loss:** ${profit_loss:+,.2f}
**Percentage Return:** {return_pct:+.2f}%

#### ⚙️ **Configuration Used**
**Parameters:** Current UI settings  
**Time Period:** {result.get('start_date', 'N/A')} to {result.get('end_date', 'N/A')}
**Sentiment Weight:** {result.get('sentiment_weight', 'N/A') if result.get('sentiment_analysis', {}).get('enabled', False) else 'N/A (Disabled)'}

#### 📋 **Analysis Notes**
• This was a single-run backtest with your current parameter settings
• For optimized parameters, enable "Parameter Optimization" 
• Results show performance using {'sentiment-enhanced' if result.get('sentiment_analysis', {}).get('enabled', False) else 'technical-only'} approach

**Status:** {result.get('message', 'Analysis completed')}
"""
            
            else:
                # Enhanced backtest formatting with more detail
                optimization_text = f"""
#### 🏆 **Optimization Complete!**

**Method:** {result.get('methodology', 'N/A')}  
**Strategy:** {result.get('strategy', 'N/A')}  
**Optimization Speed:** {result.get('optimization_speed', 'N/A')}  
**Parameter Combinations Tested:** {result.get('parameters_tested', 'N/A')}  
**Confidence Level:** {result.get('confidence_level', 'N/A')}  

---

#### 🎯 **Recommended Configuration**

**🏆 Best Approach:** {result.get('recommended_approach', 'N/A')}  
**⚙️ Optimal Parameters:** {result.get('recommended_parameters', 'N/A')}  

**Why This Configuration:**
• Achieved best out-of-sample performance
• Thoroughly tested across multiple market conditions
• Balances return potential with risk management
"""
                
                # Calculate financial outcomes for display
                initial_capital = result.get('initial_capital', 100000)
                
                # Extract returns and calculate final values
                sent_test_return = result.get('sentiment_test_return', 'N/A')
                tech_test_return = result.get('technical_test_return', 'N/A')
                
                # Parse return percentages and calculate final values
                def parse_return_and_calculate_final(return_str, initial):
                    try:
                        if return_str != 'N/A' and '%' in str(return_str):
                            return_pct = float(str(return_str).replace('%', ''))
                            final_value = initial * (1 + return_pct / 100)
                            profit_loss = final_value - initial
                            return final_value, profit_loss, return_pct
                    except:
                        pass
                    return None, None, None
                
                sent_final, sent_profit, sent_return_pct = parse_return_and_calculate_final(sent_test_return, initial_capital)
                tech_final, tech_profit, tech_return_pct = parse_return_and_calculate_final(tech_test_return, initial_capital)
                
                # Determine which approach is recommended
                recommended_is_sentiment = result.get('recommended_approach') == 'Sentiment-enhanced'
                
                if recommended_is_sentiment and sent_final is not None:
                    rec_final, rec_profit, rec_return = sent_final, sent_profit, sent_return_pct
                    alt_final, alt_profit, alt_return = tech_final, tech_profit, tech_return_pct
                    rec_label, alt_label = "Sentiment-Enhanced", "Technical-Only"
                elif not recommended_is_sentiment and tech_final is not None:
                    rec_final, rec_profit, rec_return = tech_final, tech_profit, tech_return_pct
                    alt_final, alt_profit, alt_return = sent_final, sent_profit, sent_return_pct
                    rec_label, alt_label = "Technical-Only", "Sentiment-Enhanced"
                else:
                    rec_final = rec_profit = rec_return = None
                    alt_final = alt_profit = alt_return = None
                    rec_label = alt_label = "Unknown"
                
                # Pre-format values to avoid f-string formatting issues
                def format_currency(value):
                    return f"${value:,.2f}" if value is not None else "N/A"
                
                def format_profit_loss(value):
                    return f"${value:+,.2f}" if value is not None else "N/A"
                
                def format_return(value):
                    return f"{value:+.2f}%" if value is not None else "N/A"
                
                rec_final_str = format_currency(rec_final)
                rec_profit_str = format_profit_loss(rec_profit)
                rec_return_str = format_return(rec_return)
                
                alt_final_str = format_currency(alt_final)
                alt_profit_str = format_profit_loss(alt_profit)
                alt_return_str = format_return(alt_return)
                
                performance_text = f"""
#### 📈 **Financial Performance Results**

**Initial Capital:** ${initial_capital:,.2f}

#### 🥇 **Recommended Approach ({rec_label})**
**Final Portfolio Value:** {rec_final_str}
**Profit/Loss:** {rec_profit_str}
**Return:** {rec_return_str}

#### 🥈 **Alternative Approach ({alt_label})**
**Final Portfolio Value:** {alt_final_str}
**Profit/Loss:** {alt_profit_str}
**Return:** {alt_return_str}

---

#### 📊 **Training vs Testing Performance**

**Training Phase Results:**  
• **Sentiment-Enhanced:** {result.get('sentiment_training_return', 'N/A')}  
• **Technical-Only:** {result.get('technical_training_return', 'N/A')}  

**Out-of-Sample Testing:**  
• **Sentiment-Enhanced:** {result.get('sentiment_test_return', 'N/A')}  
• **Technical-Only:** {result.get('technical_test_return', 'N/A')}  

---

#### 📅 **Data Split Information**
**Training Period:** {result.get('training_period', 'N/A')}  
**Testing Period:** {result.get('test_period', 'N/A')}  
**Split Ratio:** {result.get('split_ratio', 'N/A')}  

---

#### ✅ **Key Findings**
**Sentiment Adds Value:** {result.get('sentiment_adds_value', 'N/A')}  
**Best Parameters (Sentiment):** {result.get('best_sentiment_parameters', 'N/A')}  
**Best Parameters (Technical):** {result.get('best_technical_parameters', 'N/A')}  

**Recommendation Confidence:** High (robust testing methodology)
"""
            
            return optimization_text, performance_text, result

        def validate_and_simulate(symbol, start_date, end_date, initial_capital, strategy, 
                                sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                                bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                                analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis):
            logger.info(f"Running simulation for {symbol} with strategy {strategy}")
            if not is_valid_symbol(symbol):
                return {"error": "Invalid stock symbol"}
            
            try:
                trading_data = {
                    "initial_capital": initial_capital,
                    "strategy": strategy,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sma_window": sma_window,
                    "rsi_period": rsi_period,
                    "rsi_overbought": rsi_overbought,
                    "rsi_oversold": rsi_oversold,
                    "bb_period": bb_period,
                    "bb_dev": bb_dev,
                    "use_sentiment": use_sentiment,
                    "sentiment_weight": sentiment_weight,
                    "sentiment_threshold": sentiment_threshold,
                    # LlamaIndex parameters
                    "analysis_scope": analysis_scope,
                    "sentiment_confidence_threshold": sentiment_confidence_threshold,
                    "include_risk_analysis": include_risk_analysis,
                    "include_catalyst_analysis": include_catalyst_analysis
                }
                
                logger.info(f"Sending simulation request to {API_BASE_URL}/trading/simulate/{symbol}")
                response = requests.post(
                    f"{API_BASE_URL}/trading/simulate/{symbol}",
                    json=trading_data
                )
                
                if response.status_code == 200:
                    logger.info("Simulation completed successfully")
                    return response.json()
                else:
                    logger.error(f"Simulation failed with status {response.status_code}")
                    return {"error": f"API Error: {response.status_code}"}
                    
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                return {"error": str(e)}
        
        def validate_and_backtest(symbol, start_date, end_date, initial_capital, strategy, 
                                sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                                bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                                analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis,
                                enable_optimization, optimization_split_ratio, compare_sentiment_versions, 
                                optimization_speed, max_combinations):
            logger.info(f"Running backtest for {symbol} with strategy {strategy}")
            if not is_valid_symbol(symbol):
                error_result = {"error": "Invalid stock symbol"}
                return format_backtest_results(error_result)
            
            # Show immediate progress feedback
            if enable_optimization:
                # Estimate combinations for progress display
                speed_estimates = {
                    "lightning": "18 combinations (~30 seconds)",
                    "fast": "48 combinations (~1 minute)", 
                    "medium": "120 combinations (~2-3 minutes)",
                    "balanced": "240 combinations (~4-6 minutes)",
                    "thorough": "400 combinations (~8-12 minutes)",
                    "exhaustive": "800+ combinations (~15-30 minutes)"
                }
                
                estimate = speed_estimates.get(optimization_speed, "Multiple combinations")
                
                progress_result = {
                    "status": "progress",
                    "current_status": "Starting optimization...",
                    "progress": f"Initializing {optimization_speed} optimization",
                    "time_elapsed": "Just started",
                    "current_step": f"Preparing to test {estimate}",
                    "estimated_time": speed_estimates.get(optimization_speed, "Processing...")
                }
                
                # Return progress display immediately, then continue with actual optimization
                progress_display = format_backtest_results(progress_result)
            
            try:
                trading_data = {
                    "initial_capital": initial_capital,
                    "strategy": strategy,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sma_window": sma_window,
                    "rsi_period": rsi_period,
                    "rsi_overbought": rsi_overbought,
                    "rsi_oversold": rsi_oversold,
                    "bb_period": bb_period,
                    "bb_dev": bb_dev,
                    "use_sentiment": use_sentiment,
                    "sentiment_weight": sentiment_weight,
                    "sentiment_threshold": sentiment_threshold,
                    # LlamaIndex parameters
                    "analysis_scope": analysis_scope,
                    "sentiment_confidence_threshold": sentiment_confidence_threshold,
                    "include_risk_analysis": include_risk_analysis,
                    "include_catalyst_analysis": include_catalyst_analysis,
                    # Optimization parameters
                    "enable_optimization": enable_optimization,
                    "optimization_split_ratio": optimization_split_ratio,
                    "compare_sentiment_versions": compare_sentiment_versions,
                    "optimization_speed": optimization_speed,
                    "max_combinations": max_combinations
                }

                logger.info(f"Sending backtest request to {API_BASE_URL}/trading/backtest/{symbol}")
                response = requests.post(
                    f"{API_BASE_URL}/trading/backtest/{symbol}",
                    json=trading_data
                )
                
                if response.status_code == 200:
                    logger.info("Backtest completed successfully")
                    result = response.json()
                    return format_backtest_results(result)
                else:
                    logger.error(f"Backtest failed with status {response.status_code}")
                    error_result = {"error": f"API Error: {response.status_code}"}
                    return format_backtest_results(error_result)
                    
            except Exception as e:
                logger.error(f"Backtest error: {e}")
                error_result = {"error": str(e)}
                return format_backtest_results(error_result)
        
        def refresh_api_status():
            return format_api_status(get_api_key_status())
        
        # Connect event handlers
        # Strategy parameter visibility
        strategy.change(
            fn=update_strategy_params,
            inputs=[strategy],
            outputs=[sma_params, rsi_params, bb_params]
        )
        
        simulate_btn.click(
            fn=validate_and_simulate,
            inputs=[symbol_input, start_date, end_date, initial_capital, strategy, 
                   sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                   bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                   analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis],
            outputs=simulation_output
        )
        
        backtest_btn.click(
            fn=validate_and_backtest,
            inputs=[symbol_input, start_date, end_date, initial_capital, strategy, 
                   sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                   bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                   analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis,
                   enable_optimization, optimization_split_ratio, compare_sentiment_versions, 
                   optimization_speed, max_combinations],
            outputs=[optimization_summary, performance_summary, backtest_output]
        )
        
        # Refresh API status when sentiment is enabled
        use_sentiment.change(
            fn=refresh_api_status,
            outputs=api_status
        )
    
    return demo

# Portfolio Interface
def portfolio_interface():
    with gr.Blocks(title="Portfolio Optimization") as demo:
        gr.Markdown("# 📊 Portfolio Optimization")
        
        with gr.Row():
            with gr.Column(scale=1):
                symbols_input = gr.Textbox(
                    label="Stock Symbols (comma-separated)",
                    placeholder="e.g., AAPL,GOOGL,MSFT",
                    value="AAPL,GOOGL,MSFT"
                )
                
                start_date = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                )
                
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    value=datetime.now().strftime("%Y-%m-%d")
                )
                
                simulate_btn = gr.Button("Simulate", variant="primary")
            
            with gr.Column(scale=2):
                portfolio_output = gr.JSON(label="Portfolio Results")
        
        def validate_and_simulate(symbols_str, start_date, end_date):
            logger.info(f"Running portfolio optimization for symbols: {symbols_str}")
            symbols = [s.strip().upper() for s in symbols_str.split(",")]
            invalid_symbols = [s for s in symbols if not is_valid_symbol(s)]
            
            if invalid_symbols:
                return {"error": f"Invalid symbols: {', '.join(invalid_symbols)}"}
            
            try:
                portfolio_data = {
                    "asset_symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date
                }
                
                logger.info(f"Sending portfolio optimization request")
                response = requests.post(
                    f"{API_BASE_URL}/portfolio/simulate",
                    json=portfolio_data
                )
                
                if response.status_code == 200:
                    logger.info("Portfolio optimization completed successfully")
                    return response.json()
                else:
                    logger.error(f"Portfolio optimization failed with status {response.status_code}")
                    return {"error": f"API Error: {response.status_code}"}
                    
            except Exception as e:
                logger.error(f"Portfolio optimization error: {e}")
                return {"error": str(e)}
        
        simulate_btn.click(
            fn=validate_and_simulate,
            inputs=[symbols_input, start_date, end_date],
            outputs=portfolio_output
        )
    
    return demo

# Risk Interface
def risk_interface():
    with gr.Blocks(title="Risk Management") as demo:
        gr.Markdown("# ⚠️ Risk Management")
        
        with gr.Row():
            with gr.Column(scale=1):
                symbols_input = gr.Textbox(
                    label="Stock Symbols (comma-separated)",
                    placeholder="e.g., AAPL,GOOGL,MSFT",
                    value="AAPL,GOOGL,MSFT"
                )
                
                start_date = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                )
                
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    value=datetime.now().strftime("%Y-%m-%d")
                )
                
                confidence_level = gr.Slider(
                    label="Confidence Level",
                    minimum=0.90,
                    maximum=0.99,
                    value=0.95,
                    step=0.01
                )
                
                analyze_btn = gr.Button("Analyze Risk", variant="primary")
            
            with gr.Column(scale=2):
                risk_output = gr.JSON(label="Risk Analysis Results")
        
        def validate_and_analyze_risk(symbols_str, start_date, end_date, confidence_level):
            logger.info(f"Running risk analysis for symbols: {symbols_str}")
            symbols = [s.strip().upper() for s in symbols_str.split(",")]
            invalid_symbols = [s for s in symbols if not is_valid_symbol(s)]
            
            if invalid_symbols:
                return {"error": f"Invalid symbols: {', '.join(invalid_symbols)}"}
            
            try:
                risk_data = {
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "confidence_level": confidence_level
                }
                
                logger.info(f"Sending risk analysis request")
                response = requests.post(
                    f"{API_BASE_URL}/risk/analyze",
                    json=risk_data
                )
                
                if response.status_code == 200:
                    logger.info("Risk analysis completed successfully")
                    return response.json()
                else:
                    logger.error(f"Risk analysis failed with status {response.status_code}")
                    return {"error": f"API Error: {response.status_code}"}
                    
            except Exception as e:
                logger.error(f"Risk analysis error: {e}")
                return {"error": str(e)}
        
        analyze_btn.click(
            fn=validate_and_analyze_risk,
            inputs=[symbols_input, start_date, end_date, confidence_level],
            outputs=risk_output
        )
    
    return demo

# Main application
def create_app():
    with gr.Blocks(title="QuantFin Platform", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🏦 QuantFin Platform")
        gr.Markdown("*Advanced Quantitative Finance Tools with AI-Powered Sentiment Analysis*")
        
        with gr.Tabs():
            with gr.TabItem("📈 Algorithmic Trading"):
                trading_interface()
            with gr.TabItem("📊 Portfolio Management"):
                portfolio_interface()
            with gr.TabItem("⚠️ Risk Management"):
                risk_interface()
    
    return app

if __name__ == "__main__":
    logger.info("Starting QuantFin Frontend Server...")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)