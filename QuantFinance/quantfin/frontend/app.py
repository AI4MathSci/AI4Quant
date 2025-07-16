import gradio as gr
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Load .env file
env_path = os.path.join(os.getcwd(), '.env')
logger.info(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Load environment variables for API frequency controls with fallback defaults
OPENAI_API_FREQUENCY_MAX = int(os.getenv('OPENAI_API_FREQUENCY_MAX', 60))
OPENAI_API_FREQUENCY_DEFAULT = int(os.getenv('OPENAI_API_FREQUENCY_DEFAULT', 60))
ALPHA_VANTAGE_API_FREQUENCY_MAX = int(os.getenv('ALPHA_VANTAGE_API_FREQUENCY_MAX', 60))
ALPHA_VANTAGE_API_FREQUENCY_DEFAULT = int(os.getenv('ALPHA_VANTAGE_API_FREQUENCY_DEFAULT', 30))

# Import sentiment configuration
from quantfin.backend.config.config import (
    SENTIMENT_CLASSIFICATION_THRESHOLD,
    SENTIMENT_CONFIDENCE_THRESHOLD, 
    SENTIMENT_DECAY_FACTOR,
    SENTIMENT_COMBO_WEIGHT,
    SENTIMENT_WEIGHT_DEFAULT
)

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
        return "⚠️ OpenAI API Key: Not Configured (Sentiment analysis will be using FinBert and keyword-based methods)"

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
        /* Trading Results Styling */
        #trading-results {
            min-height: 300px;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            background-color: #fafafa;
        }
        
        #trading-results table {
            font-size: 14px;
        }
        #simulation-summary, #simulation-performance {
            min-height: 300px !important;
            max-height: 500px !important;
            overflow-y: auto !important;
            padding: 15px !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            background-color: #fafafa !important;
        }
        .simulation-display {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            line-height: 1.6 !important;
        }
        .simulation-display h4 {
            color: #2563eb !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }
        .sentiment-warning {
            background-color: #fff3cd !important;
            border: 1px solid #ffeaa7 !important;
            border-radius: 8px !important;
            padding: 10px !important;
            margin: 10px 0 !important;
            color: #856404 !important;
            font-weight: 500 !important;
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
                        value=SENTIMENT_WEIGHT_DEFAULT,
                        step=0.1
                    )
                    
                    sentiment_threshold = gr.Slider(
                        label="Sentiment Threshold (minimum strength to trigger)",
                        minimum=0.0,
                        maximum=0.5,
                        value=SENTIMENT_CLASSIFICATION_THRESHOLD,
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
                        value=SENTIMENT_CONFIDENCE_THRESHOLD,
                        step=0.1
                    )
                    
                    include_risk_analysis = gr.Checkbox(
                        label="Include Risk Factor Analysis",
                        value=False,
                        info="Analyze company-specific, industry, and market risks (must enable sentiment analysis first)"
                    )
                    
                    include_catalyst_analysis = gr.Checkbox(
                        label="Include Catalyst Analysis",
                        value=False,
                        info="Identify potential positive and negative price catalysts (must enable sentiment analysis first)"
                    )
                    
                    # Warning message for sentiment dependencies
                    sentiment_warning = gr.Markdown(
                        value="",
                        elem_classes=["sentiment-warning"],
                        visible=False
                    )
                    
                    # Sentiment API Call Controls
                    with gr.Row():
                        openai_api_frequency = gr.Slider(
                            label="OpenAI API Frequency",
                            minimum=1,
                            maximum=OPENAI_API_FREQUENCY_MAX,
                            value=OPENAI_API_FREQUENCY_DEFAULT,
                            step=1,
                            info="Call OpenAI API every N trading days (higher = fewer API calls, lower cost)"
                        )
                        
                        alpha_vantage_api_frequency = gr.Slider(
                            label="Alpha Vantage API Frequency",
                            minimum=1,
                            maximum=ALPHA_VANTAGE_API_FREQUENCY_MAX,
                            value=ALPHA_VANTAGE_API_FREQUENCY_DEFAULT,
                            step=1,
                            info="Call Alpha Vantage API every N trading days (higher = fewer API calls, lower cost)"
                        )
                        
                        sentiment_decay_factor = gr.Slider(
                            label="Sentiment Decay Factor",
                            minimum=0.7,
                            maximum=0.99,
                            value=SENTIMENT_DECAY_FACTOR,
                            step=0.01,
                            info="How quickly sentiment decays between API calls (0.9 = 10% decay per day)"
                        )
                        
                        sentiment_combo_weight = gr.Slider(
                            label="Sentiment Combo Weight",
                            minimum=0.0,
                            maximum=1.0,
                            value=SENTIMENT_COMBO_WEIGHT,
                            step=0.1,
                            info="Weight for combining decayed API sentiment with keyword sentiment (0.5 = equal weight)"
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
                        value=400,
                        minimum=50,
                        maximum=1000,
                        info="Upper limit for parameter testing (limits compute time)"
                    )
            
            with gr.Column(scale=2):
                with gr.Tab("Simulation"):
                    simulate_btn = gr.Button("Run Simulation", variant="primary")
                    
                    # Enhanced results display
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 **Simulation Results**")
                            simulation_summary = gr.Markdown(
                                value="""
**Ready to simulate...**

• Click "Run Simulation" to start forward-looking analysis
• System will project performance using current parameters
• Results will show projected portfolio value and returns

*Simulation results will appear here...*
                                """, 
                                elem_id="simulation-summary",
                                elem_classes=["simulation-display"]
                            )
                        
                        with gr.Column():
                            gr.Markdown("### 📊 **Performance Projection**")
                            simulation_performance = gr.Markdown(
                                value="""
**Performance projection will appear here...**

**Projected Results:**
• Final Portfolio Value: *pending*
• Total Return: *pending*
• Annualized Return: *pending*

**Configuration:**
• Strategy: *current settings*
• Sentiment Analysis: *enabled/disabled*

*Detailed projection analysis will be shown after simulation completes...*
                                """,
                                elem_id="simulation-performance", 
                                elem_classes=["simulation-display"]
                            )
                    
                    with gr.Row():
                        gr.Markdown("### 📋 **Detailed Results** (Advanced)")
                        simulation_output = gr.JSON(label="Complete Analysis", visible=False)
                    
                    # Toggle for detailed results
                    show_simulation_detailed = gr.Checkbox(label="Show Detailed JSON Results", value=False)
                    
                    def toggle_simulation_detailed_results(show):
                        return gr.update(visible=show)
                    
                    show_simulation_detailed.change(
                        fn=toggle_simulation_detailed_results,
                        inputs=[show_simulation_detailed],
                        outputs=[simulation_output]
                    )
                
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
**Sentiment Analysis:** {'Enabled' if use_sentiment else 'Disabled'}
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
**Sentiment Weight:** {result.get('sentiment_weight', 'N/A') if use_sentiment else 'N/A (Disabled)'}

#### 📋 **Analysis Notes**
• This was a single-run backtest with your current parameter settings
• For optimized parameters, enable "Parameter Optimization" 
• Results show performance using {'sentiment-enhanced' if use_sentiment else 'technical-only'} approach

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

        def format_simulation_results(result):
            """Format simulation results for enhanced display"""
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
**Unable to complete simulation due to error.**

Please review the error message and adjust your settings accordingly.
                    """,
                    result
                )
            
            # Extract simulation data
            initial_capital = result.get('initial_capital', 100000)
            final_value = result.get('final_portfolio_value', initial_capital)
            profit_loss = final_value - initial_capital
            
            # Fix return calculation - try multiple field names
            return_pct = result.get('total_return_pct', 
                          result.get('return_pct', 
                          result.get('total_return', 
                          result.get('return', 0))))
            
            # If still 0, calculate manually
            if return_pct == 0 and final_value != initial_capital:
                return_pct = ((final_value - initial_capital) / initial_capital) * 100
            
            # Additional metrics
            annualized_return = result.get('annualized_return', return_pct)
            sharpe_ratio = result.get('sharpe_ratio', 0.0)
            max_drawdown = result.get('max_drawdown', 0.0)
            
            # Get dates - try multiple sources
            start_date_display = (result.get('start_date') or 
                                 result.get('period_start') or 
                                 result.get('simulation_start', 'N/A'))
            end_date_display = (result.get('end_date') or 
                               result.get('period_end') or 
                               result.get('simulation_end', 'N/A'))
            
            # Get sentiment status from result (which now contains UI parameters)
            use_sentiment = result.get('use_sentiment', False)
            sentiment_weight = result.get('sentiment_weight', 0.0)
            
            # Left column: Financial results and performance metrics
            simulation_text = f"""
#### 💰 **Financial Results**

**Initial Capital:** ${initial_capital:,.2f}

**Final Portfolio Value:** ${final_value:,.2f}

**Profit/Loss:** ${profit_loss:+,.2f}

**Total Return:** {return_pct:+.2f}%

---

#### 📈 **Performance Metrics**

**Annualized Return:** {annualized_return:+.2f}%

**Sharpe Ratio:** {sharpe_ratio:.3f}

**Max Drawdown:** {max_drawdown:+.2f}%
"""
            
            # Right column: Configuration and analysis notes
            performance_text = f"""
#### ⚙️ **Configuration Used**

**Strategy:** {result.get('strategy', 'N/A')}

**Time Period:** {start_date_display} to {end_date_display}

**Sentiment Analysis:** {'Enabled' if use_sentiment else 'Disabled'}

**Sentiment Weight:** {f"{sentiment_weight*100:.0f}%" if use_sentiment else 'N/A (Disabled)'}

---

#### 📋 **Analysis Notes**

• Forward-looking projection using current parameters

• Based on historical patterns and market conditions

• Actual results may vary due to market volatility

• For historical analysis, use the Backtesting tab

**Status:** {result.get('message', 'Simulation completed')}
"""
            
            return simulation_text, performance_text, result

        def validate_and_simulate(symbol, start_date, end_date, initial_capital, strategy, 
                                sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                                bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                                analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis,
                                openai_api_frequency, sentiment_decay_factor, sentiment_combo_weight,
                                alpha_vantage_api_frequency):
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
                    "include_catalyst_analysis": include_catalyst_analysis,
                    # API call parameters
                    "openai_api_frequency": openai_api_frequency,
                    "sentiment_decay_factor": sentiment_decay_factor,
                    "sentiment_combo_weight": sentiment_combo_weight,
                    "alpha_vantage_api_frequency": alpha_vantage_api_frequency
                }
                
                logger.info(f"Sending simulation request to {API_BASE_URL}/trading/simulate/{symbol}")
                response = requests.post(
                    f"{API_BASE_URL}/trading/simulate/{symbol}",
                    json=trading_data
                )
                
                if response.status_code == 200:
                    logger.info("Simulation completed successfully")
                    result = response.json()
                    # Add the input dates and sentiment parameters to the result for proper display
                    result['start_date'] = start_date
                    result['end_date'] = end_date
                    result['use_sentiment'] = use_sentiment
                    result['sentiment_weight'] = sentiment_weight
                    return format_simulation_results(result)
                else:
                    logger.error(f"Simulation failed with status {response.status_code}")
                    error_result = {"error": f"API Error: {response.status_code}"}
                    return format_simulation_results(error_result)
                    
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                return {"error": str(e)}
        
        def validate_and_backtest(symbol, start_date, end_date, initial_capital, strategy, 
                                sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                                bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                                analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis,
                                openai_api_frequency, sentiment_decay_factor, sentiment_combo_weight,
                                alpha_vantage_api_frequency,
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
                    # Sentiment API call parameters
                    "openai_api_frequency": openai_api_frequency,
                    "sentiment_decay_factor": sentiment_decay_factor,
                    "sentiment_combo_weight": sentiment_combo_weight,
                    "alpha_vantage_api_frequency": alpha_vantage_api_frequency,                    
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
        
        def handle_sentiment_toggle(use_sentiment):
            """Handle sentiment analysis toggle - disable dependent controls when disabled"""
            if not use_sentiment:
                # Disable and uncheck dependent controls
                return (
                    gr.update(value=False, interactive=False),  # include_risk_analysis
                    gr.update(value=False, interactive=False),  # include_catalyst_analysis
                    gr.update(value="⚠️ **Sentiment Analysis Disabled**\n\nRisk Factor Analysis and Catalyst Analysis require sentiment analysis to be enabled.\n\nPlease enable sentiment analysis first to use these features.", visible=True)  # warning
                )
            else:
                # Enable dependent controls
                return (
                    gr.update(interactive=True),  # include_risk_analysis
                    gr.update(interactive=True),  # include_catalyst_analysis
                    gr.update(visible=False)  # warning
                )
        
        def handle_risk_analysis_attempt(include_risk, use_sentiment):
            """Handle attempt to check risk analysis without sentiment enabled"""
            if include_risk and not use_sentiment:
                return (
                    gr.update(value=False),  # uncheck risk analysis
                    gr.update(value="⚠️ **Cannot Enable Risk Factor Analysis**\n\nRisk Factor Analysis requires sentiment analysis to be enabled.\n\nPlease enable sentiment analysis first.", visible=True)  # warning
                )
            return (
                gr.update(),  # keep risk analysis as is
                gr.update(visible=False)  # hide warning
            )
        
        def handle_catalyst_analysis_attempt(include_catalyst, use_sentiment):
            """Handle attempt to check catalyst analysis without sentiment enabled"""
            if include_catalyst and not use_sentiment:
                return (
                    gr.update(value=False),  # uncheck catalyst analysis
                    gr.update(value="⚠️ **Cannot Enable Catalyst Analysis**\n\nCatalyst Analysis requires sentiment analysis to be enabled.\n\nPlease enable sentiment analysis first.", visible=True)  # warning
                )
            return (
                gr.update(),  # keep catalyst analysis as is
                gr.update(visible=False)  # hide warning
            )
        
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
                   analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis,
                   openai_api_frequency, sentiment_decay_factor, sentiment_combo_weight,
                   alpha_vantage_api_frequency],
            outputs=[simulation_summary, simulation_performance, simulation_output]
        )
        
        backtest_btn.click(
            fn=validate_and_backtest,
            inputs=[symbol_input, start_date, end_date, initial_capital, strategy, 
                   sma_window, rsi_period, rsi_overbought, rsi_oversold, 
                   bb_period, bb_dev, use_sentiment, sentiment_weight, sentiment_threshold,
                   analysis_scope, sentiment_confidence_threshold, include_risk_analysis, include_catalyst_analysis,
                   openai_api_frequency, sentiment_decay_factor, sentiment_combo_weight,
                   alpha_vantage_api_frequency,
                   enable_optimization, optimization_split_ratio, compare_sentiment_versions, 
                   optimization_speed, max_combinations],
            outputs=[optimization_summary, performance_summary, backtest_output]
        )
        
        # Refresh API status when sentiment is enabled
        use_sentiment.change(
            fn=refresh_api_status,
            outputs=api_status
        )
        
        # Handle sentiment toggle
        use_sentiment.change(
            fn=handle_sentiment_toggle,
            inputs=[use_sentiment],
            outputs=[include_risk_analysis, include_catalyst_analysis, sentiment_warning]
        )
        
        # Handle risk analysis attempt
        include_risk_analysis.change(
            fn=handle_risk_analysis_attempt,
            inputs=[include_risk_analysis, use_sentiment],
            outputs=[include_risk_analysis, sentiment_warning]
        )
        
        # Handle catalyst analysis attempt
        include_catalyst_analysis.change(
            fn=handle_catalyst_analysis_attempt,
            inputs=[include_catalyst_analysis, use_sentiment],
            outputs=[include_catalyst_analysis, sentiment_warning]
        )
    
    return demo

# Portfolio Interface
def portfolio_interface():
    with gr.Blocks(title="Portfolio Management", css="""
        /* Portfolio Results Styling */
        #portfolio-results {
            min-height: 300px;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            background-color: #fafafa;
        }
        
        #portfolio-results table {
            font-size: 14px;
        }
        """) as demo:
        gr.Markdown("# 📊 Portfolio Management")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input Controls
                gr.Markdown("### 📋 Portfolio Configuration")
                
                symbols_input = gr.Textbox(
                    label="Stock Symbols (comma-separated)",
                    placeholder="AAPL,GOOGL,MSFT,AMZN",
                    value="AAPL,GOOGL,MSFT,AMZN"
                )
                
                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        placeholder="2023-01-01",
                        value="2023-01-01"
                    )
                    end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        placeholder="2024-01-01",
                        value="2024-01-01"
                    )
                
                with gr.Row():
                    strategy_dropdown = gr.Dropdown(
                        choices=["Equal Weight", "Mean Variance Optimization", "Momentum"],
                        label="Strategy (for Simulate/Backtest)",
                        value="Equal Weight"
                    )
                    initial_capital = gr.Number(
                        label="Initial Capital ($)",
                        value=100000,
                        minimum=1000
                    )
                
                with gr.Row():
                    target_return = gr.Slider(
                        minimum=0.05,
                        maximum=0.25,
                        value=0.12,
                        step=0.01,
                        label="Target Return (for Mean Variance)",
                        info="Annual target return for optimization"
                    )
                    lookback_period = gr.Slider(
                        minimum=60,
                        maximum=504,
                        value=252,
                        step=30,
                        label="Lookback Period (for Momentum)",
                        info="Trading days for momentum calculation"
                    )
            
            with gr.Column(scale=2):
                # Action Buttons in upper right
                gr.Markdown("### 🚀 Analysis Options")
                
                with gr.Row():
                    simulate_btn = gr.Button("📈 Simulate", variant="primary", size="lg")
                    optimize_btn = gr.Button("🎯 Optimize", variant="primary", size="lg")
                    backtest_btn = gr.Button("📊 Backtest", variant="primary", size="lg")
                
                # Help Text below buttons
                with gr.Accordion("ℹ️ Analysis Types", open=False):
                    gr.Markdown("""
                    **Simulate**: Forward-looking projections using selected strategy and parameters
                    
                    **Optimize**: Tests multiple strategies and parameters to find the best configuration
                    
                    **Backtest**: Historical performance analysis using selected strategy and parameters
                    """)
                
                # Results Display below help text
                portfolio_output = gr.HTML(
                    value="<div style='text-align: center; padding: 50px; color: #666;'>📊 Select an analysis type and configure your portfolio to get started</div>",
                    elem_id="portfolio-results"
                )
        
        # Event Handlers
        def validate_and_process(symbols_str, start_date, end_date, strategy, initial_capital, target_return, lookback_period, action_type):
            """Validate inputs and call appropriate backend endpoint"""
            try:
                # Validate symbols
                symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
                if not symbols:
                    return "<div style='color: red;'>❌ Please enter at least one stock symbol</div>"
                
                # Validate symbols using yfinance
                try:
                    test_data = yf.download(symbols, period="5d", progress=False)
                    if test_data.empty:
                        return "<div style='color: red;'>❌ No data found for the provided symbols</div>"
                except Exception as e:
                    return f"<div style='color: red;'>❌ Error validating symbols: {str(e)}</div>"
                
                # Prepare request data
                portfolio_data = {
                    "asset_symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "strategy": strategy,
                    "initial_capital": float(initial_capital),
                    "target_return": float(target_return),
                    "lookback_period": int(lookback_period)
                }
                
                # Call appropriate endpoint
                if action_type == "simulate":
                    response = requests.post(f"{API_BASE_URL}/portfolio/simulate", json=portfolio_data)
                elif action_type == "optimize":
                    response = requests.post(f"{API_BASE_URL}/portfolio/optimize", json=portfolio_data)
                elif action_type == "backtest":
                    response = requests.post(f"{API_BASE_URL}/portfolio/backtest", json=portfolio_data)
                else:
                    return "<div style='color: red;'>❌ Invalid action type</div>"
                
                if response.status_code == 200:
                    result = response.json()
                    return format_portfolio_results(result, action_type)
                else:
                    return f"<div style='color: red;'>❌ Error: {response.text}</div>"
                    
            except Exception as e:
                return f"<div style='color: red;'>❌ Error: {str(e)}</div>"
        
        # Button click handlers
        simulate_btn.click(
            fn=lambda *args: validate_and_process(*args, "simulate"),
            inputs=[symbols_input, start_date, end_date, strategy_dropdown, initial_capital, target_return, lookback_period],
            outputs=portfolio_output
        )
        
        optimize_btn.click(
            fn=lambda *args: validate_and_process(*args, "optimize"),
            inputs=[symbols_input, start_date, end_date, strategy_dropdown, initial_capital, target_return, lookback_period],
            outputs=portfolio_output
        )
        
        backtest_btn.click(
            fn=lambda *args: validate_and_process(*args, "backtest"),
            inputs=[symbols_input, start_date, end_date, strategy_dropdown, initial_capital, target_return, lookback_period],
            outputs=portfolio_output
        )
    
    return demo

def format_portfolio_results(result: dict, action_type: str) -> str:
    """Format portfolio results for display"""
    try:
        if action_type == "optimize":
            return format_optimization_results(result)
        else:
            return format_simulation_backtest_results(result, action_type)
    except Exception as e:
        return f"<div style='color: red;'>❌ Error formatting results: {str(e)}</div>"

def format_optimization_results(result: dict) -> str:
    """Format optimization results with comprehensive analysis"""
    if not result.get("optimization_complete", False):
        return f"""
        <div style='background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; margin: 10px 0;'>
            <h3 style='color: #856404; margin-top: 0;'>⚠️ Optimization Failed</h3>
            <p><strong>Error:</strong> {result.get('error', 'Unknown error occurred')}</p>
            <p><strong>Fallback:</strong> {result.get('fallback_strategy', 'Equal Weight')}</p>
        </div>
        """
    
    best_strategy = result.get("best_strategy", {})
    recommendation = result.get("recommendation", {})
    all_results = result.get("all_results", [])
    
    # Helper functions for formatting
    def format_currency(value):
        return f"${value:,.2f}" if value is not None else "N/A"
    
    def format_percentage(value):
        return f"{value*100:+.2f}%" if value is not None else "N/A"
    
    def format_ratio(value):
        return f"{value:.3f}" if value is not None else "N/A"
    
    # Main results
    html = f"""
    <div style='background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;'>
        <h2 style='color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px;'>
            🎯 Portfolio Optimization Results
        </h2>
        
        <div style='background: #e8f5e8; border-left: 4px solid #27ae60; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #27ae60; margin-top: 0;'>🏆 Recommended Strategy</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
                <div><strong>Strategy:</strong> {best_strategy.get('strategy', 'N/A')}</div>
                <div><strong>Parameters:</strong> {str(best_strategy.get('parameters', {})) if best_strategy.get('parameters') else 'Default'}</div>
                <div><strong>Final Value:</strong> {format_currency(best_strategy.get('final_portfolio_value'))}</div>
                <div><strong>Total Return:</strong> {format_percentage(best_strategy.get('total_return'))}</div>
                <div><strong>Annual Return:</strong> {format_percentage(best_strategy.get('annual_return'))}</div>
                <div><strong>Volatility:</strong> {format_percentage(best_strategy.get('volatility'))}</div>
                <div><strong>Sharpe Ratio:</strong> {format_ratio(best_strategy.get('sharpe_ratio'))}</div>
                <div><strong>Profit/Loss:</strong> {format_currency(best_strategy.get('total_return_dollars'))}</div>
            </div>
        </div>
        
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;'>
            <h3 style='color: #495057; margin-top: 0;'>📊 Optimization Summary</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;'>
                <div><strong>Strategies Tested:</strong> {result.get('strategies_tested', 0)}</div>
                <div><strong>Improvement vs Equal Weight:</strong> {format_ratio(result.get('improvement_over_equal_weight', 0))}</div>
            </div>
        </div>
        
        <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #856404; margin-top: 0;'>💡 Portfolio Allocation</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;'>
    """
    
    # Add weight breakdown
    weights = best_strategy.get('weights', {})
    for symbol, weight in weights.items():
        html += f"<div><strong>{symbol}:</strong> {weight*100:.1f}%</div>"
    
    html += """
            </div>
        </div>
    """
    
    # Add comparison table
    if len(all_results) > 1:
        html += """
        <div style='margin: 20px 0;'>
            <h3 style='color: #495057;'>📈 Strategy Comparison (Top 5)</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white;'>
                    <thead>
                        <tr style='background: #f8f9fa;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Strategy</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Sharpe Ratio</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Annual Return</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Volatility</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Final Value</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, strategy in enumerate(all_results[:5]):
            row_style = "background: #f8f9fa;" if i % 2 == 0 else ""
            strategy_name = strategy.get('strategy', 'Unknown')
            params = strategy.get('parameters', {})
            if params:
                strategy_display = f"{strategy_name} ({params})"
            else:
                strategy_display = strategy_name
                
            html += f"""
                        <tr style='{row_style}'>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'>{strategy_display}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_ratio(strategy.get('sharpe_ratio'))}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_percentage(strategy.get('annual_return'))}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_percentage(strategy.get('volatility'))}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_currency(strategy.get('final_portfolio_value'))}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    html += """
        <div style='background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #0c5460; margin-top: 0;'>💬 Recommendation</h3>
            <p style='margin: 0; font-size: 16px;'>{}</p>
        </div>
    </div>
    """.format(result.get('message', 'Optimization completed successfully.'))
    
    return html

def format_simulation_backtest_results(result: dict, action_type: str) -> str:
    """Format simulation and backtest results"""
    # Helper functions
    def format_currency(value):
        return f"${value:,.2f}" if value is not None else "N/A"
    
    def format_percentage(value):
        return f"{value:+.2f}%" if value is not None else "N/A"
    
    def format_ratio(value):
        return f"{value:.3f}" if value is not None else "N/A"
    
    action_title = "📈 Portfolio Simulation" if action_type == "simulate" else "📊 Portfolio Backtest"
    analysis_type = "Projected" if action_type == "simulate" else "Actual"
    
    html = f"""
    <div style='background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;'>
        <h2 style='color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px;'>
            {action_title} Results
        </h2>
        
        <div style='background: #e8f5e8; border-left: 4px solid #27ae60; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #27ae60; margin-top: 0;'>💰 Financial Summary</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
                <div><strong>Strategy:</strong> {result.get('strategy', 'N/A')}</div>
                <div><strong>Initial Capital:</strong> {format_currency(result.get('initial_capital'))}</div>
    """
    
    # Different fields for simulate vs backtest
    if action_type == "simulate":
        html += f"""
                <div><strong>Projected Final Value:</strong> {format_currency(result.get('projected_final_value'))}</div>
                <div><strong>Projected Return:</strong> {format_currency(result.get('projected_return_dollars'))}</div>
                <div><strong>Projected Return %:</strong> {format_percentage(result.get('projected_return_percent'))}</div>
        """
    else:
        html += f"""
                <div><strong>Final Portfolio Value:</strong> {format_currency(result.get('final_portfolio_value'))}</div>
                <div><strong>Total Return:</strong> {format_currency(result.get('total_return_dollars'))}</div>
                <div><strong>Total Return %:</strong> {format_percentage(result.get('total_return_percent'))}</div>
        """
    
    html += """
            </div>
        </div>
        
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;'>
            <h3 style='color: #495057; margin-top: 0;'>📊 Risk Metrics</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
    """
    
    # Risk metrics
    if action_type == "simulate":
        html += f"""
                <div><strong>Expected Annual Return:</strong> {format_percentage(result.get('expected_annual_return'))}</div>
                <div><strong>Volatility:</strong> {format_percentage(result.get('volatility'))}</div>
                <div><strong>Sharpe Ratio:</strong> {format_ratio(result.get('sharpe_ratio'))}</div>
        """
    else:
        html += f"""
                <div><strong>Annual Return:</strong> {format_percentage(result.get('annual_return'))}</div>
                <div><strong>Volatility:</strong> {format_percentage(result.get('volatility'))}</div>
                <div><strong>Sharpe Ratio:</strong> {format_ratio(result.get('sharpe_ratio'))}</div>
        """
    
    html += """
            </div>
        </div>
        
        <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #856404; margin-top: 0;'>💡 Portfolio Allocation</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;'>
    """
    
    # Portfolio weights
    weights = result.get('weights', {})
    allocation_amounts = result.get('allocation_amounts', {})
    
    for symbol in weights.keys():
        weight = weights.get(symbol, 0)
        amount = allocation_amounts.get(symbol, 0)
        html += f"<div><strong>{symbol}:</strong> {weight*100:.1f}% ({format_currency(amount)})</div>"
    
    html += """
            </div>
        </div>
    """
    
    # Individual asset performance
    final_asset_values = result.get('final_asset_values', {})
    if final_asset_values:
        html += f"""
        <div style='margin: 20px 0;'>
            <h3 style='color: #495057;'>📈 Individual Asset Performance</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white;'>
                    <thead>
                        <tr style='background: #f8f9fa;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Asset</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Initial Allocation</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>{analysis_type} Final Value</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Individual Return</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, symbol in enumerate(final_asset_values.keys()):
            row_style = "background: #f8f9fa;" if i % 2 == 0 else ""
            initial_amount = allocation_amounts.get(symbol, 0)
            final_amount = final_asset_values.get(symbol, 0)
            individual_return = ((final_amount - initial_amount) / initial_amount * 100) if initial_amount > 0 else 0
            
            html += f"""
                        <tr style='{row_style}'>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'><strong>{symbol}</strong></td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_currency(initial_amount)}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_currency(final_amount)}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{individual_return:+.2f}%</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    html += f"""
        <div style='background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #0c5460; margin-top: 0;'>💬 Analysis Complete</h3>
            <p style='margin: 0; font-size: 16px;'>{result.get('message', f'{action_title} completed successfully.')}</p>
        </div>
    </div>
    """
    
    return html

# Risk Interface
def risk_interface():
    with gr.Blocks(title="Risk Management", css="""
        #risk-results {
            min-height: 300px;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            background-color: #fafafa;
        }
        """) as demo:
        gr.Markdown("# ⚠️ Risk Management")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Risk Analysis Configuration")
                
                symbols_input = gr.Textbox(
                    label="Stock Symbols (comma-separated)",
                    placeholder="AAPL,GOOGL,MSFT,AMZN",
                    value="AAPL,GOOGL,MSFT"
                )
                
                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                    )
                    end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.now().strftime("%Y-%m-%d")
                    )
                
                with gr.Row():
                    confidence_level = gr.Slider(
                        label="Confidence Level",
                        minimum=0.90,
                        maximum=0.99,
                        value=0.95,
                        step=0.01
                    )
                    portfolio_value = gr.Number(
                        label="Portfolio Value ($)",
                        value=1000000,
                        minimum=10000
                    )
                
                # Help Text
                with gr.Accordion("ℹ️ Risk Analysis Types", open=False):
                    gr.Markdown("""
                    **Single Asset**: Individual risk metrics, stress testing, VaR breakdown
                    
                    **Portfolio**: Correlation analysis, component VaR, diversification metrics
                    
                    **Stress Testing**: Market crash scenarios, volatility spikes, correlation breakdown
                    """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🚀 Risk Analysis")
                
                analyze_btn = gr.Button("📊 Analyze Risk", variant="primary", size="lg")
                
                # Results Display
                risk_output = gr.HTML(
                    value="<div style='text-align: center; padding: 50px; color: #666;'>⚠️ Configure your analysis and click 'Analyze Risk' to get started</div>",
                    elem_id="risk-results"
                )
        
        def validate_and_analyze_risk_enhanced(symbols_str, start_date, end_date, confidence_level, portfolio_value):
            """Enhanced risk analysis with better formatting"""
            logger.info(f"Running risk analysis for symbols: {symbols_str}")
            symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
            invalid_symbols = [s for s in symbols if not is_valid_symbol(s)]
            
            if invalid_symbols:
                return f"<div style='color: red;'>❌ Invalid symbols: {', '.join(invalid_symbols)}</div>"
            
            try:
                # Prepare data for both single and multiple symbols
                if len(symbols) == 1:
                    risk_data = {
                        "symbol": symbols[0],
                        "start_date": start_date,
                        "end_date": end_date,
                        "confidence_level": confidence_level,
                        "portfolio_value": portfolio_value
                    }
                else:
                    risk_data = {
                        "symbols": symbols,
                        "start_date": start_date,
                        "end_date": end_date,
                        "confidence_level": confidence_level,
                        "portfolio_value": portfolio_value
                    }
                
                response = requests.post(f"{API_BASE_URL}/risk/analyze", json=risk_data)
                
                if response.status_code == 200:
                    result = response.json()
                    return format_risk_results(result)
                else:
                    return f"<div style='color: red;'>❌ API Error: {response.status_code}</div>"
                    
            except Exception as e:
                return f"<div style='color: red;'>❌ Error: {str(e)}</div>"
        
        # Update the click handler to include portfolio_value
        analyze_btn.click(
            fn=validate_and_analyze_risk_enhanced,
            inputs=[symbols_input, start_date, end_date, confidence_level, portfolio_value],
            outputs=risk_output
        )
    
    return demo

def format_risk_results(result: dict) -> str:
    """Format risk analysis results for display"""
    if result.get("analysis_type") == "Portfolio Risk Analysis":
        return format_portfolio_risk_results(result)
    else:
        return format_single_asset_risk_results(result)

def format_portfolio_risk_results(result: dict) -> str:
    """Format portfolio risk analysis results"""
    # Helper functions
    def format_currency(value):
        return f"${value:,.2f}" if value is not None else "N/A"
    
    def format_percentage(value):
        return f"{value:+.2f}%" if value is not None else "N/A"
    
    def format_ratio(value):
        return f"{value:.3f}" if value is not None else "N/A"
    
    symbols = result.get("symbols", [])
    portfolio_metrics = result.get("portfolio_metrics", {})
    component_analysis = result.get("component_analysis", {})
    correlation_analysis = result.get("correlation_analysis", {})
    stress_testing = result.get("stress_testing", {})
    individual_metrics = result.get("individual_metrics", {})
    
    html = f"""
    <div style='background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;'>
        <h2 style='color: #2c3e50; margin-top: 0; border-bottom: 2px solid #e74c3c; padding-bottom: 10px;'>
            ⚠️ Portfolio Risk Analysis Results
        </h2>
        
        <div style='background: #ffeaa7; border-left: 4px solid #fdcb6e; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #e17055; margin-top: 0;'>📊 Portfolio Risk Metrics</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
                <div><strong>Portfolio VaR (95%):</strong> {format_currency(portfolio_metrics.get('portfolio_var_95'))}</div>
                <div><strong>Portfolio Volatility:</strong> {format_percentage(portfolio_metrics.get('portfolio_volatility', 0) * 100)}</div>
                <div><strong>Diversification Ratio:</strong> {format_ratio(portfolio_metrics.get('diversification_ratio'))}</div>
                <div><strong>Assets Analyzed:</strong> {len(symbols)}</div>
            </div>
        </div>
        
        <div style='background: #fab1a0; border-left: 4px solid #e17055; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #d63031; margin-top: 0;'>🎯 Risk Contribution Analysis</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white;'>
                    <thead>
                        <tr style='background: #f8f9fa;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Asset</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Component VaR</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Risk Contribution %</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add component VaR breakdown
    component_var = component_analysis.get('component_var', {})
    risk_contribution = component_analysis.get('risk_contribution_pct', {})
    
    for i, symbol in enumerate(symbols):
        row_style = "background: #f8f9fa;" if i % 2 == 0 else ""
        comp_var = component_var.get(symbol, 0)
        risk_contrib = risk_contribution.get(symbol, 0)
        
        html += f"""
                        <tr style='{row_style}'>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'><strong>{symbol}</strong></td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_currency(comp_var)}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{risk_contrib:.1f}%</td>
                        </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div style='background: #fd79a8; border-left: 4px solid #e84393; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #d63031; margin-top: 0;'>🔗 Correlation Analysis</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
                <div><strong>Average Correlation:</strong> {format_ratio(correlation_analysis.get('average_correlation'))}</div>
            </div>
        </div>
        
        <div style='background: #ff7675; border-left: 4px solid #d63031; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #2d3436; margin-top: 0;'>🚨 Stress Testing Results</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white;'>
                    <thead>
                        <tr style='background: #f8f9fa;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Scenario</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Description</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Loss %</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Loss $</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add stress test results
    for i, (scenario, data) in enumerate(stress_testing.items()):
        row_style = "background: #f8f9fa;" if i % 2 == 0 else ""
        scenario_name = scenario.replace('_', ' ').title()
        
        html += f"""
                        <tr style='{row_style}'>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'><strong>{scenario_name}</strong></td>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'>{data.get('description', 'N/A')}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{data.get('portfolio_loss_pct', 0):.2f}%</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_currency(data.get('portfolio_loss_dollar', 0))}</td>
                        </tr>
        """
    
    html += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <div style='background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #0c5460; margin-top: 0;'>💬 Analysis Summary</h3>
            <p style='margin: 0; font-size: 16px;'>{result.get('message', 'Portfolio risk analysis completed successfully.')}</p>
        </div>
    </div>
    """
    
    return html

def format_single_asset_risk_results(result: dict) -> str:
    """Format single asset risk analysis results"""
    # Helper functions
    def format_currency(value):
        return f"${value:,.2f}" if value is not None else "N/A"
    
    def format_percentage(value):
        return f"{value:+.2f}%" if value is not None else "N/A"
    
    def format_ratio(value):
        return f"{value:.3f}" if value is not None else "N/A"
    
    symbol = result.get("symbol", "N/A")
    risk_metrics = result.get("risk_metrics", {})
    var_breakdown = result.get("var_breakdown", {})
    stress_testing = result.get("stress_testing", {})
    
    html = f"""
    <div style='background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;'>
        <h2 style='color: #2c3e50; margin-top: 0; border-bottom: 2px solid #e74c3c; padding-bottom: 10px;'>
            ⚠️ Single Asset Risk Analysis: {symbol}
        </h2>
        
        <div style='background: #ffeaa7; border-left: 4px solid #fdcb6e; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #e17055; margin-top: 0;'>📊 Core Risk Metrics</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
                <div><strong>VaR (95%):</strong> {format_percentage(risk_metrics.get('var_95_pct', 0))}</div>
                <div><strong>VaR Dollar Amount:</strong> {format_currency(risk_metrics.get('var_95_dollar'))}</div>
                <div><strong>Expected Shortfall:</strong> {format_currency(risk_metrics.get('expected_shortfall_95'))}</div>
                <div><strong>Annual Volatility:</strong> {format_percentage(risk_metrics.get('volatility_annual', 0) * 100)}</div>
                <div><strong>Max Drawdown:</strong> {format_percentage(risk_metrics.get('max_drawdown', 0) * 100)}</div>
            </div>
        </div>
        
        <div style='background: #fab1a0; border-left: 4px solid #e17055; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #d63031; margin-top: 0;'>📅 VaR Time Breakdown</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;'>
                <div><strong>Daily VaR:</strong> {format_percentage(var_breakdown.get('daily_var_95', 0))}</div>
                <div><strong>Weekly VaR:</strong> {format_percentage(var_breakdown.get('weekly_var_95', 0))}</div>
                <div><strong>Monthly VaR:</strong> {format_percentage(var_breakdown.get('monthly_var_95', 0))}</div>
            </div>
        </div>
        
        <div style='background: #ff7675; border-left: 4px solid #d63031; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #2d3436; margin-top: 0;'>🚨 Stress Testing Results</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white;'>
                    <thead>
                        <tr style='background: #f8f9fa;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Scenario</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Description</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Loss %</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Loss $</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add stress test results
    for i, (scenario, data) in enumerate(stress_testing.items()):
        row_style = "background: #f8f9fa;" if i % 2 == 0 else ""
        scenario_name = scenario.replace('_', ' ').title()
        
        html += f"""
                        <tr style='{row_style}'>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'><strong>{scenario_name}</strong></td>
                            <td style='padding: 10px; border-bottom: 1px solid #dee2e6;'>{data.get('description', 'N/A')}</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{data.get('loss_pct', 0):.2f}%</td>
                            <td style='padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;'>{format_currency(data.get('loss_dollar', 0))}</td>
                        </tr>
        """
    
    html += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <div style='background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0; border-radius: 4px;'>
            <h3 style='color: #0c5460; margin-top: 0;'>💬 Analysis Summary</h3>
            <p style='margin: 0; font-size: 16px;'>{result.get('message', 'Single asset risk analysis completed successfully.')}</p>
        </div>
    </div>
    """
    
    return html

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