# QuantFinance Platform

A comprehensive quantitative finance platform with **AI-powered sentiment analysis** and **professional-grade backtesting**. The system provides advanced trading strategy simulation, parameter optimization, and sentiment-enhanced decision making using state-of-the-art LLM technology.

## 🚀 Key Features

### 📈 **Advanced Trading Strategies**
- **Simple Moving Average (SMA)** with configurable windows
- **RSI (Relative Strength Index)** with customizable overbought/oversold levels  
- **Bollinger Bands** with adjustable periods and deviations
- **Real-time simulation** and **historical backtesting**

### 🧠 **AI-Powered Sentiment Analysis**
- **Temporal sentiment analysis** with date-specific historical context
- **Multi-source analysis**: SEC filings, Alpha Vantage news, Yahoo Finance RSS
- **Trading days alignment** using NYSE calendar for accurate market timing
- **Global sentiment caching** for efficient optimization
- **LLM-driven sentiment parsing** with confidence scoring
- **Risk factor and catalyst analysis**

### 🎯 **Professional Parameter Optimization**
- **Split-sample methodology** (training/testing data separation)
- **Six optimization speed levels** (lightning to exhaustive)
- **Grid search across multiple dimensions** (200-800+ combinations)
- **Sentiment vs technical-only comparison**
- **Out-of-sample validation** for unbiased results
- **Robust confidence scoring** and recommendations

### 🎨 **Enhanced User Experience**
- **Modern responsive UI** with professional styling
- **Real-time progress indicators** during optimization
- **Comprehensive result displays** with financial metrics
- **Side-by-side performance comparison**
- **Detailed parameter recommendations**
- **Professional window sizing** and visual hierarchy

### 🔧 **Enterprise-Grade Features**
- **RESTful API backend** using FastAPI
- **Efficient data caching** for sentiment analysis
- **Error handling and validation**
- **Debug logging** throughout the system
- **Configurable optimization intensity**

## 🧰 **Technology Stack**

This project is built on top of exceptional open-source libraries. We give full credit to these amazing projects:

### **📈 Core Trading & Backtesting**
- **[BackTrader](https://github.com/mementum/backtrader)** - Professional Python backtesting library for trading strategy development
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Reliable Yahoo Finance data downloader

### **📊 Financial Analysis & Optimization**
- **[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)** - Portfolio optimization using modern portfolio theory
- **[Empyrical](https://github.com/quantopian/empyrical)** - Financial risk and performance metrics library

### **🧠 AI & Document Processing**
- **[LlamaIndex](https://github.com/run-llama/llama_index)** - Advanced LLM-powered document processing and RAG framework
- **[OpenAI](https://github.com/openai/openai-python)** - OpenAI Python client for GPT-4 integration

### **📡 Data & Web Services**
- **[FastAPI](https://github.com/tiangolo/fastapi)** - Modern, high-performance web framework for APIs
- **[Gradio](https://github.com/gradio-app/gradio)** - Fast, easy-to-use web UI for machine learning
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - HTML/XML parsing for web scraping
- **[Feedparser](https://github.com/kurtmckee/feedparser)** - RSS/Atom feed parsing
- **[HTTPX](https://github.com/encode/httpx)** - Modern HTTP client for async operations
- **[Requests](https://github.com/psf/requests)** - HTTP library for Python

**Special thanks to all the maintainers and contributors of these projects for making quantitative finance accessible to everyone!**

## 🛠️ Prerequisites

- **Python 3.10+**
- **OpenAI API Key** (for sentiment analysis)
- **Alpha Vantage API Key** (for historical news and SEC filings)
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **Internet connection** (for real-time data and sentiment analysis)


## 🏗️ System Architecture

```
QuantFinance/
├── quantfin/
│   ├── backend/
│   │   ├── models/                             # Core business logic
│   │   │   ├── trading_model.py                # Enhanced trading with temporal sentiment optimization
│   │   │   ├── portfolio_model.py              # Portfolio management
│   │   │   └── risk_model.py                   # Risk analysis
│   │   ├── analysis/                           # AI-powered sentiment analysis
│   │   │   └── llamaindex_engine.py            # LlamaIndex integration for document processing
│   │   ├── strategies/                         # Strategies implementation
│   │   │   ├── trading_strategy_factory.py     # Algorithmic trading strategies with sentiment integration
│   │   │   ├── portfolio_strategy_factory.py   # Portfolio management strategies
│   │   │   └── risk_strategy_factory.py        # Risk management strategies
│   │   ├── config/                             # Configuration management
│   │   │   └── config.py                       # Application configuration settings
│   │   ├── main.py                             # FastAPI server
│   ├── frontend/
│   │   └── app.py                              # Enhanced Gradio interface with sentiment controls
│   └── .env (.env.template)                    # Environment variables and API keys
├── pyproject.toml                              # Project dependencies and build configuration
└── README.md                                   # This documentation
```

## ⚡ Quick Start


### Method 1: Direct Execution

1. **Clone repository**:
```bash
git clone https://github.com/AI4MathSci/AI4Quant.git
cd AI4Quant/QuantFinance
```
2. **Configure environment** (see [Configuration](#-configuration) section below):

3. **Launch servers**:
```bash
# Terminal 1 - Backend
uv run python quantfin/backend/main.py

# Terminal 2 - Frontend
uv run python quantfin/frontend/app.py
```

### Method 2: Virtual Environment Setup

1. **Clone and setup**:
```bash
git clone https://github.com/AI4MathSci/AI4Quant.git
cd AI4Quant/QuantFinance
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install .
```

2. **Configure environment** (see [Configuration](#-configuration) section below):

3. **Launch servers**:
```bash
# Terminal 1 - Backend
python quantfin/backend/main.py

# Terminal 2 - Frontend  
python quantfin/frontend/app.py
```

Both methods will set up a Python virtual environment in the `.venu` directory. This environment includes its own Python interpreter and an independent set of installed packages for your project, preventing dependency conflicts with other projects. In Method 1, the `.venu` directory is automatically created after you run the commands to start the servers.

**Note:** Each of the backend and frontend servers requires its own terminal window to run. If you prefer useing just one terminal, you can run the servers in the background by appending an "&" to each command. For example:
```bash
uv run python quantfin/backend/main.py &
```
**Access the platform**: Once both the backend and frontend servers are up, open your browser and navigate to `http://127.0.0.1:7860/` or `http://localhost:7860` to access the user interface and begin using the system.

**Cleanup**: When you are finished using the system, make sure to terminate both the backend and frontend servers. To do this, identify and stop the processes listening on port 8000 (backend server) and port 7860 (frontend server). This cleanup is important to ensure these ports are available the next time you need to use the system.


## 🔧 Configuration

### Environment Variables Setup

1. **Create configuration file**:

```bash
cp .env.template .env
```
2. **Configure OpenAI API**:
```env
# Edit .env with your OpenAI API key

# Required for sentiment analysis
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required for historical news and SEC filings (free tier: 500 calls/day)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Optional: Model configuration
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
```

### OpenAI API Key

1. Visit [OpenAI API Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Add billing information (required for API usage)
6. Copy the key to your `.env` file

### Alpha Vantage API Key
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Get your API key (free tier: 500 calls/day)
4. Copy the key to your `.env` file

**Note**: Without an OpenAI API key, sentiment analysis will be disabled, but technical trading strategies will work normally. Alpha Vantage API key is optional and provides additional historical news data.

## 📊 Trading Features

### 🎯 **Optimization Methodology**

Our **split-sample optimization** provides professional-grade parameter tuning:

1. **Data Splitting**: Automatically divides historical data into training (70%) and testing (30%) periods
2. **Parameter Grid Search**: Tests hundreds of parameter combinations systematically  
3. **Dual Strategy Testing**: Compares sentiment-enhanced vs technical-only approaches
4. **Out-of-Sample Validation**: Ensures results are not overfitted to historical data
5. **Confidence Scoring**: Provides reliability metrics for recommendations

### 🏃 **Optimization Speed Levels**

| Speed Level | Combinations | Duration | Use Case |
|-------------|-------------|----------|----------|
| **Lightning** | ~18 | 30 sec | Quick testing |
| **Fast** | ~48 | 1 min | Rapid iteration |
| **Medium** | ~120 | 2-3 min | Balanced approach |
| **Balanced** | ~240 | 4-6 min | Thorough testing |
| **Thorough** | ~400 | 8-12 min | Comprehensive analysis |
| **Exhaustive** | 800+ | 15-30+ min | Maximum precision |

*Note: Parameter combinations automatically optimized to fit within configurable limits (default: 400)*

### 📈 **Trading Strategies**

#### **Simple Moving Average (SMA)**
- **Parameters**: Window size (5-50 periods)
- **Signals**: Buy when price crosses above SMA, sell when below
- **Optimization**: Automatically finds optimal window for your data

#### **RSI (Relative Strength Index)**
- **Parameters**: Period (5-30), Overbought (60-90), Oversold (10-40)
- **Signals**: Buy when RSI < oversold, sell when RSI > overbought
- **Optimization**: Tunes all three parameters simultaneously

#### **Bollinger Bands**
- **Parameters**: Period (10-30), Standard deviation (1.0-3.0)
- **Signals**: Buy at lower band, sell at upper band
- **Optimization**: Finds optimal band configuration

### 🧠 **Sentiment Analysis Integration**

#### **Analysis Scopes**
- **News**: Recent financial news articles with temporal filtering
- **Comprehensive**: News + SEC filings + Alpha Vantage data
- **Filings**: SEC documents only (10-K, 10-Q, etc.) with date-specific analysis

#### **Advanced Features**
- **Risk Factor Analysis**: Company, industry, and market risks
- **Catalyst Analysis**: Potential positive/negative price drivers
- **Confidence Thresholds**: Minimum confidence for sentiment signals
- **Weight Configuration**: Balance between sentiment and technical signals

#### **Performance Optimizations**
- **Temporal alignment**: Date-specific sentiment analysis for historical accuracy
- **Efficient caching**: Global sentiment data caching reduces API calls by 90%
- **Period-based slicing**: Intelligent sentiment data handling for optimization periods
- **Trading days integration**: Proper market calendar alignment for backtesting

#### **LLM Processing**
- **Intelligent parsing**: Extracts sentiment from complex financial documents
- **Confidence scoring**: Provides reliability metrics for each analysis
- **Caching system**: Efficient reuse of sentiment data

## 🎨 User Interface

### 📊 **Results Display**

#### **Optimization Results Section**
- **Recommended configuration** with optimal parameters
- **Performance confidence** based on testing methodology
- **Parameter combination** count and testing duration
- **Clear reasoning** for recommendations

#### **Financial Performance Section**
- **Initial vs Final portfolio values** with optimal parameters
- **Profit/Loss calculations** in dollars and percentages
- **Side-by-side comparison** of sentiment vs technical approaches
- **Training vs out-of-sample** performance metrics

#### **Advanced Features**
- **Real-time progress** indicators during optimization
- **Professional styling** with proper typography and spacing
- **Responsive design** that works on all screen sizes
- **Detailed JSON results** (toggle-able for advanced users)

## 🚀 Usage Examples

### Basic Trading Strategy Test
1. Select a stock symbol (e.g., AAPL)
2. Choose a strategy (e.g., Simple Moving Average)
3. Set date range (e.g., last 1 year)
4. Run backtest with current parameters

### Advanced Parameter Optimization
1. Enable "Parameter Optimization" 
2. Select optimization speed (Medium recommended)
3. Enable "Compare Sentiment vs Technical-Only"
4. Set training/test split ratio (70% default)
5. Run enhanced backtest

### Sentiment-Enhanced Trading
1. Configure OpenAI and Alpha Vantage API keys in `.env`
2. Enable "Sentiment Analysis" with temporal alignment
3. Choose analysis scope (Comprehensive recommended)
4. Set sentiment weight and API frequency controls
5. Enable risk/catalyst analysis for deeper insights

### 📊 **Sentiment Window Scheme**

The sentiment window system provides a sophisticated approach to market sentiment analysis:

#### **Window Configuration**
- **Time Window**: Configurable period for sentiment analysis (default: 30 days)
- **Document Types**: News articles, SEC filings, earnings reports
- **Update Frequency**: Real-time updates with configurable intervals

#### **Analysis Features**
- **Caching System**: Efficient storage and retrieval of sentiment results
- **Confidence Scoring**: Reliability metrics for each sentiment signal
- **Weight Integration**: Configurable influence on trading decisions

#### **Performance Optimization**
- **Batch Processing**: Efficient handling of multiple documents
- **Memory Management**: Automatic cleanup of old sentiment data
- **API Cost Optimization**: Minimizes redundant API calls

## 🤝 Contributing

- Report bugs via [GitHub Issues](https://github.com/AI4MathSci/AI4Quant/issues)
- Follow standard GitHub practices: create a fork of the repository, work on your changes in a dedicated branch, and submit a pull request when you're ready.


## 📄 License

This project is licensed under the MIT License.


## 🎯 Roadmap

- [ ] **Machine Learning Models**: LSTM, Random Forest integration
- [ ] **Real-time Trading**: Live market data streaming
- [ ] **Advanced Risk Metrics**: VaR, CVaR, Sharpe ratio optimization
- [ ] **Multi-asset Portfolios**: Cross-asset optimization
- [ ] **Cloud Deployment**: Docker containers and cloud hosting
- [ ] **API Webhooks**: External system integration

---

**Built with ❤️ for quantitative finance professionals and enthusiasts**