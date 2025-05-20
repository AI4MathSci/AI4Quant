# QuantFinance

A web application designed to allow users to choose algorithms in quant finance areas for simulation and training. The system provides both simulation and backtesting capabilities for various trading strategies, with optional sentiment analysis integration.

## Features

- Multiple trading strategies (Simple Moving Average, Buy and Hold)
- Sentiment analysis integration (News and Reddit)
- Real-time simulation and historical backtesting
- Configurable parameters for strategy optimization
- Interactive web interface using Gradio
- RESTful API backend using FastAPI

## Codebase Structure

```
quantfin/
├── backend/
│   ├── config/         # Configuration management
│   ├── models/         # Trading and data models
│   ├── services/       # Business logic and external services
│   ├── utils/          # Utility functions
│   └── main.py         # Backend server entry point
├── frontend/
│   └── app.py          # Gradio interface
├── tests/              # Test files
├── .env.template       # Template for environment variables
├── .env               # Environment variables (not in git)
├── pyproject.toml     # Project dependencies and metadata
└── README.md          # This file
```

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver
- Reddit API credentials (for sentiment analysis)

## Installation and Execution

### Method 1: Using Virtual Environment

1. Clone the repository:
```bash
git clone https://github.com/AI4MathSci/AI4Quant.git
cd AI4Quant
```
2. Create a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install dependencies with uv:
```bash
uv pip install .
```
4. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Fill in the required credentials (see Configuration section)

5. Run the backend server:
```bash
python quantfin/backend/main.py 
```
6. Run the frontend server:
```bash
python quantfin/frontend/app.py 
```
### Method 2: Direct Execution

1. Clone the repository:
```bash
git clone https://github.com/AI4MathSci/AI4Quant.git
cd AI4Quant
```
2. Set up environment variables (see Configuration section)

3. Run the backend server:
```bash
uv run python quantfin/backend/main.py
```
4. Run the frontend server:
```bash
uv run python quantfin/frontend/app.py 
```

The frontend will be available at `http://localhost:7860` or `http://127.0.0.1:7860/`

**Note:** Each of the backend and frontend servers requires its own terminal window to run. If you prefer useing just one terminal, you can run the servers in the background by appending an "&" to each command. For example:
```bash
uv run python quantfin/backend/main.py &
```
Once both the backend and frontend servers are up, open your browser and navigate to `http://127.0.0.1:7860/` or `http://localhost:7860` to access the user interface and begin using the system.

When you are finished using the system, make sure to terminate both the backend and frontend servers. To do this, identify and stop the processes listening on port 8000 (backend server) and port 7860 (frontend server). This cleanup is important to ensure these ports are available the next time you need to use the system.

## Configuration

### Environment Variables

1. Copy the template file:
```bash
cp .env.template .env
```
2. Configure the following variables in `.env`:
```
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

### Getting Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..."
3. Fill in the required information:
   - Name: Your app name
   - Type: Script
   - Description: Optional
   - About URL: Optional
   - Redirect URI: http://localhost:8080
4. Click "create app"
5. Note down the credentials:
   - Client ID: The string under "personal use script"
   - Client Secret: The string labeled "secret"
   - User Agent: Format: "platform:app_name:version (by /u/username)"

## Usage

### Trading Strategies

1. **Simple Moving Average (SMA)**
   - Uses a moving average to generate buy/sell signals
   - Configurable window size
   - Optional sentiment analysis integration

2. **Buy and Hold**
   - Basic strategy that buys at the start and holds until the end
   - Useful as a benchmark

### Sentiment Analysis

The system integrates sentiment analysis from two sources:
1. News articles (using Yahoo Finance)
2. Reddit posts and comments

Sentiment analysis can be enabled/disabled and weighted in the trading decisions.

### Parameters

- Initial Capital: Starting investment amount
- Strategy: Trading strategy selection
- SMA Window: Period for moving average calculation
- Risk Tolerance: Low/Medium/High
- Asset Classes: Available investment options
- Risk Metrics: VaR or Expected Shortfall
- Sentiment Analysis:
  - Enable/Disable
  - Weight in decision making
  - Threshold for signal generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

TBD
