import gradio as gr
import requests
import pandas as pd
from typing import Dict, Any, Optional
import logging
#from quantfin.backend.utils.data_loader import load_historical_data  # Import the data loader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend API URL (replace with your actual backend URL)
BACKEND_URL = "http://localhost:8000"  # Default, the backend should be running at port 8000

def get_server_health() -> str:
    """
    Checks the health of the backend server.

    Returns:
        str: "Online" if the server is healthy, "Offline" otherwise.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            data = response.json()
            if data and data.get("status") == "OK":
                return "Online"
            else:
                return "Offline - Invalid response"
        else:
            return f"Offline - Status Code: {response.status_code}"
    except requests.ConnectionError:
        return "Offline - Connection Error"
    except Exception as e:
        return f"Offline - Exception: {e}"

def run_simulation(model_name: str, parameters: Dict[str, Any], symbol: str = "AAPL") -> Dict[str, Any]:
    """
    Runs a simulation using the specified model and parameters via the backend.

    Args:
        model_name (str): The name of the model to simulate.
        parameters (Dict[str, Any]): The parameters for the simulation.
        symbol (str): The stock symbol to use for sentiment analysis. Defaults to "AAPL".

    Returns:
        Dict[str, Any]: The results of the simulation from the backend.
    """
    try:
        logger.info(f"Sending simulation request to backend for model: {model_name} with parameters: {parameters} and symbol: {symbol}")
        response = requests.post(f"{BACKEND_URL}/simulate", json={
            "model_name": model_name,
            "parameters": parameters,
            "symbol": symbol
        })
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with backend: {e}")
        return {"error": f"Failed to communicate with backend: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

def run_backtest(model_name: str, parameters: Dict[str, Any], symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Sends a backtest request to the backend with the specified parameters.

    Args:
        model_name (str): The name of the model to backtest.
        parameters (Dict[str, Any]): The parameters for the backtest.
        symbol (str): The stock symbol.
        start_date (str): The start date for historical data.
        end_date (str): The end date for historical data.

    Returns:
        Dict[str, Any]: The results of the backtest from the backend.
    """

    try:
        logger.info(f"Sending backtest request to backend for model: {model_name} with parameters: {parameters} and symbol: {symbol}, start_date: {start_date}, end_date: {end_date}")

        response = requests.post(f"{BACKEND_URL}/backtest", json={
            "model_name": model_name,
            "parameters": parameters,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        })
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with backend: {e}")
        return {"error": f"Failed to communicate with backend: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

def create_gradio_interface():
    """
    Creates the Gradio interface for the quantitative finance system.
    """
    with gr.Blocks() as iface:
        available_models = ["Algorithmic Trading Model", "Portfolio Management Model", "Risk Management Model"]

        # Define input components
        model_selection = gr.Dropdown(choices=available_models, label="Select Model")
        mode_input = gr.Radio(
            choices=["Simulation", "Backtest"],
            label="Select Mode",
            value="Simulation"
        )
        initial_capital_input = gr.Number(value=100000, label="Initial Capital")
        strategy_input = gr.Textbox(label="Strategy (e.g., Simple Moving Average, Buy and Hold, Equal Weight)")
        sma_window_input = gr.Number(value=20, label="SMA Window (for SMA strategy)")
        risk_tolerance_input = gr.Dropdown(choices=["Low", "Medium", "High"], label="Risk Tolerance")
        asset_classes_input = gr.CheckboxGroup(choices=["Stocks", "Bonds", "Cash", "Real Estate"], label="Asset Classes")
        risk_metric_input = gr.Dropdown(choices=["Value at Risk (VaR)", "Expected Shortfall (ES)"], label="Risk Metric")
        confidence_level_input = gr.Number(value=0.95, label="Confidence Level")
        holding_period_input = gr.Number(value=10, label="Holding Period")
        
        # Add sentiment analysis inputs
        use_sentiment_input = gr.Checkbox(label="Use Sentiment Analysis")
        sentiment_weight_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.1, label="Sentiment Weight")
        sentiment_threshold_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Sentiment Threshold")
        
        symbol_input = gr.Textbox(label="Stock Symbol for Data Loading (e.g., AAPL, MSFT)")
        start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", visible=False)
        end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", visible=False)

        # Define output components
        output_text = gr.JSON(label="Output")
        server_status_text = gr.Textbox(label="Server Status", value=get_server_health())

        # Function to update date visibility
        def update_date_visibility(mode):
            is_visible = mode == "Backtest"
            return gr.update(visible=is_visible), gr.update(visible=is_visible)

        # Update visibility when mode changes
        mode_input.change(
            fn=update_date_visibility,
            inputs=[mode_input],
            outputs=[start_date_input, end_date_input]
        )

        # Define function to handle button click
        def run_model(model_name: str, mode: str, initial_capital: float, strategy: str, sma_window: int, 
                      risk_tolerance: str, asset_classes: list, risk_metric: str, confidence_level: float, 
                      holding_period: int, use_sentiment: bool, sentiment_weight: float, 
                      sentiment_threshold: float, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
            
            parameters = {
                "initial_capital": initial_capital,
                "strategy": strategy,
                "sma_window": sma_window,
                "risk_tolerance": risk_tolerance,
                "asset_classes": asset_classes,
                "risk_metric": risk_metric,
                "confidence_level": confidence_level,
                "holding_period": holding_period,
                "use_sentiment": use_sentiment,
                "sentiment_weight": sentiment_weight,
                "sentiment_threshold": sentiment_threshold
            }

            if mode == "Backtest":
                if not all([symbol, start_date, end_date]):
                    return {"error": "Backtest mode requires symbol, start date, and end date"}
                return run_backtest(model_name, parameters, symbol, start_date, end_date)
            else:  # Simulation mode
                return run_simulation(model_name, parameters, symbol)

        # Add submit button
        submit_btn = gr.Button("Run")
        submit_btn.click(
            fn=run_model,
            inputs=[
                model_selection,
                mode_input,
                initial_capital_input,
                strategy_input,
                sma_window_input,
                risk_tolerance_input,
                asset_classes_input,
                risk_metric_input,
                confidence_level_input,
                holding_period_input,
                use_sentiment_input,
                sentiment_weight_input,
                sentiment_threshold_input,
                symbol_input,
                start_date_input,
                end_date_input
            ],
            outputs=output_text
        )

    return iface

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)  # add "share=True" to create a public link