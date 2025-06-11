from typing import Dict, Any
import empyrical

def get_risk_strategy(strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    strategies = {
        "Value at Risk (VaR)": lambda returns, portfolio_value, confidence_level: {
            "var": empyrical.value_at_risk(returns, cutoff=1-confidence_level),
            "risk_value": portfolio_value * abs(empyrical.value_at_risk(returns, cutoff=1-confidence_level)),
        },
        "Expected Shortfall (ES)": lambda returns, portfolio_value, confidence_level: {
            "es": empyrical.expected_shortfall(returns, cutoff=1-confidence_level),
            "risk_value": portfolio_value * abs(empyrical.expected_shortfall(returns, cutoff=1-confidence_level)),
        },
    }
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategies[strategy_name](params.get("returns"), params.get("portfolio_value"), params.get("confidence_level")) 