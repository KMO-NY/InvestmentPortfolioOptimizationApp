import streamlit as st

# Data handling and statistical analysis
import pandas as pd
from pandas_datareader import data
import numpy as np
from scipy import stats

# Financial data
import quantstats as qs
import ta
import yfinance as yf

# Inititalizing the session_state:
def initialize_session_state():
    if "stock_list" not in st.session_state:
        st.session_state["stock_list"] = []  # Initialize stock_list
    if "portfolio_df" not in st.session_state:
        st.session_state["portfolio_df"] = pd.DataFrame(columns=["Ticker", "Amount Invested"])  # Initialize portfolio_df
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = None  # Initialize start_date
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = None  # Initialize end_date

def load_session_state():
    ticker, weights, start_date, end_date = [], [], None, None

    if "portfolio_df" in st.session_state and not st.session_state["portfolio_df"].empty:
        portfolio_df = st.session_state["portfolio_df"]     # load portfolio_df
        ticker = portfolio_df["Ticker"].tolist()
        weights = portfolio_df["Amount Invested"].tolist()  

    if "start_date" in st.session_state:
        start_date = st.session_state["start_date"]  # load start_date
    if "end_date" in st.session_state:
        end_date = st.session_state["end_date"]  # load end_date

    return ticker, weights, start_date, end_date

# Clear session state for a different analysis
def clear_button_clicked():
    st.session_state["stock_list"] = []  
    st.session_state["portfolio_df"] = pd.DataFrame(columns=["Ticker", "Amount Invested"])
    st.session_state["start_date"] = None
    st.session_state["end_date"] = None

# Function to verify stock symbols dynamically
def validate_exchange_symbols(exchange_list):
    """
    This functions uses the ticker to fetch infromation on the example companies given in the list of dictionaries.
    Once the company is verified, the ticker is deemed as valid. It is then linked to the Country/area

    Example:
    Anglo America is traded on the JSE with the ticker suffex '.JO', the function will find it.
    '.JO' will then be linked to South Africa --> {"South Africa": ".JO"}
    """
    country_suffix_map = {}
    for exchange in exchange_list:
        country = exchange["country"]
        symbol = exchange["symbol"]
        suffix = exchange["suffix"]

        try:
            ticker = yf.Ticker(symbol)
            if "longName" in ticker.info:  # Check if the stock data is valid
                country_suffix_map[country] = suffix
                print(f"Validated: {country} -> {suffix}")
            else:
                print(f"Could not validate: {country} -> {symbol}")
        except Exception as e:
            print(f"Error validating {country} with symbol {symbol}: {e}")

    return country_suffix_map

def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    """
    Calculate the Sharpe Ratio of a portfolio.
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_std_dev

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0):
    """
    Find the portfolio weights that maximize the Sharpe Ratio.
    """
    from scipy.optimize import minimize

    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1

    # Initial guess: equal weight for all assets
    initial_weights = num_assets * [1.0 / num_assets]

    # Minimize the negative Sharpe ratio to maximize it
    result = minimize(
        lambda w: -sharpe_ratio(w, *args),  # Negative Sharpe Ratio
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x  # Optimal weights

def sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sortino Ratio.
    
    Parameters:
    - returns: array-like, portfolio returns
    - risk_free_rate: float, risk-free rate (default is 0)

    Returns:
    - float, Sortino Ratio
    """
    excess_returns = returns - risk_free_rate
    downside_returns = np.where(excess_returns < 0, excess_returns, 0)  # Only negative returns
    downside_std = np.sqrt(np.mean(downside_returns**2))  # Downside deviation

    mean_excess_return = np.mean(excess_returns)
    return mean_excess_return / downside_std

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, stock_data):
    """Calculate portfolio return and volatility."""
    trading_days = len(stock_data.index)
    portfolio_return = np.dot(weights, mean_returns) * trading_days        # multiply by number of trading days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    return portfolio_return, portfolio_volatility

def generate_efficient_frontier(mean_returns, cov_matrix, stock_data, num_portfolios=5000, risk_free_rate=0):
    """
    Simulate random portfolios, calculate their performance metrics,
    and find the minimum volatility portfolio and Sortino Ratio.
    """
    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))  
    weights_record = []
    min_volatility = {"volatility": float("inf"), "weights": None}  # Track minimum volatility portfolio
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize weights
        weights_record.append(weights)
        
        # Calculate metrics
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix, stock_data)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        sortino_ratio = sortino_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

        # Record results
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
        results[3, i] = sortino_ratio

        # Check for minimum volatility portfolio
        if portfolio_volatility < min_volatility["volatility"]:
            min_volatility["volatility"] = portfolio_volatility
            min_volatility["weights"] = weights

    return results, weights_record, min_volatility

def get_daily_returns(tickers):
    try:
        # Fetch stock data for the last 2 days
        stock_data = yf.download(tickers, period="2d")
        
        if stock_data.empty or len(stock_data) < 2:
            raise ValueError("Insufficient data to calculate daily returns.")

        # Calculate percentage change
        daily_returns = stock_data['Close'].pct_change().iloc[-1]  # Get the latest daily return
        return daily_returns
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None
    
def get_stock_data(stock_list, 
                   start_date, 
                   end_date):
    """
    This function is used to retrive stock data for the chosen stocks
    Returns the percentage of daily_returns.
    """
    stocks = [stock + ".JO" for stock in stock_list]  
    try:
        stock_data = yf.download(stocks, start=start_date, end=end_date)["Close"]
        daily_returns_df = stock_data.pct_change()
        return stock_data, daily_returns_df
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")


# Create a dict that has the extensions of the various countries
# Give one example of publicly traded companies from each listed country
exchanges = [
    {"country": "South Africa", "symbol": "AGL.JO", "suffix": ".JO"},  # JSE
]

# Creating a currency_list to add the various currencies for the chosen countries
currency_list = [    
    {"country": "South Africa", "currency": "R"}
]
zar = currency_list[0]["currency"]

risk = {"5%": 5, "10%": 10, "15%": 15, "20%": 20}