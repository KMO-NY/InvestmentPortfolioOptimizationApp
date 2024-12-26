import streamlit as st

# Data handling and statistical analysis
import pandas as pd
from pandas_datareader import data
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Data visualization
import matplotlib.pyplot as plt

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
    if "risk_choice" not in st.session_state:
        st.session_state["risk_choice"] = 0      # Initialize risk_choice
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = None  # Initialize start_date
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = None  # Initialize end_date

def load_session_state():
    ticker, weights, risk_choice, start_date, end_date = [], [], 0, None, None

    if "portfolio_df" in st.session_state and not st.session_state["portfolio_df"].empty:
        portfolio_df = st.session_state["portfolio_df"]     # load portfolio_df
        ticker = portfolio_df["Ticker"]     # .tolist()
        weights = portfolio_df["Amount Invested"]       # .tolist()  

    if "risk_choice" in st.session_state:
        risk_choice = st.session_state["risk_choice"]

    if "start_date" in st.session_state:
        start_date = st.session_state["start_date"]  # load start_date
    if "end_date" in st.session_state:
        end_date = st.session_state["end_date"]  # load end_date

    return ticker, weights, risk_choice, start_date, end_date

# Clear session state for a different analysis
def clear_button_clicked():
    st.session_state["stock_list"] = []  
    st.session_state["portfolio_df"] = pd.DataFrame(columns=["Ticker", "Amount Invested"])
    st.session_state["start_date"] = None
    st.session_state["end_date"] = None

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

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, stock_data):
    """Calculate portfolio return and volatility."""
    trading_days = len(stock_data)
    portfolio_return = np.dot(weights, mean_returns) * trading_days        # multiply by number of trading days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    annualised_portfolio_return = ((1 + portfolio_return) ** (252 / trading_days)) - 1      # Scaling to make it easier when comparing + using Sharpe ratio 
    annualised_portfolio_volatility = portfolio_volatility * np.sqrt(252/ trading_days)
    return annualised_portfolio_return, annualised_portfolio_volatility

def generate_efficient_frontier(mean_returns, cov_matrix, stock_data, num_portfolios=50000, risk_free_rate=0):
    """
    Simulates random portfolios, and calculates their performance metrics.
    """
    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))  
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize weights
        weights_record.append(weights)
        
        # Calculate metrics
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix, stock_data)
        sharpe_ratio = (portfolio_return - risk_free_rate) - portfolio_volatility

        # Record results
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    return results, weights_record

import matplotlib.pyplot as plt

def plot_efficient_frontier(results, weights_record, mean_returns, cov_matrix, stock_data, risk_choice):
    """
    Plot the efficient frontier and highlight specific portfolios.
    
    Parameters:
        - results: Array from `generate_efficient_frontier` with volatility, return, and Sharpe ratio.
        - weights_record: List of portfolio weights from `generate_efficient_frontier`.
        - mean_returns: Expected returns of each stock.
        - cov_matrix: Covariance matrix of stock returns.
        - stock_data: DataFrame with historical stock prices.
        - risk_choice: Desired portfolio return input by the user.
    """
    # Unpack results
    volatilities = results[0]
    returns = results[1]
    sharpe_ratios = results[2]

    # Find portfolios with max Sharpe and min volatility
    max_sharpe_idx = np.argmax(sharpe_ratios)
    min_vol_idx = np.argmin(volatilities)

    # Find portfolio for user-defined return
    user_weights = None
    user_volatility = None
    for i, ret in enumerate(returns):
        if np.isclose(ret, risk_choice, atol=0.001):  # Find closest return
            user_weights = weights_record[i]
            user_volatility = volatilities[i]
            break

    # Plot efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')

    # Highlight portfolios
    if user_weights is not None:
        # Highlight User-specified portfolio
        plt.scatter(user_volatility, risk_choice, marker='*', color='orange', s=200, label='User Defined Return')
        plt.legend(loc='best')
        st.success(f"A portfolio with the desired return of {risk_choice:.2%} has been found and highlighted.")
    else:
        # Fallback to other portfolios
        plt.scatter(volatilities[max_sharpe_idx], returns[max_sharpe_idx], marker='*', color='red', s=200, label='Max Sharpe Ratio')
        plt.scatter(volatilities[min_vol_idx], returns[min_vol_idx], marker='*', color='blue', s=200, label='Min Volatility')

        # If Sortino Ratio is part of results
        sortino_idx = np.argmax(results[3]) if results.shape[0] > 3 else None
        if sortino_idx is not None:
            plt.scatter(volatilities[sortino_idx], returns[sortino_idx], marker='*', color='green', s=200, label='Max Sortino Ratio')

        plt.legend(loc='best')
        st.warning(f"No portfolio with the desired return of {risk_choice}% was found. Showing other options.")

    plt.show()

