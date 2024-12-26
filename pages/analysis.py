import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yfinance import Ticker
from datetime import datetime, timedelta
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import EfficientFrontier, plotting


# Functions I created
from utils import get_stock_data
from utils import load_session_state
from utils import calculate_portfolio_performance
from utils import generate_efficient_frontier
from utils import plot_efficient_frontier

# Variables
from utils import zar

# Add Title
st.title("Portfolio Analysis")

# Load session_state
ticker, weights, risk_choice, start_date, end_date = load_session_state()
ticker = ticker.tolist()
if not ticker and not weights:
    st.error("No portfolio data available. Please add stocks on the main page.")
    st.stop()
elif start_date is None or end_date is None:
    st.error("Date range is not specified! Please set the start and end dates.")
    st.stop()

# Section: Daily Returns
st.header("Daily Returns")

try:
    # Load stock data and daily returns
    stock_data, daily_returns_df = get_stock_data(ticker, start_date, end_date)

    # Display metrics for each stock
    # Create a grid layout
    for i in range(0, len(ticker), 4):  # Step through tickers in groups of 4
        cols = st.columns(4)  # Create 4 columns per row
        for col, t in zip(cols, ticker[i:i+4]):  # Assign tickers to columns
            stock_name = t + ".JO"
            last_price = stock_data[stock_name].iloc[-1]  # Last available close price
            daily_return = daily_returns_df[stock_name].iloc[-1]  # Last daily return
            with col:
                st.metric(
                    label=f"{stock_name}", 
                    value=f"{zar}{last_price:.2f}", 
                    delta=f"{daily_return:.2%}"
                )
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
# test: st.dataframe(stock_data)
# Section: Efficient Frontier
st.header("Efficient Frontier")
# st.write("Coming soon")
# st.write("Coming soon: Maximize Sharpe Ratio & Sortino Ratio!")

mean_returns = daily_returns_df.mean()
cov_matrix = daily_returns_df.cov()
current_portfolio_returns, current_portfolio_volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix, stock_data)
results, weights_record = generate_efficient_frontier(mean_returns, cov_matrix, stock_data)
fig = plot_efficient_frontier(results, weights_record, mean_returns, cov_matrix, stock_data, risk_choice)
st.pyplot(fig)

# efficient_frontier = EfficientFrontier(mean_returns, cov_matrix, solver="OSQP")

# # Generate Efficient Frontier data
# def plot_efficient_frontier(mean_returns, cov_matrix):
#     """
#     Plot the efficient frontier.
#     """
#     try:
#         # Initialize Efficient Frontier with expected returns and covariance matrix
#         efficient_frontier = EfficientFrontier(mean_returns, cov_matrix)

#         # Calculate the Efficient Frontier
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plotting.plot_efficient_frontier(efficient_frontier, ax=ax, show_assets=True)
        
#         # Show the plot in Streamlit
#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f"Error plotting efficient frontier: {e}")

# Section: Predictions
st.header("Price Predictions")
# prediction_choice = st.radio(
#     "Select Prediction Horizon:",
#     ["1 Day", "1 Week", "1 Month"],
#     index=0
# )

# Placeholder for predictions (to be implemented)
# st.write(f"Displaying price predictions for {prediction_choice} horizon.")
st.write("Coming soon")

# Section: Portfolio Split
st.header("Portfolio Split")
# if "selected_ratio" in st.session_state:
#     selected_ratio = st.session_state["selected_ratio"]
#     st.write(f"Portfolio allocation based on {selected_ratio}.")
# else:
#     st.warning("No ratio selected. Using default allocation.")

# # Display Portfolio Split Pie Chart
# if not portfolio_df.empty:
#     st.write("Portfolio split by weights:")
#     st.write("Coming soon: Dynamic pie chart!")
st.write("Coming soon")