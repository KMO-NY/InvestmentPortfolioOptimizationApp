import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yfinance import Ticker
from datetime import datetime, timedelta
import quantstats as qs

# Functions I created
from utils import get_stock_data
from utils import load_session_state
from utils import calculate_portfolio_performance
from utils import generate_efficient_frontier
from utils import plot_efficient_frontier
from utils import tabulate_portfolio_info
from utils import suggested_portfolio_split

# Variables
from utils import zar

# Add Title
st.title("Portfolio Analysis")
st.sidebar.header("Navigation")
st.sidebar.write("Once You're Done With The Analysis, Head On Over to The Predictions Page!")

# Load session_state
ticker, weights, start_date, end_date = load_session_state()

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
            global stock_name
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

# Section: Efficient Frontier
st.header("Efficient Frontier")
st.write("Finding The Best Investment Portfolio For You!")

mean_returns = daily_returns_df.mean()
cov_matrix = daily_returns_df.cov()
results, weights_record = generate_efficient_frontier(mean_returns, cov_matrix, stock_data)
fig, max_sharpe_idx, min_vol_idx, sortino_idx = plot_efficient_frontier(results, weights_record, mean_returns, cov_matrix, stock_data)
st.pyplot(fig)
comparison_df = tabulate_portfolio_info(mean_returns, cov_matrix, stock_data, max_sharpe_idx, sortino_idx, min_vol_idx, weights_record, tickers= ticker)
st.table(comparison_df.iloc[:, :3])  # Display only the first three columns

# Section: Portfolio Split
st.write("### Suggested Portfolio Split")
cols1, cols2 = st.columns([1,2])
suggested_portfolio = cols1.selectbox("Select Your Preferred Ratio", options=["Max Sharpe Ratio", "Min Volatility", "Sortino Ratio"])

st.session_state["suggested_portfolio"] = suggested_portfolio

plot = suggested_portfolio_split(portfolio_table=comparison_df, tickers= ticker)
