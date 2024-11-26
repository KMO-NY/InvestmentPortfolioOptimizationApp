import streamlit as st

# Data handling and statistical analysis
import pandas as pd
from pandas_datareader import data
import numpy as np
from scipy import stats

# Data visualization
import plotly.express as px

# Financial data
import quantstats as qs
import ta
import yfinance as yf

st.title("INVESTMENT PORTFOLIO OPTIMISER")

# Initialize session state for storing stock entries
if "stock_list" not in st.session_state:
    st.session_state["stock_list"] = []  # List to store stock entries as dictionaries

# "Clear" button to reset form and session state
clear_col1, clear_col2 = st.columns([3,1])
if clear_col2.button("Clear Stock Data"):
    st.session_state["stock_list"] = []  # Clear session state for stocks

# Create a dict that has the extensions of the various countries
# Give one example of publicly traded companies from each listed country
exchanges = [
    {"country": "United States", "symbol": "AAPL", "suffix": ""},      # US (No suffix)
    {"country": "South Africa", "symbol": "AGL.JO", "suffix": ".JO"},  # JSE
]

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

# Generate the dictionary dynamically
country_suffix_map = validate_exchange_symbols(exchanges)

# Dropdown for country selection
selected_country = st.selectbox("Select Your Country:", options=list(country_suffix_map.keys()))

# Retrieve the suffix based on the selected country
selected_suffix = country_suffix_map[selected_country]

# dates
date_col1, date_col2 = st.columns(2)
start_date = date_col1.date_input("From", format = "DD/MM/YYYY")
end_date = date_col2.date_input("To", format = "DD/MM/YYYY")

# Creating a currency_list to add the various currencies for the chosen countries
currency_list = [
    {"country": "United States", "currency": "$"},      
    {"country": "South Africa", "currency": "R"}
]
usd = currency_list[0]["currency"]
zar = currency_list[1]["currency"]

# Section for user stock input
st.write("### Add Stocks and Amount Invested")

# Input fields for ticker and amount invested
with st.form("add_stock_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    ticker = col1.text_input("Enter the stock ticker symbol:")
    amount = col2.number_input("Amount Invested:", min_value=0.0, step=1.0, format="%.2f")
    add_stock = st.form_submit_button("Add Stock")
    
    if add_stock:
        if ticker and amount > 0:
            # Add stock entry to the list
            st.session_state["stock_list"].append({"Ticker": ticker.upper(), "Amount Invested": amount})
            st.success(f"Added {ticker} with amount {amount:.2f}.")
        else:
            st.error("Please enter a valid ticker symbol and investment amount.")

def get_stock_data(stock_list, 
                   country_mapping = country_suffix_map, 
                   start_date = start_date, 
                   end_date = end_date):
    """
    This function is used to retrive stock data for the chosen stocks
    Returns the value of the stock at market close.
    """
    stocks = [stock + selected_suffix for stock in stock_list]
    stock_data = yf.download(stocks, start=start_date, end=end_date)
    stock_data = stock_data['Close']
    return stock_data

# Display the current portfolio
if st.session_state["stock_list"]:
    st.write("Your Current Portfolio:")
    portfolio_df = pd.DataFrame(st.session_state["stock_list"])
    st.table(portfolio_df.style.format({"Amount Invested": "{:.2f}"}))      # formating so it shows up to 2 decimal places in the displayed df

# Button to finalize and show pie chart
if st.button("Load Stock Data"):
    if st.session_state["stock_list"]:
        # Generate pie chart
        fig = px.pie(
            portfolio_df,
            values="Amount Invested",
            names="Ticker",
            title="Portfolio Division",
            hole=0.4,  
        )
        st.plotly_chart(fig)
    else:
        st.error("You need to add at least one stock to view the portfolio division.")
