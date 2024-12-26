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

# Functions I created
from utils import validate_exchange_symbols
from utils import initialize_session_state
from utils import clear_button_clicked

# Variables
from utils import exchanges
from utils import zar

st.title("Investment Portfolio Optimization")
st.write("### A Data Science Project by Kgosigadi Nyepele")
st.write("Use this page to enter data on your stock portfolio.")

st.sidebar.header("Navigation")
st.sidebar.write("Use the sidebar to navigate between pages. After loading your data, head on over to the Analysis page.")

# initialize session_state:
initialize_session_state()

# "Clear" button to reset form and session state
clear_col1, clear_col2 = st.columns([3,1])
if clear_col2.button("Clear Stock Data"):
    clear_button_clicked()

#  Generate the dictionary dynamically
country_suffix_map = validate_exchange_symbols(exchanges)

# Dropdown for country selection
selected_country = st.selectbox("Select Your Country:", options=list(country_suffix_map.keys()))

# Retrieve the suffix based on the selected country
selected_suffix = country_suffix_map[selected_country]

# dates
date_col1, date_col2 = st.columns(2)

start_date = date_col1.date_input("From:", format= "DD/MM/YYYY", value=st.session_state.get("start_date", None))
end_date = date_col2.date_input("To:", format= "DD/MM/YYYY", value=st.session_state.get("end_date", None))

if start_date is not None:
    st.session_state["start_date"] = start_date
if end_date is not None:
    st.session_state["end_date"] = end_date

# risk appetite
risk_choice = st.number_input("What Return Are You Aiming For?", min_value=0, max_value=100)

if risk_choice >= 1:
    st.write(f"Your Selected Portfolio Will Be Optimised To Give You {risk_choice}% Returns on Investment.")
else:
    st.write("Please Select Your Ideal Return On Investment For Your Portfolio.")

st.session_state["risk_choice"] = risk_choice       # Updating the chosen risk for the session.
               
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
            st.success(f"Added {ticker.upper()}{selected_suffix} with amount {zar}{amount:,.2f}".replace(",", " "))        # format amount for readbility
        else:
            st.error("Please enter a valid ticker symbol and investment amount.")
            
# Display the current portfolio
if st.session_state["stock_list"]:
    st.write("Your Current Portfolio:")
    # Update portfolio DataFrame in session state
    st.session_state["portfolio_df"] = pd.DataFrame(st.session_state["stock_list"])
    st.dataframe(st.session_state["portfolio_df"])
    portfolio_df = pd.DataFrame(st.session_state["portfolio_df"])      

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
