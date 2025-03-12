import os
import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi

# Check if FAISS is installed
try:
    import faiss
except ImportError:
    st.error("FAISS is not installed. Please install it using `pip install faiss-cpu`.")
    st.stop()

# Function to download financial statements
def download_financial_statements(ticker):
    stock = yf.Ticker(ticker)
    
    # Get financial statements
    income_stmt = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T
    
    # Save as CSV files
    income_stmt_path = f"{ticker}_income_statement.csv"
    balance_sheet_path = f"{ticker}_balance_sheet.csv"
    cash_flow_path = f"{ticker}_cash_flow.csv"
    
    income_stmt.to_csv(income_stmt_path)
    balance_sheet.to_csv(balance_sheet_path)
    cash_flow.to_csv(cash_flow_path)
    
    return income_stmt_path, balance_sheet_path, cash_flow_path

# Test function
def test_download_financial_statements():
    ticker = "AAPL"  # Example ticker
    paths = download_financial_statements(ticker)
    
    print("Downloaded Files:")
    for path in paths:
        print(f"- {path}: Exists? {os.path.exists(path)}")
    
    print("\nComplete File Structure:")
    for root, dirs, files in os.walk("."):
        for file in files:
            print(os.path.join(root, file))

# Run test function
test_download_financial_statements()

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM (small open-source model)
llm_model = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(llm_model)
llm = AutoModelForCausalLM.from_pretrained(llm_model)

# Streamlit UI
st.title("Financial RAG Chatbot")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):")
if st.button("Download Financial Statements"):
    if ticker:
        income_stmt_path, balance_sheet_path, cash_flow_path = download_financial_statements(ticker)
        st.success(f"Financial statements for {ticker} downloaded!")
        st.write("Files:")
        st.write(f"- Income Statement: {income_stmt_path}")
        st.write(f"- Balance Sheet: {balance_sheet_path}")
        st.write(f"- Cash Flow Statement: {cash_flow_path}")
    else:
        st.warning("Please enter a valid stock ticker.")

query = st.text_input("Enter your financial question:")
if query:
    response = "This feature is under development."
    st.write("**Answer:**", response)
