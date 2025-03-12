import os
import streamlit as st
import faiss
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

# Function to preprocess financial data
def preprocess_financial_data(financial_data):
    # Convert financial data into readable sentences
    sentences = []
    for index, row in financial_data.iterrows():
        # Use the correct column names
        date = row.get('Date', 'Unknown Date')  # Replace 'Date' with the correct column name
        revenue = row.get('Total Revenue', 'Unknown Revenue')  # Replace 'Total Revenue' with the correct column name
        net_income = row.get('Net Income', 'Unknown Net Income')  # Replace 'Net Income' with the correct column name
        
        sentence = f"In {date}, the company reported revenue of {revenue} and net income of {net_income}."
        sentences.append(sentence)
    return sentences

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM (small open-source model)
llm_model = "EleutherAI/gpt-neo-125M"  # More capable model
tokenizer = AutoTokenizer.from_pretrained(llm_model)
llm = AutoModelForCausalLM.from_pretrained(llm_model)

# Function to generate response using LLM
def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Ensure max_length is greater than input length
    if max_length <= input_length:
        max_length = input_length + 50  # Add some buffer for generation
    
    outputs = llm.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to retrieve relevant chunks using hybrid search
def hybrid_retrieval(query, chunks, top_k=3):
    # Dense retrieval with FAISS
    query_embedding = embed_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    dense_results = [chunks[i] for i in indices[0]]
    
    # Sparse retrieval with BM25
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    sparse_results = [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]
    
    # Combine results
    combined_results = list(set(dense_results + sparse_results))
    return combined_results[:top_k]

# Streamlit UI
st.title("Financial RAG Chatbot")

# Input for stock ticker
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

# Input for financial question
query = st.text_input("Enter your financial question:")
if query:
    # Load financial data (example: income statement)
    if ticker:
        income_stmt_path = f"{ticker}_income_statement.csv"
        if os.path.exists(income_stmt_path):
            # Read financial data
            financial_data = pd.read_csv(income_stmt_path)
            print(financial_data.columns)  # Debug: Print column names
            chunks = preprocess_financial_data(financial_data)
            
            # Build FAISS index
            chunk_embeddings = embed_model.encode(chunks)
            dimension = chunk_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(chunk_embeddings)
            
            # Retrieve relevant chunks
            retrieved_chunks = hybrid_retrieval(query, chunks, top_k=3)
            
            # Generate response using LLM
            max_context_length = 512  # Adjust based on the model's max sequence length
            context = " ".join(retrieved_chunks)[:max_context_length]
            prompt = f"""
            You are a financial analyst. Your task is
