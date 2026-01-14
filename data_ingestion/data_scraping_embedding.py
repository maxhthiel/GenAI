# IMPORTS 
# Standard library imports
import os
import time
import shutil                                                       # used for clean deletion of old database states

# Data handling and scraping
import pandas as pd
import requests
import yfinance as yf

# Environment variable handling (e.g. openAI API key)
from dotenv import load_dotenv

# Hugging face pipeline for financial sentiment analysis
from transformers import pipeline

# Langchain & Vector DB imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# INITIALIZATION & CONFIGURATION

load_dotenv()                                                       # load environment variable form .env file
DATA_DIR = "/app/data"                                              # Base directory inside the Docker container
CSV_PATH = os.path.join(DATA_DIR, "nasdaq_100_final_for_RAG.csv")   # csv output path
PERSIST_DIR = DATA_DIR                                              # Directory where Chroma will persist its vector database

# ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# PART 1: DATA ENGINEERING

def get_nasdaq_100_tickers():
    """
    Scrapes the NASDAQ-100 constituents table form Wikipedia.
    Additionally appends the QQQ ETF as a synthetic entry.
    """

    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    headers = { "User-Agent": "Mozilla/5.0" }
    response = requests.get(url, headers=headers)
    # Wikipedia pages often contain multiple tables
    tables = pd.read_html(response.text)
    
    # Identify the table containing "Ticker" and "Company"
    nasdaq_df = None
    for t in tables:
        if "Ticker" in t.columns and "Company" in t.columns:
            nasdaq_df = t
            break
            
    # Manually add the NASDAQ-100 ETF (QQQ)
    nasdaq_etf_df = pd.DataFrame([{
        "Ticker": "QQQ",
        "Company": "Invesco QQQ Trust",
        "ICB Subsector[14]": "ETF",
        "ICB Industry[14]": "ETF"
    }])

    # Combine index constituents and ETF
    return pd.concat([nasdaq_df, nasdaq_etf_df], ignore_index=True)

def get_yahoo_data(nasdaq_tickers):
    """
    Retrieves financial, valuation and historical price data for each NASDAQ-100 ticker using Yahoo Finance.
    """
    data_list = []
    # Exchange rate USD -> EUR
    exchange_rate = yf.Ticker("USDEUR=X").info.get("regularMarketPrice", 1.0)
    
    for ticker in nasdaq_tickers["Ticker"]:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5y", interval="1mo")
        
        # Initialize metrics
        one_year_return = volatility = avg_monthly_return = None
        # Calculate historical performance metrics
        if not hist.empty:
            if len(hist) >= 12:
                one_year_return = (hist["Close"].iloc[-1] - hist["Close"].iloc[-12]) / hist["Close"].iloc[-12]
            monthly_returns = hist["Close"].pct_change().dropna()
            volatility = monthly_returns.std()
            avg_monthly_return = monthly_returns.mean()

        # Collect relevant fields
        data_list.append({
            "Ticker": ticker,
            "Company": nasdaq_tickers.loc[nasdaq_tickers["Ticker"] == ticker, "Company"].values[0],
            "Long Business Summary": info.get("longBusinessSummary"),
            "Sector (Yahoo)": info.get("sector"),
            "Industry (Yahoo)": info.get("industry"),
            "Country": info.get("country"),
            "Market Cap": info.get("marketCap"),
            "Current Price": info.get("currentPrice"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "Average Monthly Return": avg_monthly_return,
            "Previous Close": info.get("previousClose"),
            "Dividend Yield": info.get("dividendYield"),
            "PE Ratio": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Price to Book": info.get("priceToBook"),
            "Total Revenue": info.get("totalRevenue"),
            "Debt to Equity": info.get("debtToEquity"),
            "ROE": info.get("returnOnEquity"),
            "1y Return": one_year_return,
            "Volatility": volatility,
            "Website": info.get("website")
        })
        # Avoid rate limiting
        time.sleep(0.1)
    
    df_yahoo = pd.DataFrame(data_list)

    # Convert monetary values to EUR
    cols_to_convert = ["Market Cap", "Current Price", "Previous Close", "Total Revenue"]
    df_yahoo[cols_to_convert] = df_yahoo[cols_to_convert] * exchange_rate

    return df_yahoo

def get_latest_news_sentiment(nasdaq_tickers):
    """
    Retrieves the most recent news article per ticker and performs financial sentiment analysis using FinBERT.
    """
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    results_list = []

    for ticker in nasdaq_tickers["Ticker"]:
        stock = yf.Ticker(ticker)
        news = stock.news

        if news:
            content = news[0]["content"]
            news_summary = content.get("summary", "")
            title = content.get("title", "")
            link = content.get("canonicalUrl", {}).get("url", "")

            # Perform sentiment analysis (max 512 tokens)
            result = sentiment_pipeline(news_summary[:512])[0] if news_summary else {"label": "neutral", "score": 0.0}

            results_list.append({
                "News Summary": news_summary,
                "Ticker": ticker,
                "Latest_News_Title": title,
                "Latest_News_Link": link,
                "Sentiment": result["label"],
                "Confidence": result["score"]
            })

        time.sleep(0.1)

    return pd.DataFrame(results_list)

# PART 2: DATA INGESTION

def row_to_document(row):
    """
    Converts a single row of the financial dataframe into a LangChain Document with rich metadata.
    """
    text = f"""
    Company: {row["Company"]}  
    Ticker: {row["Ticker"]}

    Business Summary: {row["Long Business Summary"]}

    Sector: {row["Sector (Yahoo)"]}
    Industry: {row["Industry (Yahoo)"]}
    Country: {row["Country"]}

    Latest News:
    Title: {row["Latest_News_Title"]}
    Summary: {row["News Summary"]}
    Sentiment: {row["Sentiment"]}
"""
    metadata = {
        "ticker": row["Ticker"],
        "company": row["Company"],
        "sector": row["Sector (Yahoo)"],
        "industry": row["Industry (Yahoo)"],
        "country": row["Country"],
        "market_cap": row["Market Cap"],
        "current_price": row["Current Price"],
        "previous_close": row["Previous Close"],
        "52_week_high": row["52 Week High"],
        "avg_monthly_return": row["Average Monthly Return"],
        "dividend_yield": row["Dividend Yield"],
        "pe_ratio": row["PE Ratio"],
        "forward_pe": row["Forward PE"],
        "peg_ratio": row["PEG Ratio"], 
        "price_to_book": row["Price to Book"],
        "total_revenue": row["Total Revenue"],
        "debt_to_equity": row["Debt to Equity"],
        "roe": row["ROE"],
        "return_1y": row["1y Return"],
        "volatility": row["Volatility"],
        "sentiment": row["Sentiment"],
        "confidence": row["Confidence"],
        "news_link": row["Latest_News_Link"],
        "latest_news_title": row["Latest_News_Title"],
        "news_summary": row["News Summary"],
        "website": row["Website"], 
    }

    return Document(page_content=text, metadata=metadata)

def run_ingestion():
    """
    Executes the full ingestion pipeline:
    - Cleans previous vector Db
    - Scrapes financial & news data
    - Creates embeddings
    - Persists Chroma vector database
    """
    # Clean up old database entries
    print("Deleting old database entries...")
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            # Delete ONLY the chroma.sqlite3 and the index folders (UUID names)
            if item == "chroma.sqlite3" or os.path.isdir(item_path):
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    print(f"Successfuly deleted: {item}")
                except Exception as e:
                    print(f"Error deleting {item}: {e}")

    # Merge Yahoo Finance data with news sentiment
    print("Starting Scraping...")
    tickers = get_nasdaq_100_tickers()
    yahoo_data = get_yahoo_data(tickers)
    news_data = get_latest_news_sentiment(tickers)
    
    final_df = pd.merge(yahoo_data, news_data, on="Ticker", how="outer")
    
    # to ensure robust ingestion -> all missing values are replaced with empty strings 
    final_df = final_df.fillna("")
    # and dataset is explicitly cast to string format
    final_df_str = final_df.astype(str)
    
    # CSV persistence for reproducibility
    final_df.to_csv(CSV_PATH, index=False)
    print(f"CSV saved: {CSV_PATH}")

    docs = [row_to_document(row) for i, row in final_df_str.iterrows()]
    
    # Chunking for LLM context windows
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
            
    # embedding + vector storage
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(
        documents=chunked_docs, 
        embedding=embeddings, 
        collection_name="nasdaq_docs",
        persist_directory=PERSIST_DIR
    )

    db.persist()
    print(f"Successfully created database in {PERSIST_DIR} .")

# script entry point-> ensures the pipeline runs only when executed directly, not when imported
if __name__ == "__main__":
    run_ingestion()