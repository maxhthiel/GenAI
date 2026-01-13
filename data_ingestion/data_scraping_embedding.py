# Imports für Data Scraping und Data Engineering
import os
import time
import pandas as pd
import requests
import yfinance as yf
import shutil  # Für das saubere Löschen alter DB-Stände
from dotenv import load_dotenv
from transformers import pipeline

# Langchain & Vector DB imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. INITIALISIERUNG
load_dotenv()
DATA_DIR = "/app/data"
CSV_PATH = os.path.join(DATA_DIR, "nasdaq_100_final_for_RAG.csv")
PERSIST_DIR = DATA_DIR 

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- TEIL 1: DATA ENGINEERING ---

def get_nasdaq_100_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    headers = { "User-Agent": "Mozilla/5.0" }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text)
    
    nasdaq_df = None
    for t in tables:
        if "Ticker" in t.columns and "Company" in t.columns:
            nasdaq_df = t
            break
            
    nasdaq_etf_df = pd.DataFrame([{
        "Ticker": "QQQ",
        "Company": "Invesco QQQ Trust",
        "ICB Subsector[14]": "ETF",
        "ICB Industry[14]": "ETF"
    }])
    return pd.concat([nasdaq_df, nasdaq_etf_df], ignore_index=True)

def get_yahoo_data(nasdaq_tickers):
    data_list = []
    exchange_rate = yf.Ticker("USDEUR=X").info.get("regularMarketPrice", 1.0)
    
    for ticker in nasdaq_tickers["Ticker"]:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5y", interval="1mo")
        
        one_year_return = volatility = avg_monthly_return = None
        if not hist.empty:
            if len(hist) >= 12:
                one_year_return = (hist["Close"].iloc[-1] - hist["Close"].iloc[-12]) / hist["Close"].iloc[-12]
            monthly_returns = hist["Close"].pct_change().dropna()
            volatility = monthly_returns.std()
            avg_monthly_return = monthly_returns.mean()

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
        time.sleep(0.1)
    
    df_yahoo = pd.DataFrame(data_list)
    cols_to_convert = ["Market Cap", "Current Price", "Previous Close", "Total Revenue"]
    df_yahoo[cols_to_convert] = df_yahoo[cols_to_convert] * exchange_rate
    return df_yahoo

def get_latest_news_sentiment(nasdaq_tickers):
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

# --- TEIL 2: DATA INGESTION ---

def row_to_document(row):
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
        "peg_ratio": row["PEG Ratio"], # Bleibt drin!
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
    # --- BEREINIGUNG DER ALTEN CHROMA DB ---
    print("Bereinige alte Datenbank-Einträge...")
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            # Löscht NUR die chroma.sqlite3 und die Index-Ordner (UUID-Namen)
            if item == "chroma.sqlite3" or os.path.isdir(item_path):
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    print(f"Gelöscht: {item}")
                except Exception as e:
                    print(f"Fehler beim Löschen von {item}: {e}")

    print("Starte Scraping...")
    tickers = get_nasdaq_100_tickers()
    yahoo_data = get_yahoo_data(tickers)
    news_data = get_latest_news_sentiment(tickers)
    
    final_df = pd.merge(yahoo_data, news_data, on="Ticker", how="outer")
    
    # KEIN DROP MEHR VON PEG RATIO!
    final_df = final_df.fillna("")
    final_df_str = final_df.astype(str)
    
    final_df.to_csv(CSV_PATH, index=False)
    print(f"CSV gespeichert: {CSV_PATH}")

    docs = [row_to_document(row) for i, row in final_df_str.iterrows()]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunked_docs, 
        embedding=embeddings, 
        collection_name="nasdaq_docs",
        persist_directory=PERSIST_DIR
    )
    db.persist()
    print(f"Datenbank in {PERSIST_DIR} erstellt.")

if __name__ == "__main__":
    run_ingestion()