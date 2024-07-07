import mwclient
import time
import pandas as pd
import yfinance as yf
from transformers import pipeline
from statistics import mean
from datetime import datetime
import os

cryptos = ["BTC-USD", "ETH-USD", "DOGE-USD", "USDT-USD", "BNB-USD"]


sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def fetch_wikipedia_data(crypto):
    site = mwclient.Site('en.wikipedia.org')
    page = site.pages[crypto.split('-')[0]]
    revs = list(page.revisions())
    revs = sorted(revs, key=lambda rev: rev["timestamp"])
    edits = {}

    for rev in revs:
        date = time.strftime("%Y-%m-%d", rev["timestamp"])
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count=0)

        edits[date]["edit_count"] += 1
        comment = rev.get("comment", "")
        if comment:
            sent = sentiment_pipeline(comment[:250])[0]
            score = sent["score"]
            if sent["label"] == "NEGATIVE":
                score *= -1
            edits[date]["sentiments"].append(score)

    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
        else:
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0
        del edits[key]["sentiments"]

    edits_df = pd.DataFrame.from_dict(edits, orient="index")
    edits_df.index = pd.to_datetime(edits_df.index)
    return edits_df

def fetch_financial_data(crypto):
    ticker = yf.Ticker(crypto)
    if os.path.exists(f"{crypto}.csv"):
        data = pd.read_csv(f"{crypto}.csv", index_col=0, parse_dates=True)
    else:
        data = ticker.history(period="max")
        data.to_csv(f"{crypto}.csv")
    data.index = pd.to_datetime(data.index)
    data.columns = [c.lower() for c in data.columns]
    return data

def combine_data(crypto):
    financial_data = fetch_financial_data(crypto)
    wiki_data = fetch_wikipedia_data(crypto)
    financial_data.index = financial_data.index.tz_localize(None)
    wiki_data.index = wiki_data.index.tz_localize(None)
    combined_data = financial_data.merge(wiki_data, left_index=True, right_index=True, how="left")
    combined_data = combined_data.fillna(0)
    combined_data["tomorrow"] = combined_data["close"].shift(-1)
    combined_data["target"] = (combined_data["tomorrow"] > combined_data["close"]).astype(int)
    combined_data.to_csv(f"{crypto}_combined.csv")
    return combined_data

def rundata():
    for crypto in cryptos:
        print(f"Processing data for {crypto}...")
        combine_data(crypto)


    
