import yfinance as yf
import numpy as np
import pandas as pd
from newspaper import Article
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Function to fetch stock data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period)
    return stock_data

# Function to get news articles about the stock
def get_news(ticker, num_articles=5):
    query = f"{ticker} stock news"
    article = Article(f"https://www.google.com/search?q={query}")
    article.download()
    article.parse()
    return article.text

# Sentiment Analysis with TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Sentiment Analysis with VADER
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Collect data: Sentiment and Stock Data
def collect_data(tickers, period="1y"):
    data = []
    for ticker in tickers:
        print(f"Processing {ticker}...")
        stock_data = get_stock_data(ticker, period)
        news = get_news(ticker)

        sentiment_score = analyze_sentiment_vader(news)  # You can switch to TextBlob if you prefer

        # Create label (1 for up, 0 for down) based on the closing price change
        stock_data['Sentiment'] = sentiment_score
        stock_data['Target'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)

        # Collect the data for model training/testing
        for _, row in stock_data.iterrows():
            data.append([row['Sentiment'], row['Target']])

    return pd.DataFrame(data, columns=['Sentiment', 'Target'])

# Train a predictive model
def train_predictive_model(data):
    X = data[['Sentiment']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

# Predict if stock price will go up or down based on sentiment
def predict_stock_movement(ticker, model):
    news = get_news(ticker)
    sentiment_score = analyze_sentiment_vader(news)
    prediction = model.predict([[sentiment_score]])

    print("--------------------------------------------------------------\n")
    print(f"Sentiment Score for {ticker}: {sentiment_score:.2f}")
    if prediction == 1:
        print(f"Prediction: {ticker} stock will go UP. \n")
        print("--------------------------------------------------------------")
    else:
        print(f"Prediction: {ticker} stock will go DOWN. \n")
        print("--------------------------------------------------------------")

if __name__ == "__main__":
    tickers = input("Enter stock tickers (comma separated): ").split(",")
    tickers = [ticker.strip().upper() for ticker in tickers]

    period = input("Enter the period for stock data (e.g., '1y' for one year): ").strip()
    if not period:
        period = "1y"  # Default to one year if no input is provided

    data = collect_data(tickers, period)
    model = train_predictive_model(data)

    # Example prediction for each stock
    for ticker in tickers:
        predict_stock_movement(ticker, model)
