import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
TO_EMAIL = os.getenv('TO_EMAIL')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

def send_email(subject, body, attachments):
    if not isinstance(attachments, list):
        raise ValueError("Attachments should be provided as a list of filenames.")
    
    if EMAIL_ADDRESS is None or EMAIL_PASSWORD is None or TO_EMAIL is None:
        print("Error: One or more email configuration variables are missing.")
        print(f"EMAIL_ADDRESS: {EMAIL_ADDRESS}")
        print(f"EMAIL_PASSWORD: {EMAIL_PASSWORD}")
        print(f"TO_EMAIL: {TO_EMAIL}")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    for filename in attachments:
        if filename is None or not os.path.isfile(filename):
            print(f"Attachment {filename} is not found or invalid.")
            continue
        with open(filename, 'rb') as file:
            part = MIMEApplication(file.read(), Name=os.path.basename(filename))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filename)}"'
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")


def generate_results_csv(tickers):
    results = pd.DataFrame(columns=['Ticker', 'Date', 'Rate of Change', 'Close Price'])

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='5y')
            data['Rate of Change'] = data['Close'].pct_change(periods=12) * 100
            data = data.reset_index()
            data = data[['Date', 'Close', 'Rate of Change']].dropna()
            data = data.rename(columns={'Close': 'Close Price'})
            data['Ticker'] = ticker
            data = data[['Ticker', 'Date', 'Rate of Change', 'Close Price']]
            results = pd.concat([results, data], ignore_index=True)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    results.to_csv('results.csv', index=False)
    print("results.csv generated successfully")

def get_todays_recommendation(results, buy_threshold, sell_threshold):
    today = datetime.now().date()
    today_data = results[results['Date'].dt.date == today]

    if not today_data.empty:
        latest_row = today_data.iloc[-1]
        rate_of_change = latest_row['Rate of Change']
        if rate_of_change > buy_threshold:
            return 'Buy'
        elif rate_of_change < sell_threshold:
            return 'Sell'
        else:
            return 'Hold'
    else:
        return 'No Data'

def run_strategy():
    # Read tickers from ETFs.csv
    tickers_df = pd.read_csv('ETFs.csv')
    if 'Ticker' not in tickers_df.columns:
        raise ValueError("Column 'Ticker' not found in ETFs.csv")
    tickers = tickers_df['Ticker'].tolist()

    # Generate results.csv
    generate_results_csv(tickers)

    # Load the results data
    results = pd.read_csv('results.csv')

    # Convert Date column to datetime with utc=True
    results['Date'] = pd.to_datetime(results['Date'], utc=True)
    results.sort_values(by=['Ticker', 'Date'], inplace=True)

    # Define ranges for thresholds
    buy_threshold_range = np.arange(0.5, 2.5, 0.5)
    sell_threshold_range = np.arange(-2.5, -5.0, -2.5)

    # Initialize best thresholds and performance
    best_buy_threshold = None
    best_sell_threshold = None
    best_cumulative_return = -np.inf
    tax_rate = 0.20

    # Iterate over all combinations of thresholds
    for buy_threshold, sell_threshold in product(buy_threshold_range, sell_threshold_range):
        print(f"Testing Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")

        # Initialize columns for signals, positions, and returns
        results['Signal'] = 'Hold'
        results['Position'] = 0
        results['Daily Return'] = 0.0
        results['Strategy Return'] = 0.0
        results['Buy Price'] = np.nan
        results['Sell Price'] = np.nan

        tickers = results['Ticker'].unique()
        cumulative_returns = []

        for ticker in tickers:
            ticker_data = results[results['Ticker'] == ticker].copy()
            current_position = None
            buy_price = None
            ticker_returns = []

            for i in range(len(ticker_data)):
                row = ticker_data.iloc[i]

                if row['Rate of Change'] > buy_threshold and current_position is None:
                    current_position = 'Bought'
                    buy_price = row['Close Price']
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == row['Date']), 'Buy Price'] = buy_price
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == row['Date']), 'Signal'] = 'Buy'
                
                elif row['Rate of Change'] < sell_threshold and current_position == 'Bought':
                    current_position = None
                    sell_price = row['Close Price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_return_after_tax = trade_return * (1 - tax_rate)
                    ticker_returns.append(trade_return_after_tax + 1)
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == row['Date']), 'Sell Price'] = sell_price
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == row['Date']), 'Signal'] = 'Sell'
                    buy_price = None

                if current_position == 'Bought':
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == row['Date']), 'Strategy Return'] = (row['Close Price'] - buy_price) / buy_price
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == row['Date']), 'Signal'] = 'Hold'

            if current_position == 'Bought':
                last_price = ticker_data.iloc[-1]['Close Price']
                trade_return = (last_price - buy_price) / buy_price
                trade_return_after_tax = trade_return * (1 - tax_rate)
                ticker_returns.append(trade_return_after_tax + 1)
                results.loc[(results['Ticker'] == ticker) & (results['Date'] == ticker_data.iloc[-1]['Date']), 'Sell Price'] = last_price
                results.loc[(results['Ticker'] == ticker) & (results['Date'] == ticker_data.iloc[-1]['Date']), 'Signal'] = 'Sell'
                results.loc[(results['Ticker'] == ticker) & (results['Date'] == ticker_data.iloc[-1]['Date']), 'Strategy Return'] = trade_return_after_tax

            if len(ticker_returns) > 0:
                ticker_cumulative_return = np.prod(ticker_returns) - 1
                cumulative_returns.append(ticker_cumulative_return)

        if len(cumulative_returns) > 0:
            average_cumulative_return = np.mean(cumulative_returns)
            if average_cumulative_return > best_cumulative_return:
                best_cumulative_return = average_cumulative_return
                best_buy_threshold = buy_threshold
                best_sell_threshold = sell_threshold

    subject = "Trading Strategy Results"
    body = (f"Best Buy Threshold: {best_buy_threshold}\n"
            f"Best Sell Threshold: {best_sell_threshold}\n"
            f"Best Cumulative Return: {best_cumulative_return:.2%}\n"
            f"Today's Recommendation: {get_todays_recommendation(results, best_buy_threshold, best_sell_threshold)}")

    # Save results and performance summary to CSV files
    results.to_csv('strategy_results.csv', index=False)
    performance_summary = pd.DataFrame({
        'Overall Portfolio Cumulative Return (Equal-Weighted)': [best_cumulative_return * 100]
    })
    performance_summary.to_csv('performance_summary.csv', index=False)

    # Send email with results
    attachments = ['strategy_results.csv', 'performance_summary.csv']
    print("Sending email with attachments:", attachments)
    send_email(subject, body, attachments)

    print("Strategy results saved to strategy_results.csv")
    print("Performance summary saved to performance_summary.csv")

# Run the strategy
run_strategy()
