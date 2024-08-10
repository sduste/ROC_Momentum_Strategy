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
            if data.empty:
                data = stock.history(period='max')
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

def get_todays_recommendation(results, ticker, buy_threshold, sell_threshold):
    print(f"Getting today's recommendation for ticker: {ticker}")
    
    # Check for the most recent date in the results for the specific ticker
    ticker_data = results[results['Ticker'] == ticker]
    if ticker_data.empty:
        print(f"No data found for ticker {ticker}")
        return 'No Data'
    
    # Get the most recent date for this ticker
    recent_date = ticker_data['Date'].max()
    print(f"Most recent date for {ticker}: {recent_date}")

    # Filter data for the most recent date
    recent_data = ticker_data[ticker_data['Date'] == recent_date]
    print(f"Data for {ticker} on {recent_date}:\n{recent_data}")

    if not recent_data.empty:
        latest_row = recent_data.iloc[-1]
        rate_of_change = latest_row['Rate of Change']
        print(f"Rate of Change for {ticker} on {recent_date}: {rate_of_change}")
        if rate_of_change > buy_threshold:
            return 'Buy'
        elif rate_of_change < sell_threshold:
            return 'Sell'
        else:
            return 'Hold'
    else:
        return 'No Data'

def calculate_rate_of_change(data, period=12):
    # Ensure 'Close Price' column exists
    if 'Close Price' not in data.columns:
        raise ValueError("Column 'Close Price' not found in data")
    
    # Calculate Rate of Change
    data['Rate of Change'] = data['Close Price'].pct_change(periods=period) * 100
    return data

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


    # Calculate Buy-and-Hold Returns
    buy_and_hold_returns = {}

    for ticker in tickers:
        ticker_data = results[results['Ticker'] == ticker]
        if not ticker_data.empty:
            first_day_price = ticker_data.iloc[0]['Close Price']
            last_day_price = ticker_data.iloc[-1]['Close Price']
            buy_and_hold_return = (last_day_price - first_day_price) / first_day_price
            buy_and_hold_returns[ticker] = buy_and_hold_return * 100

    # Define ranges for thresholds
    buy_threshold_range = np.arange(0.25, 2.5, 0.25)
    sell_threshold_range = np.arange(-2.5, -5.5, -2.5)

    # Initialize best thresholds and performance
    best_buy_thresholds = {}
    best_sell_thresholds = {}
    best_cumulative_returns = {}

    tax_rate = 0.20

    # Define transaction costs
    stt_rate = 0.001
    brokerage_rate = 0.005
    gst_rate = 0.18
    exchange_rate = 0.0003
    sebi_rate = 0.000001
    stamp_duty_rate = 0.00015

    # Iterate over all combinations of thresholds
    for buy_threshold, sell_threshold in product(buy_threshold_range, sell_threshold_range):
        print(f"Testing Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")

        # Initialize columns for signals, positions, and returns
        results['Signal'] = 'No Action'
        results['Position'] = 0
        results['Daily Return'] = 0.0
        results['Strategy Return'] = 0.0
        results['Buy Price'] = np.nan
        results['Sell Price'] = np.nan

        cumulative_returns = {ticker: [] for ticker in tickers}

        for ticker in tickers:
            ticker_data = results[results['Ticker'] == ticker].copy()
            current_position = None
            buy_price = None

            for i in range(len(ticker_data)):
                row = ticker_data.iloc[i]
                date = row['Date']

                if row['Rate of Change'] > buy_threshold and current_position is None:
                    current_position = 'Bought'
                    buy_price = row['Close Price']
                    
                    # Calculate buy-level costs
                    brokerage_buy = brokerage_rate * buy_price
                    gst_buy = gst_rate * brokerage_buy
                    exchange_charges_buy = exchange_rate * buy_price
                    sebi_charges_buy = sebi_rate * buy_price
                    stamp_duty_buy = stamp_duty_rate * buy_price
                    total_buy_costs = brokerage_buy + gst_buy + exchange_charges_buy + sebi_charges_buy + stamp_duty_buy
                    
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == date), 'Buy Price'] = buy_price
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == date), 'Signal'] = 'Buy'
                
                elif row['Rate of Change'] < sell_threshold and current_position == 'Bought':
                    current_position = None
                    sell_price = row['Close Price']
                    
                    # Calculate sell-level costs
                    brokerage_sell = brokerage_rate * sell_price
                    gst_sell = gst_rate * brokerage_sell
                    exchange_charges_sell = exchange_rate * sell_price
                    sebi_charges_sell = sebi_rate * sell_price
                    stt = stt_rate * buy_price
                    total_sell_costs = brokerage_sell + gst_sell + exchange_charges_sell + sebi_charges_sell + stt
                    
                    # Calculate total costs including buy-level costs
                    total_costs = total_buy_costs + total_sell_costs
                    
                    trade_return = (sell_price - buy_price - total_costs) / buy_price
                    if trade_return > 0:
                        trade_return_after_tax = trade_return * (1 - tax_rate)
                    else:
                        trade_return_after_tax = trade_return
                    
                    cumulative_returns[ticker].append(trade_return_after_tax + 1)
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == date), 'Sell Price'] = sell_price
                    results.loc[(results['Ticker'] == ticker) & (results['Date'] == date), 'Signal'] = 'Sell'
                    buy_price = None
                else:
                    if current_position == 'Bought':
                        results.loc[(results['Ticker'] == ticker) & (results['Date'] == date), 'Signal'] = 'Hold'
                    else:
                        results.loc[(results['Ticker'] == ticker) & (results['Date'] == date), 'Signal'] = 'No Action'

        # Calculate the average cumulative return for each ticker and store the best
        for ticker, returns in cumulative_returns.items():
            if len(returns) > 0:
                ticker_cumulative_return = np.prod(returns) - 1
                if ticker not in best_cumulative_returns or ticker_cumulative_return > best_cumulative_returns[ticker]:
                    best_cumulative_returns[ticker] = ticker_cumulative_return
                    best_buy_thresholds[ticker] = buy_threshold
                    best_sell_thresholds[ticker] = sell_threshold

    # Save results and performance summary to CSV files
    results.to_csv('strategy_results.csv', index=False)
    performance_summary = pd.DataFrame({
        'Ticker': list(best_cumulative_returns.keys()),
        'Best Buy Threshold': [best_buy_thresholds[ticker] for ticker in best_cumulative_returns.keys()],
        'Best Sell Threshold': [best_sell_thresholds[ticker] for ticker in best_cumulative_returns.keys()],
        'Cumulative Return': [best_cumulative_returns[ticker] * 100 for ticker in best_cumulative_returns.keys()]
    })
    performance_summary.to_csv('performance_summary.csv', index=False)

    # Prepare email content
    subject = "Trading Strategy Results"
    body_lines = []
    for ticker in best_cumulative_returns.keys():
        last_signal = results[results['Ticker'] == ticker].iloc[-1]['Signal']
        body_lines.append(f"{ticker}: Best Buy Threshold: {best_buy_thresholds[ticker]}, Best Sell Threshold: {best_sell_thresholds[ticker]}, Portfolio Return: {best_cumulative_returns[ticker] * 100:.2f}%")
        body_lines.append(f"Today's Action: {last_signal}")

        # Add Buy-and-Hold Return
        buy_and_hold_return = buy_and_hold_returns.get(ticker, None)
        if buy_and_hold_return is not None:
            body_lines.append(f"Buy-and-Hold Return: {buy_and_hold_return:.2f}%")
        else:
            body_lines.append("Buy-and-Hold Return: Data not available")

    body = "\n".join(body_lines)

    # Send email with results
    attachments = ['strategy_results.csv', 'performance_summary.csv', 'trade_returns.csv']
    print("Sending email with attachments:", attachments)
    send_email(subject, body, attachments)

    print("Strategy results saved to strategy_results.csv")
    print("Performance summary saved to performance_summary.csv")

# Run the strategy
run_strategy()
