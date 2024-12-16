import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import pandas_ta as ta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

class FinancialAnalyzer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.retrieve_stock_data()
    
    def retrieve_stock_data(self):
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            if data.empty:
                raise ValueError("No data retrieved. Please check the ticker symbol or date range.")
            return data
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def calculate_technical_indicators(self):
        if self.data is None:
            print("Data not available for calculations.")
            return None
        
        data = self.data.copy()
        # Flattening MultiIndex columns
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        print("Flattened Data Columns:", data.columns)

        # Ensuring no NaN values in 'Close_AAPL' column for calculation
        if data['Close_AAPL'].isna().sum() > 0:
            print("NaN values found in 'Close_AAPL' column. Please clean the data.")
            return None

        data['SMA'] = self.calculate_moving_average(data['Close_AAPL'], 20)
        print("SMA calculated. Data Columns:", data.columns)
        
        data['RSI'] = ta.rsi(data['Close_AAPL'], length=14)
        data['EMA'] = ta.ema(data['Close_AAPL'], length=20)
        macd = ta.macd(data['Close_AAPL'])
        
        if macd is not None:
            data['MACD'] = macd['MACD_12_26_9']
            data['MACD_Signal'] = macd['MACDs_12_26_9']
        else:
            print("MACD calculation returned None.")
        
        bollinger = ta.bbands(data['Close_AAPL'], length=20)
        if bollinger is not None:
            data['Bollinger_High'] = bollinger['BBU_20_2.0']
            data['Bollinger_Low'] = bollinger['BBL_20_2.0']
        else:
            print("Bollinger Bands calculation returned None.")
        
        self.data = data
        return data
    
    def calculate_moving_average(self, series, window):
        return series.rolling(window=window).mean()

    def plot_stock_data(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        print("Data Columns Before Plotting:", self.data.columns)
        
        # Drop rows with NaN values in 'SMA'
        data_to_plot = self.data.dropna(subset=['SMA'])
        
        fig = px.line(data_to_plot, x=data_to_plot.index, y=['Close_AAPL', 'SMA'], 
                      title='Stock Price with Moving Average')
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.show()
    
    def plot_ema(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y=['Close_AAPL', 'EMA'], 
                      title='Stock Price with Exponential Moving Average')
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.show()
    
    def plot_rsi(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.update_layout(xaxis_title='Date', yaxis_title='RSI')
        fig.show()

    def plot_macd(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y=['MACD', 'MACD_Signal'], 
                      title='MACD and Signal Line')
        fig.update_layout(xaxis_title='Date', yaxis_title='Value')
        fig.show()

    def download_data(self, tickers):
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)['Close']
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame(data)

    def calculate_portfolio_weights(self, tickers):
        try:
            data = self.download_data(tickers)
            if data.empty:
                print("No valid data available for the given tickers.")
                return None
            
            mu = expected_returns.mean_historical_return(data)
            cov = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, cov)
            weights = ef.max_sharpe()
            return dict(zip(tickers, weights.values()))
        except Exception as e:
            print(f"Error calculating portfolio weights: {e}")
            return None

    def calculate_portfolio_performance(self, tickers):
        try:
            data = self.download_data(tickers)
            if data.empty:
                print("No valid data available for the given tickers.")
                return None
            
            mu = expected_returns.mean_historical_return(data)
            cov = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, cov)
            weights = ef.max_sharpe()
            portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
            return portfolio_return, portfolio_volatility, sharpe_ratio
        except Exception as e:
            print(f"Error calculating portfolio performance: {e}")
            return None
        