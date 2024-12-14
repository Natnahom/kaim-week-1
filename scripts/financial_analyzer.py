import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import pandas_ta as ta  # Changed from talib to pandas-ta
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
    
    def calculate_moving_average(self, data, window_size):
        return ta.sma(data, length=window_size)  # Changed to pandas-ta

    def calculate_technical_indicators(self):
        if self.data is None:
            print("Data not available for calculations.")
            return None
        
        data = self.data.copy()
        data['SMA'] = self.calculate_moving_average(data['Close'], 20)
        data['RSI'] = ta.rsi(data['Close'], length=14)  # Changed to pandas-ta
        data['EMA'] = ta.ema(data['Close'], length=20)  # Changed to pandas-ta
        macd = ta.macd(data['Close'])  # Changed to pandas-ta
        data['MACD'] = macd['macd']
        data['MACD_Signal'] = macd['signal']
        
        # Additional indicators can be added here
        bollinger = ta.bbands(data['Close'], length=20)  # Changed to pandas-ta
        data['Bollinger_High'] = bollinger['BBH_20']
        data['Bollinger_Low'] = bollinger['BBL_20']

        return data
    
    def plot_stock_data(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y=['Close', 'SMA'], 
                      title='Stock Price with Moving Average')
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.show()
    
    def plot_rsi(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.update_layout(xaxis_title='Date', yaxis_title='RSI')
        fig.show()

    def plot_ema(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y=['Close', 'EMA'], 
                      title='Stock Price with Exponential Moving Average')
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.show()
    
    def plot_macd(self):
        if self.data is None:
            print("Data not available for plotting.")
            return
        
        fig = px.line(self.data, x=self.data.index, y=['MACD', 'MACD_Signal'], 
                      title='MACD and Signal Line')
        fig.update_layout(xaxis_title='Date', yaxis_title='Value')
        fig.show()

    def calculate_portfolio_weights(self, tickers):
        try:
            data = yf.download(tickers, start=self.start_date, end=self.end_date)['Close']
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
            data = yf.download(tickers, start=self.start_date, end=self.end_date)['Close']
            mu = expected_returns.mean_historical_return(data)
            cov = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, cov)
            weights = ef.max_sharpe()
            portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
            return portfolio_return, portfolio_volatility, sharpe_ratio
        except Exception as e:
            print(f"Error calculating portfolio performance: {e}")
            return None