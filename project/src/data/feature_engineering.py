import pandas as pd
import numpy as np

def compute_moving_average(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute the simple moving average for a given window.
    """
    return data['Close'].rolling(window=window).mean()

def compute_RSI(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given window.
    """
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    You can adjust the window sizes as needed or dynamically compute for different time frames.
    """
    # Example: calculate moving averages
    data['MA_20'] = compute_moving_average(data, window=20)
    data['MA_50'] = compute_moving_average(data, window=50)
    
    # Calculate RSI with a 14-day window
    data['RSI_14'] = compute_RSI(data, window=14)
    
    # Volume is typically provided, but you could add derived volume features if needed.
    
    return data

if __name__ == "__main__":
    # For testing, import the preprocessing module to load our cleaned data.
    from .preprocess import clean_data, save_processed_data
    from .download_data import fetch_tsla_data

    # Fetch raw data
    raw_data = fetch_tsla_data(period="1mo", interval="1d")
    
    # Preprocess the data
    cleaned_data = clean_data(raw_data)
    
    # Add technical features
    featured_data = add_features(cleaned_data)
    print("Data with features added:")
    print(featured_data.head())
    
    # Optionally, save the data with features for further use
    save_processed_data(featured_data, filename="tsla_with_features.csv")
