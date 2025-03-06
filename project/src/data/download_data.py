import yfinance as yf

def fetch_tsla_data(period="1mo", interval="1d"):
    """
    Fetch TSLA data from Yahoo Finance.
    
    Parameters:
        period (str): The total period to fetch data for (e.g., '1mo', '1y').
        interval (str): Data interval (e.g., '1d', '1h', etc.).
    
    Returns:
        DataFrame: A pandas DataFrame containing TSLA historical data.
    """
    tsla = yf.Ticker("TSLA")
    data = tsla.history(period=period, interval=interval)
    print(data.head())
    return data

if __name__ == "__main__":
    fetch_tsla_data()
