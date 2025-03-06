import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def backtest_strategy(data: pd.DataFrame, model: LogisticRegression, feature_cols: list) -> pd.DataFrame:
    """
    Backtest a simple trading strategy.
    The strategy: when the model predicts bullish (1), we 'invest' and receive the next period's return;
              when bearish (0), we stay in cash.
              
    Parameters:
        data (pd.DataFrame): Data with features, returns, and target columns.
        model (LogisticRegression): Trained model.
        feature_cols (list): List of feature columns used in the model.
        
    Returns:
        pd.DataFrame: DataFrame containing strategy and market returns.
    """
    # Drop any rows with missing feature data
    data = data.dropna(subset=feature_cols + ['Return'])
    
    # Generate model predictions (signals)
    X = data[feature_cols]
    data['Signal'] = model.predict(X)
    
    # Compute the strategy returns: if Signal==1 (bullish), take the next period's return; else, 0.
    data['Strategy_Return'] = data['Signal'] * data['Return']
    
    # Calculate cumulative returns
    data['Cumulative_Market_Return'] = (1 + data['Return']).cumprod() - 1
    data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod() - 1
    
    return data

if __name__ == "__main__":
    # Import necessary functions from our other modules
    from src.data.download_data import fetch_tsla_data
    from src.data.preprocess import clean_data
    from src.data.feature_engineering import add_features
    from src.models.model import prepare_model_data, train_model

    # Fetch historical data (using a longer period for backtesting)
    raw_data = fetch_tsla_data(period="6mo", interval="1d")
    cleaned_data = clean_data(raw_data)
    data_with_features = add_features(cleaned_data)
    
    # Prepare the data for modeling (adding the target column)
    model_data = prepare_model_data(data_with_features)
    
    # Define the features to use (adjust as necessary)
    features = ['MA_20', 'MA_50', 'RSI_14']
    
    # Drop rows with NaNs in feature or target columns
    model_data = model_data.dropna(subset=features + ['Target'])
    
    # Train the logistic regression model
    model = train_model(model_data, features)
    
    # Run the backtesting strategy
    backtested = backtest_strategy(model_data, model, features)
    
    # Display final cumulative returns for both the market and the strategy
    print("Final Cumulative Market Return:", backtested['Cumulative_Market_Return'].iloc[-1])
    print("Final Cumulative Strategy Return:", backtested['Cumulative_Strategy_Return'].iloc[-1])
    
    # Plot the cumulative returns for a visual comparison
    ax = backtested[['Cumulative_Market_Return', 'Cumulative_Strategy_Return']].plot(title="Backtesting Performance")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    
    # Save the plot to a file instead of trying to show it interactively
    plt.savefig("backtest_performance.png")
    print("Plot saved as backtest_performance.png")
    plt.close()
