import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for modeling by creating a target column based on future returns.
    
    The target is defined as 1 if the next period's return (e.g., next day's) is positive,
    and 0 otherwise.
    """
    # Calculate daily return (shifted to use today's indicators for predicting tomorrow's return)
    data['Return'] = data['Close'].pct_change().shift(-1)
    # Define target: 1 if return is positive, 0 otherwise
    data['Target'] = (data['Return'] > 0).astype(int)
    # Drop rows with NaN values (the last row will have a NaN target)
    data = data.dropna(subset=['Return'])
    return data

def train_model(data: pd.DataFrame, feature_cols: list) -> LogisticRegression:
    # Drop rows with NaN in the specified feature columns or the target column
    data = data.dropna(subset=feature_cols + ['Target'])
    
    # Split data into features and target
    X = data[feature_cols]
    y = data['Target']
    
    # Split into training and testing datasets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train logistic regression model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    from sklearn.metrics import accuracy_score, confusion_matrix
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", acc)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model


if __name__ == "__main__":
    # Import functions from the data modules. Adjust the import paths if needed.
    from src.data.download_data import fetch_tsla_data
    from src.data.preprocess import clean_data
    from src.data.feature_engineering import add_features

    # Fetch a larger sample of historical data to improve model robustness
    raw_data = fetch_tsla_data(period="3mo", interval="1d")
    cleaned_data = clean_data(raw_data)
    
    # Add technical features (e.g., moving averages, RSI)
    data_with_features = add_features(cleaned_data)
    
    # Prepare data by creating the target column for the model
    model_data = prepare_model_data(data_with_features)
    
    # Select the features we want to use. You can adjust these based on your feature engineering.
    features = ['MA_20', 'MA_50', 'RSI_14']
    
    # Train and evaluate the model
    model = train_model(model_data, features)
