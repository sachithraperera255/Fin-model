import os
import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the TSLA data DataFrame.

    Parameters:
        data (pd.DataFrame): Raw TSLA data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Drop any rows with missing values
    data = data.dropna()

    # Reset index if needed (e.g., after dropping rows)
    data.reset_index(inplace=True)

    # Convert the index to a datetime column if it's not already
    # Sometimes the Date column is in the index, so we check and convert if needed.
    if 'Date' not in data.columns:
        data.rename(columns={'index': 'Date'}, inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])
    
    return data

def save_processed_data(data: pd.DataFrame, filename: str = "processed_tsla.csv"):
    """
    Save the cleaned data to the processed folder.
    
    Parameters:
        data (pd.DataFrame): Cleaned TSLA data.
        filename (str): Name of the output file.
    """
    # Build the file path in the processed folder
    processed_folder = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)
    file_path = os.path.join(processed_folder, filename)
    
    data.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    # For testing, import the data download function.
    # Ensure that the PYTHONPATH includes the 'src' directory or run from the project root.
    from .download_data import fetch_tsla_data

    # Fetch raw data using our earlier module
    raw_data = fetch_tsla_data(period="1mo", interval="1d")
    
    # Clean the data
    cleaned_data = clean_data(raw_data)
    print("Cleaned Data Preview:")
    print(cleaned_data.head())
    
    # Save the cleaned data
    save_processed_data(cleaned_data)
