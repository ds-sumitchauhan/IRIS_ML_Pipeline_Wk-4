import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the Iris dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError as e:
        print(f"File not found: {filepath}")
        raise e
