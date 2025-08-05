import pandas as pd
import numpy as np

def poison_data(df: pd.DataFrame, poisoning_level: float) -> pd.DataFrame:
    """Poisons a percentage of the dataset by replacing feature values with random numbers.
                Args:  df (pd.DataFrame): The original, clean DataFrame.\
                        poisoning_level (float): The percentage of data to poison (e.g., 0.05 for 5%).
                  Returns: pd.DataFrame: The poisoned DataFrame. """
    if not 0 <= poisoning_level <= 1:
        raise ValueError("Poisoning level must be between 0 and 1.")

    poisoned_df = df.copy()
    num_rows_to_poison = int(len(poisoned_df) * poisoning_level)                            
    # Get random indices to poison
    poison_indices = np.random.choice(poisoned_df.index, size=num_rows_to_poison, replace=False)                                                                             
                                                                                        
    # Define the columns to poison (all except the target 'species')
    feature_columns = [col for col in df.columns if col != 'species']
    
    # Generate random data for the selected rows and columns
    # We'll generate random numbers between 0 and 10, which is outside the normal range for Iris features
    random_data = np.random.uniform(0, 10, size=(num_rows_to_poison, len(feature_columns)))
                                                                                             
    # Replace the original data with the random (poisoned) data
    poisoned_df.loc[poison_indices, feature_columns] = random_data
                                                                                                                            
    print(f"Poisoned {num_rows_to_poison} rows ({poisoning_level * 100:.0f}% of the data).")
                                                                                                                                  
    return poisoned_df
