import pandas as pd
import os
from src.data_loader import load_data

def test_load_data():
    df = pd.read_csv("data/iris.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "sepal_length" in df.columns
