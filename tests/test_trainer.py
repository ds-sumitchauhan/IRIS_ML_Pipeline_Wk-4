import pandas as pd
from sklearn.model_selection import train_test_split
#from src.trainer import train_and_evaluate
from src.data_loader import load_data
from src.model_trainer import train_and_evaluate

def test_model_train_accuracy():
    df = load_data("data/iris.csv")
    #X = df.drop(columns=["species"])
    #y = df["species"]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, acc = train_and_evaluate(df)
    assert acc > 0.8
    assert hasattr(model, "predict")
