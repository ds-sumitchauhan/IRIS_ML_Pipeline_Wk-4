#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import sys
#import pandas as pd

#def train_and_evaluate():
#    df = pd.read_csv("data/iris.csv")
#    X = df.drop(columns=["species"])
#    y = df["species"]
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#    model = LogisticRegression(max_iter=200)
#    model.fit(X_train, y_train)
#    acc = accuracy_score(y_test, model.predict(X_test))
#    return model, acc
#
##from src.model import train_and_evaluate
#
#if __name__ == "__main__":
#    # df = sys.argv[1]
#    model, acc = train_and_evaluate()
#    print(f"Model trained successfully.\n Accuracy: {acc:.4f}")


# src/model_trainer.py

# src/model_trainer.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import your other functions
from data_loader import load_data
from data_poisoner import poison_data

def train_and_evaluate(df):
    """Trains a model and returns the model, accuracy, and test data for signature inference."""
    X = df.drop(columns=["species"])
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Infer the signature from the model's inputs and outputs
    signature = infer_signature(X_train, predictions)
    
    return model, acc, X_test, y_test, signature

# --- Main execution block ---
if __name__ == "__main__":
    mlflow.set_experiment("Iris Data Poisoning Attack with Artifacts")
    
    clean_df = load_data("data/iris.csv")
    poisoning_levels = [0.0, 0.05, 0.10, 0.50]
    
    os.makedirs("temp_artifacts", exist_ok=True)

    print("--- Starting Data Poisoning Experiment with MLflow Artifacts ---")
    
    # The 'with' statement handles starting and stopping the run
    with mlflow.start_run(run_name="Poisoning Experiment Parent Run"):
        mlflow.log_param("poisoning_levels", poisoning_levels)

        for level in poisoning_levels:
            # The 'with' statement for the nested run ensures it properly finishes
            # and commits its artifacts before the next loop iteration.
            with mlflow.start_run(run_name=f"Poisoning_Level_{int(level*100)}%", nested=True):
                print(f"\n--- Logging experiment for {level*100:.0f}% Poisoning ---")
                
                if level == 0.0:
                    training_df = clean_df.copy()
                    dataset_path = "temp_artifacts/clean_data.csv"
                else:
                    training_df = poison_data(clean_df, level)
                    dataset_path = f"temp_artifacts/poisoned_data_{int(level*100)}p.csv"
                
                # Log dataset artifact
                training_df.to_csv(dataset_path, index=False)
                mlflow.log_artifact(dataset_path, "datasets")
                
                mlflow.log_param("poisoning_level", level)
                
                model, acc, X_test, y_test, signature = train_and_evaluate(training_df)
                
                print(f"Validation Accuracy: {acc:.4f}")
                mlflow.log_metric("accuracy", acc)
                
                # Log the model with signature and input example (solves the warning)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"iris_model_poisoned_{int(level*100)}", # Correct parameter name
                    signature=signature,
                    input_example=X_test.head(5) # Provide an input example
                )

                # Log confusion matrix plot artifact
                cm = confusion_matrix(y_test, model.predict(X_test))
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=model.classes_, yticklabels=model.classes_)
                plt.title(f'Confusion Matrix (Poisoning Level: {int(level*100)}%)')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                
                plot_path = f"temp_artifacts/confusion_matrix_{int(level*100)}p.png"
                plt.savefig(plot_path)
                plt.close()
                mlflow.log_artifact(plot_path, "plots")

    print("\n--- MLflow Experiment Script Finished Successfully ---")
    print("Run 'mlflow ui' in your terminal to see the results.")
