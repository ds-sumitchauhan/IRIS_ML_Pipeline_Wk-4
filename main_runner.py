# main_runner.py

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

# Import our custom modules
from src.data_loader import load_data
from src.governed_analysis import run_fairness_and_xai
from src.drift_analysis import run_drift_detection

if __name__ == "__main__":
    mlflow.set_experiment("ML Governance Full Analysis")
    
    with mlflow.start_run() as run:
        print("--- Starting a new MLflow Run ---")
        run_id = run.info.run_id
        mlflow.set_tag("mlflow.runName", "Iris_Governed_Run")
        print(f"Run ID: {run_id}")

        df = load_data("data/iris.csv")
        df['location'] = np.random.randint(0, 2, df.shape[0])
        
        X = df.drop(columns=["species"])
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        print("Model training complete.")
        
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        
        model_path = "iris_classifier_model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved locally to {model_path}")
        mlflow.log_artifact(model_path, "model")
        
        fairness_path, shap_path = run_fairness_and_xai(
            model=model, X_test=X_test, y_test=y_test, class_name_to_explain='virginica'
        )
        mlflow.log_artifact(fairness_path, "governance_reports")
        
        # --- FIX: Only log the SHAP artifact if it was created ---
        if shap_path:
            mlflow.log_artifact(shap_path, "governance_reports")
        
        drift_report_path = run_drift_detection(reference_data=df)
        mlflow.log_artifact(drift_report_path, "governance_reports")

        print("\n--- MLflow Run Complete ---")
        print(f"All artifacts logged to run ID: {run_id}")
        print("Run 'mlflow ui' in your terminal to see the results.")
