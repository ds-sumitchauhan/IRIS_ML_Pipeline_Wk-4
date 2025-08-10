# src/fairness_explainer.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# Import Fairlearn metrics
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

# Import our existing data loader
from data_loader import load_data

def run_fairness_and_explainability_analysis():
    """
    This function performs the full analysis for the assignment:
    1. Loads data and adds a sensitive attribute.
    2. Trains a model.
    3. Evaluates fairness using Fairlearn.
    4. Creates and displays a SHAP summary plot for the 'virginica' class.
    """
    print("--- Starting Fairness and Explainability Analysis ---")

    # 1. Load data and add the sensitive 'location' attribute
    df = load_data("data/iris.csv")
    df['location'] = np.random.randint(0, 2, df.shape[0])
    print("Successfully loaded data and added random 'location' attribute.")
    
    # 2. Prepare data and train the multiclass model
    X = df.drop(columns=["species"])
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # --- Define the POSITIVE_CLASS based on your findings from Step 1 ---
    POSITIVE_CLASS = 'virginica' 

    print(f"\n--- 1. Fairlearn: Evaluating Model Fairness for '{POSITIVE_CLASS}' ---")

    # Convert multiclass problem to binary for Fairlearn
    y_test_binary = (y_test == POSITIVE_CLASS).astype(int)
    y_pred_binary = (y_pred == POSITIVE_CLASS).astype(int)
    
    sensitive_features = X_test['location']
    
    dpd = demographic_parity_difference(y_true=y_test_binary, y_pred=y_pred_binary, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_true=y_test_binary, y_pred=y_pred_binary, sensitive_features=sensitive_features)
    
    print(f"Demographic Parity Difference: {dpd:.4f}")
    print(f"Equalized Odds Difference: {eod:.4f}")

    print(f"\n--- 2. SHAP: Explaining Model Predictions for '{POSITIVE_CLASS}' ---")

    explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values = explainer(X_test)

    # Find the index corresponding to the positive class
    class_names = model.classes_
    try:
        positive_class_index = list(class_names).index(POSITIVE_CLASS)
        print(f"Found '{POSITIVE_CLASS}' at index {positive_class_index} in model outputs.")
    except ValueError:
        print(f"Error: '{POSITIVE_CLASS}' class not found. Check the POSITIVE_CLASS variable and the output of Step 1.")
        print(f"Model knows about these classes: {class_names}")
        return

    # Create the SHAP summary plot for the positive class
    print("\nGenerating SHAP summary plot... Please close the plot window to continue.")
    
    plt.title(f"SHAP Summary for {POSITIVE_CLASS} Class")
    shap.summary_plot(
        shap_values=shap_values[:, :, positive_class_index], 
        features=X_test,
        feature_names=X_test.columns,
        show=False 
    )
    plt.savefig(f'shap_summary_{POSITIVE_CLASS}.png')
    plt.show()

    print("\n--- Analysis Complete ---")
    print(f"SHAP plot saved as 'shap_summary_{POSITIVE_CLASS}.png'")


if __name__ == "__main__":
    run_fairness_and_explainability_analysis()
