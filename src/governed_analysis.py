# src/governed_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Import all necessary components from Fairlearn
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)

def run_fairness_and_xai(model, X_test, y_test, class_name_to_explain):
    """
    Runs fairness and XAI analysis and returns paths to saved artifacts.
    """
    print("--- Running Fairness & XAI Analysis ---")
    y_pred = model.predict(X_test)

    # --- 1. Detailed Fairness Analysis with MetricFrame ---
    sensitive_features = X_test['location']
    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0)
    }
    grouped_on_location = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # --- 2. Calculate Specific Fairness Difference Metrics ---
    # These metrics require a binary conversion of the target variable.
    y_test_binary = (y_test == class_name_to_explain).astype(int)
    y_pred_binary = (y_pred == class_name_to_explain).astype(int)

    dpd = demographic_parity_difference(y_true=y_test_binary, y_pred=y_pred_binary, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_true=y_test_binary, y_pred=y_pred_binary, sensitive_features=sensitive_features)

    # --- 3. Create the Complete Fairness Report ---
    fairness_report_str = "--- FAIRNESS ANALYSIS REPORT ---\n\n"
    fairness_report_str += "1. High-Level Fairness Metrics:\n"
    fairness_report_str += f"   - Demographic Parity Difference: {dpd:.4f}\n"
    fairness_report_str += f"   - Equalized Odds Difference: {eod:.4f}\n\n"
    fairness_report_str += "2. Detailed Performance Breakdown by Group:\n"
    fairness_report_str += f"   Overall Model Metrics:\n{grouped_on_location.overall}\n\n"
    fairness_report_str += f"   Metrics by Location Group:\n{grouped_on_location.by_group}"

    fairness_artifact_path = "fairness_report.txt"
    with open(fairness_artifact_path, 'w') as f:
        f.write(fairness_report_str)
    print(f"Complete fairness report saved to {fairness_artifact_path}")

    # --- 4. Explainability (XAI) Analysis ---
    explainer = shap.TreeExplainer(model)
    shap_object = explainer(X_test)
    class_names = model.classes_
    class_index = list(class_names).index(class_name_to_explain)

    plt.figure()
    plt.title(f"SHAP Feature Importance for '{class_name_to_explain}' Class")
    shap.summary_plot(shap_object.values[:, :, class_index], X_test, show=False)
    shap_artifact_path = f"shap_summary_{class_name_to_explain}.png"
    plt.savefig(shap_artifact_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP plot saved to {shap_artifact_path}")

    return fairness_artifact_path, shap_artifact_path
