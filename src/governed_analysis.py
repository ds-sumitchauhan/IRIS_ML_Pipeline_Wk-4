# src/governed_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import shap
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score

def run_fairness_and_xai(model, X_test, y_test, class_name_to_explain):
    """
    Runs fairness and XAI analysis and returns paths to saved artifacts.
    """
    print("--- Running Fairness & XAI Analysis ---")
    y_pred = model.predict(X_test)
    
    # --- Fairness Analysis ---
    sensitive_features = X_test['location']
    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0)
    }
    grouped_on_location = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)
    
    fairness_report_str = f"Overall Model Metrics:\n{grouped_on_location.overall}\n\nMetrics by Location Group:\n{grouped_on_location.by_group}"
    fairness_artifact_path = "fairness_report.txt"
    with open(fairness_artifact_path, 'w') as f:
        f.write(fairness_report_str)
    print(f"Fairness report saved to {fairness_artifact_path}")

    # --- Explainability (XAI) Analysis ---
    shap_artifact_path = None  # Initialize to None
    try:
        explainer = shap.TreeExplainer(model)
        
        # --- FIX: Use the explainer object directly, which is more robust ---
        shap_object = explainer(X_test)
        
        # Manually find the index for the desired class
        class_names = model.classes_
        class_index = list(class_names).index(class_name_to_explain)

        plt.figure()
        plt.title(f"SHAP Feature Importance for '{class_name_to_explain}' Class")
        
        # Use the base_values and values for the specific class
        shap.summary_plot(
            shap_object.values[:, :, class_index], 
            X_test,
            show=False
        )
        
        shap_artifact_path = f"shap_summary_{class_name_to_explain}.png"
        plt.savefig(shap_artifact_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP plot saved to {shap_artifact_path}")
        
    except Exception as e:
        print(f"AN ERROR OCCURRED DURING SHAP PLOTTING: {e}")
        
    return fairness_artifact_path, shap_artifact_path
