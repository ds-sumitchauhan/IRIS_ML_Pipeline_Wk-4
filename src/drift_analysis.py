# src/drift_analysis.py
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def run_drift_detection(reference_data):
    """Runs data drift detection and returns path to the saved report."""
    print("--- Running Data Drift Analysis ---")
    current_data = reference_data.copy()
    virginica_indices = current_data[current_data['species'] == 'virginica'].index
    drift_magnitude = 1.5
    current_data.loc[virginica_indices, 'petal_length'] += drift_magnitude
    print(f"Simulated drift by increasing 'petal_length' for 'virginica'.")

    data_drift_report = Report(metrics=[DataDriftPreset()])
    my_eval = data_drift_report.run(reference_data=reference_data, current_data=current_data)

    drift_artifact_path = "data_drift_report.html"
    my_eval.save_html(drift_artifact_path)
    print(f"Data drift report saved to {drift_artifact_path}")

    return drift_artifact_path
