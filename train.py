# %% [markdown]
# # Deploying Iris-detection model using Vertex AI
# 

# %% [markdown]
# ## Overview
# 
# In this tutorial, you build a scikit-learn model and deploy it on Vertex AI using the custom container method. You use the FastAPI Python web server framework to create a prediction endpoint. You also incorporate a preprocessor from training pipeline into your online serving application.
# 
# Learn more about [Custom training](https://cloud.google.com/vertex-ai/docs/training/custom-training) and [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions).

# %% [markdown]
# ### Objective
# 
# In this notebook, you learn how to create, deploy and serve a custom classification model on Vertex AI. This notebook focuses more on deploying the model than on the design of the model itself. 
# 
# 
# This tutorial uses the following Vertex AI services and resources:
# 
# - Vertex AI models
# - Vertex AI endpoints
# 
# The steps performed include:
# 
# - Train a model that uses flower's measurements as input to predict the class of iris.
# - Save the model and its serialized pre-processor.
# - Build a FastAPI server to handle predictions and health checks.
# - Build a custom container with model artifacts.
# - Upload and deploy custom container to Vertex AI Endpoints.

# %% [markdown]
# ### Dataset
# 
# This tutorial uses R.A. Fisher's Iris dataset, a small and popular dataset for machine learning experiments. Each instance has four numerical features, which are different measurements of a flower, and a target label that
# categorizes the flower into: **Iris setosa**, **Iris versicolour** and **Iris virginica**.
# 
# This tutorial uses [a version of the Iris dataset available in the
# scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris).

# %% [markdown]
# ### Costs 
# 
# This tutorial uses billable components of Google Cloud:
# 
# * Vertex AI
# * Cloud Storage
# * Artifact Registry
# * Cloud Build
# 
# Learn about [Vertex AI
# pricing](https://cloud.google.com/vertex-ai/pricing), [Cloud Storage
# pricing](https://cloud.google.com/storage/pricing), [Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing) and [Cloud Build pricing](https://cloud.google.com/build/pricing) and use the [Pricing
# Calculator](https://cloud.google.com/products/calculator/)
# to generate a cost estimate based on your projected usage.

# %% [markdown]
# ## Get started

# %% [markdown]
# ### Install Vertex AI SDK for Python and other required packages
# 
# 

# %%

# Vertex SDK for Python
#! pip3 install --upgrade --quiet  google-cloud-aiplatform

# %% [markdown]
# ### Set Google Cloud project information 
# Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

# %%
#PROJECT_ID = "soy-reporter-459913-v8"  # @param {type:"string"}
#LOCATION = "us-central1"  # @param {type:"string"}

# %% [markdown]
# ### Create a Cloud Storage bucket
# 
# Create a storage bucket to store intermediate artifacts such as datasets.

# %%
#BUCKET_URI = f"gs://mlops-course-soy-reporter-459913-v8-unique"  # @param {type:"string"}

# %% [markdown]
# **If your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.

# %%
#! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}

# %% [markdown]
# ### Initialize Vertex AI SDK for Python
# 
# To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com). 

# %%
#from google.cloud import aiplatform

#aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

# %% [markdown]
# ### Import the required libraries

# %%
import os
import sys

# %% [markdown]
# ### Configure resource names
# 
# Set a name for the following parameters:
# 
# `MODEL_ARTIFACT_DIR` - Folder directory path to your model artifacts within a Cloud Storage bucket, for example: "my-models/fraud-detection/trial-4"
# 
# `REPOSITORY` - Name of the Artifact Repository to create or use.
# 
# `IMAGE` - Name of the container image that is pushed to the repository.
# 
# `MODEL_DISPLAY_NAME` - Display name of Vertex AI model resource.

# %%
#MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-1"  # @param {type:"string"}
#REPOSITORY = "iris-classifier-repo"  # @param {type:"string"}
#IMAGE = "iris-classifier-img"  # @param {type:"string"}
#MODEL_DISPLAY_NAME = "iris-classifier"  # @param {type:"string"}

# Set the defaults if no names were specified
#if MODEL_ARTIFACT_DIR == "[your-artifact-directory]":
#    MODEL_ARTIFACT_DIR = "custom-container-prediction-model"

#if REPOSITORY == "[your-repository-name]":
#    REPOSITORY = "custom-container-prediction"

#if IMAGE == "[your-image-name]":
#    IMAGE = "sklearn-fastapi-server"

#if MODEL_DISPLAY_NAME == "[your-model-display-name]":
#    MODEL_DISPLAY_NAME = "sklearn-custom-container"

# %% [markdown]
# ## Simple Decision Tree model
# Build a Decision Tree model on iris data

# %%
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import storage
import os


# Configure GCP Bucket
BUCKET_NAME = "mlops-course-soy-reporter-459913-v8-unique"
MODEL_FILENAME = "iris_hyperparam_tuning_best_model.joblib"
MODEL_LOCAL_PATH = f"outputs/{MODEL_FILENAME}"
MODEL_GCS_PATH = f"Week_5/{MODEL_FILENAME}"

# Load Data
data = pd.read_csv('data/iris.csv')
#data.head(5)

# %%
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Enable MLflow autologging
mlflow.sklearn.autolog()

#Define hyperparameter grid
param_grid = {
        'n_estimators':[10, 50, 100],
        'max_depth' : [3,5,10],
        'min_samples_split':[2,4]
        }

# Start experiment
with mlflow.start_run():
    clf = GridSearchCV(rfc(random_state=42), param_grid, cv = 3)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Best Prams : {clf.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")


    # Save Best model
    os.makedirs("outputs",exist_ok=True)
    joblib.dump(clf.best_estimator_, MODEL_LOCAL_PATH)
    print(f"Saved best model to {MODEL_LOCAL_PATH}")

    # Upload to GCS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_GCS_PATH)
    blob.upload_from_filename(MODEL_LOCAL_PATH)
    print(f"Uploaded model to gs://{BUCKET_NAME}/{MODEL_GCS_PATH}")

    for param_name, param_value in clf.best_params_.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_metric('accuracy', acc)
    input_example = pd.DataFrame(X_test.iloc[:2]) 
    mlflow.sklearn.log_model(
        sk_model=clf.best_estimator_,
        artifact_path='wk-5_best_model',
        input_example=input_example,
        signature=mlflow.models.signature.infer_signature(X_test, y_test)
        )


#mlflow.log_metric("accuracy", acc)
#mlflow.sklearn.log_model(clf.best_estimator_, "week_5_best_model")

                       # %%
#od_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
#od_dt.fit(X_train,y_train)
#rediction=mod_dt.predict(X_test)
#rint('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# %%
# import pickle
#import joblib

#joblib.dump(mod_dt, "model.joblib")

# %% [markdown]
 ### Upload model artifacts and custom code to Cloud Storage
 
# Before you can deploy your model for serving, Vertex AI needs access to the following files in Cloud Storage:
 
# * `model.joblib` (model artifact)
# * `preprocessor.pkl` (model artifact)
 
# Run the following commands to upload your files:

# %%
# !gsutil cp artifacts/model.joblib {BUCKET_URI}/{MODEL_ARTIFACT_DIR}/



