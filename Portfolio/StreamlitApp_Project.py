import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import load


# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset = pd.read_csv(file_path)
# Drop the unnamed index column if present
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Access the AWS secrets
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ── LOAN DEFAULT MODEL CONFIG ─────────────────────────────────────────────
# Inputs are based on the top SHAP features from training:
# int_rate, term, fico_avg, emp_length, dti
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "shap_explainer.pkl",
    "pipeline"  : "finalized_loan_default_model.tar.gz",
    "keys"      : ['int_rate', 'term', 'fico_avg', 'emp_length', 'dti'],
    "inputs": [
        {"name": "int_rate",   "label": "Interest Rate (%)",       "type": "number",   "min": 5.0,   "max": 29.0,  "default": 13.5,  "step": 0.1},
        {"name": "term",       "label": "Loan Term (months)",      "type": "select",   "options": [36, 60],          "default": 36},
        {"name": "fico_avg",   "label": "Average FICO Score",      "type": "number",   "min": 660.0, "max": 850.0, "default": 700.0, "step": 5.0},
        {"name": "emp_length", "label": "Employment Length (yrs)", "type": "number",   "min": 0.0,   "max": 10.0,  "default": 5.0,   "step": 1.0},
        {"name": "dti",        "label": "Debt-to-Income Ratio",    "type": "number",   "min": 0.0,   "max": 38.0,  "default": 18.0,  "step": 0.5},
    ]
}


def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)


# ── Prediction Logic ──────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # ← LOAN-DEFAULT MAPPING (changed from fraud)
        mapping = {0: "✅ Approve (Low Default Risk)", 1: "⚠️ High Default Risk"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── Local Explainability ──────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Apply only the preprocessing step (skip SMOTE + classifier)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df = pd.DataFrame(input_df)
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    # Recover transformed feature names
    try:
        feature_names = preprocessing_pipeline.get_feature_names_out()
        feature_names = [n.split('__', 1)[-1] for n in feature_names]
    except Exception:
        feature_names = dataset.columns.tolist()

    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0])  # binary classifier — single output
    st.pyplot(fig)

    top_feature = pd.Series(
        shap_values[0].values, index=shap_values[0].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Risk Predictor")
st.markdown("Predict whether a loan applicant is likely to default. Adjust the key drivers below and run the model.")

with st.form("pred_form"):
    st.subheader("Applicant Information")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            if inp.get("type") == "select":
                user_inputs[inp['name']] = st.selectbox(
                    inp['label'],
                    options=inp['options'],
                    index=inp['options'].index(inp['default']),
                )
            else:
                user_inputs[inp['name']] = st.number_input(
                    inp['label'],
                    min_value=inp['min'],
                    max_value=inp['max'],
                    value=inp['default'],
                    step=inp['step'],
                )

    submitted = st.form_submit_button("Run Prediction")

# Build the full input row by starting from a sample of training data
# and overlaying the user's inputs for the top features.
sample_row = dataset.iloc[0].to_dict()       # flat dict of scalars
sample_row.update(user_inputs)               # override with user values

if submitted:
    res, status = call_model_api(sample_row)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(sample_row, session, aws_bucket)
    else:
        st.error(res)
