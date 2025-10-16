# Bank-Term Deposit Subscription Prediction

## Overview
Predict bank clients likely to subscribe to term deposits using ML and explainable AI.

## Structure
- `notebooks/01_data_preprocessing.ipynb`: Data loading, preprocessing, feature engineering
- `notebooks/02_feature_engineering.ipynb`: Feature Engineering
- `notebooks/03_modeling_and_evaluation.ipynb`: Baseline, advanced models and Leaderboard, metrics
- `notebooks/04_explainability.ipynb`: SHAP explainability
- `src/dashboard_st_app.py`: Streamlit app

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing and modeling scripts to train models
3. Launch dashboard: `streamlit run src/dashboard_st_app.py`

## Deployment
Deploy Streamlit app to cloud

## Deliverables
- Cleaned/preprocessed dataset
- Modeling notebooks/scripts
- Leaderboard-style comparison
- SHAP explainability
- Deployed the dashboard in the Stremlit community cloud 
- API is created and hosted in Google Cloud to predict the single customer subscription possibilities