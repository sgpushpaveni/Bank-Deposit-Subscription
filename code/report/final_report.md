# Final Report: Bank Term Deposit Subscription Prediction

## Problem Statement
Banks need to predict which clients will subscribe to term deposits.

## Approach
- Preprocessing: Encoded features, engineered interactions, managed missing values
- Modeling: Baseline (LogReg, RF), advanced (XGBoost, LGBM, DNN)
- Validation: Stratified K-Fold, leaderboard comparison
- Explainability: Feature importance, SHAP analysis
- Deployment: Streamlit dashboard, AWS-ready

## Results
- XGBoost achieved best ROC-AUC
- Duration, balance, education, previous outcome were top features
- SHAP analysis provided actionable insights

## Recommendations
- Target high-duration, high-balance clients for marketing
- Use dashboard for live predictions

## Next Steps
- Integrate with live campaign data
- Continual model monitoring and retraining