import shap
import joblib

def explain_model(model, X_sample, feature_names=None):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    shap.dependence_plot(0, shap_values, X_sample, feature_names=feature_names)

if __name__ == "__main__":
    from preprocessing import load_data, build_preprocessor, preprocess
    df = load_data("data/sample.csv")
    preprocessor = build_preprocessor()
    X, y = preprocess(df, preprocessor)
    model = joblib.load("models/XGBoost.joblib")
    explain_model(model, X[:100])