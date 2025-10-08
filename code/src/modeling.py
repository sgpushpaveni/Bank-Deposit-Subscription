import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "SVM": SVC(probability=True),
        "NaiveBayes": GaussianNB(),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
    }
    return models

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "score": model.score(X_test, y_test)
        }
        joblib.dump(model, f"models/{name}.joblib")
    return results, X_test, y_test

if __name__ == "__main__":
    from preprocessing import load_data, build_preprocessor, preprocess
    df = load_data("data/sample.csv")
    preprocessor = build_preprocessor()
    X, y = preprocess(df, preprocessor)
    results, X_test, y_test = train_and_evaluate(X, y)
    print({k: v["score"] for k,v in results.items()})