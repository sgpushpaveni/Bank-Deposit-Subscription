from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import numpy as np

def leaderboard(models, X_test, y_test):
    results = []
    for name, entry in models.items():
        model = entry["model"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        })
    import pandas as pd
    df = pd.DataFrame(results)
    print(df)
    return df

if __name__ == "__main__":
    from modeling import train_and_evaluate
    # Load previously trained models and test data