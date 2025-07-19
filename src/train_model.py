from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    RocCurveDisplay,
    precision_recall_curve
)
from catboost import CatBoostClassifier, Pool, cv
from imblearn.over_sampling import SMOTE
import joblib
from preprocessing import load_and_preprocess
import os
import matplotlib.pyplot as plt
import numpy as np

def train():  
    df = load_and_preprocess("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric='F1',
        random_seed=42,
        verbose=0,
        cat_features=cat_features 
    )

    model.fit(X_train_bal, y_train_bal)

    y_probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.3, 0.6, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred_temp = (y_probs >= t).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    y_pred = (y_probs >= best_threshold).astype(int)

    metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
    }
    joblib.dump(metrics, "models/metrics.pkl")

    print("\n=== Best Threshold Evaluation ===")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Recall Score:", recall_score(y_test, y_pred))
    print("Precision Score:", precision_score(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(pr_thresholds, precision[:-1], label="Precision")
    plt.plot(pr_thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision vs Recall vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/precision_recall_curve.png", dpi=300)
    plt.show()

    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve for Churn Prediction")
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    train()
