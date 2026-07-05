# src/evaluation.py

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = model.decision_function(X_test)

    return {

        "Accuracy":
            accuracy_score(y_test, y_pred),

        "Precision":
            precision_score(y_test, y_pred),

        "Recall":
            recall_score(y_test, y_pred),

        "F1 Score":
            f1_score(y_test, y_pred),

        "AUC":
            roc_auc_score(y_test, y_prob),

        "Hamming Loss":
            hamming_loss(y_test, y_pred),

        "Jaccard Score":
            jaccard_score(y_test, y_pred),

        "Log Loss":
            log_loss(y_test, y_prob),

        "MCC":
            matthews_corrcoef(y_test, y_pred)
    }


def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        results[name] = evaluate_model(
            model,
            X_test,
            y_test
        )

    return pd.DataFrame(results).T


def compare_models(models, X_test, y_test):

    results_df = evaluate_models(
        models,
        X_test,
        y_test
    )

    return results_df.sort_values(
        by="Accuracy",
        ascending=False
    )