# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models(config=None):

    config = config or {}

    models = {

        "Logistic Regression":
            LogisticRegression(
                random_state=42,
                **config.get("Logistic Regression", {})
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42,
                **config.get("Decision Tree", {})
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=42,
                **config.get("Random Forest", {})
            ),

        "KNN":
            KNeighborsClassifier(
                **config.get("KNN", {})
            ),

        "SVM":
            SVC(
                probability=True,
                **config.get("SVM", {})
            ),

        "XGBoost":
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                **config.get("XGBoost", {}),
                use_label_encoder=False
            )
    }

    return models