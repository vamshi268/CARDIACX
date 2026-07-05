# src/stacking.py

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


class CardiacX:

    def __init__(
        self,
        estimators,
        final_estimator=None,
        cv=5,
        n_jobs=-1
    ):

        self.estimators = estimators

        self.final_estimator = (
            final_estimator
            if final_estimator is not None
            else LogisticRegression(
                max_iter=2000,random_state=42
            )
        )

        self.cv = cv
        self.n_jobs = n_jobs

        self.model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.final_estimator,
            cv=self.cv,
            n_jobs=self.n_jobs
        )

    def fit(self, X_train, y_train):

        self.model.fit(
            X_train,
            y_train
        )

        return self

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def get_model(self):

        return self.model