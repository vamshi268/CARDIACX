# src/tuning.py

import optuna

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)


def grid_search_tune(
    model,
    param_grid,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
):

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )

    search.fit(X_train, y_train)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_
    }


def random_search_tune(
    model,
    param_distributions,
    X_train,
    y_train,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
):

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    )

    search.fit(X_train, y_train)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_
    }


def optuna_tune(
    objective,
    n_trials=30,
    direction="maximize"
):

    study = optuna.create_study(
        direction=direction
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study


def cross_validate(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
):

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores
    }