
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE

def optimizar_random_forest(df):
    X = df[FEATURES]

    y = df[TARGET]

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        random_state= RANDOM_STATE
    )
    param_grid = {

        'n_estimators': [
            50,
            100,
            200
        ],

        'max_depth': [
            3,
            5,
            10,
            None
        ],

        'min_samples_split': [
            2,
            5,
            10
        ]
    }

    grid_search = GridSearchCV(

        estimator=rf,

        param_grid=param_grid,

        cv=5,

        scoring='accuracy',

        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print(
        "Best Parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "Best Score:"
    )

    print(
        grid_search.best_score_
    )

    return (
        grid_search.best_estimator_,
        X_test,
        y_test
    )