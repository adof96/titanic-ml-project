import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE


def optimizar_random_forest(df):
    """
    Ajusta hiperparametros de un RandomForest usando GridSearchCV.

    IMPORTANTE: a diferencia de `entrenar_pipeline` (pipeline_model.py),
    esta funcion NO usa un ColumnTransformer/OneHotEncoder dentro de un
    Pipeline. Aqui la codificacion de variables categoricas se hace de
    forma manual con pd.get_dummies ANTES del split, por lo que el
    X_test que retorna ya viene con columnas dummy (ej. 'Sex_male',
    'Title_Mr', 'Title_Mrs', 'Title_Rare') en vez de las columnas
    originales 'Sex' y 'Title'.

    Por eso el modelo que retorna esta funcion (grid_search.best_estimator_)
    y su X_test/y_test SOLO son compatibles entre si. NO deben mezclarse
    con el `pipeline_model` ni con el `X_test`/`y_test` de
    `entrenar_pipeline`, ni asignarse a variables con el mismo nombre en
    el notebook, porque eso sobrescribe los datos "crudos" que el
    Pipeline completo necesita para predecir.
    """

    X = df[FEATURES]
    y = df[TARGET]

    # Codificacion manual de categoricas (Sex, Title -> dummies)
    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(
        X_train_encoded,
        y_train
    )

    print("Best Parameters:")
    print(grid_search.best_params_)

    print("Best Score:")
    print(grid_search.best_score_)

    return (
        grid_search.best_estimator_,
        X_test_encoded,
        y_test
    )