from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from config import (
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def optimizar_random_forest(df):
    """
    Optimiza un RandomForest utilizando GridSearchCV
    sobre un Pipeline completo de preprocessing + modelo.

    El Pipeline se encarga de:
    - Imputar valores numéricos
    - Escalar variables numéricas
    - Imputar variables categóricas
    - Codificar variables categóricas
    - Entrenar el RandomForest

    GridSearchCV optimiza únicamente los hiperparámetros
    del RandomForest dentro del Pipeline.
    """

    X = df[FEATURES]
    y = df[TARGET]

    # ==========================================
    # Preprocesamiento
    # ==========================================

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # ==========================================
    # Modelo
    # ==========================================

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    # ==========================================
    # Train / Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ==========================================
    # Hiperparámetros
    # ==========================================

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 10, None],
        "classifier__min_samples_split": [2, 5, 10]
    }

    # ==========================================
    # GridSearch
    # ==========================================

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    # ==========================================
    # Resultados
    # ==========================================

    print("Best Parameters:")
    print(grid_search.best_params_)

    print("Best Score:")
    print(grid_search.best_score_)

    return (
        grid_search.best_estimator_,
        X_test,
        y_test
    )