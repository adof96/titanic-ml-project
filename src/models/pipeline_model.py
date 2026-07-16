import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer
from config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE , NUMERIC_FEATURES, CATEGORICAL_FEATURES

def entrenar_pipeline(df, classifier):

    # Features
    X = df[FEATURES]
    # Target
    y = df[TARGET]

# Columnas numéricas
    numeric_features = NUMERIC_FEATURES

# Columnas categóricas
    categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='median')),

        ('scaler', StandardScaler())

    ])

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(drop='first'))

    ])

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numeric_transformer, numeric_features),

            ('cat', categorical_transformer, categorical_features)

        ]
    )

    model = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('classifier', classifier)

    ])

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state= RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions