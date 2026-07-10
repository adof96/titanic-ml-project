import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE , MAX_ITER

def entrenar_pipeline(df):

    # Features
    X = df[FEATURES]
    # Target
    y = df[TARGET]

# Columnas numéricas
    numeric_features = [
        "Pclass",
        "Age",
        "Fare",
        "FamilySize",
        "IsAlone"
    ]   

# Columnas categóricas
    categorical_features = [
        "Sex",
        "Title"
    ]

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

        ('classifier', LogisticRegression(max_iter= MAX_ITER))

    ])

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state= RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, y_test, predictions