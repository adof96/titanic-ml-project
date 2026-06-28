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

def entrenar_pipeline(df):

    # Features
    X = df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title']]
    # Target
    y = df['Survived']

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

        ('classifier', LogisticRegression(max_iter=1000))

    ])

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, y_test, predictions