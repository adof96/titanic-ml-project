import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def entrenar_random_forest(df):

    # Features
    X = df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']]

    # Target
    y = df['Survived']

    # Encoding
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicciones
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print('Random Forest Accuracy:', accuracy)

    return model, X, y_test, predictions