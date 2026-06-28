import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE

def entrenar_modelo(df):

    # Features
    X = df[FEATURES]

    # Target
    y = df[TARGET]

    # Convertir categorías a números
    X = pd.get_dummies(X, drop_first=True)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Crear modelo
    model = LogisticRegression(max_iter=1000)

    # Entrenar
    model.fit(X_train, y_train)

    # Predicciones
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print('Accuracy:', accuracy)

    return model, y_test, predictions