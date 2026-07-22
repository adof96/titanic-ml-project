import pandas as pd
from config import FEATURES
from features.build_features import construir_features


def predecir_pasajero(
    modelo,
    name,
    pclass,
    sex,
    age,
    fare,
    sibsp,
    parch
):
    """
    Realiza una predicción para un nuevo pasajero.
    """

    nuevo_pasajero = pd.DataFrame({

        "Name": [name],

        "Pclass": [pclass],

        "Sex": [sex],

        "Age": [age],

        "Fare": [fare],

        "SibSp": [sibsp],

        "Parch": [parch]

    })

    # ===== Feature Engineering =====

    nuevo_pasajero = construir_features(nuevo_pasajero)

    # Mantener únicamente las variables que espera el modelo

    nuevo_pasajero = nuevo_pasajero[FEATURES]

    prediccion = modelo.predict(nuevo_pasajero)[0]

    probabilidades = modelo.predict_proba(nuevo_pasajero)[0]

    resultado = {
    "prediction": prediccion,
    "prob_not_survival": probabilidades[0],
    "prob_survival": probabilidades[1]
}

    return resultado