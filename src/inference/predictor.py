import pandas as pd
from config import FEATURES
from features.build_features import (
    agregar_tamano_familia,
    agregar_is_alone,
    agregar_title
)


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

    nuevo_pasajero = agregar_tamano_familia(nuevo_pasajero)

    nuevo_pasajero = agregar_is_alone(nuevo_pasajero)

    nuevo_pasajero = agregar_title(nuevo_pasajero)

    # Mantener únicamente las variables que espera el modelo

    nuevo_pasajero = nuevo_pasajero[FEATURES]

    prediccion = modelo.predict(nuevo_pasajero)[0]

    probabilidades = modelo.predict_proba(nuevo_pasajero)[0]

    return prediccion, probabilidades