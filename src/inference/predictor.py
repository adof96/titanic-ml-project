import pandas as pd


def predecir_pasajero(
    modelo,
    pclass,
    sex,
    age,
    fare,
    family_size,
    is_alone,
    title
):
    """
    Realiza una predicción para un nuevo pasajero.

    Parameters
    ----------
    modelo : Pipeline
        Modelo o Pipeline entrenado.

    pclass : int
    sex : str
    age : float
    fare : float
    family_size : int
    is_alone : int
    title : str

    Returns
    -------
    tuple
        (prediccion, probabilidades)
    """

    nuevo_pasajero = pd.DataFrame({

        "Pclass": [pclass],

        "Sex": [sex],

        "Age": [age],

        "Fare": [fare],

        "FamilySize": [family_size],

        "IsAlone": [is_alone],

        "Title": [title]

    })

    prediccion = modelo.predict(nuevo_pasajero)[0]

    probabilidades = modelo.predict_proba(nuevo_pasajero)[0]

    return prediccion, probabilidades