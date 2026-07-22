from config import COMMON_TITLES

def agregar_tamano_familia(df):
    """
    Crea la variable FamilySize a partir de SibSp y Parch.
    """
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    return df

def agregar_is_alone(df):
    """
    Crea la variable IsAlone a partir de FamilySize.
    """
    df = df.copy()

    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    return df


def agregar_title(df):
    """
    Extrae y agrupa el título del pasajero a partir de Name.
    """
    df = df.copy()

    df["Title"] = (
        df["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
    )

    df["Title"] = df["Title"].apply(
        lambda x: x if x in COMMON_TITLES else "Rare"
    )

    return df

def construir_features(df):
    """
    Aplica todo el flujo de Feature Engineering del proyecto.

    Esta función centraliza la creación de todas las variables
    derivadas utilizadas por el modelo, garantizando que el mismo
    proceso se aplique tanto durante el entrenamiento como durante
    la inferencia.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame original.

    Returns
    -------
    pandas.DataFrame
        DataFrame con todas las variables derivadas.
    """


    df = agregar_tamano_familia(df)

    df = agregar_is_alone(df)

    df = agregar_title(df)

    return df