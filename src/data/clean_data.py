import pandas as pd

def revisar_nulos(df):
    return df.isnull().sum()


def imputar_datos(df):

    # Age → mediana
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Embarked → moda
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Cabin → eliminar columna
    df = df.drop(columns=['Cabin'])

    return df