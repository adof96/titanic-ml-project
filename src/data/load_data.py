import pandas as pd

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df