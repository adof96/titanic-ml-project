def supervivencia_por_sexo(df):
    return df.groupby('Sex')['Survived'].mean()

def supervivencia_por_clase(df):
    return df.groupby('Pclass')['Survived'].mean()