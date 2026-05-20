def agregar_tamano_familia(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    return df