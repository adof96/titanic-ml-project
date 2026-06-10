def agregar_tamano_familia(df):

    df['FamilySize'] = (
        df['SibSp']
        + df['Parch']
        + 1
    )

    return df

def agregar_is_alone(df):

    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    return df


def agregar_title(df):

    df = df.copy()

    df['Title'] = (
        df['Name']
        .str.extract(r' ([A-Za-z]+)\.', expand=False)
    )

    titulos_comunes = [
        'Mr',
        'Mrs',
        'Miss',
        'Master'
    ]

    df['Title'] = df['Title'].apply(
        lambda x: x if x in titulos_comunes else 'Rare'
    )

    return df