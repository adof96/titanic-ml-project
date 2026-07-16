from config import COMMON_TITLES

def agregar_tamano_familia(df):

    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    return df

def agregar_is_alone(df):

    df = df.copy()

    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    return df


def agregar_title(df):

    df = df.copy()

    df["Title"] = (
        df["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
    )

    df["Title"] = df["Title"].apply(
        lambda x: x if x in COMMON_TITLES else "Rare"
    )

    return df