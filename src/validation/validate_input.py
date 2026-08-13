def validar_datos(
    name,
    pclass,
    sex,
    age,
    fare,
    sibsp,
    parch
):
    errores = []

    if not name or not name.strip():
        errores.append("El nombre no puede estar vacío.")

    if pclass not in [1, 2, 3]:
        errores.append("La clase del pasajero debe ser 1, 2 o 3.")

    if sex not in ["male", "female"]:
        errores.append("El sexo debe ser válido.")

    if age <= 0:
        errores.append("La edad debe ser mayor que 0.")

    if fare < 0:
        errores.append("El precio del boleto no puede ser negativo.")

    if sibsp < 0:
        errores.append(
            "El número de hermanos o cónyuges no puede ser negativo."
        )

    if parch < 0:
        errores.append(
            "El número de padres o hijos no puede ser negativo."
        )

    return errores