from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def calcular_metricas(y_test, predictions):
    """
    Calcula las métricas principales de un modelo de clasificación.

    Parameters
    ----------
    y_test : array-like
        Valores reales del set de prueba.

    predictions : array-like
        Valores predichos por el modelo.

    Returns
    -------
    dict
        Diccionario con la matriz de confusión y classification report.
    """

    cm = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return {
        "confusion_matrix": cm,
        "classification_report": report
    }


def evaluar_modelo(y_test, predictions, model_name="Modelo"):
    """
    Evalúa un modelo de clasificación mostrando matriz de confusión
    y classification report.

    Parameters
    ----------
    y_test : array-like
        Valores reales del set de prueba.

    predictions : array-like
        Valores predichos por el modelo.

    model_name : str
        Nombre del modelo.
    """

    # Calcular métricas
    metricas = calcular_metricas(
        y_test,
        predictions
    )

    cm = metricas["confusion_matrix"]
    report = metricas["classification_report"]

    # -------------------------
    # Matriz de confusión
    # -------------------------

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    # -------------------------
    # Classification Report
    # -------------------------

    report_df = pd.DataFrame(report).T

    print(f"=== Classification Report: {model_name} ===")
    print(report_df)