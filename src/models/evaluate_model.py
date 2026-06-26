from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt


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
        Nombre del modelo (ej. "Logistic Regression", "Random Forest").
        Se usa en el título del gráfico y en el print del reporte,
        para identificar de un vistazo a qué modelo pertenece cada resultado.
    """

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show()

    # Classification Report
    report = classification_report(y_test, predictions)

    print(f"=== Classification Report: {model_name} ===")
    print(report)