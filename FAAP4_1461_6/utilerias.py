from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from Clustering import Cluster

from Clasificador import Clasificador
from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada

# Funciones para facilitar la presentación de datos en el Notebook


def _calcula_resultados_knn(
    Clasificador_: Clasificador, K: int, dataset: str, normaliza: bool
):
    """
    Calcula y devuelve los resultados de KNN para un conjunto de datos específico.
    """
    datos = Datos(f"{dataset}.csv")
    clasificador = Clasificador_(K=K, normaliza=normaliza)
    validacion_cruzada = ValidacionCruzada(numeroParticiones=5)

    resultados = clasificador.validacion(validacion_cruzada, datos, clasificador)

    return (K, normaliza, dataset, resultados[0], resultados[1])


def _resultados_knn(Clasificador_: Clasificador):
    """
    Calcula y devuelve un DataFrame con los resultados de KNN para múltiples configuraciones de K y conjuntos de datos.
    """
    Ks = [1, 3, 5, 11, 21, 31]
    datasets = ["heart", "wdbc"]
    columnas = ["K", "Normaliza", "Dataset", "Error Promedio", "Desviación Estándar"]
    filas = []

    with Pool(processes=4) as pool:
        results = pool.starmap(
            _calcula_resultados_knn,
            [
                (Clasificador_, K, dataset, normaliza)
                for K in Ks
                for dataset in datasets
                for normaliza in [True, False]
            ],
        )

    filas = results

    return pd.DataFrame(filas, columns=columnas)


def _grafica_resultados_knn(resultados_knn: pd.DataFrame, dataset: str):
    """
    Genera un gráfico de resultados KNN para el conjunto de datos 'dataset'.
    Muestra 'Error Promedio' y 'Desviación Estándar' en función de K.
    """
    resultados_dataset = resultados_knn[resultados_knn["Dataset"] == dataset]

    resultados_dataset_normalizado = resultados_dataset[
        resultados_dataset["Normaliza"] == True
    ]
    resultados_dataset_no_normalizado = resultados_dataset[
        resultados_dataset["Normaliza"] == False
    ]

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Resultados KNN para '{dataset}'")

    plt.subplot(1, 2, 1)  # Subplot para 'Normaliza' True
    plt.plot(
        resultados_dataset_normalizado["K"],
        resultados_dataset_normalizado["Error Promedio"],
        label="Error Promedio",
    )
    plt.fill_between(
        resultados_dataset_normalizado["K"],
        resultados_dataset_normalizado["Error Promedio"]
        - resultados_dataset_normalizado["Desviación Estándar"],
        resultados_dataset_normalizado["Error Promedio"]
        + resultados_dataset_normalizado["Desviación Estándar"],
        alpha=0.5,
        label="Desviación Estándar",
    )
    plt.title("Datos Normalizados")
    plt.xlabel("K")
    plt.ylabel("Error Promedio")

    plt.subplot(1, 2, 2)  # Subplot para 'Normaliza' False
    plt.plot(
        resultados_dataset_no_normalizado["K"],
        resultados_dataset_no_normalizado["Error Promedio"],
        label="Error Promedio",
    )
    plt.fill_between(
        resultados_dataset_no_normalizado["K"],
        resultados_dataset_no_normalizado["Error Promedio"]
        - resultados_dataset_no_normalizado["Desviación Estándar"],
        resultados_dataset_no_normalizado["Error Promedio"]
        + resultados_dataset_no_normalizado["Desviación Estándar"],
        alpha=0.5,
        label="Desviación Estándar",
    )
    plt.title("Datos No Normalizados")
    plt.xlabel("K")
    plt.ylabel("Error Promedio")

    plt.tight_layout()
    plt.show()


def _resumen_de_clusters(datos: Datos, clusters: List[Cluster]) -> pd.DataFrame:
    """
    Genera un resumen de los clusters y la distribución de clases en cada uno.
    """
    columnas = ["Cluster", "Setosa", "Versicolor", "Virginica", "Clase Mayoritaria"]
    filas = []

    for i_cluster, cluster in enumerate(clusters):
        miembros = datos.extraeDatos(cluster.i_miembros)
        cuenta_clases = miembros["Class"].value_counts()

        cuenta_setosa = cuenta_clases.get(0, 0)
        cuenta_versicolor = cuenta_clases.get(1, 0)
        cuenta_virginica = cuenta_clases.get(2, 0)

        if cuenta_setosa > max(cuenta_virginica, cuenta_versicolor):
            clase_mayoritaria = "Setosa"
        elif cuenta_versicolor > max(cuenta_virginica, cuenta_setosa):
            clase_mayoritaria = "Versicolor"
        else:
            clase_mayoritaria = "Virginica"

        filas.append(
            (
                i_cluster,
                cuenta_setosa,
                cuenta_versicolor,
                cuenta_virginica,
                clase_mayoritaria,
            )
        )

    df = pd.DataFrame(filas, columns=columnas)
    df = df.set_index("Cluster")

    return df


def _calcular_error(matriz_de_confusion: pd.DataFrame):
    matriz_de_confusion["Clase Mayoritaria"] = matriz_de_confusion[
        "Clase Mayoritaria"
    ].astype(str)

    clasificaciones_correctas = 0
    clasificaciones_incorrectas = 0

    for index, row in matriz_de_confusion.iterrows():
        clase_mayoritaria = row["Clase Mayoritaria"]
        for clase, cantidad in row.items():
            if clase != "Clase Mayoritaria":
                if clase == clase_mayoritaria:
                    clasificaciones_correctas += cantidad
                else:
                    clasificaciones_incorrectas += cantidad

    total_clasificaciones = clasificaciones_correctas + clasificaciones_incorrectas
    error = clasificaciones_incorrectas / total_clasificaciones

    return error
