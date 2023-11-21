from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Datos import Datos
from EstrategiaParticionado import ValidacionSimple

# Funciones para facilitar la presentación de datos en el Notebook

# -- APARTADO 1 Y 2 - Regresion Logistica -- #


def calcula_resultados_regresion_logistica(Clasificador, K, epocas, dataset, normaliza):
    """
    Calcula y devuelve los resultados de Regresión Logística para un conjunto de datos específico.
    """
    datos = Datos(f"{dataset}.csv")
    clasificador = Clasificador(K=K, epocas=epocas, normaliza=normaliza)
    validacion_simple = ValidacionSimple(numeroEjecuciones=3, proporcionTest=30)

    resultados = clasificador.validacion(validacion_simple, datos, clasificador)

    # K, epocas, es normalizado?, error promedio, error estandar
    return (K, epocas, normaliza, resultados[0], resultados[1])


def resultados_regresion_logistica(Clasificador, dataset, Ks, epocas):
    """
    Calcula y devuelve un DataFrame con los resultados de Regresion Logística para múltiples configuraciones de K y conjuntos de datos.
    """
    columnas = [
        "K",
        "Epocas",
        "Normaliza",
        "Error Promedio",
        "Desviación Estándar",
    ]
    filas = []

    with Pool(processes=4) as pool:
        filas = pool.starmap(
            calcula_resultados_regresion_logistica,
            [
                (Clasificador, K, epoca, dataset, normaliza)
                for K in Ks
                for epoca in epocas
                for normaliza in [True, False]
            ],
        )

    return (
        pd.DataFrame(filas, columns=columnas)
        .sort_values("Error Promedio")
        .reset_index(drop=True)
    )


def grafica_lineal_error_promedio_vs_epocas(df):
    """
    Genera un gráfico de líneas que muestra la evolución del error promedio en función del número de épocas.

    Args:
        df (pd.DataFrame): DataFrame que contiene los resultados de la regresión logística.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    for normaliza, group in df.groupby("Normaliza"):
        for k, subgroup in group.groupby("K"):
            plt.plot(
                subgroup["Epocas"],
                subgroup["Error Promedio"],
                label=f"K={k}, Normaliza={normaliza}",
            )
    plt.xlabel("Épocas")
    plt.ylabel("Error Promedio")
    plt.legend(title="Configuración")
    plt.title("Error Promedio vs. Épocas")
    plt.show()


def grafica_de_barras_error_promedio(df):
    """
    Genera un gráfico de líneas que muestra la evolución del error promedio en función del número de épocas.

    Args:
        df (pd.DataFrame): DataFrame que contiene los resultados de la regresión logística.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    df_grouped = df.groupby(["K", "Normaliza"]).mean()["Error Promedio"].unstack()
    df_grouped.plot(kind="bar", stacked=True)
    plt.xlabel("K")
    plt.ylabel("Error Promedio Promedio")
    plt.title("Error Promedio Promedio por Configuración de K y Normaliza")
    plt.show()


# -- APARTADO 3 - Análisis ROC -- #


def crea_particion_simple(datos):
    validacion_simple = ValidacionSimple(numeroEjecuciones=1, proporcionTest=30)
    validacion_simple.creaParticiones(datos.datos)

    return validacion_simple.particiones[0]


def calcular_tpr_fnr(matriz_confusion):
    """
    Calcula el TPR (True Positive Rate) y el FNR (False Negative Rate) a partir de una matriz de confusión.

    Args:
        matriz_confusion (numpy.ndarray): Matriz de confusión de la forma [[TN, FP], [FN, TP]].

    Returns:
        float: TPR (True Positive Rate).
        float: FNR (False Negative Rate).
    """
    TP = matriz_confusion[0, 0]  # Verdaderos Positivos
    TN = matriz_confusion[1, 1]  # Verdaderos Negativos
    FP = matriz_confusion[0, 1]  # Falsos Positivos
    FN = matriz_confusion[1, 0]  # Falsos Negativos

    TPR = TP / (TP + FN)  # True Positive Rate
    FNR = FP / (FP + TN)  # False Positive Rate

    return TPR, FNR


def obtener_tpr_fpr(clasificador, datos, particion):
    """
    Calcula el True Positive Rate (TPR) y False Positive Rate (FPR) para un clasificador dado y una partición de datos.

    Parameters:
        clasificador (Clasificador): El clasificador a evaluar.
        datos (Datos): El conjunto de datos completo.
        particion (Particion): La partición de datos que incluye los índices de entrenamiento y prueba.

    Returns:
        float, float: TPR y FPR calculados para el clasificador y la partición proporcionados.
    """
    datos_test = datos.extraeDatos(particion.indicesTest)
    datos_train = datos.extraeDatos(particion.indicesTrain)

    objetivo = np.array(datos_test.iloc[:, -1])

    # Entrenar el clasificador
    clasificador.entrenamiento(datos_train, datos.nominalAtributos, datos.diccionarios)

    # Realizar predicciones
    predicciones = clasificador.clasifica(
        datos_test, datos.nominalAtributos, datos.diccionarios
    )

    # Calcular la matriz de confusión
    matriz_confusion = clasificador.matriz_de_confusion(objetivo, predicciones)

    # Calcular TPR y FPR
    TPR, FPR = calcular_tpr_fnr(matriz_confusion)

    return TPR, FPR


def grafica_espacio_ROC(
    nombre_dataset, rates_knn, rates_naive_bayes, rates_regresion_logistica
):
    """
    Grafica el espacio ROC para comparar los clasificadores KNN, Naive Bayes y Regresión Logística.

    Parameters:
    - nombre_dataset (str): Nombre del conjunto de datos.
    - rates_knn (tuple): Tasa de verdaderos positivos (TPR) y tasa de falsos positivos (FPR) para KNN.
    - rates_naive_bayes (tuple): TPR y FPR para Naive Bayes.
    - rates_regresion_logistica (tuple): TPR y FPR para Regresión Logística.

    Returns:
    None
    """
    TPR_knn, FPR_knn = rates_knn
    TPR_naive_bayes, FPR_naive_bayes = rates_naive_bayes
    TPR_regresion_logistica, FPR_regresion_logistica = rates_regresion_logistica

    plt.plot([0, 1], [0, 1], "--", color="red")

    # Configuraciones del gráfico
    plt.title(f"Espacio ROC para dataset {nombre_dataset}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)

    plt.scatter(FPR_knn, TPR_knn, color="blue", label="KNN")
    plt.scatter(FPR_naive_bayes, TPR_naive_bayes, color="orange", label="Naive Bayes")
    plt.scatter(
        FPR_regresion_logistica,
        TPR_regresion_logistica,
        color="green",
        label="Regresión Logística",
    )

    plt.legend()

    # Ajustar los límites del eje X e Y para comenzar en 0 y terminar en 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Mostrar el gráfico
    plt.show()


# -- APARTADO 4 - Curva ROC -- #


def grafica_curva_ROC(nombre_dataset, clasificador, datos, particion):
    """
    Grafica la curva ROC para un clasificador dado utilizando el algoritmo de Fawcett.

    Args:
        nombre_dataset (str): Nombre del dataset.
        clasificador: Objeto del clasificador que debe tener los métodos 'entrenamiento' y 'calcula_scores_ROC'.
        datos: Objeto de datos que debe tener los métodos 'extraeDatos' y atributos 'nominalAtributos' y 'diccionarios'.
        particion: Objeto de partición que debe tener el atributo 'indicesTest'.

    Returns:
        None: Muestra la gráfica de la curva ROC.
    """
    datos_test = datos.extraeDatos(particion.indicesTest)
    datos_train = datos.extraeDatos(particion.indicesTrain)

    # Entrenar el clasificador
    clasificador.entrenamiento(datos_train, datos.nominalAtributos, datos.diccionarios)

    # -- Algoritmo de Fawcett -- #

    # 1. Calcular scores para instancias
    scores = clasificador.calcula_scores_ROC(
        datos_test, datos.nominalAtributos, datos.diccionarios
    )
    # eliminar elementos que no son TP o FP
    scores = [score for score in scores if score[2] or score[3]]

    # 2. Ordenamiento de instancias
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # 3. Calcular los puntos, desplazando en Y si es TP y desplazando en X si es FP
    x, y = obtener_puntos_ROC(scores)

    # 4. Dibujar la curva
    plt.plot([0, 1], [0, 1], "--", color="red")

    # Configuraciones del gráfico
    plt.title(f"Curva ROC para dataset {nombre_dataset}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)

    # Ajustar los límites del eje X e Y para comenzar en 0 y terminar en 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.plot(x, y, label="Curva ROC")

    # Mostrar el gráfico
    plt.show()


def obtener_puntos_ROC(scores_ordenados):
    """
    Obtiene los puntos para la construcción de la curva ROC a partir de los scores ordenados.

    Args:
        scores_ordenados (list): Lista de tuplas con scores ordenados y etiquetas de verdad.

    Returns:
        tuple: Tupla con dos listas, una para las coordenadas X y otra para las coordenadas Y de la curva ROC.
    """
    Xs, Ys = [], []
    x, y = 0, 0

    for _, _, tp, fp in scores_ordenados:
        if tp:
            y += 1
        elif fp:
            x += 1

        Xs.append(x)
        Ys.append(y)

    return normaliza_puntos(Xs, Ys)


def normaliza_puntos(Xs, Ys):
    """
    Normaliza las coordenadas X e Y de una curva ROC con base en el valor mas alto de las listas de entrada.

    Args:
        Xs (list): Lista de coordenadas X.
        Ys (list): Lista de coordenadas Y.

    Returns:
        tuple: Tupla con dos listas normalizadas, una para las coordenadas X y otra para las coordenadas Y.
    """
    Xs_norm, Ys_norm = [], []

    x_maximo = max(Xs)
    y_maximo = max(Ys)

    for x, y in zip(Xs, Ys):
        Xs_norm.append(x / x_maximo)
        Ys_norm.append(y / y_maximo)

    return Xs_norm, Ys_norm
