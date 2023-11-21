import numpy as np
from ManipuladorDatos import ManipuladorDatos

from Clasificador import Clasificador


class ClasificadorRegresionLogistica(Clasificador, ManipuladorDatos):
    def __init__(self, K: int, epocas: int, normaliza=False):
        """
        Inicializa un clasificador de regresión logística.

        Args:
            K (int): Número de iteraciones de ajuste de pesos.
            epocas (int): Número de épocas (iteraciones completas sobre los datos de entrenamiento).
        """
        self.K = K
        self.epocas = epocas
        self.normaliza = normaliza

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        """
        Realiza el entrenamiento del clasificador de regresión logística.

        Args:
            datosTrain (pd.DataFrame): DataFrame con los datos de entrenamiento.
            nominalAtributos (List[bool]): Lista de booleanos indicando si los atributos son nominales o no.
            diccionario (dict): Diccionario que mapea nombres de atributos a valores nominales.

        Returns:
            None
        """
        # Inicialización de pesos con valores aleatorios en el rango [-0.5, 0.5]
        self.pesos = np.random.uniform(-0.5, 0.5, datosTrain.shape[1])

        # Normalización de datos
        if self.normaliza:
            self.X_media, self.X_desv = self._calcular_medias_desv(
                datosTrain.iloc[:, :-1]
            )
            datosTrain.iloc[:, :-1] = self._normalizar_datos(
                datosTrain.iloc[:, :-1], self.X_media, self.X_desv
            )

        # Añadir columna para peso w0 (truco del sesgo)
        datosTrain.insert(0, "x0", 1)

        # Iteraciones sobre las épocas para ajustar los pesos
        for _ in range(self.epocas):
            for _, fila in datosTrain.iterrows():
                # Actualización de pesos mediante la función _actualiza_pesos
                self.pesos = self._actualiza_pesos(self.pesos, fila)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        """
        Realiza la clasificación de nuevos datos utilizando el clasificador de regresión logística.

        Args:
            datosTest (pd.DataFrame): DataFrame con los datos de prueba a clasificar.
            nominalAtributos (List[bool]): Lista de booleanos indicando si los atributos son nominales o no.
            diccionario (dict): Diccionario que mapea nombres de atributos a valores nominales.

        Returns:
            np.array: Un arreglo NumPy con las predicciones de clase para los datos de prueba.
        """
        predicciones = []

        # Normalización de datos
        if self.normaliza:
            datosTest.iloc[:, :-1] = self._normalizar_datos(
                datosTest.iloc[:, :-1], self.X_media, self.X_desv
            )

        # Añadir columna para peso w0 (truco del sesgo)
        datosTest.insert(0, "x0", 1)

        X_test = datosTest.iloc[:, :-1]

        for _, fila in X_test.iterrows():
            # Cálculo de la probabilidad de pertenencia a la clase 1 mediante la función sigmoid
            probabilidad = self._sigmoid(np.dot(self.pesos, fila))

            # Asignación de la clase predicha en función de la probabilidad calculada
            if probabilidad > 0.5:
                clase_predicha = 1
            elif probabilidad < 0.5:
                clase_predicha = 0
            else:
                clase_predicha = np.random.randint(0, 1)

            predicciones.append(clase_predicha)

        return np.array(predicciones)

    def calcula_scores_ROC(self, datosTest, nominalAtributos, diccionario):
        """
        Calcula los scores necesarios para construir la curva ROC.

        Args:
            datosTest (pandas.DataFrame): DataFrame con los datos de prueba.
            nominalAtributos (list): Lista de nombres de atributos nominales.
            diccionario (dict): Diccionario que contiene la información de los atributos nominales.

        Returns:
            list: Lista de tuplas que contienen información sobre clase real, probabilidad,
                  verdadero positivo (TP), y falso positivo (FP) para cada instancia de prueba.
        """
        resultados = []

        # Normalización de datos
        if self.normaliza:
            datosTest.iloc[:, :-1] = self._normalizar_datos(
                datosTest.iloc[:, :-1], self.X_media, self.X_desv
            )

        # Añadir columna para peso w0 (truco del sesgo)
        datosTest.insert(0, "x0", 1)

        for _, fila in datosTest.iterrows():
            # Cálculo de la probabilidad de pertenencia a la clase 1 mediante la función sigmoid
            probabilidad = self._sigmoid(np.dot(self.pesos, fila[:-1]))
            clase_real = fila[-1]

            # Verdadero Positivo, Falso Positivo
            tp, fp = False, False

            if probabilidad > 0.5:
                if clase_real == 1:
                    tp = True
                elif clase_real == 0:
                    fp = True

            resultados.append((clase_real, probabilidad, tp, fp))

        return resultados

    def _actualiza_pesos(self, w, x):
        """
        Actualiza los pesos del modelo de regresión logística.

        Args:
            w (np.array): Vector de pesos actual.
            x (pd.Series): Fila de datos de entrenamiento.

        Returns:
            np.array: El vector de pesos actualizado.
        """
        # Actualización de los pesos mediante el método de Máxima Verosimilitud
        clase = x.iloc[-1]
        x = np.array(x.iloc[:-1])
        return w - x * (self._sigmoid(np.dot(self.pesos, x)) - clase) * self.K

    def _sigmoid(self, x):
        """
        Función sigmoide para transformar el resultado de la combinación lineal en el rango [0, 1].

        Args:
            x (float): Resultado de la combinación lineal de pesos y características.

        Returns:
            float: Valor transformado en el rango [0, 1].
        """
        exp = -x

        # Evitar overflow
        if exp > 700:
            return 0
        elif exp < -700:
            return 1

        return 1 / (1 + np.exp(-x))
