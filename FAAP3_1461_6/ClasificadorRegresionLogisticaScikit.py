import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from Clasificador import Clasificador


class ClasificadorRegresionLogisticaScikit(Clasificador):
    def __init__(self, K: int, epocas: int, normaliza=False):
        """
        Inicializa un clasificador de regresión logística con Scikit Learn.

        Args:
            K (int): Número de iteraciones de ajuste de pesos.
            epocas (int): Número de épocas (iteraciones completas sobre los datos de entrenamiento).
        """
        self.K = K
        self.epocas = epocas
        self.normaliza = normaliza

        self.modelo = SGDClassifier(
            max_iter=epocas, loss="log_loss", tol=None, eta0=K, learning_rate="constant"
        )

        if self.normaliza:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

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
        X_train = datosTrain.iloc[:, :-1]
        y_train = datosTrain.iloc[:, -1]

        if self.normaliza:
            X_train = self.scaler.fit_transform(X_train)

        self.modelo.fit(X_train, y_train)

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
        X_test = datosTest.iloc[:, :-1]

        if self.normaliza:
            X_test = self.scaler.transform(X_test)

        predicciones = self.modelo.predict(X_test)

        return predicciones
