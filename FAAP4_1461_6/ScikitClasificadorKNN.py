from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Clasificador import Clasificador


class ScikitClasificadorKNN(Clasificador):
    def __init__(self, K: int, normaliza=False):
        """
        Inicializa un clasificador K-Nearest Neighbors (KNN) utilizando Scikit-Learn.

        Args:
            K (int): Número de vecinos a considerar para la clasificación.
            normaliza (bool): Indica si se deben normalizar los datos.
        """
        self.K = K
        self.normaliza = normaliza
        if normaliza:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        """
        Realiza el entrenamiento del clasificador KNN utilizando Scikit-Learn.

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

        self.model = KNeighborsClassifier(n_neighbors=self.K, metric="euclidean")
        self.model.fit(X_train, y_train)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        """
        Realiza la clasificación de nuevos datos utilizando el clasificador KNN de Scikit-Learn.

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

        preds = self.model.predict(X_test)

        return preds
