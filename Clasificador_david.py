from abc import ABCMeta,abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
  # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
  # datosTest: matriz numpy o dataframe con los datos de validaci�n
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
  def clasifica(self,datosTest,nominalAtributos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    # devuelve el error
	  pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  #def validacion(self,particionado,dataset,clasificador,seed=None):
   # pass
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
    # devuelve el vector con los errores por cada partici�n
    
    # pasos
    # crear particiones
    # inicializar vector de errores
    # for cada partici�n
    #     obtener datos de train
    #     obtener datos de test
    #     entrenar sobre los datos de train
    #     obtener prediciones de los datos de test (llamando a clasifica)
    #     a�adir error de la partici�n al vector de errores

class Datos:
    def __init__(self, nombreFichero: str):
        self.datosCrudos = pd.read_csv(nombreFichero)
        self.datos = self.datosCrudos.copy()

        self.nominalAtributos = []
        self.diccionarios = {}

        for columna in self.datos.columns:
            if self._es_nominal(columna):
                self.nominalAtributos.append(True)
                self.diccionarios[columna] = self._generar_mapeo(columna)
                self.datos[columna] = self._reemplazar_valores(columna)
            elif self._es_numerico(columna):
                self.nominalAtributos.append(False)
                self.diccionarios[columna] = {}
            else:
                raise ValueError(
                    f"La columna '{columna}' contiene valores que no son nominales ni enteros/decimales."
                )
    def _es_nominal(self, columna: str) -> bool:
        # es verdadero si es la columna objetivo o si sus valores son nominales
        return columna.lower() == "class" or self.datos[columna].dtype.name == "object"

    def _es_numerico(self, columna: str) -> bool:
        # es verdadero si los valores son un n�meros enteros o reales
        return self.datos[columna].dtype.name in ["int64", "float64"]

    def _generar_mapeo(self, columna_nominal: str):
        # se extraen los valores �nicos de la columna y se sortean lexicograficamente
        valores = [str(valor) for valor in self.datos[columna_nominal].unique()]
        valores = sorted(valores)

        return {valor: indice for indice, valor in enumerate(valores)}

    def _reemplazar_valores(self, columna_nominal: str) -> pd.Series:
        mapeo = self.diccionarios[columna_nominal]
        return self.datos[columna_nominal].map(lambda valor: mapeo[str(valor)])

    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self, idx: list[int]):
        return self.datos.iloc[idx]


class ClasificadorNaiveBayes:
    def __init__(self, csv_file):
        self.datos = Datos(csv_file).datos

        l_col = len(self.datos.columns) - 1
        self.y = self.datos[self.datos.columns[l_col]]
        self.X = self.datos.drop(self.datos.columns[l_col], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        self.GNB = GaussianNB()
        self.GNB.fit(self.X_train, self.y_train)

        self.pred = self.GNB.predict(self.X_test)

    def calculate_accuracy(self):
        accuracy = metrics.accuracy_score(self.y_test, self.pred) * 100
        print("Accuracy %:", accuracy)

# Beispiel der Verwendung der Klasse
classifier = ClasificadorNaiveBayes("heart.csv")
classifier.calculate_accuracy()