from abc import ABCMeta,abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

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
  def validacion(self,particionado,dataset,clasificador,seed=None):
      pass
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
import Datos

class ClasificadorNaiveBayes_heart(Clasificador):
    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):

        self.datos = Datos(datosTrain).datos

        encoder = OneHotEncoder(sparse=False, drop='first')  # 'first' um Dummy-Variablen-Falle zu vermeiden
        self.datos = encoder.fit_transform(self.datos[nominalAtributos])

        l_col = len(self.datos.columns) - 1
        self.y = self.datos[self.datos.columns[l_col]]
        self.X = self.datos.drop(self.datos.columns[l_col], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)


        return super().entrenamiento(datosTrain, nominalAtributos, diccionario)
    
    def clasifica(self, datosTest, nominalAtributos, diccionario):

        self.GNB = GaussianNB()
        self.GNB.fit(self.X_train, self.y_train)

        return super().clasifica(datosTest, nominalAtributos, diccionario)
    
    def validacion(self, particionado, dataset, clasificador, seed=None):


        return super().validacion(particionado, dataset, clasificador, seed)
    
    def error(self, datos, pred):


        return super().error(datos, pred)




    def __init__(self, csv_file):
        self.datos = Datos(csv_file).datos
        #nominals = Datos(csv_file).nominalAtributos
        #encoder = OneHotEncoder(sparse=True, drop='first')
        #self.datos = encoder.fit_transform(self.datos[nominals])


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


classifier = ClasificadorNaiveBayes_heart("heart.csv")
classifier.calculate_accuracy()

