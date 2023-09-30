# coding: utf-8
from typing import List

import pandas as pd


class Datos:
    datos: int
    nominalAtributos: List[bool]

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero: str):
        self.datos = pd.read_csv(nombreFichero)

        self.nominalAtributos = []
        self.diccionarios = {}

        for columna in self.datos.columns:
            if self._es_nominal(columna):
                self.nominalAtributos.append(True)
                self.diccionarios[columna] = self._generar_mapeo(columna)
            else:
                self.nominalAtributos.append(False)
                self.diccionarios[columna] = {}

    def _es_nominal(self, columna: str) -> bool:
        return columna.lower() == "class" or self.datos[columna].dtype.name == "object"

    def _generar_mapeo(self, columna_nominal: str):
        valores = [str(valor) for valor in self.datos[columna_nominal].unique()]
        valores = sorted(valores)

        return {valor: indice for indice, valor in enumerate(valores)}

    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx: List[int]):
        return self.datos.iloc[idx]
