# coding: utf-8
from typing import Dict, List

import pandas as pd


class Datos:
    datos: int
    nominalAtributos: List[bool]

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
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
        # es verdadero si los valores son un números enteros o reales
        return self.datos[columna].dtype.name in ["int64", "float64"]

    def _generar_mapeo(self, columna_nominal: str):
        # se extraen los valores únicos de la columna y se sortean lexicograficamente
        valores = [str(valor) for valor in self.datos[columna_nominal].unique()]
        valores = sorted(valores)

        return {valor: indice for indice, valor in enumerate(valores)}

    def _reemplazar_valores(self, columna_nominal: str) -> pd.Series:
        mapeo = self.diccionarios[columna_nominal]
        return self.datos[columna_nominal].map(lambda valor: mapeo[str(valor)])

    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx: List[int]):
        return self.datos.iloc[idx]
