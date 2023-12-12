from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class CodificadorBinario:
    def __init__(self, datos: pd.DataFrame):
        self._n_bits, self._codificacion = self._init_codificacion(datos)

    def _init_codificacion(
        self, datos: pd.DataFrame
    ) -> Tuple[int, Dict[str, Dict[str, List[int]]]]:
        atributos = datos.columns[:-1]
        objetivo = datos.columns[-1]

        codificacion = {}
        n_bits = 0  # cantidad de bits necesarios para codificar una muestra

        for atributo in atributos:
            valores = sorted(datos[atributo].astype(str).unique())
            codificacion[atributo] = {}

            for i, valor in enumerate(valores):
                codigo = [0] * len(valores)
                codigo[i] = 1

                codificacion[atributo][valor] = codigo

            n_bits += len(valores)

        # Columna de clase, tiene una codificación con un solo bit
        codificacion[objetivo] = {"0": [0], "1": [1], "+": [1], "-": [0]}
        n_bits += 1

        return n_bits, codificacion

    def codificacion(self) -> Dict[str, Dict[str, List[int]]]:
        return self._codificacion

    def codifica_datos(self, datos: pd.DataFrame) -> np.ndarray:
        filas_codificadas = []

        for _, fila in datos.iterrows():
            fila_codificada = []

            for columna, valor in fila.items():
                fila_codificada.extend(self._codificacion[columna][str(valor)])

            filas_codificadas.append(fila_codificada)

        return np.array(filas_codificadas)

    def n_bits(self) -> int:
        return self._n_bits
