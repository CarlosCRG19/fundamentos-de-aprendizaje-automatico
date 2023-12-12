import numpy as np


class Individuo:
    def __init__(self, reglas: np.ndarray):
        self.reglas = reglas

    @staticmethod
    def crea_con_reglas_aleatorias(max_reglas: int, longitud_reglas: int):
        while True:
            reglas = np.random.randint(
                2, size=(np.random.randint(1, max_reglas), longitud_reglas)
            )
            # Verificar si alguna fila tiene solo 0s o solo 1s
            if not np.any(np.all(reglas == 0, axis=1)) and not np.any(
                np.all(reglas == 1, axis=1)
            ):
                break

        return Individuo(reglas)

    def clasifica(self, dato: np.ndarray) -> int:
        votos = [0, 0]

        for regla in self.reglas:
            regla_activada = True

            for dato_bit, regla_bit in zip(dato, regla[:-1]):
                if dato_bit == 1 and regla_bit != 1:
                    regla_activada = False
                    break

            if regla_activada:
                votos[regla[-1]] += 1

        if votos[0] > votos[1]:
            return 0
        elif votos[0] < votos[1]:
            return 1
        else:
            return -1

    def fitness(self, datos_codificados: np.ndarray) -> float:
        aciertos = 0

        for dato in datos_codificados:
            prediccion = self.clasifica(dato[:-1])

            if dato[-1] == prediccion:
                aciertos += 1

        return aciertos / datos_codificados.shape[0]

    # metodo prototipo
    def copia(self):
        return Individuo(np.copy(self.reglas))
