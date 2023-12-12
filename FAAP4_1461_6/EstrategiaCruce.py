from abc import ABC, abstractmethod
from typing import List

import numpy as np
from Individuo import Individuo


class EstrategiaCruce(ABC):
    @abstractmethod
    def aplicar_cruce(
        self, individuos: List[Individuo], max_reglas: int
    ) -> List[Individuo]:
        pass


class CruceInterReglas(EstrategiaCruce):
    def aplicar_cruce(
        self, individuos: List[Individuo], max_reglas: int
    ) -> List[Individuo]:
        descendientes = []

        for _ in range(len(individuos) // 2):
            progenitor1, progenitor2 = np.random.choice(
                individuos, size=2, replace=False
            )
            punto_cruce1, punto_cruce2 = np.random.randint(
                min(len(progenitor1.reglas), len(progenitor2.reglas)), size=2
            )

            nueva_reglas1 = np.vstack(
                (progenitor1.reglas[:punto_cruce1], progenitor2.reglas[punto_cruce2:])
            )
            nueva_reglas2 = np.vstack(
                (progenitor1.reglas[punto_cruce1:], progenitor2.reglas[:punto_cruce2])
            )

            if nueva_reglas1.shape[0] > max_reglas:
                nueva_reglas1[:max_reglas]

            if nueva_reglas2.shape[0] > max_reglas:
                nueva_reglas1[:max_reglas]

            descendiente1 = Individuo(nueva_reglas1)
            descendiente2 = Individuo(nueva_reglas2)

            descendientes.extend([descendiente1, descendiente2])

        if len(individuos) % 2 != 0:
            descendientes.append(np.random.choice(individuos, size=1))

        return descendientes


class CruceIntraReglas(EstrategiaCruce):
    def aplicar_cruce(
        self, individuos: List[Individuo], max_reglas: int
    ) -> List[Individuo]:
        descendientes = []

        for _ in range(len(individuos) // 2):
            progenitor1, progenitor2 = np.random.choice(
                individuos, size=2, replace=False
            )

            # Seleccionar una regla común entre ambos progenitores
            regla_comun = np.random.randint(
                min(len(progenitor1.reglas), len(progenitor2.reglas))
            )

            # Seleccionar un punto de corte aleatorio
            punto_cruce = np.random.randint(len(progenitor1.reglas[regla_comun]))

            # Intercambiar el material genético en la regla común
            nueva_regla1 = np.copy(progenitor1.reglas)
            nueva_regla2 = np.copy(progenitor2.reglas)

            nueva_regla1[regla_comun, :punto_cruce] = progenitor2.reglas[
                regla_comun, :punto_cruce
            ]
            nueva_regla2[regla_comun, :punto_cruce] = progenitor1.reglas[
                regla_comun, :punto_cruce
            ]

            descendiente1 = Individuo(nueva_regla1)
            descendiente2 = Individuo(nueva_regla2)

            descendientes.extend([descendiente1, descendiente2])

        if len(individuos) % 2 != 0:
            descendientes.append(np.random.choice(individuos, size=1))

        return descendientes
