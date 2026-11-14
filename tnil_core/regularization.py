
"""Construccion de las ventanas espectrales para la TNIL.

Parametros de la clase SpectralWindow
    nombre: Nombre de la ventana seleccionada
    N: Numero total de muestras
    n: Malla de muestras 0:N-1
    samples: Muestras de la ventana seleccionada de dimensiones 1XN,
    este vector es creado por el metodo generate().
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatVector = NDArray[np.float64]

class SpectralWindow:

    def __init__(self, nombre: str, N: int, alpha: float | None = None):

        # Validacion del numero de muestras
        if isinstance(N, (bool, np.bool_)) or not isinstance(
            N, (int, np.integer)
        ):
            raise TypeError("N debe ser un numero entero.")

        if N <= 0:
            raise ValueError("N debe ser mayor que cero.")

        # Validacion del nombre de la ventana
        if not isinstance(nombre, str):
            raise TypeError("El nombre de la ventana debe ser texto.")

        nombre = nombre.strip().lower()

        if nombre not in ("hanning", "hamming", "lanczos", "barlett", "riesz", "blackman", "kaiser"):
            raise ValueError(
                f"Ventana '{nombre}' no reconocida."
            )

        if nombre == "kaiser":

            if alpha is None:
                raise ValueError(
                    "La ventana Kaiser requiere el parametro alpha."
                )

            if isinstance(alpha, (bool, np.bool_)):
                raise TypeError("alpha debe ser numerico.")

            try:
                alpha = float(alpha)
            except (TypeError, ValueError, OverflowError):
                raise TypeError("alpha debe ser numerico.")

            if not np.isfinite(alpha) or alpha < 0:
                raise ValueError(
                    "alpha debe ser finito y mayor o igual que cero."
                )

        elif alpha is not None:
            raise ValueError(
                "alpha solo se utiliza con la ventana de Kaiser."
            )

        self.alpha = alpha
        self.nombre = nombre
        self.N = N
        self.n = np.arange(N, dtype=np.int64)

        self.samples = None

    def generate(self) -> FloatVector:
        """Genera la ventana espectral seleccionada.
        """
        # Construccion de la ventana Hanning
        if self.nombre == "hanning":

            self.samples = (
                1.0 + np.cos(self.n * np.pi / self.N)
            ) / 2.0
        elif self.nombre == "hamming":

            self.samples = (
                0.54 + 0.46*np.cos(self.n * np.pi / self.N)
            )
        elif self.nombre == "lanczos":

            self.samples = (
                np.sin(self.n * np.pi / self.N) / (self.n * np.pi / self.N)
            )
            self.samples[0] = 1
        elif self.nombre == "barlett":

            self.samples = (
                1 - np.abs(self.n) / self.N
            )
        elif self.nombre == "riesz":

            self.samples = (
                1 - (np.abs(self.n) / self.N)**2
            )
        elif self.nombre == "blackman":

            self.samples = (
                0.42 + 0.5*np.cos(self.n * np.pi / self.N) + 0.08*np.cos(2 * self.n * np.pi / self.N)
            )

        elif self.nombre == "kaiser":

            self.samples = (
                np.i0(self.alpha * np.sqrt(1 - (self.n / self.N)**2)) / np.i0(self.alpha)
            )
        
        return self.samples