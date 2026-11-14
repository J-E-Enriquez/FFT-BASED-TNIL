
"""Construccion de la malla temporal y espectral para la TNIL.

Parametros de la clase SamplingGrid
    T: Tiempo de observacion
    N: Numero total de muestras
    epsi: Error relativo de aliasing
    n: Malla de muestras 0:N-1
    dw: Paso de frecuencia
    dt: Paso de tiempo
    c: Constante de amortiguamiento
    t: Malla temporal
    s: Malla frecuencial

La función create_sampling se encarga de
calcular y llenar estos parametros en
función de sus parametros de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Tipos de datos
FloatVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]


@dataclass(frozen=True)
class SamplingGrid:
    """Contiene los parametros y vectores de muestreo."""

    T: float
    N: int
    epsi: float

    n: NDArray[np.int64]

    dw: float
    dt: float
    c: float

    t: FloatVector
    s: ComplexVector


def create_sampling(
    T: float,
    N: int,
    epsi: float
) -> SamplingGrid:
    """Construye los vectores temporal y espectral de la TNIL.

    Parameters
    ----------
    T : float
        Tiempo de observacion.

    N : int
        Numero de muestras.

    epsi : float
        Error relativo utilizado para calcular c.

    Returns
    -------
    SamplingGrid
        Objeto que contiene los parametros y vectores calculados.
    """

    # Validacion del numero de muestras
    if isinstance(N, (bool, np.bool_)) or not isinstance(
        N, (int, np.integer)
    ):
        raise TypeError("N debe ser un numero entero.")

    if N <= 0:
        raise ValueError("N debe ser mayor que cero.")

    # Validacion de T
    try:
        T_value = float(T)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "T debe ser un escalar real, finito y positivo."
        ) from exc

    if not np.isfinite(T_value) or T_value <= 0:
        raise ValueError(
            "T debe ser un escalar real, finito y positivo."
        )

    # Validacion de epsi
    try:
        epsi_value = float(epsi)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "epsi debe ser un escalar real entre 0 y 1."
        ) from exc

    if not np.isfinite(epsi_value) or not 0.0 < epsi_value < 1.0:
        raise ValueError("epsi debe cumplir 0 < epsi < 1.")

    # Construccion del vector de indices
    n = np.arange(N, dtype=np.int64)

    # Parametros de muestreo
    dw = 2.0 * np.pi / T_value
    dt = T_value / N
    c = -np.log(epsi_value) / T_value

    # Vector temporal
    t = np.asarray(dt * n, dtype=np.float64)

    # Operador complejo de Laplace
    s = np.asarray(
        c + 1j * dw * n,
        dtype=np.complex128
    )

    # Verificacion de resultados
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(s))):
        raise FloatingPointError(
            "La construccion de la malla produjo valores no finitos."
        )

    # Retornar todos los parametros en un objeto
    return SamplingGrid(
        T=T_value,
        N=int(N),
        epsi=epsi_value,
        n=n,
        dw=float(dw),
        dt=float(dt),
        c=float(c),
        t=t,
        s=s
    )
