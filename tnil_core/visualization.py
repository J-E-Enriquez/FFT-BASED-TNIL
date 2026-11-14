
"""Utilidades para visualizar los resultados de la TNIL.

Estas funciones no modifican los datos originales ni los
resultados de la inversión numérica de Laplace.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tnil_core.models import TNILResult


def frequency_axis(
    omega: ArrayLike,
    unit: str = "rad/s"
) -> NDArray[np.float64]:
    """Convierte las frecuencias angulares a rad/s o Hz."""

    omega = np.asarray(omega, dtype=np.float64)

    if unit == "rad/s":
        return omega

    if unit == "Hz":
        return omega / (2.0 * np.pi)

    raise ValueError(
        "La unidad debe ser 'rad/s' o 'Hz'."
    )


def regularized_spectrum(
    result: TNILResult
) -> NDArray[np.complex128]:
    """Devuelve el espectro utilizado por la IFFT."""

    if not isinstance(result, TNILResult):
        raise TypeError("Se requiere un objeto TNILResult.")

    return np.asarray(
        result.input_data.values * result.window_factor,
        dtype=np.complex128
    )


def unwrapped_phase(
    values: ArrayLike
) -> NDArray[np.float64]:
    """Calcula la fase desenvuelta en radianes.

    Usa NaN en muestras donde el espectro es cero.
    Cada tramo no nulo se desenvuelve por separado.
    """

    spectrum = np.asarray(values, dtype=np.complex128)

    if spectrum.ndim != 1:
        raise ValueError("El espectro debe ser un vector 1D.")

    if not np.all(np.isfinite(spectrum)):
        raise ValueError(
            "El espectro debe contener valores finitos."
        )

    phase = np.angle(spectrum).astype(np.float64)

    valid = np.abs(spectrum) > 0.0

    # La fase no está definida cuando la magnitud es cero.
    phase[~valid] = np.nan

    if valid.all():
        return np.unwrap(phase)

    # Desenvolver cada tramo no nulo por separado.
    changes = np.flatnonzero(
        np.diff(np.r_[False, valid, False])
    )

    for start, stop in changes.reshape(-1, 2):
        phase[start:stop] = np.unwrap(phase[start:stop])

    return phase


def positive_for_log(
    values: ArrayLike
) -> NDArray[np.float64]:
    """Sustituye por NaN los valores no positivos.

    Esto permite representar magnitudes en escala
    logarítmica sin tomar logaritmos de cero.
    """

    array = np.asarray(values, dtype=np.float64)

    return np.where(array > 0, array, np.nan)
