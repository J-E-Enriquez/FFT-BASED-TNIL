"""Inversion numerica de Laplace mediante IFFT y muestreo regular.

La expresion implementada corresponde directamente a INLT.m:

    Cn = exp(c*t) * 2/dt
    WFF = FF * window
    TF = Cn * real(ifft(WFF) - WFF[0]/(2*N))

La implementacion trabaja en doble precision: float64/complex128.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector = NDArray[np.float64]


def _as_vector(values: ArrayLike, name: str) -> NDArray:
    """Convierte un vector 1D, fila o columna en un arreglo unidimensional."""
    array = np.asarray(values)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(
            f"'{name}' debe ser un vector 1D, fila o columna; "
            f"se recibio un arreglo con forma {array.shape}."
        )

    if array.size == 0:
        raise ValueError(f"'{name}' no puede estar vacio.")

    return array


def inlt(
    FF: ArrayLike,
    N: int,
    c: float,
    t: ArrayLike,
    dt: float,
    window: ArrayLike,
) -> FloatVector:
    """Calcula la TNIL usando IFFT, muestreo regular y una ventana espectral.

    Parameters
    ----------
    FF:
        Muestras complejas de F(c + j*omega), en el orden esperado por ifft.
    N:
        Numero de muestras. Debe coincidir con las longitudes de FF, t y window.
    c:
        Desplazamiento real de la recta de inversion en el plano s.
    t:
        Vector de instantes de tiempo.
    dt:
        Paso temporal positivo.
    window:
        Ventana espectral real de longitud N.

    Returns
    -------
    numpy.ndarray
        Senal reconstruida, real, de tipo float64 y longitud N.

    Notes
    -----
    No se realiza aqui el muestreo de F(s), la generacion de la ventana,
    lectura/escritura de archivos ni visualizacion. Esas tareas pertenecen
    a otros modulos.
    """
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise TypeError("N debe ser un numero entero.")
    if N <= 0:
        raise ValueError("N debe ser mayor que cero.")

    try:
        c_array = np.asarray(c)
        if c_array.ndim != 0 or np.iscomplexobj(c_array):
            raise ValueError
        c_value = float(c_array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("c debe ser un escalar real y finito.") from exc
    if not np.isfinite(c_value):
        raise ValueError("c debe ser un escalar real y finito.")

    try:
        dt_array = np.asarray(dt)
        if dt_array.ndim != 0 or np.iscomplexobj(dt_array):
            raise ValueError
        dt_value = float(dt_array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt debe ser un escalar real, finito y positivo.") from exc
    if not np.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("dt debe ser un escalar real, finito y positivo.")

    try:
        ff = _as_vector(FF, "FF").astype(np.complex128, copy=False)
        time_raw = _as_vector(t, "t")
        window_raw = _as_vector(window, "window")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"No fue posible interpretar los datos de entrada: {exc}") from exc

    if np.iscomplexobj(time_raw) and np.any(np.imag(time_raw) != 0):
        raise ValueError("t debe contener instantes de tiempo reales.")
    try:
        time = np.asarray(np.real(time_raw), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("t debe contener valores numericos reales.") from exc

    if np.iscomplexobj(window_raw) and np.any(np.imag(window_raw) != 0):
        raise ValueError("window debe ser real.")
    win = np.asarray(np.real(window_raw), dtype=np.float64)

    vectors = (("FF", ff), ("t", time), ("window", win))
    for name, vector in vectors:
        if vector.size != N:
            raise ValueError(
                f"La longitud de '{name}' es {vector.size}, pero N={N}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"'{name}' contiene valores NaN o infinitos.")

    # Traduccion directa de las operaciones de INLT.m.
    weighted_spectrum = ff * win
    inverse_fft = np.fft.ifft(weighted_spectrum)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correction = weighted_spectrum[0] / (2.0 * N)
        scale = np.exp(c_value * time) * (2.0 / dt_value)
        result = scale * np.real(inverse_fft - correction)

    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "La TNIL produjo valores no finitos. Revisa c, t, dt y la magnitud "
            "del espectro: el factor exp(c*t) puede amplificar el error numerico."
        )

    return np.asarray(result, dtype=np.float64)
