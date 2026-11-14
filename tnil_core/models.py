
"""Estructuras de datos compartidas por la aplicacion TNIL."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]


def _as_vector(values: ArrayLike, name: str) -> NDArray:
    """Convierte un vector fila o columna en un vector 1D."""

    array = np.asarray(values)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)

    elif array.ndim != 1:
        raise ValueError(
            f"'{name}' debe ser un vector unidimensional."
        )

    if array.size == 0:
        raise ValueError(f"'{name}' no puede estar vacio.")

    return array


def _real_scalar(value, name: str) -> float:
    """Valida un escalar real y finito."""

    array = np.asarray(value)

    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"'{name}' debe ser un escalar real.")

    try:
        result = float(array)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"'{name}' debe ser un escalar real."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(f"'{name}' debe ser finito.")

    return result


@dataclass(frozen=True)
class SpectrumData:
    """Representa las muestras espectrales de una funcion de transferencia."""

    values: ArrayLike
    omega: ArrayLike

    T: float
    c: float

    window_applied: bool = False

    source: str = "unknown"
    expression: str | None = None

    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):

        # Validar parametros escalares
        T = _real_scalar(self.T, "T")
        c = _real_scalar(self.c, "c")

        if T <= 0:
            raise ValueError("T debe ser mayor que cero.")

        # Construir vector complejo
        try:
            values = np.asarray(
                _as_vector(self.values, "values"),
                dtype=np.complex128
            ).copy()

        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "No fue posible interpretar el espectro complejo."
            ) from exc

        # Construir vector de frecuencias
        omega_raw = _as_vector(self.omega, "omega")

        if np.iscomplexobj(omega_raw) and np.any(
            np.imag(omega_raw) != 0
        ):
            raise ValueError("omega debe contener valores reales.")

        try:
            omega = np.asarray(
                np.real(omega_raw),
                dtype=np.float64
            ).copy()

        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "No fue posible interpretar el vector de frecuencias."
            ) from exc

        # Validar dimensiones
        N = values.size

        if N < 2:
            raise ValueError(
                "El espectro debe contener al menos dos muestras."
            )

        if omega.size != N:
            raise ValueError(
                "values y omega deben tener la misma longitud."
            )

        # Validar valores numericos
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "El espectro contiene NaN o valores infinitos."
            )

        if not np.all(np.isfinite(omega)):
            raise ValueError(
                "omega contiene NaN o valores infinitos."
            )

        # Validar malla espectral regular de la TNIL
        dw = 2.0 * np.pi / T

        expected_omega = np.arange(N, dtype=np.float64) * dw

        tolerance = max(1e-12, abs(dw) * 1e-8)

        if not np.allclose(
            omega,
            expected_omega,
            rtol=1e-8,
            atol=tolerance
        ):
            raise ValueError(
                "Las frecuencias no corresponden a la malla "
                "omega[n] = n*2*pi/T."
            )

        if not isinstance(self.window_applied, (bool, np.bool_)):
            raise TypeError("window_applied debe ser booleano.")

        if not isinstance(self.source, str):
            raise TypeError("source debe ser una cadena de texto.")

        if self.expression is not None and not isinstance(
            self.expression, str
        ):
            raise TypeError("expression debe ser texto o None.")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata debe ser un diccionario o Mapping.")

        # Evitar modificaciones accidentales de los vectores
        values.setflags(write=False)
        omega.setflags(write=False)

        # Actualizar los atributos del objeto
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "c", c)
        object.__setattr__(
            self, "window_applied", bool(self.window_applied)
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata))
        )

    @property
    def N(self) -> int:
        return int(self.values.size)

    @property
    def dw(self) -> float:
        return 2.0 * np.pi / self.T

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def t(self) -> FloatVector:
        return np.arange(self.N, dtype=np.float64) * self.dt

    @property
    def s(self) -> ComplexVector:
        return np.asarray(
            self.c + 1j * self.omega,
            dtype=np.complex128
        )


@dataclass(frozen=True)
class TNILResult:
    """Contiene los resultados de una inversion numerica de Laplace."""

    input_data: SpectrumData
    signal: ArrayLike
    window_factor: ArrayLike
    window_description: str

    def __post_init__(self):

        if not isinstance(self.input_data, SpectrumData):
            raise TypeError(
                "input_data debe ser un objeto SpectrumData."
            )

        # Validar señal reconstruida
        signal = np.asarray(
            _as_vector(self.signal, "signal"),
            dtype=np.float64
        ).copy()

        # Validar ventana utilizada
        window_raw = _as_vector(
            self.window_factor,
            "window_factor"
        )

        if np.iscomplexobj(window_raw) and np.any(
            np.imag(window_raw) != 0
        ):
            raise ValueError("La ventana debe ser real.")

        window = np.asarray(
            np.real(window_raw),
            dtype=np.float64
        ).copy()

        N = self.input_data.N

        if signal.size != N or window.size != N:
            raise ValueError(
                "La señal y la ventana deben tener N muestras."
            )

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                "La señal contiene NaN o valores infinitos."
            )

        if not np.all(np.isfinite(window)):
            raise ValueError(
                "La ventana contiene NaN o valores infinitos."
            )

        if not isinstance(self.window_description, str):
            raise TypeError(
                "window_description debe ser una cadena de texto."
            )

        signal.setflags(write=False)
        window.setflags(write=False)

        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "window_factor", window)

    @property
    def t(self) -> FloatVector:
        return self.input_data.t

    @property
    def N(self) -> int:
        return self.input_data.N

    @property
    def T(self) -> float:
        return self.input_data.T

    @property
    def c(self) -> float:
        return self.input_data.c

    @property
    def dt(self) -> float:
        return self.input_data.dt

    @property
    def source(self) -> str:
        return self.input_data.source
