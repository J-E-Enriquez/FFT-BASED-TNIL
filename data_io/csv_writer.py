
"""Exportacion de espectros y senales temporales a CSV TNIL."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_vector(values: ArrayLike, name: str) -> NDArray:

    array = np.asarray(values)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)

    elif array.ndim != 1:
        raise ValueError(
            f"'{name}' debe ser un vector 1D, fila o columna."
        )

    if array.size == 0:
        raise ValueError(f"'{name}' no puede estar vacio.")

    return array


def _validated_metadata(metadata, reserved):

    if metadata is None:
        return []

    conflicts = reserved.intersection(
        str(key) for key in metadata
    )

    if conflicts:
        raise ValueError(
            "No puedes sobrescribir metadatos reservados: "
            + ", ".join(sorted(conflicts))
        )

    items = []

    for key, value in metadata.items():

        key_text = str(key).strip()
        value_text = str(value).strip()

        if not key_text or any(
            char in key_text for char in "=\r\n"
        ):
            raise ValueError(
                "Las claves de metadatos deben ser texto simple."
            )

        if any(char in value_text for char in "\r\n"):
            raise ValueError(
                "Los valores de metadatos no pueden contener saltos de linea."
            )

        items.append((key_text, value_text))

    return items


def write_spectrum_csv(
    file_path: str | Path,
    spectrum: ArrayLike,
    T: float,
    c: float,
    *,
    omega: ArrayLike | None = None,
    window_applied: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Exporta un espectro en formato TNIL-SPECTRUM-V1."""

    values_raw = _as_vector(spectrum, "spectrum")

    values = np.asarray(
        values_raw,
        dtype=np.complex128
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "'spectrum' contiene NaN o valores infinitos."
        )

    if values.size < 2:
        raise ValueError(
            "El espectro debe contener al menos dos muestras."
        )

    # Validar parametros
    T_value = float(T)
    c_value = float(c)

    if not np.isfinite(T_value) or T_value <= 0:
        raise ValueError("T debe ser finito y mayor que cero.")

    if not np.isfinite(c_value):
        raise ValueError("c debe ser finito.")

    if not isinstance(window_applied, (bool, np.bool_)):
        raise TypeError(
            "window_applied debe ser True o False."
        )

    # Construir o validar vector de frecuencias
    N = values.size
    dw = 2.0 * np.pi / T_value

    if omega is None:

        omega_values = (
            np.arange(N, dtype=np.float64) * dw
        )

    else:

        omega_raw = _as_vector(omega, "omega")

        if np.iscomplexobj(omega_raw) and np.any(
            np.imag(omega_raw) != 0
        ):
            raise ValueError(
                "'omega' debe contener frecuencias reales."
            )

        omega_values = np.asarray(
            np.real(omega_raw),
            dtype=np.float64
        )

        if omega_values.size != N:
            raise ValueError(
                "'omega' y 'spectrum' deben tener la misma longitud."
            )

        expected = np.arange(N) * dw

        tolerance = max(1e-12, abs(dw) * 1e-8)

        if not np.allclose(
            omega_values,
            expected,
            rtol=1e-8,
            atol=tolerance
        ):
            raise ValueError(
                "'omega' no coincide con la malla esperada."
            )

    if not np.all(np.isfinite(omega_values)):
        raise ValueError(
            "'omega' contiene NaN o valores infinitos."
        )

    extra_metadata = _validated_metadata(
        metadata,
        {
            "T_s",
            "c_per_s",
            "frequency_order",
            "window_applied"
        }
    )

    path = Path(file_path)

    try:
        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            # Metadatos
            file.write("# TNIL-SPECTRUM-V1\n")
            file.write(f"# T_s={T_value:.17g}\n")
            file.write(f"# c_per_s={c_value:.17g}\n")
            file.write("# frequency_order=ifft_input\n")
            file.write(
                f"# window_applied={str(bool(window_applied)).lower()}\n"
            )

            for key, value in extra_metadata:
                file.write(f"# {key}={value}\n")

            # Encabezado CSV
            writer = csv.writer(file, lineterminator="\n")

            writer.writerow([
                "index",
                "omega_rad_s",
                "real",
                "imag"
            ])

            # Escribir muestras
            for index, (frequency, value) in enumerate(
                zip(omega_values, values)
            ):

                writer.writerow([
                    index,
                    f"{frequency:.17g}",
                    f"{value.real:.17g}",
                    f"{value.imag:.17g}"
                ])

    except OSError as exc:
        raise OSError(
            f"No fue posible escribir el archivo '{path}': {exc}"
        ) from exc


def write_time_signal_csv(
    file_path: str | Path,
    t: ArrayLike,
    signal: ArrayLike,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Exporta una senal temporal real en formato TNIL-TIME-V1."""

    time_raw = _as_vector(t, "t")
    signal_raw = _as_vector(signal, "signal")

    if np.iscomplexobj(time_raw) and np.any(
        np.imag(time_raw) != 0
    ):
        raise ValueError("'t' debe contener valores reales.")

    if np.iscomplexobj(signal_raw) and np.any(
        np.imag(signal_raw) != 0
    ):
        raise ValueError("'signal' debe ser real.")

    time = np.asarray(np.real(time_raw), dtype=np.float64)
    values = np.asarray(np.real(signal_raw), dtype=np.float64)

    if time.size != values.size:
        raise ValueError(
            "'t' y 'signal' deben tener la misma longitud."
        )

    if not np.all(np.isfinite(time)) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "'t' y 'signal' no pueden contener NaN o infinitos."
        )

    if time.size > 1 and np.any(np.diff(time) <= 0):
        raise ValueError(
            "Los instantes de tiempo deben ser estrictamente crecientes."
        )

    extra_metadata = _validated_metadata(
        metadata,
        {"N", "time_unit"}
    )

    path = Path(file_path)

    try:
        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            file.write("# TNIL-TIME-V1\n")
            file.write(f"# N={time.size}\n")
            file.write("# time_unit=s\n")

            for key, value in extra_metadata:
                file.write(f"# {key}={value}\n")

            writer = csv.writer(file, lineterminator="\n")

            writer.writerow(["time_s", "value"])

            for time_value, signal_value in zip(time, values):

                writer.writerow([
                    f"{time_value:.17g}",
                    f"{signal_value:.17g}"
                ])

    except OSError as exc:
        raise OSError(
            f"No fue posible escribir el archivo '{path}': {exc}"
        ) from exc
