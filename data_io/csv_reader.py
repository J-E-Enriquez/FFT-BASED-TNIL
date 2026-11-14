
"""Lectura y validacion de espectros complejos en formato CSV TNIL."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tnil_core.models import SpectrumData


FloatVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]

def _read_metadata_and_csv(path: Path):
    """Separa los metadatos del contenido CSV."""

    try:
        lines = path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

    except OSError as exc:
        raise OSError(
            f"No fue posible leer el archivo '{path}': {exc}"
        ) from exc

    except UnicodeError as exc:
        raise ValueError(
            "El CSV debe estar codificado en UTF-8."
        ) from exc

    metadata = {}
    csv_lines = []
    schema_found = False

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):

            content = stripped[1:].strip()

            if content == "TNIL-SPECTRUM-V1":
                schema_found = True

            elif "=" in content:
                key, value = content.split("=", 1)
                metadata[key.strip()] = value.strip()

        else:
            csv_lines.append(line)

    if not schema_found:
        raise ValueError(
            "Formato no reconocido. Se esperaba "
            "'# TNIL-SPECTRUM-V1'."
        )

    if not csv_lines:
        raise ValueError("El archivo no contiene una tabla CSV.")

    return metadata, "\n".join(csv_lines)


def _parse_finite_float(value: str, field_name: str) -> float:

    try:
        parsed = float(value)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"El metadato '{field_name}' debe ser numerico."
        ) from exc

    if not np.isfinite(parsed):
        raise ValueError(
            f"El metadato '{field_name}' debe ser finito."
        )

    return parsed


def read_spectrum_csv(file_path: str | Path) -> SpectrumData:
    """Lee un espectro CSV TNIL-SPECTRUM-V1."""

    path = Path(file_path)

    metadata, csv_text = _read_metadata_and_csv(path)

    # Metadatos obligatorios
    required_metadata = (
        "T_s",
        "c_per_s",
        "frequency_order",
        "window_applied",
    )

    missing = [
        key for key in required_metadata
        if key not in metadata
    ]

    if missing:
        raise ValueError(
            "Faltan metadatos obligatorios: "
            + ", ".join(missing)
        )

    # Recuperar parametros
    T = _parse_finite_float(metadata["T_s"], "T_s")
    c = _parse_finite_float(metadata["c_per_s"], "c_per_s")

    if T <= 0:
        raise ValueError("T_s debe ser mayor que cero.")

    if metadata["frequency_order"].lower() != "ifft_input":
        raise ValueError(
            "frequency_order debe ser 'ifft_input'. "
            "No se reordenan las muestras automaticamente."
        )

    window_text = metadata["window_applied"].lower()

    if window_text not in {"true", "false"}:
        raise ValueError(
            "window_applied debe ser 'true' o 'false'."
        )

    window_applied = window_text == "true"

    # Leer tabla CSV
    reader = csv.DictReader(io.StringIO(csv_text))

    if reader.fieldnames is None:
        raise ValueError("No se encontro el encabezado CSV.")

    reader.fieldnames = [
        name.strip().lower()
        for name in reader.fieldnames
    ]

    required_columns = {
        "index",
        "omega_rad_s",
        "real",
        "imag"
    }

    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(
            "El CSV contiene nombres de columna duplicados."
        )

    missing_columns = required_columns.difference(
        reader.fieldnames
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas: "
            + ", ".join(sorted(missing_columns))
        )

    indices = []
    frequencies = []
    real_values = []
    imag_values = []

    for row_number, row in enumerate(reader, start=2):

        if None in row:
            raise ValueError(
                f"La fila {row_number} contiene "
                "mas campos que el encabezado."
            )

        try:
            index = int(row["index"].strip())
            frequency = float(row["omega_rad_s"])
            real_part = float(row["real"])
            imag_part = float(row["imag"])

        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"La fila {row_number} contiene "
                "datos faltantes o no numericos."
            ) from exc

        if not all(np.isfinite(x) for x in (
            frequency, real_part, imag_part
        )):
            raise ValueError(
                f"La fila {row_number} contiene "
                "NaN o valores infinitos."
            )

        indices.append(index)
        frequencies.append(frequency)
        real_values.append(real_part)
        imag_values.append(imag_part)

    # Validar cantidad de muestras
    N = len(indices)

    if N < 2:
        raise ValueError(
            "El espectro debe contener al menos dos muestras."
        )

    # Validar orden de las muestras
    if indices != list(range(N)):
        raise ValueError(
            "Los indices deben ser 0, 1, ..., N-1 "
            "y conservar su orden original."
        )

    # Construir vectores
    omega = np.asarray(frequencies, dtype=np.float64)

    values = np.asarray(
        real_values,
        dtype=np.float64
    ) + 1j * np.asarray(
        imag_values,
        dtype=np.float64
    )

    values = np.asarray(values, dtype=np.complex128)

    # Validar malla espectral
    expected_dw = 2.0 * np.pi / T

    expected_omega = (
        np.arange(N, dtype=np.float64) * expected_dw
    )

    tolerance = max(1e-12, abs(expected_dw) * 1e-8)

    if not np.allclose(
        omega,
        expected_omega,
        rtol=1e-8,
        atol=tolerance
    ):
        raise ValueError(
            "Las frecuencias no coinciden con la malla "
            "omega[n] = n*2*pi/T_s. "
            "Verifica T_s, las unidades y el orden."
        )

    return SpectrumData(
        values=values,
        omega=omega,
        T=T,
        c=c,
        window_applied=window_applied,
        source = "csv",
        metadata=metadata
    )
