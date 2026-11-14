
"""Coordinador del procedimiento de inversion numerica de Laplace."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tnil_core.sampling import create_sampling
from tnil_core.regularization import SpectralWindow
from tnil_core.inversion import inlt

from tnil_core.models import SpectrumData, TNILResult

from expressions.transfer_function import TransferFunction
from data_io.csv_reader import read_spectrum_csv


def _build_window(
    data: SpectrumData,
    window_name: str | None,
    alpha: float | None = None
):
    """Construye la ventana que se utilizara en la inversion."""

    # Si el espectro ya fue ventaneado, no aplicar otra ventana.
    if data.window_applied:

        if window_name not in (None, "auto"):
            raise ValueError(
                "El espectro de entrada ya tiene una ventana aplicada. "
                "No se permite aplicar una segunda ventana."
            )

        return (
            np.ones(data.N, dtype=np.float64),
            "Ventana previamente aplicada al espectro"
        )

    # Sin ventana: ventana rectangular
    if window_name is None:

        return (
            np.ones(data.N, dtype=np.float64),
            "Sin ventana (rectangular)"
        )

    if not isinstance(window_name, str):
        raise TypeError(
            "window_name debe ser texto o None."
        )

    name = window_name.strip().lower()

    # La opcion automatica utiliza Hanning.
    if name == "auto":
        name = "hanning"

    # Construccion de la ventana seleccionada.
    window = SpectralWindow(
        name, 
        data.N,
        alpha = alpha if name == "kaiser" else None
    )

    samples = window.generate()

    if name == "kaiser":
        description = f"Kaiser (alpha={window.alpha:g})"
    else:
        description = window.nombre.capitalize()
        
    return samples, description


def run_tnil(
    data: SpectrumData,
    window_name: str | None = "auto",
    alpha: float | None = None
) -> TNILResult:
    """Ejecuta la TNIL a partir de un objeto SpectrumData.

    Parameters
    ----------
    data:
        Espectro complejo y parametros de inversion.

    window_name:
        Nombre de la ventana espectral.
        'auto' selecciona Hanning.
        None indica que no se aplicara una ventana adicional.

    Returns
    -------
    TNILResult:
        Resultado de la inversion numerica.
    """

    if not isinstance(data, SpectrumData):
        raise TypeError(
            "La entrada debe ser un objeto SpectrumData."
        )

    # Generar o recuperar la ventana.
    window_samples, window_description = _build_window(
        data,
        window_name,
        alpha = alpha
    )

    # Ejecutar la inversion numerica de Laplace.
    signal = inlt(
        FF=data.values,
        N=data.N,
        c=data.c,
        t=data.t,
        dt=data.dt,
        window=window_samples
    )

    # Construir objeto de resultados.
    result = TNILResult(
        input_data=data,
        signal=signal,
        window_factor=window_samples,
        window_description=window_description
    )

    return result


def run_from_expression(
    expression: str,
    T: float,
    N: int,
    epsi: float,
    window_name: str | None = "auto",
    alpha: float | None = None
) -> TNILResult:
    """Evalua una funcion de transferencia y ejecuta la TNIL.

    Parameters
    ----------
    expression:
        Expresion matematica de F(s).

    T:
        Tiempo de observacion.

    N:
        Numero de muestras.

    epsi:
        Error relativo utilizado para calcular el desplazamiento c.

    window_name:
        Ventana espectral que se utilizara.

    Returns
    -------
    TNILResult:
        Resultado de la inversion numerica.
    """

    # 1. Construccion de la malla temporal y espectral.
    grid = create_sampling(T, N, epsi)

    # 2. Interpretacion de la funcion de transferencia.
    transfer_function = TransferFunction(expression)

    # 3. Evaluacion de F(s) en los puntos complejos.
    FF = transfer_function.evaluate(grid.s)

    # 4. Construccion del objeto de datos espectrales.
    spectrum_data = SpectrumData(
        values=FF,
        omega=grid.dw * grid.n,
        T=grid.T,
        c=grid.c,
        window_applied=False,
        source="expression",
        expression=expression,
        metadata={
            "epsi": grid.epsi,
            "input_type": "mathematical_expression"
        }
    )

    # 5. Ejecutar la inversion.
    return run_tnil(
        spectrum_data,
        window_name=window_name,
        alpha = alpha
    )


def run_from_csv(
    file_path: str | Path,
    window_name: str | None = "auto",
    alpha: float | None = None
) -> TNILResult:
    """Importa un espectro CSV y ejecuta la TNIL.

    Parameters
    ----------
    file_path:
        Ruta del archivo CSV.

    window_name:
        Ventana espectral que se utilizara.
        Si el archivo indica que ya se aplico una ventana,
        no se aplicara una segunda.

    Returns
    -------
    TNILResult:
        Resultado de la inversion numerica.
    """

    # 1. Importar y validar el espectro.
    spectrum_data = read_spectrum_csv(file_path)

    # 2. Ejecutar la inversion.
    return run_tnil(
        spectrum_data,
        window_name=window_name,
        alpha = alpha
    )
