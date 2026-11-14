# TNIL — Transformada Numérica Inversa de Laplace

Aplicación desarrollada en Python para el cálculo de la Transformada Numérica Inversa de Laplace (TNIL), mediante el algoritmo de la Transformada Rápida de Fourier Inversa (IFFT). Consulte los siguientes artículos para una revisión detallada del sustento matemático detrás de la TNIL: [Numerical laplace inversion methods for electromagnetic transient simulations](https://ieeexplore.ieee.org/document/7747864), [Frequency Domain Aspects of Electromagnetic Transient Analysis of Power Systems](https://ieeexplore.ieee.org/document/6990290) y [Implementation of the Numerical Laplace Transform: A Review](https://ieeexplore.ieee.org/document/4511480).

## Funcionalidades

- Importación de datos espectrales desde archivos CSV.
- Evaluación de funciones en el dominio de Laplace.
- Regularización espectral mediante funciones de ventana.
- Reconstrucción de señales en el dominio del tiempo.
- Visualización y exportación de resultados.

## Tecnologías

- Python
- NumPy
- Matplotlib
- PySide6
- PyInstaller

## Instalación

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación desde la carpeta principal:

```bash
python gui.py
```

## Estado del proyecto

**Versión:** 0.1.0 — Desarrollo inicial.

Aplicación en desarrollo para investigación y análisis de reconstrucción de señales a partir de su descripción espectral.


## Software desarrollado en colaboración con UNISON y Cinvestav

Navarro Burruel, M. de Universidad de Sonora
Enriquez Jamangape, J. E. de Cinvestav Unidad Guadalajara
Naredo Villagrán, J. L. de Cinvestav Unidad Guadalajara