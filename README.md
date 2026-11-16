# TNIL: Transformada Numérica Inversa de Laplace

<p align="justify">
Aplicación desarrollada en Python para el cálculo de la Transformada Numérica Inversa de Laplace (TNIL), mediante el algoritmo de la Transformada Rápida de Fourier Inversa (IFFT). Consulte los siguientes artículos para una revisión detallada del sustento matemático detrás de la TNIL:
</p>

- [Numerical Laplace Inversion Methods for Electromagnetic Transient Simulations.](https://ieeexplore.ieee.org/document/7747864)
- [Frequency Domain Aspects of Electromagnetic Transient Analysis of Power Systems.](https://ieeexplore.ieee.org/document/6990290)
- [Implementation of the Numerical Laplace Transform: A Review.](https://ieeexplore.ieee.org/document/4511480)
- [Numerical Laplace Transformation and Inversion.](https://journals.sagepub.com/doi/abs/10.1177/002072097801500309)

## Funcionalidades

- Importación de datos espectrales desde archivos CSV.
- Evaluación de funciones en el dominio de Laplace.
- Regularización espectral mediante técnicas de ventaneo.
- Reconstrucción de señales en el dominio del tiempo.
- Visualización de datos mediante Matplotlib.
- Exportación de resultados en archivos CSV.

## Tecnologías

- Python
- NumPy
- Matplotlib
- PySide6
- PyInstaller



## Instalación y uso

Existen tres alternativas para ejecutar la aplicación **TNIL**, dependiendo de si el usuario prefiere ejecutar directamente el código fuente en Python, compilar la aplicación localmente o utilizar un ejecutable previamente compilado.

### Opción 1: Ejecutar desde el código fuente

Esta opción requiere tener instalado **Python** y las dependencias necesarias para el funcionamiento de la aplicación.

1. Descarga o clona este repositorio.
2. Instala las dependencias de Python necesarias (incluidas en `requirements.txt`).
3. Ejecuta el archivo `gui.py` utilizando Python.

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

Y ejecuta la aplicación desde la carpeta principal:

```bash
python gui.py
```

Esta alternativa es recomendada para desarrolladores y usuarios interesados en explorar o modificar el código fuente de la aplicación.

### Opción 2: Compilar el ejecutable localmente

Esta opción también requiere tener instalado **Python** y las dependencias necesarias.

1. Descarga o clona este repositorio.
2. Haz doble clic en el archivo `compilar_TNIL.bat`.
3. Una vez finalizado el proceso de compilación (puede demorar varios minutos), encontrarás el ejecutable generado en la carpeta `dist`.

El archivo `.bat` automatiza el proceso de compilación mediante **PyInstaller**, permitiendo generar una aplicación ejecutable **exclusivamente** para Windows.

### Opción 3: Descargar el ejecutable precompilado (Recomendado)

Esta es la alternativa más sencilla para los usuarios que únicamente desean utilizar la aplicación.

1. Dirígete a la sección **Releases** de este repositorio de GitHub.
2. Descarga la versión más reciente del ejecutable o paquete de distribución para Windows.
3. Descomprime los archivos descargados (aquí mismo encontrarás el manual de la aplicación).
4. Ejecuta el archivo `TNIL.exe`.

**No es necesario instalar Python ni sus dependencias**, ya que el ejecutable incluye el intérprete de Python y las bibliotecas necesarias para el funcionamiento de la aplicación.

Esta opción es recomendada para usuarios que no necesitan acceder al código fuente ni modificarlo.

## Estado del proyecto

**Versión:** 0.1.0 — Desarrollo inicial.

Aplicación en desarrollo para investigación y análisis de reconstrucción de señales a partir de su descripción espectral.


## Software desarrollado en colaboración con UNISON y Cinvestav

- Navarro Burruel, M. de la Universidad de Sonora.
- Enríquez Jamangape, J. E. del Cinvestav Unidad Guadalajara.
- Naredo Villagrán, J. L. del Cinvestav Unidad Guadalajara.
