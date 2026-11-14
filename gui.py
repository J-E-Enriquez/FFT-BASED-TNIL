"""Interfaz gráfica de TNIL (PySide6 + Matplotlib).

La GUI no implementa fórmulas de inversión: consume exclusivamente engine.py
para mantener el cálculo separado de la presentación.

Ejecutar desde la raíz del proyecto: python gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtGui import QIcon
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QTabBar,
)

from data_io import write_spectrum_csv, write_time_signal_csv
from engine import run_from_csv, run_from_expression
from tnil_core.models import TNILResult
from tnil_core.visualization import (
    frequency_axis,
    positive_for_log,
    regularized_spectrum,
    unwrapped_phase,
)


class InversionWorker(QObject):
    """Ejecuta la inversión fuera del hilo de la ventana."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, source_type: str, parameters: dict):
        super().__init__()
        self.source_type = source_type
        self.parameters = parameters

    @Slot()
    def run(self) -> None:
        try:
            if self.source_type == "expression":
                result = run_from_expression(**self.parameters)
            elif self.source_type == "csv":
                result = run_from_csv(**self.parameters)
            else:
                raise ValueError("Fuente de entrada no reconocida.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit(result)


class TNILWindow(QMainWindow):
    """Ventana principal: entradas, opciones, resultados y exportación."""

    def __init__(self) -> None:
        super().__init__()
        icon_path = Path(__file__).resolve().parent / "assets" / "TNIL.ico"
        self.setWindowIcon(QIcon(str(icon_path)))
        self.setWindowTitle("TNIL Lab")
        self.resize(1180, 740)
        self.setMinimumSize(850, 550)

        self.result: TNILResult | None = None
        self._thread: QThread | None = None
        self._worker: InversionWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 10)
        root_layout.setSpacing(9)

        heading = QLabel("Laboratorio de inversión de Laplace: TNIL")
        heading.setStyleSheet("font-size: 19px; font-weight: 600;")
        root_layout.addWidget(heading)
        subheading = QLabel("Software académico desarrollado en colaboración: UNISON-Cinvestav")
        subheading.setStyleSheet("color: gray;")
        root_layout.addWidget(subheading)

        splitter = QSplitter()
        root_layout.addWidget(splitter, 1)

        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([385, 795])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        self.statusBar().showMessage("Laboratorio inicializado con éxito.")
        self._on_source_changed(0)

    def _build_sidebar(self) -> QWidget:
        container = QScrollArea()
        container.setWidgetResizable(True)
        container.setMinimumWidth(315)
        container.setMaximumWidth(530)
        container.setFrameShape(QScrollArea.Shape.NoFrame)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 10, 3)
        layout.setSpacing(11)

        source_group = QGroupBox("1. Formato del espectro")
        source_layout = QVBoxLayout(source_group)
        self.source_tabs = QTabWidget()

        expression_tab = QWidget()
        expression_layout = QVBoxLayout(expression_tab)
        expression_layout.addWidget(QLabel("Expresión matemática de F(s):"))
        self.expression_input = QLineEdit("1/sqrt(s**2 + 1)")
        self.expression_input.setPlaceholderText("Ej.: 1/(s+1)")
        self.expression_input.setToolTip("Usa s, +, -, *, /, **, sqrt, exp, sin, cos, etc. También acepta ^.")
        expression_layout.addWidget(self.expression_input)
        expression_tip = QLabel("Ejemplo: 1/sqrt(s**2 + 1)\nUsa * para multiplicar y ** (o ^) para potencias.")
        expression_tip.setWordWrap(True)
        expression_tip.setStyleSheet("color: gray;")
        expression_layout.addWidget(expression_tip)
        expression_layout.addStretch()
        self.source_tabs.addTab(expression_tab, "Expresión")

        csv_tab = QWidget()
        csv_layout = QVBoxLayout(csv_tab)
        csv_layout.addWidget(QLabel("Archivo de muestras espectrales:"))
        csv_path_row = QHBoxLayout()
        self.csv_input = QLineEdit()
        self.csv_input.setPlaceholderText("Selecciona un archivo .csv")
        self.csv_input.setToolTip("CSV de espectro con encabezado # TNIL-SPECTRUM-V1")
        browse_button = QPushButton("Examinar…")
        browse_button.clicked.connect(self._browse_csv)
        csv_path_row.addWidget(self.csv_input, 1)
        csv_path_row.addWidget(browse_button)
        csv_layout.addLayout(csv_path_row)
        csv_tip = QLabel(
            "Formato TNIL-SPECTRUM-V1: debe contener T, c, orden de frecuencias "
            "y si el espectro ya fue ventaneado."
        )
        csv_tip.setWordWrap(True)
        csv_tip.setStyleSheet("color: gray;")
        csv_layout.addWidget(csv_tip)
        csv_layout.addStretch()
        self.source_tabs.addTab(csv_tab, "Archivo CSV")

        source_layout.addWidget(self.source_tabs)
        layout.addWidget(source_group)

        self.sampling_group = QGroupBox("2. Parámetros de muestreo regular")
        sampling_form = QFormLayout(self.sampling_group)
        self.T_input = QLineEdit("30")
        self.T_input.setToolTip(
            "Tiempo de observación T en segundos. Admite notación científica, ej. 1e-6."
        )

        self.N_input = QSpinBox()
        self.N_input.setRange(2, 2_000_000)
        self.N_input.setValue(1024)
        self.N_input.setToolTip("Número de muestras. No es obligatorio que sea potencia de dos.")

        # QDoubleSpinBox redondearía 1e-8 a cero con sus decimales habituales.
        self.epsi_input = QLineEdit("1e-5")
        self.epsi_input.setToolTip("Parámetro 0 < ε₀ < 1. Se admite notación científica, ej. 1e-8.")

        sampling_form.addRow("Tiempo de observación T:", self.T_input)
        sampling_form.addRow("Muestras N:", self.N_input)
        sampling_form.addRow("Error relativo:", self.epsi_input)
        layout.addWidget(self.sampling_group)
        self.source_tabs.currentChanged.connect(self._on_source_changed)

        regularization_group = QGroupBox("3. Regularización espectral con ventaneo")
        regularization_layout = QVBoxLayout(regularization_group)
        self.window_combo = QComboBox()
        self.window_combo.addItem("Por defecto", "auto")
        self.window_combo.addItem("Rectangular", None)
        self.window_combo.addItem("Barlett", "barlett")
        self.window_combo.addItem("Lanczos", "lanczos")
        self.window_combo.addItem("Riesz", "riesz")
        self.window_combo.addItem("Hanning", "hanning")
        self.window_combo.addItem("Hamming", "hamming")
        self.window_combo.addItem("Blackman", "blackman")
        self.window_combo.addItem("Kaiser", "kaiser")
        # Etiqueta para el tipo de ventana
        window_label = QLabel("Tipo de ventana:")
        regularization_layout.addWidget(window_label)
        regularization_layout.addWidget(self.window_combo)

        # Campo para el parametro alpha de Kaiser
        self.alpha_label = QLabel("Parámetro α:")

        self.alpha_input = QDoubleSpinBox()
        self.alpha_input.setRange(0.0, 50.0)
        self.alpha_input.setDecimals(3)
        self.alpha_input.setSingleStep(0.5)
        self.alpha_input.setValue(5.0)
        self.alpha_input.setToolTip("Parámetro: 0 < α ≤ 50")

        # Organizar horizontalmente etiqueta y entrada
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(self.alpha_label)
        alpha_row.addWidget(self.alpha_input)

        regularization_layout.addLayout(alpha_row)
        self.window_combo.currentIndexChanged.connect(
            self._update_alpha_enabled
        )
        self._update_alpha_enabled()

        window_info = QLabel(
            "Parámetro α solo se activa para ventana Kaiser.\n"
            "Valores decimales admitidos en el rango 0 ≤ α ≤ 50.\n"
            "Si α = 0, no hay truncamiento suave.\n"
            "Si α = 50, se obtiene máximo suavizado."
        )
        window_info.setWordWrap(True)
        window_info.setStyleSheet("color: gray;")
        regularization_layout.addWidget(window_info)
        layout.addWidget(regularization_group)

        self.run_button = QPushButton("Calcular TNIL")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("font-weight: 600;")
        self.run_button.clicked.connect(self._start_inversion)
        layout.addWidget(self.run_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminado: el motor no reporta progreso parcial.
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()
        container.setWidget(panel)
        return container

    
    def _build_results_panel(self) -> QWidget:

        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.setContentsMargins(10, 3, 3, 3)
        layout.setSpacing(9)

        # ==========================================
        # 1. Pestañas para seleccionar la gráfica
        # ==========================================

        self.plot_tabs = QTabBar()
        self.plot_tabs.setExpanding(True)

        self.plot_tabs.addTab("Señal temporal reconstruida")
        self.plot_tabs.addTab("Espectro sin regularización")
        self.plot_tabs.addTab("Espectro con regularización")
        self.plot_tabs.addTab("Ventana espectral")

        layout.addWidget(self.plot_tabs)

        # ==========================================
        # 2. Controles de visualización
        # ==========================================

        plot_controls = QHBoxLayout()

        # Intervalo temporal
        self.plot_range_label = QLabel("Intervalo temporal:")

        self.plot_range = QComboBox()
        # self.plot_range.addItem("0–75 % de T", 0.75)
        # self.plot_range.addItem("0–90 % de T", 0.90)
        # self.plot_range.addItem("0–100 % de T", 1.0)
        self.plot_range.addItem("Intervalo regular (0–95 % de T)", 0.95)
        self.plot_range.addItem("Intervalo completo (0–100 % de T)", 1.0)
        self.plot_range.addItem("Intervalo reducido (0–75 % de T)", 0.75)
        self.plot_range.addItem("Mitad del intervalo (0–50 % de T)", 0.50)

        self.plot_range.currentIndexChanged.connect(
            self._draw_result
        )

        plot_controls.addWidget(self.plot_range_label)
        plot_controls.addWidget(self.plot_range, 1)

        # Unidades de frecuencia
        self.frequency_label = QLabel("Frecuencia:")

        self.frequency_unit = QComboBox()
        self.frequency_unit.addItem("rad/s", "rad/s")
        self.frequency_unit.addItem("Hz", "Hz")

        self.frequency_unit.currentIndexChanged.connect(
            self._draw_result
        )

        plot_controls.addWidget(self.frequency_label)
        plot_controls.addWidget(self.frequency_unit)

        # Escala del eje de frecuencia
        self.frequency_scale_label = QLabel("Eje horizontal:")

        self.frequency_scale = QComboBox()
        self.frequency_scale.addItem("Lineal", "linear")
        self.frequency_scale.addItem("Logarítmico", "log")

        self.frequency_scale.currentIndexChanged.connect(
            self._draw_result
        )

        plot_controls.addWidget(self.frequency_scale_label)
        plot_controls.addWidget(self.frequency_scale)

        # Escala de magnitud
        self.magnitude_label = QLabel("Escala de magnitud:")

        self.magnitude_scale = QComboBox()
        self.magnitude_scale.addItem("Lineal", "linear")
        self.magnitude_scale.addItem("Logarítmica", "log")

        self.magnitude_scale.currentIndexChanged.connect(
            self._draw_result
        )

        plot_controls.addWidget(self.magnitude_label)
        plot_controls.addWidget(self.magnitude_scale)

        layout.addLayout(plot_controls)

        # ==========================================
        # 3. Figura Matplotlib
        # ==========================================

        self.figure = Figure(
            figsize=(8, 5),
            constrained_layout=True
        )

        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(
            self.canvas, panel
        )

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # ==========================================
        # 4. Nota informativa
        # ==========================================

        self.plot_note = QLabel(
            "Inversión numérica basada en el algoritmo FFT."
        )

        self.plot_note.setWordWrap(True)
        self.plot_note.setStyleSheet("color: gray;")

        layout.addWidget(self.plot_note)

        # ==========================================
        # 5. Resumen numérico
        # ==========================================

        self.summary = QLabel(
            "Todavía no hay resultados. "
            "Ejecuta una inversión para comenzar."
        )

        self.summary.setWordWrap(True)

        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        layout.addWidget(self.summary)

        # ==========================================
        # 6. Botones de exportación
        # ==========================================

        export_row = QHBoxLayout()

        self.export_time_button = QPushButton(
            "Exportar señal CSV…"
        )

        self.export_time_button.setEnabled(False)
        self.export_time_button.clicked.connect(
            self._export_time
        )

        export_row.addWidget(self.export_time_button)

        self.export_spectrum_button = QPushButton(
            "Exportar espectro CSV…"
        )

        self.export_spectrum_button.setEnabled(False)
        self.export_spectrum_button.clicked.connect(
            self._export_spectrum
        )

        export_row.addWidget(self.export_spectrum_button)

        layout.addLayout(export_row)

        # Conectar las pestañas después de construir
        # todos los controles.
        self.plot_tabs.currentChanged.connect(
            self._on_plot_view_changed
        )

        self._on_plot_view_changed(0)

        return panel


    # def _build_results_panel(self) -> QWidget:
    #     panel = QWidget()
    #     layout = QVBoxLayout(panel)
    #     layout.setContentsMargins(10, 3, 3, 3)
    #     layout.setSpacing(9)

    #     plot_controls = QHBoxLayout()
    #     plot_controls.addWidget(QLabel("Visualizar:"))
    #     self.plot_range = QComboBox()
    #     self.plot_range.addItem("Intervalo regular (0–95 % de T)", 0.95)
    #     self.plot_range.addItem("Intervalo completo (0–100 % de T)", 1.0)
    #     self.plot_range.addItem("Intervalo reducido (0–75 % de T)", 0.75)
    #     self.plot_range.addItem("Mitad del intervalo (0–50 % de T)", 0.50)
    #     self.plot_range.currentIndexChanged.connect(self._draw_result)
    #     plot_controls.addWidget(self.plot_range, 1)
    #     layout.addLayout(plot_controls)

    #     self.figure = Figure(figsize=(8, 5), constrained_layout=True)
    #     self.canvas = FigureCanvas(self.figure)
    #     self.axes = self.figure.add_subplot(111)
    #     self.toolbar = NavigationToolbar(self.canvas, panel)
    #     layout.addWidget(self.toolbar)
    #     layout.addWidget(self.canvas, 1)
    #     self._draw_placeholder()

    #     self.summary = QLabel("Todavía no hay resultados. Ejecuta una inversión para comenzar.")
    #     self.summary.setWordWrap(True)
    #     self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    #     layout.addWidget(self.summary)

    #     export_row = QHBoxLayout()
    #     self.export_time_button = QPushButton("Exportar señal en formato CSV…")
    #     self.export_time_button.setEnabled(False)
    #     self.export_time_button.clicked.connect(self._export_time)
    #     export_row.addWidget(self.export_time_button)
    #     self.export_spectrum_button = QPushButton("Exportar espectro en formato CSV…")
    #     self.export_spectrum_button.setEnabled(False)
    #     self.export_spectrum_button.clicked.connect(self._export_spectrum)
    #     export_row.addWidget(self.export_spectrum_button)
    #     layout.addLayout(export_row)
    #     return panel

    def _draw_placeholder(self) -> None:

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        # ax.set_title("Señal reconstruida")
        # ax.set_xlabel("Tiempo [s]")
        # ax.set_ylabel("Amplitud")
        ax.grid(True, alpha=0.3)

        ax.text(
            0.5,
            0.5,
            "Los resultados aparecerán aquí",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray"
        )

        self.canvas.draw_idle()

    # def _draw_placeholder(self) -> None:
    #     self.axes.clear()
    #     self.axes.set_title("Señal reconstruida")
    #     self.axes.set_xlabel("Tiempo [s]")
    #     self.axes.set_ylabel("Amplitud")
    #     self.axes.grid(True, alpha=0.3)
    #     self.axes.text(
    #         0.5, 0.5, "La reconstrucción temporal aparecerá aquí",
    #         transform=self.axes.transAxes, ha="center", va="center", color="gray"
    #     )
    #     self.canvas.draw_idle()

    def _on_plot_view_changed(self, *_args) -> None:

        # La pestaña 0 corresponde a la señal temporal.
        temporal = self.plot_tabs.currentIndex() == 0

        # Controles temporales
        self.plot_range_label.setVisible(temporal)
        self.plot_range.setVisible(temporal)

        # Controles espectrales
        self.frequency_label.setVisible(not temporal)
        self.frequency_unit.setVisible(not temporal)

        self.frequency_scale_label.setVisible(not temporal)
        self.frequency_scale.setVisible(not temporal)

        self.magnitude_label.setVisible(not temporal)
        self.magnitude_scale.setVisible(not temporal)

        # Actualizar la gráfica
        self._draw_result()

    def _on_source_changed(self, index: int) -> None:
        self.sampling_group.setEnabled(index == 0)
        self.sampling_group.setToolTip(
            "" if index == 0 else "En el modo CSV, T, c y N se leen del propio archivo."
        )

    def _update_alpha_enabled(self, *_args):

        is_kaiser = (
            self.window_combo.currentData() == "kaiser"
        )

        self.alpha_input.setEnabled(is_kaiser)
        self.alpha_label.setEnabled(is_kaiser)

    def _browse_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Importar espectro", "", "Espectros CSV (*.csv);;Todos los archivos (*)"
        )
        if filename:
            self.csv_input.setText(filename)

    def _read_parameters(self) -> tuple[str, dict]:
        window_name = self.window_combo.currentData()
        alpha = (
            self.alpha_input.value()
            if window_name == "kaiser"
            else None
        )
        if self.source_tabs.currentIndex() == 0:
            expression = self.expression_input.text().strip()
            if not expression:
                raise ValueError("Escribe una función de transferencia F(s).")
            try:
                T = float(self.T_input.text().strip().replace(",", "."))
            except ValueError as exc:
                raise ValueError("T debe ser un número positivo, por ejemplo 30 o 1e-6.") from exc
            if not np.isfinite(T) or T <= 0:
                raise ValueError("T debe ser finito y mayor que cero.")
            try:
                epsi = float(self.epsi_input.text().strip().replace(",", "."))
            except ValueError as exc:
                raise ValueError("ε₀ debe ser un número, por ejemplo 1e-8.") from exc
            if not np.isfinite(epsi) or not 0.0 < epsi < 1.0:
                raise ValueError("ε₀ debe cumplir 0 < ε₀ < 1.")
            return "expression", dict(
                expression=expression,
                T=T,
                N=self.N_input.value(),
                epsi=epsi,
                window_name=window_name,
                alpha = alpha,
            )

        file_path = Path(self.csv_input.text().strip())
        if not self.csv_input.text().strip() or not file_path.is_file():
            raise ValueError("Selecciona un archivo CSV de espectro que exista.")
        return "csv", dict(file_path=str(file_path), window_name=window_name, alpha = alpha)

    def _start_inversion(self) -> None:
        if self._thread is not None:
            return
        try:
            source_type, parameters = self._read_parameters()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Revisa las entradas", str(exc))
            return

        self.run_button.setEnabled(False)
        self.export_time_button.setEnabled(False)
        self.export_spectrum_button.setEnabled(False)
        self.progress.setVisible(True)
        self.statusBar().showMessage("Calculando la TNIL…")

        self._thread = QThread(self)
        self._worker = InversionWorker(source_type, parameters)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.succeeded.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    @Slot(object)
    def _on_success(self, result: TNILResult) -> None:
        self.result = result
        self._draw_result()
        self._update_summary()
        self.statusBar().showMessage("Inversión completada.")

    @Slot(str)
    def _on_failure(self, message: str) -> None:
        self.statusBar().showMessage("No fue posible completar la inversión.")
        QMessageBox.critical(self, "Error de inversión", message)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        self.export_time_button.setEnabled(self.result is not None)
        self.export_spectrum_button.setEnabled(self.result is not None)

    @staticmethod
    def _display_step(count: int) -> int:

        """Limita los puntos representados en la gráfica."""

        return max(
            1,
            int(np.ceil(count / 50_000))
        )

    
    def _draw_result(self, *_args) -> None:

        if self.result is None:
            self._draw_placeholder()
            return

        result = self.result

        # Pestaña seleccionada
        view = self.plot_tabs.currentIndex()

        # Limpiar la figura antes de crear nuevos ejes
        self.figure.clear()

        # ==========================================
        # VISTA 0: SEÑAL TEMPORAL
        # ==========================================

        if view == 0:

            fraction = float(self.plot_range.currentData())

            stop = max(
                2,
                min(
                    result.N,
                    int(np.floor(fraction * result.N)) + 1
                )
            )

            step = self._display_step(stop)

            ax = self.figure.add_subplot(111)

            ax.plot(
                result.t[:stop:step],
                result.signal[:stop:step],
                linewidth=1.3,
                # label="TNIL"
            )

            ax.set_xlabel("Tiempo [s]")
            ax.set_ylabel("Amplitud")
            ax.set_title("Señal reconstruida")

            ax.grid(True, alpha=0.3)
            # ax.legend(loc="best")

            self.plot_note.setText(
                "La selección del rango temporal solo afecta la gráfica. "
                "La señal completa permanece disponible "
                "para exportación."
            )

        # ==========================================
        # VISTAS 1, 2 Y 3: DOMINIO ESPECTRAL
        # ==========================================

        else:

            # Unidades de frecuencia
            unit = self.frequency_unit.currentData()

            frequency = frequency_axis(
                result.input_data.omega,
                unit
            )

            if unit == "rad/s":
                frequency_label = "Frecuencia angular ω [rad/s]"
            else:
                frequency_label = "Frecuencia f [Hz]"

            # Opciones de escala
            is_log = (
                self.magnitude_scale.currentData() == "log"
            )

            frequency_is_log = (
                self.frequency_scale.currentData() == "log"
            )

            # Submuestreo exclusivamente gráfico
            step = self._display_step(result.N)

            x = frequency[::step]

            # Frecuencia cero no puede dibujarse
            # en un eje logarítmico.
            if frequency_is_log:
                x = positive_for_log(x)

            # ======================================
            # VISTA 3: VENTANA ESPECTRAL
            # ======================================

            if view == 3:

                ax = self.figure.add_subplot(111)

                # Ventana realmente aplicada por el motor
                magnitudes = result.window_factor

                if is_log:
                    y = positive_for_log(magnitudes)
                else:
                    y = magnitudes

                ax.plot(
                    x,
                    y[::step],
                    linewidth=1.5
                )

                if is_log:

                    if np.any(magnitudes > 0):
                        ax.set_yscale("log")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No hay amplitudes positivas",
                            transform=ax.transAxes,
                            ha="center"
                        )

                if frequency_is_log:
                    ax.set_xscale("log")

                ax.set_xlabel(frequency_label)
                ax.set_ylabel("σ(ω)")

                ax.set_title(
                    f"Ventana utilizada: "
                    f"{result.window_description}"
                )

                ax.grid(
                    True,
                    alpha=0.3,
                    which="both"
                )

                if result.input_data.window_applied:

                    self.plot_note.setText(
                        "El archivo CSV ya contiene un espectro "
                        "ventaneado. El factor adicional aplicado "
                        "por el motor es σ(ω) = 1; la ventana previa "
                        "no puede reconstruirse a partir del CSV."
                    )

                else:

                    self.plot_note.setText(
                        "Magnitud de la ventana espectral σ(ω). "
                        "Observe que σ(ω) es real."
                    )

            # ======================================
            # VISTAS 1 Y 2: ESPECTROS COMPLEJOS
            # ======================================

            else:

                if view == 1:

                    # Espectro original
                    spectrum = result.input_data.values

                    title = "Espectro frecuencial de entrada"

                    if result.input_data.window_applied:
                        title += " · CSV previamente ventaneado"

                else:

                    # Espectro modificado por la ventana
                    spectrum = regularized_spectrum(result)

                    title = (
                        "Espectro frecuencial regularizado"
                    )

                # ==================================
                # Magnitud y fase
                # ==================================

                magnitude = np.abs(spectrum)

                # Desenvolver ANTES del submuestreo gráfico.
                phase = unwrapped_phase(spectrum)

                if is_log:
                    y = positive_for_log(magnitude)
                else:
                    y = magnitude

                # Dos ejes verticales con X compartido:
                # arriba magnitud, abajo fase.
                ax_mag, ax_phase = self.figure.subplots(
                    2,
                    1,
                    sharex=True
                )

                # ==================================
                # Gráfica superior: magnitud
                # ==================================

                ax_mag.plot(
                    x,
                    y[::step],
                    linewidth=1.25
                )

                if is_log:

                    if np.any(magnitude > 0):
                        ax_mag.set_yscale("log")
                    else:
                        ax_mag.text(
                            0.5,
                            0.5,
                            "Magnitud nula en todas "
                            "las frecuencias",
                            transform=ax_mag.transAxes,
                            ha="center"
                        )

                ax_mag.set_ylabel("Magnitud")
                ax_mag.set_title(title)

                ax_mag.grid(
                    True,
                    alpha=0.3,
                    which="both"
                )

                # ==================================
                # Gráfica inferior: fase
                # ==================================

                ax_phase.plot(
                    x,
                    phase[::step],
                    linewidth=1.25
                )

                if frequency_is_log:
                    ax_phase.set_xscale("log")

                ax_phase.set_xlabel(frequency_label)

                ax_phase.set_ylabel(
                    "Fase [rad]"
                )

                ax_phase.grid(
                    True,
                    alpha=0.3
                )

                self.plot_note.setText(
                    "La fase siempre se expresa en radianes y con escala lineal en el eje vertical."
                )

                if result.input_data.window_applied:

                    self.plot_note.setText(
                        self.plot_note.text() +
                        " El CSV ya estaba ventaneado, "
                        "por lo que no se aplicó "
                        "una segunda ventana."
                    )

        # ==========================================
        # NOTAS Y ACTUALIZACIÓN DEL LIENZO
        # ==========================================

        if (
            view != 0
            and self.frequency_scale.currentData() == "log"
        ):

            self.plot_note.setText(
                self.plot_note.text() +
                " La frecuencia cero se omite "
                "del eje horizontal logarítmico."
            )

        # Actualizar herramientas y figura
        self.toolbar.update()
        self.canvas.draw_idle()


    # def _draw_result(self, *_args) -> None:
    #     if self.result is None:
    #         return
    #     result = self.result
    #     fraction = float(self.plot_range.currentData())
    #     # Por convenio, t contiene 0, dt, ..., T-dt.
    #     stop = max(2, min(result.N, int(np.floor(fraction * result.N)) + 1))
    #     t = result.t[:stop]
    #     signal = result.signal[:stop]

    #     # Se limita el número de puntos que se envían a Matplotlib; no se
    #     # reducen las muestras del resultado ni del CSV que se exporta.
    #     plot_step = max(1, int(np.ceil(len(t) / 50_000)))
    #     self.axes.clear()
    #     self.axes.plot(t[::plot_step], signal[::plot_step], linewidth=1.4)
    #     self.axes.set_xlabel("Tiempo [s]")
    #     self.axes.set_ylabel("Amplitud")
    #     self.axes.set_title("Señal reconstruida")
    #     self.axes.grid(True, alpha=0.3)
    #     self.canvas.draw_idle()

    def _update_summary(self) -> None:
        if self.result is None:
            return
        result = self.result
        origin = "Expresión matemática" if result.source == "expression" else "Espectro CSV"
        self.summary.setText(
            f"Origen: {origin}    |    N = {result.N:,}    |    "
            f"T = {result.T:.8g} s    |    Δt = {result.dt:.6g} s    |    "
            f"c = {result.c:.6g}    |    Ventana: {result.window_description}"
        )

    def _export_time(self) -> None:
        if self.result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar señal temporal", "resultado_tnil.csv", "CSV (*.csv)"
        )
        if not filename:
            return
        result = self.result
        try:
            write_time_signal_csv(
                filename,
                result.t,
                result.signal,
                metadata={
                    "method": "IFFT",
                    "source": result.source,
                    "window": result.window_description,
                    "T_s": result.T,
                    "c_per_s": result.c,
                },
            )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))
        else:
            self.statusBar().showMessage(f"Señal temporal guardada: {filename}")

    def _export_spectrum(self) -> None:
        if self.result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar espectro de entrada", "espectro_tnil.csv", "CSV (*.csv)"
        )
        if not filename:
            return
        data = self.result.input_data
        # Exportamos el espectro de ENTRADA. La ventana que el motor aplicó
        # después no se añade a esas muestras ni al indicador window_applied.
        metadata = {"source": data.source}
        if data.expression is not None:
            metadata["expression"] = data.expression
        try:
            write_spectrum_csv(
                filename,
                spectrum=data.values,
                T=data.T,
                c=data.c,
                omega=data.omega,
                window_applied=data.window_applied,
                metadata=metadata,
            )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))
        else:
            self.statusBar().showMessage(f"Espectro de entrada guardado: {filename}")

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            event.ignore()
            QMessageBox.information(
                self, "Cálculo en curso", "La ventana no puede cerrarse mientras se ejecuta la inversión."
            )
            return
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TNILWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())