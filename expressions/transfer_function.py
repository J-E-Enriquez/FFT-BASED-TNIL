
"""Interpretacion y evaluacion numerica de funciones de transferencia."""

from __future__ import annotations

import ast

import numpy as np


class ExpressionError(ValueError):
    """Error relacionado con una expresion matematica invalida."""


# Operaciones matematicas permitidas
BINARY_OPERATORS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}

UNARY_OPERATORS = {
    ast.UAdd: lambda x: x,
    ast.USub: np.negative,
}

# Funciones matematicas disponibles
MATH_FUNCTIONS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
}

# Constantes matematicas
CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
    "j": 1j,
}


class TransferFunction:

    def __init__(self, expression: str):

        if not isinstance(expression, str):
            raise TypeError("La expresion debe ser una cadena de texto.")

        expression = expression.strip()

        if not expression:
            raise ExpressionError("La expresion no puede estar vacia.")

        # Compatibilidad con operadores elementales de MATLAB
        expression = expression.replace(".^", "**")
        expression = expression.replace(".*", "*")
        expression = expression.replace("./", "/")
        expression = expression.replace("^", "**")

        self.expression = expression

        try:
            self.tree = ast.parse(expression, mode="eval")

        except SyntaxError as exc:
            raise ExpressionError(
                "La sintaxis de la funcion de transferencia no es valida."
            ) from exc

        # Validar estructura de la expresion
        self._validate(self.tree)

    def _validate(self, node):

        """Comprueba que la expresion solo contenga operaciones permitidas."""

        if isinstance(node, ast.Expression):
            self._validate(node.body)

        elif isinstance(node, ast.Constant):

            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float, complex)
            ):
                raise ExpressionError(
                    "Solo se permiten constantes numericas."
                )

        elif isinstance(node, ast.Name):

            allowed_names = {"s"} | set(CONSTANTS)

            if node.id not in allowed_names:
                raise ExpressionError(
                    f"Variable o constante no permitida: '{node.id}'."
                )

        elif isinstance(node, ast.BinOp):

            if type(node.op) not in BINARY_OPERATORS:
                raise ExpressionError(
                    "Operacion matematica no permitida."
                )

            self._validate(node.left)
            self._validate(node.right)

        elif isinstance(node, ast.UnaryOp):

            if type(node.op) not in UNARY_OPERATORS:
                raise ExpressionError(
                    "Operacion unaria no permitida."
                )

            self._validate(node.operand)

        elif isinstance(node, ast.Call):

            if not isinstance(node.func, ast.Name):
                raise ExpressionError(
                    "Solo se permiten funciones matematicas autorizadas."
                )

            if node.func.id not in MATH_FUNCTIONS:
                raise ExpressionError(
                    f"Funcion no permitida: '{node.func.id}'."
                )

            if len(node.args) != 1 or node.keywords:
                raise ExpressionError(
                    "Las funciones deben recibir exactamente un argumento."
                )

            self._validate(node.args[0])

        else:
            raise ExpressionError(
                "La expresion contiene una construccion no permitida."
            )

    def _evaluate(self, node, s):

        """Evalua recursivamente el arbol matematico."""

        if isinstance(node, ast.Expression):
            return self._evaluate(node.body, s)

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):

            if node.id == "s":
                return s

            return CONSTANTS[node.id]

        elif isinstance(node, ast.BinOp):

            operation = BINARY_OPERATORS[type(node.op)]

            left = self._evaluate(node.left, s)
            right = self._evaluate(node.right, s)

            return operation(left, right)

        elif isinstance(node, ast.UnaryOp):

            operation = UNARY_OPERATORS[type(node.op)]

            value = self._evaluate(node.operand, s)

            return operation(value)

        elif isinstance(node, ast.Call):

            function = MATH_FUNCTIONS[node.func.id]

            argument = self._evaluate(node.args[0], s)

            return function(argument)

        raise ExpressionError("No fue posible evaluar la expresion.")

    def evaluate(self, s):

        """Evalua F(s) en uno o varios puntos complejos."""

        s = np.asarray(s, dtype=np.complex128)

        if not np.all(np.isfinite(s)):
            raise ValueError(
                "El vector s contiene valores no finitos."
            )

        try:

            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore"
            ):

                result = self._evaluate(self.tree, s)

                result = np.asarray(
                    result,
                    dtype=np.complex128
                )

                # Permitir expresiones constantes
                if result.ndim == 0:
                    result = np.full(
                        s.shape,
                        result.item(),
                        dtype=np.complex128
                    )

                # Comprobar dimensiones
                elif result.shape != s.shape:
                    raise ExpressionError(
                        "El resultado no tiene las dimensiones esperadas."
                    )

        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:

            raise ExpressionError(
                "Error numerico al evaluar la funcion."
            ) from exc

        # Comprobar resultados no finitos
        if not np.all(np.isfinite(result)):

            raise ExpressionError(
                "La funcion produjo valores no finitos. "
                "Verifica posibles singularidades en los puntos de muestreo."
            )

        return result
