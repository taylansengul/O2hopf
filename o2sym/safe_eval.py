from __future__ import annotations

import ast
from typing import Mapping

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def safe_eval_expr(expr: str, params: Mapping[str, float]) -> float:
    """Evaluate a small arithmetic expression in numeric parameters.

    Permits numbers, parameter names, +, -, *, /, **, and parentheses.
    No function calls, attributes, subscriptions, or imports are allowed.
    """
    node = ast.parse(expr, mode="eval")
    value = _eval_node(node.body, params)
    return float(value)


def _eval_node(node: ast.AST, params: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported literal in expression: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise ValueError(f"Unknown parameter in expression: {node.id}")
        return float(params[node.id])
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator in expression: {op_type.__name__}")
        return _ALLOWED_BINOPS[op_type](_eval_node(node.left, params), _eval_node(node.right, params))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator in expression: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand, params))
    raise ValueError(f"Unsupported expression syntax: {ast.dump(node)}")
