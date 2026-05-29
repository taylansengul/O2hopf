from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from .core import O2HopfNormalForm


def complex_to_str(z: complex) -> str:
    z = complex(z)
    return f"{z.real:.12g} {z.imag:+.12g}i"


def latex_complex(z: complex) -> str:
    z = complex(z)
    if abs(z.imag) < 1e-12:
        return f"{z.real:.12g}"
    if abs(z.real) < 1e-12:
        return f"{z.imag:.12g}i"
    return f"({z.real:.12g}{z.imag:+.12g}i)"


def _monomial_latex(alpha_u: Tuple[int, ...], alpha_v: Tuple[int, ...]) -> str:
    pieces: list[str] = []
    names_u = ["u", "u_x", "u_{xx}", "u_{xxx}", "u_{xxxx}", "u_{xxxxx}"]
    names_v = ["v", "v_x", "v_{xx}", "v_{xxx}", "v_{xxxx}", "v_{xxxxx}"]
    for alpha, names in [(alpha_u, names_u), (alpha_v, names_v)]:
        for r, power in enumerate(alpha):
            if int(power) == 0:
                continue
            name = names[r] if r < len(names) else rf"\partial_x^{{{r}}}"
            pieces.append(name if int(power) == 1 else f"{name}^{{{int(power)}}}")
    return " ".join(pieces) if pieces else "1"


def component_equation_latex(nonlinearity: Mapping[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray], component: int) -> str:
    terms: list[str] = []
    for (alpha_u, alpha_v), coeff_vec in nonlinearity.items():
        coeff = complex(coeff_vec[component])
        if abs(coeff) < 1e-12:
            continue
        mon = _monomial_latex(alpha_u, alpha_v)
        if abs(coeff - 1) < 1e-12:
            terms.append(mon)
        elif abs(coeff + 1) < 1e-12:
            terms.append(f"- {mon}")
        else:
            terms.append(f"{latex_complex(coeff)} {mon}")
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def eval_b(system: O2HopfNormalForm, key: Tuple[int, int], params: Mapping[str, float]) -> complex:
    value = system.b_coeffs.get(key, 0.0)
    return complex(value(params) if callable(value) else value)


def _latex_coeff_times(coeff: complex, term: str) -> str:
    coeff = complex(coeff)
    if abs(coeff) < 1e-12:
        return ""
    if abs(coeff - 1) < 1e-12:
        return term
    if abs(coeff + 1) < 1e-12:
        return f"- {term}"
    return f"{latex_complex(coeff)} {term}"


def _dx_term(k: int) -> str:
    if k == 0:
        return "I"
    if k == 1:
        return r"\partial_x"
    return rf"\partial_x^{{{k}}}"


def linear_component_latex(system: O2HopfNormalForm, params: Mapping[str, float], entry: str) -> str:
    terms: list[str] = []
    for k in range(system.m_L + 1):
        if entry == "11":
            order, key = 2 * k, (1, 2 * k)
        elif entry == "22":
            order, key = 2 * k, (2, 2 * k)
        elif entry == "12":
            order, key = 2 * k + 1, (1, 2 * k + 1)
        elif entry == "21":
            order, key = 2 * k + 1, (2, 2 * k + 1)
        else:
            raise ValueError(entry)
        piece = _latex_coeff_times(eval_b(system, key, params), _dx_term(order))
        if piece:
            terms.append(piece)
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def symbol_entry_latex(system: O2HopfNormalForm, params: Mapping[str, float], entry: str) -> str:
    terms: list[str] = []
    for k in range(system.m_L + 1):
        if entry == "11":
            order, key = 2 * k, (1, 2 * k)
        elif entry == "22":
            order, key = 2 * k, (2, 2 * k)
        elif entry == "12":
            order, key = 2 * k + 1, (1, 2 * k + 1)
        elif entry == "21":
            order, key = 2 * k + 1, (2, 2 * k + 1)
        else:
            raise ValueError(entry)
        coeff = eval_b(system, key, params)
        if abs(coeff) < 1e-12:
            continue
        term = "1" if order == 0 else rf"(im)^{{{order}}}"
        piece = _latex_coeff_times(coeff, term)
        if piece:
            terms.append(piece)
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def linear_operator_latex(system: O2HopfNormalForm, params: Mapping[str, float]) -> str:
    L11 = linear_component_latex(system, params, "11")
    L12 = linear_component_latex(system, params, "12")
    L21 = linear_component_latex(system, params, "21")
    L22 = linear_component_latex(system, params, "22")
    return rf"L=\begin{{pmatrix}} {L11} & {L12} \\ {L21} & {L22} \end{{pmatrix}}"


def fourier_symbol_latex(system: O2HopfNormalForm, params: Mapping[str, float]) -> str:
    M11 = symbol_entry_latex(system, params, "11")
    M12 = symbol_entry_latex(system, params, "12")
    M21 = symbol_entry_latex(system, params, "21")
    M22 = symbol_entry_latex(system, params, "22")
    return rf"M_m=\begin{{pmatrix}} {M11} & {M12} \\ {M21} & {M22} \end{{pmatrix}}"
