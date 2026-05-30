"""Minimal Streamlit web interface for the O(2)-Hopf calculator.

Run locally with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np
import streamlit as st

from o2sym import O2HopfNormalForm, application_1_nonlinearity, application_1_system
from o2sym.checks import hypothesis_checklist
from o2sym.classification import classify_region
from o2sym.formatting import (
    component_equation_latex,
    complex_to_str,
    fourier_symbol_latex,
    linear_operator_latex,
)
from o2sym.safe_eval import safe_eval_expr


def parse_params(text: str) -> Dict[str, float]:
    data = json.loads(text)
    return {str(k): float(v) for k, v in data.items()}


def parse_linear_coeffs(text: str):
    """Parse JSON linear coeffs.

    Format: keys are strings "i,k". Values are either numbers or simple
    arithmetic expressions in params, e.g. "c**2" or "-delta".
    """
    raw = json.loads(text)
    out = {}
    for key, value in raw.items():
        i_str, k_str = key.split(",")
        index = (int(i_str.strip()), int(k_str.strip()))
        if isinstance(value, (int, float)):
            out[index] = float(value)
        elif isinstance(value, str):
            expr = value
            out[index] = lambda p, expr=expr: safe_eval_expr(expr, p)
        else:
            raise ValueError(f"Unsupported coefficient value for {key}: {value!r}")
    return out


def parse_nonlinearity(text: str):
    """Parse nonlinearity JSON rows.

    Each row: {"alpha_u": [..], "alpha_v": [..], "a1": number, "a2": number}
    """
    rows = json.loads(text)
    out = {}
    for row in rows:
        alpha_u = tuple(int(x) for x in row["alpha_u"])
        alpha_v = tuple(int(x) for x in row["alpha_v"])
        out[(alpha_u, alpha_v)] = np.array([complex(row.get("a1", 0)), complex(row.get("a2", 0))], dtype=complex)
    return out


def status_symbol(status) -> str:
    if status == "N/A":
        return "N/A"
    return "✅" if bool(status) else "❌"


st.set_page_config(page_title="O(2)-Hopf Calculator", layout="wide")
st.title("O(2)-Hopf normal-form coefficient calculator")
st.caption(
    "Prototype companion tool. It implements the simple-spectrum/eigenbasis formulas. "
    "Checklist items marked sampled/N/A are diagnostics, not proofs."
)

preset = st.sidebar.selectbox("Input mode", ["Application 1 preset", "Custom JSON input"])
beta_sign = st.sidebar.selectbox(
    "Branch side for Table I--VI",
    ["+", "-"],
    help="Sign of Re beta_{m_c,1}. Used only for Regions III and VI stability entries.",
)

if preset == "Application 1 preset":
    eta = st.sidebar.number_input("eta", value=0.0, step=0.1)
    a = st.sidebar.number_input("a", value=0.0, min_value=0.0, step=0.1,
                                help="fourth-order coefficient b_{1,4} = -a; "
                                     "0 <= a < c for the rigorous regime")
    c = st.sidebar.number_input("c", value=1.0, min_value=0.0001, step=0.1)
    delta = st.sidebar.number_input("delta", value=1.0, step=0.1)
    m_c = st.sidebar.number_input("critical mode m_c", value=1, step=1)
    system = application_1_system()
    params = {"a": float(a), "delta": float(delta), "c": float(c)}
    nonlinearity = application_1_nonlinearity(float(eta))
else:
    st.sidebar.write("Use JSON inputs below.")
    m_c = st.sidebar.number_input("critical mode m_c", value=1, step=1)
    m_L = st.sidebar.number_input("maximum linear order index m_L", value=2, step=1)
    params_text = st.text_area("Parameters JSON", '{"a": 0.0, "delta": 1.0, "c": 1.0}', height=90)
    linear_text = st.text_area(
        "Linear coefficients JSON",
        '{\n  "1,0": 0.0,\n  "1,4": "-a",\n  "1,1": 1.0,\n  "2,1": "c**2",\n  "2,0": 0.0,\n  "2,2": "-delta",\n  "2,4": -1.0\n}',
        height=170,
    )
    nonlinear_text = st.text_area(
        "Nonlinearity JSON rows",
        '[\n  {"alpha_u": [1, 1], "alpha_v": [0], "a1": 0.0, "a2": 1.0}\n]',
        height=150,
    )
    params = parse_params(params_text)
    system = O2HopfNormalForm(parse_linear_coeffs(linear_text), int(m_L))
    nonlinearity = parse_nonlinearity(nonlinear_text)

try:
    st.subheader("Entered linear operator")
    st.latex(linear_operator_latex(system, params))
    with st.expander("Fourier symbol shown from the same coefficients"):
        st.latex(fourier_symbol_latex(system, params))

    st.subheader("Entered nonlinear operator")
    g1 = component_equation_latex(nonlinearity, 0)
    g2 = component_equation_latex(nonlinearity, 1)
    st.latex(r"G(u,v)=\begin{pmatrix}g_1\\ g_2\end{pmatrix}")
    st.latex(rf"g_1={g1}")
    st.latex(rf"g_2={g2}")

    zeta, xi = system.compute_normal_form(int(m_c), params, nonlinearity)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Normal-form coefficients")
        st.metric("Re zeta", f"{zeta.real:.12g}")
        st.metric("Re xi", f"{xi.real:.12g}")
        st.write("zeta =", complex_to_str(zeta))
        st.write("xi =", complex_to_str(xi))
        st.write("Re(zeta+xi) =", f"{(zeta.real + xi.real):.12g}")
        st.write("Re(zeta-xi) =", f"{(zeta.real - xi.real):.12g}")

    with col2:
        st.subheader("Region classification from the paper")
        region = classify_region(zeta, xi, beta_sign)
        st.metric("Region", region["region"])
        st.write(region["conditions"])
        st.write("TW:", region["tw"])
        st.write("SW:", region["sw"])
        st.write("Transition:", region["transition"])

    st.subheader("Checklist")
    checklist = hypothesis_checklist(system, int(m_c), params, nonlinearity)
    display_rows = [
        {"status": status_symbol(row["status"]), "check": row["check"], "details": row["details"]}
        for row in checklist
    ]
    st.table(display_rows)

    with st.expander("Linear diagnostics"):
        data = system.diagnostic_data(int(m_c), params)
        st.write("M_mc")
        st.write(data["M_mc"])
        st.write("M_2mc")
        st.write(data["M_2mc"])
        st.write("beta_mc", tuple(complex_to_str(z) for z in data["beta_mc"]))
        st.write("beta_2mc", tuple(complex_to_str(z) for z in data["beta_2mc"]))

except Exception as exc:
    st.error(str(exc))
    st.info("Check the simple-spectrum assumptions, denominators, and JSON input format.")

