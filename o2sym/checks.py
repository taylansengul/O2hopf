from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .conservative import ConservativeChecker
from .core import O2HopfNormalForm
from .formatting import complex_to_str, eval_b

TOL_DEFAULT = 1e-8

StatusRow = Dict[str, Any]


def nonlinear_o2_component_ok(nonlinearity, component: int) -> bool:
    """Reflection check for kappa(u,v)(x)=(u(-x),-v(-x)).

    u_r has parity r and v_r has parity r+1. Therefore g_1 must be even
    and g_2 must be odd.
    """
    required = 0 if component == 0 else 1
    for (alpha_u, alpha_v), coeff_vec in nonlinearity.items():
        if abs(coeff_vec[component]) < 1e-12:
            continue
        parity = 0
        for r, a in enumerate(alpha_u):
            parity = (parity + int(a) * r) % 2
        for r, a in enumerate(alpha_v):
            parity = (parity + int(a) * (r + 1)) % 2
        if parity != required:
            return False
    return True


def linear_o2_status() -> StatusRow:
    return {
        "check": "linear O(2) parity pattern",
        "status": "N/A",
        "details": "enforced by input format: diagonal even orders, off-diagonal odd orders",
    }


def condition_D(system: O2HopfNormalForm, params: Mapping[str, float]) -> Tuple[bool, str]:
    m_L = system.m_L
    d1 = eval_b(system, (1, 2 * m_L), params)
    d2 = eval_b(system, (2, 2 * m_L), params)
    off1 = eval_b(system, (1, 2 * m_L + 1), params)
    off2 = eval_b(system, (2, 2 * m_L + 1), params)
    ok = abs(d1) > TOL_DEFAULT and abs(d2) > TOL_DEFAULT and abs(off1) < TOL_DEFAULT and abs(off2) < TOL_DEFAULT
    return ok, f"b_{{1,{2*m_L}}}={complex_to_str(d1)}, b_{{2,{2*m_L}}}={complex_to_str(d2)}, b_{{1,{2*m_L+1}}}={complex_to_str(off1)}, b_{{2,{2*m_L+1}}}={complex_to_str(off2)}"


def trace_det_at(system: O2HopfNormalForm, m: int, params: Mapping[str, float]) -> Tuple[complex, complex]:
    M = system.M_matrix(m, params)
    return complex(np.trace(M)), complex(np.linalg.det(M))


def asymptotic_gap_condition(system: O2HopfNormalForm, params: Mapping[str, float]) -> Tuple[bool, str]:
    # Numerical reconstruction of tau(x) and Delta(x), x=m^2. This is a practical
    # diagnostic, not a Sturm/root-isolation proof.
    mL = system.m_L
    xs = np.array([float(j) for j in range(1, max(8, 2 * mL + 8))])
    tau_vals: list[float] = []
    det_vals: list[float] = []
    for x in xs:
        tau = sum((eval_b(system, (1, 2 * k), params) + eval_b(system, (2, 2 * k), params)) * ((-1) ** k) * (x ** k) for k in range(mL + 1))
        A = sum(eval_b(system, (1, 2 * k), params) * ((-1) ** k) * (x ** k) for k in range(mL + 1))
        D = sum(eval_b(system, (2, 2 * k), params) * ((-1) ** k) * (x ** k) for k in range(mL + 1))
        B = sum(eval_b(system, (1, 2 * k + 1), params) * ((-1) ** k) * (x ** k) for k in range(mL + 1))
        C = sum(eval_b(system, (2, 2 * k + 1), params) * ((-1) ** k) * (x ** k) for k in range(mL + 1))
        det = A * D + x * B * C
        tau_vals.append(float(complex(tau).real))
        det_vals.append(float(complex(det).real))

    def degree_and_lead(vals: list[float]) -> Tuple[int | None, float | None]:
        for d in range(0, 2 * mL + 2):
            coeffs = np.polyfit(xs, vals, d)
            if np.max(np.abs(np.polyval(coeffs, xs) - vals)) < 1e-6:
                return d, float(coeffs[0])
        return None, None

    d_tau, a_tau = degree_and_lead(tau_vals)
    d_det, a_det = degree_and_lead(det_vals)
    ok = bool(d_tau is not None and d_det is not None and a_tau is not None and a_det is not None and a_tau < 0 and d_det >= d_tau and a_det > 0)
    return ok, f"numerical polynomial test: deg(det)={d_det}, deg(tr)={d_tau}, lead(det)={a_det:.4g}, lead(tr)={a_tau:.4g}"


def sample_global_hopf_conditions(system: O2HopfNormalForm, m_c: int, params: Mapping[str, float], sample_max: int = 40) -> list[StatusRow]:
    rows: list[StatusRow] = []
    tau_c, _det_c = trace_det_at(system, m_c, params)
    rows.append({"check": "Hopf (i) tr M_mc = 0", "status": abs(tau_c) < 1e-7, "details": f"tr M_mc={complex_to_str(tau_c)}"})
    bad_det: list[int] = []
    bad_tau: list[int] = []
    min_det = float("inf")
    max_tau_noncritical = -float("inf")
    for m in range(1, sample_max + 1):
        tau, det = trace_det_at(system, m, params)
        min_det = min(min_det, det.real)
        if m != m_c:
            max_tau_noncritical = max(max_tau_noncritical, tau.real)
        if not (abs(det.imag) < 1e-7 and det.real > 0):
            bad_det.append(m)
        if m != m_c and not (abs(tau.imag) < 1e-7 and tau.real < 0):
            bad_tau.append(m)
    rows.append({"check": f"Hopf (ii) det M_m > 0, sampled m≤{sample_max}", "status": len(bad_det) == 0, "details": f"min det={min_det:.6g}; bad m={bad_det[:8]}"})
    rows.append({"check": "Hopf (iii) d_lambda tr M_mc != 0", "status": "N/A", "details": "requires lambda-dependent coefficients or an explicit derivative"})
    rows.append({"check": f"Hopf (iv) tr M_m < 0 for m≠m_c, sampled m≤{sample_max}", "status": len(bad_tau) == 0, "details": f"max noncritical tr={max_tau_noncritical:.6g}; bad m={bad_tau[:8]}"})
    ok_gap, gap_details = asymptotic_gap_condition(system, params)
    rows.append({"check": "Hopf (v) asymptotic det/|tr| liminf > 0", "status": ok_gap, "details": gap_details})
    return rows


def hypothesis_checklist(system: O2HopfNormalForm, m_c: int, params: Mapping[str, float], nonlinearity) -> list[StatusRow]:
    rows: list[StatusRow] = []
    try:
        data = system.diagnostic_data(m_c, params)
        M_mc = data["M_mc"]
        M_2mc = data["M_2mc"]
        beta_1, beta_2 = data["beta_mc"]
        beta_21, beta_22 = data["beta_2mc"]
        rows.extend(sample_global_hopf_conditions(system, m_c, params))
        rows.append(linear_o2_status())
        rows.append({"check": "nonlinear O(2) parity for g1", "status": nonlinear_o2_component_ok(nonlinearity, 0), "details": "g1 reflection-even"})
        rows.append({"check": "nonlinear O(2) parity for g2", "status": nonlinear_o2_component_ok(nonlinearity, 1), "details": "g2 reflection-odd"})
        for component in [0, 1]:
            checker = ConservativeChecker.from_nonlinearity_component(nonlinearity, component)
            rows.append({"check": f"conservative form for g{component+1}", "status": checker.is_conservative(), "details": f"E_u terms={len(checker.euler_u())}, E_v terms={len(checker.euler_v())}"})
        cond_d, cond_details = condition_D(system, params)
        rows.append({"check": "Condition D/high-order diagonal part", "status": cond_d, "details": cond_details})
        rows.append({"check": "critical determinant positive", "status": abs(np.linalg.det(M_mc).imag) < 1e-7 and np.linalg.det(M_mc).real > 0, "details": f"det M_mc={complex_to_str(np.linalg.det(M_mc))}"})
        rows.append({"check": "critical pair purely imaginary", "status": abs(beta_1.real) < 1e-7 and abs(beta_2.real) < 1e-7 and abs(beta_1.imag) > 1e-7, "details": f"beta={complex_to_str(beta_1)}, {complex_to_str(beta_2)}"})
        rows.append({"check": "2m_c block invertible", "status": abs(np.linalg.det(M_2mc)) > 1e-9, "details": f"det M_2mc={complex_to_str(np.linalg.det(M_2mc))}"})
        denom1 = 2 * beta_1 - beta_21
        denom2 = 2 * beta_1 - beta_22
        rows.append({"check": "quadratic CM denominators nonzero", "status": abs(denom1) > 1e-9 and abs(denom2) > 1e-9 and abs(beta_21) > 1e-9 and abs(beta_22) > 1e-9, "details": f"2beta-beta_2mc=({complex_to_str(denom1)}, {complex_to_str(denom2)})"})
    except Exception as exc:
        rows.append({"check": "diagnostic computation", "status": False, "details": str(exc)})
    return rows
