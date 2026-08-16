from __future__ import annotations

import numpy as np
import pytest

from o2sym import ConservativeChecker, O2HopfNormalForm, application_1_nonlinearity, application_1_system, classify_region, hopf_params
from o2sym.coverage import bilinear_nonlinearity
from o2sym.checks import nonlinear_o2_component_ok
from o2sym.conservative import Term
from o2sym.formatting import component_equation_latex
from o2sym.safe_eval import safe_eval_expr


def test_import_and_application_1_eta_zero():
    system = application_1_system()
    zeta, xi = system.compute_normal_form(1, {"delta": 1.0, "c": 1.0}, application_1_nonlinearity(0.0))
    assert zeta.real == pytest.approx(-1.0 / 24.0)
    assert xi.real == pytest.approx(0.0)
    assert xi.imag == pytest.approx(-0.5)


def test_application_1_eta_one():
    system = application_1_system()
    zeta, xi = system.compute_normal_form(1, {"delta": 1.0, "c": 1.0}, application_1_nonlinearity(1.0))
    assert zeta.real == pytest.approx(-1.0 / 24.0)
    assert zeta.imag == pytest.approx(0.5)
    assert xi.real == pytest.approx(0.0)
    assert xi.imag == pytest.approx(0.5)


def test_conservative_uvx_plus_uxv():
    checker = ConservativeChecker([
        Term((1,), (0, 1), 1.0),
        Term((0, 1), (1,), 1.0),
    ])
    assert checker.is_conservative()


def test_nonconservative_ux_squared():
    checker = ConservativeChecker([Term((0, 2), (), 1.0)])
    assert not checker.is_conservative()


def test_formatter_u_uxx():
    nonlinearity = {((1, 0, 1), (0,)): [1.0, 0.0]}
    assert component_equation_latex(nonlinearity, 0) == "u u_{xx}"


def test_safe_eval_blocks_calls():
    assert safe_eval_expr("1/(2*c) + c**2", {"c": 2.0}) == pytest.approx(4.25)
    with pytest.raises(ValueError):
        safe_eval_expr('__import__("os").system("echo bad")', {})


def test_region_classifier_boundary_and_region_v():
    assert classify_region(0, 1)["region"] == "Degenerate/boundary"
    assert classify_region(-2, -3)["region"] == "V"


def _cubic_flux(h11: float, h12: float, h21: float, h22: float):
    """Return G = d_x F_3 for the general O(2)-equivariant cubic flux."""
    return {
        ((1, 1), (1,)): np.array([2 * h11, 0.0]),
        ((2,), (0, 1)): np.array([h11, 0.0]),
        ((), (2, 1)): np.array([3 * h12, 0.0]),
        ((2, 1), ()): np.array([0.0, 3 * h21]),
        ((0, 1), (2,)): np.array([0.0, h22]),
        ((1,), (1, 1)): np.array([0.0, 2 * h22]),
    }


def _generic_mc1_system(A: float = 0.37):
    """Admissible second-order family with its only Hopf pair at m_c = 1."""
    p, q = 0.22, 0.31
    return O2HopfNormalForm(
        {
            (1, 0): lambda params: A + p + 0.5 * params.get("lambda", 0.0),
            (1, 2): p,
            (2, 0): lambda params: -A + q + 0.5 * params.get("lambda", 0.0),
            (2, 2): q,
            (1, 1): 1.4,
            (2, 1): 0.9,
        },
        m_L=1,
    )


def _purpose_built_mc2_system():
    """Admissible fourth-order family with tr M_m = -(m^2-4)^2 at lambda=0."""
    return O2HopfNormalForm(
        {
            (1, 0): lambda params: -8.9 + 0.5 * params.get("lambda", 0.0),
            (1, 2): -4.0,
            (1, 4): -0.5,
            (2, 0): lambda params: -7.1 + 0.5 * params.get("lambda", 0.0),
            (2, 2): -4.0,
            (2, 4): -0.5,
            (1, 1): 1.0,
            (2, 1): 0.5625,
        },
        m_L=2,
    )


def _purpose_built_mc1_mL2_system(A: float = 0.37):
    """Fourth-order family with tr M_m = -(m^2-1)^2 at lambda=0."""
    return O2HopfNormalForm(
        {
            (1, 0): lambda params: -0.5 + A + 0.5 * params.get("lambda", 0.0),
            (1, 2): -1.0,
            (1, 4): -0.5,
            (2, 0): lambda params: -0.5 - A + 0.5 * params.get("lambda", 0.0),
            (2, 2): -1.0,
            (2, 4): -0.5,
            (1, 1): 1.4,
            (2, 1): 0.9,
        },
        m_L=2,
    )


def _trim(alpha):
    alpha = tuple(alpha)
    while alpha and alpha[-1] == 0:
        alpha = alpha[:-1]
    return alpha


def _differentiate_scalar_polynomial(polynomial):
    """Apply one total x derivative, expanding every Leibniz term."""
    result = {}
    for (alpha_u, alpha_v), coefficient in polynomial.items():
        for component, alpha in enumerate((alpha_u, alpha_v)):
            for derivative_order, exponent in enumerate(alpha):
                if exponent == 0:
                    continue
                updated = list(alpha) + [0]
                updated[derivative_order] -= 1
                updated[derivative_order + 1] += 1
                key = [_trim(alpha_u), _trim(alpha_v)]
                key[component] = _trim(updated)
                key = tuple(key)
                result[key] = result.get(key, 0.0) + coefficient * exponent
    return {key: value for key, value in result.items() if abs(value) > 1e-14}


def _differentiate_scalar_polynomial_n(polynomial, order):
    result = dict(polynomial)
    for _ in range(order):
        result = _differentiate_scalar_polynomial(result)
    return result


def _apply_odd_operator(first, second, operator_terms):
    """Apply sum_r a_r d_x^r componentwise to a two-component polynomial."""
    nonlinearity = {}
    for component, polynomial in enumerate((first, second)):
        for order, multiplier in operator_terms.items():
            differentiated = _differentiate_scalar_polynomial_n(polynomial, order)
            for key, coefficient in differentiated.items():
                vector = nonlinearity.setdefault(key, np.zeros(2, dtype=complex))
                vector[component] += multiplier * coefficient
    return {key: value for key, value in nonlinearity.items() if np.linalg.norm(value) > 1e-14}


def _combine_nonlinearities(*nonlinearities):
    result = {}
    for nonlinearity in nonlinearities:
        for key, value in nonlinearity.items():
            result[key] = result.get(key, np.zeros(2, dtype=complex)) + value
    return {key: value for key, value in result.items() if np.linalg.norm(value) > 1e-14}


def _max_derivative_order(nonlinearity):
    return max(
        derivative_order
        for (alpha_u, alpha_v), vector in nonlinearity.items()
        if np.linalg.norm(vector) > 1e-14
        for alpha in (alpha_u, alpha_v)
        for derivative_order, exponent in enumerate(alpha)
        if exponent
    )


def _assert_every_generated_term_is_o2_equivariant(nonlinearity):
    """Check Assumption S(ii) separately on every expanded Leibniz term."""
    for key, vector in nonlinearity.items():
        for component in (0, 1):
            if abs(vector[component]) <= 1e-14:
                continue
            singleton = {key: np.eye(2, dtype=complex)[component]}
            assert nonlinear_o2_component_ok(singleton, component)


def _assert_admissible_generated_nonlinearity(nonlinearity, m_L=2):
    _assert_every_generated_term_is_o2_equivariant(nonlinearity)
    assert nonlinear_o2_component_ok(nonlinearity, 0)
    assert nonlinear_o2_component_ok(nonlinearity, 1)
    for component in (0, 1):
        assert ConservativeChecker.from_nonlinearity_component(nonlinearity, component).is_conservative()
    assert _max_derivative_order(nonlinearity) <= 2 * m_L - 1


def _variational_density_nonlinearity(name, operator_terms=None):
    """Return S K delta H for one of the four derivative-dependent densities."""
    operator_terms = operator_terms or {1: 1.0}
    if name == "u_ux2":  # H_3 = integral u u_x^2 / 2
        grad_u = {((0, 2), ()): -0.5, ((1, 0, 1), ()): -1.0}
        grad_v = {}
    elif name == "u_vx2":  # H_3 = integral u v_x^2 / 2
        grad_u = {((), (0, 2)): 0.5}
        grad_v = {((0, 1), (0, 1)): -1.0, ((1,), (0, 0, 1)): -1.0}
    elif name == "u2_ux2":  # H_4 = integral u^2 u_x^2 / 4
        grad_u = {((1, 2), ()): -0.5, ((2, 0, 1), ()): -0.5}
        grad_v = {}
    elif name == "v2_ux2":  # H_4 = integral v^2 u_x^2 / 4
        grad_u = {((0, 1), (1, 1)): -1.0, ((0, 0, 1), (2,)): -0.5}
        grad_v = {((0, 2), (1,)): 0.5}
    else:
        raise ValueError(f"unknown density {name}")
    return _apply_odd_operator(grad_v, grad_u, operator_terms)


def _derivative_free_cubic_flux(h11, h12, h21, h22):
    return (
        {((2,), (1,)): h11, ((), (3,)): h12},
        {((3,), ()): h21, ((1,), (2,)): h22},
    )


def _derivative_free_quadratic_flux(g11, g21, g22):
    return (
        {((1,), (1,)): g11},
        {((2,), ()): 0.5 * g21, ((), (2,)): 0.5 * g22},
    )


def test_purpose_built_mc2_system_is_a_genuine_hopf_family():
    system = _purpose_built_mc2_system()
    for m in range(1, 21):
        M = system.M_matrix(m, {})
        assert np.trace(M).real == pytest.approx(-(m**2 - 4) ** 2)
        assert np.linalg.det(M).real > 0
        if m != 2:
            assert np.trace(M).real < 0
    beta_1, beta_2 = system.eigenvalues(2, {})
    assert beta_1 == pytest.approx(1.2j)
    assert beta_2 == pytest.approx(-1.2j)
    assert np.trace(system.M_matrix(2, {"lambda": 1.0})).real == pytest.approx(1.0)


@pytest.mark.parametrize(
    "system,m_c,params",
    [
        (application_1_system(), 1, hopf_params(0.5, 1.0)),
        (_generic_mc1_system(), 1, {}),
        (_purpose_built_mc2_system(), 2, {}),
    ],
)
def test_general_cubic_real_part_identities(system, m_c, params):
    h11, h12, h21, h22 = 0.8, -0.4, 1.1, -0.25
    zeta, xi = system.compute_normal_form(m_c, params, _cubic_flux(h11, h12, h21, h22))
    q_1 = system.diagnostic_data(m_c, params)["q_mc"][0]
    chi = np.imag(q_1[0] * np.conj(q_1[1]))
    defect = h11 - h22
    assert zeta.real == pytest.approx(-m_c * chi * defect, abs=1e-12)
    assert xi.real == pytest.approx(2 * m_c * chi * defect, abs=1e-12)
    assert zeta.real == pytest.approx(-0.5 * xi.real, abs=1e-12)


def test_h12_h21_do_not_affect_real_parts():
    system = _generic_mc1_system()
    reference = system.compute_normal_form(1, {}, _cubic_flux(0.7, 0.0, 0.0, -0.2))
    changed = system.compute_normal_form(1, {}, _cubic_flux(0.7, 4.3, -2.8, -0.2))
    assert changed[0].real == pytest.approx(reference[0].real, abs=1e-12)
    assert changed[1].real == pytest.approx(reference[1].real, abs=1e-12)


def test_compatible_pure_cubic_flux_has_zero_real_parts():
    for system, m_c in ((_generic_mc1_system(), 1), (_purpose_built_mc2_system(), 2)):
        zeta, xi = system.compute_normal_form(m_c, {}, _cubic_flux(0.6, -1.3, 2.1, 0.6))
        assert zeta.real == pytest.approx(0.0, abs=1e-12)
        assert xi.real == pytest.approx(0.0, abs=1e-12)


def test_zero_chi_is_the_exception_to_compatibility_characterization():
    system = _generic_mc1_system(A=0.0)
    zeta, xi = system.compute_normal_form(1, {}, _cubic_flux(0.9, 1.7, -0.4, -0.3))
    q_1 = system.diagnostic_data(1, {})["q_mc"][0]
    assert np.imag(q_1[0] * np.conj(q_1[1])) == pytest.approx(0.0, abs=1e-12)
    assert zeta.real == pytest.approx(0.0, abs=1e-12)
    assert xi.real == pytest.approx(0.0, abs=1e-12)


def test_compatible_cubic_part_does_not_change_mixed_real_parts():
    system = _generic_mc1_system()
    quadratic = bilinear_nonlinearity(0.3, 1.0, 0.3)
    cubic = _cubic_flux(0.75, -1.2, 0.4, 0.75)
    zeta_q, xi_q = system.compute_normal_form(1, {}, quadratic)
    zeta_mixed, xi_mixed = system.compute_normal_form(1, {}, {**quadratic, **cubic})
    assert zeta_mixed.real == pytest.approx(zeta_q.real, abs=1e-12)
    assert xi_mixed.real == pytest.approx(xi_q.real, abs=1e-12)
    assert xi_mixed.real == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "system,m_c",
    [(_purpose_built_mc1_mL2_system(), 1), (_purpose_built_mc2_system(), 2)],
)
@pytest.mark.parametrize(
    "density,degree",
    [("u_ux2", 3), ("u_vx2", 3), ("u2_ux2", 4), ("v2_ux2", 4)],
)
def test_derivative_dependent_variational_hamiltonian_cancellation(system, m_c, density, degree):
    nonlinearity = _variational_density_nonlinearity(density)
    _assert_admissible_generated_nonlinearity(nonlinearity)
    assert _max_derivative_order(nonlinearity) == 3

    zeta, xi = system.compute_normal_form(m_c, {}, nonlinearity)
    assert xi.real == pytest.approx(0.0, abs=1e-12)
    if degree == 4:
        assert zeta.real == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "system,m_c",
    [(_purpose_built_mc1_mL2_system(), 1), (_purpose_built_mc2_system(), 2)],
)
@pytest.mark.parametrize(
    "cubic_density,quartic_density",
    [("u_ux2", "u2_ux2"), ("u_vx2", "v2_ux2")],
)
def test_derivative_dependent_mixtures_leave_zeta_real_unchanged(
    system, m_c, cubic_density, quartic_density
):
    quadratic = _variational_density_nonlinearity(cubic_density)
    cubic = _variational_density_nonlinearity(quartic_density)
    mixed = _combine_nonlinearities(quadratic, cubic)
    _assert_admissible_generated_nonlinearity(mixed)

    zeta_quadratic, xi_quadratic = system.compute_normal_form(m_c, {}, quadratic)
    zeta_mixed, xi_mixed = system.compute_normal_form(m_c, {}, mixed)
    assert zeta_mixed.real == pytest.approx(zeta_quadratic.real, abs=1e-12)
    assert xi_quadratic.real == pytest.approx(0.0, abs=1e-12)
    assert xi_mixed.real == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "system,m_c",
    [(_purpose_built_mc1_mL2_system(), 1), (_purpose_built_mc2_system(), 2)],
)
@pytest.mark.parametrize(
    "operator_terms,mu",
    [
        ({3: 1.0}, lambda m: -(m**3)),
        ({1: 1.0, 3: 0.17}, lambda m: m - 0.17 * m**3),
    ],
)
def test_odd_hamiltonian_multiplier_cubic_defect_and_ratio(system, m_c, operator_terms, mu):
    h11, h12, h21, h22 = 0.8, -0.4, 1.1, -0.25
    flux = _derivative_free_cubic_flux(h11, h12, h21, h22)
    nonlinearity = _apply_odd_operator(*flux, operator_terms)
    _assert_admissible_generated_nonlinearity(nonlinearity)

    zeta, xi = system.compute_normal_form(m_c, {}, nonlinearity)
    q_1 = system.diagnostic_data(m_c, {})["q_mc"][0]
    chi = np.imag(q_1[0] * np.conj(q_1[1]))
    defect = h11 - h22
    assert zeta.real == pytest.approx(-mu(m_c) * chi * defect, abs=1e-12)
    assert xi.real == pytest.approx(2 * mu(m_c) * chi * defect, abs=1e-12)
    assert zeta.real == pytest.approx(-0.5 * xi.real, abs=1e-12)


@pytest.mark.parametrize(
    "system,m_c",
    [(_purpose_built_mc1_mL2_system(), 1), (_purpose_built_mc2_system(), 2)],
)
@pytest.mark.parametrize(
    "operator_terms,mu",
    [
        ({3: 1.0}, lambda m: -(m**3)),
        ({1: 1.0, 3: 0.17}, lambda m: m - 0.17 * m**3),
    ],
)
def test_odd_hamiltonian_multiplier_quadratic_rescaling(system, m_c, operator_terms, mu):
    flux = _derivative_free_quadratic_flux(0.4, 1.1, -0.3)
    baseline = _apply_odd_operator(*flux, {1: 1.0})
    nonlinearity = _apply_odd_operator(*flux, operator_terms)
    _assert_admissible_generated_nonlinearity(nonlinearity)

    zeta_0, xi_0 = system.compute_normal_form(m_c, {}, baseline)
    zeta, xi = system.compute_normal_form(m_c, {}, nonlinearity)
    scale = mu(m_c) * mu(2 * m_c) / (2 * m_c**2)
    assert zeta == pytest.approx(scale * zeta_0, abs=1e-12)
    assert xi == pytest.approx(scale * xi_0, abs=1e-12)


def test_order_budget_is_combined_for_derivative_dependent_density_and_multiplier():
    derivative_dependent_dx = _variational_density_nonlinearity("u_ux2", {1: 1.0})
    derivative_dependent_dxxx = _variational_density_nonlinearity("u_ux2", {3: 1.0})
    derivative_free_dxxx = _apply_odd_operator(
        *_derivative_free_quadratic_flux(1.0, 0.5, 1.0), {3: 1.0}
    )

    assert _max_derivative_order(derivative_dependent_dx) == 3 == 2 * 2 - 1
    assert _max_derivative_order(derivative_dependent_dxxx) == 5 > 2 * 2 - 1
    assert _max_derivative_order(derivative_free_dxxx) == 3 == 2 * 2 - 1


def test_quadratic_variational_example_does_not_force_c11_real_part_to_zero():
    system = _purpose_built_mc1_mL2_system()
    nonlinearity = _variational_density_nonlinearity("u_ux2")
    zeta, xi = system.compute_normal_form(1, {}, nonlinearity)
    c11_real = zeta.real  # There is no cubic part, so zeta = c_hat_{1,1}.
    assert abs(c11_real) > 1e-8
    assert xi.real == pytest.approx(0.0, abs=1e-12)


def _li_yao_2015_coefficients(a: float, c: float) -> tuple[complex, complex]:
    """Cubic coefficients (b_0, c_0) of Li--Yao, Physica D 310 (2015), Eqs. (4.9)--(4.11).

    Specialized to the quadratic stress law sigma(u) = c^2 u + u^2 / 2 at wavenumber
    k_0 = 1, i.e. sigma'(0) = c^2, sigma''(0) = 2, sigma'''(0) = 0, with a_c = a and
    delta_c = a + 1.
    """
    k0, sp, spp, sppp, ac, dc = 1.0, c**2, 2.0, 0.0, a, a + 1.0
    w = np.sqrt(sp * k0**2 - ac**2 * k0**8)
    alpha = (-2 * sp * k0**2 + 2 * w**2 - 32 * k0**8 * ac * (3 - ac)) ** 2 \
        + 144 * k0**4 * w**2 * dc**2
    b0 = complex(
        -6 * k0**6 * spp**2 * dc / alpha,
        k0**4 * spp**2 * (-sp * k0**2 / w + w - 16 * k0**8 * ac * (3 - ac) / w) / alpha,
    )
    c0 = complex(
        0.0,
        -k0**2 * spp**2 / (2 * w * (sp - 16 * ac * k0**4 * (dc - 4 * k0**2)))
        + sppp * k0**2 / (2 * w),
    )
    return b0, c0


@pytest.mark.parametrize("a,c", [(0.5, 1.0), (0.2, 0.9), (0.1, 2.0), (0.3, 1.5)])
def test_application_1_quadratic_case_matches_li_yao_2015(a, c):
    """The eta = 0 coefficients equal those of Li--Yao up to the 4c^2 normalization.

    This is the agreement asserted in the Applications section of the paper; the factor
    4c^2 is the same one relating our coefficients to Yao's b and c.
    """
    system = application_1_system()
    zeta, xi = system.compute_normal_form(
        1, hopf_params(a, c), application_1_nonlinearity(0.0)
    )
    b0, c0 = _li_yao_2015_coefficients(a, c)
    assert b0 == pytest.approx(4 * c**2 * zeta)
    assert c0 == pytest.approx(4 * c**2 * xi)
