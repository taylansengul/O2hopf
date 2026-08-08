"""Tests for the bilinear-family coverage and the pseudo-spectral confirmation."""
import numpy as np

from o2sym import (
    application_1_system,
    expm2,
    hopf_params,
    minimal_subsets,
    normal_form_coeffs,
    reachable_regions,
    simulate_system2,
)
from o2sym.coverage import bilinear_nonlinearity
from o2sym.conservative import ConservativeChecker


def _dense_expm(M, t):
    w, V = np.linalg.eig(M)
    return V @ np.diag(np.exp(w * t)) @ np.linalg.inv(V)


def test_expm2_matches_eigendecomposition():
    """Closed-form 2x2 exponential agrees with an eigen-based reference on the operator modes."""
    sys = application_1_system()
    p = dict(hopf_params(0.5, 1.0)); p["delta"] = 1.55
    Mstack = np.array([sys.M_matrix(m, p) for m in range(-42, 43)], dtype=complex)
    got = expm2(Mstack, 0.2)
    for k, m in enumerate(range(-42, 43)):
        assert np.allclose(got[k], _dense_expm(Mstack[k], 0.2), atol=1e-12)


def test_expm2_defective_jordan_block():
    """The s -> 0 (repeated-eigenvalue) branch reproduces a true Jordan block exponential."""
    M = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=complex)
    got = expm2(M[None], 0.7)[0]
    want = np.exp(2 * 0.7) * np.array([[1.0, 0.7], [0.0, 1.0]])
    assert np.allclose(got, want, atol=1e-12)


def test_both_components_conservative():
    nl = bilinear_nonlinearity(1.0, 1.0, 1.0)
    for comp in (0, 1):
        assert ConservativeChecker.from_nonlinearity_component(nl, comp).is_conservative()


def test_all_six_regions_reachable_at_positive_a():
    rr = reachable_regions(0.5, 1.0)
    assert rr["count"] == 6
    assert rr["regions"] == {"I", "II", "III", "IV", "V", "VI"}


def test_g11_indispensable_and_triple_minimal():
    ms = minimal_subsets(0.5, 1.0)
    assert ms["reach_all_six"] == [("g11", "g21", "g22")]      # only the full triple
    assert ms[("g21", "g22")] == frozenset({"IV", "V", "VI"})  # g11 = 0 loses I, II, III


def test_elasticity_point_on_xi_zero_line():
    """The stress--strain member g = (0, 1, 0) has Re xi = 0 (pinned to standing waves)."""
    zeta, xi = normal_form_coeffs((0.0, 1.0, 0.0), 0.5, 1.0)
    assert abs(xi.real) < 1e-9
    assert zeta.real < 0


def test_hamiltonian_diagonal_forces_re_xi_zero():
    """The Hamiltonian compatibility g11 = g22 forces Re xi = 0 along the diagonal."""
    for g in (-0.4, 0.0, 0.3):
        _, xi = normal_form_coeffs((g, 1.0, g), 0.5, 1.0)
        assert abs(xi.real) < 1e-9


def test_re_xi_zero_does_not_imply_hamiltonian_compatibility():
    """The operator-specific second factor can vanish off the Hamiltonian diagonal."""
    _, xi = normal_form_coeffs((-4.0, 1.0, 0.0), 0.5, 1.0)
    assert abs(xi.real) < 1e-9


def test_wave_selection_split():
    """Direct integration selects a Hamiltonian standing wave and a non-Hamiltonian traveling wave."""
    r_iv = simulate_system2(0.5, 1.0, (0.0, 1.0, 0.0), delta_minus_dc=0.05, T=1000.0, tail=45.0, seed=1)
    r_v = simulate_system2(0.5, 1.0, (0.35, 1.0, -0.30), delta_minus_dc=0.05, T=1000.0, tail=45.0, seed=1)
    assert r_iv.modulation_depth > 0.5 and r_iv.label == "standing wave"
    assert r_v.modulation_depth < 0.1 and r_v.label == "traveling wave"
