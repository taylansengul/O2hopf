"""Two gaps left by test_o2sym.py's Hamiltonian-cancellation coverage.

That module already checks the cancellation for named derivative-dependent
densities, the defect identities, the ratio Re s_11 = -1/2 Re s_12, the
multiplier rescaling, and the positive control on Re c_11 -- all on fixed,
purpose-built linear operators.  Two things it does not check:

1.  The theorem quantifies over *every* admissible L, not over a fixed one.
    Here the linear part is sampled and its admissibility (Assumption H) is
    verified rather than assumed, over two structurally different families.

2.  The quadratic defect identity itself.  test_o2sym.py checks the two cubic
    identities in absolute value and the ratio, but for the quadratic one it
    only checks how the coefficients rescale with the multiplier, never the
    identity's own right-hand side in the resolvent entries.

3.  test_bilinear.py shows that the diagonal g11 = g22 forces Re xi = 0 and
    that one off-diagonal point also cancels.  Neither shows that the diagonal
    is the *whole* variational locus, which is what makes the off-diagonal
    branch a witness that no converse holds.  That is settled symbolically.

The Hamiltonians and the Euler operator below are built with sympy, so this
module is skipped when sympy is absent.
"""
import itertools
import random

import numpy as np
import pytest

from o2sym.core import CoeffDict, O2HopfNormalForm

sp = pytest.importorskip("sympy", reason="symbolic Hamiltonians need sympy")

R = 7  # symbol depth; must exceed every derivative order reached
U = sp.symbols(f"u0:{R+2}")
V = sp.symbols(f"v0:{R+2}")


def Dx(expr):
    """Total x-derivative.  Raises rather than silently truncating."""
    free = getattr(expr, "free_symbols", set())
    if U[R + 1] in free or V[R + 1] in free:
        raise RuntimeError("symbol table too shallow; raise R")
    return sum(
        sp.diff(expr, U[j]) * U[j + 1] + sp.diff(expr, V[j]) * V[j + 1]
        for j in range(R + 1)
    )


def euler(expr, W):
    """Variational derivative delta/delta w = sum_j (-D_x)^j d/dw_j."""
    out = sp.Integer(0)
    for j in range(R + 1):
        term = sp.diff(expr, W[j])
        for _ in range(j):
            term = -Dx(term)
        out += term
    return sp.expand(out)


def K_apply(expr, P):
    """K = d_x P(-d_x^2), with P(z) = sum_k P[k] z^k."""
    out = sp.Integer(0)
    for k, c in enumerate(P):
        term = expr
        for _ in range(2 * k):
            term = Dx(term)
        out += c * (-1) ** k * term
    return sp.expand(Dx(out))


def reflect(expr):
    """Reflection on a density: u_j -> (-1)^j u_j, v_j -> (-1)^(j+1) v_j."""
    sub = {U[j]: (-1) ** j * U[j] for j in range(R + 2)}
    sub.update({V[j]: (-1) ** (j + 1) * V[j] for j in range(R + 2)})
    return sp.expand(expr.subs(sub, simultaneous=True))


def invariant_monomials(degree, max_deriv=1):
    out = []
    syms = [U[j] for j in range(max_deriv + 1)] + [V[j] for j in range(max_deriv + 1)]
    for combo in itertools.combinations_with_replacement(syms, degree):
        m = sp.prod(combo)
        if sp.expand(reflect(m) - m) == 0:
            out.append(m)
    return out


def random_density(degree, rng, max_deriv=1):
    terms = invariant_monomials(degree, max_deriv)
    assert terms
    return sum(sp.Integer(rng.randint(-4, 4)) * t for t in terms)


def to_nonlinearity(g1, g2):
    """Encode two differential polynomials in this package's coefficient format."""
    out = {}
    for comp, g in enumerate((g1, g2)):
        g = sp.expand(g)
        if g == 0:
            continue
        poly = sp.Poly(g, *U, *V)
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            au = tuple(int(e) for e in monom[: R + 2])
            av = tuple(int(e) for e in monom[R + 2 :])
            while au and au[-1] == 0:
                au = au[:-1]
            while av and av[-1] == 0:
                av = av[:-1]
            key = (au or (0,), av or (0,))
            out.setdefault(key, np.zeros(2, dtype=complex))[comp] += complex(coeff)
    return out


def variational_flux(H3, H4, P):
    """G = S K delta H / delta U, i.e. (K dH/dv, K dH/du)."""
    g1 = g2 = sp.Integer(0)
    for H in (H3, H4):
        g1 += K_apply(euler(H, V), P)
        g2 += K_apply(euler(H, U), P)
    return to_nonlinearity(sp.expand(g1), sp.expand(g2))


def admissible(system, m_c, params, mmax=60):
    """Assumption H, checked on the symbol: the crossing, positivity, a gap."""
    if abs(np.trace(system.M_matrix(m_c, params)).real) > 1e-9:
        return False
    for m in range(1, mmax + 1):
        M = system.M_matrix(m, params)
        if np.linalg.det(M).real <= 1e-9:
            return False
        if m != m_c and np.trace(M).real >= -1e-9:
            return False
    ratios = [
        np.linalg.det(system.M_matrix(m, params)).real
        / abs(np.trace(system.M_matrix(m, params)).real)
        for m in range(40, mmax + 1)
    ]
    return min(ratios) >= 1e-6


LI_YAO: CoeffDict = {
    (1, 0): 0.0, (1, 4): lambda p: -p["a"], (1, 1): 1.0,
    (2, 1): lambda p: p["c"] ** 2, (2, 0): 0.0,
    (2, 2): lambda p: -p["delta"], (2, 4): -1.0,
}

# M11 = M22 = r - s m^2 - m^4; the crossing at m_c = 1 forces r = s + 1.
ALT: CoeffDict = {
    (1, 0): lambda p: p["r"], (1, 2): lambda p: p["s"], (1, 4): -1.0,
    (1, 1): lambda p: p["g1"], (1, 3): lambda p: p["g3"],
    (2, 0): lambda p: p["r"], (2, 2): lambda p: p["s"], (2, 4): -1.0,
    (2, 1): lambda p: p["h1"], (2, 3): lambda p: p["h3"],
}


def li_yao_params(rng):
    a = rng.uniform(0.05, 2.0)
    return {"a": a, "c": rng.uniform(0.5, 2.5), "delta": 1.0 + a}


def alt_params(rng):
    s = rng.uniform(0.1, 3.0)
    return {"r": s + 1.0, "s": s, "g1": rng.uniform(-2, 2), "g3": rng.uniform(-1, 1),
            "h1": rng.uniform(-2, 2), "h3": rng.uniform(-1, 1)}


@pytest.mark.parametrize(
    "coeffs,sampler", [(LI_YAO, li_yao_params), (ALT, alt_params)]
)
@pytest.mark.parametrize("P", [[1.0], [0.0, 1.0], [2.0, -1.0]])
def test_cancellation_holds_over_sampled_admissible_operators(coeffs, sampler, P):
    """Re xi = 0 for every admissible L, not only for a purpose-built one."""
    rng = random.Random(11)
    H3, H4 = random_density(3, rng), random_density(4, rng)
    system = O2HopfNormalForm(coeffs, m_L=2)
    nl = variational_flux(H3, H4, P)

    rng = random.Random(5)
    tested = 0
    for _ in range(120):
        params = sampler(rng)
        if not admissible(system, 1, params):
            continue
        zeta, xi = system.compute_normal_form(1, params, nl)
        if not (np.isfinite(zeta.real) and np.isfinite(xi.real)):
            continue
        tested += 1
        assert abs(xi.real) <= 1e-9 * max(abs(zeta), abs(xi), 1.0)
    assert tested >= 20, "too few admissible operators sampled to be meaningful"


def test_bilinear_variational_locus_is_exactly_the_diagonal():
    """The g11 = g22 diagonal is the whole variational locus of the family.

    test_bilinear.py checks that the diagonal cancels and that one off-diagonal
    point cancels too.  Together with this, that second point is a genuine
    counterexample to a converse: Re xi = 0 there without variationality.
    """
    g11, g21, g22 = sp.symbols("g11 g21 g22", real=True)
    mono = invariant_monomials(3)
    coeffs = [sp.Symbol(f"h{i}") for i in range(len(mono))]
    H3 = sum(c * m for c, m in zip(coeffs, mono))

    residual_1 = sp.expand(K_apply(euler(H3, V), [1]) - g11 * (U[1] * V[0] + U[0] * V[1]))
    residual_2 = sp.expand(K_apply(euler(H3, U), [1]) - (g21 * U[0] * U[1] + g22 * V[0] * V[1]))
    gens = [U[0], U[1], U[2], U[3], V[0], V[1], V[2], V[3]]
    equations = [c for r in (residual_1, residual_2) for c in sp.Poly(r, *gens).coeffs()]

    solutions = sp.solve(equations, coeffs + [g11, g21, g22], dict=True)
    assert len(solutions) == 1
    assert solutions[0][g11] == g22

    # H_3 = g21 u^3 / 6 + g22 u v^2 / 2 realizes the locus, and the closed-form
    # numerator of Re xi in the paper's Section 5.3 vanishes on it.
    c = sp.Symbol("c", positive=True)
    numerator = (g11 - g22) * (c ** 2 * (g11 - 4 * g22) + 4 * g21)
    assert sp.simplify(numerator.subs({g11: g22})) == 0


def ksym(m, P):
    """Real odd symbol of K: k(m) = m P(m^2)."""
    return m * sum(c * (m ** 2) ** k for k, c in enumerate(P))


def general_flux(g, h, P):
    """K F, with F the general reflection-compatible derivative-free flux."""
    g11, g21, g22 = g
    h11, h12, h21, h22 = h
    F1 = g11 * U[0] * V[0] + h11 * U[0] ** 2 * V[0] + h12 * V[0] ** 3
    F2 = (sp.Rational(1, 2) * (g21 * U[0] ** 2 + g22 * V[0] ** 2)
          + h21 * U[0] ** 3 + h22 * U[0] * V[0] ** 2)
    return to_nonlinearity(K_apply(F1, P), K_apply(F2, P))


@pytest.mark.parametrize("P", [[1.0], [0.0, 1.0], [2.0, -1.0], [0.0, 0.0, 1.0]])
def test_quadratic_defect_identity_holds_in_absolute_value(P):
    """Re c_12 equals its right-hand side, not merely its rescaling.

    Re c_12 = (k(m_c) k(2 m_c) / 2) D_2
              [ rho_22 (g22 |v_c|^2 - g21 |u_c|^2) - 2 rho_21 g11 chi ],
    with D_2 = g22 - g11 and rho the entries of the inverse symbol at 2 m_c.
    """
    system = O2HopfNormalForm(LI_YAO, m_L=2)
    rng = random.Random(2)
    tested = 0
    for _ in range(120):
        params = li_yao_params(rng)
        if not admissible(system, 1, params):
            continue
        g = [rng.uniform(-2, 2) for _ in range(3)]
        h = [rng.uniform(-2, 2) for _ in range(4)]
        nl = general_flux(g, h, P)

        b1, b2, q1, q2, s1, s2 = system.eigendata(1, params)
        _c11, c12 = system.compute_c_hat(1, params, (q1, q2), (s1, s2), (b1, b2), nl)

        u_c, v_c = q1
        chi = (u_c * np.conj(v_c)).imag
        Minv = np.linalg.inv(system.M_matrix(2, params))
        rho22, rho21 = Minv[1, 1].real, (Minv[1, 0] / 1j).real
        predicted = (ksym(1, P) * ksym(2, P) / 2) * (g[2] - g[0]) * (
            rho22 * (g[2] * abs(v_c) ** 2 - g[1] * abs(u_c) ** 2) - 2 * rho21 * g[0] * chi
        )

        tested += 1
        assert c12.real == pytest.approx(predicted, abs=1e-9 * max(abs(c12), 1.0))
    assert tested >= 20
