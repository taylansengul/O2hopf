from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Dict, Iterable, List, Tuple

Monomial = Tuple[int, ...]
Poly = Dict[Monomial, complex]


def _trim(t: Tuple[int, ...]) -> Tuple[int, ...]:
    t = tuple(int(x) for x in t)
    while t and t[-1] == 0:
        t = t[:-1]
    return t


def _pad(t: Tuple[int, ...], n: int) -> Tuple[int, ...]:
    return t + (0,) * max(0, n - len(t))


def _add_poly(a: Poly, b: Poly, scale: complex = 1.0) -> Poly:
    out = dict(a)
    for mon, coeff in b.items():
        out[mon] = out.get(mon, 0.0j) + scale * coeff
        if abs(out[mon]) < 1e-12:
            del out[mon]
    return out


def _mul_poly(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            n = max(len(ma), len(mb))
            mon = tuple(x + y for x, y in zip(_pad(ma, n), _pad(mb, n)))
            mon = _trim(mon)
            out[mon] = out.get(mon, 0.0j) + ca * cb
    return {m: c for m, c in out.items() if abs(c) >= 1e-12}


def _dx_monomial(mon: Monomial) -> Poly:
    """Total x-derivative of prod_j w_j^{mon_j}, where w may mean u or v variables."""
    out: Poly = {}
    mon = _trim(mon)
    for j, exponent in enumerate(mon):
        if exponent == 0:
            continue
        n = max(len(mon), j + 2)
        new = list(_pad(mon, n))
        new[j] -= 1
        new[j + 1] += 1
        new_t = _trim(tuple(new))
        out[new_t] = out.get(new_t, 0.0j) + exponent
    return out


def _dx_poly(poly: Poly) -> Poly:
    out: Poly = {}
    for mon, coeff in poly.items():
        out = _add_poly(out, _dx_monomial(mon), scale=coeff)
    return out


def _dx_power(poly: Poly, order: int) -> Poly:
    out = dict(poly)
    for _ in range(order):
        out = _dx_poly(out)
    return out


def _partial_monomial(alpha: Monomial, r: int) -> Tuple[complex, Monomial] | None:
    alpha = _trim(alpha)
    exponent = alpha[r] if r < len(alpha) else 0
    if exponent == 0:
        return None
    new = list(_pad(alpha, r + 1))
    new[r] -= 1
    return exponent, _trim(tuple(new))


@dataclass(frozen=True)
class Term:
    """One scalar differential monomial coefficient*(Du)^alpha*(Dv)^beta."""

    alpha_u: Tuple[int, ...]
    alpha_v: Tuple[int, ...]
    coefficient: complex

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_u", _trim(self.alpha_u))
        object.__setattr__(self, "alpha_v", _trim(self.alpha_v))
        object.__setattr__(self, "coefficient", complex(self.coefficient))


class ConservativeChecker:
    """Euler-operator test for 1D differential-polynomial conservative form.

    For a scalar expression g(u,v,u_x,v_x,...), the expression is a total
    x-derivative iff E_u(g)=E_v(g)=0, modulo constants, in the differential
    polynomial algebra.
    """

    def __init__(self, terms: Iterable[Term], tolerance: float = 1e-10):
        self.terms = list(terms)
        self.tolerance = float(tolerance)

    def _euler_component(self, variable: str) -> Dict[Tuple[Monomial, Monomial], complex]:
        out: Dict[Tuple[Monomial, Monomial], complex] = {}
        for term in self.terms:
            alpha = term.alpha_u if variable == "u" else term.alpha_v
            other = term.alpha_v if variable == "u" else term.alpha_u
            max_r = len(alpha)
            for r in range(max_r):
                partial = _partial_monomial(alpha, r)
                if partial is None:
                    continue
                multiplier, alpha_reduced = partial
                base_coeff = term.coefficient * multiplier

                # Apply (-D_x)^r to (selected variable monomial)*(other variable monomial).
                selected_poly: Poly = {alpha_reduced: 1.0}
                other_poly: Poly = {other: 1.0}
                deriv_sum: Dict[Tuple[Monomial, Monomial], complex] = {}
                for a in range(r + 1):
                    left = _dx_power(selected_poly, a)
                    right = _dx_power(other_poly, r - a)
                    prod_coeff = comb(r, a)
                    for ml, cl in left.items():
                        for mr, cr in right.items():
                            if variable == "u":
                                key = (ml, mr)
                            else:
                                key = (mr, ml)
                            deriv_sum[key] = deriv_sum.get(key, 0.0j) + prod_coeff * cl * cr
                sign = (-1) ** r
                for key, coeff in deriv_sum.items():
                    out[key] = out.get(key, 0.0j) + sign * base_coeff * coeff
                    if abs(out[key]) < self.tolerance:
                        del out[key]
        return {k: v for k, v in out.items() if abs(v) >= self.tolerance}

    def euler_u(self) -> Dict[Tuple[Monomial, Monomial], complex]:
        return self._euler_component("u")

    def euler_v(self) -> Dict[Tuple[Monomial, Monomial], complex]:
        return self._euler_component("v")

    def is_conservative(self) -> bool:
        return not self.euler_u() and not self.euler_v()

    @staticmethod
    def from_nonlinearity_component(nonlinearity: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], object], component: int) -> "ConservativeChecker":
        terms: List[Term] = []
        for (alpha_u, alpha_v), coeff_vec in nonlinearity.items():
            coeff = coeff_vec[component]
            if abs(coeff) > 1e-14:
                terms.append(Term(alpha_u, alpha_v, coeff))
        return ConservativeChecker(terms)
