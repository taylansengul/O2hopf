"""Wave-selection coverage for the conservative bilinear family on the Li--Yao operator.

The nonlinearity is the three-parameter conservative, O(2)-equivariant quadratic

    G(u, v) = [ g11 (u_x v + u v_x) ;  g21 u u_x + g22 v v_x ],   g = (g11, g21, g22) in R^3,

each component a total x-derivative.  On the fixed linear operator of
:func:`o2sym.core.application_1_system` (Hopf at m_c = 1, delta = 1 + a), the cubic
normal-form coefficients zeta = c_hat_{1,1} and xi = c_hat_{1,2} are homogeneous
quadratic forms in g, so their real parts -- which alone decide the region of
Table (signs) -- are

    Re zeta = g^T A g,    Re xi = g^T B g,

with symmetric 3x3 matrices A, B returned by :func:`extract_quadratic_forms`.

This module reproduces the paper's coverage claims:
  * at a > 0 the full family realizes all six regions (:func:`reachable_regions`);
  * g11 is indispensable and, at moderate a, no two parameters suffice
    (:func:`minimal_subsets`).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Sequence, Tuple

import numpy as np

from .classification import classify_region
from .core import application_1_system

Vector3 = Sequence[float]
PARAM_NAMES: Tuple[str, str, str] = ("g11", "g21", "g22")


def bilinear_nonlinearity(g11: float, g21: float, g22: float) -> Dict:
    """Return the nonlinearity dictionary for G with parameters (g11, g21, g22)."""
    nl: Dict = {}

    def add(au: Tuple[int, ...], av: Tuple[int, ...], vec) -> None:
        au = tuple(au)
        av = tuple(av)
        v = np.array(vec, dtype=complex)
        nl[(au, av)] = v if (au, av) not in nl else nl[(au, av)] + v

    add((0, 1), (1,), [g11, 0.0])   # u_x v   in equation 1
    add((1,), (0, 1), [g11, 0.0])   # u v_x   in equation 1
    add((1, 1), (0,), [0.0, g21])   # u u_x   in equation 2
    add((0,), (1, 1), [0.0, g22])   # v v_x   in equation 2
    return nl


def hopf_params(a: float, c: float) -> Dict[str, float]:
    """Parameter dictionary placing the Hopf point at m_c = 1 (delta = 1 + a)."""
    return {"a": float(a), "c": float(c), "delta": 1.0 + float(a)}


def normal_form_coeffs(g: Vector3, a: float, c: float) -> Tuple[complex, complex]:
    """Return the complex coefficients (zeta, xi) for parameters g on the Li--Yao operator."""
    system = application_1_system()
    z, x = system.compute_normal_form(1, hopf_params(a, c), bilinear_nonlinearity(*g))
    return complex(z), complex(x)


def region_for_g(g: Vector3, a: float, c: float) -> str:
    """Classify the wave-selection region (string label) for parameters g."""
    z, x = normal_form_coeffs(g, a, c)
    return classify_region(z, x)["region"]


def extract_quadratic_forms(a: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """Recover symmetric A, B with Re zeta = g^T A g and Re xi = g^T B g by polarization."""
    e = np.eye(3)
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    diag_z = np.zeros(3)
    diag_x = np.zeros(3)
    for i in range(3):
        z, x = normal_form_coeffs(e[i], a, c)
        A[i, i] = diag_z[i] = z.real
        B[i, i] = diag_x[i] = x.real
    for i in range(3):
        for j in range(i + 1, 3):
            z, x = normal_form_coeffs(e[i] + e[j], a, c)
            A[i, j] = A[j, i] = 0.5 * (z.real - diag_z[i] - diag_z[j])
            B[i, j] = B[j, i] = 0.5 * (x.real - diag_x[i] - diag_x[j])
    return A, B


def real_parts(A: np.ndarray, B: np.ndarray, g: Vector3) -> Tuple[float, float]:
    """Evaluate (Re zeta, Re xi) = (g^T A g, g^T B g)."""
    g = np.asarray(g, dtype=float)
    return float(g @ A @ g), float(g @ B @ g)


def _region_index(z: np.ndarray, x: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Vectorized region index in {1,...,6}; 0 marks the boundary set."""
    zp = z + x
    zm = z - x
    R = np.zeros(np.shape(z), dtype=int)
    R[(zp > tol) & (zm > tol)] = 1
    R[(z > tol) & (zm < -tol)] = 2
    R[(z < -tol) & (zp > tol)] = 3
    R[(zp < -tol) & (zm < -tol)] = 4
    R[(z < -tol) & (zm > tol)] = 5
    R[(z > tol) & (zp < -tol)] = 6
    return R


@dataclass
class GridScan:
    """A scan of the (g11, g22)-plane at fixed g21."""
    g11: np.ndarray
    g22: np.ndarray
    re_zeta: np.ndarray
    re_xi: np.ndarray
    region: np.ndarray   # integer array in {0,...,6}

    @property
    def regions_present(self) -> FrozenSet[int]:
        return frozenset(int(k) for k in np.unique(self.region) if k != 0)


def scan_grid(a: float, c: float, g21: float = 1.0,
              half_width: float = 0.5, n: int = 400) -> GridScan:
    """Scan the (g11, g22)-plane at fixed g21 over [-half_width, half_width]^2."""
    A, B = extract_quadratic_forms(a, c)
    t = np.linspace(-half_width, half_width, n)
    G11, G22 = np.meshgrid(t, t)

    def form(M: np.ndarray) -> np.ndarray:
        return (M[0, 0] * G11**2 + M[2, 2] * G22**2 + 2 * M[0, 2] * G11 * G22
                + 2 * M[0, 1] * g21 * G11 + 2 * M[1, 2] * g21 * G22 + M[1, 1] * g21**2)

    Z = form(A)
    X = form(B)
    return GridScan(G11, G22, Z, X, _region_index(Z, X))


def reachable_regions(a: float, c: float, npts: int = 60000, seed: int = 1) -> Dict:
    """Sample the full 3-parameter family on the unit sphere; report regions hit and angular span.

    Returns a dict with keys ``regions`` (set of labels), ``count`` (number of distinct
    regions), and ``angular_coverage_deg`` (angular span of the reachable cone in the
    (Re zeta, Re xi)-plane; ~360 means the whole plane, hence all six regions).
    """
    A, B = extract_quadratic_forms(a, c)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((npts, 3))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    z = np.einsum("ni,ij,nj->n", G, A, G)
    x = np.einsum("ni,ij,nj->n", G, B, G)
    R = _region_index(z, x)
    labels = {_INDEX_TO_LABEL[int(k)] for k in np.unique(R) if k != 0}
    ang = np.sort(np.arctan2(x, z))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    coverage_deg = float(np.degrees(2 * np.pi - gaps.max()))
    return {"regions": labels, "count": len(labels), "angular_coverage_deg": coverage_deg}


def _subset_regions(A: np.ndarray, B: np.ndarray, idx: Sequence[int],
                    npts: int, seed: int) -> FrozenSet[str]:
    sub = list(idx)
    As = A[np.ix_(sub, sub)]
    Bs = B[np.ix_(sub, sub)]
    rng = np.random.default_rng(seed)
    k = len(sub)
    Q = rng.standard_normal((npts, k))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    z = np.einsum("ni,ij,nj->n", Q, As, Q)
    x = np.einsum("ni,ij,nj->n", Q, Bs, Q)
    R = _region_index(z, x)
    return frozenset(_INDEX_TO_LABEL[int(kk)] for kk in np.unique(R) if kk != 0)


def minimal_subsets(a: float, c: float, npts: int = 40000, seed: int = 7) -> Dict:
    """Regions reachable by each nonempty subset of {g11, g21, g22}.

    Returns a dict mapping the subset (tuple of names) to the frozenset of regions it
    reaches, plus the key ``reach_all_six`` listing the subsets that attain all six.
    """
    A, B = extract_quadratic_forms(a, c)
    out: Dict = {}
    all_six = []
    for r in (1, 2, 3):
        for combo in combinations(range(3), r):
            names = tuple(PARAM_NAMES[i] for i in combo)
            regs = _subset_regions(A, B, combo, npts, seed)
            out[names] = regs
            if len(regs) == 6:
                all_six.append(names)
    out["reach_all_six"] = all_six
    return out


_INDEX_TO_LABEL = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
