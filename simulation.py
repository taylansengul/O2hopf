"""Pseudo-spectral confirmation of wave selection for the bilinear family.

The two-component system on the 1D torus [0, 2*pi),

    u_t = -a u_xxxx + v_x + g11 (u v)_x,
    v_t =  c^2 u_x - delta v_xx - v_xxxx + g21 u u_x + g22 v v_x,

is integrated by an integrating-factor RK4 (IFRK4) scheme: the linear part is
advanced exactly mode-by-mode with the 2x2 matrix exponential of the Fourier
symbol M_m (the operator of :func:`o2sym.core.application_1_system`), and the
conservative quadratic nonlinearity is evaluated pseudo-spectrally with 2/3
dealiasing.

The matrix exponential is computed in closed form (no SciPy) via

    exp(M t) = e^{(tr M / 2) t} [ cosh(s t) I + t * sinhc(s t) (M - (tr M / 2) I) ],
    s = sqrt((tr M / 2)^2 - det M),    sinhc(z) = sinh(z)/z,

which is exact for every 2x2 M (the s -> 0 / defective limit is handled by a
series in :func:`_sinhc`) and vectorizes over all modes at once.

A rotating (traveling) wave saturates to |u_hat_{+1}(t)| = const; a standing
wave pulses to zero, so the tail modulation depth (max-min)/(max+min) separates
the two: ~0 for traveling, ~1 for standing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .core import application_1_system
from .coverage import hopf_params

Vector3 = Sequence[float]


def _sinhc(z: np.ndarray) -> np.ndarray:
    """sinh(z)/z, evaluated stably (including the z -> 0 limit) for complex z."""
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 1e-4
    z_safe = np.where(small, 1.0, z)
    ratio = np.sinh(z) / z_safe
    series = 1.0 + z * z / 6.0 + z**4 / 120.0
    return np.where(small, series, ratio)


def expm2(M: np.ndarray, t: float) -> np.ndarray:
    """Exact matrix exponential exp(M t) for a stack of 2x2 matrices.

    ``M`` has shape (..., 2, 2); the return value has the same shape.  Evaluated in the
    spectral form ``0.5(e^{l+ t}+e^{l- t}) I + (e^{l+ t}-e^{l- t})/(2s) (M - 0.5 tr M I)``
    with ``l+- = 0.5 tr M +- s`` and ``s = sqrt((0.5 tr M)^2 - det M)``.  This folds the
    average-eigenvalue prefactor into the exponentials, so strongly damped modes decay to
    zero instead of producing overflow; the defective ``s -> 0`` limit uses a series.
    """
    M = np.asarray(M, dtype=complex)
    tr = M[..., 0, 0] + M[..., 1, 1]
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    half = 0.5 * tr
    s = np.sqrt(half * half - det)
    ep = np.exp((half + s) * t)
    em = np.exp((half - s) * t)
    co_I = 0.5 * (ep + em)
    st = s * t
    small = np.abs(st) < 1e-4
    denom = np.where(small, 1.0, 2.0 * s)
    series = np.exp(half * t) * t * _sinhc(np.where(small, st, 0.0))
    co_M = np.where(small, series, (ep - em) / denom)
    eye = np.eye(2, dtype=complex)
    return (co_M[..., None, None] * M
            + (co_I - co_M * half)[..., None, None] * eye)


@dataclass
class SimulationResult:
    """Outcome of a single integration."""
    a: float
    c: float
    g: Tuple[float, float, float]
    delta_minus_dc: float
    x: np.ndarray                 # spatial grid
    snap_times: np.ndarray        # times of the saved tail snapshots
    snapshots: np.ndarray         # real u(x, t) on the tail, shape (n_snap, N)
    mode1_amplitude: np.ndarray   # |u_hat_{+1}(t)| on the tail
    saturated_amplitude: float    # final |u_hat_{+1}|
    modulation_depth: float       # (max - min) / (max + min) on the tail
    drift_speed: float            # tail phase speed of mode +1 (|.| ~ const for TW)

    @property
    def label(self) -> str:
        return selection_label(self)


def selection_label(result: "SimulationResult", tol: float = 0.5) -> str:
    """'standing wave' if the tail modulation depth exceeds ``tol``, else 'traveling wave'."""
    return "standing wave" if result.modulation_depth > tol else "traveling wave"


def simulate_system2(a: float, c: float, g: Vector3, delta_minus_dc: float = 0.05,
                     N: int = 128, h: float = 0.2, T: float = 1200.0,
                     tail: float = 200.0, seed: int = 1) -> SimulationResult:
    """Integrate the system at supercriticality ``delta = (1 + a) + delta_minus_dc``.

    Returns a :class:`SimulationResult`; ``result.label`` reports the selected wave.
    """
    g11, g21, g22 = (float(g[0]), float(g[1]), float(g[2]))
    params = dict(hopf_params(a, c))
    params["delta"] = (1.0 + a) + delta_minus_dc

    system = application_1_system()
    L = 2 * np.pi
    x = L * np.arange(N) / N
    m = np.fft.fftfreq(N, d=1.0 / N)            # integer wavenumbers
    im = 1j * m
    mask = (np.abs(m) < (N / 3)).astype(float)  # 2/3 dealiasing

    Mmat = np.array([system.M_matrix(int(round(mk)), params) for mk in m], dtype=complex)
    E = expm2(Mmat, h)
    E2 = expm2(Mmat, 0.5 * h)

    def Nhat(V: np.ndarray) -> np.ndarray:
        uh, vh = V[:, 0], V[:, 1]
        u = np.fft.ifft(uh); v = np.fft.ifft(vh)
        ux = np.fft.ifft(im * uh); vx = np.fft.ifft(im * vh)
        n1 = g11 * (ux * v + u * vx)
        n2 = g21 * u * ux + g22 * v * vx
        return np.stack([np.fft.fft(n1) * mask, np.fft.fft(n2) * mask], axis=1)

    def lin(P: np.ndarray, W: np.ndarray) -> np.ndarray:
        return np.einsum("kij,kj->ki", P, W)

    def step(V: np.ndarray) -> np.ndarray:
        a1 = h * Nhat(V)
        b1 = h * Nhat(lin(E2, V + 0.5 * a1))
        c1 = h * Nhat(lin(E2, V) + 0.5 * b1)
        d1 = h * Nhat(lin(E, V) + lin(E2, c1))
        return lin(E, V) + (lin(E, a1) + 2 * lin(E2, b1 + c1) + d1) / 6.0

    rng = np.random.default_rng(seed)
    u0 = 1e-3 * rng.standard_normal(N)
    v0 = 1e-3 * rng.standard_normal(N)
    u0 -= u0.mean(); v0 -= v0.mean()
    V = np.stack([np.fft.fft(u0), np.fft.fft(v0)], axis=1)

    kp = int(np.where(np.round(m) == 1)[0][0])
    nst = int(T / h)
    snaps, snt, amp1, phase1 = [], [], [], []
    for n in range(nst):
        V = step(V)
        t = (n + 1) * h
        if t >= T - tail:
            snaps.append(np.real(np.fft.ifft(V[:, 0])))
            snt.append(t)
            amp1.append(abs(V[kp, 0]))
            phase1.append(np.angle(V[kp, 0]))

    snt = np.array(snt)
    amp = np.array(amp1)
    md = float((amp.max() - amp.min()) / (amp.max() + amp.min() + 1e-30))
    if len(snt) > 2:
        drift = float(np.polyfit(snt, np.unwrap(np.array(phase1)), 1)[0])
    else:
        drift = float("nan")

    return SimulationResult(
        a=a, c=c, g=(g11, g21, g22), delta_minus_dc=delta_minus_dc,
        x=x, snap_times=snt, snapshots=np.array(snaps),
        mode1_amplitude=amp, saturated_amplitude=float(amp[-1]),
        modulation_depth=md, drift_speed=drift,
    )
