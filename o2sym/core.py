from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple, Union, Any

import numpy as np

Coeff = Union[float, complex, Callable[[Mapping[str, float]], Union[float, complex]]]
CoeffDict = Dict[Tuple[int, int], Coeff]
Nonlinearity = Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray]


@dataclass(frozen=True)
class MultiIndex:
    """Finite multi-index alpha=(alpha_0, alpha_1, ...)."""

    coeffs: Tuple[int, ...]

    def __post_init__(self) -> None:
        coeffs = tuple(int(c) for c in self.coeffs)
        while coeffs and coeffs[-1] == 0:
            coeffs = coeffs[:-1]
        object.__setattr__(self, "coeffs", coeffs)

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, index: int) -> int:
        return self.coeffs[index] if index < len(self.coeffs) else 0

    def norm(self) -> int:
        """Total degree |alpha|."""
        return sum(self.coeffs)

    def m_e(self) -> int:
        """Even-order count sum alpha_{2j}."""
        return sum(self.coeffs[i] for i in range(0, len(self.coeffs), 2))

    def m_o(self) -> int:
        """Odd-order count sum alpha_{2j+1}."""
        return sum(self.coeffs[i] for i in range(1, len(self.coeffs), 2))

    def m_t(self) -> int:
        """Total derivative order sum j alpha_j."""
        return sum(i * self.coeffs[i] for i in range(len(self.coeffs)))

    def weight(self) -> complex:
        """Weight factor sum (-2)^p alpha_p used in the quadratic interaction formulas."""
        return sum((-2) ** p * self.coeffs[p] for p in range(len(self.coeffs)))


def _eval_coeff(value: Coeff, params: Mapping[str, float]) -> complex:
    return complex(value(params) if callable(value) else value)


class O2HopfNormalForm:
    """Compute cubic O(2)-Hopf normal-form coefficients for the 2x2 symbol setup.

    The implementation follows the simple-spectrum/eigenbasis formulas. It is a
    companion calculator, not a replacement for the hypotheses in the paper.
    """

    def __init__(self, b_coeffs: CoeffDict, m_L: int, tolerance: float = 1e-10):
        self.b_coeffs = dict(b_coeffs)
        self.m_L = int(m_L)
        self.tolerance = float(tolerance)

    def M_matrix(self, m: int, params: Mapping[str, float]) -> np.ndarray:
        """Return the Fourier symbol M_m."""
        M = np.zeros((2, 2), dtype=complex)
        for k in range(self.m_L + 1):
            if (1, 2 * k) in self.b_coeffs:
                M[0, 0] += _eval_coeff(self.b_coeffs[(1, 2 * k)], params) * (1j * m) ** (2 * k)
            if (2, 2 * k) in self.b_coeffs:
                M[1, 1] += _eval_coeff(self.b_coeffs[(2, 2 * k)], params) * (1j * m) ** (2 * k)
            if (1, 2 * k + 1) in self.b_coeffs:
                M[0, 1] += _eval_coeff(self.b_coeffs[(1, 2 * k + 1)], params) * (1j * m) ** (2 * k + 1)
            if (2, 2 * k + 1) in self.b_coeffs:
                M[1, 0] += _eval_coeff(self.b_coeffs[(2, 2 * k + 1)], params) * (1j * m) ** (2 * k + 1)
        return M

    def eigendata(self, m: int, params: Mapping[str, float]) -> Tuple[complex, complex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return beta_1, beta_2, q_1, q_2, q*_1, q*_2 at mode m.

        Uses the explicit eigenvector convention from the draft. A warning-level
        error is raised when the simple formula is close to singular.
        """
        M = self.M_matrix(m, params)
        tr_M = np.trace(M)
        det_M = np.linalg.det(M)
        discriminant = tr_M ** 2 - 4 * det_M
        beta_1 = (tr_M + np.sqrt(discriminant)) / 2
        beta_2 = (tr_M - np.sqrt(discriminant)) / 2

        if abs(M[0, 1]) > self.tolerance:
            if abs(beta_1 - M[0, 0]) < self.tolerance or abs(beta_2 - M[0, 0]) < self.tolerance:
                raise ValueError("Eigenvector formula is near singular. Use a matrix-resolvent/eigensolver version.")
            q_1 = np.array([M[0, 1] / (beta_1 - M[0, 0]), 1.0], dtype=complex)
            q_2 = np.array([M[0, 1] / (beta_2 - M[0, 0]), 1.0], dtype=complex)
        else:
            q_1 = np.array([1.0, 0.0], dtype=complex)
            q_2 = np.array([0.0, 1.0], dtype=complex)

        if abs(M[1, 0]) > self.tolerance:
            beta_minus_m_1 = np.conj(beta_1)
            beta_minus_m_2 = np.conj(beta_2)
            if abs(beta_minus_m_1 - M[0, 0]) < self.tolerance or abs(beta_minus_m_2 - M[0, 0]) < self.tolerance:
                raise ValueError("Adjoint eigenvector formula is near singular.")
            q_star_1_unnorm = np.array([-M[1, 0] / (beta_minus_m_1 - M[0, 0]), 1.0], dtype=complex)
            q_star_2_unnorm = np.array([-M[1, 0] / (beta_minus_m_2 - M[0, 0]), 1.0], dtype=complex)
            d1 = np.dot(q_1, np.conj(q_star_1_unnorm))
            d2 = np.dot(q_2, np.conj(q_star_2_unnorm))
            if abs(d1) < self.tolerance or abs(d2) < self.tolerance:
                raise ValueError("Biorthogonal normalization is near singular.")
            q_star_1 = q_star_1_unnorm / (2 * np.pi * d1)
            q_star_2 = q_star_2_unnorm / (2 * np.pi * d2)
        else:
            q_star_1 = np.array([1.0 / (2 * np.pi * q_1[0]), 0.0], dtype=complex)
            q_star_2 = np.array([0.0, 1.0 / (2 * np.pi * q_2[1])], dtype=complex)

        return beta_1, beta_2, q_1, q_2, q_star_1, q_star_2

    def sigma_coefficient(self, alpha1: MultiIndex, alpha2: MultiIndex, m_c: int, q_star: np.ndarray, a_alpha: np.ndarray) -> complex:
        inner_prod = np.dot(np.asarray(a_alpha, dtype=complex), np.conj(q_star))
        phase = (1j * m_c) ** (alpha1.m_t() + alpha2.m_t())
        return 2 * np.pi * inner_prod * phase

    def compute_s_coefficients(self, m_c: int, q_mc: Tuple[np.ndarray, np.ndarray], q_star_mc: Tuple[np.ndarray, np.ndarray], nonlinearity: Nonlinearity) -> np.ndarray:
        # Only the critical adjoint q_1^* enters the cubic normal-form projection.
        # Earlier debug versions also computed the q_2^* row, but that value is not
        # used in zeta or xi and can suggest a false dependence on the stable adjoint.
        q_star_1, _q_star_2 = q_star_mc
        s = np.zeros(6, dtype=complex)
        for (alpha1_tuple, alpha2_tuple), a_alpha in nonlinearity.items():
            alpha1 = MultiIndex(alpha1_tuple)
            alpha2 = MultiIndex(alpha2_tuple)
            if alpha1.norm() + alpha2.norm() != 3:
                continue
            sigma = self.sigma_coefficient(alpha1, alpha2, m_c, q_star_1, a_alpha)
            if alpha1.norm() == 3 and alpha2.norm() == 0:
                s[0] += sigma * (alpha1.m_e() - alpha1.m_o())
            elif alpha1.norm() == 1 and alpha2.norm() == 2:
                s[1] += sigma * (alpha2.m_e() - alpha2.m_o())
                s[4] += sigma * (alpha1.m_e() - alpha1.m_o())
            elif alpha1.norm() == 2 and alpha2.norm() == 1:
                s[2] += sigma * (alpha1.m_e() - alpha1.m_o())
                s[5] += sigma * (alpha2.m_e() - alpha2.m_o())
            elif alpha1.norm() == 0 and alpha2.norm() == 3:
                s[3] += sigma * (alpha2.m_e() - alpha2.m_o())
        return s

    def compute_s_hat(self, s: np.ndarray, q_1: np.ndarray, q_2: np.ndarray) -> Tuple[complex, complex]:
        u_1, v_1 = q_1
        u_2, v_2 = q_2
        s_hat_11 = (
            s[0] * u_1 * abs(u_1) ** 2
            + s[1] * u_1 * abs(v_1) ** 2
            + s[2] * v_1 * abs(u_1) ** 2
            + s[3] * v_1 * abs(v_1) ** 2
            + s[4] * np.conj(u_1) * v_1 ** 2
            + s[5] * u_1 ** 2 * np.conj(v_1)
        )
        s_hat_12 = (
            2 * s[0] * u_1 * abs(u_2) ** 2
            + s[1] * np.conj(v_2) * (u_1 * v_2 + u_2 * v_1)
            + s[2] * np.conj(u_2) * (v_1 * u_2 + v_2 * u_1)
            + 2 * s[3] * v_1 * abs(v_2) ** 2
            + 2 * s[4] * np.conj(u_2) * v_1 * v_2
            + 2 * s[5] * u_1 * u_2 * np.conj(v_2)
        )
        return s_hat_11, s_hat_12

    def compute_c_hat(
        self,
        m_c: int,
        params: Mapping[str, float],
        q_mc: Tuple[np.ndarray, np.ndarray],
        q_star_mc: Tuple[np.ndarray, np.ndarray],
        q_2mc: Tuple[np.ndarray, np.ndarray],
        q_star_2mc: Tuple[np.ndarray, np.ndarray],
        beta_mc: Tuple[complex, complex],
        beta_2mc: Tuple[complex, complex],
        nonlinearity: Nonlinearity,
    ) -> Tuple[complex, complex]:
        q_1, q_2 = q_mc
        q_star_1, _q_star_2 = q_star_mc
        q_2mc_1, q_2mc_2 = q_2mc
        q_star_2mc_1, q_star_2mc_2 = q_star_2mc
        beta_1, _beta_2 = beta_mc
        beta_2mc_1, beta_2mc_2 = beta_2mc
        u_1, v_1 = q_1
        u_2, v_2 = q_2

        C_U_1_1 = 0.0j
        C_U_2_1 = 0.0j
        C_V_1_1 = 0.0j
        C_V_2_1 = 0.0j

        for (alpha1_tuple, alpha2_tuple), a_alpha in nonlinearity.items():
            alpha1 = MultiIndex(alpha1_tuple)
            alpha2 = MultiIndex(alpha2_tuple)
            if alpha1.norm() + alpha2.norm() != 2:
                continue
            sigma = self.sigma_coefficient(alpha1, alpha2, m_c, q_star_1, a_alpha)
            sign_factor = (-1) ** (alpha1.m_o() + alpha2.m_o())
            A_alpha_1 = sign_factor * sigma * alpha1.weight()
            A_alpha_2 = sign_factor * sigma * alpha2.weight()
            if alpha1.norm() == 2 and alpha2.norm() == 0:
                C_U_1_1 += A_alpha_1
            elif alpha1.norm() == 1 and alpha2.norm() == 1:
                C_U_2_1 += A_alpha_1
                C_V_1_1 += A_alpha_2
            elif alpha1.norm() == 0 and alpha2.norm() == 2:
                C_V_2_1 += A_alpha_2

        Phi_AA = np.zeros(2, dtype=complex)
        Phi_AB = np.zeros(2, dtype=complex)
        for k, (beta_2mc_k, q_star_2mc_k) in enumerate(((beta_2mc_1, q_star_2mc_1), (beta_2mc_2, q_star_2mc_2))):
            sigma_sum_AA = 0.0j
            sigma_sum_AB = 0.0j
            for (alpha1_tuple, alpha2_tuple), a_alpha in nonlinearity.items():
                alpha1 = MultiIndex(alpha1_tuple)
                alpha2 = MultiIndex(alpha2_tuple)
                if alpha1.norm() + alpha2.norm() != 2:
                    continue
                sigma_2mc = self.sigma_coefficient(alpha1, alpha2, m_c, q_star_2mc_k, a_alpha)
                n1, n2 = alpha1.norm(), alpha2.norm()
                sigma_sum_AA += sigma_2mc * u_1 ** n1 * v_1 ** n2
                Q_AB = 0.0j
                if n1 >= 1:
                    Q_AB += n1 * u_1 * u_2 ** (n1 - 1) * v_2 ** n2
                if n2 >= 1:
                    Q_AB += n2 * u_2 ** n1 * v_1 * v_2 ** (n2 - 1)
                sigma_sum_AB += sigma_2mc * Q_AB
            if abs(2 * beta_1 - beta_2mc_k) < self.tolerance or abs(beta_2mc_k) < self.tolerance:
                raise ValueError("Small denominator in center-manifold quadratic correction.")
            Phi_AA[k] = sigma_sum_AA / (2 * beta_1 - beta_2mc_k)
            Phi_AB[k] = sigma_sum_AB / (-beta_2mc_k)

        c_hat_11 = 0.0j
        c_hat_12 = 0.0j
        for k, q_2mc_k in enumerate((q_2mc_1, q_2mc_2)):
            u_2mc_k, v_2mc_k = q_2mc_k
            c_hat_11 += Phi_AA[k] * (
                (C_U_1_1 * u_2mc_k + C_V_1_1 * v_2mc_k) * np.conj(u_1)
                + (C_U_2_1 * u_2mc_k + C_V_2_1 * v_2mc_k) * np.conj(v_1)
            )
            c_hat_12 += Phi_AB[k] * (
                (C_U_1_1 * u_2mc_k + C_V_1_1 * v_2mc_k) * np.conj(u_2)
                + (C_U_2_1 * u_2mc_k + C_V_2_1 * v_2mc_k) * np.conj(v_2)
            )
        return c_hat_11, c_hat_12

    def compute_normal_form(self, m_c: int, params: Mapping[str, float], nonlinearity: Nonlinearity) -> Tuple[complex, complex]:
        beta_1, beta_2, q_1, q_2, q_star_1, q_star_2 = self.eigendata(m_c, params)
        beta_2mc_1, beta_2mc_2, q_2mc_1, q_2mc_2, q_star_2mc_1, q_star_2mc_2 = self.eigendata(2 * m_c, params)
        s = self.compute_s_coefficients(m_c, (q_1, q_2), (q_star_1, q_star_2), nonlinearity)
        s_hat_11, s_hat_12 = self.compute_s_hat(s, q_1, q_2)
        c_hat_11, c_hat_12 = self.compute_c_hat(
            m_c,
            params,
            (q_1, q_2),
            (q_star_1, q_star_2),
            (q_2mc_1, q_2mc_2),
            (q_star_2mc_1, q_star_2mc_2),
            (beta_1, beta_2),
            (beta_2mc_1, beta_2mc_2),
            nonlinearity,
        )
        return s_hat_11 + c_hat_11, s_hat_12 + c_hat_12

    def diagnostic_data(self, m_c: int, params: Mapping[str, float]) -> Dict[str, Any]:
        beta_1, beta_2, q_1, q_2, q_star_1, q_star_2 = self.eigendata(m_c, params)
        beta_2mc_1, beta_2mc_2, q_2mc_1, q_2mc_2, q_star_2mc_1, q_star_2mc_2 = self.eigendata(2 * m_c, params)
        return {
            "M_mc": self.M_matrix(m_c, params),
            "M_2mc": self.M_matrix(2 * m_c, params),
            "beta_mc": (beta_1, beta_2),
            "beta_2mc": (beta_2mc_1, beta_2mc_2),
            "q_mc": (q_1, q_2),
            "q_star_mc": (q_star_1, q_star_2),
            "q_2mc": (q_2mc_1, q_2mc_2),
            "q_star_2mc": (q_star_2mc_1, q_star_2mc_2),
        }


def application_1_system() -> O2HopfNormalForm:
    b_coeffs: CoeffDict = {
        (1, 0): 0.0,
        (1, 1): 1.0,
        (2, 1): lambda p: p["c"] ** 2,
        (2, 0): 0.0,
        (2, 2): lambda p: -p["delta"],
        (2, 4): -1.0,
    }
    return O2HopfNormalForm(b_coeffs, m_L=2)


def application_1_nonlinearity(eta: float = 0.0) -> Nonlinearity:
    nonlinearity: Nonlinearity = {((1, 1), (0,)): np.array([0.0, 1.0], dtype=complex)}
    if abs(eta) > 1e-14:
        nonlinearity[((2, 1), (0,))] = np.array([0.0, eta], dtype=complex)
    return nonlinearity
