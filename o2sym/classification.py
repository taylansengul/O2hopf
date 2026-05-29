from __future__ import annotations

from typing import Any, Dict


def classify_region(zeta: complex, xi: complex, beta_sign: str = "+", tol: float = 1e-8) -> Dict[str, Any]:
    z = float(complex(zeta).real)
    x = float(complex(xi).real)
    zp = z + x
    zm = z - x
    if min(abs(z), abs(zp), abs(zm)) <= tol:
        return {
            "region": "Degenerate/boundary",
            "conditions": f"Re zeta={z:.6g}, Re xi={x:.6g}, Re(zeta+xi)={zp:.6g}, Re(zeta-xi)={zm:.6g}",
            "tw": "not classified",
            "sw": "not classified",
            "transition": "degenerate case; outside Regions I--VI",
        }
    if zp > 0 and zm > 0:
        region, tw, sw = "I", "saddle", "unstable"
    elif z > 0 and zm < 0:
        region, tw, sw = "II", "unstable", "saddle"
    elif z < 0 and zp > 0:
        region = "III"
        tw, sw = ("saddle", "---") if beta_sign == "+" else ("---", "saddle")
    elif zp < 0 and zm < 0:
        region, tw, sw = "IV", "saddle", "stable"
    elif z < 0 and zm > 0:
        region, tw, sw = "V", "stable", "saddle"
    elif z > 0 and zp < 0:
        region = "VI"
        tw, sw = ("---", "saddle") if beta_sign == "+" else ("saddle", "---")
    else:
        region, tw, sw = "Unclassified", "not classified", "not classified"
    transition = "Type-I/continuous" if (z < 0 and zp < 0) else "Type-II/jump or non-continuous"
    return {
        "region": region,
        "conditions": f"Re zeta={z:.6g}, Re xi={x:.6g}, Re(zeta+xi)={zp:.6g}, Re(zeta-xi)={zm:.6g}",
        "tw": tw,
        "sw": sw,
        "transition": transition,
    }
