from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution via `python examples/application_1.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o2sym import application_1_nonlinearity, application_1_system


if __name__ == "__main__":
    system = application_1_system()
    params = {"delta": 1.0, "c": 1.0}
    for eta in [0.0, 1.0]:
        zeta, xi = system.compute_normal_form(1, params, application_1_nonlinearity(eta))
        print(f"eta={eta}: zeta={zeta}, xi={xi}")
