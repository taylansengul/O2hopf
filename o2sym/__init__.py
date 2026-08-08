"""O(2)-Hopf normal-form coefficient calculator."""

from .core import MultiIndex, O2HopfNormalForm, application_1_system, application_1_nonlinearity
from .conservative import ConservativeChecker, Term
from .classification import classify_region
from .checks import hypothesis_checklist
from .safe_eval import safe_eval_expr
from .coverage import (
    bilinear_nonlinearity,
    hopf_params,
    normal_form_coeffs,
    region_for_g,
    extract_quadratic_forms,
    real_parts,
    scan_grid,
    GridScan,
    reachable_regions,
    minimal_subsets,
)
from .simulation import (
    expm2,
    simulate_system2,
    selection_label,
    SimulationResult,
)

__all__ = [
    "MultiIndex", "O2HopfNormalForm", "application_1_system", "application_1_nonlinearity",
    "ConservativeChecker", "Term", "classify_region", "hypothesis_checklist", "safe_eval_expr",
    "bilinear_nonlinearity", "hopf_params", "normal_form_coeffs", "region_for_g",
    "extract_quadratic_forms", "real_parts", "scan_grid", "GridScan",
    "reachable_regions", "minimal_subsets",
    "expm2", "simulate_system2", "selection_label", "SimulationResult",
]
