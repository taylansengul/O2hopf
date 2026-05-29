from __future__ import annotations

import pytest

from o2sym import ConservativeChecker, O2HopfNormalForm, application_1_nonlinearity, application_1_system, classify_region
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
