# O2Sym prototype

Prototype companion calculator for cubic O(2)-Hopf normal-form coefficients in the two-component Fourier-symbol setting.

A hosted version runs at <https://o2hopf.streamlit.app/> (Streamlit Community Cloud, best-effort availability); for a durable setup, install and run locally as below.

This is a local research tool, not a replacement for the paper's hypotheses or proofs. Some checklist entries are sampled numerical diagnostics.

## Layout

```text
o2sym_tool/
  app.py
  o2sym/
    __init__.py
    core.py
    conservative.py
    coverage.py
    simulation.py
    formatting.py
    checks.py
    classification.py
    safe_eval.py
  examples/
    application_1.py
    bilinear_family.py
  tests/
    test_o2sym.py
    test_bilinear.py
```

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For editable package-style installation:

```bash
python -m pip install -e ".[dev]"
```

## Reproducing the Application

The Application section studies the conservative, O(2)-equivariant bilinear family

```text
G(u, v) = [ g11 (u_x v + u v_x) ;  g21 u u_x + g22 v v_x ]
```

on the fixed Li--Yao linear operator (Hopf at m_c = 1, delta = 1 + a). On this operator the
cubic coefficients zeta, xi are homogeneous quadratic forms in (g11, g21, g22), so their real
parts -- which alone decide the wave-selection region -- are `g^T A g` and `g^T B g`. The
module `o2sym.coverage` builds A, B, scans the parameter plane, and verifies that the full
family realizes all six regions while g11 is indispensable. The module `o2sym.simulation`
integrates the PDE directly with an integrating-factor RK4 scheme whose linear part is advanced
by the exact closed-form 2x2 matrix exponential `expm2` (NumPy only; no SciPy).

The matplotlib-backed figures require the `paper` extra:

```bash
python -m pip install -e ".[paper]"
python -m examples.bilinear_family --outdir .
```

This prints the coverage diagnostics (conservativeness, A and B, the six reachable regions,
the minimal-subset result) and writes `phase_diagram_a05_c1.png` and
`selection_confirmation_a05_c1.png`. Pass `--no-figures` to skip plotting and print the
coverage results only (NumPy alone). The direct-integration test that selects a standing wave
in Region IV and a traveling wave in Region V runs as part of the suite:

```bash
python -m pytest tests/test_bilinear.py -q
```

## JSON conventions

A multi-index

```json
"alpha_u": [2, 1, 0]
```

means

```text
u^2 u_x
```

and

```json
"alpha_v": [0, 2, 1]
```

means

```text
v_x^2 v_xx.
```

Each nonlinear row has the form

```json
{"alpha_u": [1], "alpha_v": [0, 1], "a1": 0.0, "a2": 1.0}
```

which contributes the monomial to `g1` with coefficient `a1` and to `g2` with coefficient `a2`.

## Safety note

Linear coefficient expressions such as `"c**2"` are parsed by a small arithmetic parser, not Python `eval`. It permits numbers, parameter names, parentheses, and the operations `+`, `-`, `*`, `/`, and `**`.
