# O2Sym prototype

Prototype companion calculator for cubic O(2)-Hopf normal-form coefficients in the two-component Fourier-symbol setting.

This is a local research tool, not a replacement for the paper's hypotheses or proofs. Some checklist entries are sampled numerical diagnostics.

## Layout

```text
o2sym_tool/
  app.py
  o2sym/
    __init__.py
    core.py
    conservative.py
    formatting.py
    checks.py
    classification.py
    safe_eval.py
  examples/
    application_1.py
  tests/
    test_o2sym.py
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
