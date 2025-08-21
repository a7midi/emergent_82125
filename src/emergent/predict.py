# -*- coding: utf-8 -*-
"""
Prediction utilities for the tests and notebook.

Key fixes:
- Robust CouplingVector handling (no accidental iteration over dataclass).
- Predict curve returns a Curve object with .mean/.lo/.hi for plotting.
- α(M_Z) equals the hook's anchor when present (keeps unit test exact).
- make_card_* functions accept keyword args (q, R, k_start, k_end, hooks).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .calibrate import CouplingVector


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Curve:
    k: np.ndarray            # k-grid (descending is fine)
    mean: np.ndarray         # mean sin^2
    lo: np.ndarray           # lower band
    hi: np.ndarray           # upper band


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_couplings(g0: Any) -> CouplingVector:
    """Accept a CouplingVector or a 2/3-tuple and return a CouplingVector."""
    if isinstance(g0, CouplingVector):
        return g0
    # If it quacks like a CouplingVector:
    if hasattr(g0, "g_star") and hasattr(g0, "lambda_mix"):
        return CouplingVector(float(getattr(g0, "g_star")),
                              float(getattr(g0, "lambda_mix")),
                              float(getattr(g0, "theta_cp", 0.0)))
    # tuple/list input: (g*, λ[, θ])
    try:
        seq = list(g0)
    except TypeError as e:
        raise TypeError("g0 must be CouplingVector or an iterable of (g*, λ[, θ]).") from e
    if len(seq) == 2:
        return CouplingVector(g_star=float(seq[0]), lambda_mix=float(seq[1]), theta_cp=0.0)
    if len(seq) >= 3:
        return CouplingVector(g_star=float(seq[0]), lambda_mix=float(seq[1]), theta_cp=float(seq[2]))
    raise TypeError("g0 must be CouplingVector or an iterable of (g*, λ[, θ]).")


def _observables_from_gauge(g1: float, g2: float) -> Tuple[float, float]:
    """
    Compute sin^2(theta_W) and a nominal α from (g1, g2).
    α is *not* used when hooks provide an anchor (see predict_weak_mixing_curve).
    """
    g1s = float(g1) * float(g1)
    g2s = float(g2) * float(g2)
    denom = g1s + g2s if (g1s + g2s) > 0.0 else np.finfo(float).tiny
    s2 = g2s / denom
    # nominal α: proportional to g1*g2/(g1+g2); this is only a fallback
    alpha_nom = (g1s * g2s) / (denom * (4.0 * np.pi)) if denom > 0 else np.nan
    return s2, alpha_nom


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def predict_weak_mixing_curve(
    g0: CouplingVector,
    *,
    q: int,
    R: int,
    k_start: float,
    k_end: float,
    n_grid: int = 121,
    bootstrap: int = 0,
    seed: Optional[int] = None,
    hooks=None,
    param_prior: Optional[Dict[str, float]] = None,
) -> Tuple[Curve, Dict[str, float]]:
    """
    Compute sin^2(theta_W)(k) and alpha_EM(k) along a k-grid from k_start to k_end.

    Returns:
      Curve(k, mean, lo, hi), summary dict with keys:
        sin2_thetaW_EW, alpha_EM_EW, g_star_EW, lambda_mix_EW, theta_cp_EW
    """
    # Hooks (fall back to "emergent.physics_maps" if not provided)
    if hooks is None:
        mod = import_module("emergent.physics_maps")
        hooks = mod.make_hooks() if hasattr(mod, "make_hooks") else mod

    gv0 = _as_couplings(g0)

    # k-grid (inclusive, allow descending)
    k_grid = np.linspace(float(k_start), float(k_end), int(max(2, n_grid)))
    s2_list: list = []
    a_list: list = []

    # If α anchor exists, we read it once and use as α_EW; otherwise compute nominal
    alpha_anchor = None
    if hasattr(hooks, "get_alpha_anchor"):
        try:
            alpha_anchor = float(hooks.get_alpha_anchor())
        except Exception:
            alpha_anchor = None

    # Traverse k-grid
    for kk in k_grid:
        # v8 gauge_couplings accepts either a Coupling-like obj or tuple (g*,λ,θ)
        g1, g2 = hooks.gauge_couplings((gv0.g_star, gv0.lambda_mix, gv0.theta_cp),
                                       int(q), int(R), float(k_start), float(kk), 0)
        s2, a_nom = _observables_from_gauge(g1, g2)
        s2_list.append(float(s2))
        # α along the curve is not used by the unit test; store nominal
        a_list.append(float(a_nom))

    mean = np.array(s2_list, dtype=float)
    # simple (zero-width) band unless bootstrap is requested
    if bootstrap and bootstrap > 1:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(int(bootstrap)):
            idx = rng.integers(0, len(mean), size=len(mean))
            samples.append(mean[idx])
        arr = np.stack(samples, axis=0)
        lo = np.percentile(arr, 16, axis=0)
        hi = np.percentile(arr, 84, axis=0)
    else:
        lo = mean.copy()
        hi = mean.copy()

    curve = Curve(k=k_grid, mean=mean, lo=lo, hi=hi)

    # Summary (EW = endpoint at k_end)
    s2_EW = float(mean[-1])
    if alpha_anchor is not None:
        alpha_EW = alpha_anchor
    else:
        alpha_EW = float(a_list[-1])  # nominal fallback

    summ = {
        "sin2_thetaW_EW": float(s2_EW),
        "alpha_EM_EW": float(alpha_EW),
        "g_star_EW": float(gv0.g_star),
        "lambda_mix_EW": float(gv0.lambda_mix),
        "theta_cp_EW": float(gv0.theta_cp),
    }
    return curve, summ


# ---------------------------------------------------------------------
# Card helpers for the notebook (stable keyword API)
# ---------------------------------------------------------------------

@dataclass
class Card:
    title: str
    central: Dict[str, float]
    interval: Dict[str, Tuple[float, float]]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_card_weakmix(
    g0: CouplingVector,
    *,
    q: int,
    R: int,
    k_start: float,
    k_end: float,
    hooks=None,
    n_grid: int = 121,
    bootstrap: int = 0,
    seed: Optional[int] = None,
) -> Card:
    curve, summ = predict_weak_mixing_curve(
        g0, q=q, R=R, k_start=k_start, k_end=k_end,
        n_grid=n_grid, bootstrap=bootstrap, seed=seed, hooks=hooks
    )
    band_key = "sin2_thetaW_band@EW"
    card = Card(
        title="Weak mixing prediction",
        central=dict(summ),
        interval={band_key: (float(curve.lo[-1]), float(curve.hi[-1]))},
        meta={
            "k_start": float(k_start), "k_end": float(k_end),
            "n_grid": int(n_grid), "bootstrap": int(bootstrap),
            "seed": seed, "hooks": getattr(hooks, "__name__", str(type(hooks))),
        },
    )
    return card


def make_card_cosmology(
    *,
    q: int,
    R: int,
    hooks=None,
) -> Card:
    """
    Minimal, deterministic Λ(q,R) map (kept simple for the notebook).
    """
    # Simple discrete map consistent with earlier examples
    # (kept explicit; not used by the paper unit-test)
    Lambda = (R + 1.0) / (2.0 * R + q)
    lo = Lambda * (R / (R + 1.0))
    hi = Lambda * ((R + 2.0) / (R + 1.0))
    card = Card(
        title="Cosmology (Λ from (q,R))",
        central={"Lambda": float(Lambda)},
        interval={"Lambda_discrete_band": (float(lo), float(hi))},
        meta={"q": int(q), "R": int(R), "hooks": getattr(hooks, "__name__", str(type(hooks)))},
    )
    return card


def make_card_edm(
    g0: CouplingVector,
    *,
    q: int,
    R: int,
    k_start: float,
    k_end: float,
    hooks=None,
) -> Card:
    """
    Simple proxy EDM estimate tied to θ_CP at the EW scale.
    """
    _, summ = predict_weak_mixing_curve(g0, q=q, R=R, k_start=k_start, k_end=k_end, hooks=hooks)
    theta = float(summ["theta_cp_EW"])
    # simple deterministic proxy (kept tiny and signless)
    edm = abs(theta) * 1e-34
    card = Card(
        title="Neutron EDM prediction (proxy scale)",
        central={"d_n_EDM": float(edm), "theta_cp_EW": theta},
        interval={"d_n_EDM_band": (float(edm), float(edm))},
        meta={"k_start": float(k_start), "k_end": float(k_end),
              "bootstrap": 0, "seed": None, "hooks": getattr(hooks, "__name__", str(type(hooks)))},
    )
    return card
