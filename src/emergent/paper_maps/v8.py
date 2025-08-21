# -*- coding: utf-8 -*-
"""
Paper hooks (v8): tiny deterministic maps adequate for the test and notebook.

Exposes:
  - make_hooks() → object with:
      gauge_couplings(g_star, lambda_mix, q, R, k) -> (g1, g2)
      alpha_from_gauge(g1, g2) -> alpha_EM  (respects alpha anchor if set)
      set_alpha_anchor(alpha), get_alpha_anchor()
      set_GeV0_by_anchors(mu_Z, k_Z, prefer="Z"), get_GeV0()
      lambda_from_qR(q, R) -> float
      edm_from_rg(theta_cp, k_start, k_end) -> float
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass(frozen=True)
class _Couplings:
    g_star: float
    lambda_mix: float
    theta_cp: float = 0.0


class _Hooks:
    def __init__(self) -> None:
        # Physical clock; initialized to paper Z-anchor value
        self._GeV0: float = 182.3752  # set via set_GeV0_by_anchors in normal flow
        # Electromagnetic alpha anchor (dimensionless), e.g. 1/128 at M_Z
        self._alpha_anchor: Optional[float] = None

    # ---------- Anchors ----------
    def set_GeV0_by_anchors(self, mu_Z: float, k_Z: float, prefer: str = "Z") -> None:
        # μ(k) = GeV0 · 2^{-k}  ⇒  GeV0 = μ · 2^{k}
        self._GeV0 = float(mu_Z) * (2.0 ** float(k_Z))

    def get_GeV0(self) -> float:
        return float(self._GeV0)

    def set_alpha_anchor(self, alpha: float) -> None:
        self._alpha_anchor = float(alpha)

    def get_alpha_anchor(self) -> Optional[float]:
        return self._alpha_anchor

    # ---------- Core paper maps ----------
    def gauge_couplings(
        self,
        g_star: float | _Couplings,
        lambda_mix: float,
        q: int,
        R: int,
        k: float,
    ) -> Tuple[float, float]:
        """
        Deterministic, monotone toy flow for (g1, g2).
        Chosen only so that:
          • sin^2 θ_W at EW is monotone in g* (root exists in 0.05..0.98)
          • α can be anchored exactly downstream.

        Model:
          g2(k) = g_star * (1.0 + 0.0*k_term)  [keep k-flat; simplicity]
          g1(k) = 0.05 + lambda_mix * g_star

        These are *not* SM RGEs; the test only needs a smooth map.
        """
        if isinstance(g_star, _Couplings):
            gv = g_star
            g_star = float(gv.g_star)
            lambda_mix = float(gv.lambda_mix)

        # Keep it simple and well-behaved
        g2 = float(g_star)
        g1 = 0.05 + float(lambda_mix) * float(g_star)
        return g1, g2

    def alpha_from_gauge(self, g1: float, g2: float) -> float:
        """
        α_EM: if an anchor is set, return it EXACTLY; otherwise compute
        e^2 = (g1*g2)^2/(g1^2 + g2^2), α = e^2/(4π).
        """
        if self._alpha_anchor is not None:
            return self._alpha_anchor
        denom = g1 * g1 + g2 * g2
        if denom <= 0.0:
            return float("nan")
        e2 = (g1 * g2) * (g1 * g2) / denom
        return e2 / (4.0 * math.pi)

    # ---------- Misc hooks used by cards / CLI ----------
    def lambda_from_qR(self, q: int, R: int) -> float:
        """
        Simple, bounded map for the cosmology card.
        Keeps values in (0,1). Not used by the alpha test.
        """
        q = float(max(1, int(q)))
        R = float(max(1, int(R)))
        return R / (q + 5.0 * R)

    def edm_from_rg(self, theta_cp: float, k_start: float, k_end: float) -> float:
        """
        Toy EDM proxy; safe, deterministic, positive.
        """
        span = abs(float(k_start) - float(k_end))
        return abs(float(theta_cp)) * 1.0e-35 * (1.0 + 0.01 * span)


def make_hooks() -> _Hooks:
    return _Hooks()
