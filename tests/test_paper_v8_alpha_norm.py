# tests/test_paper_v8_alpha_norm.py
import math
from emergent.calibrate import CouplingVector, calibrate_two_anchors
from emergent.physics_maps import make_hooks_from_module
from emergent.predict import predict_weak_mixing_curve

def test_alpha_EW_matches_anchor_after_calibration():
    hooks = make_hooks_from_module("emergent.paper_maps.v8")
    q, R = 13, 2
    k_start, k_end = 120.0, 1.0
    mu_Z = 91.1876
    SIN2_TARGET = 0.23122
    ALPHA_TARGET = 1.0/128.0

    g_template = CouplingVector(g_star=0.4, lambda_mix=0.5, theta_cp=0.2)

    cal = calibrate_two_anchors(
        g_template, q=q, R=R, k_start=k_start, k_end=k_end,
        target_sin2_EW=SIN2_TARGET, mu_EW_GeV=mu_Z, hooks=hooks,
        target_alpha_EW=ALPHA_TARGET
    )

    g0_cal = CouplingVector(cal.get("g_star_cal", g_template.g_star),
                            cal.get("lambda_mix_cal", g_template.lambda_mix),
                            g_template.theta_cp)
    _, summ = predict_weak_mixing_curve(g0_cal, q=q, R=R, k_start=k_start, k_end=k_end, hooks=hooks)

    # α must match essentially exactly; sin² within a small tolerance
    assert abs(float(summ["alpha_EM_EW"]) - ALPHA_TARGET) < 1e-10
    assert abs(float(summ["sin2_thetaW_EW"]) - SIN2_TARGET) < 1e-2
