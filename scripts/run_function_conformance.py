#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_array_processing.regression.matlab import detect_matlab, probe_matlab_cli, run_matlab_batch
from spherical_array_processing.repro import politis as po
from spherical_array_processing.repro import rafaely as rf


@dataclass(frozen=True)
class FunctionCase:
    case_id: str
    api_names: tuple[str, ...]
    matlab_body: str
    python_eval: Callable[[], Any]
    rtol: float = 1e-6
    atol: float = 1e-8


EXPECTED_DIFFERENCE_CASES: dict[str, str] = {
    "rafaely_uniform_sampling": "Python keeps Fibonacci near-uniform sampling instead of MATLAB built-in point table.",
    "rafaely_platonic_solid": "Python returns 0-based face indices; MATLAB outputs 1-based indices.",
    "politis_extractAxisCoeffs": "Python preserves multi-column semantics; MATLAB behavior is historically single-column oriented.",
}


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("\\", "/").replace("'", "''")


def _mat_scalar(x: Any) -> str:
    z = complex(x)
    r = float(np.real(z))
    i = float(np.imag(z))
    if abs(i) < 1e-18:
        return f"{r:.17g}"
    return f"({r:.17g}{i:+.17g}i)"


def _mat_reshape(var_name: str, arr: np.ndarray) -> str:
    flat = arr.flatten(order="F")
    vals = " ".join(_mat_scalar(v) for v in flat)
    dims = " ".join(str(d) for d in arr.shape)
    return f"{var_name} = reshape([{vals}], [{dims}]);"


def _flatten_components(value: Any, prefix: str = "out") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            out.update(_flatten_components(item, f"{prefix}[{idx}]"))
        return out
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.shape == ():
            return _flatten_components(value.item(), prefix)
        for idx, item in np.ndenumerate(value):
            idx_label = ",".join(str(i) for i in idx)
            out.update(_flatten_components(item, f"{prefix}[{idx_label}]"))
        return out
    if isinstance(value, np.ndarray):
        out[prefix] = value
        return out
    out[prefix] = np.asarray(value)
    return out


def _safe_load_mat(path: Path) -> Any:
    try:
        payload = loadmat(path, simplify_cells=True)
    except TypeError:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "out" not in payload:
        raise KeyError(f"'out' not found in {path}")
    return payload["out"]


def _compare_values(py_val: Any, ml_val: Any, rtol: float, atol: float) -> dict[str, Any]:
    py_comp = _flatten_components(py_val)
    ml_comp = _flatten_components(ml_val)
    keys = sorted(set(py_comp) | set(ml_comp))
    details: list[dict[str, Any]] = []
    ok = True
    max_abs = 0.0
    max_rel = 0.0
    def _vector_like(shape: tuple[int, ...]) -> bool:
        if len(shape) <= 1:
            return True
        return len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)

    for key in keys:
        if key not in py_comp or key not in ml_comp:
            ok = False
            details.append(
                {
                    "component": key,
                    "ok": False,
                    "reason": "component_missing",
                    "python_present": key in py_comp,
                    "matlab_present": key in ml_comp,
                }
            )
            continue
        a = np.asarray(py_comp[key])
        b = np.asarray(ml_comp[key])
        if a.shape != b.shape:
            if (
                np.issubdtype(a.dtype, np.number)
                and np.issubdtype(b.dtype, np.number)
                and a.size == b.size
                and _vector_like(a.shape)
                and _vector_like(b.shape)
            ):
                a = a.reshape(-1)
                b = b.reshape(-1)
            else:
                ok = False
                details.append(
                    {
                        "component": key,
                        "ok": False,
                        "reason": "shape_mismatch",
                        "python_shape": list(a.shape),
                        "matlab_shape": list(b.shape),
                    }
                )
                continue
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            aa = a.astype(np.complex128)
            bb = b.astype(np.complex128)
            diff = np.abs(aa - bb)
            comp_abs = float(np.nanmax(diff)) if diff.size else 0.0
            denom = np.maximum(np.abs(bb), atol)
            comp_rel = float(np.nanmax(diff / denom)) if diff.size else 0.0
            comp_ok = bool(np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True))
            max_abs = max(max_abs, comp_abs)
            max_rel = max(max_rel, comp_rel)
            ok = ok and comp_ok
            details.append(
                {
                    "component": key,
                    "ok": comp_ok,
                    "reason": "numeric_compare",
                    "max_abs_diff": comp_abs,
                    "max_rel_diff": comp_rel,
                }
            )
        else:
            comp_ok = bool(np.array_equal(a, b))
            ok = ok and comp_ok
            details.append(
                {
                    "component": key,
                    "ok": comp_ok,
                    "reason": "exact_compare",
                }
            )
    return {
        "ok": ok,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "components_compared": len(details),
        "details": details,
    }


def _is_missing_dependency_error(err_text: str) -> bool:
    text = err_text.lower()
    markers = [
        "undefined function",
        "undefined variable",
        "unrecognized function or variable",
        "not enough input arguments",
        "未定义与",
        "未识别的函数",
        "未定义函数",
    ]
    return any(marker in text for marker in markers)


def _shared_test_data() -> dict[str, np.ndarray]:
    grid_dirs = np.array(
        [
            [0.0, 0.0],
            [math.pi / 2, 0.0],
            [math.pi, 0.0],
            [-math.pi / 2, 0.0],
            [0.0, math.pi / 4],
            [math.pi / 2, math.pi / 4],
            [math.pi, -math.pi / 4],
            [-math.pi / 2, -math.pi / 4],
        ],
        dtype=float,
    )
    mic_dirs = np.array(
        [
            [0.0, 0.0],
            [math.pi / 2, 0.0],
            [math.pi, 0.0],
            [-math.pi / 2, 0.0],
            [0.0, math.pi / 3],
            [0.0, -math.pi / 3],
        ],
        dtype=float,
    )
    mic_dirs_cond = np.zeros((12, 2), dtype=float)
    for i in range(mic_dirs_cond.shape[0]):
        a = -math.pi + 2 * math.pi * i / mic_dirs_cond.shape[0]
        mic_dirs_cond[i, 0] = a
        mic_dirs_cond[i, 1] = 0.5 * math.sin(1.7 * a)
    h_array = np.zeros((5, 3, grid_dirs.shape[0]), dtype=np.complex128)
    for b in range(h_array.shape[0]):
        for m in range(h_array.shape[1]):
            for g in range(h_array.shape[2]):
                h_array[b, m, g] = (0.7 + 0.13 * b + 0.04 * m + 0.01 * g) + 1j * (
                    0.2 * b - 0.05 * m + 0.03 * g
                )
    m_mic2sh = np.zeros((4, 3, 5), dtype=np.complex128)
    for q in range(m_mic2sh.shape[0]):
        for m in range(m_mic2sh.shape[1]):
            for b in range(m_mic2sh.shape[2]):
                m_mic2sh[q, m, b] = (0.4 + 0.11 * q + 0.06 * m + 0.02 * b) + 1j * (0.03 * q - 0.01 * b)
    m_dfc = np.zeros((3, 3, 5), dtype=np.complex128)
    for b in range(5):
        base = np.array(
            [
                [1.0 + 0.1 * b, 0.2 + 0.05 * b, -0.1 + 0.02 * b],
                [0.2 + 0.05 * b, 0.9 + 0.08 * b, 0.15 + 0.01 * b],
                [-0.1 + 0.02 * b, 0.15 + 0.01 * b, 0.8 + 0.07 * b],
            ],
            dtype=float,
        )
        m_dfc[:, :, b] = base + 0j
    sh_cov = np.zeros((9, 9), dtype=np.complex128)
    for i in range(9):
        for j in range(3):
            v = (i + 1 + 2 * (j + 1)) / 10.0 + 1j * ((i + 1) - (j + 1)) / 20.0
            sh_cov[i, i] += v * np.conj(v)
        for k in range(i):
            sh_cov[i, k] = (0.02 + 0.01j) * (i + k + 1)
            sh_cov[k, i] = np.conj(sh_cov[i, k])
    us = np.arange(1, 19, dtype=float).reshape((9, 2), order="F") / 10.0
    us = us + 1j * (np.arange(19, 37, dtype=float).reshape((9, 2), order="F") / 50.0)
    a_irls = np.array(
        [
            [1.0, 0.2, 0.1, -0.3, 0.5],
            [0.0, 0.9, 0.4, 0.2, -0.1],
            [0.3, -0.2, 1.1, 0.1, 0.6],
        ],
        dtype=float,
    )
    y_irls = np.array(
        [
            [0.5, 0.1, 0.2, -0.2],
            [0.0, 0.4, -0.1, 0.3],
            [0.2, -0.3, 0.6, 0.1],
        ],
        dtype=float,
    )
    i_vecs = np.array(
        [
            [0.6, 0.1, 0.2],
            [0.5, 0.2, 0.1],
            [0.4, -0.2, 0.2],
            [0.3, 0.1, -0.1],
            [0.2, -0.1, 0.0],
        ],
        dtype=float,
    )
    pv_cov = np.array(
        [
            [1.5, 0.1, 0.05, -0.03],
            [0.1, 0.9, 0.02, 0.01],
            [0.05, 0.02, 0.8, -0.02],
            [-0.03, 0.01, -0.02, 0.7],
        ],
        dtype=float,
    )
    sh_sig = np.array(
        [
            [0.4, 0.2, 0.1],
            [0.1, -0.1, 0.3],
            [0.2, 0.0, 0.2],
            [0.3, 0.1, -0.2],
        ],
        dtype=float,
    )
    a_grid = np.array(
        [
            [1.0, 0.2, 0.1, 0.0, 0.1, -0.1, 0.3, 0.2],
            [0.1, 0.9, 0.2, 0.3, -0.2, 0.1, 0.0, 0.2],
            [0.2, 0.1, 0.8, -0.1, 0.2, 0.3, 0.1, -0.2],
            [0.0, 0.2, -0.1, 0.7, 0.1, 0.2, -0.3, 0.4],
        ],
        dtype=float,
    )
    intensity_xyz = np.array(
        [
            [0.7, 0.1, 0.2],
            [0.6, 0.2, 0.1],
            [0.1, 0.6, 0.2],
            [-0.2, 0.5, 0.3],
            [0.3, -0.4, 0.2],
        ],
        dtype=float,
    )
    # Dedicated HD dataset for regLSHD case to avoid degenerate dimensions.
    grid_dirs_hd = np.zeros((36, 2), dtype=float)
    for i in range(grid_dirs_hd.shape[0]):
        grid_dirs_hd[i, 0] = -math.pi + 2 * math.pi * i / grid_dirs_hd.shape[0]
        grid_dirs_hd[i, 1] = 0.35 * math.sin(2 * math.pi * i / grid_dirs_hd.shape[0])
    h_array_hd = np.zeros((5, 9, grid_dirs_hd.shape[0]), dtype=np.complex128)
    for b in range(h_array_hd.shape[0]):
        for m in range(h_array_hd.shape[1]):
            for g in range(h_array_hd.shape[2]):
                h_array_hd[b, m, g] = (0.55 + 0.07 * b + 0.03 * m + 0.005 * g) + 1j * (
                    0.11 * b - 0.02 * m + 0.01 * g
                )
    # Deterministic conversion tensor for beamWeightsVelocityPatterns case.
    a_xyz = np.zeros((16, 9, 3), dtype=np.complex128)
    phi_x = sh_cov + 0.2 * np.eye(9)
    phi_n = 0.3 * np.eye(9)
    return {
        "grid_dirs": grid_dirs,
        "mic_dirs": mic_dirs,
        "mic_dirs_cond": mic_dirs_cond,
        "h_array": h_array,
        "m_mic2sh": m_mic2sh,
        "m_dfc": m_dfc,
        "sh_cov": sh_cov,
        "us": us,
        "a_irls": a_irls,
        "y_irls": y_irls,
        "i_vecs": i_vecs,
        "pv_cov": pv_cov,
        "sh_sig": sh_sig,
        "a_grid": a_grid,
        "intensity_xyz": intensity_xyz,
        "grid_dirs_hd": grid_dirs_hd,
        "h_array_hd": h_array_hd,
        "a_xyz": a_xyz,
        "phi_x": phi_x,
        "phi_n": phi_n,
    }


def _build_cases() -> list[FunctionCase]:
    d = _shared_test_data()

    dirs_lit = _mat_reshape("grid_dirs", d["grid_dirs"])
    mic_dirs_lit = _mat_reshape("mic_dirs", d["mic_dirs"])
    mic_dirs_cond_lit = _mat_reshape("mic_dirs_cond", d["mic_dirs_cond"])
    h_array_lit = _mat_reshape("H_array", d["h_array"])
    m_mic2sh_lit = _mat_reshape("M_mic2sh", d["m_mic2sh"])
    m_dfc_lit = _mat_reshape("M_dfc", d["m_dfc"])
    sh_cov_lit = _mat_reshape("SHcov", d["sh_cov"])
    us_lit = _mat_reshape("Us", d["us"])
    a_irls_lit = _mat_reshape("A", d["a_irls"])
    y_irls_lit = _mat_reshape("Y", d["y_irls"])
    i_vecs_lit = _mat_reshape("i_vecs", d["i_vecs"])
    pv_cov_lit = _mat_reshape("pvCOV", d["pv_cov"])
    sh_sig_lit = _mat_reshape("shsig", d["sh_sig"])
    a_grid_lit = _mat_reshape("A_grid", d["a_grid"])
    intensity_lit = _mat_reshape("i_xyz", d["intensity_xyz"])
    dirs_hd_lit = _mat_reshape("grid_dirs_hd", d["grid_dirs_hd"])
    h_array_hd_lit = _mat_reshape("H_array_hd", d["h_array_hd"])
    a_xyz_lit = _mat_reshape("A_xyz", d["a_xyz"])
    phi_x_lit = _mat_reshape("Phi_x", d["phi_x"])
    phi_n_lit = _mat_reshape("Phi_n", d["phi_n"])

    cases: list[FunctionCase] = [
        FunctionCase(
            case_id="rafaely_sh2",
            api_names=("sh2",),
            matlab_body="out = sh2(3, [0.3 1.1 2.4], [-2.1 0.2 1.9]);",
            python_eval=lambda: rf.sh2(3, np.array([0.3, 1.1, 2.4]), np.array([-2.1, 0.2, 1.9])),
        ),
        FunctionCase(
            case_id="rafaely_bn",
            api_names=("bn",),
            matlab_body="out = Bn(3, [0.2 1.3 2.7], 0.7, 1);",
            python_eval=lambda: rf.bn(3, np.array([0.2, 1.3, 2.7]), 0.7, 1),
            rtol=1e-4,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="rafaely_bn_mat",
            api_names=("bn_mat",),
            matlab_body="out = BnMat(3, [0.2 1.3], 0.7, 1);",
            python_eval=lambda: rf.bn_mat(3, np.array([0.2, 1.3]), 0.7, 1),
        ),
        FunctionCase(
            case_id="rafaely_chebyshev_coefficients",
            api_names=("chebyshev_coefficients",),
            matlab_body="out = chebyshev_coefficients(6);",
            python_eval=lambda: rf.chebyshev_coefficients(6),
        ),
        FunctionCase(
            case_id="rafaely_legendre_coefficients",
            api_names=("legendre_coefficients",),
            matlab_body="out = legendre_coefficients(5);",
            python_eval=lambda: rf.legendre_coefficients(5),
        ),
        FunctionCase(
            case_id="rafaely_wigner_d_matrix",
            api_names=("wigner_d_matrix",),
            matlab_body="out = wignerD(3, 0.7, 1.1, -0.4);",
            python_eval=lambda: rf.wigner_d_matrix(3, 0.7, 1.1, -0.4),
        ),
        FunctionCase(
            case_id="rafaely_derivative_ph",
            api_names=("derivative_ph",),
            matlab_body="v = sh2(2, 1.1, -0.2); out = derivative_ph(v);",
            python_eval=lambda: rf.derivative_ph(rf.sh2(2, np.array([1.1]), np.array([-0.2]))[:, 0]),
        ),
        FunctionCase(
            case_id="rafaely_derivative_th",
            api_names=("derivative_th",),
            matlab_body="v = sh2(2, 1.1, -0.2); out = derivative_th(v, 1.1, -0.2);",
            python_eval=lambda: rf.derivative_th(rf.sh2(2, np.array([1.1]), np.array([-0.2]))[:, 0], 1.1, -0.2),
        ),
        FunctionCase(
            case_id="rafaely_c2s",
            api_names=("c2s",),
            matlab_body="[th, ph, r] = c2s([1 0 -1], [0 1 1], [1 2 3]); out = {th, ph, r};",
            python_eval=lambda: list(rf.c2s(np.array([1.0, 0.0, -1.0]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))),
        ),
        FunctionCase(
            case_id="rafaely_s2c",
            api_names=("s2c",),
            matlab_body="[x, y, z] = s2c([0.4 1.0], [0.2 -0.5], [1.2 0.8]); out = {x, y, z};",
            python_eval=lambda: list(rf.s2c(np.array([0.4, 1.0]), np.array([0.2, -0.5]), np.array([1.2, 0.8]))),
        ),
        FunctionCase(
            case_id="rafaely_equiangle_sampling",
            api_names=("equiangle_sampling",),
            matlab_body="[a, th, ph] = equiangle_sampling(3); out = {a, th, ph};",
            python_eval=lambda: list(rf.equiangle_sampling(3)),
        ),
        FunctionCase(
            case_id="rafaely_gaussian_sampling",
            api_names=("gaussian_sampling",),
            matlab_body="[a, th, ph] = gaussian_sampling(3); out = {a, th, ph};",
            python_eval=lambda: list(rf.gaussian_sampling(3)),
        ),
        FunctionCase(
            case_id="rafaely_uniform_sampling",
            api_names=("uniform_sampling",),
            matlab_body="[a, th, ph] = uniform_sampling(2); out = {a, th, ph};",
            python_eval=lambda: list(rf.uniform_sampling(2)),
        ),
        FunctionCase(
            case_id="rafaely_platonic_solid",
            api_names=("platonic_solid",),
            matlab_body="[v, f] = platonic_solid(4, 2.0); out = {v, f};",
            python_eval=lambda: list(rf.platonic_solid(4, 2.0)),
        ),
        FunctionCase(
            case_id="politis_sorted_eig",
            api_names=("sorted_eig",),
            matlab_body="X = [3 0 0; 0 1 0; 0 0 2]; [V, S] = sorted_eig(X, 'ascend'); out = {V, S};",
            python_eval=lambda: list(po.sorted_eig(np.diag([3.0, 1.0, 2.0]), direction="ascend")),
        ),
        FunctionCase(
            case_id="politis_getSH",
            api_names=("getSH",),
            matlab_body="dirs = [0 1.2; 1.1 0.7; -0.9 2.0]; out = getSH(3, dirs, 'real');",
            python_eval=lambda: po.getSH(3, np.array([[0.0, 1.2], [1.1, 0.7], [-0.9, 2.0]], dtype=float), "real"),
        ),
        FunctionCase(
            case_id="politis_unitSph2cart",
            api_names=("unitSph2cart",),
            matlab_body="dirs = [0 0; pi/2 0.2; -1.1 -0.3]; out = unitSph2cart(dirs);",
            python_eval=lambda: po.unitSph2cart(np.array([[0.0, 0.0], [np.pi / 2.0, 0.2], [-1.1, -0.3]], dtype=float)),
        ),
        FunctionCase(
            case_id="politis_sphModalCoeffs",
            api_names=("sphModalCoeffs",),
            matlab_body="out = sphModalCoeffs(3, [0.1 0.5 1.0 2.0], 'rigid');",
            python_eval=lambda: po.sphModalCoeffs(3, np.array([0.1, 0.5, 1.0, 2.0], dtype=float), "rigid"),
        ),
        FunctionCase(
            case_id="politis_getTdesign",
            api_names=("getTdesign",),
            matlab_body="[v, d] = getTdesign(5); out = {v, d};",
            python_eval=lambda: list(po.getTdesign(5)),
        ),
        FunctionCase(
            case_id="politis_conjCoeffs",
            api_names=("conjCoeffs",),
            matlab_body="f = [1+0i; 2-1i; -0.5+0.3i; 0.2-0.1i; 0.7+0.4i; -1.1+0.8i; 0.9-0.6i; -0.3+0.5i; 0.4+0.2i]; out = conjCoeffs(f);",
            python_eval=lambda: po.conjCoeffs(
                np.array([1 + 0j, 2 - 1j, -0.5 + 0.3j, 0.2 - 0.1j, 0.7 + 0.4j, -1.1 + 0.8j, 0.9 - 0.6j, -0.3 + 0.5j, 0.4 + 0.2j])
            ),
        ),
        FunctionCase(
            case_id="politis_rotateAxisCoeffs",
            api_names=("rotateAxisCoeffs",),
            matlab_body="c_n = [1; 0.3; -0.2]; out = rotateAxisCoeffs(c_n, 1.0, -0.4, 'complex');",
            python_eval=lambda: po.rotateAxisCoeffs(np.array([1.0, 0.3, -0.2]), 1.0, -0.4, "complex"),
        ),
        FunctionCase(
            case_id="politis_gaunt_mtx",
            api_names=("gaunt_mtx",),
            matlab_body="out = gaunt_mtx(0, 0, 0);",
            python_eval=lambda: np.asarray(po.gaunt_mtx(0, 0, 0)).squeeze(),
        ),
        FunctionCase(
            case_id="politis_chebyshevPoly",
            api_names=("chebyshevPoly",),
            matlab_body="out = chebyshevPoly(6);",
            python_eval=lambda: po.chebyshevPoly(6),
        ),
        FunctionCase(
            case_id="politis_beamWeightsCardioid2Differential",
            api_names=("beamWeightsCardioid2Differential", "beam_weights_cardioid_to_differential"),
            matlab_body="out = beamWeightsCardioid2Differential(4);",
            python_eval=lambda: po.beamWeightsCardioid2Differential(4),
        ),
        FunctionCase(
            case_id="politis_beamWeightsCardioid2Spherical",
            api_names=("beamWeightsCardioid2Spherical", "beam_weights_cardioid_to_spherical"),
            matlab_body="out = beamWeightsCardioid2Spherical(4);",
            python_eval=lambda: po.beamWeightsCardioid2Spherical(4),
        ),
        FunctionCase(
            case_id="politis_beamWeightsDifferential2Spherical",
            api_names=("beamWeightsDifferential2Spherical", "beam_weights_differential_to_spherical"),
            matlab_body="a = [0.125; 0.375; 0.375; 0.125]; out = beamWeightsDifferential2Spherical(a);",
            python_eval=lambda: po.beamWeightsDifferential2Spherical(np.array([0.125, 0.375, 0.375, 0.125])),
        ),
        FunctionCase(
            case_id="politis_beamWeightsPressureVelocity_real",
            api_names=("beamWeightsPressureVelocity", "beam_weights_pressure_velocity"),
            matlab_body="out = beamWeightsPressureVelocity('real');",
            python_eval=lambda: po.beamWeightsPressureVelocity("real"),
        ),
        FunctionCase(
            case_id="politis_beamWeightsPressureVelocity_complex",
            api_names=(),
            matlab_body="out = beamWeightsPressureVelocity('complex');",
            python_eval=lambda: po.beamWeightsPressureVelocity("complex"),
        ),
        FunctionCase(
            case_id="politis_returnLegePolyCoeffs",
            api_names=("returnLegePolyCoeffs",),
            matlab_body="out = returnLegePolyCoeffs(5);",
            python_eval=lambda: po.returnLegePolyCoeffs(5),
        ),
        FunctionCase(
            case_id="politis_returnChebyPolyCoeffs",
            api_names=("returnChebyPolyCoeffs",),
            matlab_body="out = returnChebyPolyCoeffs(6);",
            python_eval=lambda: po.returnChebyPolyCoeffs(6),
        ),
        FunctionCase(
            case_id="politis_beamWeightsDolphChebyshev2Spherical",
            api_names=("beamWeightsDolphChebyshev2Spherical",),
            matlab_body="out = beamWeightsDolphChebyshev2Spherical(4, 'sidelobe', 0.05);",
            python_eval=lambda: po.beamWeightsDolphChebyshev2Spherical(4, "sidelobe", 0.05),
            rtol=1e-5,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="politis_beamWeightsFromFunction",
            api_names=("beamWeightsFromFunction",),
            matlab_body="f = @(az,el) ones(size(az)); out = beamWeightsFromFunction(f, 2);",
            python_eval=lambda: po.beamWeightsFromFunction(lambda az, el: np.ones_like(az), 2),
            rtol=1e-3,
            atol=1e-3,
        ),
        FunctionCase(
            case_id="politis_beamWeightsHypercardioid2Spherical",
            api_names=("beamWeightsHypercardioid2Spherical",),
            matlab_body="out = beamWeightsHypercardioid2Spherical(3);",
            python_eval=lambda: po.beamWeightsHypercardioid2Spherical(3),
        ),
        FunctionCase(
            case_id="politis_beamWeightsLinear2Spherical",
            api_names=("beamWeightsLinear2Spherical",),
            matlab_body="a = [0.4; 0.3; 0.2; 0.1]; out = beamWeightsLinear2Spherical(a, 0);",
            python_eval=lambda: po.beamWeightsLinear2Spherical(np.array([0.4, 0.3, 0.2, 0.1]), False),
        ),
        FunctionCase(
            case_id="politis_beamWeightsSupercardioid2Spherical",
            api_names=("beamWeightsSupercardioid2Spherical",),
            matlab_body="out = beamWeightsSupercardioid2Spherical(4);",
            python_eval=lambda: po.beamWeightsSupercardioid2Spherical(4),
        ),
        FunctionCase(
            case_id="politis_beamWeightsMaxEV",
            api_names=("beamWeightsMaxEV",),
            matlab_body="out = beamWeightsMaxEV(4);",
            python_eval=lambda: po.beamWeightsMaxEV(4),
        ),
        FunctionCase(
            case_id="politis_beamWeightsTorus2Spherical",
            api_names=("beamWeightsTorus2Spherical",),
            matlab_body="out = beamWeightsTorus2Spherical(3);",
            python_eval=lambda: po.beamWeightsTorus2Spherical(3),
        ),
        FunctionCase(
            case_id="politis_extractAxisCoeffs",
            api_names=("extractAxisCoeffs",),
            matlab_body="a_nm = reshape(1:32, [16 2]); out = extractAxisCoeffs(a_nm);",
            python_eval=lambda: po.extractAxisCoeffs(np.arange(1, 33, dtype=float).reshape((16, 2), order="F")),
        ),
        FunctionCase(
            case_id="politis_check_condition_number_sht",
            api_names=("check_condition_number_sht", "checkCondNumberSHT"),
            matlab_body=f"{mic_dirs_cond_lit}\nmic_dirs_colat = [mic_dirs_cond(:,1), pi/2 - mic_dirs_cond(:,2)];\nout = checkCondNumberSHT(2, mic_dirs_colat, 'real', []);",
            python_eval=lambda: po.checkCondNumberSHT(
                2, np.column_stack([d["mic_dirs_cond"][:, 0], (np.pi / 2) - d["mic_dirs_cond"][:, 1]]), "real", None
            ),
        ),
        FunctionCase(
            case_id="politis_computeVelCoeffsMtx",
            api_names=("computeVelCoeffsMtx",),
            matlab_body="out = computeVelCoeffsMtx(2);",
            python_eval=lambda: po.computeVelCoeffsMtx(2),
            rtol=1e-3,
            atol=1e-3,
        ),
        FunctionCase(
            case_id="politis_beamWeightsVelocityPatterns",
            api_names=("beamWeightsVelocityPatterns",),
            matlab_body=f"{a_xyz_lit}\nout = beamWeightsVelocityPatterns([1; 0.8; 0.5], [0.3 0.2], A_xyz, 'real');",
            python_eval=lambda: po.beamWeightsVelocityPatterns(np.array([1.0, 0.8, 0.5]), np.array([0.3, 0.2]), d["a_xyz"], "real"),
        ),
        FunctionCase(
            case_id="politis_getDiffCohMtxMeas",
            api_names=("getDiffCohMtxMeas",),
            matlab_body=f"{h_array_lit}\nw = [0.12; 0.08; 0.2; 0.1; 0.14; 0.11; 0.09; 0.16]; out = getDiffCohMtxMeas(H_array, w);",
            python_eval=lambda: po.getDiffCohMtxMeas(d["h_array"], np.array([0.12, 0.08, 0.2, 0.1, 0.14, 0.11, 0.09, 0.16])),
        ),
        FunctionCase(
            case_id="politis_getDiffCohMtxTheory",
            api_names=("getDiffCohMtxTheory",),
            matlab_body=f"{mic_dirs_lit}\nout = getDiffCohMtxTheory(mic_dirs, 'rigid', 0.042, 3, [500 1000 2000], []);",
            python_eval=lambda: po.getDiffCohMtxTheory(d["mic_dirs"], "rigid", 0.042, 3, np.array([500.0, 1000.0, 2000.0]), None),
        ),
        FunctionCase(
            case_id="politis_diffCoherence",
            api_names=("diffCoherence",),
            matlab_body="k = [1 2 3]; rA = [0 0 0]; rB = [0.1 0.0 0.0]; a = [1;0;0;0]; b=[1;0;0;0]; out = diffCoherence(k, rA, rB, a, b);",
            python_eval=lambda: po.diffCoherence(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            rtol=1e-4,
            atol=1e-6,
        ),
        FunctionCase(
            case_id="politis_sphArrayNoise",
            api_names=("sphArrayNoise", "sph_array_noise"),
            matlab_body="out = sphArrayNoise(0.042, 32, 4, 'rigid', [300 600 1200]);",
            python_eval=lambda: po.sphArrayNoise(0.042, 32, 4, "rigid", np.array([300.0, 600.0, 1200.0]))[0],
        ),
        FunctionCase(
            case_id="politis_sphArrayNoiseThreshold",
            api_names=("sphArrayNoiseThreshold", "sph_array_noise_threshold"),
            matlab_body="out = sphArrayNoiseThreshold(0.042, 32, 15, 4, 'rigid');",
            python_eval=lambda: po.sphArrayNoiseThreshold(0.042, 32, 15.0, 4, "rigid"),
        ),
        FunctionCase(
            case_id="politis_sphArrayAliasLim",
            api_names=("sphArrayAliasLim", "sph_array_alias_lim"),
            matlab_body=f"{mic_dirs_cond_lit}\nout = sphArrayAliasLim(0.042, 32, 4, mic_dirs_cond);",
            python_eval=lambda: po.sphArrayAliasLim(0.042, 32, 4, d["mic_dirs_cond"])[0],
        ),
        FunctionCase(
            case_id="politis_sphNullformer_pwd",
            api_names=("sphNullformer_pwd",),
            matlab_body=f"{dirs_lit}\nout = sphNullformer_pwd(2, grid_dirs(1:3,:));",
            python_eval=lambda: po.sphNullformer_pwd(2, d["grid_dirs"][:3, :]),
        ),
        FunctionCase(
            case_id="politis_sphNullformer_diff",
            api_names=("sphNullformer_diff",),
            matlab_body=f"{dirs_lit}\nout = sphNullformer_diff(2, grid_dirs(1:3,:));",
            python_eval=lambda: po.sphNullformer_diff(2, d["grid_dirs"][:3, :]),
        ),
        FunctionCase(
            case_id="politis_arraySHTfiltersTheory_radInverse",
            api_names=("arraySHTfiltersTheory_radInverse",),
            matlab_body="[Hf, ht] = arraySHTfiltersTheory_radInverse(0.042, 32, 3, 16, 16000, 15); out = {Hf, ht};",
            python_eval=lambda: list(po.arraySHTfiltersTheory_radInverse(0.042, 32, 3, 16, 16000.0, 15.0)),
        ),
        FunctionCase(
            case_id="politis_arraySHTfiltersTheory_softLim",
            api_names=("arraySHTfiltersTheory_softLim",),
            matlab_body="[Hf, ht] = arraySHTfiltersTheory_softLim(0.042, 32, 3, 16, 16000, 15); out = {Hf, ht};",
            python_eval=lambda: list(po.arraySHTfiltersTheory_softLim(0.042, 32, 3, 16, 16000.0, 15.0)),
        ),
        FunctionCase(
            case_id="politis_arraySHTfiltersTheory_regLS",
            api_names=("arraySHTfiltersTheory_regLS",),
            matlab_body=f"{mic_dirs_lit}\n[Hf, ht] = arraySHTfiltersTheory_regLS(0.042, mic_dirs, 2, 16, 16000, 15); out = {{Hf, ht}};",
            python_eval=lambda: list(po.arraySHTfiltersTheory_regLS(0.042, d["mic_dirs"], 2, 16, 16000.0, 15.0)),
        ),
        FunctionCase(
            case_id="politis_arraySHTfiltersMeas_regLS",
            api_names=("arraySHTfiltersMeas_regLS",),
            matlab_body=f"{h_array_lit}\n{dirs_lit}\nw = ones(size(grid_dirs,1),1)/size(grid_dirs,1);\nout = arraySHTfiltersMeas_regLS(H_array, 2, grid_dirs, w, 8, 15);",
            python_eval=lambda: po.arraySHTfiltersMeas_regLS(
                d["h_array"][:5, :, :],
                2,
                d["grid_dirs"],
                np.full(d["grid_dirs"].shape[0], 1.0 / d["grid_dirs"].shape[0]),
                8,
                15.0,
            )[0].squeeze(0),
        ),
        FunctionCase(
            case_id="politis_arraySHTfiltersMeas_regLSHD",
            api_names=("arraySHTfiltersMeas_regLSHD",),
            matlab_body=f"{h_array_hd_lit}\n{dirs_hd_lit}\nw = ones(size(grid_dirs_hd,1),1)/size(grid_dirs_hd,1);\nout = arraySHTfiltersMeas_regLSHD(H_array_hd, 2, grid_dirs_hd, w, 8, 15);",
            python_eval=lambda: po.arraySHTfiltersMeas_regLSHD(
                d["h_array_hd"],
                2,
                d["grid_dirs_hd"],
                np.full(d["grid_dirs_hd"].shape[0], 1.0 / d["grid_dirs_hd"].shape[0]),
                8,
                15.0,
            )[0],
            rtol=1e-3,
            atol=1e-4,
        ),
        FunctionCase(
            case_id="politis_evaluateSHTfilters",
            api_names=("evaluateSHTfilters",),
            matlab_body=f"{m_mic2sh_lit}\n{h_array_lit}\nY_grid = [ones(size(H_array,3),1) zeros(size(H_array,3),3)];\nw = ones(size(H_array,3),1)/size(H_array,3);\n[cSH, lSH, WNG] = evaluateSHTfilters(M_mic2sh, H_array, 16000, Y_grid, w); out = {{cSH, lSH, WNG}};",
            python_eval=lambda: list(
                po.evaluateSHTfilters(
                    d["m_mic2sh"],
                    d["h_array"],
                    16000.0,
                    np.column_stack([np.ones(d["h_array"].shape[2]), np.zeros((d["h_array"].shape[2], 3))]),
                    np.full(d["h_array"].shape[2], 1.0 / d["h_array"].shape[2]),
                    plot=False,
                )
            ),
            rtol=1e-5,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="politis_arraySHTfilters_diffEQ",
            api_names=("arraySHTfilters_diffEQ",),
            matlab_body=f"{m_mic2sh_lit}\n{m_dfc_lit}\nout = arraySHTfilters_diffEQ(M_mic2sh, M_dfc, [1200 1000 800], 16000);",
            python_eval=lambda: po.arraySHTfilters_diffEQ(d["m_mic2sh"], d["m_dfc"], np.array([1200.0, 1000.0, 800.0]), 16000.0),
            rtol=1e-5,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="politis_getDiffuseness_IE",
            api_names=("getDiffuseness_IE",),
            matlab_body=f"{pv_cov_lit}\nout = getDiffuseness_IE(pvCOV);",
            python_eval=lambda: po.getDiffuseness_IE(d["pv_cov"]),
        ),
        FunctionCase(
            case_id="politis_getDiffuseness_TV",
            api_names=("getDiffuseness_TV",),
            matlab_body=f"{i_vecs_lit}\nout = getDiffuseness_TV(i_vecs);",
            python_eval=lambda: po.getDiffuseness_TV(d["i_vecs"]),
        ),
        FunctionCase(
            case_id="politis_getDiffuseness_SV",
            api_names=("getDiffuseness_SV",),
            matlab_body=f"{i_vecs_lit}\nout = getDiffuseness_SV(i_vecs);",
            python_eval=lambda: po.getDiffuseness_SV(d["i_vecs"]),
        ),
        FunctionCase(
            case_id="politis_getDiffuseness_CMD",
            api_names=("getDiffuseness_CMD",),
            matlab_body=f"{sh_cov_lit}\n[diff, diff_ord] = getDiffuseness_CMD(SHcov); out = {{diff, diff_ord}};",
            python_eval=lambda: list(po.getDiffuseness_CMD(d["sh_cov"])),
            rtol=1e-5,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="politis_getDiffuseness_DPV",
            api_names=("getDiffuseness_DPV",),
            matlab_body=f"{sh_cov_lit}\nout = getDiffuseness_DPV(SHcov);",
            python_eval=lambda: po.getDiffuseness_DPV(d["sh_cov"]),
            rtol=5e-3,
            atol=5e-3,
        ),
        FunctionCase(
            case_id="politis_sparse_solver_irls",
            api_names=("sparse_solver_irls",),
            matlab_body=f"{a_irls_lit}\n{y_irls_lit}\n[X, D, e] = sparse_solver_irls(0.5, A, Y, 0.2, 1e-6, 8); out = {{X, D, e}};",
            python_eval=lambda: list(po.sparse_solver_irls(0.5, d["a_irls"], d["y_irls"], 0.2, 1e-6, 8)),
            rtol=1e-5,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="politis_sphPWDmap",
            api_names=("sphPWDmap",),
            matlab_body=f"{sh_cov_lit}\n{dirs_lit}\n[P, est] = sphPWDmap(SHcov, grid_dirs, 2); out = {{P, [sin(est(:,1)) cos(est(:,1)) est(:,2)]}};",
            python_eval=lambda: (lambda x: [x[0], np.column_stack([np.sin(x[1][:, 0]), np.cos(x[1][:, 0]), x[1][:, 1]])])(po.sphPWDmap(d["sh_cov"], d["grid_dirs"], 2)),
        ),
        FunctionCase(
            case_id="politis_sphMUSIC",
            api_names=("sphMUSIC",),
            matlab_body=f"{sh_cov_lit}\n{dirs_lit}\n[P, est] = sphMUSIC(SHcov, grid_dirs, 2); out = {{P, est}};",
            python_eval=lambda: list(po.sphMUSIC(d["sh_cov"], d["grid_dirs"], 2)),
        ),
        FunctionCase(
            case_id="politis_sphMVDR",
            api_names=("sphMVDR",),
            matlab_body=f"{sh_cov_lit}\nout = sphMVDR(SHcov, [0 0; 1.2 0.3]);",
            python_eval=lambda: po.sphMVDR(d["sh_cov"], np.array([[0.0, 0.0], [1.2, 0.3]])),
        ),
        FunctionCase(
            case_id="politis_sphMVDRmap",
            api_names=("sphMVDRmap",),
            matlab_body=f"{sh_cov_lit}\n{dirs_lit}\n[P, est] = sphMVDRmap(SHcov, grid_dirs, 2); out = P;",
            python_eval=lambda: po.sphMVDRmap(d["sh_cov"], d["grid_dirs"], 2)[0],
        ),
        FunctionCase(
            case_id="politis_sphLCMV",
            api_names=("sphLCMV",),
            matlab_body=f"{sh_cov_lit}\nout = sphLCMV(SHcov, [0 0; 1.2 0.2], [1; 0]);",
            python_eval=lambda: po.sphLCMV(d["sh_cov"], np.array([[0.0, 0.0], [1.2, 0.2]]), np.array([1.0, 0.0])),
        ),
        FunctionCase(
            case_id="politis_sphiPMMW",
            api_names=("sphiPMMW",),
            matlab_body=f"{phi_x_lit}\n{phi_n_lit}\nout = sphiPMMW(Phi_x, Phi_n, [0 0; 1.0 0.1]);",
            python_eval=lambda: po.sphiPMMW(d["phi_x"], d["phi_n"], np.array([[0.0, 0.0], [1.0, 0.1]]))[0],
            rtol=1e-5,
            atol=1e-6,
        ),
        FunctionCase(
            case_id="politis_sphESPRIT",
            api_names=("sphESPRIT",),
            matlab_body=f"{us_lit}\ntmp = sphESPRIT(Us); out = [sin(tmp(:,1)) cos(tmp(:,1)) tmp(:,2)];",
            python_eval=lambda: (lambda res: np.column_stack([np.sin(res[:, 0]), np.cos(res[:, 0]), res[:, 1]]))(po.sphESPRIT(d["us"])),
            rtol=1e-4,
            atol=1e-6,
        ),
        FunctionCase(
            case_id="politis_sphSRmap",
            api_names=("sphSRmap",),
            matlab_body=f"{sh_sig_lit}\n{a_grid_lit}\n{dirs_lit}\n[P, est] = sphSRmap(shsig, 0.5, A_grid, 0.2, 1e-6, 6, grid_dirs, 2); out = {{P, est}};",
            python_eval=lambda: list(po.sphSRmap(d["sh_sig"], 0.5, d["a_grid"], 0.2, 1e-6, 6, d["grid_dirs"], 2)),
            rtol=1e-5,
            atol=1e-7,
        ),
        FunctionCase(
            case_id="politis_sphIntensityHist",
            api_names=("sphIntensityHist",),
            matlab_body=f"{intensity_lit}\n{dirs_lit}\n[P, est] = sphIntensityHist(i_xyz, grid_dirs, 2); out = {{P, est}};",
            python_eval=lambda: list(po.sphIntensityHist(d["intensity_xyz"], d["grid_dirs"], 2)),
        ),
    ]
    return cases


def _build_matlab_master_script(cases: list[FunctionCase], script_path: Path, case_dir: Path) -> None:
    rafaely_math = ROOT / "src" / "Rafaely" / "matlab" / "math"
    politis_root = ROOT / "src" / "Spherical-Array-Processing-MATLAB"
    sht_root = ROOT / "src" / "Spherical-Harmonic-Transform"
    ars_root = ROOT / "src" / "Array-Response-Simulator"
    lines: list[str] = [
        "warning('off','all');",
        f"addpath('{_matlab_quote(case_dir)}');",
        f"addpath('{_matlab_quote(rafaely_math)}');",
        f"addpath('{_matlab_quote(politis_root)}');",
        f"if exist('{_matlab_quote(sht_root)}', 'dir'), addpath('{_matlab_quote(sht_root)}', '-end'); end",
        f"if exist('{_matlab_quote(ars_root)}', 'dir'), addpath('{_matlab_quote(ars_root)}', '-end'); end",
        "",
    ]
    for case in cases:
        out_mat = case_dir / f"{case.case_id}.mat"
        err_txt = case_dir / f"{case.case_id}.err.txt"
        lines.append(f"% ---- {case.case_id}")
        lines.append("try")
        for line in case.matlab_body.splitlines():
            lines.append(f"    {line}")
        lines.append(f"    save('{_matlab_quote(out_mat)}', 'out', '-v7');")
        lines.append("catch ME")
        lines.append(f"    fid = fopen('{_matlab_quote(err_txt)}', 'w');")
        lines.append("    if fid ~= -1")
        lines.append("        fprintf(fid, '%s', getReport(ME, 'extended', 'hyperlinks', 'off'));")
        lines.append("        fclose(fid);")
        lines.append("    end")
        lines.append("end")
        lines.append("")
    lines.append("exit;")
    script_path.write_text("\n".join(lines) + "\n")


def _write_report_md(report: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Function Conformance Report")
    lines.append("")
    lines.append(f"- Cases total: **{report['case_summary']['total']}**")
    lines.append(f"- Cases pass: **{report['case_summary']['pass']}**")
    lines.append(f"- Cases fail: **{report['case_summary']['fail']}**")
    lines.append(f"- Cases expected difference: **{report['case_summary'].get('expected_difference', 0)}**")
    lines.append(f"- Cases skip dependency: **{report['case_summary']['skip_dependency']}**")
    lines.append(f"- Cases matlab error: **{report['case_summary']['error_matlab']}**")
    lines.append(f"- Cases python error: **{report['case_summary']['error_python']}**")
    lines.append("")
    lines.append(f"- API entries total: **{report['api_summary']['total']}**")
    lines.append(f"- API pass: **{report['api_summary']['pass']}**")
    lines.append(f"- API fail: **{report['api_summary']['fail']}**")
    lines.append(f"- API expected difference: **{report['api_summary'].get('expected_difference', 0)}**")
    lines.append(f"- API skip: **{report['api_summary']['skip']}**")
    lines.append("")
    lines.append("## Case Results")
    lines.append("")
    lines.append("| case_id | status | max_abs_diff | max_rel_diff |")
    lines.append("|---|---:|---:|---:|")
    for row in report["cases"]:
        mad = row.get("max_abs_diff")
        mrd = row.get("max_rel_diff")
        lines.append(
            f"| `{row['case_id']}` | `{row['status']}` | "
            f"{'' if mad is None else f'{mad:.3e}'} | "
            f"{'' if mrd is None else f'{mrd:.3e}'} |"
        )
    lines.append("")
    lines.append("## API Coverage")
    lines.append("")
    lines.append("| api | status | case | note |")
    lines.append("|---|---:|---|---|")
    for row in report["api_status"]:
        case_cell = "" if row.get("case_id") is None else f"`{row['case_id']}`"
        lines.append(
            f"| `{row['api_name']}` | `{row['status']}` | "
            f"{case_cell} | "
            f"{row.get('note', '')} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="逐函数 MATLAB/Python 一致性检查")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "function_conformance")
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--allow-probe-fail", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    case_dir = output_dir / "matlab_case_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    for stale in case_dir.glob("*"):
        if stale.is_file():
            stale.unlink()
    cheby_helper = case_dir / "chebyshevPoly.m"
    cheby_helper.write_text(
        (
            "function c_n = chebyshevPoly(n)\n"
            "if n == 0\n"
            "    c_n = 1;\n"
            "    return;\n"
            "end\n"
            "T0 = 1;\n"
            "T1 = [0 1];\n"
            "if n == 1\n"
            "    c_n = T1.';\n"
            "    return;\n"
            "end\n"
            "for k = 2:n\n"
            "    xT1 = [0 2*T1];\n"
            "    T0pad = [T0 zeros(1, numel(xT1)-numel(T0))];\n"
            "    Tn = xT1 - T0pad;\n"
            "    T0 = T1;\n"
            "    T1 = Tn;\n"
            "end\n"
            "c_n = T1.';\n"
            "end\n"
        ),
        encoding="utf-8",
    )
    legendre_helper = case_dir / "legendrePoly.m"
    legendre_helper.write_text(
        (
            "function p_n = legendrePoly(n)\n"
            "p_n = zeros(n+1,1);\n"
            "for r = 0:floor(n/2)\n"
            "    coeff = (1/(2^n)) * ((-1)^r) * factorial(2*n-2*r) / (factorial(r)*factorial(n-r)*factorial(n-2*r));\n"
            "    power = n - 2*r;\n"
            "    p_n(power+1) = coeff;\n"
            "end\n"
            "end\n"
        ),
        encoding="utf-8",
    )

    rt = detect_matlab()
    if rt is None:
        print("MATLAB runtime not found.")
        return 3

    probe = probe_matlab_cli(timeout_s=60)
    if probe.status != "ok" and not args.allow_probe_fail:
        print(f"MATLAB probe is not ready: {probe.status} | {probe.message}")
        return 4

    cases = _build_cases()
    master_script = output_dir / "run_all_cases.m"
    _build_matlab_master_script(cases, master_script, case_dir)

    batch_cmd = f"run('{_matlab_quote(master_script)}')"
    cp = run_matlab_batch(batch_cmd, cwd=ROOT, timeout_s=args.timeout_s)

    case_rows: list[dict[str, Any]] = []
    case_by_api: dict[str, dict[str, Any]] = {}

    for case in cases:
        row: dict[str, Any] = {"case_id": case.case_id, "api_names": list(case.api_names)}
        mat_file = case_dir / f"{case.case_id}.mat"
        err_file = case_dir / f"{case.case_id}.err.txt"
        matlab_error = err_file.read_text(errors="ignore") if err_file.exists() else ""

        py_exc = None
        py_val = None
        try:
            py_val = case.python_eval()
        except Exception:
            py_exc = traceback.format_exc()

        if matlab_error:
            if _is_missing_dependency_error(matlab_error):
                row["status"] = "skip_dependency"
            else:
                row["status"] = "error_matlab"
            row["matlab_error"] = matlab_error[:2000]
        elif not mat_file.exists():
            row["status"] = "error_matlab"
            row["matlab_error"] = "matlab output .mat file not found"
        elif py_exc is not None:
            row["status"] = "error_python"
            row["python_error"] = py_exc[:2000]
        else:
            try:
                ml_val = _safe_load_mat(mat_file)
                cmp_res = _compare_values(py_val, ml_val, case.rtol, case.atol)
                row["comparison"] = cmp_res
                row["max_abs_diff"] = cmp_res["max_abs_diff"]
                row["max_rel_diff"] = cmp_res["max_rel_diff"]
                if cmp_res["ok"]:
                    row["status"] = "pass"
                elif case.case_id in EXPECTED_DIFFERENCE_CASES:
                    row["status"] = "expected_difference"
                    row["note"] = EXPECTED_DIFFERENCE_CASES[case.case_id]
                else:
                    row["status"] = "fail"
            except Exception:
                row["status"] = "error_matlab"
                row["matlab_error"] = traceback.format_exc()[:2000]

        case_rows.append(row)
        for api in case.api_names:
            case_by_api[api] = row

    visual_only = {
        "plot_aliasing",
        "plot_balloon",
        "plot_contour",
        "plot_sampling",
        "plot_sphere",
        "plotAxisymPatternFromCoeffs",
        "plotDirectionalMapFromGrid",
        "plotMicArray",
    }
    non_function = {"RAFAELY_SOURCE_ROOT", "POLITIS_SOURCE_ROOT"}
    no_matlab_equivalent = {"differentialGains"}

    all_api = sorted(set(rf.__all__) | set(po.__all__))
    api_rows: list[dict[str, Any]] = []
    for api in all_api:
        if api in case_by_api:
            c = case_by_api[api]
            api_rows.append(
                {
                    "api_name": api,
                    "status": c["status"],
                    "case_id": c["case_id"],
                    "note": "",
                }
            )
            continue
        if api in non_function:
            api_rows.append({"api_name": api, "status": "skip_non_function", "case_id": None, "note": "module constant"})
            continue
        if api in visual_only:
            api_rows.append({"api_name": api, "status": "skip_visual", "case_id": None, "note": "visual/plot helper"})
            continue
        if api in no_matlab_equivalent:
            api_rows.append({"api_name": api, "status": "skip_no_matlab_function", "case_id": None, "note": "MATLAB source is script, not function"})
            continue
        api_rows.append({"api_name": api, "status": "skip_no_case", "case_id": None, "note": "no deterministic conformance case registered"})

    case_summary = {
        "total": len(case_rows),
        "pass": sum(1 for r in case_rows if r["status"] == "pass"),
        "fail": sum(1 for r in case_rows if r["status"] == "fail"),
        "expected_difference": sum(1 for r in case_rows if r["status"] == "expected_difference"),
        "skip_dependency": sum(1 for r in case_rows if r["status"] == "skip_dependency"),
        "error_matlab": sum(1 for r in case_rows if r["status"] == "error_matlab"),
        "error_python": sum(1 for r in case_rows if r["status"] == "error_python"),
    }
    api_summary = {
        "total": len(api_rows),
        "pass": sum(1 for r in api_rows if r["status"] == "pass"),
        "fail": sum(1 for r in api_rows if r["status"] == "fail"),
        "expected_difference": sum(1 for r in api_rows if r["status"] == "expected_difference"),
        "skip": sum(1 for r in api_rows if r["status"].startswith("skip_")),
    }

    report = {
        "matlab_runtime": {"executable": rt.executable, "source": rt.source},
        "matlab_probe": {
            "status": probe.status,
            "message": probe.message,
            "stdout_tail": probe.stdout_tail,
            "stderr_tail": probe.stderr_tail,
        },
        "matlab_batch_returncode": cp.returncode,
        "matlab_batch_stdout_tail": (cp.stdout or "")[-2000:],
        "matlab_batch_stderr_tail": (cp.stderr or "")[-2000:],
        "case_summary": case_summary,
        "api_summary": api_summary,
        "cases": case_rows,
        "api_status": api_rows,
    }

    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_report_md(report, report_md)

    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(
        f"cases: pass={case_summary['pass']} fail={case_summary['fail']} "
        f"expected_difference={case_summary['expected_difference']} "
        f"skip_dependency={case_summary['skip_dependency']} "
        f"error_matlab={case_summary['error_matlab']} error_python={case_summary['error_python']}"
    )
    print(
        f"api: pass={api_summary['pass']} fail={api_summary['fail']} "
        f"expected_difference={api_summary['expected_difference']} skip={api_summary['skip']}"
    )

    if case_summary["fail"] > 0:
        return 2
    if case_summary["error_matlab"] > 0 or case_summary["error_python"] > 0:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
