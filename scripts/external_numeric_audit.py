#!/usr/bin/env python3
"""Cross-check core numerics against independent open-source packages.

This script is intentionally repository-only. It verifies shared mathematical
contracts against optional reference packages without adding them as runtime
dependencies of spherical-array-processing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import spherical_array_processing as sap
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


def _optional_import(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, f"{type(exc).__name__}: {exc}"


def _max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _record(
    records: list[dict[str, Any]],
    name: str,
    status: str,
    *,
    max_abs_error: float | None = None,
    tolerance: float | None = None,
    detail: str = "",
    source: str = "",
) -> None:
    records.append(
        {
            "name": name,
            "status": status,
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
            "detail": detail,
            "source": source,
        }
    )


def _check_close(
    records: list[dict[str, Any]],
    name: str,
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
    source: str,
    detail: str = "",
) -> None:
    err = _max_abs(actual, expected)
    _record(
        records,
        name,
        "pass" if err <= tolerance else "fail",
        max_abs_error=err,
        tolerance=tolerance,
        detail=detail,
        source=source,
    )


def _per_order_from_repeated(weights: np.ndarray, order: int) -> np.ndarray:
    weights = np.asarray(weights)
    out = np.empty(order + 1, dtype=float)
    cursor = 0
    for n in range(order + 1):
        out[n] = float(weights[cursor])
        cursor += 2 * n + 1
    return out


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def run_audit(repo_root: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260414)
    max_order = 4
    n_points = 37
    azimuth = rng.uniform(0.0, 2.0 * np.pi, n_points)
    colatitude = rng.uniform(0.0, np.pi, n_points)
    grid = SphericalGrid(azimuth=azimuth, angle2=colatitude, convention="az_colat")

    complex_spec = SHBasisSpec(max_order=max_order, basis="complex", normalization="orthonormal")
    real_spec = SHBasisSpec(max_order=max_order, basis="real", normalization="orthonormal")
    y_complex = sap.sh.complex_matrix(complex_spec, grid)
    y_real = sap.sh.real_matrix(real_spec, grid)

    scipy_ref = np.zeros_like(y_complex)
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            scipy_ref[:, sap.sh.acn_index(n, m)] = special.sph_harm(m, n, azimuth, colatitude)
    _check_close(
        records,
        "scipy.complex_spherical_harmonics",
        y_complex,
        scipy_ref,
        tolerance=5e-15,
        source="scipy.special.sph_harm",
        detail="Complex orthonormal SH basis, ACN order, azimuth/colatitude convention.",
    )

    spa_sph, err = _optional_import("spaudiopy.sph")
    if spa_sph is None:
        _record(records, "spaudiopy", "skip", detail=err or "not installed", source="spaudiopy")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _check_close(
                records,
                "spaudiopy.complex_sh_matrix",
                y_complex,
                spa_sph.sh_matrix(max_order, azimuth, colatitude, "complex"),
                tolerance=5e-15,
                source="spaudiopy.sph.sh_matrix",
            )
            _check_close(
                records,
                "spaudiopy.real_sh_matrix",
                y_real,
                spa_sph.sh_matrix(max_order, azimuth, colatitude, "real"),
                tolerance=5e-15,
                source="spaudiopy.sph.sh_matrix",
            )
            hyper_errs = []
            cardioid_errs = []
            butterworth_errs = []
            mode_open_errs = []
            mode_rigid_errs = []
            kr = np.linspace(0.2, 8.0, 13)
            for order in range(1, 5):
                hyper_errs.append(
                    _max_abs(
                        sap.beamforming.beam_weights_hypercardioid(order),
                        spa_sph.hypercardioid_modal_weights(order),
                    )
                )
                cardioid_errs.append(
                    _max_abs(
                        sap.beamforming.beam_weights_cardioid(order),
                        spa_sph.cardioid_modal_weights(order),
                    )
                )
                butterworth_errs.append(
                    _max_abs(
                        sap.beamforming.beam_weights_butterworth(order, 5.0, 3.0),
                        spa_sph.butterworth_modal_weights(order, 5.0, 3.0),
                    )
                )
            for n in range(max_order + 1):
                mode_open_errs.append(
                    _max_abs(
                        sap.acoustics.plane_wave_radial_bn(n, kr, sphere="open"),
                        spa_sph.mode_strength(n, kr, sphere_type="open"),
                    )
                )
                mode_rigid_errs.append(
                    _max_abs(
                        sap.acoustics.plane_wave_radial_bn(n, kr, sphere="rigid"),
                        spa_sph.mode_strength(n, kr, sphere_type="rigid"),
                    )
                )
            _record(
                records,
                "spaudiopy.hypercardioid_weights",
                "pass" if max(hyper_errs) <= 5e-15 else "fail",
                max_abs_error=float(max(hyper_errs)),
                tolerance=5e-15,
                source="spaudiopy.sph.hypercardioid_modal_weights",
            )
            _record(
                records,
                "spaudiopy.cardioid_weights",
                "pass" if max(cardioid_errs) <= 5e-14 else "fail",
                max_abs_error=float(max(cardioid_errs)),
                tolerance=5e-14,
                source="spaudiopy.sph.cardioid_modal_weights",
            )
            _record(
                records,
                "spaudiopy.butterworth_modal_weights",
                "pass" if max(butterworth_errs) <= 5e-14 else "fail",
                max_abs_error=float(max(butterworth_errs)),
                tolerance=5e-14,
                source="spaudiopy.sph.butterworth_modal_weights",
            )
            _record(
                records,
                "spaudiopy.open_rigid_mode_strength",
                "pass" if max(mode_open_errs + mode_rigid_errs) <= 5e-13 else "fail",
                max_abs_error=float(max(mode_open_errs + mode_rigid_errs)),
                tolerance=5e-13,
                source="spaudiopy.sph.mode_strength",
            )

    # sound_field_analysis 2021.2.4 still uses NumPy aliases removed in NumPy 2.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128  # type: ignore[attr-defined]
    sfa_sph, err = _optional_import("sound_field_analysis.sph")
    if sfa_sph is None:
        _record(records, "sound_field_analysis", "skip", detail=err or "not installed", source="sound_field_analysis")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _check_close(
                records,
                "sound_field_analysis.complex_sh_all",
                y_complex,
                sfa_sph.sph_harm_all(max_order, azimuth, colatitude, "complex"),
                tolerance=5e-15,
                source="sound_field_analysis.sph.sph_harm_all",
            )
            real_convention_err = _max_abs(
                y_real,
                sfa_sph.sph_harm_all(max_order, azimuth, colatitude, "real"),
            )
            _record(
                records,
                "sound_field_analysis.real_sh_convention_delta",
                "info",
                max_abs_error=real_convention_err,
                detail="Real SH convention differs; complex basis is the comparable contract.",
                source="sound_field_analysis.sph.sph_harm_all",
            )
            kr = np.linspace(0.2, 8.0, 13)
            bessel_errs = []
            bn_errs = []
            for n in range(max_order + 1):
                bessel_errs.extend(
                    [
                        _max_abs(sap.acoustics.besseljs(n, kr), sfa_sph.spbessel(n, kr)),
                        _max_abs(sap.acoustics.besseljsd(n, kr), sfa_sph.dspbessel(n, kr)),
                        _max_abs(sap.acoustics.besselys(n, kr), sfa_sph.spneumann(n, kr)),
                        _max_abs(sap.acoustics.besselysd(n, kr), sfa_sph.dspneumann(n, kr)),
                        _max_abs(sap.acoustics.besselhs(n, kr), sfa_sph.sphankel1(n, kr)),
                        _max_abs(sap.acoustics.besselhsd(n, kr), sfa_sph.dsphankel1(n, kr)),
                        _max_abs(sap.acoustics.besselhs2(n, kr), sfa_sph.sphankel2(n, kr)),
                        _max_abs(sap.acoustics.besselhs2d(n, kr), sfa_sph.dsphankel2(n, kr)),
                    ]
                )
                scale = 4.0 * np.pi * (1j ** n)
                bn_errs.extend(
                    [
                        _max_abs(
                            sap.acoustics.plane_wave_radial_bn(n, kr, sphere="open") / scale,
                            sfa_sph.bn_open_omni(n, kr),
                        ),
                        _max_abs(
                            sap.acoustics.plane_wave_radial_bn(n, kr, sphere="rigid") / scale,
                            sfa_sph.bn_rigid_omni(n, kr, kr),
                        ),
                    ]
                )
            _record(
                records,
                "sound_field_analysis.radial_special_functions",
                "pass" if max(bessel_errs) <= 1e-8 else "fail",
                max_abs_error=float(max(bessel_errs)),
                tolerance=1e-8,
                source="sound_field_analysis.sph spherical Bessel/Hankel wrappers",
            )
            _record(
                records,
                "sound_field_analysis.normalized_modal_coefficients",
                "pass" if max(bn_errs) <= 5e-13 else "fail",
                max_abs_error=float(max(bn_errs)),
                tolerance=5e-13,
                source="sound_field_analysis.sph bn_open_omni/bn_rigid_omni",
            )

    spharpy_spherical, spharpy_err = _optional_import("spharpy.spherical")
    spharpy_samplings, samplings_err = _optional_import("spharpy.samplings")
    spharpy_beamforming, beamforming_err = _optional_import("spharpy.beamforming")
    if spharpy_spherical is None or spharpy_samplings is None or spharpy_beamforming is None:
        _record(
            records,
            "spharpy",
            "skip",
            detail=spharpy_err or samplings_err or beamforming_err or "not installed",
            source="spharpy",
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = spharpy_samplings.Coordinates.from_spherical(
                np.ones(n_points), colatitude, azimuth
            )
            _check_close(
                records,
                "spharpy.complex_harmonic_basis",
                y_complex,
                spharpy_spherical.spherical_harmonic_basis(max_order, coords),
                tolerance=5e-14,
                source="spharpy.spherical.spherical_harmonic_basis",
            )
            _check_close(
                records,
                "spharpy.real_harmonic_basis",
                y_real,
                spharpy_spherical.spherical_harmonic_basis_real(max_order, coords),
                tolerance=5e-14,
                source="spharpy.spherical.spherical_harmonic_basis_real",
            )
            super_errs = []
            maxre_errs = []
            dolph_errs = []
            for order in range(1, 7):
                spharpy_super = _per_order_from_repeated(
                    spharpy_beamforming.maximum_front_back_ratio_weights(order, normalize=True),
                    order,
                )
                spharpy_maxre = _per_order_from_repeated(
                    spharpy_beamforming.rE_max_weights(order, normalize=True),
                    order,
                )
                super_errs.append(_max_abs(sap.beamforming.beam_weights_supercardioid(order), spharpy_super))
                maxre_errs.append(_max_abs(sap.beamforming.beam_weights_maxev(order), spharpy_maxre))
                spharpy_dolph = _per_order_from_repeated(
                    spharpy_beamforming.dolph_chebyshev_weights(order, 10.0, "sidelobe"),
                    order,
                )
                dolph_errs.append(
                    _max_abs(sap.beamforming.beam_weights_dolph_chebyshev(order, 10.0, "sidelobe"), spharpy_dolph)
                )
            _record(
                records,
                "spharpy.supercardioid_front_back_ratio",
                "pass" if max(super_errs) <= 2e-6 else "fail",
                max_abs_error=float(max(super_errs)),
                tolerance=2e-6,
                source="spharpy.beamforming.maximum_front_back_ratio_weights",
            )
            _record(
                records,
                "spharpy.maxre_weight_family_delta",
                "info",
                max_abs_error=float(max(maxre_errs)),
                detail="Both implement max-rE-style tapers but use slightly different published parameterizations.",
                source="spharpy.beamforming.rE_max_weights",
            )
            _record(
                records,
                "spharpy.dolph_chebyshev_weights",
                "pass" if max(dolph_errs) <= 5e-13 else "fail",
                max_abs_error=float(max(dolph_errs)),
                tolerance=5e-13,
                source="spharpy.beamforming.dolph_chebyshev_weights",
            )
            kr = np.linspace(0.2, 8.0, 13)
            spharpy_open = np.diagonal(
                spharpy_spherical.modal_strength(max_order, kr, arraytype="open"),
                axis1=1,
                axis2=2,
            )
            spharpy_rigid = np.diagonal(
                spharpy_spherical.modal_strength(max_order, kr, arraytype="rigid"),
                axis1=1,
                axis2=2,
            )
            sap_open = sap.acoustics.bn_matrix(max_order, kr, sphere="open", repeat_per_order=True)
            sap_rigid = sap.acoustics.bn_matrix(max_order, kr, sphere="rigid", repeat_per_order=True)
            _check_close(
                records,
                "spharpy.open_modal_strength",
                sap_open,
                spharpy_open,
                tolerance=5e-13,
                source="spharpy.spherical.modal_strength",
            )
            _check_close(
                records,
                "spharpy.rigid_modal_strength_global_sign_adjusted",
                sap_rigid,
                -spharpy_rigid,
                tolerance=5e-13,
                source="spharpy.spherical.modal_strength",
                detail="Rigid-sphere response has a global sign convention difference relative to spaudiopy/SFA.",
            )

    sphericart_mod, err = _optional_import("sphericart")
    if sphericart_mod is None:
        _record(records, "sphericart", "skip", detail=err or "not installed", source="sphericart")
    else:
        xyz = sap.coords.unit_sph_to_cart(azimuth, colatitude, convention="az_colat").astype(np.float64)
        _check_close(
            records,
            "sphericart.real_spherical_harmonics",
            y_real,
            sphericart_mod.SphericalHarmonics(max_order).compute(xyz),
            tolerance=5e-14,
            source="sphericart.SphericalHarmonics.compute",
        )

    repo_heads = {}
    if repo_root is not None:
        for name in ["spaudiopy", "sound_field_analysis-py", "spharpy", "sphericart", "pyroomacoustics"]:
            head = _git_head(repo_root / name)
            if head is not None:
                repo_heads[name] = head

    return {
        "package_version": sap.__version__,
        "sample_seed": 20260414,
        "max_order": max_order,
        "n_points": n_points,
        "repo_heads": repo_heads,
        "records": records,
    }


def _print_summary(result: dict[str, Any]) -> None:
    for record in result["records"]:
        err = record["max_abs_error"]
        tol = record["tolerance"]
        if err is None:
            metric = ""
        elif tol is None:
            metric = f" max_abs_error={err:.3e}"
        else:
            metric = f" max_abs_error={err:.3e} tolerance={tol:.3e}"
        detail = f" :: {record['detail']}" if record["detail"] else ""
        print(f"{record['status']:>4} {record['name']}{metric}{detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional directory containing downloaded external git repositories.",
    )
    args = parser.parse_args()

    result = run_audit(args.repo_root)
    _print_summary(result)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    failed = [r for r in result["records"] if r["status"] == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
