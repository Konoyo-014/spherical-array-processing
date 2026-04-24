"""End-to-end plumbing tests for ``array_type="directional"``.

Added in 0.4.0b15 as part of a scorecard item: the public surface
(:class:`ArrayGeometry`, :func:`simulate_sh_array_response`,
:func:`radial_equalizer` and friends) must all accept the
``"directional"`` capsule kind with a ``dir_coeff`` parameter, and the
three well-known limits (``α=1`` ≡ open, ``α=0.5`` ≡ cardioid, ``α=0``
= dipole) must hold at the numerical level.  These tests lock the
contract so refactors do not silently break first-order capsule
arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.array.simulation import (
    simulate_sh_array_response,
)
from spherical_array_processing.encoding.radial_filters import (
    radial_equalizer,
    radial_equalizer_tikhonov,
    radial_equalizer_wng_limited,
)
from spherical_array_processing.types import ArrayGeometry, SphericalGrid


def _geom(n: int = 12) -> ArrayGeometry:
    return ArrayGeometry(
        radius_m=0.042,
        sensor_grid=fibonacci_grid(n),
        array_type="directional",
        metadata={"dir_coeff": 0.5},
    )


def _src_grid() -> SphericalGrid:
    return SphericalGrid(
        azimuth=np.array([0.0, np.pi / 2]),
        angle2=np.array([np.pi / 2, np.pi / 2]),
        convention="az_colat",
    )


def test_array_geometry_accepts_directional_array_type():
    geom = _geom()
    assert geom.array_type == "directional"
    assert geom.metadata["dir_coeff"] == 0.5


# ------------------------------------------------------------------ #
# simulate_sh_array_response                                         #
# ------------------------------------------------------------------ #


def test_simulate_directional_half_matches_cardioid():
    """``α=0.5`` directional must be bitwise identical to ``cardioid``.

    The two closed forms reduce to the same polynomial in ``j_n`` and
    ``j_n'`` (``2π·iⁿ·(j_n - i·j_n')``), so the DC bin *and* every
    non-DC bin must coincide to machine precision.
    """
    geom = ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12))
    src = _src_grid()
    _, h_d = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4,
        array_type="directional", dir_coeff=0.5,
    )
    _, h_c = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4, array_type="cardioid",
    )
    assert np.max(np.abs(h_d - h_c)) == 0.0


def test_simulate_directional_unit_matches_open():
    """``α=1.0`` directional degenerates into the open-sphere response."""
    geom = ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12))
    src = _src_grid()
    _, h_d = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4,
        array_type="directional", dir_coeff=1.0,
    )
    _, h_o = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4, array_type="open",
    )
    assert np.max(np.abs(h_d - h_o)) == 0.0


def test_simulate_directional_zero_is_radial_dipole_at_dc():
    """``α=0`` directional: DC bin is ``cos γ`` (radial dipole pattern).

    A capsule at azimuth 0, colat π/2 looks at source azimuth 0 with
    γ = 0 ⇒ cos γ = +1, and at source azimuth π with γ = π ⇒
    cos γ = -1.  The DC bin collapses to those exact values.
    """
    single_mic = ArrayGeometry(
        radius_m=0.042,
        sensor_grid=SphericalGrid(
            azimuth=np.array([0.0]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        ),
    )
    src = SphericalGrid(
        azimuth=np.array([0.0, np.pi]),
        angle2=np.array([np.pi / 2, np.pi / 2]),
        convention="az_colat",
    )
    _, h = simulate_sh_array_response(
        256, 16000.0, single_mic, src, max_order=6,
        array_type="directional", dir_coeff=0.0,
    )
    assert np.allclose(h[0, 0, 0], 1.0, atol=1e-12)
    assert np.allclose(h[0, 0, 1], -1.0, atol=1e-12)


def test_simulate_directional_requires_dir_coeff():
    geom = ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12))
    src = _src_grid()
    with pytest.raises(ValueError, match="directional.*dir_coeff"):
        simulate_sh_array_response(
            256, 16000.0, geom, src, max_order=2, array_type="directional",
        )


def test_simulate_rejects_stray_dir_coeff_on_non_directional():
    """Parity with the ``radial_equalizer_*`` family: the simulation
    layer must also reject a stray ``dir_coeff`` paired with a
    non-directional ``array_type``, so a typo does not silently fall
    back to (say) the rigid response."""
    geom = ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12))
    src = _src_grid()
    for kind in ("open", "rigid", "cardioid"):
        with pytest.raises(ValueError, match="dir_coeff is only meaningful"):
            simulate_sh_array_response(
                256, 16000.0, geom, src, max_order=2,
                array_type=kind, dir_coeff=0.5,
            )


def test_simulate_rejects_out_of_range_dir_coeff():
    geom = ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12))
    src = _src_grid()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"dir_coeff must be in \[0, 1\]"):
            simulate_sh_array_response(
                256, 16000.0, geom, src, max_order=2,
                array_type="directional", dir_coeff=bad,
            )


# ------------------------------------------------------------------ #
# radial_equalizer family                                            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("func", [radial_equalizer_tikhonov, radial_equalizer_wng_limited])
def test_radial_equalizer_directional_half_matches_cardioid(func):
    kr = np.linspace(0.1, 5.0, 32)
    eq_d = func(3, kr, array_type="directional", dir_coeff=0.5)
    eq_c = func(3, kr, array_type="cardioid")
    assert np.max(np.abs(eq_d - eq_c)) == 0.0


def test_radial_equalizer_directional_unit_matches_open():
    kr = np.linspace(0.1, 5.0, 32)
    eq_d = radial_equalizer(3, kr, array_type="directional", dir_coeff=1.0)
    eq_o = radial_equalizer(3, kr, array_type="open")
    assert np.max(np.abs(eq_d - eq_o)) == 0.0


def test_radial_equalizer_directional_requires_dir_coeff():
    kr = np.linspace(0.1, 5.0, 8)
    with pytest.raises(ValueError, match="directional.*dir_coeff"):
        radial_equalizer(2, kr, array_type="directional")


def test_radial_equalizer_rejects_stray_dir_coeff_on_non_directional():
    """A typo like ``array_type='rigid', dir_coeff=0.5`` must not pass
    silently — it would look correct but produce the rigid response and
    bury a first-order directional-pattern intention."""
    kr = np.linspace(0.1, 5.0, 8)
    for kind in ("open", "rigid", "cardioid"):
        with pytest.raises(ValueError, match="dir_coeff is only meaningful"):
            radial_equalizer(2, kr, array_type=kind, dir_coeff=0.5)


def test_radial_equalizer_rejects_out_of_range_dir_coeff():
    kr = np.linspace(0.1, 5.0, 8)
    with pytest.raises(ValueError, match=r"dir_coeff must be in \[0, 1\]"):
        radial_equalizer(2, kr, array_type="directional", dir_coeff=-0.1)
    with pytest.raises(ValueError, match=r"dir_coeff must be in \[0, 1\]"):
        radial_equalizer(2, kr, array_type="directional", dir_coeff=1.5)


# ------------------------------------------------------------------ #
# Acoustics-layer validation symmetry (0.4.0b15 polish — addresses    #
# the "bottom-layer validation still asymmetric" item in              #
# CODEX_SCORECARD_B15.md).                                            #
# ------------------------------------------------------------------ #


def test_bn_matrix_rejects_stray_dir_coeff_on_non_directional():
    """``bn_matrix(..., sphere="open", dir_coeff=0.5)`` used to
    silently succeed.  After b15 it raises at the acoustics layer
    with the same three-case semantics as the equalizer / simulation
    layers, so ``directional`` validation is now symmetric across the
    entire public API surface."""
    from spherical_array_processing.acoustics import bn_matrix
    kr = np.linspace(0.1, 3.0, 8)
    for kind in ("open", "rigid", "cardioid"):
        with pytest.raises(ValueError, match="dir_coeff is only meaningful"):
            bn_matrix(2, kr, sphere=kind, dir_coeff=0.5)
    # And the integer-coded variants follow suit.
    for kind in (0, 1, 2):
        with pytest.raises(ValueError, match="dir_coeff is only meaningful"):
            bn_matrix(2, kr, sphere=kind, dir_coeff=0.5)


def test_bn_matrix_rejects_out_of_range_dir_coeff():
    from spherical_array_processing.acoustics import bn_matrix
    kr = np.linspace(0.1, 3.0, 8)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"dir_coeff must be in \[0, 1\]"):
            bn_matrix(2, kr, sphere="directional", dir_coeff=bad)


def test_plane_wave_radial_bn_rejects_stray_dir_coeff():
    from spherical_array_processing.acoustics import plane_wave_radial_bn
    kr = np.linspace(0.1, 3.0, 8)
    with pytest.raises(ValueError, match="dir_coeff is only meaningful"):
        plane_wave_radial_bn(1, kr, sphere="rigid", dir_coeff=0.5)


def test_sph_modal_coeffs_rejects_stray_dir_coeff():
    """``sph_modal_coeffs`` talks to users in terms of ``array_type``
    — the error message must reflect that vocabulary, not the
    internal ``sphere`` code."""
    from spherical_array_processing.acoustics.radial import sph_modal_coeffs
    kR = np.linspace(0.1, 3.0, 8)
    with pytest.raises(ValueError, match="dir_coeff is only meaningful"):
        sph_modal_coeffs(2, kR, array_type="open", dir_coeff=0.5)


# ------------------------------------------------------------------ #
# ArrayGeometry single-source-of-truth (0.4.0b15 polish — addresses   #
# the "阵列配置双真相源" item in CODEX_SCORECARD_B15.md).              #
# ------------------------------------------------------------------ #


def test_simulate_reads_array_type_from_geometry_when_omitted():
    """When the caller omits ``array_type``, :func:`simulate_sh_array_response`
    must fall back to ``geometry.array_type`` so that
    :class:`ArrayGeometry` can act as the single source of truth for
    the array spec."""
    geom_cardioid = ArrayGeometry(
        radius_m=0.042, sensor_grid=fibonacci_grid(12),
        array_type="cardioid",
    )
    src = _src_grid()
    _, h_implicit = simulate_sh_array_response(
        256, 16000.0, geom_cardioid, src, max_order=4,
    )
    _, h_explicit = simulate_sh_array_response(
        256, 16000.0, geom_cardioid, src, max_order=4, array_type="cardioid",
    )
    assert np.max(np.abs(h_implicit - h_explicit)) == 0.0


def test_simulate_reads_dir_coeff_from_geometry_metadata():
    """For a directional geometry with ``metadata["dir_coeff"]`` set,
    the caller should not need to re-state the coefficient."""
    geom = ArrayGeometry(
        radius_m=0.042, sensor_grid=fibonacci_grid(12),
        array_type="directional", metadata={"dir_coeff": 0.3},
    )
    src = _src_grid()
    _, h_implicit = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4,
    )
    _, h_explicit = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4,
        array_type="directional", dir_coeff=0.3,
    )
    assert np.max(np.abs(h_implicit - h_explicit)) == 0.0


def test_simulate_explicit_kwarg_overrides_geometry():
    """Explicit ``array_type`` / ``dir_coeff`` must win over whatever
    the geometry carries — otherwise users couldn't do a quick
    what-if analysis without reconstructing the geometry object."""
    geom = ArrayGeometry(
        radius_m=0.042, sensor_grid=fibonacci_grid(12),
        array_type="directional", metadata={"dir_coeff": 0.3},
    )
    src = _src_grid()
    _, h_override = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4, array_type="rigid",
    )
    _, h_rigid_direct = simulate_sh_array_response(
        256, 16000.0,
        ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12)),
        src, max_order=4, array_type="rigid",
    )
    assert np.max(np.abs(h_override - h_rigid_direct)) == 0.0


# ------------------------------------------------------------------ #
# ArrayGeometry.sensor_kind now participates in validation (0.4.0b15  #
# polish round 3 — gives the formerly-informational field a genuine   #
# behavioural role so it stops being a free-floating dataclass tag).  #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "array_type, implied",
    [
        ("open", "pressure"),
        ("rigid", "pressure"),
        ("cardioid", "directional"),
        ("directional", "directional"),
    ],
)
def test_sensor_kind_auto_derives_from_array_type(array_type, implied):
    meta = {"dir_coeff": 0.5} if array_type == "directional" else {}
    geom = ArrayGeometry(
        radius_m=0.042, sensor_grid=fibonacci_grid(12),
        array_type=array_type, metadata=meta,
    )
    assert geom.sensor_kind == implied


def test_sensor_kind_explicit_match_is_allowed():
    """An explicit ``sensor_kind`` that agrees with ``array_type`` must
    pass through unchanged — the validator only rejects the
    inconsistent case."""
    geom = ArrayGeometry(
        radius_m=0.042, sensor_grid=fibonacci_grid(12),
        array_type="rigid", sensor_kind="pressure",
    )
    assert geom.sensor_kind == "pressure"


def test_sensor_kind_inconsistent_with_array_type_raises():
    """The whole point of giving ``sensor_kind`` a behavioural role:
    setting it to a value that disagrees with the baffle spec now
    fails loudly at construction time instead of silently creating an
    incoherent geometry."""
    with pytest.raises(ValueError, match="sensor_kind.*inconsistent"):
        ArrayGeometry(
            radius_m=0.042, sensor_grid=fibonacci_grid(12),
            array_type="rigid", sensor_kind="directional",
        )
    with pytest.raises(ValueError, match="sensor_kind.*inconsistent"):
        ArrayGeometry(
            radius_m=0.042, sensor_grid=fibonacci_grid(12),
            array_type="directional", sensor_kind="pressure",
            metadata={"dir_coeff": 0.5},
        )


def test_simulate_default_geometry_still_produces_rigid_response():
    """Backward compat: a plain :class:`ArrayGeometry` (no
    ``array_type`` argument) still defaults to ``"rigid"``, because
    :class:`ArrayGeometry`'s own default is ``"rigid"``.  Users who
    never heard of the b15 single-source-of-truth wiring see no
    behavioural change."""
    geom = ArrayGeometry(radius_m=0.042, sensor_grid=fibonacci_grid(12))
    src = _src_grid()
    _, h_new = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4,
    )
    _, h_old_default = simulate_sh_array_response(
        256, 16000.0, geom, src, max_order=4, array_type="rigid",
    )
    assert np.max(np.abs(h_new - h_old_default)) == 0.0
