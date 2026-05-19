"""Ambisonic decoder constructions and application helpers."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, QhullError

from ..array.sampling import get_tdesign_fallback
from ..coords import unit_sph_to_cart
from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid


DecoderMethod = Literal["sad", "mmd", "mad", "mode_matching", "epad", "allrad"]
BasisKind = Literal["real", "complex"]
DecoderTaper = Literal["none", "max_re", "in_phase"]


def max_re_sh_weights(max_order: int) -> NDArray[np.float64]:
    """Per-SH-coefficient max-rE amplitude taper for ``N``-th-order decoding.

    Following Zotter & Frank (2012) the max-rE weights are Legendre
    polynomials evaluated at ``cos(2.4068 / (N + 1.51))``, replicated
    ``2n+1`` times so they apply uniformly within every SH degree.
    Normalised so that ``max_re_sh_weights(N)[0] == 1`` (the omni
    channel keeps unit gain).
    """
    from math import cos
    from scipy.special import eval_legendre as _eval_legendre

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    theta = 2.4068 / (float(max_order) + 1.51)
    ct = float(np.cos(theta))
    taper = np.array(
        [float(_eval_legendre(n, ct)) for n in range(max_order + 1)],
        dtype=float,
    )
    # Replicate the per-order taper to per-coefficient ACN layout.
    lengths = [2 * n + 1 for n in range(max_order + 1)]
    per_coeff = np.repeat(taper, lengths)
    return per_coeff


def in_phase_sh_weights(max_order: int) -> NDArray[np.float64]:
    """Per-SH-coefficient in-phase decoder taper.

    The per-order weight is

    ``g_n = N! (N + 1)! / ((N + n + 1)! (N - n)!)``.

    It is replicated over all ``2n + 1`` coefficients of order ``n``.
    The taper is stronger than max-rE and trades spatial sharpness for
    strictly same-sign axisymmetric decoder lobes.
    """
    import math

    N = int(max_order)
    if N < 0:
        raise ValueError("max_order must be non-negative")
    weights = []
    numerator = math.factorial(N) * math.factorial(N + 1)
    for n in range(N + 1):
        denom = math.factorial(N + n + 1) * math.factorial(N - n)
        weights.extend([numerator / denom] * (2 * n + 1))
    return np.asarray(weights, dtype=float)


def decoder_taper_weights(
    max_order: int,
    taper: DecoderTaper = "none",
) -> NDArray[np.float64]:
    """Return a named Ambisonic decoder taper in ACN coefficient order."""
    if taper == "none":
        return np.ones((int(max_order) + 1) ** 2, dtype=float)
    if taper == "max_re":
        return max_re_sh_weights(int(max_order))
    if taper == "in_phase":
        return in_phase_sh_weights(int(max_order))
    raise ValueError("taper must be 'none', 'max_re', or 'in_phase'")


def apply_decoder_taper(
    decoder: ArrayLike,
    max_order: int,
    taper: DecoderTaper = "none",
    *,
    rms_preserving: bool = False,
) -> NDArray[np.floating]:
    """Apply a named per-order taper to decoder columns.

    Parameters
    ----------
    decoder : array_like, shape ``(L, (N+1)^2)``
        Decoder matrix.
    max_order : int
        Ambisonic order ``N``.
    taper : {"none", "max_re", "in_phase"}, optional
        Per-order weighting policy.
    rms_preserving : bool, optional
        If true, scale the tapered decoder so the average coefficient
        energy under a full-band unit prior matches the untapered
        decoder.  This is useful for high-frequency dual-band decoders.
    """
    d = np.asarray(decoder)
    weights = decoder_taper_weights(int(max_order), taper)
    if d.ndim != 2 or d.shape[1] != weights.size:
        raise ValueError(
            f"decoder must have shape (L, {weights.size}) for max_order={max_order}"
        )
    if rms_preserving and taper != "none":
        denom = float(np.mean(weights ** 2))
        if denom > 0:
            weights = weights / np.sqrt(denom)
    return d * weights[None, :]


# --------------------------------------------------------------------------- #
# Loudspeaker-side utilities                                                  #
# --------------------------------------------------------------------------- #


def _speaker_sh(
    spk: SphericalGrid, max_order: int, basis: BasisKind
) -> NDArray[np.floating]:
    """Evaluate the SH basis at loudspeaker directions."""
    spec = SHBasisSpec(
        max_order=int(max_order),
        basis=basis,
        angle_convention=spk.convention,
    )
    return np.asarray(sh_matrix(spec, spk))


def layout_from_directions(
    directions_rad: ArrayLike | SphericalGrid,
    *,
    convention: Literal["az_el", "az_colat"] = "az_el",
    weights: ArrayLike | None = None,
) -> SphericalGrid:
    """Create a loudspeaker grid from ``(azimuth, elevation/colatitude)`` rows."""
    if isinstance(directions_rad, SphericalGrid):
        if weights is None:
            return directions_rad
        return SphericalGrid(
            azimuth=directions_rad.azimuth,
            angle2=directions_rad.angle2,
            convention=directions_rad.convention,
            weights=np.asarray(weights, dtype=float).reshape(-1),
        )
    dirs = np.asarray(directions_rad, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError("directions_rad must have shape (n_speakers, 2)")
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    return SphericalGrid(
        azimuth=dirs[:, 0],
        angle2=dirs[:, 1],
        weights=w,
        convention=convention,
    )


def layout_from_directions_deg(
    directions_deg: ArrayLike,
    *,
    convention: Literal["az_el", "az_colat"] = "az_el",
    weights: ArrayLike | None = None,
) -> SphericalGrid:
    """Create a loudspeaker grid from degree-valued direction rows."""
    return layout_from_directions(
        np.deg2rad(np.asarray(directions_deg, dtype=float)),
        convention=convention,
        weights=weights,
    )


def layout_t_design(order: int, n_points: int | None = None) -> SphericalGrid:
    """Return an equal-weight synthetic t-design fallback layout."""
    return get_tdesign_fallback(max(1, int(order)), n_points=n_points)


def layout_itu_5_1() -> SphericalGrid:
    """Return an ITU-style 5.0 horizontal layout without the LFE channel."""
    return layout_from_directions_deg(
        np.array(
            [
                [30.0, 0.0],
                [-30.0, 0.0],
                [0.0, 0.0],
                [110.0, 0.0],
                [-110.0, 0.0],
            ],
            dtype=float,
        ),
        convention="az_el",
    )


def layout_itu_7_1_4() -> SphericalGrid:
    """Return an 11-channel 7.0.4 layout without the LFE channel."""
    return layout_from_directions_deg(
        np.array(
            [
                [30.0, 0.0],
                [-30.0, 0.0],
                [0.0, 0.0],
                [90.0, 0.0],
                [-90.0, 0.0],
                [150.0, 0.0],
                [-150.0, 0.0],
                [45.0, 45.0],
                [-45.0, 45.0],
                [135.0, 45.0],
                [-135.0, 45.0],
            ],
            dtype=float,
        ),
        convention="az_el",
    )


# --------------------------------------------------------------------------- #
# SAD — Sampling Ambisonic Decoder                                            #
# --------------------------------------------------------------------------- #


def sad_decoder(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    *,
    basis: BasisKind = "real",
) -> NDArray[np.floating]:
    """Sampling Ambisonic Decoder.

    For a loudspeaker grid of size ``L`` the decoder matrix is
    ``D = (4π / L) · Y_spk`` where ``Y_spk[l, q] = Y_q(θ_l)``.  This is
    the closed-form inverse SHT interpretation: the decoded loudspeaker
    signal at ``l`` is the band-limited field value evaluated at
    ``θ_l`` scaled by the uniform surface weight ``4π / L``.

    SAD is exact whenever the loudspeaker layout forms an
    orthonormalising sphere sample (e.g. a t-design of degree ≥ ``2N``
    or a Fibonacci grid at large ``L``).  For irregular layouts its
    front/back energy balance degrades; prefer EPAD or AllRAD in that
    regime.
    """
    y_spk = _speaker_sh(loudspeaker_grid, max_order, basis)
    n_spk = y_spk.shape[0]
    return (4.0 * np.pi / float(n_spk)) * y_spk


# --------------------------------------------------------------------------- #
# MMD — Mode-Matching Decoder                                                 #
# --------------------------------------------------------------------------- #


def mmd_decoder(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    *,
    basis: BasisKind = "real",
    rcond: float = 1e-8,
) -> NDArray[np.floating]:
    """Mode-Matching Decoder — least-squares pseudoinverse of ``Y_spk^T``.

    ``D = pinv(Y_spk^T)`` in the sense of :func:`numpy.linalg.pinv`.
    The resulting decoder produces loudspeaker signals that best match
    each SH mode at the loudspeaker positions.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Loudspeaker directions.
    max_order : int
        Ambisonic order ``N``.
    basis : {"real", "complex"}, optional
        SH basis to target.
    rcond : float, optional
        Singular-value cutoff passed to :func:`numpy.linalg.pinv`.

    Notes
    -----
    For regular / t-design loudspeaker layouts MMD coincides with SAD
    up to numerical precision.  It is the method of choice when the
    layout is irregular but still well-covered (``L ≳ (N+1)²``).
    """
    y_spk = _speaker_sh(loudspeaker_grid, max_order, basis)
    return np.linalg.pinv(y_spk.T, rcond=rcond)


# --------------------------------------------------------------------------- #
# EPAD — Energy-Preserving Ambisonic Decoder (Zotter-Frank 2012)              #
# --------------------------------------------------------------------------- #


def epad_decoder(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    *,
    basis: BasisKind = "real",
) -> NDArray[np.floating]:
    """Energy-Preserving Ambisonic Decoder.

    Constructs ``D`` such that ``Dᵀ · D ∝ I_Q`` via the SVD of
    ``Y_spk = U · Σ · Vᵀ``.  Using ``D = √(4π/L) · U[:, :Q] · Vᵀ``
    replaces all singular values of ``Y_spk`` with a constant, which
    guarantees total-energy preservation regardless of whether the
    loudspeaker layout is regular or not.  For uniform / t-design
    layouts the result coincides with SAD.

    References
    ----------
    .. [1] F. Zotter, H. Pomberger, M. Noisternig, "Energy-preserving
       ambisonic decoding", *Acta Acustica united with Acustica*,
       98(1), 2012.
    """
    y_spk = _speaker_sh(loudspeaker_grid, max_order, basis)
    n_spk, n_coeffs = y_spk.shape
    if n_spk < n_coeffs:
        raise ValueError(
            "EPAD requires at least (N+1)² loudspeakers; got "
            f"{n_spk} speakers for (N+1)² = {n_coeffs} modes."
        )
    u, _s, vt = np.linalg.svd(y_spk, full_matrices=False)
    # u: (L, Q), vt: (Q, Q)
    scale = np.sqrt(4.0 * np.pi / float(n_spk))
    return scale * (u @ vt)


# --------------------------------------------------------------------------- #
# VBAP — Vector Base Amplitude Panning over the loudspeaker convex hull      #
# --------------------------------------------------------------------------- #


def _loudspeaker_triangles(spk_xyz: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return the (n_tri, 3) triangle index table of the loudspeaker
    convex hull on the unit sphere.

    Raises
    ------
    ValueError
        If the loudspeaker layout is coplanar or otherwise degenerate
        (e.g. a single horizontal ring), so that :class:`scipy.spatial.ConvexHull`
        cannot build a 3-D hull.  The caller should either add
        imaginary loudspeakers to close the hull or switch to a
        layout-aware decoder such as :func:`epad_decoder`.
    """
    try:
        hull = ConvexHull(spk_xyz)
    except QhullError as exc:
        raise ValueError(
            "VBAP requires a loudspeaker layout whose positions form a "
            "3-D convex hull on the unit sphere.  The supplied layout "
            "appears coplanar or degenerate (e.g. a purely horizontal "
            "ring).  Add imaginary-loudspeaker positions to close the "
            "hull, or choose a layout-aware decoder such as "
            "`epad_decoder` / `mmd_decoder`."
        ) from exc
    return np.asarray(hull.simplices, dtype=np.int64)


def check_layout_coverage(
    loudspeaker_grid: SphericalGrid,
    *,
    n_probe_points: int = 2562,
) -> dict:
    """Report how well a loudspeaker layout covers the unit sphere.

    Dense-probes the sphere with a Fibonacci lattice and measures, for
    every probe, the angular distance to the nearest real loudspeaker.
    Returns a dict with:

    * ``max_gap_deg`` — largest nearest-speaker angular distance in
      degrees (the worst-case "hole" half-width),
    * ``mean_gap_deg`` — average nearest-speaker distance, a proxy for
      overall coverage uniformity,
    * ``uncovered_fraction_above_30deg`` — fraction of probe points
      whose nearest speaker is more than 30° away.

    Use this as a pre-flight check for VBAP / AllRAD: gaps above
    ~30° are a strong indicator that you need imaginary loudspeakers
    (or a different decoder) to avoid silent centroid fallbacks.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Real loudspeaker directions.
    n_probe_points : int, optional
        Fibonacci probe density.  Default ``2562`` yields ≈ 2.3°
        angular resolution, plenty for the gap statistic.
    """
    spk_xyz = unit_sph_to_cart(
        loudspeaker_grid.azimuth,
        loudspeaker_grid.angle2,
        convention=loudspeaker_grid.convention,
    )
    probe = _fibonacci_unit_sphere(int(n_probe_points))
    nearest_cos = np.clip(np.max(probe @ spk_xyz.T, axis=1), -1.0, 1.0)
    distances_deg = np.degrees(np.arccos(nearest_cos))
    return {
        "max_gap_deg": float(distances_deg.max()),
        "mean_gap_deg": float(distances_deg.mean()),
        "uncovered_fraction_above_30deg": float(
            np.mean(distances_deg > 30.0)
        ),
        "n_probe_points": int(n_probe_points),
    }


def suggest_imaginary_loudspeakers(
    loudspeaker_grid: SphericalGrid,
    *,
    min_cap_half_width_deg: float = 30.0,
    max_imaginary: int = 6,
    strict: bool = False,
    n_probe_points: int = 2562,
) -> SphericalGrid:
    """Heuristically pick auxiliary *imaginary* loudspeaker directions
    that close the convex hull of a partial layout (e.g. an upper-dome).

    The algorithm probes the unit sphere with a dense Fibonacci lattice
    and iteratively adds an imaginary speaker at the worst-covered
    probe point until either every probe's nearest-speaker distance
    is ≤ *min_cap_half_width_deg* or *max_imaginary* speakers have
    been placed.

    The returned grid can be fed to :func:`vbap_gains` and
    :func:`allrad_decoder` via the ``imaginary_loudspeakers`` keyword
    argument; their gains are zeroed out in the final decoder, so the
    convex-hull closure does not change the output energy in covered
    directions.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Real loudspeaker directions.
    min_cap_half_width_deg : float, optional
        Target maximum gap (half-width) between any probe point and
        its nearest (real or imaginary) speaker.  Default ``30°``.
    max_imaginary : int, optional
        Cap on the number of imaginary speakers placed.  Raising this
        lets the routine close tighter layouts at the cost of a bigger
        augmented hull.  Default ``6``.
    strict : bool, optional
        If ``True``, raise :class:`ValueError` when the target gap
        cannot be reached within *max_imaginary* additions.  If
        ``False`` (default), return the best-effort grid and let the
        caller decide.
    n_probe_points : int, optional
        Fibonacci probe density (passed through to the internal
        coverage diagnostic).  Default ``2562``.

    Returns
    -------
    SphericalGrid
        Imaginary speaker grid in the same ``convention`` as the
        input (may have zero points if the layout is already enclosing).

    Raises
    ------
    ValueError
        If ``strict=True`` and the layout still has a gap larger than
        *min_cap_half_width_deg* after placing ``max_imaginary``
        imaginary speakers.
    """
    spk_xyz = unit_sph_to_cart(
        loudspeaker_grid.azimuth,
        loudspeaker_grid.angle2,
        convention=loudspeaker_grid.convention,
    )
    cos_thresh = np.cos(np.radians(float(min_cap_half_width_deg)))
    candidates_xyz: list[np.ndarray] = []
    probe = _fibonacci_unit_sphere(int(n_probe_points))
    current = spk_xyz.copy()
    max_imag = max(0, int(max_imaginary))
    for _ in range(max_imag):
        nearest_cos = np.max(probe @ current.T, axis=1)
        farthest = int(np.argmin(nearest_cos))
        if nearest_cos[farthest] > cos_thresh:
            break
        new_point = probe[farthest]
        candidates_xyz.append(new_point)
        current = np.vstack([current, new_point[None, :]])

    # Final residual — whether we're inside target or not.
    residual_cos = np.max(probe @ current.T, axis=1)
    residual_gap_deg = float(
        np.degrees(np.arccos(np.clip(residual_cos.min(), -1.0, 1.0)))
    )
    if strict and residual_gap_deg > float(min_cap_half_width_deg) + 1e-9:
        raise ValueError(
            "suggest_imaginary_loudspeakers could not close the layout "
            f"to {min_cap_half_width_deg}° within max_imaginary="
            f"{max_imag}; residual gap is {residual_gap_deg:.1f}°. "
            "Raise max_imaginary or accept a larger min_cap_half_width_deg."
        )

    if not candidates_xyz:
        return SphericalGrid(
            azimuth=np.asarray([], dtype=float),
            angle2=np.asarray([], dtype=float),
            convention=loudspeaker_grid.convention,
        )
    arr = np.stack(candidates_xyz, axis=0)
    # Convert back to (azimuth, angle2) with the grid's own convention.
    az = np.arctan2(arr[:, 1], arr[:, 0]) % (2 * np.pi)
    if loudspeaker_grid.convention == "az_colat":
        ang2 = np.arccos(np.clip(arr[:, 2], -1.0, 1.0))
    else:
        ang2 = np.arcsin(np.clip(arr[:, 2], -1.0, 1.0))
    return SphericalGrid(
        azimuth=az,
        angle2=ang2,
        convention=loudspeaker_grid.convention,
    )


def _fibonacci_unit_sphere(n_points: int) -> NDArray[np.float64]:
    """Unit-sphere Fibonacci lattice — no weights, just Cartesian points."""
    i = np.arange(n_points, dtype=float)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    z = 1.0 - (2.0 * i + 1.0) / n_points
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * i / phi
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def vbap_gains(
    loudspeaker_grid: SphericalGrid,
    source_directions_xyz: ArrayLike,
    *,
    imaginary_loudspeakers: SphericalGrid | None = None,
    strict: bool = False,
    uncovered_tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute VBAP gain matrix mapping virtual sources → loudspeakers.

    For each virtual source direction, finds the convex-hull triangle
    containing it and solves the 3-speaker amplitude-panning
    equations ``G · [l_a, l_b, l_c] = k̂`` with all gains non-negative
    and ``sum(g²) = 1``.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Real loudspeaker directions.
    source_directions_xyz : array_like, shape (S, 3)
        Unit-norm Cartesian directions of the ``S`` virtual sources.
    imaginary_loudspeakers : SphericalGrid or None, optional
        Auxiliary speaker directions used only to close the convex
        hull for hemispherical / partial-coverage layouts.  Their
        amplitude-panning gains are computed and then **dropped from
        the returned matrix**, so the final decoder has exactly ``L``
        real loudspeakers.  The trade-off is that virtual sources
        whose nearest triangle includes an imaginary speaker
        contribute less energy than fully-real triangles, but they
        remain correctly panned toward the nearest real speakers.
        Use :func:`suggest_imaginary_loudspeakers` to auto-generate
        a sensible list for a new layout.
    strict : bool, optional
        If ``True``, raise :class:`ValueError` when a virtual source
        lies outside the (possibly augmented) convex hull.  If
        ``False`` (default), emit a :class:`RuntimeWarning` listing
        how many sources are outside the hull and fall back to the
        triangle whose outward normal is closest to the source.
    uncovered_tolerance : float, optional
        Slack on the ``gains ≥ 0`` test when classifying whether a
        virtual source is "inside" a triangle.  Default ``1e-6``.

    Returns
    -------
    ndarray, shape (L, S)
        Gain matrix over **real** loudspeakers only.  Columns have at
        most three non-zero entries when the source is reached by a
        fully-real triangle; they can have as few as one or two
        non-zero entries when the remaining slots go to (dropped)
        imaginary speakers.

    Raises
    ------
    ValueError
        If the augmented loudspeaker layout is still degenerate
        (coplanar, etc.) or — when ``strict=True`` — when any virtual
        source lies outside the hull.
    """
    src = np.asarray(source_directions_xyz, dtype=float)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source_directions_xyz must have shape (S, 3)")
    # Re-normalise defensively.
    src = src / np.maximum(np.linalg.norm(src, axis=1, keepdims=True), 1e-30)

    real_xyz = unit_sph_to_cart(
        loudspeaker_grid.azimuth,
        loudspeaker_grid.angle2,
        convention=loudspeaker_grid.convention,
    )  # (L, 3)
    n_real = real_xyz.shape[0]

    if imaginary_loudspeakers is not None and imaginary_loudspeakers.size > 0:
        imag_xyz = unit_sph_to_cart(
            imaginary_loudspeakers.azimuth,
            imaginary_loudspeakers.angle2,
            convention=imaginary_loudspeakers.convention,
        )
        spk_xyz = np.vstack([real_xyz, imag_xyz])
    else:
        spk_xyz = real_xyz
    n_spk = spk_xyz.shape[0]
    triangles = _loudspeaker_triangles(spk_xyz)  # (T, 3)
    # Pre-compute each triangle's inverse for the linear solve.
    bases = spk_xyz[triangles, :]  # (T, 3, 3) — rows are speaker positions
    inverses = np.linalg.pinv(bases.swapaxes(1, 2))  # invert transposed: (T, 3, 3)
    normals = np.cross(
        bases[:, 1] - bases[:, 0], bases[:, 2] - bases[:, 0]
    )  # (T, 3)
    # Make outward-pointing (dot with any vertex positive).
    flip_mask = np.einsum("tj,tj->t", normals, bases[:, 0]) < 0
    normals[flip_mask] = -normals[flip_mask]

    n_src = src.shape[0]
    gains = np.zeros((n_spk, n_src), dtype=float)
    uncovered = 0
    max_deficit = 0.0
    for s_idx in range(n_src):
        u = src[s_idx]
        candidates = inverses @ u  # (T, 3)
        positive = np.all(candidates >= -float(uncovered_tolerance), axis=1)
        alignment = normals @ u
        scores = np.where(positive, alignment, -np.inf)
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            uncovered += 1
            # Source is outside every triangle.  Fall back to the
            # closest-centroid triangle so the decoder keeps producing
            # output, but record the worst "how far outside" distance
            # to report in the warning message.
            max_deficit = max(
                max_deficit, float(-np.min(candidates))
            )
            centroid_align = (bases.sum(axis=1) / 3.0) @ u
            best = int(np.argmax(centroid_align))
        g = np.clip(candidates[best], 0.0, None)
        norm = np.linalg.norm(g)
        if norm > 0:
            g = g / norm
        for local_idx, spk_idx in enumerate(triangles[best]):
            gains[spk_idx, s_idx] = g[local_idx]

    if uncovered > 0:
        msg = (
            f"{uncovered}/{n_src} virtual-source directions fell outside "
            f"the loudspeaker convex hull (max negative-gain deficit "
            f"{max_deficit:.3e}). These sources were amplitude-panned to "
            "the closest-centroid triangle, which is approximate and can "
            "produce sizeable angular error for uncovered regions."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, category=RuntimeWarning, stacklevel=2)
    # Drop imaginary-loudspeaker rows: the caller only wanted real speakers.
    return gains[:n_real, :]


# --------------------------------------------------------------------------- #
# AllRAD — virtual-source SAD + VBAP                                          #
# --------------------------------------------------------------------------- #


def allrad_decoder(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    *,
    basis: BasisKind = "real",
    virtual_order: int | None = None,
    imaginary_loudspeakers: SphericalGrid | None = None,
    auto_close_hull: bool = False,
    strict: bool = False,
) -> NDArray[np.floating]:
    """All-Round Ambisonic Decoder (Zotter & Frank 2012).

    Builds a SAD decoder on a dense t-design virtual grid, then maps
    each virtual source onto the actual loudspeaker layout via VBAP
    over the convex-hull triangulation.  This makes the decoder robust
    to irregular loudspeaker geometries while preserving the
    front-back energy balance better than plain SAD / MMD on the same
    speakers.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Real loudspeaker directions.
    max_order : int
        Ambisonic order ``N``.
    basis : {"real", "complex"}, optional
        SH basis to target.
    virtual_order : int or None, optional
        t-design order for the virtual grid.  Defaults to ``2 · N + 1``
        (the minimum that supports exact SH integration to order ``N``).
    imaginary_loudspeakers : SphericalGrid or None, optional
        Auxiliary speaker directions used only to close the convex
        hull for hemispherical / partial-coverage layouts.  Their
        gains are computed and dropped from the final decoder matrix;
        see :func:`vbap_gains` for the exact semantics.
    auto_close_hull : bool, optional
        If ``True`` and *imaginary_loudspeakers* is ``None``, call
        :func:`suggest_imaginary_loudspeakers` to auto-select a
        minimal set of imaginary speakers that close the hull.
        Default ``False`` so that explicit control is the norm.
    strict : bool, optional
        If ``True``, raise :class:`ValueError` when any virtual source
        direction lies outside the (possibly augmented) loudspeaker
        hull.  If ``False`` (default), emit a :class:`RuntimeWarning`
        listing the number of uncovered directions.  Prefer
        ``strict=True`` in production pipelines on hemispherical
        / partial-coverage layouts.

    References
    ----------
    .. [1] F. Zotter and M. Frank, "All-round ambisonic panning and
       decoding", *J. Audio Eng. Soc.*, 60(10), 2012.
    """
    virt_n = 2 * int(max_order) + 1 if virtual_order is None else int(virtual_order)
    virt_grid = get_tdesign_fallback(virt_n)
    y_virt = _speaker_sh(virt_grid, max_order, basis)  # (J, Q)
    n_virt = y_virt.shape[0]
    d_virt = (4.0 * np.pi / float(n_virt)) * y_virt  # SAD on the virtual grid

    virt_xyz = unit_sph_to_cart(
        virt_grid.azimuth, virt_grid.angle2, convention=virt_grid.convention
    )

    imag = imaginary_loudspeakers
    if imag is None and auto_close_hull:
        imag = suggest_imaginary_loudspeakers(loudspeaker_grid)
        if imag.size == 0:
            imag = None
    g_vbap = vbap_gains(
        loudspeaker_grid,
        virt_xyz,
        imaginary_loudspeakers=imag,
        strict=strict,
    )
    return g_vbap @ d_virt


# --------------------------------------------------------------------------- #
# Dispatch + application                                                      #
# --------------------------------------------------------------------------- #


def decoder_matrix(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    method: DecoderMethod = "allrad",
    *,
    basis: BasisKind = "real",
    **kwargs,
) -> NDArray[np.floating]:
    """Construct the decoder matrix ``D`` of shape ``(L, (N+1)²)``.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Loudspeaker directions.
    max_order : int
        Ambisonic order ``N``.
    method : {"sad", "mmd", "epad", "allrad"}, optional
        Decoder family.  Default ``"allrad"`` is the best general-purpose
        choice for irregular loudspeaker layouts; for regular
        (t-design-like) layouts ``"sad"`` is faster and produces an
        equivalent result.
    basis : {"real", "complex"}, optional
        SH basis the ambisonic signal lives in.  Real SH is the
        customary choice for audio pipelines (AmbiX / SN3D / ACN).
    **kwargs :
        Method-specific options (``rcond`` for MMD, ``virtual_order``
        for AllRAD).

    Returns
    -------
    ndarray, shape (n_speakers, (max_order + 1) ** 2)
        Decoder matrix.
    """
    if method == "sad":
        return sad_decoder(loudspeaker_grid, max_order, basis=basis)
    if method in ("mmd", "mad", "mode_matching"):
        return mmd_decoder(loudspeaker_grid, max_order, basis=basis, **kwargs)
    if method == "epad":
        return epad_decoder(loudspeaker_grid, max_order, basis=basis)
    if method == "allrad":
        return allrad_decoder(loudspeaker_grid, max_order, basis=basis, **kwargs)
    raise ValueError(
        f"method must be one of 'sad'/'mmd'/'epad'/'allrad', got {method!r}"
    )


def dual_band_decoder_matrix(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    method: DecoderMethod = "allrad",
    *,
    basis: BasisKind = "real",
    rms_preserving: bool = True,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Construct a low-frequency / high-frequency decoder pair.

    Returns the same ``(L, (N+1)²)`` decoder matrix twice, the second
    copy post-multiplied by the per-SH-coefficient **max-rE amplitude
    taper** so that its output optimises the energy vector (Makita /
    Gerzon / Daniel).  When the two matrices are crossfaded around a
    transition frequency (typically ≈ 700 Hz for human listeners) the
    decoder delivers velocity-vector-optimal reproduction at low
    frequencies and energy-vector-optimal reproduction at high
    frequencies — the Gerzon "dual-band" prescription.

    Parameters
    ----------
    loudspeaker_grid : SphericalGrid
        Loudspeaker directions.
    max_order : int
        Ambisonic order ``N``.
    method : {"sad", "mmd", "epad", "allrad"}, optional
        Base decoder family; default ``"allrad"``.
    basis : {"real", "complex"}, optional
    rms_preserving : bool, optional
        If ``True`` (default), scale the max-rE matrix so the
        loudspeaker-energy sum matches the LF matrix under the
        band-limited unit plane-wave prior.  Disabling this gives the
        raw tapered matrix, which is useful when building custom
        crossover scalings.
    **kwargs :
        Extra arguments forwarded to :func:`decoder_matrix` for the
        chosen method (e.g. ``virtual_order`` for AllRAD).

    Returns
    -------
    D_lf : ndarray, shape (L, (N+1)²)
        Basic (velocity-preserving) decoder.
    D_hf : ndarray, shape (L, (N+1)²)
        Max-rE (energy-preserving) decoder.

    References
    ----------
    .. [1] J. Daniel, *Représentation de champs acoustiques,
       application à la transmission et à la reproduction de scènes
       sonores complexes dans un contexte multimédia*, PhD thesis,
       Université Paris VI, 2000.
    .. [2] F. Zotter and M. Frank, "All-round ambisonic panning and
       decoding", *J. Audio Eng. Soc.*, 60(10), 2012.
    """
    d_lf = decoder_matrix(
        loudspeaker_grid, max_order, method=method, basis=basis, **kwargs
    )
    taper = max_re_sh_weights(int(max_order))
    if rms_preserving:
        lengths = np.asarray([2 * n + 1 for n in range(int(max_order) + 1)])
        per_order_taper = np.asarray(
            [taper[n ** 2] for n in range(int(max_order) + 1)], dtype=float
        )
        weighted_sum = float(np.sum(lengths * per_order_taper ** 2))
        total_channels = float((int(max_order) + 1) ** 2)
        energy_scale = float(np.sqrt(total_channels / weighted_sum)) if weighted_sum > 0 else 1.0
    else:
        energy_scale = 1.0
    d_hf = d_lf * (energy_scale * taper)[None, :]
    return d_lf, d_hf


def frequency_dependent_decoder_matrix(
    loudspeaker_grid: SphericalGrid,
    max_order: int,
    freqs_hz: ArrayLike,
    *,
    method: DecoderMethod = "allrad",
    basis: BasisKind = "real",
    low_taper: DecoderTaper = "none",
    high_taper: DecoderTaper = "max_re",
    crossover_hz: float = 700.0,
    crossover_order: int = 4,
    rms_preserving: bool = True,
    **kwargs,
) -> NDArray[np.floating]:
    """Construct a smooth frequency-dependent decoder bank.

    Returns ``D[f]`` with shape ``(F, L, Q)`` by crossfading between a
    low-frequency taper and a high-frequency taper.  The default is the
    classic Gerzon/Daniel velocity-to-energy transition: no LF taper
    and max-rE HF taper around 700 Hz.
    """
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if f.size == 0:
        raise ValueError("freqs_hz must contain at least one frequency")
    if float(crossover_hz) <= 0.0:
        raise ValueError("crossover_hz must be positive")
    if int(crossover_order) <= 0:
        raise ValueError("crossover_order must be positive")
    base = decoder_matrix(
        loudspeaker_grid,
        max_order,
        method=method,
        basis=basis,
        **kwargs,
    )
    d_low = apply_decoder_taper(
        base,
        max_order,
        low_taper,
        rms_preserving=False,
    )
    d_high = apply_decoder_taper(
        base,
        max_order,
        high_taper,
        rms_preserving=bool(rms_preserving),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(f > 0, (f / float(crossover_hz)) ** int(crossover_order), 0.0)
    alpha = 1.0 / (1.0 + ratio)
    g_low = np.sqrt(alpha).reshape(-1, 1, 1)
    g_high = np.sqrt(1.0 - alpha).reshape(-1, 1, 1)
    return g_low * d_low[None, :, :] + g_high * d_high[None, :, :]


def decoder_diagnostics(
    decoder: ArrayLike,
    loudspeaker_grid: SphericalGrid,
    *,
    max_order: int | None = None,
    basis: BasisKind = "real",
    n_probe_points: int = 512,
) -> dict:
    """Compute loudspeaker-decoder health metrics.

    The report is intentionally numeric and machine-readable.  It
    includes matrix rank/conditioning, per-speaker diffuse-level error,
    layout gap statistics, and probe-grid velocity / energy vector
    behaviour.  It is meant as a pre-flight check before publishing a
    layout-specific decoder.
    """
    d = np.asarray(decoder)
    if d.ndim != 2:
        raise ValueError("decoder must be a 2-D matrix")
    q = d.shape[1]
    if max_order is None:
        root = int(round(np.sqrt(q)))
        if root * root != q:
            raise ValueError(
                "max_order is required when decoder column count is not a square"
            )
        max_order = root - 1
    expected_q = (int(max_order) + 1) ** 2
    if q != expected_q:
        raise ValueError(
            f"decoder has {q} columns but max_order={max_order} expects {expected_q}"
        )
    if d.shape[0] != loudspeaker_grid.size:
        raise ValueError(
            "decoder row count must match loudspeaker_grid.size"
        )

    spk_xyz = unit_sph_to_cart(
        loudspeaker_grid.azimuth,
        loudspeaker_grid.angle2,
        convention=loudspeaker_grid.convention,
    )
    singular_values = np.linalg.svd(d, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 0:
        condition = np.inf
    else:
        condition = float(singular_values[0] / singular_values[-1])
    speaker_power = np.sum(np.abs(d) ** 2, axis=1)
    mean_power = float(np.mean(speaker_power))
    diffuse_level_error_db = 10.0 * np.log10(
        np.maximum(speaker_power, 1e-30) / max(mean_power, 1e-30)
    )

    probe = get_tdesign_fallback(
        2 * int(max_order) + 2,
        n_points=max(int(n_probe_points), 4 * expected_q, 32),
    )
    y_probe = np.asarray(
        sh_matrix(
            SHBasisSpec(
                max_order=int(max_order),
                basis=basis,
                angle_convention=probe.convention,
            ),
            probe,
        )
    )
    speaker_signals = y_probe @ d.T
    amp = np.real(speaker_signals)
    power = np.abs(speaker_signals) ** 2
    energy = np.sum(power, axis=1)
    energy_vec = (power @ spk_xyz) / np.maximum(energy[:, None], 1e-30)
    velocity_denom = np.sum(np.abs(amp), axis=1)
    velocity_vec = (amp @ spk_xyz) / np.maximum(velocity_denom[:, None], 1e-30)
    target_xyz = unit_sph_to_cart(
        probe.azimuth,
        probe.angle2,
        convention=probe.convention,
    )

    def _angle_errors(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        norm = np.linalg.norm(vec, axis=1)
        dots = np.sum(vec * target_xyz, axis=1) / np.maximum(norm, 1e-30)
        err = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        return np.where(norm > 1e-12, err, np.nan)

    e_err = _angle_errors(energy_vec)
    v_err = _angle_errors(velocity_vec)
    return {
        "n_speakers": int(d.shape[0]),
        "n_coeffs": int(q),
        "max_order": int(max_order),
        "rank": int(np.linalg.matrix_rank(d)),
        "condition_number": condition,
        "singular_values": singular_values,
        "diffuse_level_error_db": diffuse_level_error_db,
        "diffuse_level_error_db_max_abs": float(np.max(np.abs(diffuse_level_error_db))),
        "layout_coverage": check_layout_coverage(
            loudspeaker_grid,
            n_probe_points=max(int(n_probe_points), 32),
        ),
        "energy_mean": float(np.mean(energy)),
        "energy_min": float(np.min(energy)),
        "energy_max": float(np.max(energy)),
        "energy_vector_magnitude_mean": float(np.mean(np.linalg.norm(energy_vec, axis=1))),
        "energy_vector_angle_error_deg_mean": float(np.nanmean(e_err)),
        "energy_vector_angle_error_deg_max": float(np.nanmax(e_err)),
        "velocity_vector_magnitude_mean": float(np.mean(np.linalg.norm(velocity_vec, axis=1))),
        "velocity_vector_angle_error_deg_mean": float(np.nanmean(v_err)),
        "velocity_vector_angle_error_deg_max": float(np.nanmax(v_err)),
        "n_probe_points": int(probe.size),
    }


def apply_dual_band_decoder(
    ambi_stft: ArrayLike,
    freqs_hz: ArrayLike,
    decoder_lf: ArrayLike,
    decoder_hf: ArrayLike,
    *,
    crossover_hz: float = 700.0,
    crossover_order: int = 4,
    coeff_axis: int = -1,
) -> np.ndarray:
    """Crossfade between the LF/HF decoders of
    :func:`dual_band_decoder_matrix` using a power-complementary
    Butterworth-style magnitude response.

    The crossover weights are

    .. math::

        a(f) = \\frac{1}{1 + (f / f_c)^p}, \\qquad
        g_\\text{LF}(f) = \\sqrt{a(f)}, \\qquad
        g_\\text{HF}(f) = \\sqrt{1 - a(f)}

    so that ``g_LF² + g_HF² = 1`` at all frequencies (power-preserving
    crossover).  This keeps the overall energy constant through the
    transition band.

    Parameters
    ----------
    ambi_stft : array_like
        STFT ambisonic signal with ``(F, ..., Q)`` layout; ``F`` must
        match *freqs_hz*, ``Q`` must match the decoder column count.
    freqs_hz : array_like, shape (F,)
        Frequency axis in Hz.
    decoder_lf, decoder_hf : array_like, shape (L, Q)
        Matrices from :func:`dual_band_decoder_matrix`.
    crossover_hz : float, optional
        Transition frequency.  Default ``700`` Hz matches the classic
        velocity / energy perceptual boundary.
    crossover_order : int, optional
        Butterworth-style slope parameter.  Default ``4`` gives an
        approximately 24 dB/oct transition.
    coeff_axis : int, optional
        Axis of *ambi_stft* that indexes SH coefficients.

    Returns
    -------
    ndarray
        Loudspeaker-domain STFT; the coefficient axis is replaced by
        ``L`` speaker channels.
    """
    a = np.asarray(ambi_stft)
    d_lf = np.asarray(decoder_lf)
    d_hf = np.asarray(decoder_hf)
    if d_lf.shape != d_hf.shape:
        raise ValueError("decoder_lf and decoder_hf must have matching shapes")
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    coeff_axis_norm = int(coeff_axis) % a.ndim
    if coeff_axis_norm == 0:
        raise ValueError("coeff_axis cannot index the frequency axis (0)")
    a_moved = np.moveaxis(a, coeff_axis, -1)
    if a_moved.shape[0] != f.size:
        raise ValueError(
            f"ambi_stft leading axis must equal freqs_hz length; "
            f"got {a_moved.shape[0]} vs {f.size}"
        )
    # Butterworth-style split that is power-complementary at every bin.
    cutoff = float(crossover_hz)
    order = int(crossover_order)
    if cutoff <= 0.0:
        raise ValueError("crossover_hz must be positive")
    if order <= 0:
        raise ValueError("crossover_order must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(f > 0, (f / cutoff) ** order, 0.0)
    alpha = 1.0 / (1.0 + ratio)
    g_lf = np.sqrt(alpha)
    g_hf = np.sqrt(1.0 - alpha)
    lf_part = a_moved @ d_lf.T  # (F, ..., L)
    hf_part = a_moved @ d_hf.T
    # Broadcast the frequency weights back over the ... / L axes.
    extra_ndim = lf_part.ndim - 1
    shape_weights = (f.size,) + (1,) * (extra_ndim)
    out = g_lf.reshape(shape_weights) * lf_part + g_hf.reshape(shape_weights) * hf_part
    return np.moveaxis(out, -1, coeff_axis)


def apply_decoder(
    ambi_signal: ArrayLike,
    decoder: ArrayLike,
    *,
    coeff_axis: int = -1,
) -> np.ndarray:
    """Apply a decoder matrix along a coefficient axis.

    Parameters
    ----------
    ambi_signal : array_like
        Ambisonic signal tensor with ``(N+1)²`` entries along
        *coeff_axis*.
    decoder : array_like, shape ``(L, (N+1)²)``
        Decoder matrix (typically from :func:`decoder_matrix`).
    coeff_axis : int, optional
        Axis in ``ambi_signal`` that indexes SH coefficients.  Default
        is ``-1``.  The decoder is contracted against that axis; the
        length of the new axis is ``L``.

    Returns
    -------
    ndarray
        Loudspeaker-domain signal with ``L`` entries replacing the
        coefficient axis.
    """
    a = np.asarray(ambi_signal)
    d = np.asarray(decoder)
    if d.ndim != 2:
        raise ValueError("decoder must be a 2-D matrix")
    n_coeffs = d.shape[1]
    a_m = np.moveaxis(a, coeff_axis, -1)
    if a_m.shape[-1] != n_coeffs:
        raise ValueError(
            f"ambi_signal has {a_m.shape[-1]} coefficients along the requested "
            f"axis, decoder expects {n_coeffs}."
        )
    out = a_m @ d.T  # (..., L)
    return np.moveaxis(out, -1, coeff_axis)
