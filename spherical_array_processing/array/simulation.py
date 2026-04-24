from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre

from ..acoustics.radial import (
    _validate_sphere_and_dir_coeff,
    bn_matrix,
    kr as kr_func,
)
from ..coords import unit_sph_to_cart
from ..types import ArrayGeometry, SphericalGrid


def simulate_plane_wave_array_response(
    fft_len: int,
    fs: float,
    geometry: ArrayGeometry,
    source_grid: SphericalGrid,
    c: float = 343.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-field plane-wave transfer functions under the DOA convention.

    Returns ``(freqs_hz, H)`` where ``H[f, m, s] = exp(+j k · û_s · r_m)`` is
    the frequency response of sensor *m* to a unit plane wave arriving
    **from** source direction ``û_s`` (i.e. *source_grid* is the
    direction-of-arrival), evaluated at ``freqs[f]``.  The ``+j`` sign
    convention matches the Jacobi–Anger expansion
    ``exp(ik·r) = Σ 4π iⁿ j_n(kr) Y_n^m*(k̂) Y_n^m(r̂)``, and is consistent
    with :func:`spherical_array_processing.acoustics.plane_wave_radial_bn`
    as well as the MATLAB ``simulateSphArray`` reference.

    Parameters
    ----------
    fft_len : int
        FFT length in samples.  The number of non-negative frequency bins
        is ``fft_len // 2 + 1``.
    fs : float
        Sampling rate in Hz.
    geometry : ArrayGeometry
        Microphone array geometry (positions and radius).
    source_grid : SphericalGrid
        Directions **from which** the incident plane waves arrive
        (a.k.a. directions of arrival).
    c : float, optional
        Speed of sound in m/s.  Default is 343.0.

    Returns
    -------
    freqs : np.ndarray, shape (n_bins,)
        Frequency vector in Hz.
    H : np.ndarray, shape (n_bins, n_sensors, n_sources)
        Complex transfer function matrix.

    Notes
    -----
    Prior to v0.4.0 this function used ``exp(-j k · û_s · r_m)``, i.e. it
    treated *source_grid* as propagation direction.  The sign has been
    corrected so that the open-sphere special case of
    :func:`~spherical_array_processing.array.simulation.simulate_sh_array_response`
    reduces to this function, and so that a DOA estimator such as
    :func:`~spherical_array_processing.doa.pwd_spectrum` peaks at the
    true direction of arrival without an external conjugation.

    Examples
    --------
    >>> from spherical_array_processing.types import ArrayGeometry, SphericalGrid
    >>> import numpy as np
    >>> sg = SphericalGrid(azimuth=np.array([0.0]), angle2=np.array([1.57]),
    ...                    weights=np.array([1.0]), convention="az_colat")
    >>> geom = ArrayGeometry(sensor_grid=sg, radius_m=0.042)
    >>> freqs, H = simulate_plane_wave_array_response(512, 16000, geom, sg)
    >>> freqs.shape
    (257,)
    >>> H.shape
    (257, 1, 1)
    """
    n_bins = fft_len // 2 + 1
    freqs = np.arange(n_bins, dtype=float) * fs / fft_len
    k = 2.0 * np.pi * freqs / c
    mic_xyz = unit_sph_to_cart(
        geometry.sensor_grid.azimuth,
        geometry.sensor_grid.angle2,
        convention=geometry.sensor_grid.convention,
    ) * geometry.radius_m
    src_u = unit_sph_to_cart(source_grid.azimuth, source_grid.angle2, convention=source_grid.convention)
    # Plane wave from direction û_s: p(r) = exp(+j k û_s · r) under the
    # Jacobi–Anger / DOA convention matched to plane_wave_radial_bn and
    # simulate_sh_array_response.
    proj = mic_xyz @ src_u.T  # [M, S]
    h = np.exp(1j * k[:, None, None] * proj[None, :, :])
    return freqs, h


ArrayType = Literal["open", "rigid", "cardioid", "directional"]


def simulate_sh_array_response(
    fft_len: int,
    fs: float,
    geometry: ArrayGeometry,
    source_grid: SphericalGrid,
    max_order: int,
    array_type: ArrayType | None = None,
    c: float = 343.0,
    sphere_radius_m: float | None = None,
    *,
    dir_coeff: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Plane-wave array response from modal coefficients for an open, rigid,
    or cardioid spherical array.

    Evaluates the closed-form sum

    ``H[f, m, s] = Σ_{n=0..N} B_n(k r_m, k a) · (2n+1)/(4π) · P_n(cos γ_{ms})``

    where ``γ_{ms}`` is the angle between microphone direction ``r̂_m`` and
    source direction ``k̂_s`` (DOA), and ``B_n`` is obtained from
    :func:`spherical_array_processing.acoustics.bn_matrix` with the
    requested array type.  All microphones are assumed to lie at the same
    radius ``geometry.radius_m``; the scatterer surface radius is
    ``sphere_radius_m`` (defaults to ``geometry.radius_m``, i.e. microphones
    flush with the sphere).

    For ``array_type="open"`` with ``sphere_radius_m == geometry.radius_m``
    this reduces exactly to the free-field plane-wave response
    :func:`simulate_plane_wave_array_response` up to truncation error at
    order ``max_order``.

    Parameters
    ----------
    fft_len : int
        FFT length.  Non-negative bin count is ``fft_len // 2 + 1``.
    fs : float
        Sampling rate in Hz.
    geometry : ArrayGeometry
        Uniform-radius microphone geometry.
    source_grid : SphericalGrid
        DOA grid (directions *from which* the plane waves arrive).
    max_order : int
        SH truncation order ``N``.  For a well-sampled response keep
        ``k r_m ≲ N`` throughout the audible band of interest.
    array_type : {"open", "rigid", "cardioid", "directional"} or None, optional
        Baffle / capsule configuration.  When ``None`` (the default
        as of 0.4.0b15), the value is read from
        ``geometry.array_type``; an explicit kwarg always wins.  This
        lets callers treat :class:`~spherical_array_processing.types.ArrayGeometry`
        as the single source of truth for the array spec while
        preserving full backward compatibility — existing callers
        that pass ``array_type="rigid"`` (or any other explicit
        value) are unaffected, and a geometry that still uses its
        ``"rigid"`` default likewise keeps the pre-b15 behaviour.
    c : float, optional
        Speed of sound in m/s.  Default ``343.0``.
    sphere_radius_m : float or None, optional
        Radius of the scattering baffle.  Defaults to
        ``geometry.radius_m`` (microphones on the sphere surface).
    dir_coeff : float or None, optional
        Directional coefficient ``α ∈ [0, 1]`` required when
        ``array_type="directional"``.  When ``None`` and the resolved
        ``array_type`` is ``"directional"``, the value is read from
        ``geometry.metadata["dir_coeff"]`` so an ``ArrayGeometry``
        that already records its directional coefficient does not
        need the caller to re-state it.  See
        :func:`spherical_array_processing.acoustics.plane_wave_radial_bn`.

    Returns
    -------
    freqs : np.ndarray, shape (n_bins,)
        Non-negative frequency axis in Hz.
    H : np.ndarray, shape (n_bins, n_sensors, n_sources), complex
        Complex transfer function matrix.  For ``array_type="open"`` and
        ``"rigid"`` the DC bin collapses to unity (``B_0(0) = 4π`` and
        ``B_n(0) = 0`` for ``n ≥ 1``); for ``"cardioid"`` the DC bin
        carries the capsule's own directional response
        ``0.5·(1 + cos γ_{ms})`` because ``B_1(0) = 2π/3 ≠ 0``.

    Notes
    -----
    For heterogeneous array radii, iterate the call per microphone subset
    or build the response manually via :func:`bn_matrix` and the addition
    theorem.

    References
    ----------
    .. [1] B. Rafaely, *Fundamentals of Spherical Array Processing*,
       2nd ed., Springer, 2019, eq. (4.19–4.22).

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.array import fibonacci_grid
    >>> from spherical_array_processing.types import ArrayGeometry, SphericalGrid
    >>> geom = ArrayGeometry(sensor_grid=fibonacci_grid(12), radius_m=0.042)
    >>> src = SphericalGrid(azimuth=np.array([0.0]),
    ...                     angle2=np.array([np.pi / 2]),
    ...                     convention="az_colat")
    >>> freqs, H = simulate_sh_array_response(
    ...     256, 16000.0, geom, src, max_order=4, array_type="rigid")
    >>> H.shape
    (129, 12, 1)
    >>> np.allclose(H[0, :, 0], 1.0, atol=1e-12)
    True
    """
    # Resolve the array spec: explicit kwargs win; otherwise fall back
    # to the values carried by :class:`ArrayGeometry`.  This makes the
    # geometry the single source of truth for the array type when the
    # caller doesn't need to override it — eliminating the pre-b15
    # dual-config-source between ``geometry.array_type`` and the
    # function default.
    if array_type is None:
        array_type = geometry.array_type
    if array_type == "directional" and dir_coeff is None:
        dir_coeff = geometry.metadata.get("dir_coeff")
    if array_type not in ("open", "rigid", "cardioid", "directional"):
        raise ValueError(
            "array_type must be one of 'open'/'rigid'/'cardioid'/'directional',"
            f" got {array_type!r}"
        )
    # Delegate the three-case directional-coefficient contract to the
    # shared helper in :mod:`spherical_array_processing.acoustics.radial`
    # so simulation / encoding / acoustics all enforce the exact same
    # rules from one implementation — no room for silent drift.
    _validate_sphere_and_dir_coeff(
        array_type, dir_coeff, arg_name="array_type",
    )
    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    n_bins = fft_len // 2 + 1
    freqs = np.arange(n_bins, dtype=float) * fs / float(fft_len)
    sensor_grid = geometry.sensor_grid
    mic_u = unit_sph_to_cart(
        sensor_grid.azimuth, sensor_grid.angle2, convention=sensor_grid.convention
    )  # [M, 3]
    src_u = unit_sph_to_cart(
        source_grid.azimuth, source_grid.angle2, convention=source_grid.convention
    )  # [S, 3]
    cos_gamma = np.clip(mic_u @ src_u.T, -1.0, 1.0)  # [M, S]

    radius = float(geometry.radius_m)
    a_m = radius if sphere_radius_m is None else float(sphere_radius_m)

    kr_vec = kr_func(freqs, radius_m=radius, c=c)  # [F]
    ka_vec = kr_func(freqs, radius_m=a_m, c=c)  # [F]
    # bn[f, n]  — per-order modal coefficient at measurement radius with
    # scattering surface at a_m.
    bn = bn_matrix(
        max_order,
        kr=kr_vec,
        ka=ka_vec,
        sphere=array_type,
        repeat_per_order=False,
        dir_coeff=dir_coeff,
    )  # [F, N+1]

    # Legendre table P_n(cos γ) for all orders/mic/source triples.
    n_arr = np.arange(max_order + 1)
    # Shape [N+1, M, S].  For N ≲ 50 the memory footprint is modest.
    p_nms = np.empty((max_order + 1, cos_gamma.shape[0], cos_gamma.shape[1]), dtype=float)
    for n in range(max_order + 1):
        p_nms[n] = eval_legendre(n, cos_gamma)

    weights_per_order = (2.0 * n_arr + 1.0) / (4.0 * np.pi)  # [N+1]
    # H[f, m, s] = Σ_n bn[f, n] · weights[n] · P_n[m, s]
    h = np.einsum(
        "fn,n,nms->fms",
        bn,
        weights_per_order,
        p_nms,
        optimize=True,
    )
    # Zero-frequency bin — evaluate in closed form to avoid the modal-sum
    # truncation error that remains at `max_order`:
    #
    #   * open / rigid: ``B_0(0) = 4π`` and ``B_n(0) = 0`` for n ≥ 1, so
    #     the omnidirectional limit is ``H(0) = 1``.
    #   * cardioid: ``B_0(0) = 2π`` and ``B_1(0) = 2π/3``; the sum above
    #     becomes ``0.5 + 0.5·cos γ``, i.e. the capsule's own DC pattern.
    #   * directional: ``B_0(0) = 4πα`` and ``B_1(0) = 4π(1-α)/3``, giving
    #     ``H(0) = α + (1-α)·cos γ`` exactly.
    if array_type in ("open", "rigid"):
        h[0, :, :] = 1.0
    elif array_type == "cardioid":
        h[0, :, :] = 0.5 * (1.0 + cos_gamma)
    else:  # directional
        alpha = float(dir_coeff)  # type: ignore[arg-type]
        h[0, :, :] = alpha + (1.0 - alpha) * cos_gamma
    return freqs, h
