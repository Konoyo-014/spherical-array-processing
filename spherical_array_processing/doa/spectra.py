from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..coords import unit_sph_to_cart
from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SpatialSpectrumResult, SphericalGrid


def peak_pick_spectrum(spectrum: ArrayLike, n_peaks: int) -> np.ndarray:
    """Return indices of the *n_peaks* largest values in *spectrum*.

    Parameters
    ----------
    spectrum : array_like
        1-D real-valued spatial spectrum.
    n_peaks : int
        Number of peaks to return.  Clamped to ``[1, len(spectrum)]``.

    Returns
    -------
    np.ndarray, shape (n_peaks,)
        Integer indices into *spectrum*, sorted in descending order of
        magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([1.0, 5.0, 3.0, 4.0])
    >>> peak_pick_spectrum(s, 2)
    array([1, 3])
    """
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("spectrum must be non-empty")
    n = max(1, min(n_peaks, s.size))
    idx = np.argpartition(s, -n)[-n:]
    idx = idx[np.argsort(s[idx])[::-1]]
    return idx.astype(int)


def peak_pick_spectrum_nms(
    spectrum: ArrayLike,
    grid: SphericalGrid,
    n_peaks: int,
    *,
    min_separation_deg: float = 10.0,
) -> np.ndarray:
    """Peak-pick a spatial spectrum with angular non-maximum suppression.

    Returns up to *n_peaks* indices into *spectrum*, ordered by
    descending spectrum value, such that every pair of returned peaks
    is separated by at least *min_separation_deg* on the sphere.  Useful
    when picking multiple DOAs from a spatial spectrum whose main lobes
    span several grid cells — vanilla top-N selection tends to report
    neighbouring cells of the strongest peak and miss weaker sources.

    Parameters
    ----------
    spectrum : array_like, shape (G,)
        Real-valued spatial spectrum evaluated on *grid*.
    grid : SphericalGrid
        Grid the spectrum was evaluated on.  Must have ``grid.size == G``.
    n_peaks : int
        Maximum number of peaks to return.
    min_separation_deg : float, optional
        Minimum angular separation between returned peaks, in degrees.
        Default ``10°``.

    Returns
    -------
    np.ndarray
        Array of peak indices, at most ``n_peaks`` long.  May contain
        fewer entries if no more well-separated peaks exist.
    """
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("spectrum must be non-empty")
    if s.size != grid.size:
        raise ValueError("spectrum length must match grid size")
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1")

    u = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)
    cos_thresh = np.cos(np.radians(float(min_separation_deg)))

    sorted_idx = np.argsort(s)[::-1]
    picked: list[int] = []
    for idx in sorted_idx:
        if len(picked) >= int(n_peaks):
            break
        keep = True
        for p in picked:
            if u[idx] @ u[p] > cos_thresh:
                keep = False
                break
        if keep:
            picked.append(int(idx))
    return np.asarray(picked, dtype=np.int64)


def spatial_spectrum_from_map(spectrum: ArrayLike, grid: SphericalGrid, n_peaks: int, metadata: dict | None = None, *, min_separation_deg: float | None = None) -> SpatialSpectrumResult:
    """Wrap a spatial spectrum array and its grid into a SpatialSpectrumResult with peak picking.

    Parameters
    ----------
    spectrum : array_like
        1-D real-valued spatial spectrum evaluated on *grid*.
    grid : SphericalGrid
        Spherical grid at which the spectrum was evaluated.
    n_peaks : int
        Number of dominant peaks to extract.
    metadata : dict or None, optional
        Arbitrary metadata to attach to the result (e.g., method name).
    min_separation_deg : float or None, optional
        If provided, apply angular non-maximum suppression when picking
        peaks: peaks are required to be at least this many degrees apart.
        Default ``None`` reproduces the 0.3.0 vanilla top-N behaviour —
        useful for diagnostics but may report neighbouring cells of the
        same main lobe when *n_peaks > 1*.

    Returns
    -------
    SpatialSpectrumResult
        Named result containing the spectrum, grid, peak indices, and
        peak directions in radians.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.types import SphericalGrid
    >>> g = SphericalGrid(azimuth=np.linspace(0, 2*np.pi, 36, endpoint=False),
    ...                   angle2=np.full(36, 1.57), weights=np.ones(36),
    ...                   convention="az_colat")
    >>> s = np.zeros(36); s[10] = 5.0
    >>> res = spatial_spectrum_from_map(s, g, n_peaks=1)
    >>> int(res.peak_indices[0])
    10
    """
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    if s.size != grid.size:
        raise ValueError("spectrum length must match grid size")
    if min_separation_deg is None:
        idx = peak_pick_spectrum(s, n_peaks)
    else:
        idx = peak_pick_spectrum_nms(
            s, grid, n_peaks, min_separation_deg=min_separation_deg
        )
    dirs = np.stack([grid.azimuth[idx], grid.elevation[idx]], axis=1)
    return SpatialSpectrumResult(
        spectrum=s,
        grid=grid,
        peak_indices=idx.astype(np.int64),
        peak_dirs_rad=dirs.astype(float),
        metadata={} if metadata is None else dict(metadata),
    )


def pwd_spectrum(
    cov: ArrayLike,
    grid: SphericalGrid,
    basis: SHBasisSpec,
    n_peaks: int = 1,
    *,
    min_separation_deg: float | None = None,
) -> SpatialSpectrumResult:
    """Compute the plane-wave decomposition (steered response power) spatial spectrum.

    Computes ``P(q̂) = y(q̂)ᵀ · cov · y(q̂)*`` where ``y(q̂)`` is the SH
    basis evaluated at the scan direction *q̂*.  For a rank-1 covariance
    ``cov = c cᴴ`` this reduces to ``|Σ_{n,m} Y_n^m(q̂) · c_{nm}|²`` —
    the natural addition-theorem form ``(2n+1)/(4π)·P_n(cos γ)`` that
    peaks at the source direction when ``c`` is the physical SHT
    coefficient of a plane wave (``c_{nm} ∝ Y_n^m*(k̂_src)``).

    Covariance-construction convention
    ----------------------------------
    The formula is matched to the SHT output produced by this package.
    For a plane-wave simulation followed by :func:`direct_sht`, the
    natural pipeline is::

        nm = direct_sht(p_mic, Y, mic_grid)
        nm_eq = nm / bn                       # optional radial compensation
        R = np.outer(nm_eq, nm_eq.conj())
        result = pwd_spectrum(R, scan, basis, n_peaks=1)

    The real (tesseral) SH basis is self-conjugate and needs no change.

    Parameters
    ----------
    cov : array_like, shape (Q, Q)
        SH-domain covariance; ``Q = (max_order+1)²`` in ACN ordering.
    grid : SphericalGrid
        Scanning grid of look directions.
    basis : SHBasisSpec
        Spherical harmonic basis specification used to build the
        steering matrix.
    n_peaks : int, optional
        Number of peaks to extract from the spectrum.  Default is 1.

    Returns
    -------
    SpatialSpectrumResult
        Steered response power spectrum with peak directions.

    Raises
    ------
    ValueError
        If the SH matrix is not 2-D or its size does not match *cov*.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.eye(4, dtype=complex)
    >>> # (full usage requires a valid grid and basis)
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square matrix")
    y = np.asarray(sh_matrix(basis, grid))
    if y.ndim != 2:
        raise ValueError("SH matrix must be 2D")
    if y.shape[1] != r.shape[0]:
        raise ValueError("basis/grid and covariance size mismatch")
    p = np.real(np.einsum("gi,ij,gj->g", y, r, np.conj(y)))
    return spatial_spectrum_from_map(
        p,
        grid,
        n_peaks=n_peaks,
        metadata={"method": "pwd"},
        min_separation_deg=min_separation_deg,
    )


def music_spectrum(
    cov: ArrayLike,
    grid: SphericalGrid,
    basis: SHBasisSpec,
    n_sources: int,
    n_peaks: int | None = None,
    *,
    min_separation_deg: float | None = None,
) -> SpatialSpectrumResult:
    """Compute the MUSIC (MUltiple SIgnal Classification) DOA spectrum.

    Uses the same ``y(q̂)ᵀ · R · y(q̂)*`` steering convention as
    :func:`pwd_spectrum`: for a physical SHT covariance
    ``R = E[c cᴴ]`` with ``c_{nm} ∝ Y_n^m*(k̂_src)`` the pseudo-spectrum
    peaks at the source direction without an external conjugation step.

    Parameters
    ----------
    cov : array_like, shape (N, N)
        Covariance matrix in the spherical harmonic domain.
    grid : SphericalGrid
        Scanning grid of look directions.
    basis : SHBasisSpec
        Spherical harmonic basis specification used to build the
        steering matrix.
    n_sources : int
        Assumed number of active sources.  Must satisfy
        ``1 <= n_sources < N``.
    n_peaks : int or None, optional
        Number of peaks to extract.  Defaults to *n_sources*.

    Returns
    -------
    SpatialSpectrumResult
        MUSIC pseudo-spectrum with peak directions.

    Raises
    ------
    ValueError
        If *n_sources* is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.diag([10, 1, 1, 1]).astype(complex)
    >>> # (full usage requires a valid grid and basis)
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square matrix")
    evals, evecs = np.linalg.eigh(r)
    order = np.argsort(evals.real)
    evecs = evecs[:, order]
    n_sources = int(n_sources)
    if n_sources < 1 or n_sources >= r.shape[0]:
        raise ValueError("n_sources must be in [1, n_channels-1]")
    en = evecs[:, : r.shape[0] - n_sources]
    proj = en @ en.conj().T
    y = np.asarray(sh_matrix(basis, grid))
    # Use the same ``y · R · y*`` steering convention as
    # :func:`pwd_spectrum` so both functions agree with the physical
    # SHT covariance ``R = E[c cᴴ]`` (``c ∝ Y_n^m*(k̂_src)``).
    denom = np.real(np.einsum("gi,ij,gj->g", y, proj, np.conj(y)))
    spec = 1.0 / np.maximum(denom, 1e-15)
    return spatial_spectrum_from_map(
        spec,
        grid,
        n_peaks=n_peaks or n_sources,
        metadata={"method": "music"},
        min_separation_deg=min_separation_deg,
    )
