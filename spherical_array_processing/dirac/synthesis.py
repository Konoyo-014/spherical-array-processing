"""DirAC synthesis: re-render per-bin DOA / diffuseness to loudspeakers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..decoding import vbap_gains
from ..types import SphericalGrid
from .analysis import DirACParameters


def _diffuse_mix_matrix(
    n_speakers: int,
    n_frames: int,
    n_freqs: int,
    rng: np.random.Generator | None,
    decorrelate: bool,
) -> NDArray[np.complex128]:
    """Build a per-(frequency, frame) complex mix pattern that spreads
    a diffuse-component amplitude equally across all loudspeakers,
    optionally with bin-wise random phase for decorrelation.
    """
    base = np.ones((n_freqs, n_frames, n_speakers), dtype=np.complex128) / np.sqrt(
        float(n_speakers)
    )
    if not decorrelate:
        return base
    rng = rng if rng is not None else np.random.default_rng()
    phases = rng.uniform(
        0.0, 2.0 * np.pi, size=(n_freqs, n_frames, n_speakers)
    )
    return base * np.exp(1j * phases)


def dirac_synthesize(
    parameters: DirACParameters,
    loudspeaker_grid: SphericalGrid,
    *,
    imaginary_loudspeakers: SphericalGrid | None = None,
    decorrelate_diffuse: bool = True,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Re-render DirAC parameters to loudspeaker STFT signals.

    For each time-frequency bin the omni pressure is split into a
    **direct** component (amplitude-panned to the estimated DOA via
    VBAP) and a **diffuse** component (equal-amplitude distribution,
    optionally per-bin decorrelated).  The two parts are mixed with
    weights ``√(1−ψ)`` and ``√ψ`` respectively so the loudspeaker
    energy preserves the omni pressure's energy on average.

    Parameters
    ----------
    parameters : DirACParameters
        Output of :func:`dirac_analysis`.
    loudspeaker_grid : SphericalGrid
        Loudspeaker directions.  Must form a 3-D convex hull; use the
        ``imaginary_loudspeakers`` argument for hemispherical layouts.
    imaginary_loudspeakers : SphericalGrid or None, optional
        Extra VBAP-helper directions; see
        :func:`spherical_array_processing.decoding.vbap_gains`.
    decorrelate_diffuse : bool, optional
        If ``True`` (default), apply a per-bin random-phase dither to
        the diffuse distribution.  Removes lateral-sum coloration at
        the expense of a synthetic "diffuse" timbre.  Disable for
        critical comparison with real diffuse fields.
    rng : numpy.random.Generator or None, optional
        Random-number source for the decorrelation dither.  Default:
        a fresh :func:`numpy.random.default_rng()`.

    Returns
    -------
    ndarray, shape (F, L, T), complex128
        Loudspeaker STFT ready to be fed back through
        :func:`spherical_array_processing.stft.istft`.
    """
    n_freqs, n_frames, _ = parameters.direction_xyz.shape
    pressure = parameters.pressure  # (F, T) complex
    diffuseness = parameters.diffuseness  # (F, T) real
    directions = parameters.direction_xyz.reshape(-1, 3)  # (F·T, 3)

    n_dirs = directions.shape[0]
    # Replace any zero-direction (no dominant source) with +x by default
    # so VBAP has a well-defined fallback; those bins contribute with
    # weight sqrt(1 - 1) = 0 anyway when ψ = 1.
    norms = np.linalg.norm(directions, axis=1)
    zero_mask = norms < 1e-10
    if np.any(zero_mask):
        directions = directions.copy()
        directions[zero_mask] = np.array([1.0, 0.0, 0.0])

    gains = vbap_gains(
        loudspeaker_grid,
        directions,
        imaginary_loudspeakers=imaginary_loudspeakers,
    )  # (L, F·T)
    n_spk = gains.shape[0]
    gains = gains.reshape(n_spk, n_freqs, n_frames)  # (L, F, T)

    weight_direct = np.sqrt(np.clip(1.0 - diffuseness, 0.0, 1.0))
    weight_diffuse = np.sqrt(np.clip(diffuseness, 0.0, 1.0))

    # Direct part: pressure · weight_direct · gains (broadcast per bin)
    direct = pressure[None, :, :] * weight_direct[None, :, :] * gains

    diffuse_mix = _diffuse_mix_matrix(
        n_spk, n_frames, n_freqs, rng, decorrelate_diffuse
    )  # (F, T, L)
    # Pressure · weight_diffuse · diffuse_mix[..., l]
    diffuse = (
        pressure[:, :, None] * weight_diffuse[:, :, None] * diffuse_mix
    )
    # Reorder direct to match (F, T, L) then add.
    direct_ft_l = np.moveaxis(direct, 0, -1)  # (F, T, L)
    out_ft_l = direct_ft_l + diffuse
    # Return as (F, L, T).
    return np.moveaxis(out_ft_l, 1, 2)
