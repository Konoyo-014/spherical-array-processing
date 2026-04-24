from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def diffuse_coherence_matrix_omni(sensor_xyz: ArrayLike, freqs_hz: ArrayLike, c: float = 343.0) -> np.ndarray:
    """Compute the theoretical diffuse-field coherence matrix for omnidirectional sensors.

    Parameters
    ----------
    sensor_xyz : array_like
        Sensor positions of shape ``(M, 3)`` in Cartesian coordinates
        (metres).
    freqs_hz : array_like
        Frequency bins in Hz.  Will be broadcast to a 1-D array.
    c : float, optional
        Speed of sound in m/s.  Default is 343.0.

    Returns
    -------
    numpy.ndarray
        Complex coherence matrices of shape ``(F, M, M)`` where *F* is
        the number of frequencies and *M* is the number of sensors.
        Each ``(M, M)`` slice contains ``sinc(k * d / pi)`` values.

    Raises
    ------
    ValueError
        If *sensor_xyz* is not two-dimensional or does not have 3 columns.

    Examples
    --------
    >>> import numpy as np
    >>> xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    >>> C = diffuse_coherence_matrix_omni(xyz, np.array([0.0]))
    >>> C.shape
    (1, 2, 2)
    >>> float(np.real(C[0, 0, 1]))
    1.0
    """
    xyz = np.asarray(sensor_xyz, dtype=float)
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("sensor_xyz must be [M,3]")
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    k = 2 * np.pi * f / c
    out = np.ones((f.size, xyz.shape[0], xyz.shape[0]), dtype=np.complex128)
    for i, kk in enumerate(k):
        x = kk * d
        out[i] = np.sinc(x / np.pi)  # sinc(x) = sin(pi x)/(pi x)
    return out


def diffuse_coherence_from_weights(w_a: ArrayLike, w_b: ArrayLike) -> complex:
    """Compute the diffuse-field coherence between two SH-domain weight vectors.

    Parameters
    ----------
    w_a : array_like
        First spherical-harmonic weight vector of length ``(N+1)^2``.
    w_b : array_like
        Second spherical-harmonic weight vector, same length as *w_a*.

    Returns
    -------
    complex
        Normalised inner product ``<w_a, w_b> / sqrt(<w_a, w_a> * <w_b, w_b>)``.
        Returns ``0+0j`` if either vector has zero energy.

    Raises
    ------
    ValueError
        If *w_a* and *w_b* do not have the same length.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([1.0, 0.0, 0.0, 0.0])
    >>> complex(diffuse_coherence_from_weights(w, w))
    (1+0j)
    """
    a = np.asarray(w_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(w_b, dtype=np.complex128).reshape(-1)
    if a.size != b.size:
        raise ValueError("weight vectors must have same length")
    na = np.vdot(a, a).real
    nb = np.vdot(b, b).real
    if na <= 0 or nb <= 0:
        return 0.0 + 0.0j
    return np.vdot(a, b) / np.sqrt(na * nb)

