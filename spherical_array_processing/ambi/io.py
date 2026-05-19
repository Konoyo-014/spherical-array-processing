"""AmbiX WAV I/O helpers.

AmbiX files are plain multichannel WAV with the channel layout fixed
to **ACN** ordering and — by convention — **SN3D** normalisation.
Nothing in the WAV container distinguishes AmbiX from any other
multichannel audio file, so the caller (or these helpers) has to know
both the order and the normalisation.

This module requires `soundfile` (installable as the ``[audio]``
extra).  The ``read_ambix_wav`` helper converts into the package's
internal *orthonormal* basis by default; ``write_ambix_wav`` converts
out again to SN3D.  Override either with the ``normalization`` /
``target_normalization`` arguments if your pipeline uses a different
scaling.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .format import Normalization, convert_ambi_normalization
from .spec import AmbisonicFrame, AmbisonicSpec


AxisLayout = Literal["channels_first", "channels_last"]


def _validate_axis_layout(axis: AxisLayout) -> AxisLayout:
    if axis not in {"channels_first", "channels_last"}:
        raise ValueError(
            "axis must be 'channels_first' or 'channels_last', got "
            f"{axis!r}"
        )
    return axis


def _require_soundfile():
    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:  # pragma: no cover — guarded by extras
        raise ImportError(
            "soundfile is required for AmbiX WAV I/O; install with "
            "`pip install 'spherical-array-processing[audio]'` or "
            "`pip install soundfile`."
        ) from exc
    return sf


def _infer_max_order(n_channels: int) -> int:
    """Return N such that (N+1)² == n_channels, or raise."""
    root = int(round(np.sqrt(n_channels)))
    if root * root != n_channels:
        raise ValueError(
            f"{n_channels} channels is not a valid ambisonic "
            f"(N+1)² count; files must have 1, 4, 9, 16, 25, … channels"
        )
    return root - 1


def read_ambix_wav(
    path: str | os.PathLike[str],
    *,
    max_order: int | None = None,
    normalization: Normalization = "sn3d",
    target_normalization: Normalization = "orthonormal",
    axis: AxisLayout = "channels_first",
    dtype: Literal["float32", "float64"] = "float64",
) -> tuple[NDArray[np.floating], float]:
    """Read an AmbiX WAV file into an SH signal.

    Parameters
    ----------
    path : str or Path
        AmbiX ``.wav`` file location.
    max_order : int, optional
        Expected ambisonic order ``N``.  Defaults to the value inferred
        from the channel count (``N = √(n_ch) − 1``).  Passing an
        explicit *max_order* raises ``ValueError`` on mismatch.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation convention of the *file*.  AmbiX default is
        ``"sn3d"``.
    target_normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Convert the loaded coefficients to this convention before
        returning.  Default ``"orthonormal"`` (package internal).
    axis : {"channels_first", "channels_last"}, optional
        Output layout.  ``"channels_first"`` (default) returns ``(Q, T)``;
        ``"channels_last"`` returns ``(T, Q)``.
    dtype : {"float32", "float64"}, optional
        Sample dtype.  Default ``"float64"``.

    Returns
    -------
    sh_signal : ndarray
        The SH coefficient time series.
    fs : float
        Sampling rate in Hz.
    """
    axis = _validate_axis_layout(axis)
    sf = _require_soundfile()
    data, fs = sf.read(str(path), dtype=dtype, always_2d=True)
    # soundfile returns (T, Q) in always_2d=True mode.
    n_samples, n_channels = data.shape
    inferred = _infer_max_order(n_channels)
    if max_order is not None:
        if int(max_order) != inferred:
            raise ValueError(
                f"file has {n_channels} channels implying max_order="
                f"{inferred}, but max_order={max_order} was requested"
            )
        n_order = int(max_order)
    else:
        n_order = inferred
    # Convert normalisation on the channel axis.
    if normalization != target_normalization:
        data = convert_ambi_normalization(
            data, max_order=n_order,
            from_=normalization, to=target_normalization, axis=1,
        )
    if axis == "channels_first":
        return np.ascontiguousarray(data.T), float(fs)
    return np.ascontiguousarray(data), float(fs)


def read_ambix_frame(
    path: str | os.PathLike[str],
    *,
    max_order: int | None = None,
    normalization: Normalization = "sn3d",
    target_normalization: Normalization = "sn3d",
    dtype: Literal["float32", "float64"] = "float64",
    metadata: dict | None = None,
) -> AmbisonicFrame:
    """Read an AmbiX WAV file into an :class:`AmbisonicFrame`.

    Unlike :func:`read_ambix_wav`, the default target normalisation is
    ``"sn3d"`` so the frame preserves the usual AmbiX stream
    convention unless the caller explicitly asks for an internal
    orthonormal representation.
    """
    data, fs = read_ambix_wav(
        path,
        max_order=max_order,
        normalization=normalization,
        target_normalization=target_normalization,
        axis="channels_first",
        dtype=dtype,
    )
    order = _infer_max_order(data.shape[0])
    spec = AmbisonicSpec(
        max_order=order,
        basis="real",
        normalization=target_normalization,
        channel_order="acn",
        domain="time",
    )
    return AmbisonicFrame(
        data,
        spec,
        channel_axis=0,
        sample_rate_hz=fs,
        metadata={} if metadata is None else dict(metadata),
    )


def write_ambix_wav(
    path: str | os.PathLike[str],
    sh_signal: ArrayLike,
    fs: float,
    *,
    max_order: int | None = None,
    source_normalization: Normalization = "orthonormal",
    file_normalization: Normalization = "sn3d",
    axis: AxisLayout = "channels_first",
    subtype: str = "FLOAT",
) -> None:
    """Write an SH signal as an AmbiX WAV file.

    Parameters
    ----------
    path : str or Path
        Destination ``.wav`` path.
    sh_signal : array_like
        SH coefficients laid out as ``(Q, T)`` (default) or ``(T, Q)``
        depending on *axis*.
    fs : float
        Sampling rate in Hz.
    max_order : int, optional
        Expected ambisonic order ``N``.  Defaults to the value inferred
        from the channel axis.
    source_normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation convention of *sh_signal*.  Default
        ``"orthonormal"`` (package internal).
    file_normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation convention to write to disk.  AmbiX default is
        ``"sn3d"``.
    axis : {"channels_first", "channels_last"}, optional
        Layout of *sh_signal*.  Default ``"channels_first"``.
    subtype : str, optional
        WAV sample subtype as understood by :mod:`soundfile`.  Default
        ``"FLOAT"`` (32-bit IEEE float, the common AmbiX choice).
    """
    sf = _require_soundfile()
    sig = np.asarray(sh_signal)
    if sig.ndim != 2:
        raise ValueError(
            f"sh_signal must be 2-D; got shape {sig.shape}"
        )
    axis = _validate_axis_layout(axis)
    if axis == "channels_first":
        data_tq = sig.T
    else:
        data_tq = sig
    n_channels = data_tq.shape[1]
    inferred = _infer_max_order(n_channels)
    if max_order is not None:
        if int(max_order) != inferred:
            raise ValueError(
                f"sh_signal has {n_channels} channels implying max_order="
                f"{inferred}, but max_order={max_order} was requested"
            )
        n_order = int(max_order)
    else:
        n_order = inferred
    if source_normalization != file_normalization:
        data_tq = convert_ambi_normalization(
            data_tq, max_order=n_order,
            from_=source_normalization, to=file_normalization, axis=1,
        )
    sf.write(str(path), np.asarray(data_tq), int(fs), subtype=subtype)


def write_ambix_frame(
    path: str | os.PathLike[str],
    frame: AmbisonicFrame,
    *,
    fs: float | None = None,
    file_normalization: Normalization = "sn3d",
    subtype: str = "FLOAT",
) -> None:
    """Write a time-domain :class:`AmbisonicFrame` as an AmbiX WAV file."""
    if frame.spec.domain != "time":
        raise ValueError("write_ambix_frame requires a time-domain frame")
    if frame.data.ndim != 2:
        raise ValueError("write_ambix_frame requires 2-D time-domain data")
    sample_rate = frame.sample_rate_hz if fs is None else float(fs)
    if sample_rate is None:
        raise ValueError("frame.sample_rate_hz is required unless fs is provided")
    axis = "channels_first" if frame.channel_axis == 0 else "channels_last"
    write_ambix_wav(
        path,
        frame.data,
        sample_rate,
        max_order=frame.spec.max_order,
        source_normalization=frame.spec.normalization,
        file_normalization=file_normalization,
        axis=axis,
        subtype=subtype,
    )


__all__ = [
    "read_ambix_frame",
    "read_ambix_wav",
    "write_ambix_frame",
    "write_ambix_wav",
]
