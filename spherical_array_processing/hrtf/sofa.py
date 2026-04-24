"""Minimal SOFA (AES69) reader for ``SimpleFreeFieldHRIR`` files.

Covers the single SOFA convention that accounts for the vast majority
of published HRTF datasets (HUTUBS, ARI, SADIE, LISTEN, IRCAM,
CIPIC/SCUT, …).  Implemented on top of :mod:`h5py` so that the
optional dependency scope is narrow — no ``netCDF4`` with its HDF5
system dependency required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..coords import azel_to_az_colat
from ..types import SphericalGrid
from .dataset import HRTFDataset


_SUPPORTED_CONVENTIONS = {"SimpleFreeFieldHRIR"}


def _require_h5py():
    try:
        import h5py  # noqa: F401
    except ImportError as exc:  # pragma: no cover — trivial guard
        raise ImportError(
            "SOFA I/O requires the optional h5py dependency. "
            "Install with `pip install h5py` or "
            "`pip install 'spherical-array-processing[hrtf]'`."
        ) from exc
    import h5py

    return h5py


def _decode(value) -> str:
    """Decode a SOFA attribute value to a plain string.

    SOFA attributes may be scalar bytes, numpy scalars, or arrays of
    single-element bytes depending on the writer.  Normalise to ``str``.
    """
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        item = value.flatten()[0]
        return _decode(item)
    if isinstance(value, (np.bytes_, np.str_)):
        return value.tolist() if isinstance(value, np.str_) else value.decode(
            "ascii", errors="replace"
        )
    return str(value)


def _build_source_grid(
    positions: NDArray[np.float64],
    units: str | None,
    positon_type: str | None,
) -> SphericalGrid:
    """Convert SOFA ``SourcePosition`` to a :class:`SphericalGrid`.

    ``SimpleFreeFieldHRIR`` stores source positions as
    ``(azimuth_deg, elevation_deg, radius_m)`` per row by default
    (``Type = "spherical"``).  Some writers use radians or the
    ``cartesian`` representation; we honour both.
    """
    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"SourcePosition must be (M, 3); got {arr.shape}"
        )
    pos_type = (positon_type or "spherical").lower()
    if pos_type == "cartesian":
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        az = np.arctan2(y, x) % (2.0 * np.pi)
        # Elevation from horizon, matching the SOFA convention.
        el = np.arcsin(np.clip(np.where(r > 0, z / np.maximum(r, 1e-30), 0.0), -1.0, 1.0))
        azimuth = az
        elevation = el
    elif pos_type == "spherical":
        units_norm = (units or "degree, degree, metre").split(",")
        az_unit = units_norm[0].strip().lower()
        el_unit = units_norm[1].strip().lower() if len(units_norm) > 1 else "degree"
        azimuth = arr[:, 0]
        elevation = arr[:, 1]
        if az_unit.startswith("deg"):
            azimuth = np.radians(azimuth)
        if el_unit.startswith("deg"):
            elevation = np.radians(elevation)
        azimuth = azimuth % (2.0 * np.pi)
    else:
        raise ValueError(
            "SourcePosition.Type must be 'spherical' or 'cartesian'; "
            f"got {positon_type!r}"
        )
    return SphericalGrid(
        azimuth=azimuth.astype(float),
        angle2=elevation.astype(float),
        convention="az_el",
    )


def load_sofa(
    path: str | Path,
    *,
    preserve_zero_delay: bool = False,
) -> HRTFDataset:
    """Load a ``SimpleFreeFieldHRIR`` SOFA file into an :class:`HRTFDataset`.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the ``.sofa`` file.
    preserve_zero_delay : bool, optional
        When ``False`` (default, backward compatible), an all-zero
        ``Data.Delay`` block in the file is folded into
        ``data_delay_samples=None`` because zero-delay and
        no-pre-roll are identical downstream.  When ``True``, the
        loader preserves the explicit zero block as a ``(2,)`` or
        ``(M, 2)`` array of zeros.  A file-side ``(1, 2)`` block is
        intentionally normalised to ``(2,)`` in the returned
        :class:`HRTFDataset`, and a subsequent
        :func:`save_sofa` call performs a byte-level round-trip
        (the output ``Data.Delay`` block has the exact shape seen
        in the input).  Non-zero delays are always preserved
        regardless of this flag.

    Returns
    -------
    HRTFDataset
        Container with HRIRs, sampling rate, source grid, ear
        positions (when available), and a metadata dict carrying the
        SOFA global attributes.

    Raises
    ------
    ImportError
        If :mod:`h5py` is not installed.
    ValueError
        If the file's ``SOFAConventions`` attribute is not one of the
        supported conventions.
    """
    h5py = _require_h5py()
    path = Path(path).expanduser()
    with h5py.File(str(path), "r") as fh:
        conventions = _decode(fh.attrs.get("SOFAConventions", "")).strip()
        if conventions and conventions not in _SUPPORTED_CONVENTIONS:
            raise ValueError(
                f"Unsupported SOFA convention {conventions!r}; this "
                f"reader handles only {sorted(_SUPPORTED_CONVENTIONS)}."
            )
        hrirs = np.asarray(fh["Data.IR"][...], dtype=float)
        fs_arr = np.asarray(fh["Data.SamplingRate"][...], dtype=float)
        fs = float(fs_arr.flat[0])
        source_positions = np.asarray(fh["SourcePosition"][...], dtype=float)
        pos_type = None
        pos_units = None
        if "SourcePosition" in fh:
            sp_attrs = fh["SourcePosition"].attrs
            pos_type = _decode(sp_attrs.get("Type", b"")) or None
            pos_units = _decode(sp_attrs.get("Units", b"")) or None
        if "ReceiverPosition" in fh:
            ear_positions = np.asarray(fh["ReceiverPosition"][...], dtype=float)
            # SOFA stores ReceiverPosition as (R, C, I) with C = 3 Cartesian
            # coordinates and I = 1 measurement index for static arrays.
            if ear_positions.ndim == 3 and ear_positions.shape == (2, 3, 1):
                ear_positions = ear_positions[:, :, 0]
            elif ear_positions.ndim == 2 and ear_positions.shape == (2, 3):
                pass
            else:
                ear_positions = None  # unusable for BiMagLS
            if ear_positions is not None:
                ear_positions = ear_positions.astype(float, copy=False)
        else:
            ear_positions = None
        # Data.Delay: pre-roll delays per ear (and optionally per-direction).
        # Accept the standard SOFA layouts (1, 2) and (M, 2), plus a plain
        # (2,) vector for permissive interoperability.  By default an
        # all-zero block folds to ``None`` (no pre-roll); ``preserve_zero_delay=True``
        # keeps the explicit zeros so ``save_sofa`` can round-trip the
        # file byte-faithfully.
        if "Data.Delay" in fh:
            delay_raw = np.asarray(fh["Data.Delay"][...], dtype=float)
            if delay_raw.ndim == 1 and delay_raw.shape == (2,):
                data_delay = delay_raw
            elif delay_raw.ndim == 2 and delay_raw.shape == (1, 2):
                data_delay = delay_raw[0]                      # (2,)
            elif delay_raw.ndim == 2 and delay_raw.shape[1] == 2:
                if delay_raw.shape[0] != hrirs.shape[0]:
                    raise ValueError(
                        "Data.Delay must have shape (1, 2) or "
                        f"({hrirs.shape[0]}, 2); got {delay_raw.shape}"
                    )
                data_delay = delay_raw                         # (M, 2)
            else:
                raise ValueError(
                    "Data.Delay must have shape (1, 2) or "
                    f"({hrirs.shape[0]}, 2); got {delay_raw.shape}"
                )
            if (
                not preserve_zero_delay
                and np.all(delay_raw == 0.0)
            ):
                data_delay = None
        else:
            data_delay = None
        attrs = {
            key: _decode(value) for key, value in fh.attrs.items()
        }

    grid = _build_source_grid(source_positions, pos_units, pos_type)
    if hrirs.ndim != 3 or hrirs.shape[1] != 2:
        raise ValueError(
            f"Data.IR must have shape (M, 2, N); got {hrirs.shape}"
        )
    if hrirs.shape[0] != grid.size:
        raise ValueError(
            "Data.IR and SourcePosition disagree on direction count "
            f"({hrirs.shape[0]} vs {grid.size})"
        )
    return HRTFDataset(
        hrirs=hrirs,
        fs=fs,
        source_grid=grid,
        ear_positions_m=ear_positions,
        data_delay_samples=data_delay,
        metadata=attrs,
    )


def _source_grid_to_spherical_deg(
    grid: SphericalGrid,
) -> NDArray[np.float64]:
    """Return ``(M, 3)`` SourcePosition in ``(azimuth_deg, elevation_deg, 1 m)``."""
    az = grid.azimuth
    el = grid.elevation  # property converts az_colat → elevation if needed
    radius = np.ones_like(az, dtype=float)
    return np.stack(
        [np.degrees(az), np.degrees(el), radius], axis=-1,
    ).astype(np.float64)


def save_sofa(
    dataset: HRTFDataset,
    path: str | Path,
    *,
    database_name: str | None = None,
    license_str: str = "",
) -> None:
    """Write an :class:`HRTFDataset` to a ``SimpleFreeFieldHRIR`` SOFA file.

    The on-disk layout follows AES69-2022 ``SimpleFreeFieldHRIR``:

    * ``Data.IR`` shape ``(M, 2, N)`` float64.
    * ``Data.SamplingRate`` scalar double, attribute ``Units="hertz"``.
    * ``Data.Delay`` float64 array ``(1, 2)`` for per-ear delays or
      ``(M, 2)`` for per-direction delays.  ``dataset.data_delay_samples
      is None`` writes the conventional all-zero ``(1, 2)`` block.
    * ``SourcePosition`` shape ``(M, 3)`` with ``Type="spherical"``
      and ``Units="degree, degree, metre"`` (azimuth, elevation,
      distance).
    * ``ReceiverPosition`` shape ``(2, 3, 1)`` with Cartesian ear
      positions in metres.  Zeros when the dataset has no ear
      positions.
    * ``ListenerPosition`` / ``EmitterPosition`` zero blocks to keep
      common SOFA readers happy.

    Extra keys from ``dataset.metadata`` are written as global string
    attributes unless they collide with the structural keys above.

    Parameters
    ----------
    dataset : HRTFDataset
    path : str or Path
    database_name : str, optional
        Overrides the ``DatabaseName`` global attribute.  Defaults to
        ``dataset.metadata["DatabaseName"]`` if present, else
        ``"spherical-array-processing"``.
    license_str : str, optional
        Value for the ``License`` global attribute.

    Raises
    ------
    ImportError
        If :mod:`h5py` is not installed.
    """
    h5py = _require_h5py()
    path = Path(path).expanduser()
    hrirs = np.ascontiguousarray(dataset.hrirs, dtype=np.float64)
    if hrirs.ndim != 3 or hrirs.shape[1] != 2:
        raise ValueError(
            f"dataset.hrirs must have shape (M, 2, N); got {hrirs.shape}"
        )
    m, _, n_taps = hrirs.shape
    if dataset.source_grid.size != m:
        raise ValueError(
            "source_grid.size does not match dataset.hrirs.shape[0] "
            f"({dataset.source_grid.size} vs {m})"
        )

    source_positions = _source_grid_to_spherical_deg(dataset.source_grid)

    if dataset.ear_positions_m is None:
        ear_positions = np.zeros((2, 3), dtype=np.float64)
    else:
        ear_positions = np.asarray(dataset.ear_positions_m, dtype=np.float64)
        if ear_positions.shape != (2, 3):
            raise ValueError(
                f"ear_positions_m must be (2, 3); got {ear_positions.shape}"
            )
    # SOFA layout: (R=2, C=3, I=1).
    ear_positions_sofa = ear_positions[:, :, None]

    with h5py.File(str(path), "w") as fh:
        fh.attrs["Conventions"] = np.bytes_("SOFA")
        fh.attrs["Version"] = np.bytes_("2.1")
        fh.attrs["SOFAConventions"] = np.bytes_("SimpleFreeFieldHRIR")
        fh.attrs["SOFAConventionsVersion"] = np.bytes_("1.0")
        fh.attrs["APIName"] = np.bytes_("spherical-array-processing")
        fh.attrs["APIVersion"] = np.bytes_("0.4")
        fh.attrs["DataType"] = np.bytes_("FIR")
        fh.attrs["RoomType"] = np.bytes_("free field")
        fh.attrs["Title"] = np.bytes_("HRIR dataset")
        fh.attrs["AuthorContact"] = np.bytes_(
            str(dataset.metadata.get("AuthorContact", ""))
        )
        fh.attrs["Organization"] = np.bytes_(
            str(dataset.metadata.get("Organization", ""))
        )
        fh.attrs["ListenerShortName"] = np.bytes_(
            str(dataset.metadata.get("ListenerShortName", ""))
        )
        fh.attrs["DateCreated"] = np.bytes_("")
        fh.attrs["DateModified"] = np.bytes_("")
        fh.attrs["License"] = np.bytes_(license_str)
        db_name = (
            database_name
            if database_name is not None
            else dataset.metadata.get("DatabaseName", "spherical-array-processing")
        )
        fh.attrs["DatabaseName"] = np.bytes_(str(db_name))

        # Extra metadata → string attributes (string type only).
        reserved = {
            "Conventions", "Version", "SOFAConventions",
            "SOFAConventionsVersion", "APIName", "APIVersion",
            "DataType", "RoomType", "Title", "AuthorContact",
            "Organization", "ListenerShortName", "DateCreated",
            "DateModified", "License", "DatabaseName",
        }
        for key, value in dataset.metadata.items():
            if key in reserved:
                continue
            fh.attrs[key] = np.bytes_(str(value))

        fh.create_dataset("Data.IR", data=hrirs)
        fs_ds = fh.create_dataset(
            "Data.SamplingRate",
            data=np.array([float(dataset.fs)], dtype=np.float64),
        )
        fs_ds.attrs["Units"] = np.bytes_("hertz")
        # Data.Delay preserves per-ear (and optionally per-direction)
        # pre-roll delays from the dataset.  ``None`` → zero block.
        if dataset.data_delay_samples is None:
            delay_block = np.zeros((1, 2), dtype=np.float64)
        else:
            delay_arr = np.asarray(
                dataset.data_delay_samples, dtype=np.float64,
            )
            if delay_arr.shape == (2,):
                delay_block = delay_arr[None, :]  # (1, 2)
            elif delay_arr.shape == (m, 2):
                delay_block = delay_arr           # (M, 2)
            else:
                raise ValueError(
                    "data_delay_samples must have shape (2,) or "
                    f"({m}, 2); got {delay_arr.shape}"
                )
        fh.create_dataset("Data.Delay", data=delay_block)
        sp_ds = fh.create_dataset("SourcePosition", data=source_positions)
        sp_ds.attrs["Type"] = np.bytes_("spherical")
        sp_ds.attrs["Units"] = np.bytes_("degree, degree, metre")
        rp_ds = fh.create_dataset(
            "ReceiverPosition", data=ear_positions_sofa,
        )
        rp_ds.attrs["Type"] = np.bytes_("cartesian")
        rp_ds.attrs["Units"] = np.bytes_("metre")
        lp_ds = fh.create_dataset(
            "ListenerPosition",
            data=np.zeros((1, 3), dtype=np.float64),
        )
        lp_ds.attrs["Type"] = np.bytes_("cartesian")
        lp_ds.attrs["Units"] = np.bytes_("metre")
        lv_ds = fh.create_dataset(
            "ListenerView",
            data=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        )
        lv_ds.attrs["Type"] = np.bytes_("cartesian")
        lv_ds.attrs["Units"] = np.bytes_("metre")
        lu_ds = fh.create_dataset(
            "ListenerUp",
            data=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        )
        ep_ds = fh.create_dataset(
            "EmitterPosition",
            data=np.zeros((1, 3, 1), dtype=np.float64),
        )
        ep_ds.attrs["Type"] = np.bytes_("cartesian")
        ep_ds.attrs["Units"] = np.bytes_("metre")


__all__ = ["load_sofa", "save_sofa"]
