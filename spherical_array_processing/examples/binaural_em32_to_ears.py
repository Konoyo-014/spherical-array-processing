"""End-to-end example: Eigenmike-32 → SH encoding → MagLS binaural render.

Simulates a plane-wave source captured by a rigid-sphere Eigenmike-32,
encodes the microphone signals to a 4th-order real ACN/SN3D ambisonic
signal with Tikhonov-regularised radial equalisation, then renders the
result to binaural via Magnitude Least Squares using a *synthetic*
HRTF (delay + soft shadow) so the example has no external data
dependency.

Run after installing the wheel::

    python -m spherical_array_processing.examples.binaural_em32_to_ears

Or from the repo source tree, using the back-compatible top-level
shim::

    python examples/binaural_em32_to_ears.py
"""

from __future__ import annotations

import numpy as np

import spherical_array_processing as sap
from spherical_array_processing.array import em32_eigenmike, fibonacci_grid
from spherical_array_processing.binaural import magls_binaural_filters
from spherical_array_processing.coords import unit_sph_to_cart
from spherical_array_processing.encoding import apply_radial_equalizer, radial_equalizer
from spherical_array_processing.hrtf import HRTFDataset
from spherical_array_processing.sh import direct_sht, matrix as sh_matrix  # noqa: F401 — kept for example docs
from spherical_array_processing.stft import istft, stft
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


def _synthetic_sofa(
    grid: SphericalGrid, fs: float, n_taps: int = 256
) -> HRTFDataset:
    """Build a toy HRTF dataset (delay + head shadow) so the example
    runs without an external ``.sofa`` file."""
    u = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)
    ear_positions = np.array([[-0.09, 0.0, 0.0], [0.09, 0.0, 0.0]])
    tau = u @ ear_positions.T / 343.0  # (G, 2)
    shadow = np.stack(
        [
            1.0 - 0.5 * np.clip(u[:, 0], 0.0, 1.0),   # left ear
            1.0 - 0.5 * np.clip(-u[:, 0], 0.0, 1.0),  # right ear
        ],
        axis=1,
    )
    freqs = np.fft.rfftfreq(n_taps, d=1.0 / fs)
    hrtfs = shadow[None, :, :] * np.exp(
        -2j * np.pi * freqs[:, None, None] * tau[None, :, :]
    )  # (F, G, 2)
    # Inverse-FFT back to HRIRs shaped (G, 2, n_taps).
    hrirs = np.fft.irfft(hrtfs, n=n_taps, axis=0).transpose(1, 2, 0)
    return HRTFDataset(
        hrirs=hrirs,
        fs=fs,
        source_grid=grid,
        ear_positions_m=ear_positions,
        metadata={"DatabaseName": "synthetic"},
    )


def run_example(
    source_az_deg: float = 60.0,
    source_col_deg: float = 70.0,
    fs: float = 16000.0,
    duration_s: float = 0.5,
    max_order: int = 4,
) -> dict:
    """Run the full Eigenmike → MagLS binaural pipeline.  Returns the
    intermediate signals and a few sanity metrics so tests / notebooks
    can inspect the output.
    """
    # --- 1.  Simulate EM32 plane-wave response at the source direction.
    em32 = em32_eigenmike()
    src = SphericalGrid(
        azimuth=[np.radians(source_az_deg)],
        angle2=[np.radians(source_col_deg)],
        convention="az_colat",
    )
    fft_len = 512
    _, array_tf = sap.array.simulate_sh_array_response(
        fft_len, fs, em32, src, max_order=12, array_type="rigid"
    )  # (F, M, 1)

    # --- 2.  Build a time-domain source signal and convolve per mic.
    n = int(fs * duration_s)
    rng = np.random.default_rng(0)
    src_signal = rng.normal(size=n) * np.hanning(n) * 0.2
    # Apply frequency-dependent array transfer function via overlap-add.
    # Use ceil framing + a padded buffer so any tail shorter than one
    # hop still gets convolved and written back into *n* samples.
    frame = fft_len
    hop = frame // 2
    padded_len = int(np.ceil(n / hop)) * hop + frame
    src_padded = np.zeros(padded_len, dtype=float)
    src_padded[:n] = src_signal
    n_frames = (padded_len - frame) // hop + 1
    mic_buffer = np.zeros((em32.sensor_grid.size, padded_len), dtype=float)
    win = np.hanning(frame)
    for k in range(n_frames):
        start = k * hop
        chunk = src_padded[start : start + frame] * win
        spec_in = np.fft.rfft(chunk)
        for m in range(em32.sensor_grid.size):
            filtered = spec_in * array_tf[:, m, 0]
            mic_buffer[m, start : start + frame] += np.fft.irfft(
                filtered, n=frame
            )
    mic_signal = mic_buffer[:, :n]

    # --- 3.  Encode to 4th-order ambi via SHT + radial equalisation.
    spec = SHBasisSpec(max_order=max_order, basis="real")
    y_mic = sh_matrix(spec, em32.sensor_grid)
    freqs_stft, _, mic_stft = stft(mic_signal, fs, nperseg=frame)
    # (F, M, T) → SH per frame per bin.
    n_bins, n_mics, n_t = mic_stft.shape
    ambi_stft = np.einsum("fmt,mq->fqt", mic_stft, np.linalg.pinv(y_mic.T))
    kr_vec = sap.acoustics.kr(freqs_stft, radius_m=em32.radius_m)
    eq = radial_equalizer(
        max_order, kr_vec, array_type="rigid",
        regularization="tikhonov", tikhonov_lambda=0.01,
    )
    ambi_stft = apply_radial_equalizer(
        ambi_stft, eq, freq_axis=0, coeff_axis=1
    )

    # --- 4.  Build MagLS binaural filters on a synthetic HRTF set.
    hrtf_grid = fibonacci_grid(180)
    hrtf_ds = _synthetic_sofa(hrtf_grid, fs, n_taps=fft_len)
    hrtf_freqs, hrtfs = hrtf_ds.to_frequency_domain(fft_len)
    # Re-sample MagLS onto the STFT grid: their frequencies match because
    # we used the same FFT length and ``fs``.
    assert np.allclose(hrtf_freqs, freqs_stft, atol=1e-6)
    bin_filters = magls_binaural_filters(
        hrtfs, hrtf_freqs, hrtf_grid, max_order,
        f_cut_hz=1500.0, n_iterations=10,
    )  # (F, Q, 2)

    # --- 5.  Apply the filters per bin and inverse-STFT back to time.
    binaural_stft = np.einsum("fqt,fqe->fet", ambi_stft, bin_filters)
    _, binaural = istft(binaural_stft, fs, nperseg=frame)

    # --- Sanity check: left / right ear energy ratio should favour the
    # ear nearest the source direction.
    left_energy = float(np.mean(binaural[0] ** 2))
    right_energy = float(np.mean(binaural[1] ** 2))
    return {
        "binaural": np.asarray(binaural, dtype=float),
        "ambi_stft": ambi_stft,
        "left_energy": left_energy,
        "right_energy": right_energy,
        "source_az_deg": source_az_deg,
        "source_col_deg": source_col_deg,
        "fs": fs,
    }


def main() -> None:
    """CLI entry point used by ``python -m`` invocations."""
    out = run_example()
    print(
        f"source at az={out['source_az_deg']}°, colat={out['source_col_deg']}° — "
        f"left ear energy {out['left_energy']:.3e}, right ear energy "
        f"{out['right_energy']:.3e}"
    )


if __name__ == "__main__":  # pragma: no cover — manual invocation
    main()
