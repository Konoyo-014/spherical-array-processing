import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spherical_array_processing.repro.politis import (
    arraySHTfiltersTheory_radInverse,
    arraySHTfiltersTheory_regLS,
    arraySHTfiltersTheory_softLim,
    arraySHTfiltersMeas_regLS,
    arraySHTfiltersMeas_regLSHD,
    arraySHTfilters_diffEQ,
    beamWeightsCardioid2Differential,
    beamWeightsCardioid2Spherical,
    beamWeightsDifferential2Spherical,
    getDiffuseness_CMD,
    getDiffuseness_DPV,
    getDiffuseness_IE,
    getDiffuseness_SV,
    getDiffuseness_TV,
    beam_weights_cardioid_to_differential,
    beam_weights_differential_to_spherical,
    beam_weights_pressure_velocity,
    beamWeightsDolphChebyshev2Spherical,
    beamWeightsFromFunction,
    beamWeightsPressureVelocity,
    differentialGains,
    beamWeightsLinear2Spherical,
    beamWeightsTorus2Spherical,
    beamWeightsVelocityPatterns,
    computeVelCoeffsMtx,
    diffCoherence,
    getDiffCohMtxMeas,
    getDiffCohMtxTheory,
    evaluateSHTfilters,
    extractAxisCoeffs,
    plotAxisymPatternFromCoeffs,
    returnChebyPolyCoeffs,
    returnLegePolyCoeffs,
    sorted_eig,
    sparse_solver_irls,
    sphiPMMW,
    sphESPRIT,
    sphIntensityHist,
    sphNullformer_diff,
    sphNullformer_pwd,
    sphLCMV,
    sphMUSIC,
    sphMVDR,
    sphMVDRmap,
    sphPWDmap,
    sphSRmap,
    sph_array_alias_lim,
    sph_array_noise,
    sph_array_noise_threshold,
)
from spherical_array_processing.repro.rafaely import (
    chebyshev_coefficients,
    derivative_ph,
    equiangle_sampling as rafaely_equiangle_sampling,
    gaussian_sampling as rafaely_gaussian_sampling,
    legendre_coefficients,
    platonic_solid,
    sh2,
    uniform_sampling as rafaely_uniform_sampling,
    wigner_d_matrix,
)


def test_sorted_eig_descend():
    x = np.diag([1.0, 3.0, 2.0])
    v, s = sorted_eig(x, "descend")
    vals = np.diag(s).real
    assert np.allclose(vals, [3.0, 2.0, 1.0])
    assert v.shape == (3, 3)


def test_politis_noise_functions_shapes():
    f = np.linspace(100, 1000, 8)
    g2, g2_lin = sph_array_noise(0.042, 32, 3, "rigid", f)
    th = sph_array_noise_threshold(0.042, 32, 10.0, 3, "rigid")
    assert g2.shape == (8, 4)
    assert g2_lin.shape == (8, 3)
    assert th.shape == (3,)


def test_politis_alias_limit_outputs():
    dirs = np.stack([np.linspace(0, 2 * np.pi, 32, endpoint=False), np.zeros(32)], axis=1)
    f_alias, cond = sph_array_alias_lim(0.042, 32, 4, dirs)
    assert f_alias.shape == (3,)
    assert cond.ndim == 1


def test_differential_to_spherical_conversion_runs():
    a = beam_weights_cardioid_to_differential(2)
    b = beam_weights_differential_to_spherical(a)
    assert b.shape == (3,)
    assert np.isfinite(b).all()


def test_rafaely_math_shapes():
    th = np.array([0.1, 0.2, -0.3])
    ph = np.array([0.2, -0.4, 0.7])
    y = sh2(2, th, ph)
    assert y.shape == (9, 3)
    d = derivative_ph(y[:, 0])
    assert d.shape == (9,)
    w = wigner_d_matrix(2, 0.1, 0.2, 0.3)
    assert w.shape == (9, 9)
    a1, th1, ph1 = rafaely_equiangle_sampling(2)
    a2, th2, ph2 = rafaely_gaussian_sampling(2)
    a3, th3, ph3 = rafaely_uniform_sampling(2)
    assert a1.shape == th1.shape == ph1.shape
    assert a2.shape == th2.shape == ph2.shape
    assert a3.shape == th3.shape == ph3.shape


def test_polynomial_coeff_helpers():
    tc = chebyshev_coefficients(2)
    lc = legendre_coefficients(2)
    # T2 = 2x^2 - 1  (descending order)
    assert np.allclose(tc, [2.0, 0.0, -1.0])
    # P2 = (3x^2 -1)/2, coefficients arranged as MATLAB helper [x^2, x^1, x^0]
    assert np.allclose(lc, [1.5, 0.0, -0.5])


def test_platonic_solid_radius():
    v, f = platonic_solid(4, radius=2.0)
    r = np.linalg.norm(v, axis=1)
    assert np.allclose(r, 2.0)
    assert f.ndim == 2


def test_politis_wrapper_doa_and_diffuseness_interfaces():
    order = 1
    n = (order + 1) ** 2
    cov = np.eye(n, dtype=complex)
    grid_dirs = np.stack([np.linspace(0, 2 * np.pi, 30, endpoint=False), np.zeros(30)], axis=1)
    p_pwd, dirs_pwd = sphPWDmap(cov, grid_dirs, n_src=2)
    p_music, dirs_music = sphMUSIC(cov + 0.01 * np.eye(n), grid_dirs, n_src=1)
    p_mvdr, dirs_mvdr = sphMVDRmap(cov + 0.1 * np.eye(n), grid_dirs, n_src=1)
    w_mvdr = sphMVDR(cov, np.array([0.0, 0.0]))
    w_lcmv = sphLCMV(cov, np.array([[0.0, 0.0]]), np.array([1.0]))
    assert p_pwd.shape == (30,)
    assert p_music.shape == (30,)
    assert p_mvdr.shape == (30,)
    assert dirs_pwd.shape == (2, 2)
    assert dirs_music.shape == (1, 2)
    assert dirs_mvdr.shape == (1, 2)
    assert w_mvdr.shape == (n,)
    assert w_lcmv.shape == (n,)
    assert 0.0 <= getDiffuseness_IE(np.eye(4)) <= 1.0
    i_vecs = np.random.default_rng(0).normal(size=(100, 3))
    assert 0.0 <= getDiffuseness_TV(i_vecs) <= 1.0
    assert 0.0 <= getDiffuseness_SV(i_vecs) <= 1.0
    cmd, evals = getDiffuseness_CMD(cov)
    assert 0.0 <= cmd <= 1.0
    assert evals.ndim == 1
    assert 0.0 <= getDiffuseness_DPV(cov) <= 1.0


def test_politis_nullformers_and_pv_matrix():
    w_pwd = sphNullformer_pwd(1, np.array([[0.0, 0.0], [np.pi / 2, 0.0]]))
    w_diff = sphNullformer_diff(1, np.array([[0.0, 0.0]]))
    m_real = beam_weights_pressure_velocity("real")
    m_cplx = beam_weights_pressure_velocity("complex")
    assert w_pwd.shape == (4, 2)
    assert w_diff.shape == (4,)
    assert m_real.shape == (4, 4)
    assert m_cplx.shape == (4, 4)


def test_politis_diffcoh_and_encoding_filters_shapes():
    # synthetic array response [bins,mics,grid]
    H = np.ones((5, 4, 10), dtype=np.complex128)
    d_meas = getDiffCohMtxMeas(H)
    dirs = np.stack([np.linspace(0, 2 * np.pi, 4, endpoint=False), np.zeros(4)], axis=1)
    d_theory = getDiffCohMtxTheory(dirs, "rigid", 0.042, 3, np.linspace(100, 1000, 5))
    h_rad_f, h_rad_t = arraySHTfiltersTheory_radInverse(0.042, 32, 3, 128, 48000, 15.0)
    h_soft_f, h_soft_t = arraySHTfiltersTheory_softLim(0.042, 32, 3, 128, 48000, 15.0)
    h_reg_f, h_reg_t = arraySHTfiltersTheory_regLS(0.042, dirs, 1, 64, 48000, 15.0)
    H_meas = np.ones((64, 4, 20), dtype=np.complex128)
    grid_meas = np.stack([np.linspace(0, 2 * np.pi, 20, endpoint=False), np.zeros(20)], axis=1)
    h_mreg_f, h_mreg_t = arraySHTfiltersMeas_regLS(H_meas, 1, grid_meas, None, 64, 15.0)
    h_mreg_hd_f, h_mreg_hd_t = arraySHTfiltersMeas_regLSHD(H_meas, 1, grid_meas, None, 64, 15.0)
    d_meas_match = np.repeat(np.eye(4, dtype=np.complex128)[:, :, None], h_mreg_f.shape[2], axis=2)
    h_diffeq = arraySHTfilters_diffEQ(h_mreg_f, d_meas_match, np.array([500.0, 700.0, 900.0]), 48000)
    assert d_meas.shape == (4, 4, 5)
    assert d_theory.shape == (4, 4, 5)
    assert h_rad_f.shape == (65, 4)
    assert h_rad_t.shape == (128, 4)
    assert h_soft_f.shape == (65, 4)
    assert h_soft_t.shape == (128, 4)
    assert h_reg_f.shape == (4, 4, 33)
    assert h_reg_t.shape == (4, 4, 64)
    assert h_mreg_f.shape == (4, 4, 33)
    assert h_mreg_t.shape == (4, 4, 64)
    assert h_mreg_hd_f.shape == (4, 4, 33)
    assert h_mreg_hd_t.shape == (4, 4, 64)
    assert h_diffeq.shape == h_mreg_f.shape


def test_politis_sparse_recovery_wrappers():
    rng = np.random.default_rng(0)
    M, K, T = 4, 12, 6
    A = rng.normal(size=(M, K))
    X_true = np.zeros((K, T))
    X_true[[2, 7], :] = rng.normal(size=(2, T))
    Y = A @ X_true
    X, D, e = sparse_solver_irls(0.8, A, Y, beta=0.2, termination_value=1e-6, max_iterations=3)
    grid_dirs = np.stack([np.linspace(0, 2 * np.pi, K, endpoint=False), np.zeros(K)], axis=1)
    P_sr, est_sr = sphSRmap(Y, 0.8, A, 0.2, 1e-6, 3, grid_dirs, nSrc=2)
    i_xyz = rng.normal(size=(30, 3))
    hist, est_i = sphIntensityHist(i_xyz, np.stack([np.linspace(0, 2 * np.pi, 40, endpoint=False), np.zeros(40)], axis=1), nSrc=2)
    assert X.shape[0] == K
    assert D.shape == (K, M)
    assert e.shape == (K,)
    assert P_sr.shape == (K,)
    assert est_sr.shape == (2, 2)
    assert hist.shape == (40,)
    assert est_i.shape == (2, 2)
    Phi_x = np.eye(M, dtype=complex) * 2.0
    Phi_n = np.eye(M, dtype=complex) * 0.1
    src_dirs = np.array([[0.0, 0.0], [np.pi / 2, 0.0]])
    W_pmmw, Pd_est, Ps_est = sphiPMMW(Phi_x, Phi_n, src_dirs)
    assert W_pmmw.shape == (M, 2)
    assert np.isscalar(Pd_est)
    assert Ps_est.shape == (2,)


def test_politis_more_weight_and_eval_helpers():
    d = beamWeightsDolphChebyshev2Spherical(2, "sidelobe", 0.1)
    b_lin = beamWeightsLinear2Spherical(np.array([1.0, 0.5, 0.25]))
    b_fun = beamWeightsFromFunction(lambda az, el: np.ones_like(az), order=1)
    assert d.shape == (3,)
    assert b_lin.shape == (3,)
    assert b_fun.shape == (4,)

    n_bins, n_mics, n_grid = 5, 4, 20
    n_sh = 4
    M = np.zeros((n_sh, n_mics, n_bins), dtype=np.complex128)
    for k in range(n_bins):
        M[:, :, k] = 0.5 * np.ones((n_sh, n_mics))
    H = np.ones((n_bins, n_mics, n_grid), dtype=np.complex128)
    Y_grid = np.ones((n_grid, n_sh), dtype=np.float64)
    csh, lsh, wng = evaluateSHTfilters(M, H, fs=48000, Y_grid=Y_grid)
    assert csh.shape == (n_bins, 2)
    assert lsh.shape == (n_bins, 2)
    assert wng.shape == (n_bins, 1)


def test_politis_poly_axisym_helpers():
    tc = returnChebyPolyCoeffs(3)
    lc = returnLegePolyCoeffs(3)
    assert tc.shape == (4, 1)
    assert lc.shape == (4, 1)
    b_torus = beamWeightsTorus2Spherical(2)
    assert b_torus.shape == (3,)
    coeffs = np.arange(16, dtype=float)
    a0 = extractAxisCoeffs(coeffs)
    assert a0.shape == (4,)
    ax = plotAxisymPatternFromCoeffs(np.array([1.0, 0.5]))
    assert ax is not None
    plt.close(ax.figure)


def test_politis_remaining_aliases_and_numerical_helpers():
    a = beamWeightsCardioid2Differential(2)
    b = beamWeightsCardioid2Spherical(2)
    c = beamWeightsDifferential2Spherical(a)
    Mpv = beamWeightsPressureVelocity("real")
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)
    assert Mpv.shape == (4, 4)

    Axyz = computeVelCoeffsMtx(1)
    assert Axyz.shape == (9, 4, 3)
    vel = beamWeightsVelocityPatterns(np.array([1.0, 0.5]), np.array([0.0, 0.0]), A_xyz=Axyz, basisType="real")
    assert vel.shape == (9, 3)
    # diffuse coherence numerical integration
    k = np.linspace(0.1, 2.0, 4)
    a_nm = np.zeros(4, dtype=complex)
    b_nm = np.zeros(4, dtype=complex)
    a_nm[0] = 1.0
    b_nm[0] = 1.0
    g = diffCoherence(k, np.array([0, 0, 0]), np.array([0.02, 0, 0]), a_nm, b_nm)
    assert g.shape == (4,)
    assert np.all(np.isfinite(g))

    # ESPRIT shape test with random subspace
    rng = np.random.default_rng(0)
    Us = rng.normal(size=(9, 2)) + 1j * rng.normal(size=(9, 2))
    dirs = sphESPRIT(Us)
    assert dirs.shape == (2, 2)


def test_politis_differential_gains_table():
    gains = differentialGains()
    assert set(gains.keys()) == {"cardioid", "supercardioid", "hypercardioid"}
    for family, table in gains.items():
        assert set(table.keys()) == {1, 2, 3, 4}
        for order, coeffs in table.items():
            assert coeffs.shape == (order + 1,)
            assert np.all(np.isfinite(coeffs))
        if family in {"cardioid", "supercardioid"}:
            for coeffs in table.values():
                assert np.isclose(np.sum(coeffs), 1.0, atol=1e-3)
