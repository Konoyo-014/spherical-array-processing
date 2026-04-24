from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from scipy.special import lpmv

from ...array.sampling import get_tdesign_fallback
from ...coords import cart_to_sph, sph_to_cart, unit_sph_to_cart
from ...sh import (
    complex_to_real_coeffs,
    direct_sht,
    matrix as sh_matrix,
    real_to_complex_coeffs,
    replicate_per_order,
)
from ...types import SHBasisSpec, SphericalGrid
from ..politis import (
    checkCondNumberSHT,
    conjCoeffs,
    gaunt_mtx,
    getSH,
    getTdesign,
    rotateAxisCoeffs,
    unitSph2cart,
)
from ..rafaely.math import wigner_d_matrix

_ROOT = Path(__file__).resolve().parents[3]
_FLIEGE_MAT = _ROOT / "src" / "Spherical-Harmonic-Transform" / "fliegeMaierNodes_1_30.mat"


def grid2dirs(aziRes: float, polarRes: float, POLAR_OR_ELEV: int = 1, ZEROED_OR_CENTERED: int = 1) -> np.ndarray:
    if (360 % aziRes) != 0 or (180 % polarRes) != 0:
        raise ValueError("azimuth/polar resolution should divide 360/180 exactly")

    if ZEROED_OR_CENTERED:
        phi = np.deg2rad(np.arange(0.0, 360.0, aziRes))
    else:
        phi = np.deg2rad(np.arange(-180.0, 180.0, aziRes))

    if POLAR_OR_ELEV:
        theta = np.deg2rad(np.arange(0.0, 180.0 + polarRes, polarRes))
    else:
        theta = np.deg2rad(np.arange(-90.0, 90.0 + polarRes, polarRes))

    out = []
    for i in range(1, len(theta) - 1):
        for p in phi:
            out.append([p, theta[i]])
    if POLAR_OR_ELEV:
        out = [[0.0, 0.0]] + out + [[0.0, math.pi]]
    else:
        out = [[0.0, -math.pi / 2.0]] + out + [[0.0, math.pi / 2.0]]
    return np.asarray(out, dtype=float)


def Fdirs2grid(W: ArrayLike, aziRes: float, polarRes: float, CLOSED: int = 0) -> np.ndarray:
    w = np.asarray(W)
    if w.ndim == 1:
        w = w[:, None]
    if (360 % aziRes) != 0 or (180 % polarRes) != 0:
        raise ValueError("azimuth/polar resolution should divide 360/180 exactly")

    nphi = int(round(360 / aziRes))
    ntheta = int(round(180 / polarRes + 1))
    nf = w.shape[1]
    out = np.zeros((nphi, ntheta, nf), dtype=w.dtype)
    for i in range(nf):
        out[:, 1:-1, i] = w[1:-1, i].reshape(nphi, ntheta - 2, order="F")
        out[:, 0, i] = w[0, i]
        out[:, -1, i] = w[-1, i]

    if nf != 1:
        out = np.transpose(out, (1, 0, 2))
    else:
        out = out[:, :, 0].T

    if CLOSED:
        out = np.concatenate([out, out[:, :1, ...]], axis=1)
    return out


def directSHT(N: int, F: ArrayLike, dirs: ArrayLike, basisType: str, weights: ArrayLike | None = None):
    y = getSH(N, dirs, basisType)
    f = np.asarray(F)
    if f.ndim == 1:
        f = f[:, None]
    npoints = y.shape[0]
    if f.shape[0] != npoints:
        raise ValueError("F rows must match dirs")
    if weights is None:
        fn = (4.0 * np.pi / npoints) * y.conj().T @ f
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        fn = y.conj().T @ (w[:, None] * f)
    return fn, y


def leastSquaresSHT(N: int, F: ArrayLike, dirs: ArrayLike, basisType: str, weights: ArrayLike | None = None):
    y = getSH(N, dirs, basisType)
    f = np.asarray(F)
    if f.ndim == 1:
        f = f[:, None]
    if weights is None:
        fn = np.linalg.pinv(y) @ f
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        yw = y.conj().T @ (w[:, None] * y)
        fn = np.linalg.solve(yw, y.conj().T @ (w[:, None] * f))
    return fn, y


def inverseSHT(F_N: ArrayLike, dirs: ArrayLike, basisType: str):
    fn = np.asarray(F_N)
    if fn.ndim == 1:
        fn = fn[:, None]
    n = int(round(np.sqrt(fn.shape[0]) - 1))
    y = getSH(n, dirs, basisType)
    return y @ fn


def complex2realSHMtx(N: int) -> np.ndarray:
    n = int(N)
    t = np.zeros(((n + 1) ** 2, (n + 1) ** 2), dtype=np.complex128)
    t[0, 0] = 1.0
    idx = 1
    for band in range(1, n + 1):
        m = np.arange(1, band + 1)
        diag_t = np.concatenate([1j * np.ones(band), [np.sqrt(2) / 2], ((-1) ** m)]).astype(np.complex128) / np.sqrt(2)
        adiag_t = np.concatenate([-1j * ((-1) ** m[::-1]), [np.sqrt(2) / 2], np.ones(band)]).astype(np.complex128) / np.sqrt(2)
        temp = np.diag(diag_t) + np.fliplr(np.diag(adiag_t))
        sl = slice(idx, idx + 2 * band + 1)
        t[sl, sl] = temp
        idx += 2 * band + 1
    return t


def real2complexSHMtx(N: int) -> np.ndarray:
    n = int(N)
    t = np.zeros(((n + 1) ** 2, (n + 1) ** 2), dtype=np.complex128)
    t[0, 0] = 1.0
    idx = 1
    for band in range(1, n + 1):
        m = np.arange(1, band + 1)
        diag_t = np.concatenate([-1j * np.ones(band), [np.sqrt(2) / 2], ((-1) ** m)]).astype(np.complex128) / np.sqrt(2)
        adiag_t = np.concatenate([np.ones(band), [np.sqrt(2) / 2], 1j * ((-1) ** m)]).astype(np.complex128) / np.sqrt(2)
        temp = np.diag(diag_t) + np.fliplr(np.diag(adiag_t))
        sl = slice(idx, idx + 2 * band + 1)
        t[sl, sl] = temp
        idx += 2 * band + 1
    return t


def complex2realCoeffs(C_N: ArrayLike) -> np.ndarray:
    c = np.asarray(C_N, dtype=np.complex128)
    if c.ndim == 1:
        c = c[:, None]
    n = int(round(np.sqrt(c.shape[0]) - 1))
    t = complex2realSHMtx(n)
    return np.conj(t) @ c


def real2complexCoeffs(R_N: ArrayLike) -> np.ndarray:
    r = np.asarray(R_N)
    if r.ndim == 1:
        r = r[:, None]
    n = int(round(np.sqrt(r.shape[0]) - 1))
    t = real2complexSHMtx(n)
    return np.conj(t) @ r


def euler2rotationMatrix(alpha: float, beta: float, gamma: float, convention: str = "zyz") -> np.ndarray:
    rx = lambda th: np.array([[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]], dtype=float)
    ry = lambda th: np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]], dtype=float)
    rz = lambda th: np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]], dtype=float)

    m = {"x": rx, "y": ry, "z": rz}
    r1 = m[convention[0]](alpha)
    r2 = m[convention[1]](beta)
    r3 = m[convention[2]](gamma)
    return r3 @ r2 @ r1


def getSHrotMtx(Rxyz: ArrayLike, L: int, basisType: str = "real") -> np.ndarray:
    rxyz = np.asarray(Rxyz, dtype=float)
    if rxyz.shape != (3, 3):
        raise ValueError("Rxyz must be 3x3")
    l = int(L)

    # Numerical construction through rotated samples, robust across real/complex bases.
    g = get_tdesign_fallback(order=max(2 * l + 2, 4), n_points=max(128, 4 * (l + 1) ** 2))
    dirs_incl = np.column_stack([g.azimuth, g.colatitude])
    y = getSH(l, dirs_incl, basisType)  # [G,C]

    xyz = unit_sph_to_cart(g.azimuth, g.elevation, convention="az_el")
    xyz_rot = xyz @ rxyz.T
    az, el, _ = cart_to_sph(xyz_rot[:, 0], xyz_rot[:, 1], xyz_rot[:, 2], convention="az_el")
    dirs_rot_incl = np.column_stack([az, (np.pi / 2.0) - el])
    y_rot = getSH(l, dirs_rot_incl, basisType)

    return np.linalg.pinv(y) @ y_rot


def gaunt_mtx_fast(N1: int, N2: int, N: int) -> np.ndarray:
    return gaunt_mtx(N1, N2, N)


def getRealGauntMtx(cGmtx: ArrayLike) -> np.ndarray:
    c = np.asarray(cGmtx, dtype=np.complex128)
    n1 = int(round(np.sqrt(c.shape[0]) - 1))
    n2 = int(round(np.sqrt(c.shape[1]) - 1))
    n = int(round(np.sqrt(c.shape[2]) - 1))
    deg = min(21, max(1, 2 * (n1 + n2 + n)))
    _, dirs = getTdesign(deg)
    incl_dirs = np.column_stack([dirs[:, 0], (np.pi / 2.0) - dirs[:, 1]])
    g = SphericalGrid(azimuth=incl_dirs[:, 0], angle2=incl_dirs[:, 1], convention="az_colat")
    y1 = np.asarray(sh_matrix(SHBasisSpec(max_order=n1, basis="real"), g))
    y2 = np.asarray(sh_matrix(SHBasisSpec(max_order=n2, basis="real"), g))
    y = np.asarray(sh_matrix(SHBasisSpec(max_order=n, basis="real"), g))
    w = np.full(g.size, 4 * np.pi / g.size)
    out = np.zeros(((n1 + 1) ** 2, (n2 + 1) ** 2, (n + 1) ** 2), dtype=float)
    for q in range((n + 1) ** 2):
        out[:, :, q] = (y1.T * w[None, :]) @ (y2 * y[:, [q]])
    return out


def legendre2(N: int, x: ArrayLike) -> np.ndarray:
    xx = np.asarray(x, dtype=float).reshape(-1)
    n = int(N)
    pos = np.vstack([lpmv(m, n, xx) for m in range(0, n + 1)])
    if n == 0:
        return pos
    m = np.arange(0, n + 1)
    norm = ((-1) ** m) * np.array([math.factorial(n - mm) / math.factorial(n + mm) for mm in m], dtype=float)
    neg = (norm[:, None] * pos)[-1:0:-1, :]
    return np.vstack([neg, pos])


def w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    try:
        from scipy.special import wigner_3j  # type: ignore

        return float(wigner_3j(j1, j2, j3, m1, m2, m3))
    except Exception:
        from sympy.physics.wigner import wigner_3j as _w3j

        return float(_w3j(j1, j2, j3, m1, m2, m3).evalf())


def w3j_stirling(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    return w3j(j1, j2, j3, m1, m2, m3)


def sym_w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int):
    from sympy.physics.wigner import wigner_3j as _w3j

    return _w3j(j1, j2, j3, m1, m2, m3)


def unitCart2sph(xyz: ArrayLike) -> np.ndarray:
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2:
        raise ValueError("xyz must be [N,3] or [3,N]")
    if arr.shape[1] != 3 and arr.shape[0] == 3:
        arr = arr.T
    if arr.shape[1] != 3:
        raise ValueError("xyz must have three columns")
    az, el, _ = cart_to_sph(arr[:, 0], arr[:, 1], arr[:, 2], convention="az_el")
    return np.column_stack([az, el])


def sphConvolution(x_nm: ArrayLike, h_n: ArrayLike) -> np.ndarray:
    x = np.asarray(x_nm, dtype=np.complex128).reshape(-1)
    h = np.asarray(h_n, dtype=np.complex128).reshape(-1)
    nx = int(round(np.sqrt(x.size) - 1))
    nh = h.size - 1
    nmax = min(nx, nh)
    y = np.zeros((nmax + 1) ** 2, dtype=np.complex128)
    for n in range(nmax + 1):
        for m in range(-n, n + 1):
            q = n * (n + 1) + m
            y[q] = 2 * np.pi * np.sqrt(4 * np.pi / (2 * n + 1)) * x[q] * h[n]
    return y


def sphMultiplication(a_nm: ArrayLike, b_nm: ArrayLike, G: ArrayLike | None = None) -> np.ndarray:
    a = np.asarray(a_nm, dtype=np.complex128).reshape(-1)
    b = np.asarray(b_nm, dtype=np.complex128).reshape(-1)
    na = int(round(np.sqrt(a.size) - 1))
    nb = int(round(np.sqrt(b.size) - 1))
    nc = na + nb
    g = np.asarray(gaunt_mtx(na, nb, nc) if G is None else G, dtype=np.complex128)
    out = np.zeros((nc + 1) ** 2, dtype=np.complex128)
    for n in range(nc + 1):
        for m in range(-n, n + 1):
            q = n * (n + 1) + m
            out[q] = a.conj().T @ g[:, :, q] @ b
    return out


def sphDelaunay(dirs: ArrayLike) -> np.ndarray:
    d = np.asarray(dirs, dtype=float)
    xyz = unit_sph_to_cart(d[:, 0], d[:, 1], convention="az_el")
    faces = ConvexHull(xyz).simplices[:, ::-1]
    for i in range(faces.shape[0]):
        f = faces[i]
        faces[i] = np.roll(f, -int(np.argmin(f)))
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    return faces[order]


def sphVoronoi(dirs: ArrayLike, faces: ArrayLike | None = None):
    from scipy.spatial import SphericalVoronoi

    d = np.asarray(dirs, dtype=float)
    xyz = unit_sph_to_cart(d[:, 0], d[:, 1], convention="az_el")
    sv = SphericalVoronoi(xyz)
    sv.sort_vertices_of_regions()
    vor = {"vert": sv.vertices, "face": [np.asarray(r, dtype=int) for r in sv.regions]}
    duplicates = np.zeros(sv.vertices.shape[0], dtype=int)
    return vor, duplicates


def sphVoronoiAreas(voronoi: dict[str, Any]) -> dict[str, Any]:
    vert = np.asarray(voronoi["vert"], dtype=float)
    faces = voronoi["face"]

    def _area(face_idx: np.ndarray) -> float:
        pts = vert[np.asarray(face_idx, dtype=int)]
        m = pts.shape[0]
        angles = np.zeros(m, dtype=float)
        for i in range(m):
            p0 = pts[(i - 1) % m]
            p1 = pts[i]
            p2 = pts[(i + 1) % m]
            n1 = np.cross(p1, p0)
            n2 = np.cross(p1, p2)
            n1 /= np.linalg.norm(n1)
            n2 /= np.linalg.norm(n2)
            angles[i] = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return float(np.sum(angles) - (m - 2) * np.pi)

    areas = np.array([_area(np.asarray(f, dtype=int)) for f in faces], dtype=float)
    out = dict(voronoi)
    out["area"] = areas
    return out


def getVoronoiWeights(dirs: ArrayLike) -> np.ndarray:
    vor, _ = sphVoronoi(dirs)
    vor = sphVoronoiAreas(vor)
    return np.asarray(vor["area"], dtype=float)


def getFliegeNodes(index: int):
    i = int(index)
    if i > 30 or i < 2:
        raise ValueError("index must be in [2, 30]")
    if not _FLIEGE_MAT.exists():
        raise FileNotFoundError(f"Missing Fliege nodes mat file: {_FLIEGE_MAT}")
    data = loadmat(_FLIEGE_MAT)
    nodes = data["fliegeNodes"][0, i - 1]
    vecs = np.asarray(nodes[:, 0:3], dtype=float)
    az, el, _ = cart_to_sph(vecs[:, 0], vecs[:, 1], vecs[:, 2], convention="az_el")
    dirs = np.column_stack([az, el])
    weights = np.asarray(nodes[:, 3], dtype=float)
    return vecs, dirs, weights


def plotSphFunctionGrid(F_grid: ArrayLike, *args, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    arr = np.asarray(F_grid)
    ax.plot_surface(np.real(arr), np.imag(arr), np.abs(arr), linewidth=0)
    return fig, ax


def plotSphFunctionTriangle(*args, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    return fig, ax


def plotSphFunctionCoeffs(coeffs: ArrayLike, *args, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(np.abs(np.asarray(coeffs).reshape(-1)))
    return fig, ax


__all__ = [
    "Fdirs2grid",
    "checkCondNumberSHT",
    "complex2realCoeffs",
    "complex2realSHMtx",
    "conjCoeffs",
    "directSHT",
    "euler2rotationMatrix",
    "gaunt_mtx",
    "gaunt_mtx_fast",
    "getFliegeNodes",
    "getRealGauntMtx",
    "getSH",
    "getSHrotMtx",
    "getTdesign",
    "getVoronoiWeights",
    "grid2dirs",
    "inverseSHT",
    "leastSquaresSHT",
    "legendre2",
    "plotSphFunctionCoeffs",
    "plotSphFunctionGrid",
    "plotSphFunctionTriangle",
    "real2complexCoeffs",
    "real2complexSHMtx",
    "replicatePerOrder",
    "rotateAxisCoeffs",
    "sphConvolution",
    "sphDelaunay",
    "sphMultiplication",
    "sphVoronoi",
    "sphVoronoiAreas",
    "sym_w3j",
    "unitCart2sph",
    "unitSph2cart",
    "w3j",
    "w3j_stirling",
    "wignerD",
]


# MATLAB-name compatibility alias.
def wignerD(N: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    return wigner_d_matrix(int(N), float(alpha), float(beta), float(gamma))


def replicatePerOrder(values: ArrayLike) -> np.ndarray:
    return replicate_per_order(values)
