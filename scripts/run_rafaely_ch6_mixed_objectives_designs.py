#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from spherical_array_processing.repro.rafaely import bn_mat, uniform_sampling


def compute_mixed_objectives_designs():
    # Columns: sphere, N, kr, sigma_a^2, sigma_s^2
    par = np.array(
        [
            [0, 2, 2, 1, 0],
            [0, 2, 2, 0, 1],
            [1, 3, 3, 1, 0],
            [1, 3, 3, 0, 1],
            [1, 3, 3, 1, 1],
            [1, 4, 2, 1, 0],
            [1, 4, 2, 0, 1],
            [1, 4, 2, 0.4, 1],
        ],
        dtype=float,
    )
    rows: list[dict[str, float]] = []

    for row in par:
        sphere = int(row[0])
        order = int(row[1])
        kr = float(row[2])
        sigma2a = float(row[3])
        sigma2s = float(row[4])

        a, th, ph = uniform_sampling(order)
        Q = int(len(a))
        NN = (2 * np.arange(order + 1) + 1).astype(float)
        vn = NN / (4 * np.pi)

        bn = bn_mat(order, np.array([kr]), np.array([kr]), sphere)[:, 0]
        R = sigma2a * (1 / (4 * np.pi)) * np.diag(NN) + sigma2s * (1 / Q) * np.diag(NN / np.maximum(np.abs(bn) ** 2, 1e-12))
        Ri = np.linalg.pinv(R)
        dn = (Ri @ vn) / np.maximum(vn.T @ Ri @ vn, 1e-12)

        DI = float((dn.T @ NN) ** 2 / np.maximum(dn.T @ np.diag(NN) @ dn, 1e-12))
        WNG = float((Q / (4 * np.pi) ** 2) * (dn.T @ NN) ** 2 / np.maximum(dn.T @ np.diag(NN / np.maximum(np.abs(bn) ** 2, 1e-12)) @ dn, 1e-12))

        rows.append(
            {
                "sphere": float(sphere),
                "N": float(order),
                "Q": float(Q),
                "kr": kr,
                "sigma2a": sigma2a,
                "sigma2s": sigma2s,
                "DI": DI,
                "WNG": WNG,
            }
        )
    return rows


def _format_table(rows: list[dict[str, float]]) -> str:
    lines = []
    lines.append("Sph\tN\tQ\tkr\tsig_a\tsig_s\tDF\tWNG")
    for r in rows:
        lines.append(
            f"{r['sphere']:.0f}\t{r['N']:.0f}\t{r['Q']:.0f}\t{r['kr']:.1f}\t{r['sigma2a']:.1f}\t{r['sigma2s']:.1f}\t{r['DI']:.2f}\t{r['WNG']:.2f}"
        )
    return "\n".join(lines)


def main(print_table: bool = True):
    rows = compute_mixed_objectives_designs()
    if print_table:
        print(_format_table(rows))
    return rows


if __name__ == "__main__":
    raise SystemExit(0 if main(print_table=True) else 1)
