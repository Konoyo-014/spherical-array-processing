"""Tests for Ambisonic channel metadata helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing import ambi


def test_channel_vectors_and_labels_are_acn_ordered():
    assert ambi.acn_sequence(2).tolist() == list(range(9))
    assert ambi.channel_degrees(2).tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert ambi.channel_orders(2).tolist() == [0, -1, 0, 1, -2, -1, 0, 1, 2]
    assert ambi.channel_labels(1, style="acn") == ("ACN0", "ACN1", "ACN2", "ACN3")
    assert ambi.channel_labels(1, style="nm") == ("Y0_+0", "Y1_-1", "Y1_+0", "Y1_+1")
    assert ambi.fuma_channel_labels(1) == ("W", "X", "Y", "Z")
    assert ambi.channel_name(3, style="fuma") == "X"
    assert ambi.channel_index(2, -1) == 5
    assert ambi.channel_label_to_acn("ACN3") == 3
    assert ambi.channel_label_to_acn("X") == 3
    assert ambi.channel_label_to_acn("Y2_-1") == 5
    assert ambi.channel_label_to_acn("2.-1") == 5
    assert ambi.first_acn_for_degree(2) == 4
    assert ambi.last_acn_for_degree(2) == 8
    assert ambi.order_block_slice(2) == slice(4, 9)
    assert_allclose(ambi.per_order_channel_counts(2), [1, 3, 5])
    assert_allclose(ambi.order_weight_vector(2, [1.0, 2.0, 3.0]), [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0])


def test_channel_table_flags():
    table = ambi.channel_table(2)
    assert table[0].is_zonal
    assert not table[0].is_horizontal
    assert table[4].is_sectoral
    assert table[6].label_nm == "Y2_+0"
    assert table[6].label_fuma == "R"


def test_masks_are_consistent():
    assert ambi.degree_mask(2, 1).tolist() == [False, True, True, True, False, False, False, False, False]
    assert ambi.degree_channel_indices(2, 1).tolist() == [1, 2, 3]
    assert ambi.signed_order_mask(2, 0).tolist() == [True, False, True, False, False, False, True, False, False]
    assert ambi.signed_order_channel_indices(2, 0).tolist() == [0, 2, 6]
    assert ambi.horizontal_channel_mask(1).tolist() == [False, True, False, True]
    assert ambi.horizontal_channel_indices(1).tolist() == [1, 3]
    assert ambi.zonal_channel_mask(1).tolist() == [True, False, True, False]
    assert ambi.zonal_channel_indices(1).tolist() == [0, 2]
    assert ambi.sectoral_channel_mask(2).tolist() == [True, True, False, True, True, False, False, False, True]
    assert ambi.sectoral_channel_indices(1).tolist() == [0, 1, 3]
    assert ambi.omni_channel_mask(1).tolist() == [True, False, False, False]
    assert ambi.non_omni_channel_mask(1).tolist() == [False, True, True, True]
    assert ambi.first_order_channel_mask(1).tolist() == [False, True, True, True]
    assert ambi.truncate_channel_mask(2, 1).tolist() == [True, True, True, True, False, False, False, False, False]
    assert ambi.mask_to_indices([True, False, True, False]).tolist() == [0, 2]
    assert ambi.indices_to_mask([0, 2], 1).tolist() == [True, False, True, False]
    assert ambi.invert_channel_mask([True, False, True, False]).tolist() == [False, True, False, True]
    assert ambi.combine_channel_masks([True, False, False, False], [False, True, False, False]).tolist() == [True, True, False, False]


def test_mixed_order_compaction_round_trip():
    mask = ambi.mixed_order_mask(3, 1)
    assert mask.size == 16
    assert ambi.mixed_order_channel_count(3, 1) == int(np.count_nonzero(mask))
    spec = ambi.infer_mixed_order(mask)
    assert spec.horizontal_order == 3
    assert spec.vertical_order == 1
    coeffs = np.arange(2 * 16).reshape(2, 16)
    compact = ambi.compact_mixed_order_coeffs(coeffs, mask, axis=1)
    expanded = ambi.expand_mixed_order_coeffs(compact, mask, axis=1, fill_value=-1)
    assert_allclose(expanded[:, mask], coeffs[:, mask])
    assert np.all(expanded[:, ~mask] == -1)


def test_fuma_permutation_and_channel_selection_helpers():
    data = np.arange(8).reshape(2, 4)
    assert ambi.acn_to_fuma_permutation(1).tolist() == [0, 3, 1, 2]
    assert ambi.fuma_to_acn_permutation(1).tolist() == [0, 2, 3, 1]
    fuma = ambi.reorder_acn_to_fuma(data, axis=1)
    assert_allclose(fuma, data[:, [0, 3, 1, 2]])
    assert_allclose(ambi.reorder_fuma_to_acn(fuma, axis=1), data)
    assert_allclose(ambi.select_channels(data, [0, 2], axis=1), data[:, [0, 2]])
    zeroed = ambi.zero_channels(data, [1, 3], axis=1)
    assert_allclose(zeroed[:, [1, 3]], 0)
    assert_allclose(ambi.drop_channels(data, [1, 3], axis=1), data[:, [0, 2]])


def test_channel_energy_peak_and_metadata_helpers():
    data = np.zeros((2, 4))
    data[:, 0] = [1.0, -1.0]
    data[:, 3] = [2.0, 0.0]
    assert_allclose(ambi.channel_energy(data, axis=1), [2.0, 0.0, 0.0, 4.0])
    assert_allclose(ambi.channel_rms(data, axis=1), [1.0, 0.0, 0.0, np.sqrt(2.0)])
    assert_allclose(ambi.channel_peak(data, axis=1), [1.0, 0.0, 0.0, 2.0])
    assert ambi.active_channel_mask(data, axis=1).tolist() == [True, False, False, True]
    assert ambi.active_channel_indices(data, axis=1).tolist() == [0, 3]
    assert_allclose(ambi.per_order_energy(data, axis=1), [2.0, 4.0])
    assert_allclose(ambi.per_order_peak(data, axis=1), [1.0, 2.0])
    assert_allclose(ambi.per_order_energy_fraction(data, axis=1), [1.0 / 3.0, 2.0 / 3.0])
    assert ambi.channel_metadata_dicts(1)[0]["label_acn"] == "ACN0"


def test_split_and_join_by_order():
    coeffs = np.arange(18).reshape(2, 9)
    blocks = ambi.split_coeffs_by_order(coeffs, axis=1)
    assert [b.shape for b in blocks] == [(2, 1), (2, 3), (2, 5)]
    joined = ambi.join_coeffs_by_order(blocks, axis=1)
    assert_allclose(joined, coeffs)
    coeffs_t = coeffs.T
    blocks_t = ambi.split_coeffs_by_order(coeffs_t, axis=0)
    assert [b.shape for b in blocks_t] == [(1, 2), (3, 2), (5, 2)]
    assert_allclose(ambi.join_coeffs_by_order(blocks_t, axis=0), coeffs_t)


def test_validate_axis_and_reports():
    data = np.zeros((4, 9))
    axis, order = ambi.validate_ambisonic_channel_axis(data, axis=1)
    assert axis == 1
    assert order == 2
    data[:, 3] = 1.0
    report = ambi.active_channel_report(data, axis=1)
    assert report["active_channel_count"] == 1
    assert report["active_indices"] == [3]
    assert report["active_degrees"] == [1]
    assert_allclose(ambi.per_order_rms(data, axis=1), [0.0, np.sqrt(1 / 3), 0.0])


def test_validation_rejects_bad_channel_specs():
    with pytest.raises(ValueError, match="order"):
        ambi.channel_index(1, 2)
    with pytest.raises(ValueError, match="FuMa"):
        ambi.fuma_channel_labels(4)
    with pytest.raises(ValueError, match="channel axis"):
        ambi.validate_ambisonic_channel_axis(np.zeros((2, 5)), axis=1)
    with pytest.raises(ValueError, match="mixed-order"):
        ambi.infer_mixed_order([True, False, True, False])
    with pytest.raises(ValueError, match="cannot parse"):
        ambi.channel_label_to_acn("not-a-channel")
    with pytest.raises(ValueError, match="out of range"):
        ambi.validate_channel_indices([9], 1)
    with pytest.raises(ValueError, match="matching shape"):
        ambi.combine_channel_masks([True], [True, False, False, False])
