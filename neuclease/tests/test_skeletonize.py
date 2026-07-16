import re

import numpy as np
import pytest

from neuclease.dvid.rle import runlength_encode_mask_to_ranges
from neuclease.util.skeleton import neuroglancer_skeleton_to_df
import neuclease.misc.skeletonize as skel_module
from neuclease.misc.skeletonize import (
    treeify_coords,
    skeletonize_neuron,
    skeletonize_neuron_from_ranges,
    HaloComponentTracker,
)


def _edges(df):
    """Set of undirected (node, parent) edges in a skeleton DataFrame, excluding roots."""
    e = set()
    for row in df.itertuples(index=False):
        if row.parent != -1:
            e.add(frozenset((row.node, row.parent)))
    return e


def test_treeify_groups_prevent_cross_component_edges():
    # Two small, spatially-close clusters. Their points are close enough that an
    # unconstrained nearest-neighbor MST would join them into one component.
    cluster_a = np.array([[0, 0, i] for i in range(5)])
    cluster_b = np.array([[0, 2, i] for i in range(5)])  # only 2 voxels away in y
    coords = np.concatenate([cluster_a, cluster_b])
    cc_ids = np.array([0] * 5 + [1] * 5)

    # Without cc_ids, the two clusters get merged into a single component.
    df_merged = treeify_coords(coords)
    assert df_merged['cc'].nunique() == 1

    # With cc_ids, they remain distinct and NO edge crosses between them.
    df = treeify_coords(coords, cc_ids=cc_ids)
    assert df['cc'].nunique() == 2
    assert len(df) == len(coords)

    node_cc = dict(zip(df['node'], df['cc']))
    for edge in _edges(df):
        u, v = tuple(edge)
        assert node_cc[u] == node_cc[v], "An edge crossed between two components"


def test_treeify_isolated_points_are_retained():
    # A single-point component must not be dropped from the output.
    coords = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [50, 50, 50]])
    cc_ids = np.array([0, 0, 0, 1])
    df = treeify_coords(coords, cc_ids=cc_ids, first_node=0)
    assert len(df) == len(coords)
    assert set(df['node']) == {0, 1, 2, 3}
    # The lone point is its own root.
    lone = df[df['node'] == 3].iloc[0]
    assert lone['parent'] == -1


def test_treeify_heal_respects_max_distance():
    cluster_a = np.array([[0, 0, i] for i in range(5)])
    cluster_b = np.array([[0, 0, i] for i in range(20, 25)])  # ~15 voxels away
    coords = np.concatenate([cluster_a, cluster_b])
    cc_ids = np.array([0] * 5 + [1] * 5)

    # No healing -> stays split.
    assert treeify_coords(coords, cc_ids=cc_ids)['cc'].nunique() == 2

    # Small max distance -> too far to bridge, stays split.
    assert treeify_coords(coords, cc_ids=cc_ids, heal_max_distance=5)['cc'].nunique() == 2

    # Large max distance -> bridged into one.
    assert treeify_coords(coords, cc_ids=cc_ids, heal_max_distance=100)['cc'].nunique() == 1


def test_treeify_anisotropy_scales_heal_distance():
    # Two clusters separated by 4 voxels along Z.
    cluster_a = np.array([[0, 0, i] for i in range(5)])
    cluster_b = np.array([[4, 0, i] for i in range(5)])
    coords = np.concatenate([cluster_a, cluster_b])
    cc_ids = np.array([0] * 5 + [1] * 5)

    # Isotropic: the gap is 4 units.
    assert treeify_coords(coords, cc_ids=cc_ids, heal_max_distance=3)['cc'].nunique() == 2
    assert treeify_coords(coords, cc_ids=cc_ids, heal_max_distance=5)['cc'].nunique() == 1

    # Anisotropic Z (10x): the gap becomes 40 units, so the thresholds shift.
    aniso = (10, 1, 1)  # zyx
    assert treeify_coords(coords, cc_ids=cc_ids, heal_max_distance=30, anisotropy_zyx=aniso)['cc'].nunique() == 2
    assert treeify_coords(coords, cc_ids=cc_ids, heal_max_distance=50, anisotropy_zyx=aniso)['cc'].nunique() == 1


class _FakeSparsevol:
    """Monkeypatch target for fetch_sparsevol: returns ranges from a dense mask."""
    def __init__(self, mask):
        self.ranges = runlength_encode_mask_to_ranges(mask.astype(bool))

    def __call__(self, *args, scale=0, format='ranges', **kwargs):
        assert format == 'ranges'
        # The caller fetches at a coarser scale; our synthetic mask is already
        # at that scale, so just return it unchanged.
        return self.ranges


def _skeletonize_mask(monkeypatch, mask, **kwargs):
    monkeypatch.setattr(skel_module, 'fetch_sparsevol', _FakeSparsevol(mask))
    # Provide voxel_size_xyz explicitly so we never hit the network for instance info.
    kwargs.setdefault('voxel_size_xyz', (1, 1, 1))
    tracker = HaloComponentTracker()
    df = skeletonize_neuron(
        'fake-server', 'fake-uuid', 'fake-seg', body=1,
        scale=0, block_shape=(20, 20, 20), halo=6,
        closing_radius=0, tracker=tracker, threads=1,
        **kwargs
    )
    return df, tracker


def test_single_component_spanning_blocks(monkeypatch):
    # A long tube along X that crosses several 20-voxel blocks.
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True

    df, tracker = _skeletonize_mask(monkeypatch, mask)

    assert len(df) > 0
    assert df['cc'].nunique() == 1, "A single tube spanning blocks should be one component"
    # Neighbor faces are registered in matched pairs, so nothing should be left over.
    assert len(tracker.pending) == 0


def test_scales_produce_integer_coords(monkeypatch):
    # scale 0 (no centering offset) and scale > 0 (half-voxel offset) must both
    # yield integer coordinates without raising.
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True

    for scale in (0, 2):
        monkeypatch.setattr(skel_module, 'fetch_sparsevol', _FakeSparsevol(mask))
        df = skeletonize_neuron(
            's', 'u', 'seg', body=1,
            scale=scale, block_shape=(20, 20, 20), halo=6,
            closing_radius=0, tracker=HaloComponentTracker(), threads=1,
        )
        assert len(df) > 0
        for c in 'xyz':
            assert np.array_equal(df[c], df[c].astype(int)), f"non-integer {c} at scale {scale}"


def test_two_separate_bodies_stay_separate(monkeypatch):
    # Two parallel tubes, close together but not touching, each spanning blocks.
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[10:15, 10:15, 4:76] = True   # tube A
    mask[10:15, 18:23, 4:76] = True   # tube B, 3-voxel gap in Y

    df, tracker = _skeletonize_mask(monkeypatch, mask)

    assert df['cc'].nunique() == 2, "Two disjoint tubes should remain two components"
    assert len(tracker.pending) == 0

    node_cc = dict(zip(df['node'], df['cc']))
    for edge in _edges(df):
        u, v = tuple(edge)
        assert node_cc[u] == node_cc[v]


def test_separate_bodies_can_be_healed(monkeypatch):
    # Same two tubes, but with a generous heal distance they merge into one.
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[10:15, 10:15, 4:76] = True
    mask[10:15, 18:23, 4:76] = True

    df, _ = _skeletonize_mask(monkeypatch, mask, heal_max_distance=100)
    assert df['cc'].nunique() == 1


def test_radii_reported_in_physical_units(monkeypatch):
    # The distance transform is anisotropy-aware, so doubling the (isotropic)
    # voxel size should (roughly) double the physical radius estimates.
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True

    df1, _ = _skeletonize_mask(monkeypatch, mask, return_radii=True, voxel_size_xyz=(1, 1, 1))
    df2, _ = _skeletonize_mask(monkeypatch, mask, return_radii=True, voxel_size_xyz=(2, 2, 2))

    assert 'radius' in df1.columns
    assert df1['radius'].max() > 0
    assert np.isclose(df2['radius'].mean(), 2 * df1['radius'].mean(), rtol=0.05)


def test_skeletonize_swc_format(monkeypatch, tmp_path):
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True
    monkeypatch.setattr(skel_module, 'fetch_sparsevol', _FakeSparsevol(mask))
    # skeletonize_neuron fetches the mutation id internally for the SWC header.
    monkeypatch.setattr(skel_module, 'fetch_lastmod', lambda *a, **k: {"mutation id": 12345})

    out = tmp_path / "1.swc"
    swc = skeletonize_neuron(
        's', 'u', 'seg', body=1, scale=0, block_shape=(20, 20, 20), halo=6,
        closing_radius=0, voxel_size_xyz=(8, 8, 8),
        format='swc', output_path=str(out), threads=1,
    )
    assert isinstance(swc, str)

    # Header: generator line + backwards-compatible NeuTu-style keys.
    assert swc.startswith('#Generated by neuclease')
    assert '"downresLevel": 0' in swc
    assert '"dataName": "seg"' in swc
    m = re.search(r'{"mutation id": (\d+)}', swc)
    assert m and int(m.group(1)) == 12345

    # Body rows: 7 whitespace-separated SWC columns (id kind x y z radius parent).
    body_lines = [ln for ln in swc.splitlines() if ln and not ln.startswith('#')]
    assert len(body_lines) > 0
    assert all(len(ln.split()) == 7 for ln in body_lines)

    # output_path was written with the same content.
    assert out.read_text() == swc


def test_skeletonize_neuroglancer_format(monkeypatch):
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True

    kwargs = dict(scale=0, block_shape=(20, 20, 20), halo=6, closing_radius=0,
                  voxel_size_xyz=(8, 8, 8), threads=1)

    monkeypatch.setattr(skel_module, 'fetch_sparsevol', _FakeSparsevol(mask))
    buf = skeletonize_neuron('s', 'u', 'seg', body=1, format='neuroglancer', **kwargs)
    assert isinstance(buf, (bytes, bytearray))

    monkeypatch.setattr(skel_module, 'fetch_sparsevol', _FakeSparsevol(mask))
    df = skeletonize_neuron('s', 'u', 'seg', body=1, format='pandas', **kwargs)

    ng_df = neuroglancer_skeleton_to_df(buf)
    assert len(ng_df) == len(df)

    # Neuroglancer positions are in nm (df voxel coords scaled by the 8nm voxel size).
    ng_coords = np.sort(ng_df[[*'xyz']].values / 8.0, axis=0)
    df_coords = np.sort(df[[*'xyz']].values.astype(float), axis=0)
    assert np.allclose(ng_coords, df_coords)


def test_skeletonize_from_ranges_no_dvid(monkeypatch):
    # skeletonize_neuron_from_ranges must not touch DVID at all: sabotage the
    # fetch functions so any DVID access would raise.
    def _boom(*a, **k):
        raise AssertionError("skeletonize_neuron_from_ranges should not access DVID")
    monkeypatch.setattr(skel_module, 'fetch_sparsevol', _boom)
    monkeypatch.setattr(skel_module, 'fetch_instance_info', _boom)
    monkeypatch.setattr(skel_module, 'fetch_lastmod', _boom)

    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True
    ranges = runlength_encode_mask_to_ranges(mask)

    common = dict(scale=0, block_shape=(20, 20, 20), halo=6, closing_radius=0,
                  voxel_size_xyz=(8, 8, 8), threads=1)

    # pandas
    df = skeletonize_neuron_from_ranges(ranges, format='pandas', **common)
    assert len(df) > 0
    assert df['cc'].nunique() == 1

    # swc (caller must supply the header metadata)
    swc = skeletonize_neuron_from_ranges(
        ranges, format='swc', uuid='u', segmentation_instance='seg', mutid=999, **common
    )
    assert re.search(r'{"mutation id": (\d+)}', swc).group(1) == '999'
    assert '"dataName": "seg"' in swc

    # neuroglancer
    buf = skeletonize_neuron_from_ranges(ranges, format='neuroglancer', **common)
    assert isinstance(buf, (bytes, bytearray)) and len(buf) > 0


def test_skeletonize_from_ranges_swc_requires_metadata():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    ranges = runlength_encode_mask_to_ranges(mask)
    # Missing uuid/segmentation_instance/mutid for swc -> AssertionError.
    with pytest.raises(AssertionError):
        skeletonize_neuron_from_ranges(ranges, format='swc', scale=0, threads=1)


def test_skeletonize_output_path_rejected_for_pandas(monkeypatch):
    monkeypatch.setattr(skel_module, 'fetch_sparsevol', _FakeSparsevol(np.ones((4, 4, 4), bool)))
    with pytest.raises(AssertionError):
        skeletonize_neuron('s', 'u', 'seg', body=1, format='pandas', output_path='/tmp/x.swc')


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_skeletonize'])
