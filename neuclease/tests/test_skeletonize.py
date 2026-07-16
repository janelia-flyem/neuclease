import numpy as np
import pytest

from neuclease.dvid.rle import runlength_encode_mask_to_ranges
import neuclease.misc.skeletonize as skel_module
from neuclease.misc.skeletonize import (
    treeify_coords,
    skeletonize_neuron,
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


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_skeletonize'])
