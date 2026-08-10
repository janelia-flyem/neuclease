import re

import numpy as np
import pandas as pd
import pytest

from neuclease.dvid.rle import runlength_encode_mask_to_ranges
from neuclease.util.skeleton import neuroglancer_skeleton_to_df
import neuclease.misc.skeletonize as skel_module
from neuclease.misc.skeletonize import (
    treeify_coords,
    skeletonize_neuron,
    skeletonize_neuron_from_ranges,
    heal_skeleton,
    HaloComponentTracker,
)


def _assert_valid_forest(df):
    """Every connected component (by node/parent edges) is a tree with exactly one root."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(df['node'])
    edges = df.loc[df['parent'] != -1, ['node', 'parent']]
    for node, parent in edges.itertuples(index=False):
        g.add_edge(node, parent)
    roots = set(df.loc[df['parent'] == -1, 'node'])
    for comp in nx.connected_components(g):
        assert len(comp & roots) == 1, "component must have exactly one root"
    # A forest has (#nodes - #components) edges (no cycles).
    assert g.number_of_edges() == g.number_of_nodes() - nx.number_connected_components(g)


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
    # min_component_size=1 to exercise retention (the default of 2 would drop the lone point).
    df = treeify_coords(coords, cc_ids=cc_ids, min_component_size=1, first_node=0)
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


def test_treeify_one_tree_per_cc():
    # Two dense clusters far apart (>> the kNN reach).  Each point's nearest
    # neighbors lie entirely within its own cluster, so the kNN MST leaves them as
    # two separate sub-trees.
    a = np.array([[0, 0, i] for i in range(10)])
    b = np.array([[50, 0, i] for i in range(10)])
    coords = np.concatenate([a, b])

    # Same cc_id => same physical component => the sub-trees must be rejoined into
    # a single tree (exactly one root), regardless of the gap.
    df = treeify_coords(coords, cc_ids=np.zeros(20, dtype=int))
    assert (df['parent'] == -1).sum() == 1
    assert df['cc'].nunique() == 1
    assert len(df) == len(coords)

    # Distinct cc_ids => distinct physical components => stays two trees.
    df2 = treeify_coords(coords, cc_ids=np.array([0] * 10 + [1] * 10))
    assert (df2['parent'] == -1).sum() == 2
    assert df2['cc'].nunique() == 2


def test_treeify_min_component_size_drops_orphans():
    # A 5-node line plus two isolated single points (each its own component).
    line = np.array([[0, 0, i] for i in range(5)])
    orphans = np.array([[100, 0, 0], [0, 100, 0]])
    coords = np.concatenate([line, orphans])
    cc_ids = np.array([0, 0, 0, 0, 0, 1, 2])

    # min_component_size=1 keeps everything: 3 components (1 line + 2 orphans).
    df = treeify_coords(coords, cc_ids=cc_ids, min_component_size=1)
    assert df['cc'].nunique() == 3
    assert len(df) == 7

    # The default (2) drops the two single-node orphans.
    df2 = treeify_coords(coords, cc_ids=cc_ids)
    assert df2['cc'].nunique() == 1
    assert len(df2) == 5
    assert (df2['parent'] == -1).sum() == 1

    # If everything is single-node, dropping yields an empty (well-formed) result.
    singletons = np.array([[0, 0, 0], [50, 0, 0]])
    df3 = treeify_coords(singletons, cc_ids=np.array([0, 1]))
    assert len(df3) == 0
    assert list(df3.columns) == ['node', *'xyz', 'parent', 'cc']


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

    df1, _ = _skeletonize_mask(monkeypatch, mask, return_radii=True, radius_units='nanometers', voxel_size_xyz=(1, 1, 1))
    df2, _ = _skeletonize_mask(monkeypatch, mask, return_radii=True, radius_units='nanometers', voxel_size_xyz=(2, 2, 2))

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


def test_treeify_empty_input():
    # A body that skeletonizes to zero points must not crash treeify_coords.
    empty = np.zeros((0, 3), dtype=int)

    df = treeify_coords(empty, cc_ids=np.array([], dtype=int))
    assert len(df) == 0
    assert list(df.columns) == ['node', *'xyz', 'parent', 'cc']

    df_r = treeify_coords(empty, radii=np.array([]), cc_ids=np.array([], dtype=int))
    assert len(df_r) == 0
    assert 'radius' in df_r.columns


def test_skeletonize_empty_skeleton_all_formats():
    # An even-width symmetric bar skeletonizes to zero points (skimage quirk),
    # which previously crashed treeify_coords with "No objects to concatenate".
    mask = np.zeros((10, 10, 40), dtype=bool)
    mask[3:7, 3:7, 4:36] = True
    ranges = runlength_encode_mask_to_ranges(mask)

    common = dict(scale=0, block_shape=(20, 20, 20), halo=4, closing_radius=0,
                  voxel_size_xyz=(8, 8, 8), threads=1)

    df = skeletonize_neuron_from_ranges(ranges, format='pandas', **common)
    assert len(df) == 0

    swc = skeletonize_neuron_from_ranges(
        ranges, format='swc', uuid='u', segmentation_instance='seg', mutid=1, **common
    )
    # Header present, but no body (node) rows.
    assert '"mutation id": 1' in swc
    assert not [ln for ln in swc.splitlines() if ln and not ln.startswith('#')]

    buf = skeletonize_neuron_from_ranges(ranges, format='neuroglancer', **common)
    # 8-byte header: num_vertices=0, num_edges=0.
    assert np.frombuffer(buf, np.uint32).tolist() == [0, 0]


def test_skeletonize_units_options():
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True
    ranges = runlength_encode_mask_to_ranges(mask)

    vs = (8, 8, 8)  # isotropic
    common = dict(scale=0, block_shape=(20, 20, 20), halo=6, closing_radius=0,
                  voxel_size_xyz=vs, return_radii=True, threads=1)

    # voxels / nanometers: coords in scale-0 voxels, radii in nm.
    ref = skeletonize_neuron_from_ranges(
        ranges, format='pandas', coordinate_units='voxels', radius_units='nanometers', **common)
    assert len(ref) > 0

    # nanometer coordinates: scaled by voxel size (radii still nm here).
    nm = skeletonize_neuron_from_ranges(
        ranges, format='pandas', coordinate_units='nanometers', radius_units='nanometers', **common)
    assert np.allclose(nm[[*'xyz']].to_numpy(), ref[[*'xyz']].to_numpy() * 8)
    assert np.allclose(np.sort(nm['radius']), np.sort(ref['radius']))

    # voxel radii: divided by the isotropic voxel size (coords unchanged voxels).
    vv = skeletonize_neuron_from_ranges(
        ranges, format='pandas', coordinate_units='voxels', radius_units='voxels', **common)
    assert np.allclose(vv[[*'xyz']].to_numpy(), ref[[*'xyz']].to_numpy())
    assert np.allclose(np.sort(vv['radius']), np.sort(ref['radius']) / 8)

    # The two axes are independent: nm coordinates with voxel radii is allowed.
    mixed = skeletonize_neuron_from_ranges(
        ranges, format='pandas', coordinate_units='nanometers', radius_units='voxels', **common)
    assert np.allclose(mixed[[*'xyz']].to_numpy(), ref[[*'xyz']].to_numpy() * 8)
    assert np.allclose(np.sort(mixed['radius']), np.sort(ref['radius']) / 8)


def test_skeletonize_voxel_radii_requires_isotropy():
    mask = np.zeros((30, 30, 80), dtype=bool)
    mask[13:18, 13:18, 4:76] = True
    ranges = runlength_encode_mask_to_ranges(mask)

    common = dict(scale=0, block_shape=(20, 20, 20), halo=6, closing_radius=0,
                  return_radii=True, threads=1)

    # Anisotropic voxels + voxel radii -> ValueError.
    with pytest.raises(ValueError):
        skeletonize_neuron_from_ranges(ranges, format='pandas', radius_units='voxels',
                                       voxel_size_xyz=(16, 16, 15), **common)

    # But nanometer radii are fine on anisotropic data.
    df = skeletonize_neuron_from_ranges(ranges, format='pandas', radius_units='nanometers',
                                        voxel_size_xyz=(16, 16, 15), **common)
    assert len(df) > 0


def test_skeletonize_nanometer_units_require_voxel_size():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    ranges = runlength_encode_mask_to_ranges(mask)

    # nanometer coordinates but no voxel size -> ValueError.
    with pytest.raises(ValueError):
        skeletonize_neuron_from_ranges(ranges, format='pandas', coordinate_units='nanometers',
                                       scale=0, threads=1)


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


def _make_skeleton_df(rows):
    # rows: list of (node, x, y, z, parent, radius)
    return pd.DataFrame(rows, columns=['node', 'x', 'y', 'z', 'parent', 'radius'])


def test_heal_skeleton_reorients_opposed_fragments():
    # Fragment A: 1->2->3 rooted at node 1, running along x = 0,1,2.
    # Fragment B: 6->5->4 rooted at node 6 (the FAR end), running along x = 3,4,5.
    # The nearest cross-fragment points are node 3 (x=2) and node 4 (x=3).
    rows = [
        (1, 0.0, 0.0, 0.0, -1, 1.0),
        (2, 1.0, 0.0, 0.0,  1, 1.0),
        (3, 2.0, 0.0, 0.0,  2, 1.0),
        (4, 3.0, 0.0, 0.0,  5, 1.0),
        (5, 4.0, 0.0, 0.0,  6, 1.0),
        (6, 5.0, 0.0, 0.0, -1, 1.0),
    ]
    df = _make_skeleton_df(rows)
    assert (df['parent'] == -1).sum() == 2  # two roots to begin with

    healed = heal_skeleton(df)
    # One connected tree now, exactly one root, valid (acyclic) structure.
    assert (healed['parent'] == -1).sum() == 1
    assert healed['cc'].nunique() == 1
    _assert_valid_forest(healed)

    # Node ids, coordinates, and radius are preserved.
    assert set(healed['node']) == {1, 2, 3, 4, 5, 6}
    assert 'radius' in healed.columns
    # Fragment B was re-oriented: node 6 is no longer a root.
    assert healed.set_index('node').loc[6, 'parent'] != -1


def test_heal_skeleton_respects_max_distance():
    rows = [
        (1, 0.0, 0.0, 0.0, -1, 1.0),
        (2, 1.0, 0.0, 0.0,  1, 1.0),
        (3, 20.0, 0.0, 0.0, -1, 1.0),
        (4, 21.0, 0.0, 0.0,  3, 1.0),
    ]
    df = _make_skeleton_df(rows)

    # Gap between the two fragments is ~19 (node 2 -> node 3).
    assert heal_skeleton(df, max_distance=5)['parent'].eq(-1).sum() == 2    # too far, stays split
    assert heal_skeleton(df, max_distance=100)['parent'].eq(-1).sum() == 1  # joined
    assert heal_skeleton(df)['parent'].eq(-1).sum() == 1                    # default inf -> joined

    # Anisotropy (XYZ) scales the gap: with 10x on x, the ~19-voxel gap becomes ~190 > 100.
    healed = heal_skeleton(df, max_distance=100, anisotropy_xyz=(10, 1, 1))
    assert (healed['parent'] == -1).sum() == 2
    _assert_valid_forest(healed)


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_skeletonize'])
