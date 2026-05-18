import threading
from functools import partial

import numpy as np
import pandas as pd
import vigra
import skimage.measure
import networkx as nx
from scipy.spatial import KDTree

from neuclease.util import compute_parallel, compute_nonzero_box, extract_subvol
from neuclease.util.segmentation import distance_transform
from neuclease.dvid import fetch_sparsevol
from neuclease.dvid.rle import blockwise_masks_from_ranges


def skeletonize_neuron(
    dvid_server,
    uuid,
    segmentation_instance,
    body,
    scale=2,
    block_shape=(128, 128, 128),
    halo=128,
    closing_radius=5,
    return_radii=False,
    threads=12
):
    """
    Skeletonize a body in DVID by from its sparsevol.

    The body's sparsevol is fetched from DVID at the given scale.  The sparsevol is processed in big
    overlapping chunks ("blocks"), each of which is skeletonized with scikit-image after a bit of
    preprocessing (morphological closing, followed by hole filling).

    Args:
        dvid_server:
            The DVID server to connect to.
        uuid:
            The UUID of the DVID instance to connect to.
        segmentation_instance:
            The name of the segmentation volume to use.
        body:
            The ID of the body to skeletonize.
        scale:
            The scale at which to fetch the sparsevol.
        block_shape:
            The processing will be split into to blocks of this shape (not including the halo).
        halo:
            The amount of overlap between blocks.
            Since we skeletonize each block independently, this halo minimizes edge effects between blocks.
            The skeleton points in the halo region are discarded before returning.
        closing_radius:
            The radius of the morphological closing operation, applied before hole-filling and skeletonization.
        return_radii:
            If True, also return an approximate "radius" for each skeleton point.
            This is computed via the distance transform within the neuron mask.
        threads:
            Blocks are processed in parallel using a thread pool of this size.

    Returns:
        A pandas DataFrame with columns:
            ['node', 'x', 'y', 'z', 'parent', 'cc', 'radius']
        
        (If return_radii=False, the 'radius' column is omitted.)
    """
    all_coords, radii = skeleton_coords(
        dvid_server,
        uuid,
        segmentation_instance,
        body,
        scale,
        block_shape,
        halo,
        closing_radius,
        return_radii=return_radii,
        threads=threads
    )
    df = treeify_coords(all_coords, radii)
    return df


def skeleton_coords(
    dvid_server,
    uuid,
    segmentation_instance,
    body,
    scale=2,
    block_shape=(256, 256, 256),
    halo=256,
    closing_radius=5,
    trim_halo=True,
    return_radii=False,
    threads=12
):
    """
    For a given body in DVID, produce a set of coordinates which can be joined together to form a skeleton.

    The body's sparsevol is fetched from DVID at the given scale.  The sparsevol is processed in big
    overlapping chunks ("blocks"), each of which is skeletonized with scikit-image after a bit of
    preprocessing (morphological closing, followed by hole filling).

    Args:
        dvid_server:
            The DVID server to connect to.
        uuid:
            The UUID of the DVID instance to connect to.
        segmentation_instance:
            The name of the segmentation volume to use.
        body:
            The ID of the body to skeletonize.
        scale:
            The scale at which to fetch the sparsevol.
        block_shape:
            The processing will be split into to blocks of this shape (not including the halo).
        halo:
            The amount of overlap between blocks.
            Since we skeletonize each block independently, this halo minimizes edge effects between blocks.
            The skeleton points in the halo region are discarded before returning unless trim_halo is False.
        closing_radius:
            The radius of the morphological closing operation, applied before hole-filling and skeletonization.
        trim_halo:
            If False, DON'T discard coordinates in the halo region.
            (Useful for debugging.)
        return_radii:
            If True, also return an approximate "radius" for each skeleton point.
            This is computed via the distance transform within the neuron mask.
        threads:
            Blocks are processed in parallel using a thread pool of this size.
    Returns:
        (coords_zyx, radii)
            coords_zyx has shape (N, 3) and contains the skeleton points. Note that the coordinates are given in Z,Y,X order.
            If return_radii is True, then radii has shape (N,) and contains the mask radius for each skeleton point.
            If return_radii is False, then radii is None.
    """
    dvid_seg = (dvid_server, uuid, segmentation_instance)
    ranges = fetch_sparsevol(*dvid_seg, body, scale=scale, format='ranges')
    mask_boxes, mask_iterator = blockwise_masks_from_ranges(ranges, block_shape=block_shape, halo=halo)
    
    def threadsafe_generator(g):
        lock = threading.Lock()
        with lock:
            yield from g

    # I don't think it's safe for multiple threads to use
    # mask_iterator simultaneously, so I wrapped it with a lock.
    mask_iterator = threadsafe_generator(mask_iterator)

    # The function to process one block.
    _process_block = partial(
        _skeleton_coords_for_mask,
        closing_radius=closing_radius,
        halo=halo,
        trim_halo=trim_halo,
        return_radii=return_radii
    )

    # Use a thread pool.
    results = compute_parallel(
        _process_block,
        zip(mask_boxes, mask_iterator),
        starmap=True,
        total=len(mask_boxes),
        leave_progress=True,
        threads=threads,
    )

    all_coords, all_radii = zip(*results)
    all_coords = np.concatenate(all_coords)
    all_coords *= 2**scale

    if not return_radii:
        return all_coords, None

    all_radii = np.concatenate(all_radii)
    all_radii *= 2**scale
    return all_coords, all_radii


def _skeleton_coords_for_mask(box, mask, closing_radius, halo, trim_halo, return_radii):
    """
    Helper for skeleton_coords().  Processes a single block of the sparsevol.
    """
    # We only care about the part that actually contains non-zero values.
    # (Note that the closing operation would not result in pixels outside the bounding box.)
    nz_box = compute_nonzero_box(mask)
    mask = extract_subvol(mask, nz_box)

    if closing_radius > 0:
        # After trying skimage and scipy, vigra seems ~2x faster at closing.
        mask = vigra.filters.multiBinaryClosing(mask, radius=closing_radius)

    mask = fill_holes(mask)
    skeleton = skimage.morphology.skeletonize(mask)
    coords = np.array(skeleton.nonzero()).T

    if return_radii:
        dt = distance_transform(mask)
        radii = np.array(dt[tuple(coords.T)])

    coords += box[0] + nz_box[0]
    
    if trim_halo:
        non_halo = (coords >= box[0] + halo).all(axis=1) & (coords < box[1] - halo).all(axis=1)
        coords = coords[non_halo]
        radii = radii[non_halo]

    if return_radii:
        return coords, radii
    else:
        return coords, None


def fill_holes(mask):
    """
    Fill holes in a binary mask.
    Computes the connected components of the mask (inside and out),
    and fills all non-masked regions except for the largest one
    (which we presume is the true background).
    """
    if mask.all():
        return mask
    cc = skimage.measure.label(mask.astype(np.uint8), background=99, connectivity=1)
    nonmask_cc_counts = np.bincount(cc[~mask])
    background_cc = np.argmax(nonmask_cc_counts)
    return (cc != background_cc)


def treeify_coords(all_coords, radii=None):
    """
    Given an array of coordinates, join them into a minimum spanning tree.
    We only consider possible edges between each point and its N closest neighbors,
    so it is possible for the output to consist of multiple connected components.

    Args:
        all_coords:
            An array of shape (N, 3) containing the coordinates of the points to join.
        radii:
            Optional column of radii to include in the result as column.
            The radii are not used in the computation of the MST.
    
    Returns:
        A pandas DataFrame with columns:
            ['node', 'x', 'y', 'z', 'parent', 'cc', 'radius']
        
        (If radii are not provided, the 'radius' column is omitted.)
    """
    # Select edges for every node's N closest neighbors
    # (Ignore the first "neighbor", which is the node itself.)    
    # TODO:
    #   Consider using a max radius instead (or in addition to)
    #   of a fixed number of neighbors.
    num_neighbors = 7
    kdtree = KDTree(all_coords)
    distances, neighbors = kdtree.query(all_coords, k=tuple(range(2, 2+num_neighbors)))
    nodes = np.arange(len(all_coords))
    edges = []
    for i in range(num_neighbors):
        nth_edges = pd.DataFrame({
            'u': nodes,
            'v': neighbors[:, i],
            'distance': distances[:, i]
        })
        edges.append(nth_edges)
    edges = pd.concat(edges, ignore_index=True)

    # Load into nx.Graph and compute MST
    g = nx.Graph()
    for row in edges.itertuples():
        g.add_edge(row.u, row.v, weight=row.distance)

    dfs_edges = []
    mst = nx.minimum_spanning_tree(g, weight='weight')
    for component in nx.connected_components(mst):
        root = min(component)
        dfs_edges.extend(nx.dfs_edges(mst, source=root))
        dfs_edges.append((-1, root))

    df = pd.DataFrame(dfs_edges, columns=['parent', 'node'])
    df[[*'zyx']] = all_coords[df['node']]

    df = df.set_index('node').sort_index()
    df['cc'] = np.int32(-1)
    g = nx.Graph()
    g.add_edges_from(dfs_edges)
    for i, cc in enumerate(nx.connected_components(g)):
        for node in cc:
            if node == -1:
                continue
            df.loc[node, 'cc'] = i

    cols = ['node', *'xyz', 'parent', 'cc']
    if radii is not None:
        df['radius'] = radii[df.index]
        cols.append('radius')
    return df.reset_index()[cols]
