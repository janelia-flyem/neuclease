import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import navis

from neuclease.dvid import fetch_skeleton, fetch_label, fetch_instance_info
from neuclease.misc.skeletonize import heal_skeleton


def tree_neuron_from_dvid(dvid_server, uuid, seg_instance, synapse_instance, soma_instance, body,
                           heal=np.inf, anisotropy_xyz=None, min_synapse_soma_dist_nm=None):
    """
    Construct a navis TreeNeuron for a single body, using data fetched from DVID:
    the skeleton (nodes), the synapses (as connectors), and the soma location (if any).

    Args:
        dvid_server, uuid:
            DVID node to fetch from.
        seg_instance:
            Name of the segmentation instance, e.g. 'segmentation'.
            The skeleton is fetched from f"{seg_instance}_skeletons".
        synapse_instance:
            Name of the DVID annotation instance holding synapse points (e.g. 'synapses').
        soma_instance:
            Name of the DVID annotation instance holding soma annotations (e.g. 'segmentation_soma').
        body:
            Body ID to fetch.
        heal:
            Max distance (in units of anisotropy_xyz) for stitching disconnected
            skeleton fragments together via heal_skeleton(). Use None to skip healing.
        anisotropy_xyz:
            Physical size of a voxel in nm, in XYZ order. If not provided, it's
            fetched from the segmentation instance's metadata.
        min_synapse_soma_dist_nm:
            If given, discard any synapse whose distance to the soma is less than
            this many nanometers (e.g. to filter out unreliable synapse detections
            near the soma). Distance is measured along the skeleton (from the
            synapse's nearest node to the soma's node), plus the (Euclidean) offset
            between the synapse and its nearest node. Ignored if the body has no soma.

    Returns:
        navis.TreeNeuron
    """
    if anisotropy_xyz is None:
        anisotropy_xyz = fetch_instance_info(dvid_server, uuid, seg_instance)['Extended']['VoxelSize']
    anisotropy_xyz = np.asarray(anisotropy_xyz, dtype=float)

    skel_df = fetch_skeleton(dvid_server, uuid, f"{seg_instance}_skeletons", body, format='pandas')
    skel_df = skel_df.drop(columns=['kind'])

    if heal is not None:
        skel_df = heal_skeleton(skel_df, max_distance=heal, anisotropy_xyz=anisotropy_xyz)
        skel_df = skel_df.drop(columns=['cc'])

    n = navis.TreeNeuron(skel_df, units=_voxel_size_to_units(anisotropy_xyz))
    n.id = body

    # Columns: ['x', 'y', 'z', 'kind', ...] where 'kind' is either 'PreSyn' or 'PostSyn'
    syn_df = fetch_label(dvid_server, uuid, synapse_instance, body, format='pandas')

    # Columns: ['x', 'y', 'z', 'kind', ...] where 'kind' is always 'Note'
    # If it's empty, then the body has no annotated soma.
    soma_df = fetch_label(dvid_server, uuid, soma_instance, body, format='pandas')

    soma_xyz = None
    if len(soma_df):
        # If more than one soma point was annotated, just take first one in scan order.
        soma_xyz = soma_df[['x', 'y', 'z']].sort_values([*'zyx']).values[:1]

    node_xyz_nm = None
    if len(syn_df) or soma_xyz is not None:
        # Scale by anisotropy_xyz so that nearest-node distances are measured
        # in physical space rather than (possibly anisotropic) voxel units.
        node_xyz_nm = n.nodes[['x', 'y', 'z']].values * anisotropy_xyz
        tree = cKDTree(node_xyz_nm)

    soma_node_id = None
    if soma_xyz is not None:
        _, node_ix = tree.query(soma_xyz * anisotropy_xyz)
        soma_node_id = n.nodes['node_id'].values[node_ix[0]]

    if len(syn_df):
        syn_offset_nm, node_ix = tree.query(syn_df[['x', 'y', 'z']].values * anisotropy_xyz)
        syn_node_ids = n.nodes['node_id'].values[node_ix]

        if min_synapse_soma_dist_nm is not None and soma_node_id is not None:
            # Distance from the synapse to the soma = distance along the skeleton
            # from the synapse's nearest node to the soma's node, plus the
            # (Euclidean) offset between the synapse and that node.
            nodes_nm = n.nodes[['node_id', 'parent_id']].copy()
            nodes_nm[['x', 'y', 'z']] = node_xyz_nm
            n_nm = navis.TreeNeuron(nodes_nm)

            geo = navis.geodesic_matrix(n_nm, from_=[soma_node_id])
            geo_dist_nm = geo.loc[soma_node_id][syn_node_ids].values

            total_dist_nm = geo_dist_nm + syn_offset_nm
            keep = total_dist_nm >= min_synapse_soma_dist_nm
            syn_df = syn_df[keep]
            syn_node_ids = syn_node_ids[keep]

        n.connectors = pd.DataFrame({
            'connector_id': np.arange(len(syn_df)),
            'node_id': syn_node_ids,
            'x': syn_df['x'].values,
            'y': syn_df['y'].values,
            'z': syn_df['z'].values,
            'type': syn_df['kind'].map({'PreSyn': 'pre', 'PostSyn': 'post'}).values,
        })

    if soma_node_id is not None:
        n.soma = soma_node_id
        n.reroot(n.soma, inplace=True)
    else:
        n.soma = None

    return n


def _voxel_size_to_units(voxel_size_xyz):
    """Convert a voxel size (in nm, XYZ order) into a navis-compatible units string/list."""
    voxel_size_xyz = np.asarray(voxel_size_xyz, dtype=float)
    if (voxel_size_xyz == voxel_size_xyz[0]).all():
        return f"{voxel_size_xyz[0]} nm"
    return [f"{vs} nm" for vs in voxel_size_xyz]
