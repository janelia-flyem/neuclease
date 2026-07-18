import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import navis

from neuclease.dvid import fetch_skeleton, fetch_label, fetch_instance_info
from neuclease.misc.skeletonize import heal_skeleton


def tree_neuron_from_dvid(dvid_server, uuid, seg_instance, synapse_instance, soma_instance, body, heal=np.inf, anisotropy_xyz=None):
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

    if len(syn_df) or len(soma_df):
        # Scale by anisotropy_xyz so that nearest-node distances are measured
        # in physical space rather than (possibly anisotropic) voxel units.
        tree = cKDTree(n.nodes[['x', 'y', 'z']].values * anisotropy_xyz)

    if len(syn_df):
        _, node_ix = tree.query(syn_df[['x', 'y', 'z']].values * anisotropy_xyz)
        n.connectors = pd.DataFrame({
            'connector_id': np.arange(len(syn_df)),
            'node_id': n.nodes['node_id'].values[node_ix],
            'x': syn_df['x'].values,
            'y': syn_df['y'].values,
            'z': syn_df['z'].values,
            'type': syn_df['kind'].map({'PreSyn': 'pre', 'PostSyn': 'post'}).values,
        })

    if len(soma_df):
        # If more than one soma point was annotated, just take first one in scan order.
        soma_xyz = soma_df[['x', 'y', 'z']].sort_values([*'zyx']).values[:1]
        _, node_ix = tree.query(soma_xyz * anisotropy_xyz)
        n.soma = n.nodes['node_id'].values[node_ix[0]]
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
