# builtin
import sys
import json
import argparse
import logging
from collections import namedtuple

# third-party
import numpy as np
import pandas as pd

# flyem
from dvidutils import LabelMapper

# local
from .util import Timer, read_csv_col, NumpyConvertingEncoder
from .merge_table import load_mapping
from .merge_graph import LabelmapMergeGraph

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-mapping', required=False)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--stop-after', required=False)
    #parser.add_argument('--subset-bodies', required=False)
    parser.add_argument('--output-path', '-o', required=True)
    parser.add_argument('dvid_server')
    parser.add_argument('dvid_uuid')
    parser.add_argument('dvid_instance')
    parser.add_argument('original_mapping')
    parser.add_argument('important_bodies')
    parser.add_argument('speculative_merge_table')
    args = parser.parse_args()

    instance_info = DvidInstanceInfo(args.dvid_server, args.dvid_uuid, args.dvid_instance)

    # Note: speculative merge table must include original merge graph, at least for 'unimportant' bodies.
    all_paths = compute_focused_paths( args.speculative_merge_table,
                                       args.original_mapping,
                                       args.important_bodies,
                                       instance_info,
                                       args.split_mapping,
                                       args.max_depth,
                                       args.stop_after )
    
    with open(args.output_path) as f:
        json.dump(f, all_paths, cls=NumpyConvertingEncoder)
    
    logger.info("DONE.")


DvidInstanceInfo = namedtuple("DvidInstanceInfo", "server uuid instance")
def compute_focused_paths( instance_info,
                           original_mapping,
                           important_bodies,
                           speculative_merge_tables,
                           split_mapping=None,
                           max_depth=10,
                           stop_after_endpoint_num=None,
                           return_after_setup=False ):

    instance_info = DvidInstanceInfo(*instance_info)
    with Timer("Loading speculative merge graph", logger):
        merge_graph = LabelmapMergeGraph(speculative_merge_tables, instance_info.uuid)   

    if split_mapping is not None:
        _bad_edges = merge_graph.append_edges_for_split_supervoxels( split_mapping,
                                                                     instance_info.server,
                                                                     instance_info.uuid,
                                                                     instance_info.instance,
                                                                     parent_sv_handling='drop' )
    merge_table_df = merge_graph.merge_table_df
        
    if isinstance(original_mapping, str):
        with Timer("Loading mapping", logger):
            original_mapping = load_mapping(original_mapping)
    else:
        assert isinstance(original_mapping, pd.Series)

    with Timer("Applying mapping", logger):
        mapper = LabelMapper(original_mapping.index.values, original_mapping.values)
        merge_table_df['body_a'] = mapper.apply(merge_table_df['id_a'].values, allow_unmapped=True)
        merge_table_df['body_b'] = mapper.apply(merge_table_df['id_b'].values, allow_unmapped=True)

    with Timer("Reading importances", logger):
        if isinstance(important_bodies, str):
            important_bodies = read_csv_col(important_bodies).values
        important_bodies = pd.Index(important_bodies, dtype=np.uint64)

    with Timer("Assigning importances", logger):
        merge_table_df['important_a'] = merge_table_df['body_a'].isin(important_bodies)
        merge_table_df['important_b'] = merge_table_df['body_b'].isin(important_bodies)

    with Timer("Discarding merged edges within 'important' bodies ", logger):
        size_before = len(merge_table_df)
        merge_table_df.query('(body_a != body_b) and not (important_a and important_b)', inplace=True)
        size_after = len(merge_table_df)
        logger.info(f"Discarded {size_before - size_after} edges")

    edges = merge_table_df[['id_a', 'id_b']].values
    assert edges.dtype == np.uint64

    if return_after_setup:
        logger.info("Returning setup instead of searching for paths")
        return (edges, original_mapping, important_bodies, max_depth, stop_after_endpoint_num)

    logger.info(f"Finding paths among {len(important_bodies)} important bodies")
    # FIXME: Need to augment original_mapping with rows for single-sv bodies that are 'important'
    all_paths = find_all_paths(edges, original_mapping, important_bodies, max_depth, stop_after_endpoint_num)
    return all_paths


def find_all_paths(edges, original_mapping, important_bodies, max_depth=10, stop_after_endpoint_num=None):
    """
    Returns a dict of { start_sv: VertexPath }
    """
    with Timer("Loading into gt.Graph", logger):
        import graph_tool as gt
        g = gt.Graph(directed=False)
        sv_pmap = g.add_edge_list( edges, hashed=True )

    # Mappings from Graph vertices (which are consecutive) to SV ids and vice-versa
    v_to_sv = sv_pmap.get_array().astype(np.uint64)
    sv_to_v = pd.Series(index=v_to_sv, data=np.arange(len(v_to_sv), dtype=np.uint32))

    with Timer("Mapping SV importances"):
        sv_importances = pd.DataFrame(original_mapping)
        sv_importances['v'] = sv_to_v.loc[original_mapping.index]
        sv_importances['important'] = original_mapping.index.isin(important_bodies)
        
        important_svs = sv_importances[sv_importances['important']].index
        important_verts = pd.Index(sv_importances[sv_importances['important']]['v'])

    if stop_after_endpoint_num is None:
        stop_after_endpoint_num = len(important_svs)

    with Timer(f"Finding paths from {stop_after_endpoint_num} endpoints", logger):
        all_paths = {}
        for sv in important_svs[:stop_after_endpoint_num]:
            v = sv_to_v.loc[sv]
            v_paths = []
            find_paths_from( max_depth,
                             important_verts,
                             g,
                             v,
                             v_paths,
                             current_path=[v],
                             current_path_edge_ids=[None],
                             current_depth=0 )
    
            # map v_paths to sv_paths
            sv_paths = []
            for v_path in v_paths:
                sv_paths.append(v_to_sv[v_path.vertices])
            all_paths[sv] = sv_paths

    return all_paths

VertexPath = namedtuple("VertexPath", "vertices edge_ids")
def find_paths_from(max_depth, important_verts, g, v, v_paths, current_path, current_path_edge_ids, current_depth):
    """
    """
    if current_depth > max_depth:
        return

    _sources, targets, edge_ids = g.get_out_edges(v).transpose()    
    for t, eid in zip(targets, edge_ids):
        if t in current_path:
            continue

        current_path.append(t)
        current_path_edge_ids.append(eid)
        if t in important_verts:
            # Endpoint found.  Path is complete.
            v_paths.append( VertexPath(list(current_path), list(current_path_edge_ids)) )
        else:
            # Keep going
            find_paths_from(max_depth, important_verts, g, t, v_paths, current_path, current_path_edge_ids, current_depth+1)
        current_path.pop()
        current_path_edge_ids.pop()


if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    main()    
