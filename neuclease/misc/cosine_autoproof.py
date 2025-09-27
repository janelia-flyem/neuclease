import argparse
import copy
import json
import logging
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from neuprint import Client, fetch_neurons, fetch_adjacencies, fetch_mean_synapses, NeuronCriteria as NC, NotNull
from neuclease.util import compute_parallel, tqdm_proxy
from neuclease.misc.neuroglancer import parse_nglink, layer_dict, segment_properties_json, upload_to_bucket, upload_ngstate

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore_connections_below', type=int, default=10)
    parser.add_argument('--processes', type=int, default=0)
    parser.add_argument('--max-target-types', type=int, default=20)
    parser.add_argument('neuprint_server', type=str)
    parser.add_argument('neuprint_dataset', type=str)
    parser.add_argument('template_link', type=str)
    parser.add_argument('bucket_dir', type=str)
    parser.add_argument('orphans_csv', type=str)
    parser.add_argument('output_targets_csv', type=str, nargs='?')
    parser.add_argument('output_links_csv', type=str, nargs='?')

    args = parser.parse_args()

    from neuclease import configure_default_logging
    configure_default_logging()

    c = Client(args.neuprint_server, args.neuprint_dataset, progress=False)
    threshold_strength = args.ignore_connections_below

    if not args.orphans_csv.endswith('.csv') and str.isalnum(args.orphans_csv):
        orphan_df = pd.DataFrame({'orphan': [int(args.orphans_csv)]})
        args.orphans_csv = args.orphans_csv + '.csv'
    else:
        orphan_df = pd.read_csv(args.orphans_csv)

    orphans_df = orphan_df.rename(columns={'body': 'orphan', 'bodyId': 'orphan'})
    orphans = orphans_df['orphan'].tolist()

    if not args.output_targets_csv:
        args.output_targets_csv = args.orphans_csv.replace('.csv', '_targets.csv')
    if not args.output_links_csv:
        args.output_links_csv = args.orphans_csv.replace('.csv', '_links.csv')

    show_internal_progress = (args.processes == 0)
    results = compute_parallel(
        partial(_process_orphan, threshold_strength, args.max_target_types, args.template_link, args.bucket_dir, c, show_internal_progress),
        orphans,
        processes=args.processes,
        show_progress=True,
    )

    urls = []
    targets = []
    for orphan, orphan_roi, targets_df, url in results:
        if orphan_roi is None:
            continue
        targets.append(targets_df)
        urls.append((orphan, orphan_roi, url))

    targets_df = pd.concat(targets, ignore_index=True)
    targets_df = orphans_df.merge(targets_df, 'inner', on='orphan')
    targets_df.to_csv(args.output_targets_csv, index=False)
    print(f"Wrote {len(targets_df)} targets to {args.output_targets_csv}")

    url_df = pd.DataFrame(urls, columns=['orphan', 'orphan_roi', 'url'])
    url_df = orphans_df.merge(url_df, 'inner', on='orphan')
    url_df.to_csv(args.output_links_csv, index=False)
    print(f"Wrote {len(urls)} links to {args.output_links_csv}")


def _process_orphan(threshold_strength, max_target_types, template_link, bucket_dir, client, show_progress, orphan):
    orphan_roi, targets_df = fetch_orphan_targets(orphan, threshold_strength, client, show_progress)
    if orphan_roi is None:
        return orphan, None, None, None
    url = _neuroglancer_link(orphan, max_target_types, template_link, targets_df, bucket_dir, client)
    return orphan, orphan_roi, targets_df, url


def fetch_orphan_targets(orphan, threshold_strength, client, show_progress):
    _, orphan_syndist = fetch_neurons(orphan, client=client)
    orphan_syndist['synweight'] = orphan_syndist.eval('upstream + downstream')
    orphan_rois = orphan_syndist.query('roi in @client.primary_rois').sort_values('synweight', ascending=False)['roi']
    if len(orphan_rois) == 0:
        return None, None
    orphan_roi = orphan_rois.iloc[0]

    orphan_type_strengths, orphan_upstream_types, orphan_downstream_types = _orphan_type_strengths(orphan, orphan_roi, client)
    try:
        target_type_strengths = _target_type_strengths(orphan, orphan_upstream_types, orphan_downstream_types, orphan_roi, client)
    except Exception as e:
        logger.error(f"Error fetching target type strengths for {orphan}, {orphan_roi}: {e}")
        return None, None

    if len(target_type_strengths) == 0:
        return None, None

    # If the orphan connects to some cell type that none of the targets connect to,
    # then the orphan table (just one row) will have column(s) for that type that
    # the target table lacks. If not for that, the columns in the orphan and target
    # dataframes would already be identical. But since they're not, we must explicitly align their columns.
    assert set(orphan_type_strengths.columns) >= set(target_type_strengths.drop(columns=['type']).columns)
    target_type_strengths = target_type_strengths.reindex(columns=[*orphan_type_strengths.columns, 'type'], fill_value=0)
    assert (target_type_strengths.drop(columns=['type']).columns == orphan_type_strengths.columns).all()

    improvements = _improvements(orphan, orphan_type_strengths, target_type_strengths, threshold_strength, show_progress)
    return orphan_roi, improvements


def _orphan_type_strengths(orphan, orphan_roi, client):
    bodies_downstream_of_orphan, conn_downstream_of_orphan = fetch_adjacencies(orphan, NC(type=NotNull), rois=orphan_roi, client=client)
    bodies_upstream_of_orphan, conn_upstream_of_orphan = fetch_adjacencies(NC(type=NotNull), orphan, rois=orphan_roi, client=client)

    orphan_downstream_types = bodies_downstream_of_orphan.query('bodyId != @orphan')['type'].unique()
    orphan_upstream_types = bodies_upstream_of_orphan.query('bodyId != @orphan')['type'].unique()

    conn_downstream_of_orphan['type_post'] = conn_downstream_of_orphan['bodyId_post'].map(bodies_downstream_of_orphan.set_index('bodyId')['type'])
    conn_upstream_of_orphan['type_pre'] = conn_upstream_of_orphan['bodyId_pre'].map(bodies_upstream_of_orphan.set_index('bodyId')['type'])

    upstream_of_orphan_types_conn = conn_upstream_of_orphan.groupby(['bodyId_post', 'type_pre'])['weight'].sum().unstack()
    downstream_of_orphan_types_conn = conn_downstream_of_orphan.groupby(['bodyId_pre', 'type_post'])['weight'].sum().unstack()

    upstream_of_orphan_types_conn.columns = [f'{c}_pre' for c in upstream_of_orphan_types_conn.columns]
    downstream_of_orphan_types_conn.columns = [f'{c}_post' for c in downstream_of_orphan_types_conn.columns]

    orphan_type_strengths = pd.concat([upstream_of_orphan_types_conn, downstream_of_orphan_types_conn], axis=1).fillna(0).astype(int)
    return orphan_type_strengths, orphan_upstream_types, orphan_downstream_types


def _target_type_strengths(orphan, orphan_upstream_types, orphan_downstream_types, orphan_roi, client):
    if len(orphan_upstream_types) == 0:
        bodies_downstream_of_upstream = pd.DataFrame({'bodyId': [], 'type': []})
        conn_downstream_of_upstream = pd.DataFrame({'bodyId_pre': [], 'bodyId_post': [], 'weight': [], 'type_pre': [], 'type_post': []})
    else:
        bodies_downstream_of_upstream, conn_downstream_of_upstream = fetch_adjacencies(NC(type=orphan_upstream_types), NC(type=NotNull), rois=orphan_roi, client=client)

    if len(orphan_downstream_types) == 0:
        bodies_upstream_of_downstream = pd.DataFrame({'bodyId': [], 'type': []})
        conn_upstream_of_downstream = pd.DataFrame({'bodyId_pre': [], 'bodyId_post': [], 'weight': [], 'type_pre': [], 'type_post': []})
    else:
        bodies_upstream_of_downstream, conn_upstream_of_downstream = fetch_adjacencies(NC(type=NotNull), NC(type=orphan_downstream_types), rois=orphan_roi, client=client)

    if len(bodies_downstream_of_upstream) == 0 and len(bodies_upstream_of_downstream) == 0:
        return pd.DataFrame({'type': [], 'type_pre': [], 'type_post': []})

    bodies_downstream_of_upstream = bodies_downstream_of_upstream.query('bodyId != @orphan')
    bodies_upstream_of_downstream = bodies_upstream_of_downstream.query('bodyId != @orphan')
    conn_upstream_of_downstream = conn_upstream_of_downstream.query('bodyId_pre != @orphan and bodyId_post != @orphan').copy()
    conn_downstream_of_upstream = conn_downstream_of_upstream.query('bodyId_pre != @orphan and bodyId_post != @orphan').copy()

    conn_downstream_of_upstream['type_pre'] = conn_downstream_of_upstream['bodyId_pre'].map(bodies_downstream_of_upstream.set_index('bodyId')['type'])
    conn_upstream_of_downstream['type_post'] = conn_upstream_of_downstream['bodyId_post'].map(bodies_upstream_of_downstream.set_index('bodyId')['type'])

    assert set(conn_downstream_of_upstream['type_pre'].unique()) <= set(orphan_upstream_types), \
        f"{orphan}, {orphan_roi}: {set(conn_downstream_of_upstream['type_pre'].unique())} <= {set(orphan_upstream_types)}"
    assert set(conn_upstream_of_downstream['type_post'].unique()) <= set(orphan_downstream_types), \
        f"{orphan}, {orphan_roi}: {set(conn_upstream_of_downstream['type_post'].unique())} <= {set(orphan_downstream_types)}"

    downstream_of_upstream_types_conn = conn_downstream_of_upstream.groupby(['bodyId_post', 'type_pre'])['weight'].sum().unstack()
    upstream_of_downstream_types_conn = conn_upstream_of_downstream.groupby(['bodyId_pre', 'type_post'])['weight'].sum().unstack()

    downstream_of_upstream_types_conn.columns = [f'{c}_pre' for c in downstream_of_upstream_types_conn.columns]
    upstream_of_downstream_types_conn.columns = [f'{c}_post' for c in upstream_of_downstream_types_conn.columns]

    target_type_strengths = pd.concat([downstream_of_upstream_types_conn, upstream_of_downstream_types_conn], axis=1).fillna(0).astype(int)

    types = fetch_neurons(target_type_strengths.index, omit_rois=True, client=client).set_index('bodyId')['type']
    target_type_strengths['type'] = target_type_strengths.index.map(types)
    return target_type_strengths


def _improvements(orphan, orphan_type_strengths, target_type_strengths, threshold_strength, show_progress):
    improvements = []
    progress = tqdm_proxy(total=len(target_type_strengths), leave=False, disable=not show_progress)
    with progress:
        for t, ts in target_type_strengths.groupby('type'):
            if len(ts) == 1:
                progress.update(1)
                continue

            progress.set_description_str("{:10}".format(f"{t}:"))

            ts = ts.drop(columns=['type'])

            # For each body in this type group, compute the cosine simiilarity between the body and the mean of its peers.
            # Then try adding the orphan to the body and compute similarity again.  Does it get better or worse?

            for body in ts.index:
                body_ts = ts.loc[body:body].values
                body_ts[body_ts < threshold_strength] = 0

                peer_ts = ts.drop(body)
                peer_mean_ts = peer_ts.mean()
                peer_mean_ts[peer_mean_ts < threshold_strength] = 0
                cs = cosine_similarity(body_ts, peer_mean_ts.values[None])

                merged_body_ts = ts.loc[body:body].values + orphan_type_strengths.values
                merged_body_ts[merged_body_ts < threshold_strength] = 0
                merged_cs = cosine_similarity(merged_body_ts, peer_mean_ts.values[None])

                improvement = (merged_cs - cs)[0, 0]
                improvements.append((
                    orphan,
                    body,
                    t,
                    improvement,
                    (body_ts > 0).any().sum(),
                    (ts > 0).any().sum()
                ))
                progress.update(1)

    columns = ['orphan', 'target', 'type', 'score', 'similar_body_connections', 'similar_type_connections']
    improvements = (
        pd.DataFrame(improvements, columns=columns)
        .sort_values(['similar_body_connections', 'similar_type_connections', 'score'], ascending=False)
    )
    return improvements


def _neuroglancer_link(orphan, max_target_types, template_link, target_df, bucket_dir, client):
    segment_properties = segment_properties_json(
        target_df.set_index('target').rename_axis('body')
        [['type', 'similar_body_connections', 'similar_type_connections', 'score']],
        'type'
    )
    assert bucket_dir.startswith('gs://')
    bucket, *_dir_parts = bucket_dir[len('gs://'):].split('/')
    segprop_path = '/'.join(_dir_parts) + f'/orphan_{orphan}_segment_properties/info'
    upload_to_bucket(bucket, segprop_path, json.dumps(segment_properties), disable_cache=True)
    segprop_src =  {
        "url": f"precomputed://gs://{bucket}/{segprop_path}"[:-len('/info')],
        "subsources": {
            "default": True,
        },
        "enableDefaultSubsources": False,
    }

    mean_synapses = fetch_mean_synapses(orphan, by_roi=False, client=client)
    mean_synapses[[*'xyz']] *= mean_synapses[['count']].values
    mean_position = mean_synapses[[*'xyz']].sum().values / mean_synapses['count'].sum()
    mean_position = mean_position.astype(int).tolist()

    state = parse_nglink(template_link)
    state['title'] = f"Orphan {orphan}: Proposed targets"
    state["position"] = mean_position
    state["layerListPanel"] = {
        "visible": True
    }
    
    layers = layer_dict(state)
    if 'orphan' not in layers or 'targets' not in layers:
        raise ValueError("The template link must have 'orphan' and 'targets' layers")

    target_types_df = (
        target_df
        .reset_index()
        .groupby('type', sort=False)
        .agg({
            'target': list,
            'score': 'max',
            'similar_body_connections': 'max',
            'similar_type_connections': 'max'
        })
        .sort_values(['similar_body_connections', 'similar_type_connections', 'score'], ascending=False)
        .iloc[:max_target_types]
    )

    orphan_layer = layers['orphan']
    orphan_layer['name'] = f"orphan-{orphan}"
    orphan_layer['segments'] = [str(orphan)]
    orphan_layer['segmentQuery'] = str(orphan)

    layers['orphan-synapses']["linkedSegmentationLayer"] = {
        "segments": orphan_layer['name']
      }

    targets_layer = layers['targets']
    targets_position = list(layers.keys()).index('targets')
    del state['layers'][targets_position]

    for celltype, targets in target_types_df['target'].items():
        layer = copy.deepcopy(targets_layer)
        layer['name'] = celltype
        layer['segments'] = [*map(str, targets)]
        layer['segmentQuery'] = ", ".join(layer['segments'])
        layer['source'].append(segprop_src)
        state['layers'].insert(targets_position, layer)
        targets_position += 1

    bucket_path = f"{bucket_dir}/orphan_{orphan}_proposed_targets.json"
    url = upload_ngstate(bucket_path, state, True, 'https://clio-ng.janelia.org')
    return url


if __name__ == '__main__':
    main()
