import re
import copy
from itertools import chain

import requests
import numpy as np
import pandas as pd

from neuclease.dvid.keyvalue import DEFAULT_BODY_STATUS_CATEGORIES

from neuclease.util import tqdm_proxy
from neuclease.misc.neuroglancer import parse_nglink, layer_dict, upload_json, segment_properties_json
from neuprint.wrangle import bilateral_syndist, syndist_matrix, assign_sides_in_groups


def review_central_brain_groups(template_link, task_bucket, neurons=None, syndist=None, group_size=2):
    from neuprint import Client, fetch_neurons, NeuronCriteria as NC
    template_state = parse_nglink(template_link)

    c = Client('neuprint-cns.janelia.org', 'cns')
    if neurons is None or syndist is None:
        neurons, syndist = fetch_neurons(NC())

    syndist = syndist.rename(columns={'vnc-neuopil-shell': 'VNC'})

    olr = {'ME(R)', 'AME(R)', 'LA(R)', 'LO(R)', 'LOP(R)'}
    oll = {'ME(L)', 'AME(L)', 'LA(L)', 'LO(L)', 'LOP(L)'}
    vnc = {'VNC', 'vnc-neuropil-shell'}
    nerves = {roi for roi, info in c.meta['roiInfo'].items() if info['isNerve']}
    cb = set(c.meta['primaryRois']) - olr - oll - nerves - vnc  # noqa

    olr_bodies = set(syndist.query('roi in @olr')['bodyId'].unique())
    oll_bodies = set(syndist.query('roi in @oll')['bodyId'].unique())
    vnc_bodies = set(syndist.query('roi in @vnc')['bodyId'].unique())
    _cb_bodies = set(syndist.query('roi in @cb')['bodyId'].unique())
    cb_only_bodies = _cb_bodies - olr_bodies - oll_bodies - vnc_bodies

    task_links = review_groups(neurons, syndist, cb_only_bodies, c.primary_rois, template_state, 'brain-neuropils', task_bucket, group_size=group_size)
    task_links = pd.Series(task_links, name='link').rename_axis('group')
    if group_size is None:
        group_size = 'all'
    task_links.to_csv(f'central-group-tasks-{group_size}.csv', index=True, header=True)
    return task_links


def review_groups(neurons, syndist, bodies, primary_rois, template_state, roi_layer, task_bucket, untraced_only=True, group_size=None):
    neurons.index = neurons['bodyId']
    neurons = neurons.query('bodyId in @bodies and group.notnull() and group != 0').copy()
    if untraced_only:
        status_dtype = pd.CategoricalDtype(DEFAULT_BODY_STATUS_CATEGORIES, ordered=True)
        neurons['statusLabel'] = neurons['statusLabel'].astype(status_dtype)
        neurons['is_traced'] = neurons['statusLabel'].fillna('') > 'Prelim Roughly traced'
        neurons['group_is_traced'] = neurons.groupby('group')['is_traced'].transform('all').fillna(False)
        neurons = neurons.loc[~neurons['group_is_traced']].copy()

    if group_size is not None:
        neurons['group_size'] = neurons.groupby('group').transform('size')
        neurons = neurons.loc[neurons['group_size'] == group_size].copy()

    syndist = syndist.query('roi in @primary_rois and bodyId in @neurons.bodyId')
    syndist = syndist[['bodyId', 'roi', 'pre', 'post']]

    syn_bidist = bilateral_syndist(syndist)
    syn_bidist_mat = syndist_matrix(syn_bidist, flatten_column_index=True)

    sides = assign_sides_in_groups(neurons, syndist, primary_rois)
    neurons = pd.concat(
        (
            neurons.set_index('bodyId'),
            sides,
            syn_bidist_mat,
        ),
        axis=1
    )

    roicols = sorted(syn_bidist_mat.columns)
    task_links = prepare_task_links(neurons, roicols, task_bucket, primary_rois, template_state, roi_layer)
    return task_links


def prepare_task_links(neurons, roicols, task_bucket, primary_rois, template_state, roi_layer):
    assert neurons.index.name == 'bodyId'
    roi_ids = determine_segment_ids(template_state, roi_layer)

    task_links = {}
    try:
        for group, df in tqdm_proxy(neurons.groupby('group'), total=neurons['group'].nunique()):
            task_links[group] = _prepare_task(group, df, roicols, task_bucket, primary_rois, template_state, roi_layer, roi_ids)
    except KeyboardInterrupt:
        print("Stopping early due to KeyboardInterrupt!")

    return task_links


def _prepare_task(group, df, roicols, task_bucket, primary_rois, template_state, roi_layer, roi_ids):
    assert task_bucket.startswith('gs://')
    group = int(group)

    SIGNIFICANT_PRE = 20
    SIGNIFICANT_POST = 100

    # Which ROI columns contain at least one neuron with a significant count?
    # We only show those ROIs and associated properties.
    pre_roi_df = df[[c for c in roicols if 'pre' in c]]
    post_roi_df = df[[c for c in roicols if 'post' in c]]
    sig_pre = pre_roi_df.loc[:, pre_roi_df.max() >= SIGNIFICANT_PRE].columns
    sig_post = post_roi_df.loc[:, post_roi_df.max() >= SIGNIFICANT_POST].columns
    sig_roi_props = sorted([*sig_pre, *sig_post])

    # Drop any that weren't found to be 'significant' above.
    # Since the properties are aggregated (LR), we have to match by prefix only.
    prefix_pattern = r'([^\(\)-]+)'
    sig_roi_prefixes = pd.Series([*sig_roi_props]).str.extract(prefix_pattern)[0].unique()

    roi_df = df[roicols]
    roi_spreads = (
        pd.concat(
            (
                roi_df.min().rename('min'),
                roi_df.mean().rename('mean'),
                roi_df.max().rename('max'),
            ),
            axis=1
        )
        .rename_axis('stat')
        .reset_index()
    )
    roi_spreads['spread'] = roi_spreads.eval('(max - min) / mean').fillna(0.0)

    # Neuroglancer has horrible formatting for floats,
    # so we will try to make this come out pretty by rounding a bit.
    roi_spreads['spread'] = (roi_spreads['spread'] * 32 // 1 / 32)

    roi_spreads[['roi', 'stat']] = roi_spreads['stat'].str.extract(r'(.*)-(.*)')
    roi_spreads = roi_spreads.set_index(['roi', 'stat']).unstack()
    roi_spreads.columns = [f"{stat}-{col}" for (col, stat) in roi_spreads.columns]

    # Keeping all of the stats (min, mean, max.) would be too many.
    roi_prop_names = ['pre-spread', 'post-spread']
    roi_spreads = roi_spreads[roi_prop_names]

    roi_spreads = roi_spreads.reset_index()
    roi_spreads['prefix'] = roi_spreads['roi'].str.extract(prefix_pattern)[0]
    roi_spreads = roi_spreads.query('prefix in @sig_roi_prefixes')
    roi_spreads = roi_spreads.drop(columns=['roi'])

    roi_ids_df = pd.Series(roi_ids, name='segment').rename_axis('roi').reset_index()
    roi_ids_df['prefix'] = roi_ids_df['roi'].str.extract(prefix_pattern)[0]
    roi_ids_df = roi_ids_df.query('prefix in @sig_roi_prefixes')

    # print('roi_spreads')
    # print(roi_spreads)
    # print('roi_ids_df')
    # print(roi_ids_df)
    # assert False

    roi_spreads = (
        roi_ids_df
        .merge(roi_spreads, 'left', on='prefix')
        .set_index('segment')
    )

    roi_segprops = segment_properties_json(roi_spreads.drop(columns=['prefix']), 'roi')
    roi_segprops_file = f'{task_bucket}/{group}/roi_properties/info'
    upload_json(roi_segprops, roi_segprops_file)

    prop_df = df.rename(columns={'statusLabel': 'status', 'status': '_status'})
    prop_df = prop_df.astype({'pre': np.int32, 'post': np.int32})
    prop_df = prop_df[['status', 'type', 'pre', 'post', *sig_roi_props]]

    assert prop_df.index.name == 'bodyId'
    segprops = segment_properties_json(prop_df.rename_axis('segment'), 'status')
    segprops_file = f'{task_bucket}/{group}/info'
    upload_json(segprops, segprops_file)

    sides = {}
    for bodyId, side in df['consensusSide'].items():
        sides.setdefault(side, []).append(bodyId)

    # Figure out which ROIs we actually need:
    # If we're mirroring left-onto-right, then reflect those ROIs
    df['use_mirror'] = False
    if set(df['consensusSide']) == {'L', 'R'}:
        df.loc[df['consensusSide'] == 'L', 'use_mirror'] = True

    df.loc[df['use_mirror'], 'inputRois'] = df.loc[df['use_mirror'], 'inputRois'].map(
        lambda rois: [
            roi.replace('(R)', '(L)') if '(R)' in roi else
            roi.replace('(L)', '(R)') if '(L)' in roi else
            roi
            for roi in rois
        ]
    )

    df.loc[df['use_mirror'], 'outputRois'] = df.loc[df['use_mirror'], 'outputRois'].map(
        lambda rois: [
            roi.replace('(R)', '(L)') if '(R)' in roi else
            roi.replace('(L)', '(R)') if '(L)' in roi else
            roi
            for roi in rois
        ]
    )

    selected_rois = {*chain(*df['inputRois'])} | {*chain(*df['outputRois'])}
    selected_rois &= {*primary_rois}

    # Keep only the ROI segments with a prefix that matches one of our sig_roi_prefixes
    selected_rois = {
        r for r in selected_rois
        if re.match(prefix_pattern, r).groups()[0] in sig_roi_prefixes
    }

    state = _task_state(template_state, sides, group, segprops_file, prop_df.columns[2:], selected_rois, roi_layer, roi_ids, roi_segprops_file, roi_prop_names)
    state_url = upload_json(state, f'{task_bucket}/{group}/state.json')
    state_blob = state_url[len('https://storage.googleapis.com/'):]
    return f'https://clio-ng.janelia.org/#!gs://{state_blob}'


def _task_state(template_state, sides, group, segprops_file, prop_names, selected_rois, roi_layer_name, roi_ids, roi_segprops_file, roi_prop_names):
    assert set(sides) <= {'L', 'R', None}, sides

    if isinstance(template_state, str):
        template_state = parse_nglink(template_state)

    state = copy.deepcopy(template_state)
    state['title'] = f'group-{group}'

    assert segprops_file.endswith('/info')
    if segprops_file.startswith('http'):
        prop_source = 'precomputed://' + segprops_file[:-len("/info")]
    elif segprops_file.startswith('gs://'):
        prop_source = f'precomputed://{segprops_file[:-len("/info")]}'
    else:
        raise AssertionError(f"bad segprop location: {segprops_file}")

    layers = layer_dict(state)
    roi_layer = layers[roi_layer_name]
    selected_rois = set(selected_rois) & set(roi_ids)
    roi_layer['segments'] = [*map(str, sorted(roi_ids[k] for k in selected_rois))]
    if isinstance(roi_layer['source'], list):
        old_roi_source = roi_layer['source'][0]
    else:
        old_roi_source = roi_layer['source']
        roi_layer['source'] = [old_roi_source]

    if isinstance(old_roi_source, str):
        roi_layer['source'][0] = {
            "url": old_roi_source,
            "subsources": {
                "default": True,
                "bounds": False,
                "properties": False,
                "mesh": True
            },
            "enableDefaultSubsources": False
        }

    roi_layer['source'][0]['subsources']['properties'] = False
    roi_layer['source'].append(f"precomputed://{roi_segprops_file[:-len('/info')]}")
    roi_layer['segmentQuery'] = ' '.join(f'|{prop}' for prop in roi_prop_names)

    seg_layer = layers['brain-seg']
    mirror_layer = layers['brain-mirror']

    task_layers = {
        (None, False): seg_layer,
        (None, True): mirror_layer,
        ('R', False): copy.deepcopy(seg_layer),    # right neurons
        ('L', True): copy.deepcopy(mirror_layer),  # left mirrored neurons
        ('L', False): copy.deepcopy(seg_layer),    # left neurons
        ('R', True): copy.deepcopy(mirror_layer),  # right mirrored neurons
    }
    for (side, is_mirror), layer in task_layers.items():
        # Everything is archived by default
        layer['archived'] = True
        layer['visible'] = False

        if isinstance(layer['source'], list):
            layer['source'].append(prop_source)
        else:
            layer['source'] = [layer['source'], prop_source]

        layer['segmentQuery'] = ' '.join(f'|{prop}' for prop in prop_names)
        if side:
            if is_mirror:
                layer['name'] = f'{side}-neurons-mirrored'
            else:
                layer['name'] = f'{side}-neurons'
        else:
            if is_mirror:
                layer['name'] = 'neurons-mirrored'
            else:
                layer['name'] = 'neurons'

        if layer['name'] != 'neurons':
            layer['linkedSegmentationColorGroup'] = 'neurons'

        side_segments = [str(b) for b in sorted(sides.get(side, []))]
        if (side, is_mirror) != (None, True):
            task_layers[(side, is_mirror)]['segments'] = side_segments

    for layer in reversed(task_layers.values()):
        state['layers'].insert(1, layer)

    task_layers[('R', True)]['linkedSegmentationGroup'] = task_layers[('R', False)]['name']
    task_layers[('L', True)]['linkedSegmentationGroup'] = task_layers[('L', False)]['name']

    # Four cases:
    # - left and right are present
    # - left only
    # - right only
    # - neither is present

    # If both L and are present, then use actual right and mirrored left
    if {'L', 'R'} == {*sides}:
        task_layers[('R', False)]['visible'] = True
        task_layers[('L', True)]['visible'] = True
        task_layers[('R', False)]['archived'] = False
        task_layers[('L', True)]['archived'] = False
        state['selectedLayer'] = {
            "layer": task_layers[('R', False)]['name'],
            "visible": True
        }
    # if only R is present, use actual right
    elif 'R' in sides:
        task_layers[('R', False)]['visible'] = True
        task_layers[('R', False)]['archived'] = False
        task_layers[('L', True)]['archived'] = False
        state['selectedLayer'] = {
            "layer": task_layers[('R', False)]['name'],
            "visible": True
        }
    # if only L is present, use actual left
    elif 'L' in sides:
        task_layers[('L', False)]['visible'] = True
        task_layers[('L', False)]['archived'] = False
        task_layers[('R', True)]['archived'] = False
        state['selectedLayer'] = {
            "layer": task_layers[('L', False)]['name'],
            "visible": True
        }
    # if no sides were determined, then just show everything in the 'brain-seg' layer.
    else:
        task_layers[(None, False)]['visible'] = True
        task_layers[(None, False)]['archived'] = False
        task_layers[(None, True)]['archived'] = False
        state['selectedLayer'] = {
            "layer": task_layers[(None, False)]['name'],
            "visible": True
        }

    return state


def determine_segment_ids(state, layer):
    """
    Determine the mapping of segment name to segment ID
    in a neuroglancer precomputed volume.
    """
    if isinstance(state, str):
        state = parse_nglink(state)

    src = layer_dict(state)[layer]['source']

    src_url = None
    while src_url is None:
        if isinstance(src, str):
            src_url = src
        if isinstance(src, dict):
            src_url = src['url']
        if isinstance(src, list):
            src = src[0]

    assert src_url.startswith('precomputed://gs://')
    bucket_src = src_url[len('precomputed://gs://'):]
    info_url = f'https://storage.googleapis.com/{bucket_src}/info'
    info = requests.get(info_url, timeout=4.0).json()
    props_dir = info['segment_properties']

    props_url = f'https://storage.googleapis.com/{bucket_src}/{props_dir}/info'
    props = requests.get(props_url, timeout=4.0).json()

    label_props = [p for p in props['inline']['properties'] if p['type'] == 'label']
    if len(label_props) == 0:
        raise RuntimeError(f"Segment properties for layer '{layer}' contain no 'label' property.")

    # returns {name: id}
    segment_ids = {
        label: int(i)
        for label, i in zip(label_props[0]['values'], props['inline']['ids'])
    }
    return segment_ids

