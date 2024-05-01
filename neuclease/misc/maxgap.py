import re
import copy
import getpass
from functools import partial

import numpy as np
import pandas as pd

from neuclease.util import upload_to_bucket, dump_json, compute_parallel
from neuclease.util.graph import euclidean_mst
from neuclease.dvid.labelmap import fetch_sparsevol_coarse
from neuclease.misc.neuroglancer import download_ngstate, annotation_layer_json


# Make sure you set GOOGLE_APPLICATION_CREDENTIALS as needed for whichever bucket you're using.
DEFAULT_LINK_BUCKET = f'flyem-user-links/{getpass.getuser()}'

SHADER = """\
void main() {
    setLineColor(defaultColor());
    setEndpointMarkerSize(0.0, 0.0);
    setEndpointMarkerColor(defaultColor(), defaultColor());
    setEndpointMarkerBorderWidth(0.0, 0.0);
    //setEndpointMarkerBorderColor(defaultColor(), defaultColor());
}
"""


def maxgap_for_bodies(server, uuid, instance, bodies, processes=8):
    fn = partial(maxgap_for_body, server, uuid, instance)
    maxgaps = compute_parallel(fn, bodies, processes=processes)
    bodies = pd.Index(bodies, name='body')
    maxgaps = pd.Series(index=bodies, data=maxgaps, name='maxgap')
    return maxgaps


def maxgap_for_body(server, uuid, instance, body):
    """
    Fetch the coarse sparsevol for the given body and find the
    largest gap between blocks in the body's blockwise
    minimum spanning tree.
    Note:
        This returns the gap between block centers,
        so a fully connected body will have a 'max gap' of 1,
        and a body with diagonal connectivity will have a
        'max gap' of √3.
    """
    svc = fetch_sparsevol_coarse(server, uuid, instance, body)
    return euclidean_mst(svc, format='max-or-initial')


def mst_edges_for_body(server, uuid, instance, body):
    """
    Fetch the coarse sparsevol for the given body and
    return the edges in the minimum spanning tree across
    the blocks occupied by the body, along with the
    lengths of those edges.

    Note:
        This returns the distances between block centers,
        so a fully connected body will have a 'max gap' of 1,
        and a body with diagonal connectivity will have a
        'max gap' of √3.
    """
    svc = fetch_sparsevol_coarse(server, uuid, instance, body)
    df = euclidean_mst(svc, format='pandas')
    df[['za', 'ya', 'xa']] = svc[df['u'].values] * 64 + 32
    df[['zb', 'yb', 'xb']] = svc[df['v'].values] * 64 + 32
    df['distance'] = 64 * df['distance'].astype(np.float32)
    return df


def view_edges_for_body(edges, body, initial_ngstate, seg_layer_name='.*', link_bucket_location=DEFAULT_LINK_BUCKET):
    """
    Construct a neuroglancer view of a body with a local://annotations
    layer to display the given set of edges as line annotations.

    In the future, it would be nice to offer an option to construct
    a skeleton for tree data, rather than line annotations.
    (Can skeleton nodes/edges have properties associated with them?)
    """
    assert isinstance(initial_ngstate, (dict, str))
    if isinstance(initial_ngstate, dict):
        state = copy.deepcopy(initial_ngstate)
    else:
        state = download_ngstate(initial_ngstate)

    # Select the first segmentation layer that matches the given pattern
    seg_layer = [
        layer
        for layer in state['layers']
        if layer['type'] == 'segmentation' and re.match(seg_layer_name, layer['name'])
    ][0]

    state['title'] = f'Body {body}'
    seg_layer['segments'] = [str(body)]
    seg_layer['segmentQuery'] = str(body)
    line_layer = annotation_layer_json(
        edges.assign(type='line', segments=body),
        f'{body}-edges',
        linkedSegmentationLayer=seg_layer['name'],
        properties=['distance'],
        shader=SHADER
    )
    state['layers'].append(line_layer)

    if link_bucket_location.startswith('gs://'):
        link_bucket_location = link_bucket_location[len('gs://'):]
    bucket, *bucket_dir = link_bucket_location.split('/')
    bucket_dir = '/'.join(bucket_dir)
    bucket_path = f'{bucket_dir}/body-{body}.json'
    upload_to_bucket(bucket, bucket_path, dump_json(state), disable_cache=True)
    return f'https://clio-ng.janelia.org/#!gs://{bucket}/{bucket_path}'
