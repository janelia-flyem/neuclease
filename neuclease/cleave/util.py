import re
import copy
import getpass
from textwrap import dedent
from io import BytesIO

import numpy as np
import pandas as pd
import requests

from ..util import Timer
from ..dvid import fetch_instance_info, fetch_supervoxels
from ..misc.neuroglancer import parse_nglink, layer_state, annotation_layer_json, upload_ngstate


def fetch_body_edge_table(cleave_server, dvid_server, uuid, instance, body, timeout=60.0):
    """
    Note:
        If you give a body that doesn't exist, the server returns a 404 error.
    """
    if not dvid_server.startswith('http'):
        dvid_server = 'http://' + dvid_server

    dvid_server, dvid_port = dvid_server.split('://')[1].split(':')

    if not cleave_server.startswith('http'):
        cleave_server = 'http://' + cleave_server

    data = { "body-id": body,
             "port": dvid_port,
             "server": dvid_server,
             "uuid": uuid,
             "segmentation-instance": instance,
             "user": getpass.getuser() }

    r = requests.post(f'{cleave_server}/body-edge-table', json=data, timeout=timeout)
    r.raise_for_status()

    df = pd.read_csv(BytesIO(r.content), header=0)
    df = df.astype({'id_a': np.uint64, 'id_b': np.uint64, 'score': np.float32})
    return df


SHADER = dedent("""\
    float saturate( float x ) { return clamp( x, 0.0, 1.0 ); }

    float SCORE_MIN = __SCORE_MIN__;
    float SCORE_MAX = __SCORE_MAX__;

    vec3 viridis_quintic( float x )
    {
        x = saturate( x );
        vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
        vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
        return vec3(
            dot( x1.xyzw, vec4( +0.280268003, -0.143510503, +2.225793877, -14.815088879 ) ) + dot( x2.xy, vec2( +25.212752309, -11.772589584 ) ),
            dot( x1.xyzw, vec4( -0.002117546, +1.617109353, -1.909305070, +2.701152864 ) ) + dot( x2.xy, vec2( -1.685288385, +0.178738871 ) ),
            dot( x1.xyzw, vec4( +0.300805501, +2.614650302, -12.019139090, +28.933559110 ) ) + dot( x2.xy, vec2( -33.491294770, +13.762053843 ) ) );
    }

    void main() {
    vec3 color = defaultColor();
    if (int(prop_source()) == 0) {
        color = vec3(1.0, 0.0, 0.0);
    }
    else {
        float normalized_score = 1.0 - (prop_score() - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
        color = viridis_quintic(normalized_score);
    }
    setLineColor(color);
    setEndpointMarkerSize(8.0, 8.0);
    setEndpointMarkerColor(color, color);
    setEndpointMarkerBorderWidth(1.0, 1.0);
    setEndpointMarkerBorderColor(defaultColor(), defaultColor());
    }

""")


def visualize_edges_table(cleave_server, dvid_server, uuid, instance, body, ngstate, agglo_layer_name='', sv_layer_name='', bucket_path=None, timeout=60.0):
    """
    Download the cleaving edge table for a particular body and construct a point annotation layer for those edges.
    Add that layer to a neuroglancer scene, starting with a user-supplied template scene.
    Upload the neuroglancer state file to a bucket and return the complete link for the uploaded scene.
    """
    dvid_seg = (dvid_server, uuid, instance)
    res_nm_xyz = fetch_instance_info(*dvid_seg)['Extended']['VoxelSize']

    neuroglancer_domain = 'https://neuroglancer-demo.appspot.com'

    if isinstance(ngstate, dict):
        ngstate = copy.deepcopy(ngstate)
    else:
        if ngstate.startswith('http'):
            neuroglancer_domain = re.match(r"(https?://.+?)/", ngstate).groups()[0]
        ngstate = parse_nglink(ngstate)

    if agglo_layer_name:
        agglo_layer = layer_state(ngstate, agglo_layer_name)
        agglo_layer['segments'] = [str(body)]
        agglo_layer['segmentQuery'] = str(body)
        agglo_layer['archived'] = False
        agglo_layer['visible'] = True

    if sv_layer_name:
        supervoxels = fetch_supervoxels(*dvid_seg, body).tolist()
        sv_layer = layer_state(ngstate, sv_layer_name)
        sv_layer['segments'] = [*map(str, supervoxels)]
        sv_layer['segmentQuery'] = ', '.join(sv_layer['segments'])
        sv_layer['archived'] = False
        sv_layer['visible'] = False

    with Timer(f"Fetching cleave edges for body {body}"):
        edges = fetch_body_edge_table(cleave_server, *dvid_seg, body, timeout)

        # Neuroglancer JSON can't handle infinity,
        # so convert infinities to something very big (but finite).
        inf_scores = np.isinf(edges['score'])

        SCORE_MIN = edges.loc[~inf_scores, 'score'].min()
        SCORE_MAX = edges.loc[~inf_scores, 'score'].max()

        edges.loc[inf_scores, 'score'] = 1e6-1

    edge_layer = annotation_layer_json(
        edges.assign(type='line').astype({'source': 'category'}),
        f'cleave-edges-{body}',
        color='#ffffff',
        properties=['source', 'score'],
        res_nm_xyz=res_nm_xyz,
        shader=(
            SHADER
            .replace('__SCORE_MIN__', str(SCORE_MIN))
            .replace('__SCORE_MAX__', str(SCORE_MAX))
        )
    )
    ngstate['layers'].append(edge_layer)

    if bucket_path:
        url = upload_ngstate(bucket_path, ngstate, True, return_prefix=neuroglancer_domain)
        url = url.replace('https://storage.googleapis.com/', 'gs://')
        return url

    return ngstate
