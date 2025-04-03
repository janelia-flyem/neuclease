import re
import copy
import getpass
import logging
from textwrap import dedent
from io import BytesIO

import numpy as np
import pandas as pd
import requests

from ..util import Timer
from ..dvid import fetch_instance_info, fetch_supervoxels, generate_sample_coordinates
from ..misc.neuroglancer import parse_nglink, layer_state, annotation_layer_json, upload_ngstate


logger = logging.getLogger(__name__)


def compute_cleave(cleave_server, dvid_server, uuid, instance, body, seeds, timeout=60.0):
    if not cleave_server.startswith('http'):
        cleave_server = f"http://{cleave_server}"
    dvid_server, dvid_port = dvid_server.split(':')

    if isinstance(seeds, dict):
        seeds = {str(k): v for k, v in seeds.items()}
    elif isinstance(seeds, list):
        assert all(isinstance(s, list) for s in seeds)
        seeds = {str(i): v for i, v in enumerate(seeds, start=1)}

    data = { "user": getpass.getuser(),
             "body-id": body,
             "port": dvid_port,
             "seeds": seeds,
             "server": dvid_server,
             "uuid": uuid,
             "segmentation-instance": instance,
             "mesh-instance": "segmentation_meshes_tars" }

    r = requests.post(f'{cleave_server}/compute-cleave', json=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


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
        float normalized_score = 1.0 - (prop_score() - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
        color = viridis_quintic(normalized_score);
        setLineColor(color);
        setEndpointMarkerSize(8.0, 8.0);
        setEndpointMarkerColor(color, color);
        setEndpointMarkerBorderWidth(1.0, 1.0);
        setEndpointMarkerBorderColor(defaultColor(), defaultColor());
    }
""")


# https://github.com/janelia-flyem/NeuTu/blob/develop/neurolabi/gui/protocols/taskbodycleave.cpp#L84C39-L106
SEED_COLORS = [
    "#{0:02x}{1:02x}{2:02x}".format(*c)
    for c in
    [
        (0,    0,    0),  # noqa
        (230,  25,  75),  # noqa
        (255, 225,  25),  # noqa
        (  0, 130, 200),  # noqa
        (245, 130,  48),  # noqa
        (145,  30, 180),  # noqa
        ( 70, 240, 240),  # noqa
        ( 60, 180,  75),  # noqa
        (240,  50, 230),  # noqa
        (210, 245,  60),  # noqa
        (250, 190, 190),  # noqa
        (  0, 128, 128),  # noqa
        (230, 190, 255),  # noqa
        (170, 110,  40),  # noqa
        (255, 250, 200),  # noqa
        (128,   0,   0),  # noqa
        (170, 195, 255),  # noqa
        (128, 128,   0),  # noqa
        (255, 215, 180),  # noqa
        (  0,   0, 128),  # noqa
        (128, 128, 128),  # noqa
    ]
]


def visualize_edges_table(cleave_server, dvid_server, uuid, instance, body, ngstate, agglo_layer_name='', sv_layer_name='', bucket_path=None, timeout=60.0, seeds=None, show_sv_nodes=False):
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

    with Timer(f"Fetching cleave edges for body {body}", logger):
        edges = fetch_body_edge_table(cleave_server, *dvid_seg, body, timeout)

        # Neuroglancer JSON can't handle infinity,
        # so convert infinities to something very big (but finite).
        inf_scores = np.isinf(edges['score'])

        SCORE_MIN = edges.loc[~inf_scores, 'score'].min()
        SCORE_MAX = edges.loc[~inf_scores, 'score'].max()

        edges.loc[inf_scores, 'score'] = 1e6-1

    if sv_layer_name:
        supervoxels = fetch_supervoxels(*dvid_seg, body).tolist()
        sv_layer = layer_state(ngstate, sv_layer_name)
        sv_layer['segments'] = [*map(str, supervoxels)]
        sv_layer['segmentQuery'] = ', '.join(sv_layer['segments'])
        sv_layer['archived'] = False
        sv_layer['visible'] = False

        if seeds:
            seed_layer = copy.deepcopy(sv_layer)
            seed_layer['name'] = 'seeds'
            seed_layer['segments'] = []
            seed_layer['segmentColors'] = {}
            for seed, svs in seeds.items():
                seed_layer['segments'].extend(map(str, svs))
                seed_layer['segmentColors'] |= {str(sv): SEED_COLORS[int(seed)] for sv in svs}
            seed_layer['segmentQuery'] = ' '.join(seed_layer['segments'])
            ngstate['layers'].append(seed_layer)

            with Timer("Computing cleave assignments", logger):
                cleave_result = compute_cleave(cleave_server, *dvid_seg, body, seeds, timeout)

            assignments_layer = copy.deepcopy(sv_layer)
            assignments_layer['name'] = 'cleave-assignments'
            assignments_layer['segments'] = []
            assignments_layer['segmentColors'] = {}
            for seed, svs in cleave_result['assignments'].items():
                assignments_layer['segments'].extend(map(str, svs))
                assignments_layer['segmentColors'] |= {str(sv): SEED_COLORS[int(seed)] for sv in svs}
            assignments_layer['segmentQuery'] = ' '.join(assignments_layer['segments'])
            ngstate['layers'].append(assignments_layer)

        edges['segments'] = edges[['id_a', 'id_b']].apply(list, axis=1)

    edge_layer = annotation_layer_json(
        edges.assign(type='line').astype({'source': 'category'}),
        f'cleave-edges-{body}',
        color='#ffffff',
        linkedSegmentationLayer=sv_layer_name,
        properties=['source', 'score'],
        res_nm_xyz=res_nm_xyz,
        shader=(
            SHADER
            .replace('__SCORE_MIN__', str(SCORE_MIN))
            .replace('__SCORE_MAX__', str(SCORE_MAX))
        )
    )
    ngstate['layers'].append(edge_layer)

    if show_sv_nodes:
        all_svs = pd.unique(edges[['id_a', 'id_b']].values.reshape(-1))
        logger.info("Selecting points for supervoxel nodes")
        sv_coords = generate_sample_coordinates(*dvid_seg, all_svs, supervoxels=True, interior=True, processes=8)
        edges = edges.merge(sv_coords.rename_axis('id_a'), 'left', on='id_a')
        edges = edges.merge(sv_coords.rename_axis('id_b'), 'left', on='id_b', suffixes=('_id_a', '_id_b'))

        sv_node_edges = pd.concat((
            edges.assign(type='line').drop(columns=['xb', 'yb', 'zb']).rename(columns={'z_id_a': 'zb', 'y_id_a': 'yb', 'x_id_a': 'xb'}).assign(segments=edges['id_a']),
            edges.assign(type='line').drop(columns=['xa', 'ya', 'za']).rename(columns={'z_id_b': 'za', 'y_id_b': 'ya', 'x_id_b': 'xa'}).assign(segments=edges['id_b'])
        ))

        sv_node_edge_layer = annotation_layer_json(
            sv_node_edges[['xa', 'ya', 'za', 'xb', 'yb', 'zb', 'segments']].assign(type='line'),
            f'cleave-sv-nodes-{body}',
            color='#ffffff',
            linkedSegmentationLayer=sv_layer_name,
            res_nm_xyz=res_nm_xyz,
        )
        ngstate['layers'].append(sv_node_edge_layer)

    if bucket_path:
        url = upload_ngstate(bucket_path, ngstate, True, return_prefix=neuroglancer_domain)
        url = url.replace('https://storage.googleapis.com/', 'gs://')
        return url

    return ngstate
