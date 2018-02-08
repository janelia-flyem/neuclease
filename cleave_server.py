from __future__ import print_function
import sys
import os
import logging
import signal
import sqlite3
from flask import Flask, request, abort, redirect, url_for, jsonify, Response

from agglomeration_split_tool import AgglomerationGraph, do_split

root_logger = logging.getLogger()
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('show_debug_info'))

@app.route('/debug')
def show_debug_info():
    return "TODO: Show cleaving log data here, for debugging purposes."

@app.route('/compute-cleave', methods=['POST'])
def compute_cleave():
    """
    Example body json:
    
    {
        "body-id": 123,
        "seeds": {
            "1" : [1234, 1235, 1236],
            "2": [],
            "4": [234, 235, 236],
        }
    }
    """
    data = request.json
    if not data:
        abort(Response('Request is missing a JSON body', status=400))

    logger.info("Received cleave request: {}".format(data))

    body_id = data["body-id"]
    seeds = { int(k): v for k,v in data["seeds"].items() }

    # Remove empty seed classes (if any)
    for label in list(seeds.keys()):
        if len(seeds[label]) == 0:
            del seeds[label]

    # Structure seed data for do_split()
    # (These dicts would have more members when using Neuroglancer viewers,
    #  but for do_split() we only need to provide the supervoxel_id.)
    agglo_tool_split_seeds = {}
    for label, supervoxel_ids in data["seeds"].items():
        for sv in supervoxel_ids:
            agglo_tool_split_seeds.setdefault(int(label), []).append({ "supervoxel_id": sv })
    
    split_result = do_split(GRAPH, agglo_tool_split_seeds, body_id)

    agglo_id = split_result['agglo_id']
    cur_eqs = split_result['cur_eqs']
    _supervoxel_map = split_result['supervoxel_map']

    # Notes:    
    # - supervoxel_map is a dict of { sv_id: set([seed_class, seed_class,...]) }
    #   (Most supervoxels contain only one seed class, though.)
    #   It does NOT contain all supervoxels in the body, so we have to use cur_eq
    # - cur_eq is a neuroglancer.equivalence_map.EquivalenceMap
    
    assert agglo_id == body_id

    cleave_results = { "body-id": body_id,
                       "assignments": {} }

    for label in seeds.keys():
        first_member = seeds[label][0]
        label_equivalences = cur_eqs.members(first_member)
        cleave_results["assignments"][str(label)] = list(label_equivalences)

    logger.info("Sending cleave results: {}".format(cleave_results))
    return jsonify(cleave_results)

if __name__ == '__main__':
    import argparse

#     ## DEBUG
#     # Careful:
#     # The flask debug server's "reloader" feature may cause this section to be executed more than once!
#     if len(sys.argv) == 1:
#         sys.argv += ["--graph-db", "exported_merge_graphs/274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0.sqlite"]

    print(sys.argv)

    # Don't log ordinary GET, POST, etc.
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5555, type=int)
    parser.add_argument('--graph-db', required=True)
    args = parser.parse_args()

    graph_name = os.path.split(args.graph_db)[1].split(':')[-1]
    GRAPH = AgglomerationGraph(sqlite3.connect(args.graph_db, check_same_thread=False))
    
    # Clear any handlers that were automatically added (by flask? by neuroglancer?)
    root_logger.handlers = []
    logger.handlers = []

    # Configure logging
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    print("Starting server on 0.0.0.0:{}".format(args.port))
    app.run(host='0.0.0.0', port=args.port, debug=True)
