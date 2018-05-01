from __future__ import print_function
import sys
import os
import json
import copy
import signal
import multiprocessing
from itertools import chain
from http import HTTPStatus
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, request, abort, redirect, url_for, jsonify, Response, make_response

from .logging_setup import init_logging, log_exceptions, ProtectedLogger
from .merge_table import load_merge_table, extract_rows
from .dvid import get_supervoxels_for_body
from .cleave import cleave
from .util import Timer

# FIXME: multiprocessing has unintended consequences for the log rollover procedure.
USE_MULTIPROCESSING = False

# Globals
pool = None # Must be instantiated after this module definition, at the bottom of main().
LOGFILE = None # Will be set in __main__, below
app = Flask(__name__)
logger = ProtectedLogger(__name__)
MERGE_TABLE = None
PRIMARY_UUID = None

def main(debug_mode=False):
    global MERGE_TABLE
    global PRIMARY_UUID
    global pool
    global LOGFILE
    global logger
    global app
    global compute_cleave
    global show_log

    import argparse

    # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5555, type=int)
    parser.add_argument('--merge-table', required=True)
    parser.add_argument('--mapping-file', required=False)
    parser.add_argument('--log-dir', required=False)
    parser.add_argument('--primary-uuid', required=False,
                        help="If provided, do not update the internal cached merge table mapping except for the given UUID. "
                        "(Prioritizes speed of the primary UUID over all others.)")
    args = parser.parse_args()

    ##
    ## Configure logging
    ##
    LOGFILE = init_logging(logger, args.log_dir, args.merge_table, debug_mode)
    logger.info("Server started with command: {}".format(' '.join(sys.argv)))

    ##
    ## Load merge table
    ##
    if not os.path.exists(args.merge_table):
        sys.stderr.write("Merge table not found: {}\n".format(args.graph_db))
        sys.exit(-1)

    with Timer(f"Loading merge table from: {args.merge_table}", logger):
        MERGE_TABLE = load_merge_table(args.merge_table, args.mapping_file, set_multiindex=False, scores_only=True)

    PRIMARY_UUID = args.primary_uuid
        
    if USE_MULTIPROCESSING:
        # Pool must be started LAST, after we've configured all the global variables (logger, etc.),
        # so that the forked (child) processes have the same setup as the parent process.
        pool = multiprocessing.Pool(8)

    print("Starting app...")
    app.run(host='0.0.0.0', port=args.port, debug=debug_mode, threaded=not debug_mode, use_reloader=debug_mode)


@app.route('/')
def index():
    return redirect(url_for('show_log', page='0'))


@app.route('/log')
@log_exceptions(logger)
def show_log():
    page = request.args.get('page')
    if page and page != '0':
        path = LOGFILE + '.' + page
    else:
        path = LOGFILE

    path = os.path.abspath(path)

    if not os.path.exists(path):
        msg = "Error 404: Could not find log page " + page + ".\n"
        msg += "File does not exist:\n"
        msg += path
        response = make_response(msg)
        response.headers['Content-Type'] = 'text/plain'
        return response, 404

    with open(path, 'r') as f:
        contents = f.read()
    
    response = make_response(contents)
    response.headers['Content-Type'] = 'text/plain'
    return response
    return contents
    

@app.route('/compute-cleave', methods=['POST'])
@log_exceptions(logger)
def compute_cleave():
    """
    Example body json:
    
    {
        "body-id": 123,
        "seeds": {
            "1" : [1234, 1235, 1236],
            "2": [],
            "4": [234, 235, 236],
        },

        "user": "bergs",
        "server": "emdata2.int.janelia.org",
        "port": 8700,
        "uuid": "f73ce97d08064bcba34f2637c356e490",
        "segmentation-instance": "segmentation",
        "mesh-instance": "segmentation_meshes_tars"
    }
    """
    with Timer() as timer:
        data = request.json
        if not data:
            abort(Response('Request is missing a JSON body', status=400))
    
        body_id = data["body-id"]
        user = data.get("user", "unknown")
        
        # This is injected into the request so that it will be echoed back to the client
        data['request-timestamp'] = str(datetime.now())
    
        req_string = json.dumps(data, sort_keys=True)
        logger.info(f"User {user}: Body {body_id}: Received cleave request: {req_string}")
        if USE_MULTIPROCESSING:
            cleave_results, status_code = pool.apply(_run_cleave, [data])
        else:
            cleave_results, status_code = _run_cleave(data)

        json_response = jsonify(cleave_results)
    
    logger.info(f"User {user}: Body {body_id}: Total time: {timer.timedelta}")
    return json_response, status_code


@log_exceptions(logger)
def _run_cleave(data):
    """
    Helper function that actually performs the cleave,
    and can be run in a separate process.
    Must not use any flask functions.
    """
    global logger
    global MERGE_TABLE

    user = data.get("user", "unknown")
    method = data.get("method", "seeded-watershed")
    body_id = data["body-id"]
    seeds = { int(k): v for k,v in data["seeds"].items() }
    server = data["server"] + ':' + str(data["port"])
    uuid = data["uuid"]
    segmentation_instance = data["segmentation-instance"]

    if not server.startswith('http://'):
        server = 'http://' + server

    # Remove empty seed classes (if any)
    for label in list(seeds.keys()):
        if len(seeds[label]) == 0:
            del seeds[label]

    cleave_response = copy.copy(data)
    cleave_response["seeds"] = dict(sorted((k, sorted(v)) for (k,v) in data["seeds"].items()))
    cleave_response["assignments"] = {}
    cleave_response["warnings"] = []

    if not data["seeds"]:
        msg = f"User {user}: Body {body_id}: Request contained no seeds!"
        logger.error(msg)
        logger.info(f"User {user}: Body {body_id}: Responding with error PRECONDITION_FAILED.")
        cleave_response.setdefault("errors", []).append(msg)
        return cleave_response, HTTPStatus.PRECONDITION_FAILED # code 412

    with Timer(f"User {user}: Body {body_id}: Retrieving supervoxel list from DVID"):
        supervoxels = get_supervoxels_for_body(server, uuid, segmentation_instance, body_id)
        supervoxels = np.asarray(supervoxels, np.uint64)
        supervoxels.sort()
    
    unexpected_seeds = set(chain(*seeds.values())) - set(supervoxels)
    if unexpected_seeds:
        msg = f"User {user}: Body {body_id}: Request contained seeds that do not belong to body: {sorted(unexpected_seeds)}"
        logger.error(msg)
        logger.info(f"User {user}: Body {body_id}: Responding with error PRECONDITION_FAILED.")
        cleave_response.setdefault("errors", []).append(msg)
        return cleave_response, HTTPStatus.PRECONDITION_FAILED # code 412

    # Extract this body's edges from the complete merge graph
    with Timer(f"User {user}: Body {body_id}: Extracting body graph", logger):
        permit_table_update = (PRIMARY_UUID is None or uuid == PRIMARY_UUID)
        df = extract_rows(MERGE_TABLE, body_id, supervoxels, update_inplace=permit_table_update)
        
        edges = df[['id_a', 'id_b']].values.astype(np.uint64)
        weights = df['score'].values
    
    # Perform the computation
    with Timer(f"User {user}: Body {body_id}: Computing cleave", logger):
        results = cleave(edges, weights, seeds, supervoxels, method=method)

    # Convert assignments to JSON
    with Timer(f"User {user}: Body {body_id}: Populating response", logger):
        df = pd.DataFrame({'node': results.node_ids, 'label': results.output_labels})
        df.sort_values('node', inplace=True)
        for label, group in df.groupby('label'):
            cleave_response["assignments"][str(label)] = group['node'].tolist()

    if results.disconnected_components:
        msg = (f"User {user}: Body {body_id}: Cleave result contains non-contiguous objects for seeds: "
               f"{sorted(results.disconnected_components)}")
        logger.warning(msg)
        cleave_response["warnings"].append(msg)

    if results.contains_unlabeled_components:
        msg = f"User {user}: Body {body_id}: Cleave result is not complete."
        logger.error(msg)
        cleave_response.setdefault("errors", []).append(msg)
        logger.info("User {user}: Body {body_id}: Responding with error PRECONDITION_FAILED.")
        return ( cleave_response, HTTPStatus.PRECONDITION_FAILED )

    logger.info(f"User {user}: Body {body_id}: Sending cleave results")
    return ( cleave_response, HTTPStatus.OK )

