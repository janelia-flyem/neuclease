import os
import sys
import json
import copy
import signal
import logging
import argparse
from itertools import chain
from http import HTTPStatus
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, request, abort, redirect, url_for, jsonify, Response, make_response

from .logging_setup import init_logging, log_exceptions, PrefixedLogger
from .merge_graph import  LabelmapMergeGraph
from .cleave import cleave
from .util import Timer

# Globals
MERGE_GRAPH = None
LOGFILE = None # Will be set in __main__, below
logger = logging.getLogger(__name__)
app = Flask(__name__)


def main(debug_mode=False):
    global MERGE_GRAPH
    global LOGFILE

    # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5555, type=int)
    parser.add_argument('--log-dir', required=False)
    parser.add_argument('--merge-table', required=True)
    parser.add_argument('--mapping-file', required=False)
    parser.add_argument('--split-mapping', required=False)
    parser.add_argument('--primary-dvid-server')
    parser.add_argument('--primary-uuid', required=False,
                        help="If provided, do not update the internal cached merge table mapping except for the given UUID. "
                             "(Prioritizes speed of the primary UUID over all others.)")
    parser.add_argument('--primary-labelmap-instance')
    parser.add_argument('--suspend-before-launch', action='store_true',
                        help="After loading the merge graph, suspend the process before launching the server, and await a SIGCONT. "
                             "Allows you to ALMOST hot-swap a running cleave server. (You can load the new merge graph before killing the old server).")
    args = parser.parse_args()

    # This check is to ensure that this initialization is only run once,
    # even in the presence of the flask debug 'reloader'.
    if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        ##
        ## Configure logging
        ##
        print("Configuring logging...")
        if not args.log_dir:
            args.log_dir = os.path.dirname(args.merge_table)

        LOGFILE = init_logging(logger, args.log_dir, args.merge_table, debug_mode)
        logger.info("Server started with command: " + ' '.join(sys.argv))
    
        ##
        ## Load merge table
        ##
        if not os.path.exists(args.merge_table):
            sys.stderr.write(f"Merge table not found: {args.merge_table}\n")
            sys.exit(-1)

        print("Loading merge table...")
        with Timer(f"Loading merge table from: {args.merge_table}", logger):
            MERGE_GRAPH = LabelmapMergeGraph(args.merge_table, args.mapping_file, args.primary_uuid)

        # If no mappings file was given, fetch it from DVID.
        if not args.mapping_file and args.primary_dvid_server and args.primary_uuid and args.primary_labelmap_instance:
            MERGE_GRAPH.fetch_and_apply_mapping(args.primary_dvid_server, args.primary_uuid, args.primary_labelmap_instance)

        if args.split_mapping:
            if not args.primary_dvid_server or not args.primary_uuid or not args.primary_labelmap_instance:
                raise RuntimeError("Can't append split supervoxel edges without all primary server/uuid/instance info")
            with Timer(f"Appending split supervoxel edges for supervoxels in {args.split_mapping}", logger):
                bad_edges = MERGE_GRAPH.append_edges_for_split_supervoxels( args.split_mapping,
                                                                            args.primary_dvid_server,
                                                                            args.primary_uuid,
                                                                            args.primary_labelmap_instance )

                if len(bad_edges) > 0:
                    split_mapping_name = os.path.split(args.split_mapping)[1]
                    bad_edges_name = os.path.splitext(split_mapping_name)[0] + '.csv'
                    bad_edges_filepath = args.log_dir + '/' + bad_edges_name
                    bad_edges.to_csv(bad_edges_filepath, index=False, header=True)
                    logger.error(f"Some edges belonging to split supervoxels could not be preserved, due to {len(bad_edges)} bad representative points.")
                    logger.error(f"See {bad_edges_filepath}")

        if args.suspend_before_launch:
            pid = os.getpid()
            print(f"Suspending process.  Please use 'kill -CONT {pid}' to resume app startup.")
            sys.stdout.flush()
            os.kill(pid, signal.SIGSTOP)
            print(f"Process resumed.  Starting server.")

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
        json_loading_time = timer.seconds
        
        if not data:
            abort(Response('Request is missing a JSON body', status=400))
    
        body_id = data["body-id"]
        user = data.get("user", "unknown")
        body_logger = PrefixedLogger(logger, f"User {user}: Body {body_id}: ")
        
        if json_loading_time > 0.5:
            body_logger.warning(f"Loading JSON for the following request took {json_loading_time:.2f} seconds!")

        # This is injected into the request so that it will be echoed back to the client
        data['request-timestamp'] = str(datetime.now())
    
        req_string = json.dumps(data, sort_keys=True)
        body_logger.info(f"Received cleave request: {req_string}")
        cleave_results, status_code = _run_cleave(data)

        json_response = jsonify(cleave_results)
    
    body_logger.info(f"Total time: {timer.timedelta}")
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
    body_logger = PrefixedLogger(logger, f"User {user}: Body {body_id}: ")

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
        msg = "Request contained no seeds!"
        body_logger.error(msg)
        body_logger.info(f"Responding with error PRECONDITION_FAILED.")
        cleave_response.setdefault("errors", []).append(msg)
        return cleave_response, HTTPStatus.PRECONDITION_FAILED # code 412

    # Extract this body's edges from the complete merge graph
    with Timer() as timer:
        df, supervoxels = MERGE_GRAPH.extract_rows(server, uuid, segmentation_instance, body_id, body_logger)
        edges = df[['id_a', 'id_b']].values.astype(np.uint64)
        weights = df['score'].values
    body_logger.info(f"Extracting body graph took {timer.timedelta}")

    unexpected_seeds = set(chain(*seeds.values())) - set(supervoxels)
    if unexpected_seeds:
        msg = f"Request contained seeds that do not belong to body: {sorted(unexpected_seeds)}"
        body_logger.error(msg)
        body_logger.info("Responding with error PRECONDITION_FAILED.")
        cleave_response.setdefault("errors", []).append(msg)
        return cleave_response, HTTPStatus.PRECONDITION_FAILED # code 412

    # Perform the computation
    with Timer() as timer:
        results = cleave(edges, weights, seeds, supervoxels, method=method)
    body_logger.info(f"Computing cleave took {timer.timedelta}")

    # Convert assignments to JSON
    df = pd.DataFrame({'node': results.node_ids, 'label': results.output_labels})
    df.sort_values('node', inplace=True)
    for label, group in df.groupby('label'):
        cleave_response["assignments"][str(label)] = group['node'].tolist()

    if results.disconnected_components:
        msg = (f"Cleave result contains non-contiguous objects for seeds: "
               f"{sorted(results.disconnected_components)}")
        body_logger.warning(msg)
        cleave_response["warnings"].append(msg)

    if results.contains_unlabeled_components:
        msg = "Cleave result is not complete."
        body_logger.error(msg)
        body_logger.warning(msg)
        cleave_response["warnings"].append(msg)

    body_logger.info("Sending cleave results")
    return ( cleave_response, HTTPStatus.OK )

