import os
import sys
import copy
import signal
import logging
import argparse
from io import StringIO
from itertools import chain
from http import HTTPStatus
from datetime import datetime

import ujson
import pandas as pd

import requests
from flask import Flask, request, abort, redirect, url_for, jsonify, Response, make_response

from .logging_setup import init_logging
from .merge_graph import LabelmapMergeGraphLocalTable, LabelmapMergeGraphBigQuery
from .cleave import cleave, InvalidCleaveMethodError
from ..dvid import DvidInstanceInfo, default_dvid_session
from ..util import Timer, PrefixedLogger, log_exceptions


# Globals
MERGE_GRAPH = None
DEFAULT_METHOD = "seeded-mst"
LOGFILE = None # Will be set in __main__, below
logger = logging.getLogger(__name__)
app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge-table', required=False)
    parser.add_argument('--bigquery-table', required=False)

    parser.add_argument('-p', '--port', default=5555, type=int)
    parser.add_argument('--primary-dvid-server', required=True)
    parser.add_argument('--primary-uuid', required=False,
                        help="In case of a local merge table, do not update the internal cached "
                             "merge table mapping except for the given UUID. "
                             "(Prioritizes speed of the primary UUID over all others.)"
                             "Also, the merge graph is updated with split supervoxels for the given UUID.")
    parser.add_argument('--primary-labelmap-instance', required=True)
    parser.add_argument('--max-cached-bodies', type=int, default=100_000)
    parser.add_argument('--disable-extra-edge-cache', action='store_true', help='For debugging, it can be useful to disable the "extra edge" cache')
    parser.add_argument('--log-dir', required=False)
    parser.add_argument('--debug-export-dir', required=False, help="For debugging only. Enables export of certain intermediate results.")
    parser.add_argument('--suspend-before-launch', action='store_true',
                        help="After loading the merge graph, suspend the process before launching the server, and await a SIGCONT. "
                             "Allows you to ALMOST hot-swap a running cleave server. (You can load the new merge graph before killing the old server).")
    parser.add_argument('--testing', action='store_true')

    args ,_ = parser.parse_known_args()
    if bool(args.merge_table) == bool(args.bigquery_table):
        raise RuntimeError("Please provide either --merge-table or --bigquery-table (not both)")

    if args.merge_table:
        return _parse_args_local_table(parser)
    return _parse_args_bigquery_table(parser)


def _parse_args_local_table(parser):
    parser.add_argument('--mapping-file', required=False)
    parser.add_argument('--initialization-dvid-server',
                        help="Which DVID server to use for initializing the edge table (for splits and focused edges)")
    parser.add_argument('--initialization-uuid')
    parser.add_argument('--initialization-labelmap-instance')
    parser.add_argument('--primary-kafka-log', required=False,
                        help="Normally the startup procedure involves reading the entire kafka log for the primary dvid instance. "
                        "But if you supply one here in 'jsonl' format, it will be used instead of downloading the log from kafka.")

    parser.add_argument('--skip-focused-merge-update', action='store_true')
    parser.add_argument('--skip-split-sv-update', action='store_true')
    args = parser.parse_args()

    # By default, initialization is same as primary unless otherwise specified
    args.initialization_dvid_server = args.initialization_dvid_server or args.primary_dvid_server
    args.initialization_uuid = args.initialization_uuid or args.primary_uuid
    args.initialization_labelmap_instance = args.initialization_labelmap_instance or args.primary_labelmap_instance

    return args


def _parse_args_bigquery_table(parser):
    # I may want to implement these features for the BigQuery case,
    # but I'm not going to bother right now.
    # parser.add_argument('--skip-focused-merge-update', action='store_true')
    # parser.add_argument('--skip-split-sv-update', action='store_true')
    return parser.parse_args()


def main(debug_mode=False, stdout_logging=False):
    global MERGE_GRAPH
    global LOGFILE

    # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

    args = parse_args()

    # This check is to ensure that this initialization is only run once,
    # even in the presence of the flask debug 'reloader'.
    if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("Configuring logging...")
        if not args.log_dir:
            assert args.merge_table, \
                "If you don't supply a merge-table, please provide an explicit --log-dir"
            args.log_dir = os.path.dirname(args.merge_table)

        LOGFILE = init_logging(logger, args.log_dir, args.merge_table or args.bigquery_table, stdout_logging)
        logger.info("Server started with command: " + ' '.join(sys.argv))  # noqa

        MERGE_GRAPH = _init_merge_graph(args)

        if args.suspend_before_launch:
            pid = os.getpid()
            print(f"Suspending process.  Please use 'kill -CONT {pid}' to resume app startup.")
            sys.stdout.flush()
            os.kill(pid, signal.SIGSTOP)
            print("Process resumed.")

    logger.info("Merge graph loaded. Starting server.")
    print("Merge graph loaded. Starting server.")
    app.run(host='0.0.0.0', port=args.port, debug=debug_mode, threaded=not debug_mode, use_reloader=debug_mode)


def _init_merge_graph(args):
    if args.merge_table:
        return _init_local_merge_graph(args)
    else:
        return _init_bigquery_merge_graph(args)


def _init_local_merge_graph(args):
    if args.merge_table and not os.path.exists(args.merge_table):
        sys.stderr.write(f"Merge table not found: {args.merge_table}\n")
        sys.exit(-1)

    primary_instance_info = DvidInstanceInfo(args.primary_dvid_server, args.primary_uuid, args.primary_labelmap_instance)
    initialization_instance_info = DvidInstanceInfo(args.initialization_dvid_server, args.initialization_uuid, args.initialization_labelmap_instance)

    kafka_msgs = None
    if args.primary_kafka_log:
        assert args.primary_kafka_log.endswith('.jsonl'), \
            "Supply the kafka log in .jsonl format"
        kafka_msgs = []
        for line in open(args.primary_kafka_log, 'r'):
            kafka_msgs.append(ujson.loads(line))

    print("Loading merge table...")
    with Timer(f"Loading merge table from: {args.merge_table or 'NONE'}", logger):
        merge_graph = LabelmapMergeGraphLocalTable(
            args.merge_table,
            primary_instance_info.uuid,
            args.max_cached_bodies,
            args.disable_extra_edge_cache,
            args.debug_export_dir,
            no_kafka=args.testing
        )

    if not args.skip_focused_merge_update:
        with Timer("Loading focused merge decisions", logger):
            num_focused_merges = merge_graph.append_edges_for_focused_merges(*initialization_instance_info[:2], 'segmentation_merged')
        logger.info(f"Loaded {num_focused_merges} merge decisions.")

    # Apply splits first
    if all(primary_instance_info) and not args.skip_split_sv_update:
        with Timer("Appending split supervoxel edges for supervoxels in", logger):
            bad_edges = merge_graph.append_edges_for_split_supervoxels( initialization_instance_info, read_from='dvid', kafka_msgs=kafka_msgs )

            if len(bad_edges) > 0:
                bad_edges_name = f'BAD-SPLIT-EDGES-{args.primary_uuid[:4]}.csv'
                bad_edges_filepath = args.log_dir + '/' + bad_edges_name
                bad_edges.to_csv(bad_edges_filepath, index=False, header=True)
                logger.error(f"Some edges belonging to split supervoxels could not be preserved, due to {len(bad_edges)} bad representative points.")
                logger.error(f"See {bad_edges_filepath}")

    # Apply mapping (after splits), either from file or from DVID.
    if args.mapping_file:
        merge_graph.apply_mapping(args.mapping_file)
    elif all(primary_instance_info):
        merge_graph.fetch_and_apply_mapping(*primary_instance_info, kafka_msgs)

    return merge_graph


def _init_bigquery_merge_graph(args):
    logger.info(f"Using BigQuery table: {args.bigquery_table}")
    merge_graph = LabelmapMergeGraphBigQuery(
        args.bigquery_table,
        args.primary_uuid,
        args.max_cached_bodies,
        args.disable_extra_edge_cache,
        args.debug_export_dir
    )
    return merge_graph


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

        "method": "seeded-mst",
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
    
        req_string = ujson.dumps(data, sort_keys=True)
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
    method = data.get("method", DEFAULT_METHOD)
    body_id = data["body-id"]
    seeds = { int(k): v for k,v in data["seeds"].items() }
    server = data["server"] + ':' + str(data["port"])
    uuid = data["uuid"]
    segmentation_instance = data["segmentation-instance"]
    find_missing_edges = data.get("find-missing-edges", True)

    body_logger = PrefixedLogger(logger, f"User {user}: Body {body_id}: ")

    instance_info = DvidInstanceInfo(server, uuid, segmentation_instance)

    # Remove empty seed classes (if any)
    for label in list(seeds.keys()):
        if len(seeds[label]) == 0:
            del seeds[label]

    cleave_response = copy.copy(data)
    cleave_response["seeds"] = dict(sorted((k, sorted(v)) for (k,v) in data["seeds"].items()))
    cleave_response["assignments"] = {}
    cleave_response["warnings"] = []
    cleave_response["info"] = []

    if not data["seeds"]:
        msg = "Request contained no seeds!"
        body_logger.error(msg)
        body_logger.info(f"Responding with error PRECONDITION_FAILED.")
        cleave_response.setdefault("errors", []).append(msg)
        return cleave_response, HTTPStatus.PRECONDITION_FAILED # code 412

    # Extract this body's edges from the complete merge graph
    with Timer() as timer:
        try:
            session = default_dvid_session(appname='cleave-server', user=user)
            mutid, supervoxels, edges, scores = MERGE_GRAPH.extract_edges(*instance_info, body_id, find_missing_edges, session=session, logger=body_logger)
        except requests.HTTPError as ex:
            status_name = str(HTTPStatus(ex.response.status_code)).split('.')[1]
            if ex.response.status_code == HTTPStatus.NOT_FOUND:
                msg = f"Body not found: {body_id}"
            else:
                msg = f"Received error from DVID: {status_name}"
            body_logger.error(msg)
            body_logger.info(f"Responding with error {status_name}.")
            cleave_response.setdefault("errors", []).append(msg)
            return cleave_response, ex.response.status_code

    body_logger.info(f"Extracting body graph (mutid={mutid}) took {timer.timedelta}")

    unexpected_seeds = set(chain(*seeds.values())) - set(supervoxels)
    if unexpected_seeds:
        msg = f"Request contained seeds that do not belong to body: {sorted(unexpected_seeds)}"
        body_logger.error(msg)
        body_logger.info("Responding with error PRECONDITION_FAILED.")
        cleave_response.setdefault("errors", []).append(msg)
        return cleave_response, HTTPStatus.PRECONDITION_FAILED # code 412

    try:
        # Perform the cleave computation
        with Timer() as timer:
            results = cleave(edges[['id_a', 'id_b']].values, scores, seeds, supervoxels, method=method)
    except InvalidCleaveMethodError as ex:
        body_logger.error(str(ex))
        body_logger.info("Responding with error BAD_REQUEST.")
        cleave_response.setdefault("errors", []).append(str(ex))
        return cleave_response, HTTPStatus.BAD_REQUEST # code 400
        
    body_logger.info(f"Computing cleave took {timer.timedelta}")

    # Convert assignments to JSON
    df = pd.DataFrame({'node': supervoxels, 'label': results.output_labels})
    df.sort_values('node', inplace=True)
    for label, group in df.groupby('label'):
        cleave_response["assignments"][str(label)] = group['node'].tolist()

    if results.disconnected_components:
        msg = (f"Cleave result contains non-contiguous objects for seeds: "
               f"{sorted(results.disconnected_components)}")
        body_logger.warning(msg)
        cleave_response["info"].append(msg)

    if results.contains_unlabeled_components:
        num_unlabeled = len(cleave_response["assignments"]["0"])
        msg = f"Cleave result is not complete. {num_unlabeled} supervoxels remain unassigned."
        body_logger.error(msg)
        body_logger.warning(msg)
        cleave_response["warnings"].append(msg)

    body_logger.info("Sending cleave results")
    return ( cleave_response, HTTPStatus.OK )


@app.route('/primary-uuid')
def get_primary_uuid():
    global MERGE_GRAPH
    response = jsonify( { "uuid": MERGE_GRAPH.primary_uuid } )
    return response, HTTPStatus.OK
    

@app.route('/primary-uuid', methods=['POST'])
def set_primary_uuid():
    global MERGE_GRAPH
    data = request.json
    MERGE_GRAPH.set_primary_uuid(data["uuid"])
    response = jsonify( { "uuid": MERGE_GRAPH.primary_uuid } )
    return response, HTTPStatus.OK


@app.route('/body-edge-table', methods=['POST'])
def body_edge_table():
    """
    Extract rows for a particular body from the merge table.
    Useful for debugging, or for warming up the merge graph cache.
    """
    global logger
    global MERGE_TABLE

    data = request.json
    user = data.get("user", "unknown")
    body_id = data["body-id"]
    server = data["server"] + ':' + str(data["port"])
    uuid = data["uuid"]
    segmentation_instance = data["segmentation-instance"]
    find_missing_edges = data.get("find-missing-edges", True)

    body_logger = PrefixedLogger(logger, f"User {user}: Body {body_id}: ")

    instance_info = DvidInstanceInfo(server, uuid, segmentation_instance)

    body_logger.info("Received body-edge-table request")

    try:
        session = default_dvid_session(appname='cleave-server', user=user)
        _mutid, _supervoxels, edges, scores = MERGE_GRAPH.extract_edges(*instance_info, body_id, find_missing_edges, session=session, logger=body_logger)
    except requests.HTTPError as ex:
        status_name = str(HTTPStatus(ex.response.status_code)).split('.')[1]
        if ex.response.status_code == HTTPStatus.NOT_FOUND:
            msg = f"Body not found: {body_id}"
        else:
            msg = f"Received error from DVID: {status_name}"
        body_logger.error(msg)
        return msg, ex.response.status_code

    response = StringIO()
    edges['score'] = scores
    edges.to_csv(response, index=False, header=True)
    return response.getvalue()


@app.route('/debug', methods=['POST'])
def debug():
    """
    Endpoint for posting arbitrary messages to be written into the cleave server log.
    """
    global logger
    global MERGE_TABLE
    
    # Must be json-parseable
    debug_data = copy.copy(request.json)
    user = debug_data.setdefault("user", "unknown")
    if "seeds" in debug_data:
        debug_data["seeds"] = dict(sorted((k, sorted(v)) for (k,v) in debug_data["seeds"].items()))
        for k in debug_data["seeds"]:
            num_seeds = len(debug_data["seeds"][k])
            if num_seeds > 100:
                debug_data["seeds"][k] = f"too many to show in log (N={len(num_seeds)})"
    
    debug_logger = logger
    if "body-id" in debug_data:
        body_id = debug_data["body-id"]
        debug_logger = PrefixedLogger(logger, f"User {user}: Body {body_id}: ")
    else:
        debug_logger = PrefixedLogger(logger, f"User {user}: ")

    debug_string = ujson.dumps(debug_data, sort_keys=True)
    debug_logger.info(f"Client debug: {debug_string}")

    return (debug_string, HTTPStatus.OK)


@app.route('/set-default-params', methods=['POST'])
def set_default_params():
    global DEFAULT_METHOD
    DEFAULT_METHOD = request.args.get('method', DEFAULT_METHOD)
    
    defaults = {
        "method": DEFAULT_METHOD
    }
    
    return (jsonify(defaults), HTTPStatus.OK)

