from __future__ import print_function
import sys
import os
import json
import copy
import signal
import httplib
import multiprocessing
from itertools import chain

from flask import Flask, request, abort, redirect, url_for, jsonify, Response, make_response

from agglomeration_split_tool import AgglomerationGraph, do_split
from logging_setup import init_logging, log_exceptions, ProtectedLogger

# FIXME: multiprocessing has unintended consequences for the log rollover procedure.
USE_MULTIPROCESSING = False

# Globals
pool = None # Must be instantiated after this module definition, at the bottom of main().
LOGFILE = None # Will be set in __main__, below
app = Flask(__name__)
logger = ProtectedLogger(__name__)

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
        }
    }
    """
    data = request.json
    if not data:
        abort(Response('Request is missing a JSON body', status=400))

    logger.info("Received cleave request: {}".format(json.dumps(data, sort_keys=True)))
    if USE_MULTIPROCESSING:
        cleave_results, status_code = pool.apply(_run_cleave, [data])
    else:
        cleave_results, status_code = _run_cleave(data)
    return jsonify(cleave_results), status_code

@log_exceptions(logger)
def _run_cleave(data):
    """
    Helper function that actually performs the cleave,
    and can be run in a separate process.
    Must not use any flask functions.
    """
    global logger
    global GRAPH

    body_id = data["body-id"]
    seeds = { int(k): v for k,v in data["seeds"].items() }

    # Remove empty seed classes (if any)
    for label in list(seeds.keys()):
        if len(seeds[label]) == 0:
            del seeds[label]

    cleave_results = copy.copy(data)
    cleave_results["seeds"] = dict(sorted((k, sorted(v)) for (k,v) in data["seeds"].items()))
    cleave_results["assignments"] = {}
    cleave_results["warnings"] = []

    if not data["seeds"]:
        msg = "Request contained no seeds!"
        logger.error(msg)
        logger.info("Responding with error PRECONDITION_FAILED.")
        cleave_results.setdefault("errors", []).append(msg)
        return cleave_results, httplib.PRECONDITION_FAILED # code 412

    # Structure seed data for do_split()
    # (These dicts would have more members when using Neuroglancer viewers,
    #  but for do_split() we only need to provide the supervoxel_id.)
    agglo_tool_split_seeds = {}
    for label, supervoxel_ids in data["seeds"].items():
        for sv in supervoxel_ids:
            agglo_tool_split_seeds.setdefault(int(label), []).append({ "supervoxel_id": sv })
    
    # Note: The GRAPH object is thread-safe (for reads, at least)
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

    disconnected_seeds = set()
    for label in seeds.keys():
        first_member = seeds[label][0]
        label_equivalences = set(cur_eqs.members(first_member))
        for seed in seeds[label][1:]:
            if seed not in label_equivalences:
                disconnected_seeds.add(label)
                label_equivalences.update(cur_eqs.members(seed))
        cleave_results["assignments"][str(label)] = sorted(list(label_equivalences))

    if disconnected_seeds:
        msg = "Cleave result for body {} contains non-contiguous objects for seeds: {}".format(body_id, sorted(list(disconnected_seeds)))
        logger.warning(msg)
        cleave_results["warnings"].append(msg)

    CHECK_MISSING_SUPERVOXELS = True
    all_body_edges = None

    if CHECK_MISSING_SUPERVOXELS:
        logger.info("Checking for missing supervoxels in cleave results for body {}".format(body_id))
        if all_body_edges is None:
            all_body_edges = GRAPH.get_agglo_edges(body_id)
        all_body_ids = set(chain(*(edge.segment_ids for edge in all_body_edges)))
        assigned_ids = set(chain(*cleave_results["assignments"].values()))
        if set(all_body_ids) != assigned_ids:
            msg = "Cleave result is not complete for body {body_id}, using seeds {seeds}.".format(body_id=body_id, seeds=sorted(map(int, data["seeds"])))
            logger.error(msg)
            cleave_results.setdefault("errors", []).append(msg)
            logger.info("Responding with error PRECONDITION_FAILED.")
            return ( cleave_results, httplib.PRECONDITION_FAILED )

    logger.info("Sending cleave results for body: {}".format(cleave_results['body-id']))
    return ( cleave_results, httplib.OK )

def main():
    global GRAPH
    global pool
    global LOGFILE
    global logger
    global app
    global compute_cleave
    global show_log

    import argparse

    # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

#     ## DEBUG
#     # Careful:
#     # The flask debug server's "reloader" feature may cause this section to be executed more than once!
#     if len(sys.argv) == 1:
#         sys.argv += ["--graph-db", "exported_merge_graphs/274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0.sqlite",
#                      "--log-dir", "logs"]

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5555, type=int)
    parser.add_argument('--graph-db', required=True)
    parser.add_argument('--log-dir', required=False)
    args = parser.parse_args()

    if not os.path.exists(args.graph_db):
        sys.stderr.write("Graph database not found: {}\n".format(args.graph_db))
        sys.exit(-1)

    GRAPH = AgglomerationGraph(args.graph_db)

    ##
    ## Configure logging
    ##
    LOGFILE = init_logging(logger, args.log_dir, args.graph_db)
    logger.info("Server started with command: {}".format(' '.join(sys.argv)))

    # Pool must be started LAST, after we've configured all the global variables (logger, etc.),
    # so that the forked (child) processes have the same setup as the parent process.
    pool = multiprocessing.Pool(8)

    # Start app
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()