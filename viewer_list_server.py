from __future__ import print_function
import sys
import os
import time
import json
from datetime import datetime
from collections import OrderedDict, defaultdict, namedtuple
import tempfile
import traceback
import logging
import socket
import signal
from flask import Flask, request, render_template, abort, make_response, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import agglomeration_split_tool

LAUNCHER_PORT = 5000

SplitterTuple = namedtuple("SplitterTuple", "body_id splitter seeds_filepath")

ViewerListRow = namedtuple("ViewerListRow", "body_id viewer_url initial_seeds_name initial_seeds_url current_seeds_url current_seeds_filename")

splitters = OrderedDict()
seed_files = OrderedDict()

VolumeDetails = namedtuple("VolumeDetails", "graph_id grayscale_id base_segmentation_id")
volume_details = ()

ERR_MSG = ""

UPLOAD_FOLDER = os.path.join(os.path.split(__file__)[0] + './seeds')

ALLOWED_EXTENSIONS = set(['json'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return redirect(url_for('show_viewer_list'))


@app.route('/viewers')
def show_viewer_list():
    global ERR_MSG
    err_msg = ERR_MSG
    column_names=['Viewer', 'Initial Seeds', 'Current Seeds']
    rows = []
    for body_id, splitter_tuple in splitters.items():
        initial_seeds_name = os.path.split(splitter_tuple.seeds_filepath)[1]
        
        # For some reason socket.getfqdn() doesn't always work correctly on the Janelia network,
        # so we may have to fix the neuroglancer link.
        viewer_url = str(splitter_tuple.splitter.viewer)
        if socket.gethostname() == 'bergs-lm3':
            # Hack to get my development machine working on Janelia's network,
            # which somehow refuses to return the proper FQDN
            viewer_url = 'http://localhost:' + viewer_url.split(':')[-1]

        initial_seeds_url = url_for('initial_seeds_file', filename=initial_seeds_name)
        current_seeds_url = url_for('download_current_seeds', body_id=body_id)
        current_seeds_filename = 'interactive-seeds-{}.json'.format(body_id)
        rows.append( ViewerListRow(body_id, viewer_url, initial_seeds_name, initial_seeds_url, current_seeds_url, current_seeds_filename) )

    ERR_MSG = ""
    return render_template('viewer-list.html.jinja',
                           hostname=socket.gethostname(),
                           volume_details=volume_details,
                           err_msg=err_msg,
                           rows=rows,
                           column_names=column_names)


@app.route('/create-split-viewer', methods=['POST'])
def create_viewer():
    global ERR_MSG

    try:
        body_id = int(request.form['body_id'].strip())
    except:
        ERR_MSG = "Could not parse body ID: {}".format(request.form['body_id'].strip())
        return redirect(url_for('show_viewer_list'))

    try:
        uploaded_seeds_path = _save_seeds_file(request)
    except Exception as ex:
        ERR_MSG = str(ex)
        return redirect(url_for('show_viewer_list'))

    splitter = agglomeration_split_tool.run_interactive(tool_args, graph, body_id, uploaded_seeds_path)
    splitters[body_id] = SplitterTuple(body_id, splitter, uploaded_seeds_path)
    return redirect(url_for('show_viewer_list'))

@app.route('/initial-seeds/<filename>')
def initial_seeds_file(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _save_seeds_file(request):
    if 'seeds_file' not in request.files:
        return ''
    
    seeds_file = request.files['seeds_file']
    if not seeds_file or seeds_file.filename == '':
        return ''

    if not allowed_file(seeds_file.filename):
        raise RuntimeError("Wrong file type: {}".format(seeds_file.filename))

    filename = secure_filename(seeds_file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    seeds_file.save(upload_path)
    return upload_path

@app.route('/download-current-seeds/<body_id>')
def download_current_seeds(body_id):
    body_id = int(body_id)
    lookup = {}
    
    splitter = splitters[body_id].splitter
    for (label, pos), supervoxel_id in splitter.seed_position_to_supervoxel_id_map.items():
        lookup.setdefault(label, {"label": label, "supervoxels": []})
        seed_details = { "count": 1, "supervoxel_id": supervoxel_id, "position": list(pos) }
        lookup[label]["supervoxels"].append(seed_details)

    json_data = []
    for label in sorted(lookup.keys()):
        seed_entry = {"label": label, "supervoxels": []}
        for sv_item in lookup[label]["supervoxels"]:
            seed_entry["supervoxels"].append(sv_item)
        json_data.append(seed_entry)

    filename = "interactive-seeds-{}.json".format(body_id)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(save_path, 'w') as f:
        json.dump(json_data, f)

    # Tell the browser not to cache
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response
    

def mkdir_p(path):
    """
    Like the bash command 'mkdir -p' [bash]
    or makedirs(path, exists_ok=True) [python 3]
    """
    import os, errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == "__main__":
    import argparse
    import sys
    print(sys.argv)
    
    # DEBUG
#     if len(sys.argv) == 1:
#         sys.argv.extend( "interactive"\
#                          " --graph-db=exported_merge_graphs/274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0.sqlite"\
#                          " --image-url=brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec24_image"\
#                          " --segmentation-url=brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a"\
#                          .split() )

    mkdir_p(UPLOAD_FOLDER)

    # Don't log ordinary GET, POST, etc.
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
     # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

    # Start the neuroglancer server
    tool_args, graph = agglomeration_split_tool.main()
    
    graph_name = os.path.split(tool_args.graph_db)[1].split(':')[-1]
    grayscale_name = tool_args.image_url.split('://')[1]
    seg_name = tool_args.segmentation_url.split('://')[1]
    volume_details = VolumeDetails(graph_name, grayscale_name, seg_name)

    print("Starting server on 0.0.0.0:{}".format(LAUNCHER_PORT))
    app.run(host='0.0.0.0', port=LAUNCHER_PORT, debug=True)
