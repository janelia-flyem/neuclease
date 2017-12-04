from __future__ import print_function
import sys
import os
import time
from datetime import datetime
from collections import OrderedDict, defaultdict
import tempfile
import traceback
import logging
import socket
import signal
from flask import Flask, request, render_template, abort, make_response, redirect, url_for

import agglomeration_split_tool

app = Flask(__name__)

LAUNCHER_PORT = 3000
splitters = OrderedDict()
ERR_MSG = ""

@app.route('/')
def index():
    return redirect(url_for('show_viewer_list'))

@app.route('/viewers')
def show_viewer_list():
    global ERR_MSG
    err_msg = ERR_MSG
    column_names=['Viewer']
    viewer_tuples = []
    for body_id, splitter in splitters.items():
        url = str(splitters[body_id].viewer)
        
        # For some reason socket.getfqdn() doesn't always work correctly on the Janelia network,
        # so we may have to fix the neuroglancer link.
        if not url.startswith('http://{}'.format(socket.gethostname())):
            url = 'http://' + '.'.join([socket.gethostname()] + url.split('.')[1:])
        viewer_tuples.append( (body_id, url) )
    ERR_MSG = ""
    return render_template('viewer-list.html.jinja',
                           hostname=socket.gethostname(),
                           err_msg=err_msg,
                           viewer_tuples=viewer_tuples,
                           column_names=column_names)

@app.route('/create-split-viewer', methods=['POST'])
def create_viewer():
    global ERR_MSG
    try:
        body_id = int(request.form['body_id'].strip())
    except:
        ERR_MSG = "Could not parse body ID: {}".format(request.form['body_id'].strip())
    else:
        splitters[body_id] = agglomeration_split_tool.run_interactive(tool_args, graph, body_id)
    return redirect(url_for('show_viewer_list'))

if __name__ == '__main__':
    import argparse
    import sys
    print(sys.argv)

    # Don't log ordinary GET, POST, etc.
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
     # Terminate results in normal shutdown
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

    # Start the neuroglancer server
    tool_args, graph = agglomeration_split_tool.main()

    print("Starting server on 0.0.0.0:{}".format(LAUNCHER_PORT))
    app.run(host='0.0.0.0', port=LAUNCHER_PORT, debug=True)
